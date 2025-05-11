import os
import re
import pandas as pd
import psycopg2
from uuid import uuid4
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader

# Initialize folders
os.makedirs('csvs', exist_ok=True)
os.makedirs('vectors', exist_ok=True)

# Load environment variables
load_dotenv()

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY in .env file")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-pro-latest')

# Dummy embeddings for schema-vector store
class SimpleEmbeddings:
    def embed_documents(self, texts):
        return [[0.1] * 384 for _ in texts]

    def embed_query(self, text):
        return [0.1] * 384

embeddings = SimpleEmbeddings()

def get_basic_table_details(cursor):
    """Retrieve table/column names from public schema."""
    cursor.execute("""
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public';
    """)
    return cursor.fetchall()

def get_foreign_key_info(cursor):
    """Retrieve foreign-key relationships."""
    cursor.execute("""
        SELECT
            conrelid::regclass AS table_name,
            conname AS foreign_key,
            confrelid::regclass AS referred_table
        FROM pg_constraint
        WHERE contype = 'f' AND connamespace = 'public'::regnamespace;
    """)
    return cursor.fetchall()

def create_vectors(filename, persist_directory):
    """Build a Chroma vector store from the saved CSV schema."""
    loader = CSVLoader(file_path=filename, metadata_columns=['table_name'])
    docs = loader.load()
    texts = [str(d.page_content) for d in docs]
    return Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        persist_directory=persist_directory,
        metadatas=[d.metadata for d in docs]
    )

def save_db_details(db_uri):
    """
    Extracts schema + foreign keys, writes to CSV, and vectorizes.
    Returns a unique_id to reference these files.
    """
    unique_id = str(uuid4()).replace("-", "_")
    try:
        with psycopg2.connect(db_uri) as conn:
            with conn.cursor() as cur:
                # Tables & columns
                tbls = get_basic_table_details(cur)
                df_tbl = pd.DataFrame(tbls, columns=['table_name','column_name','data_type'])
                df_tbl.to_csv(f'csvs/tables_{unique_id}.csv', index=False)
                # Foreign keys
                fks = get_foreign_key_info(cur)
                df_fk = pd.DataFrame(fks, columns=['table_name','foreign_key','referred_table'])
                df_fk.to_csv(f'csvs/foreign_keys_{unique_id}.csv', index=False)

                # Build vectors on columns CSV
                create_vectors(f'csvs/tables_{unique_id}.csv', f'vectors/tables_{unique_id}')
    except Exception as e:
        raise Exception(f"Database error: {e}")
    return unique_id

def generate_sql(query, schema_info, foreign_keys):
    """
    Ask Gemini to produce a SQL snippet wrapped in ```sql ... ``` markers.
    """
    prompt = f"""You are a PostgreSQL expert. Generate SQL for this database if and only if user's messagge is understandable:

Database Schema:
{schema_info}

Foreign Key Relationships:
{foreign_keys}

User Question: {query}

Rules:
1. Use proper JOINs based on foreign keys.
2. Only select necessary columns.
3. Format SQL between ```sql``` markers.
4. Add comments for any complex logic.

Respond **ONLY** with the SQL query, wrapped like:

```sql
SELECT ...
```"""
    try:
        resp = gemini_model.generate_content(prompt)
        text = resp.text or ""
        match = re.search(r"```sql\n(.*?)\n```", text, re.DOTALL)
        if match:
            return {"sql": match.group(1).strip(), "success": True}
        else:
            return {"error": f"No SQL found in AI response:\n{text}", "success": False}
    except Exception as e:
        return {"error": f"AI Error: {e}", "success": False}

def get_relevant_tables(query, unique_id):
    """Use vector similarity to find the top-5 relevant tables."""
    vectordb = Chroma(
        persist_directory=f'vectors/tables_{unique_id}',
        embedding_function=embeddings
    )
    docs = vectordb.similarity_search(query, k=5)
    return list({d.metadata['table_name'] for d in docs if 'table_name' in d.metadata})

def format_schema_info(df_tables, tables):
    """Produce a human-readable schema snippet for the prompt."""
    parts = []
    for t in tables:
        cols = df_tables[df_tables['table_name']==t]
        parts.append(f"Table {t}:\n" + cols[['column_name','data_type']].to_string(index=False))
    return "\n\n".join(parts)

def execute_the_solution(sql_query, db_uri, format_output=True):
    """
    Run the SQL, return a DataFrame.
    If format_output=True, format numbers/dates for display.
    """
    try:
        with psycopg2.connect(db_uri) as conn:
            with conn.cursor() as cur:
                cur.execute(sql_query)
                if not cur.description:
                    return pd.DataFrame() if not format_output else pd.DataFrame()
                cols = [d[0] for d in cur.description]
                rows = cur.fetchall()
        df = pd.DataFrame(rows, columns=cols)
        if not format_output:
            return df
        # Format numeric & datetime columns
        for c in df.select_dtypes(include='number').columns:
            df[c] = df[c].map(lambda x: f"{x:,.2f}")
        for c in df.select_dtypes(include='datetime').columns:
            df[c] = df[c].dt.strftime('%Y-%m-%d')
        return df
    except Exception as e:
        return f"Execution error: {e}"

def get_the_output_from_llm(query, unique_id, db_uri):
    """
    Load saved schema CSVs, pick relevant tables, ask AI for SQL,
    then execute and return either a DataFrame payload or an error text.
    """
    try:
        df_tbl = pd.read_csv(f'csvs/tables_{unique_id}.csv', dtype=str)
        df_fk  = pd.read_csv(f'csvs/foreign_keys_{unique_id}.csv', dtype=str)

        relevant = get_relevant_tables(query, unique_id) or df_tbl['table_name'].unique()[:3].tolist()
        schema_info = format_schema_info(df_tbl, relevant)
        fk_info = "\n".join(f"{r.table_name}.{r.foreign_key} → {r.referred_table}"
                            for r in df_fk.itertuples() if r.table_name in relevant)

        gen = generate_sql(query, schema_info, fk_info)
        if not gen.get("success"):
            return {"type":"text","text":gen.get("error"),"metrics":{"processing_time":0,"query_complexity":"low"}}

        sql = gen["sql"]
        result = execute_the_solution(sql, db_uri, format_output=False)

        if isinstance(result, pd.DataFrame):
            return {
                "type": "dataframe",
                "data": result,
                "sql": sql,
                "metrics": {
                    "processing_time": 0,
                    "query_complexity": "high" if len(result)>100 else "medium",
                    "rows_returned": len(result)
                }
            }
        else:
            # execution error string
            return {"type":"text","text":f"```sql\n{sql}\n```\n\n{result}","metrics":{"processing_time":0,"query_complexity":"low"}}
    except Exception as e:
        return {"type":"text","text":f"Processing error: {e}","metrics":{"processing_time":0,"query_complexity":"low"}}

def explain_chart(df: pd.DataFrame, x_col: str, y_col: str, chart_type: str) -> str:
    """
    Ask Gemini to explain the chart in very simple bullet points.
    """
    sample = df[[x_col,y_col]].head(5).to_dict(orient='records')
    prompt = f"""
You are a helpful assistant.
Explain this {chart_type} chart in simple bullet points:
- X axis: {x_col}
- Y axis: {y_col}
- Sample points: {sample}

Write:
• This chart shows…
• On the horizontal axis…
• On the vertical axis…
• The trend might be because…
"""
    resp = gemini_model.generate_content(prompt)
    return resp.text.strip()
