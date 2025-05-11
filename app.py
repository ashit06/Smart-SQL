import streamlit as st
import pandas as pd
import psycopg2
import time
from utils import save_db_details, get_the_output_from_llm, execute_the_solution, explain_chart

import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# -------- Helper: Make DataFrame PyArrow-safe --------
def convert_object_columns_to_str(df):
    safe_df = df.copy()
    for col in safe_df.select_dtypes(include="object").columns:
        safe_df[col] = safe_df[col].astype(str)
    return safe_df

# -------- Session State Initialization --------
defaults = {
    "messages": [],
    "metrics": {"processing_time": None, "query_complexity": "low", "rows_returned": None},
    "db_uri": None,
    "unique_id": None,
    "raw_df": None,
}
for key, default in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default

# -------- Page Config --------
st.set_page_config(page_title="Universal Database Assistant", layout="wide")
page = st.sidebar.radio("Navigate to:", ["Chat", "Visualize & Analyze"])

# -------- Chat Page --------
def run_chat_page():
    st.title("🧠 Smart SQL Chat Assistant")

    # -- Sidebar: Connection --
    with st.sidebar:
        st.header("Database Connection")
        uri = st.text_input(
            "PostgreSQL URI",
            placeholder="postgresql://user:pass@host:port/dbname"
        )
        if st.button("Connect"):
            try:
                with psycopg2.connect(uri):
                    pass
                st.session_state.db_uri = uri
                st.session_state.unique_id = save_db_details(uri)
                st.success("✅ Connected successfully!")
            except Exception as e:
                st.error(f"❌ Connection failed: {e}")

        # -- Sidebar: Metrics --
        st.header("Query Metrics")
        m = st.session_state.metrics
        c1, c2 = st.columns(2)
        if m["processing_time"] is not None:
            c1.metric("⏱️ Time", f"{m['processing_time']:.2f}s")
        if m["query_complexity"]:
            c2.metric("⚙️ Complexity", m["query_complexity"].capitalize())
        if m["rows_returned"] is not None:
            st.metric("📊 Rows", m["rows_returned"])

    # -- Display Chat History --
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and isinstance(msg["content"], dict):
                st.code(msg["content"]["sql"], language="sql")
                df = convert_object_columns_to_str(msg["content"]["formatted_df"])
                st.dataframe(df)
            else:
                st.write(msg["content"])

    # -- Chat Input --
    if prompt := st.chat_input("Ask about your data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        if not st.session_state.db_uri:
            st.chat_message("assistant").warning("Please connect to a database first.")
            return

        with st.chat_message("assistant"):
            start_time = time.time()
            response = get_the_output_from_llm(
                prompt,
                st.session_state.unique_id,
                st.session_state.db_uri
            )
            duration = time.time() - start_time

            if isinstance(response, dict) and response.get("type") == "dataframe":
                sql = response["sql"]
                st.code(sql, language="sql")

                # Raw and formatted results
                raw_df = execute_the_solution(sql, st.session_state.db_uri, format_output=False)
                formatted_df = execute_the_solution(sql, st.session_state.db_uri, format_output=True)

                st.session_state.raw_df = raw_df
                df = convert_object_columns_to_str(formatted_df)
                st.dataframe(df)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": {
                        "sql": sql,
                        "formatted_df": formatted_df
                    }
                })

                rows = len(raw_df)
                complexity = "high" if rows > 100 else "medium"

            else:
                msg = response.get("text") if isinstance(response, dict) else str(response)
                st.write(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
                rows = None
                complexity = "low"

            # Update metrics
            st.session_state.metrics = {
                "processing_time": duration,
                "query_complexity": complexity,
                "rows_returned": rows
            }

def create_pdf_report(df, explanation_text):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width/2, height - 50, "Data Analysis Report")

    # Data Preview
    c.setFont("Helvetica", 12)
    text_y = height - 80
    c.drawString(40, text_y, "First 5 Rows:")
    text_y -= 20
    preview = df.head().to_string(index=False).split("\n")
    for line in preview:
        c.drawString(40, text_y, line)
        text_y -= 14

    # Explanation
    text_y -= 10
    c.drawString(40, text_y, "AI Explanation:")
    text_y -= 20
    for line in explanation_text.split("\n"):
        c.drawString(40, text_y, line)
        text_y -= 14
        if text_y < 50:
            c.showPage()
            text_y = height - 50

    c.save()
    buffer.seek(0)
    return buffer

# -------- Visualization Page --------
def run_visualization_page():
    st.title("📊 Visualization & Analysis")

    df = st.session_state.raw_df
    if df is None or df.empty:
        st.warning("⚠️ No data found. Please run a query in the Chat tab.")
        return

    # Data overview
    with st.expander("🔍 Data Overview"):
        st.write("Shape:", df.shape)
        st.write("Dtypes:")
        st.dataframe(df.dtypes.astype(str))
        st.dataframe(df.head())

    # Clean numeric columns
    numeric_df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    if numeric_df.empty:
        st.warning("⚠️ No numeric columns available for plotting.")
        return

    # Select columns & chart type
    x_col = st.selectbox("Select numeric column to plot", numeric_df.columns, key="viz_x")
    y_col = st.selectbox("Select Y-axis (optional)", ["None"] + list(numeric_df.columns), key="viz_y")
 
    chart_type = st.radio("Chart type", ["Line", "Bar", "Histogram"], horizontal=True)

    st.subheader("📈 Chart")
    if chart_type == "Line":
        st.line_chart(numeric_df[x_col])
    elif chart_type == "Bar":
        st.bar_chart(numeric_df[x_col])
    else:  # Histogram via value_counts
        counts = (
            pd.cut(numeric_df[x_col], bins=20)
              .value_counts()
              .sort_index()
        )
        st.bar_chart(counts)

    # Generate explanation
    explanation = ""
    if st.button("📝 Explain Chart"):
        with st.spinner("Generating explanation..."):
            # Prepare a small sample for context
            sample = numeric_df[[x_col]].dropna().head(5).reset_index()
            idx_col = 'index' if 'index' in sample.columns else sample.columns[0]
            explanation = explain_chart(sample, idx_col, x_col, chart_type)
        st.subheader("💡 Chart Explanation")
        st.write(explanation)

    # PDF download (table + explanation only)
    if st.button("📄 Download PDF Report"):
        pdf_buffer = create_pdf_report(df, explanation or "No explanation generated.")
        st.download_button(
            label="Download PDF",
            data=pdf_buffer,
            file_name="report.pdf",
            mime="application/pdf"
        )


 

# -------- Page Router --------
if page == "Chat":
    run_chat_page()
else:
    run_visualization_page()
