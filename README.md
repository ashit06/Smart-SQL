# 🧠 Smart SQL Assistant

Smart SQL Assistant is a Python-based application that lets users query a relational database using natural human language. It translates plain English questions into SQL queries using a language model and executes them on a PostgreSQL database.

---

## 🚀 Features

* 🗣️ Query your database using plain English
* 🤖 AI-powered SQL generation using **Google Gemini**
* 🧠 Understands schema and foreign key relationships
* 🧪 Test-ready SQL setup included

---

## 📁 Project Structure

```
.
├── app.py                 # Main application
├── utils.py               # Core utility functions
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (Gemini API key)
├── test_chat_db/          # SQL script for test database
├── csvs/                  # Table and foreign key CSVs
├── vectors/               # Embeddings for LLM queries
└── README.md              # This documentation
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/smart-sql-assistant.git
cd smart-sql-assistant
```

### 2. Install Python dependencies

Make sure Python 3.x and pip are installed, then run:

```bash
pip install -r requirements.txt
```

### 3. Install and configure PostgreSQL

* Download and install PostgreSQL: [https://www.postgresql.org/download/](https://www.postgresql.org/download/)
* Start PostgreSQL and create a database, for example:

```bash
createdb smartsql
```

* Load the first 60 lines of the `test_chat_db.sql` file to set up test tables:

```bash
psql -d smartsql -f test_chat_db/test_chat_db.sql -v start=1 -v end=60
```

> Or simply open the file in a text editor and run lines 1–60 manually via psql or pgAdmin.

### 4. Configure environment variables

Create a `.env` file in the project root and add your **Gemini API key**:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 5. Run the application (using Streamlit)

You can start the app in two ways:

* **Streamlit (recommended for UI):**

  ```bash
  streamlit run app.py
  ```

* **Directly with Python:**

  ```bash
  python app.py
  ```

---

## 💡 Example Usage

**Input:**

```
List all customers who placed an order in the last 7 days.
```

**Output SQL:**

```sql
SELECT * FROM customers WHERE customer_id IN (
  SELECT customer_id FROM orders WHERE order_date >= NOW() - INTERVAL '7 days'
);
```

---

## 🧪 Testing Setup

* Test schema is located in `test_chat_db/test_chat_db.sql`
* Load first 60 lines to get starter tables like `orders`, `customers`, etc.
* You can test queries using the app interface once PostgreSQL is running and schema is loaded.

---

## 📦 Dependencies

* Python 3.x
* PostgreSQL
* Gemini API
* pandas, sqlalchemy, python-dotenv, etc.

(Full list in `requirements.txt`)

---

## 🤝 Contributing

Pull requests are welcome! Please create issues for any bugs or enhancement requests.

---

## 📄 License
Copyright @ Arpit Agrahari 2025 
