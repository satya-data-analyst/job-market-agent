# Job Market Agent

One assistant, two tools. Ask a question in plain English — it decides whether
it needs the database (SQL) or the job descriptions (RAG), then answers with
that tool.

![Job Market Agent](screenshot.png)

**Live demo:** (add your Streamlit URL here)

## What it does

- 10,000 real job postings in SQLite — salary, location, seniority, company
- 1,000 job descriptions indexed for semantic search
- A router that picks the right tool per question, and shows you which it chose

## Examples

- "How many jobs are remote?"            -> SQL
- "Which 10 companies have the most postings?"  -> SQL
- "What skills do data analyst roles require?"  -> RAG
- "Do any jobs mention AI agents?"       -> RAG

## How it works

question -> router (SQL / RAG / chat) -> tool runs -> grounded answer

- **SQL tool**: the schema (with null-rates per column) goes to the LLM, which
  writes a query; we run it and show the result and the SQL.
- **RAG tool**: the question is embedded, matched against 1,000 job-description
  vectors by cosine similarity, and the top matches become the model's context.
- **Router**: one classification call decides which tool to use.

## Tech

Python, Streamlit, SQLite, fastembed, Groq

## Note on data

Built on the public "Job Postings with Salaries (October 2026)" dataset.
Descriptions belong to the employers that published them.
