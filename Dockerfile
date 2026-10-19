# Job Market Agent — run it anywhere with Docker.
#
#   docker build -t job-market-agent .
#   docker run -p 8501:8501 -e GROQ_API_KEY=your_key job-market-agent
#
# Then open http://localhost:8501

FROM python:3.12-slim

WORKDIR /app

# Install dependencies first, so Docker can cache this layer.
# Change your code -> only the copy below re-runs, not the slow pip install.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the app itself.
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "agent_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
