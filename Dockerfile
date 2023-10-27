FROM python:3.9-slim

WORKDIR /app

COPY . .

RUN pip install -r req_docker.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "Intro.py", "--server.port=8501", "--server.address=0.0.0.0"]
