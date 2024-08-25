FROM python:3.9-slim
COPY . /app/
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["streamlit","src/Streamlit_app.py"]
