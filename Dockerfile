from: Python:3.8.0
expose 8501
cmd mkdir -p app
WORKDIR  /app
copy requirements.txt ./requirements.txt
run pip3 install -r requirements.txt
copy . .
ENTRYPOINT ["streamlit", "run"]
CMD ["The_application.py"]