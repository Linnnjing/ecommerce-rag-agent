FROM python:3.10-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV HF_ENDPOINT=https://hf-mirror.com

RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple typing-extensions sympy networkx filelock fsspec jinja2

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.13.0+cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .
EXPOSE 8000
CMD ["uvicorn", "web_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
