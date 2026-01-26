# Label Studio ML 适配器 Dockerfile
FROM python:3.10-slim

WORKDIR /app

# 复制依赖文件
COPY services/ls_adapter/requirements.txt .
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY services/ls_adapter/ .

ENV PYTHONUNBUFFERED=1

EXPOSE 9090

CMD ["python", "app.py"]
