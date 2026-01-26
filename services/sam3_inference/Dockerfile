# SAM3 推理服务 Dockerfile
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

WORKDIR /tmp

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 安装 PyTorch 和基础依赖
RUN pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu126 --no-cache-dir
RUN pip install einops decord pycocotools huggingface-hub --no-cache-dir

# 克隆并安装 SAM3（在 /tmp 下，不干扰 /app）
RUN git clone https://github.com/facebookresearch/sam3.git && \
    cd sam3 && \
    pip install --no-cache-dir . && \
    cd .. && \
    rm -rf sam3

# 设置应用工作目录
WORKDIR /app

# 复制依赖文件并安装应用依赖
COPY services/sam3_inference/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY services/sam3_inference/ .

ENV PYTHONUNBUFFERED=1
ENV DEVICE=cuda:0

EXPOSE 8000

CMD ["python", "main.py"]
