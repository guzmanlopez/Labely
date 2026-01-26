# Labely - Makefile
# Label Studio + SAM3 自动化标注平台

.PHONY: help setup build up down logs ps clean

# 默认目标
help:
	@echo "Labely - Label Studio + SAM3 自动化标注平台"
	@echo ""
	@echo "使用方法: make [target]"
	@echo ""
	@echo "Docker 命令:"
	@echo "  build              构建 Docker 镜像"
	@echo "  build-no-cache     构建 Docker 镜像 (无缓存)"
	@echo "  up                 启动所有服务 (GPU)"
	@echo "  up-cpu             启动所有服务 (CPU)"
	@echo "  up-nginx           启动所有服务 + Nginx"
	@echo "  down               停止所有服务"
	@echo "  restart            重启所有服务"
	@echo "  ps                 查看服务状态"
	@echo "  logs               查看所有日志"
	@echo "  logs-sam3          查看 SAM3 Backend 日志"
	@echo "  logs-ls            查看 Label Studio 日志"
	@echo ""
	@echo "开发命令:"
	@echo "  dev-sam3           本地运行 SAM3 Backend"
	@echo "  dev-ls             本地运行 Label Studio"
	@echo "  test               测试后端服务"
	@echo ""
	@echo "清理命令:"
	@echo "  clean              清理 Docker 资源"
	@echo "  clean-all          清理所有 (包括数据卷)"

# ==================== Docker ====================

build:
	@echo "构建 Docker 镜像..."
	docker-compose build

build-no-cache:
	@echo "构建 Docker 镜像 (无缓存)..."
	docker-compose build --no-cache

up:
	@echo "启动服务 (GPU 版本)..."
	docker-compose up -d
	@echo ""
	@echo "服务已启动:"
	@echo "  - Label Studio:  http://localhost:8080"
	@echo "  - SAM3 Backend:  http://localhost:9090"
	@echo ""
	@echo "首次启动 SAM3 会自动下载模型，请耐心等待..."

up-cpu:
	@echo "启动服务 (CPU 版本)..."
	DEVICE=cpu docker-compose up -d
	@echo ""
	@echo "服务已启动 (CPU 模式，推理速度较慢)"

up-nginx:
	@echo "启动服务 (含 Nginx)..."
	docker-compose --profile with-nginx up -d
	@echo ""
	@echo "服务已启动:"
	@echo "  - 统一入口: http://localhost:80"

down:
	@echo "停止服务..."
	docker-compose down

restart: down up

ps:
	docker-compose ps

logs:
	docker-compose logs -f

logs-sam3:
	docker-compose logs -f sam3-backend

logs-ls:
	docker-compose logs -f label-studio

# ==================== 开发 ====================

dev-sam3:
	@echo "启动 SAM3 Backend (本地开发模式)..."
	PYTHONPATH=. python -m label_studio_ml.server start \
		--host 0.0.0.0 \
		--port 9090 \
		--model-dir ./models \
		--model-class backend.sam3_backend.SAM3Backend

dev-ls:
	@echo "启动 Label Studio (本地开发模式)..."
	label-studio start --port 8080

test:
	@echo "测试后端服务..."
	python scripts/test_backend.py --sam3-url http://localhost:9090

# ==================== 清理 ====================

clean:
	@echo "清理 Docker 资源..."
	docker-compose down --rmi local --remove-orphans
	docker system prune -f

clean-all:
	@echo "清理所有资源 (包括数据卷)..."
	docker-compose down -v --rmi all --remove-orphans
	docker system prune -af
	@echo "警告: 数据卷已删除，所有标注数据将丢失！"
