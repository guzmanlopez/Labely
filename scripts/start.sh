#!/bin/bash

# ==========================================
# Labely - 快速启动脚本
# Label Studio + SAM3 自动化标注平台
# ==========================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
echo_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
echo_error() { echo -e "${RED}[ERROR]${NC} $1"; }
echo_title() { echo -e "${BLUE}$1${NC}"; }

# 检查 Docker
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo_error "Docker 未安装，请先安装 Docker"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        echo_error "Docker 服务未运行，请启动 Docker"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        echo_error "Docker Compose 未安装"
        exit 1
    fi
}

# 检查 GPU
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            echo_info "检测到 NVIDIA GPU"
            return 0
        fi
    fi
    echo_warn "未检测到 NVIDIA GPU，将使用 CPU 模式"
    return 1
}

# 检查 HF Token
check_hf_token() {
    if [ -n "$HF_TOKEN" ]; then
        echo_info "检测到 HF_TOKEN 环境变量"
        return 0
    fi
    
    local token_file="$HOME/.cache/huggingface/token"
    if [ -f "$token_file" ]; then
        echo_info "检测到 Hugging Face 缓存 Token"
        return 0
    fi
    
    echo_warn "未检测到 Hugging Face Token"
    echo_warn "SAM3 模型下载可能受限，建议运行: huggingface-cli login"
    return 1
}

# 主逻辑
main() {
    echo_title "=========================================="
    echo_title "  Labely - Label Studio + SAM3 标注平台"
    echo_title "=========================================="
    echo ""
    
    # 检查环境
    echo_info "检查环境..."
    check_docker
    check_hf_token || true
    
    # 检测 GPU
    use_gpu=false
    if check_gpu; then
        use_gpu=true
    fi
    
    # 启动服务
    echo ""
    echo_info "启动服务..."
    
    if [ "$use_gpu" = true ]; then
        docker-compose up -d
    else
        DEVICE=cpu docker-compose up -d
    fi
    
    # 等待服务就绪
    echo_info "等待服务启动..."
    sleep 5
    
    # 显示状态
    echo ""
    echo_title "=========================================="
    echo_title "  服务已启动"
    echo_title "=========================================="
    echo ""
    echo_info "Label Studio: http://localhost:8080"
    echo_info "SAM3 Backend: http://localhost:9090"
    echo ""
    echo_warn "首次启动时 SAM3 会自动下载模型，请耐心等待..."
    echo ""
    echo_info "首次使用请在 Label Studio 中:"
    echo "  1. 创建账号并登录"
    echo "  2. 创建项目"
    echo "  3. 在 Settings -> Model 中连接 ML Backend:"
    echo "     - URL: http://sam3-backend:9090"
    echo "  4. 勾选 'Use for interactive preannotations'"
    echo ""
    echo_info "查看日志: docker-compose logs -f sam3-backend"
    echo_info "停止服务: docker-compose down"
}

main "$@"
