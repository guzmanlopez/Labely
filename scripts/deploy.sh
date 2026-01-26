#!/bin/bash
# 快速部署脚本

set -e

echo "🚀 Labely SAM3 部署"
echo "===================="
echo ""

# 停止旧服务
echo "📋 停止旧服务..."
docker compose down 2>/dev/null || true
echo ""

# 构建
echo "🔨 构建镜像..."
docker compose build
echo ""

# 启动
echo "▶️  启动服务..."
docker compose up -d
echo ""

# 等待
echo "⏳ 等待服务启动..."
sleep 10

# 健康检查
echo "🏥 健康检查..."
echo ""

echo "  SAM3 推理服务:"
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "  ✅ http://localhost:8000"
else
    echo "  ❌ 启动失败"
fi

echo ""
echo "  LS 适配器:"
if curl -s http://localhost:9090/health > /dev/null 2>&1; then
    echo "  ✅ http://localhost:9090"
else
    echo "  ❌ 启动失败"
fi

echo ""
echo "  Label Studio:"
if curl -s http://localhost:8080 > /dev/null 2>&1; then
    echo "  ✅ http://localhost:8080"
else
    echo "  ⏳ 仍在启动..."
fi

echo ""
echo "✅ 部署完成！"
echo ""
echo "下一步："
echo "  1. 访问 Label Studio: http://localhost:8080"
echo "  2. 配置 ML 后端: http://ls-adapter:9090"
echo "  3. 开始标注！"
echo ""
echo "查看日志:"
echo "  docker compose logs -f"
