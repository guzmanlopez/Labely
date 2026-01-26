#!/bin/bash
# 清理旧文件和目录

echo "🧹 清理项目..."

# 删除废弃的文档
rm -f FIX_LABEL_CONFIG.md
rm -f GET_LABEL_STUDIO_TOKEN.md
rm -f HOW_TO_ENABLE_SMART_MODE.md
rm -f INTERACTIVE_DIAGNOSTIC.md
rm -f ISSUE_SUMMARY.md
rm -f MIGRATION_GUIDE.md
rm -f PROJECT_SUMMARY.md
rm -f SUCCESS_SUMMARY.md
rm -f TROUBLESHOOTING.md
rm -f test_interactive.py
rm -f test_ml_backend.py

# 删除旧的后端代码
rm -rf backend/sam3_backend
rm -rf backend/sam3_inference

# 删除旧的脚本
rm -f scripts/check_token.sh
rm -f scripts/deploy_sam3_inference.sh
rm -f scripts/test_backend.py
rm -f scripts/test_sam3_inference.py

# 删除旧的 Dockerfile
rm -f docker/Dockerfile.sam3
rm -f docker/Dockerfile.sam3_inference

echo "✅ 清理完成"
