#!/usr/bin/env python3
"""
测试 SAM3 推理服务（独立）
"""
import requests
import base64
from io import BytesIO
from PIL import Image
import sys

INFERENCE_URL = "http://localhost:8000"

def test_health():
    """测试健康检查"""
    print("🔍 测试推理服务健康检查...")
    try:
        response = requests.get(f"{INFERENCE_URL}/health", timeout=5)
        if response.status_code == 200:
            print(f"✅ 健康检查通过: {response.json()}")
            return True
        else:
            print(f"❌ 健康检查失败: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 健康检查失败: {e}")
        return False

def test_predict():
    """测试推理"""
    print("\n🔍 测试推理...")
    
    # 创建测试图像
    img = Image.new('RGB', (640, 480), color='red')
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    payload = {
        "image": f"data:image/jpeg;base64,{img_base64}",
        "points": [
            {"x": 320, "y": 240, "label": 1}
        ]
    }
    
    try:
        response = requests.post(
            f"{INFERENCE_URL}/predict",
            json=payload,
            timeout=60
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 推理成功")
            print(f"   生成掩码数: {len(data.get('masks', []))}")
            if data.get('masks'):
                print(f"   第一个掩码分数: {data['masks'][0]['score']}")
            print(f"   图像尺寸: {data.get('image_size')}")
            return True
        else:
            print(f"❌ 推理失败: HTTP {response.status_code}")
            print(f"   响应: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 50)
    print("SAM3 推理服务测试")
    print("=" * 50)
    
    results = [
        ("健康检查", test_health()),
        ("推理测试", test_predict())
    ]
    
    print("\n" + "=" * 50)
    print("测试总结")
    print("=" * 50)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name:20s}: {status}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\n总计: {passed}/{total} 测试通过")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
