"""
Label Studio ML 适配器启动文件
"""
import os
import logging
from label_studio_ml.api import init_app
from model import SAM3Adapter

logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)

app = init_app(model_class=SAM3Adapter)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9090, debug=False)
