"""
Label Studio ML 适配器
轻量级服务，将 Label Studio 格式转换为 SAM3 推理 API 格式
"""
import os
import io
import base64
import logging
from typing import List, Dict, Optional
import requests
from PIL import Image
from label_studio_ml.model import LabelStudioMLBase
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path
import json
from label_studio_ml.response import ModelResponse

logger = logging.getLogger(__name__)

# SAM3 推理服务地址
INFERENCE_URL = os.getenv("SAM3_INFERENCE_URL", "http://sam3-inference:8000")


class SAM3Adapter(LabelStudioMLBase):
    """SAM3 适配器 - 调用独立的推理服务"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set('model_version', "SAM3-Adapter-v1.0")
        
        # 调试：打印所有kwargs
        logger.info(f"SAM3 Adapter __init__ kwargs keys: {list(kwargs.keys())}")
        logger.info(f"SAM3 Adapter __init__ full kwargs: {json.dumps({k: str(v) for k, v in kwargs.items()}, indent=2)}")
        
        # 尝试从不同位置获取extra_params
        extra_params = kwargs.get('extra_params', {})
        
        # 如果extra_params是字符串，尝试解析JSON
        if isinstance(extra_params, str):
            try:
                extra_params = json.loads(extra_params)
                logger.info("Parsed extra_params from JSON string")
            except:
                logger.warning(f"Failed to parse extra_params as JSON: {extra_params}")
                extra_params = {}
        
        # 保存extra_params
        if extra_params:
            self.set("extra_params", json.dumps(extra_params))
            logger.info(f"Stored extra_params: {json.dumps(extra_params, indent=2)}")
        else:
            logger.warning("No extra_params found in kwargs")
        
        # 打印所有可用的属性
        logger.info(f"Available attributes after init: {[attr for attr in dir(self) if not attr.startswith('_')]}")
        
        logger.info(f"SAM3 Adapter initialized, inference URL: {INFERENCE_URL}")

    
    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        # 获取标签配置
        from_name, to_name, value = self._get_labels()
        
        # 详细调试predict调用
        logger.info(f"=== Predict called ===")
        logger.info(f"kwargs keys: {list(kwargs.keys())}")
        logger.info(f"kwargs: {json.dumps({k: str(v)[:200] for k, v in kwargs.items()}, indent=2)}")
        
        # 尝试多种方式获取参数
        params = {}
        
        # 方式1: 从初始化时保存的extra_params获取
        if params_ext := self.get("extra_params"):
            try:
                params.update(json.loads(params_ext))
                logger.info(f"Got params from stored extra_params: {params}")
            except Exception as e:
                logger.error(f"Failed to parse stored extra_params: {e}")
        
        # 方式2: 从kwargs中的params获取
        if 'params' in kwargs:
            params.update(kwargs['params'])
            logger.info(f"Got params from kwargs['params']: {kwargs['params']}")
        
        # 方式3: 尝试从project配置中获取
        if hasattr(self, 'project'):
            logger.info(f"Project object exists: {type(self.project)}")
            if hasattr(self.project, 'extra_params'):
                logger.info(f"Project extra_params: {self.project.extra_params}")
        
        # 方式4: 检查是否有parsed_label_config
        if hasattr(self, 'parsed_label_config'):
            logger.info(f"Parsed label config type: {type(self.parsed_label_config)}")
        
        logger.info(f"Final params to use: {json.dumps(params, indent=2)}")
        
        prompt = params.get("prompt")
        output_type = params.get("output_type", "segment")
        label_name = params.get("label")  # 允许用户指定标签名
        
        logger.info(f"Extracted -> prompt: {prompt}, output_type: {output_type}, label: {label_name}")
        
        predictions = []
        
        for task in tasks:
            try:
                # 获取图像 URL
                image_url = task['data'].get(value)
                if not image_url:
                    predictions.append({"result": [], "model_version": self.model_version})
                    continue
                
                # 调用推理 API（自动预测模式）
                inference_result = self._call_inference_api(
                    image_url, 
                    task_id=task.get('id'),
                    prompt=prompt,
                    output_type=output_type
                )
                
                logger.info(f"Inference result: {json.dumps({k: str(v)[:100] for k, v in inference_result.items()}, indent=2)}")
                
                # 转换为 Label Studio 格式
                results = self._convert_to_ls_format(
                    inference_result, from_name, to_name, output_type,
                    prompt=prompt, label_name=label_name
                )
                
                logger.info(f"Converted {len(results)} results to Label Studio format")
                
                predictions.append({
                    "result": results,
                    "model_version": self.model_version
                })
                
            except Exception as e:
                logger.error(f"Prediction failed: {e}", exc_info=True)
                predictions.append({
                    "result": [],
                    "model_version": self.model_version,
                    "error": str(e)
                })
        
        return ModelResponse(predictions=predictions)
    
    def _get_labels(self):
        """获取标签配置"""
        for control_type in ['BrushLabels', 'PolygonLabels', 'RectangleLabels', 'KeyPointLabels']:
            try:
                result = self.get_first_tag_occurence(control_type, 'Image')
                if result:
                    logger.info(f"Found label config: {control_type} -> {result}")
                    return result
            except (ValueError, KeyError) as e:
                logger.debug(f"Control type {control_type} not found: {e}")
                continue
        
        # 如果没有找到任何配置，返回默认值
        logger.warning("No suitable label config found, using defaults")
        return ('label', 'image', 'image')
    
    def _select_label(self, available_labels: List[str], prompt: Optional[str] = None, 
                     label_name: Optional[str] = None) -> str:
        """智能选择标签
        
        优先级：
        1. 用户明确指定的label_name
        2. 从prompt中匹配到的标签
        3. 第一个可用标签
        4. 默认值"Object"
        """
        # 优先使用用户指定的标签
        if label_name:
            if label_name in available_labels:
                logger.info(f"Using user-specified label: {label_name}")
                return label_name
            else:
                logger.warning(f"Specified label '{label_name}' not in available labels, will try to match from prompt")
        
        # 尝试从prompt中匹配标签
        if prompt and available_labels:
            prompt_lower = prompt.lower()
            for label in available_labels:
                # 检查prompt中是否包含标签名（不区分大小写）
                if label.lower() in prompt_lower:
                    logger.info(f"Matched label '{label}' from prompt: {prompt}")
                    return label
        
        # 使用第一个可用标签
        if available_labels:
            logger.info(f"Using first available label: {available_labels[0]}")
            return available_labels[0]
        
        # 默认值
        logger.warning("No labels available, using default 'Object'")
        return "Object"
    
    
    def _get_image_base64(self, image_url: str, task_id: Optional[str] = None) -> str:
        """获取图片并转换为 base64"""
        try:
            # 使用 Label Studio SDK 的 get_local_path 来正确处理各种 URL 格式
            image_path = get_local_path(image_url, task_id=task_id)
            
            # 打开图片并转换为 RGB
            image = Image.open(image_path).convert("RGB")
            
            # 转换为 base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/jpeg;base64,{img_base64}"
            
        except Exception as e:
            logger.error(f"Failed to load image from {image_url}: {e}")
            raise
    
    def _call_inference_api(self, image_url: str, task_id: Optional[str] = None,
                           prompt: Optional[str] = None, output_type: str = "segment") -> Dict:
        """调用 SAM3 推理 API"""
        # 获取图片 base64
        image_base64 = self._get_image_base64(image_url, task_id=task_id)
        
        # 准备请求
        payload = {
            "image": image_base64,
            "output_type": output_type
        }
        
        # 添加可选的prompt参数
        if prompt:
            payload["prompt"] = prompt
        
        logger.info(f"Calling inference API with output_type={output_type}, prompt={prompt}")
        
        # 调用 API
        response = requests.post(
            f"{INFERENCE_URL}/predict",
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    
    def _convert_to_ls_format(self, inference_result: Dict, from_name: str, 
                              to_name: str, output_type: str = "segment",
                              prompt: Optional[str] = None, label_name: Optional[str] = None) -> List[Dict]:
        """转换推理结果为 Label Studio 格式
        
        Args:
            inference_result: SAM3推理结果
            from_name: 标签控件名称
            to_name: 目标对象名称
            output_type: 输出类型（bbox或segment）
            prompt: 文本提示词，用于智能匹配标签
            label_name: 用户指定的标签名称（优先级最高）
        """
        logger.info(f"Converting to LS format: output_type={output_type}, from_name={from_name}, to_name={to_name}")
        logger.info(f"Prompt: {prompt}, Specified label: {label_name}")
        results = []
        
        # 获取图像尺寸
        img_width, img_height = inference_result.get('image_size', [640, 480])
        logger.info(f"Image size: {img_width}x{img_height}")
        
        # 获取可用的标签列表
        available_labels = []
        try:
            # 方法1: 尝试使用parsed_label_config
            if hasattr(self, 'parsed_label_config') and isinstance(self.parsed_label_config, dict):
                # 从parsed_label_config中提取标签
                for key, value in self.parsed_label_config.items():
                    if isinstance(value, dict) and 'labels' in value:
                        available_labels.extend(value['labels'])
                        logger.info(f"Got labels from parsed_label_config: {available_labels}")
                        break
            
            # 方法2: 如果方法1失败，尝试从label_interface获取
            if not available_labels and hasattr(self, 'label_interface'):
                logger.info(f"label_interface type: {type(self.label_interface)}")
                logger.info(f"label_interface attributes: {[attr for attr in dir(self.label_interface) if not attr.startswith('_')]}")
                
                # 尝试获取控件信息
                if hasattr(self.label_interface, 'controls'):
                    for control_name, control in self.label_interface.controls.items():
                        if hasattr(control, 'labels'):
                            available_labels = [label for label in control.labels]
                            logger.info(f"Got labels from control '{control_name}': {available_labels}")
                            break
                
        except Exception as e:
            logger.warning(f"Could not get labels from config: {e}", exc_info=True)
        
        # 智能选择标签
        selected_label = self._select_label(available_labels, prompt, label_name)
        logger.info(f"Selected label: {selected_label}")
        
        masks_data = inference_result.get('masks', [])
        logger.info(f"Processing {len(masks_data)} masks")
        
        for idx, mask_data in enumerate(masks_data):
            bbox = mask_data.get('bbox', [0, 0, 0, 0])
            score = mask_data.get('score', 0.0)
            logger.info(f"Mask {idx}: bbox={bbox}, score={score}")
            
            if output_type == "bbox":
                # 返回矩形框格式
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                
                result_item = {
                    "id": f"sam3_{idx}",
                    "from_name": from_name,
                    "to_name": to_name,
                    "type": "rectanglelabels",
                    "original_width": img_width,
                    "original_height": img_height,
                    "image_rotation": 0,
                    "value": {
                        "x": (x1 / img_width) * 100,
                        "y": (y1 / img_height) * 100,
                        "width": (width / img_width) * 100,
                        "height": (height / img_height) * 100,
                        "rotation": 0,
                        "rectanglelabels": [selected_label]
                    },
                    "score": score,
                    "readonly": False
                }
                logger.info(f"Created bbox result: {json.dumps(result_item, indent=2)}")
                results.append(result_item)
            else:
                # 返回分割mask格式
                mask_rle = mask_data.get('mask')
                if mask_rle:
                    results.append({
                        "id": f"sam3_{idx}",
                        "from_name": from_name,
                        "to_name": to_name,
                        "type": "brushlabels",
                        "original_width": img_width,
                        "original_height": img_height,
                        "image_rotation": 0,
                        "value": {
                            "format": "rle",
                            "rle": mask_rle,
                            "brushlabels": [selected_label]
                        },
                        "score": score,
                        "readonly": False
                    })
        
        return results
    
    def fit(self, event, data, **kwargs):
        """训练（不需要）"""
        return {"status": "skipped"}
