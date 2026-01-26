"""
SAM3 纯推理服务 - FastAPI
无 Label Studio 依赖，只提供分割推理 API
使用文本提示词进行图像分割
"""
import os
import io
import base64
from typing import List, Optional
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger("uvicorn.info")


# SAM3 文本提示词配置
DEFAULT_PROMPT = os.getenv("SAM3_TEXT_PROMPT", "segment all objects")

# 全局模型实例
_model = None
_processor = None



def get_model():
    """获取 SAM3 模型"""
    global _model, _processor
    if _model is None:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        device = os.getenv("DEVICE", "cuda:0")
        _model = build_sam3_image_model()
        if device != "cpu":
            _model = _model.to(device)
        _model.eval()
        _processor = Sam3Processor(_model)
    return _model, _processor

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()
    yield

# FastAPI 应用
app = FastAPI(title="SAM3 Inference Service", version="1.0.0", lifespan=lifespan)


# 请求模型
class InferenceRequest(BaseModel):
    image: str  # base64 编码的图像或 URL
    prompt: Optional[str] = None  # 可选的文本提示词，如果不提供则使用环境变量中的默认值
    output_type: Optional[str] = "segment"  # 输出类型: "bbox" 或 "segment"，默认为segment


class MaskResult(BaseModel):
    mask: Optional[List[int]] = None  # RLE 格式 (仅在segment模式下返回)
    score: float
    bbox: List[int]  # [x1, y1, x2, y2]


class InferenceResponse(BaseModel):
    masks: List[MaskResult]
    image_size: List[int]  # [width, height]




@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy", 
        "model": "SAM3",
        "default_prompt": DEFAULT_PROMPT,
        "supported_output_types": ["bbox", "segment"]
    }


@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    """
    执行分割推理
    
    Args:
        request: 包含图像、可选的文本提示词和输出类型
        
    Returns:
        分割掩码列表（根据output_type返回bbox或完整的segment）
    """

    # 加载图像
    image = load_image(request.image)
    width, height = image.size
    
    # 获取模型
    model, processor = get_model()
    
    # 设置图像
    inference_state = processor.set_image(image)
    
    # 使用文本提示词
    prompt = request.prompt if request.prompt else DEFAULT_PROMPT
    
    # 验证output_type参数
    output_type = request.output_type.lower() if request.output_type else "segment"
    if output_type not in ["bbox", "segment"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid output_type: {output_type}. Must be 'bbox' or 'segment'"
        )
    
    # SAM3 核心接口：使用文本提示词进行分割
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)
    logger.info(f"Output type: {output_type}, Prompt: {prompt}")
    logger.info(f"SAM3 output keys: {list(output.keys()) if isinstance(output, dict) else type(output)}")
    
    # 提取掩码
    masks = output.get("masks", [])
    boxes = output.get("boxes", [])
    scores = output.get("scores", [])
    
    # 安全地获取长度（处理tensor和list）
    masks_len = len(masks) if isinstance(masks, (list, tuple)) else (masks.shape[0] if hasattr(masks, 'shape') and len(masks.shape) > 0 else 0)
    boxes_len = len(boxes) if isinstance(boxes, (list, tuple)) else (boxes.shape[0] if hasattr(boxes, 'shape') and len(boxes.shape) > 0 else 0)
    scores_len = len(scores) if isinstance(scores, (list, tuple)) else (scores.shape[0] if hasattr(scores, 'shape') and len(scores.shape) > 0 else 0)
    
    logger.info(f"Extracted - masks: {masks_len}, boxes: {boxes_len}, scores: {scores_len}")
    logger.info(f"Masks type: {type(masks)}, Boxes type: {type(boxes)}, Scores type: {type(scores)}")
    
    # 如果有masks，打印第一个mask的形状
    if masks_len > 0:
        first_mask = masks[0]
        logger.info(f"First mask type: {type(first_mask)}, shape: {first_mask.shape if hasattr(first_mask, 'shape') else 'N/A'}")
    
    # 转换结果
    results = []
    
    # 确保masks和scores是可迭代的
    if masks_len == 0:
        logger.warning("No masks detected by SAM3")
        return InferenceResponse(masks=[], image_size=[width, height])
    
    for i, (mask, score) in enumerate(zip(masks, scores)):
        if hasattr(mask, 'cpu'):
            mask = mask.cpu().numpy()
        if mask.ndim == 3:
            mask = mask[0]
        
        # 获取 bbox
        if i < len(boxes):
            box = boxes[i]
            if hasattr(box, 'cpu'):
                box = box.cpu().numpy()
            
            # 调试：打印原始box值
            logger.info(f"Raw box[{i}]: {box}, box range: [{box.min():.4f}, {box.max():.4f}]")
            
            # 判断box是归一化坐标还是像素坐标
            # 如果box的最大值 <= 1.0，说明是归一化坐标（0-1）
            # 否则是像素坐标
            if box.max() <= 1.0:
                # 归一化坐标，需要乘以图像尺寸
                logger.info("Box is normalized (0-1), converting to pixels")
                bbox = [int(box[0] * width), int(box[1] * height), 
                        int(box[2] * width), int(box[3] * height)]
            else:
                # 像素坐标，直接使用
                logger.info("Box is already in pixels")
                bbox = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
            
            logger.info(f"Final bbox: {bbox}")
        else:
            bbox = get_bbox(mask)
        
        # 根据output_type决定是否返回mask
        if output_type == "segment":
            # 转换为 RLE
            mask_uint8 = (mask * 255).astype(np.uint8)
            rle = mask_to_rle(mask_uint8)
            results.append(MaskResult(
                mask=rle,
                score=float(score) if hasattr(score, 'item') else float(score),
                bbox=bbox
            ))
        else:  # bbox模式
            results.append(MaskResult(
                mask=None,  # bbox模式不返回mask
                score=float(score) if hasattr(score, 'item') else float(score),
                bbox=bbox
            ))
    
    logger.info(f"Inference results count: {len(results)}, output_type: {output_type}")    
    return InferenceResponse(
        masks=results,
        image_size=[width, height]
    )
    



def load_image(image_str: str) -> Image.Image:
    """加载图像"""
    if image_str.startswith('data:image'):
        # Base64
        header, data = image_str.split(',', 1)
        image_data = base64.b64decode(data)
        return Image.open(io.BytesIO(image_data)).convert("RGB")
    elif image_str.startswith('http'):
        # URL
        import requests
        response = requests.get(image_str)
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    else:
        # 文件路径
        return Image.open(image_str).convert("RGB")


def mask_to_rle(mask: np.ndarray) -> List[int]:
    """转换掩码为 RLE 格式"""
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return runs.tolist()


def get_bbox(mask: np.ndarray) -> List[int]:
    """从掩码计算边界框"""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return [0, 0, 0, 0]
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [int(cmin), int(rmin), int(cmax), int(rmax)]



if __name__ == "__main__":
    # app.run(host="0.0.0.0", port=8000)
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=debug_mode,
    )
    # uvicorn.run(
    #     "main:app",
    #     host="0.0.0.0",
    #     port=8000,
    #     reload=os.getenv("DEBUG", "false").lower() == "true"
    # )
