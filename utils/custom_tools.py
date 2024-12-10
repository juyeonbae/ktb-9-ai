import sys
sys.path.append('Grounded-Segment-Anything')
sys.path.append('Grounded-Segment-Anything/EfficientSAM')

import os
import logging
from pydantic import BaseModel, Field
from typing import Optional, Type, Dict, Any, List
import numpy as np
import torch
from PIL import Image
from langchain.tools import BaseTool
from contextlib import contextmanager

from utils.inference import (
    instruct_pix2pix,
    sd_inpaint,
    lama_cleaner
)
from utils.util import dilate_mask
from utils.device_utils import get_device

# [수정] 로깅 설정 개선
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('image_editor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# [추가] CUDA 메모리 관리
@contextmanager
def cuda_memory_manager():
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# [수정] 상태 관리 클래스 개선
class ImageState:
    def __init__(self):
        self._state: Dict[str, Any] = {
            "inference_image": [],
            "image_state": 0,
            "mask": None,
            "coord": False,
            "freedraw": False
        }
    
    @property
    def current_image(self) -> Optional[Image.Image]:
        images = self._state.get("inference_image", [])
        current_state = self._state.get("image_state", 0)
        return images[current_state] if images and len(images) > current_state else None

    def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        if key not in self._state:
            raise KeyError(f"Invalid state key: {key}")
        self._state[key] = value
        
    def update_image(self, image: Image.Image) -> None:
        self._state["inference_image"].append(image)
        self._state["image_state"] = len(self._state["inference_image"]) - 1

image_state = ImageState()

# [수정] 이미지 변환 함수 개선
def image_transform(pil_image: Image.Image, prompt: str) -> Optional[Image.Image]:
    try:
        logger.debug(f"Starting image transform with prompt: {prompt}")
        
        with cuda_memory_manager():
            if image_state.get("coord"):
                mask = image_state.get("mask")
                if mask is None:
                    raise ValueError("No mask available")
                    
                mask_image = Image.fromarray(mask.squeeze())
                remove_keywords = ["remove", "erase", "delete", "clean"]
                prompt_final = "remove this" if any(word in prompt.lower() for word in remove_keywords) else prompt
                
                return sd_inpaint(pil_image, mask_image, prompt_final)
            
            result = instruct_pix2pix(pil_image, prompt)
            return result[0] if result else None
            
    except Exception as e:
        logger.error("Image transform failed", exc_info=e)
        raise

# [수정] 객체 제거 함수 개선
def object_erase(image: np.ndarray, mask: np.ndarray, device: Optional[str] = None) -> Image.Image:
    try:
        logger.debug("Starting object erase operation")
        device = device or get_device()
        
        with cuda_memory_manager():
            result = lama_cleaner(image, mask, device)
            if result is None:
                raise RuntimeError("Lama cleaner processing failed")
            return result
            
    except Exception as e:
        logger.error("Object erase failed", exc_info=e)
        raise

# [수정] 입력 검증 모델 개선
class ImageTransformInput(BaseModel):
    prompt: str = Field(..., description="Transform prompt", min_length=1)

# [수정] 이미지 변환 도구 개선
class ImageTransformTool(BaseTool):
    name: str = "image_transform"
    description: str = "Transform image style or replace/add objects"
    args_schema: Type[BaseModel] = ImageTransformInput
    return_direct: bool = True
    
    def _run(self, prompt: str) -> Image.Image:
        try:
            logger.debug(f"Running transform with prompt: {prompt}")
            
            image = image_state.current_image
            if image is None:
                raise ValueError("No image loaded")
                
            return image_transform(image, prompt)
            
        except Exception as e:
            logger.error("Transform tool failed", exc_info=e)
            raise

    def _arun(self, query: str):
        raise NotImplementedError("Async not supported")

# [수정] 객체 제거 도구 개선
class ObjectEraseTool(BaseTool):
    name: str = "object_erase"
    description: str = "Clean, erase or delete objects from image"
    return_direct: bool = True
    
    def _run(self, args: Any = None) -> Image.Image:
        try:
            logger.debug("Running object erase")
            
            image = image_state.current_image
            if image is None:
                raise ValueError("No image loaded")
                
            mask = image_state.get("mask")
            if mask is None:
                raise ValueError("No mask selected")
                
            if not image_state.get("freedraw"):
                mask = dilate_mask(mask, kernel_size=5, iterations=6)
                
            return object_erase(np.array(image), mask)
            
        except Exception as e:
            logger.error("Erase tool failed", exc_info=e)
            raise

    def _arun(self, query: str):
        raise NotImplementedError("Async not supported")