# 필요한 라이브러리 임포트
import sys
sys.path.append('Grounded-Segment-Anything')
sys.path.append('Grounded-Segment-Anything/EfficientSAM')

import os
import shutil
import streamlit as st
from pydantic import BaseModel, Field
from typing import Optional, Type
import numpy as np
import torch
from PIL import Image
from langchain.tools import BaseTool

# 커스텀 유틸리티 함수 임포트
from utils.inference import (
    instruct_pix2pix,    # 이미지 변환 모델
    sd_inpaint,          # 인페인팅 모델
    lama_cleaner         # 객체 제거 모델
)
from utils.util import dilate_mask
from utils.device_utils import get_device
from typing import ClassVar

def log_debug(message):
    """
    디버깅 메시지를 Streamlit UI에 출력하는 헬퍼 함수
    Args:
        message: 출력할 디버그 메시지
    """
    st.write(f"Debug: {message}")

def image_transform(pil_image, prompt):
    """
    이미지 변환을 수행하는 함수
    
    Args:
        pil_image: 변환할 PIL 이미지
        prompt: 변환 지시사항
    
    Returns:
        변환된 PIL 이미지 또는 에러 시 None
    """
    try:
        log_debug(f"Starting image transform with prompt: {prompt}")
        
        # 영역이 선택되지 않은 경우 전체 이미지 변환
        if not st.session_state.get("coord", False):
            log_debug("Transforming entire image with pix2pix")
            transform_pillow = instruct_pix2pix(pil_image, prompt)[0]
        # 영역이 선택된 경우 인페인팅 수행
        else:
            log_debug("Performing inpainting on selected area")
            if "mask" not in st.session_state or st.session_state["mask"] is None:
                st.error("No mask available for inpainting")
                return None
            
            mask = Image.fromarray(st.session_state["mask"].squeeze())
            transform_pillow = sd_inpaint(pil_image, mask, prompt)
        
        log_debug("Image transform completed successfully")
        return transform_pillow
        
    except Exception as e:
        st.error(f"Error in image_transform: {str(e)}")
        import traceback
        st.error(f"Full error: {traceback.format_exc()}")
        return None

def object_erase(image, mask, device):
    """
    이미지에서 객체를 제거하는 함수
    
    Args:
        image: 처리할 이미지 (numpy array)
        mask: 제거할 영역의 마스크
        device: 사용할 장치 (CPU/GPU)
    
    Returns:
        처리된 PIL 이미지 또는 에러 시 None
    """
    try:
        log_debug("Starting object erase operation...")
        device = get_device()
        
        # GPU 메모리 정리
        if device == "cuda":
            torch.cuda.empty_cache()
            
        transform_pillow = lama_cleaner(image, mask, device)
        
        if transform_pillow is None:
            raise Exception("Failed to process image with lama_cleaner")
            
        log_debug("Object erase completed successfully")
        return transform_pillow
        
    except Exception as e:
        st.error(f"Error in object_erase: {str(e)}")
        import traceback
        st.error(f"Full error: {traceback.format_exc()}")
        return None

# Pydantic 모델 - 입력 유효성 검사용
class ImageTransformCheckInput(BaseModel):
    """이미지 변환 입력 검증을 위한 Pydantic 모델"""
    prompt: str = Field(..., description="prompt for transform the image")

class ImageTransformTool(BaseTool):
    """
    이미지 변환을 위한 LangChain 도구
    - 이미지 스타일 변경
    - 객체 대체/추가
    """
    name: str = "image_transform"
    description: str = """
    Please use this tool when you want to change the image style or replace, add specific objects with something else.
    """
    return_direct: bool = True
    
    def _run(self, prompt: str):
        """
        이미지 변환 실행 함수
        Args:
            prompt: 변환 지시사항
        Returns:
            변환된 이미지 또는 에러 시 None
        """
        try:
            log_debug(f"Running ImageTransformTool with prompt: {prompt}")
            
            # 세션 상태 검증
            if "inference_image" not in st.session_state or "image_state" not in st.session_state:
                raise ValueError("No image loaded in session state")
                
            pil_image = st.session_state["inference_image"][st.session_state["image_state"]]
            transform_pillow = image_transform(pil_image, prompt)
            
            if transform_pillow is None:
                raise Exception("Image transformation failed")
                
            log_debug("ImageTransformTool completed successfully")
            return transform_pillow
            
        except Exception as e:
            st.error(f"Error in ImageTransformTool: {str(e)}")
            import traceback
            st.error(f"Full error: {traceback.format_exc()}")
            return None
    
    def _arun(self, query: str):
        """비동기 실행은 지원하지 않음"""
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = ImageTransformCheckInput

class ObjectEraseTool(BaseTool):
    """
    이미지에서 객체를 제거하기 위한 LangChain 도구
    - 객체 제거
    - 영역 정리
    """
    name: str = "object_erase"
    description: str = """
    Please use this tool when you want to clean, erase or delete certain objects from an image.
    """
    return_direct: bool = True
    
    def _run(self, args=None):
        """
        객체 제거 실행 함수
        Args:
            args: 사용되지 않음
        Returns:
            처리된 이미지 또는 에러 시 None
        """
        try:
            log_debug("Running ObjectEraseTool")
            
            # 세션 상태 검증
            if "inference_image" not in st.session_state or "image_state" not in st.session_state:
                raise ValueError("No image loaded in session state")
                
            pil_image = st.session_state["inference_image"][st.session_state["image_state"]]
            np_image = np.array(pil_image)
            
            # 마스크 검증
            if "mask" not in st.session_state or st.session_state["mask"] is None:
                raise ValueError("No mask available. Please select an area using the drawing tool.")
                
            mask = st.session_state["mask"]
            
            # freedraw 모드에 따른 마스크 처리
            if not st.session_state.get("freedraw", False):
                mask = dilate_mask(mask, kernel_size=5, iterations=6)
                
            device = get_device()
            transform_pillow = object_erase(np_image, mask, device)
            
            if transform_pillow is None:
                raise Exception("Object erase operation failed")
                
            log_debug("ObjectEraseTool completed successfully")
            return transform_pillow
            
        except Exception as e:
            st.error(f"Error in ObjectEraseTool: {str(e)}")
            import traceback
            st.error(f"Full error: {traceback.format_exc()}")
            return None
    
    def _arun(self, query: str):
        """비동기 실행은 지원하지 않음"""
        raise NotImplementedError("This tool does not support async")