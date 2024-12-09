from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv
import numpy as np
from PIL import Image
import io
import base64
import httpx
import torch
import openai
from loguru import logger

# 커스텀 유틸리티 및 모델 import
from utils.inference import (
    instruct_pix2pix,  # 전체 이미지 변환을 위한 모델
    sd_inpaint,        # 선택 영역 인페인팅을 위한 모델
)
from utils.device_utils import get_device  # GPU/CPU 설정 유틸리티
from utils.util import resize_image, dilate_mask  # 이미지 처리 유틸리티
from utils.model_setup import (
    get_sd_inpaint,        # Stable Diffusion 인페인팅 모델 초기화
    get_instruct_pix2pix   # Instruct Pix2Pix 모델 초기화
)
from utils.template import image_editor_template  # 프롬프트 템플릿
from utils.agent import image_editor_agent       # LangChain 에이전트

# 환경변수 로드
load_dotenv()

# OpenAI API 키 검증 및 설정
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Missing required environment variable: OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

# FastAPI 앱 초기화
app = FastAPI(title="Image Editor API")

# API 요청/응답 모델 정의
class ImageEditRequest(BaseModel):
    """이미지 편집 요청 모델"""
    image_url: str           # 편집할 이미지 URL
    mask_data: str           # base64로 인코딩된 마스크 이미지
    prompt: str              # 편집 지시사항
    edit_type: str = "inpaint"  # 편집 유형 (inpaint/transform)

class ImageEditResponse(BaseModel):
    """이미지 편집 응답 모델"""
    edited_image: str        # base64로 인코딩된 편집된 이미지
    status: str              # 처리 상태
    message: Optional[str] = None  # 추가 메시지

# 글로벌 변수로 모델들을 저장
models = {}

@app.on_event("startup")
async def startup_event():
    """서버 시작시 모델 초기화"""
    global models
    try:
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 디바이스 설정
        device = get_device()
        logger.info(f"Using device: {device}")
        
        # 모델들 초기화 - 캐싱됨
        models["sd_inpaint"] = get_sd_inpaint()
        models["instruct_pix2pix"] = get_instruct_pix2pix()
        models["agent"] = image_editor_agent()
        
        logger.info("All models initialized successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise e

@app.post("/api/edit-image", response_model=ImageEditResponse)
async def edit_image(request: ImageEditRequest):
    """
    이미지 편집 API 엔드포인트
    
    Args:
        request (ImageEditRequest): 이미지 편집 요청 데이터
        
    Returns:
        ImageEditResponse: 편집된 이미지와 상태 정보
        
    Raises:
        HTTPException: 처리 중 오류 발생시
    """
    try:
        # 이미지 다운로드 및 전처리
        original_image = await download_image(request.image_url)
        # 이미지 크기 제한 (704x704)
        resized_image, _ = resize_image(original_image, max_width=704, max_height=704)
        
        # 마스크 디코딩
        mask_array = decode_mask(request.mask_data)
        
        # 프롬프트 템플릿 적용
        formatted_prompt = image_editor_template(request.prompt)
        
        # 편집 유형에 따른 처리
        if request.edit_type == "inpaint" and mask_array.any():
            # 마스크 확장 (더 자연스러운 결과를 위해)
            dilated_mask = dilate_mask(mask_array)
            
            # 선택 영역 인페인팅
            edited_image = sd_inpaint(
                image=resized_image,
                mask=Image.fromarray(dilated_mask),
                inpaint_prompt=formatted_prompt
            )
        else:
            # 전체 이미지 변환
            edited_image = instruct_pix2pix(
                image=resized_image,
                prompt=formatted_prompt
            )[0]
            
        # 결과 이미지를 base64로 인코딩
        edited_base64 = encode_image(edited_image)
        
        return ImageEditResponse(
            edited_image=edited_base64,
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Error in edit_image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def download_image(url: str) -> Image.Image:
    """
    URL에서 이미지 다운로드
    
    Args:
        url (str): 이미지 URL
        
    Returns:
        PIL.Image: 다운로드된 이미지
        
    Raises:
        HTTPException: 다운로드 실패시
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Image download failed")
            return Image.open(io.BytesIO(response.content)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error downloading image: {str(e)}")

def decode_mask(mask_data: str) -> np.ndarray:
    """
    Base64 마스크 데이터를 numpy 배열로 디코딩
    
    Args:
        mask_data (str): base64로 인코딩된 마스크 이미지
        
    Returns:
        numpy.ndarray: 디코딩된 마스크 배열
        
    Raises:
        HTTPException: 디코딩 실패시
    """
    try:
        mask_bytes = base64.b64decode(mask_data)
        mask_image = Image.open(io.BytesIO(mask_bytes))
        return np.array(mask_image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error decoding mask: {str(e)}")

def encode_image(image: Image.Image) -> str:
    """
    PIL 이미지를 base64 문자열로 인코딩
    
    Args:
        image (PIL.Image): 인코딩할 이미지
        
    Returns:
        str: base64로 인코딩된 이미지 문자열
        
    Raises:
        HTTPException: 인코딩 실패시
    """
    try:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error encoding image: {str(e)}")

@app.get("/health")
async def health_check():
    """
    서버 상태 체크 엔드포인트
    
    Returns:
        dict: 서버 상태 정보
    """
    return {
        "status": "healthy",
        "device": get_device(),
        "models_loaded": list(models.keys())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)