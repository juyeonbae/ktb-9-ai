import os
import torch
import tritonclient.http
from functools import lru_cache
from diffusers import (
    StableDiffusionInstructPix2PixPipeline, 
    EulerAncestralDiscreteScheduler, 
    StableDiffusionInpaintPipeline
)
from dotenv import load_dotenv
from utils.lama_cleaner_helper import load_jit_model
from utils.device_utils import get_device

# 전역 디바이스 설정
device = get_device()

def get_triton_client():
    """Triton 추론 서버 클라이언트 생성"""
    load_dotenv()
    url = os.getenv("TRITON_HTTP_URL")
    return tritonclient.http.InferenceServerClient(url=url, verbose=False)

@lru_cache(maxsize=None)
def get_sd_inpaint():
    """Stable Diffusion Inpainting 모델 로드 및 캐싱"""
    print("Stable Diffusion Inpaint setup!")
    
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    pipe.scheduler.num_inference_steps = 50  # 추론 스텝 증가
    pipe.safety_checker = None  # 안전 체커 비활성화로 속도 향상
    
    return pipe

@lru_cache(maxsize=None)
def get_lama_cleaner():
    """LaMa Cleaner 모델 로드 및 캐싱"""
    print("lama cleaner setup!")
    
    LAMA_MODEL_URL = os.environ.get(
        "LAMA_MODEL_URL",
        "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
    )
    LAMA_MODEL_MD5 = os.environ.get("LAMA_MODEL_MD5", "e3aa4aaa15225a33ec84f9f4bc47e500")

    lama_model = load_jit_model(LAMA_MODEL_URL, device, LAMA_MODEL_MD5).eval()
    return lama_model

@lru_cache(maxsize=None)
def get_instruct_pix2pix():
    """Instruct Pix2Pix 모델 로드 및 캐싱"""
    print("Instruct Pix2Pix setup!")
        
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None
    ).to(device)
    
    # 스케줄러 설정
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config,
        num_train_timesteps=500)
    return pipe