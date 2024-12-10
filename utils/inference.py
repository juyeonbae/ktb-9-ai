import sys
from typing import Optional, Tuple, List, Union
from contextlib import contextmanager
import logging
import numpy as np
import torch
from PIL import Image, ImageOps, ImageFilter
import tritonclient.http
from diffusers import StableDiffusionInpaintPipeline

from utils.lama_cleaner_helper import norm_img
from utils.model_setup import (
    get_triton_client,
    get_sd_inpaint,
    get_lama_cleaner,
    get_instruct_pix2pix
)
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

def instruct_pix2pix(
    image: Image.Image,
    prompt: str,
    num_steps: int = 50,
    guidance_scale: float = 7.0
) -> Optional[List[Image.Image]]:
    try:
        with cuda_memory_manager():
            pipe = get_instruct_pix2pix()
            image = ImageOps.exif_transpose(image)
            images = pipe(
                prompt,
                image=image,
                num_inference_steps=num_steps,
                image_guidance_scale=1.5,
                guidance_scale=guidance_scale
            ).images
            return images
    except Exception as e:
        logger.error("Pix2pix 처리 실패", exc_info=e)
        raise RuntimeError(f"이미지 변환 실패: {str(e)}")

def prepare_sam_inputs(
    image: np.ndarray,
    neg_coords: np.ndarray,
    pos_coords: np.ndarray,
    labels: np.ndarray
) -> List[tritonclient.http.InferInput]:
    inputs = []
    try:
        coord_inputs = {
            "pos_coords": pos_coords,
            "neg_coords": neg_coords,
            "labels": labels,
            "input_image": image
        }
        
        for name, data in coord_inputs.items():
            input_tensor = tritonclient.http.InferInput(
                name,
                data.shape,
                "INT64" if name != "input_image" else "UINT8"
            )
            input_tensor.set_data_from_numpy(
                data.astype(np.int64 if name != "input_image" else np.uint8)
            )
            inputs.append(input_tensor)
            
        return inputs
    except Exception as e:
        logger.error("SAM 입력 준비 실패", exc_info=e)
        raise RuntimeError(f"입력 준비 실패: {str(e)}")

def sam(
    image: np.ndarray,
    neg_coords: np.ndarray,
    pos_coords: np.ndarray,
    labels: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Image.Image]]:
    try:
        with cuda_memory_manager():
            triton_client = get_triton_client()
            image = np.array(image).copy()
            
            inputs = prepare_sam_inputs(image, neg_coords, pos_coords, labels)
            outputs = [
                tritonclient.http.InferRequestedOutput(name="mask", binary_data=False),
                tritonclient.http.InferRequestedOutput(name="segmented_image", binary_data=False)
            ]
            
            response = triton_client.infer(
                model_name="sam",
                model_version="1",
                inputs=inputs,
                outputs=outputs
            )
            
            return (
                image,
                response.as_numpy("mask"),
                Image.fromarray(response.as_numpy("segmented_image"))
            )
    except Exception as e:
        logger.error("SAM 처리 실패", exc_info=e)
        raise RuntimeError(f"세그멘테이션 실패: {str(e)}")

def preprocess_mask(mask: Union[Image.Image, np.ndarray]) -> Image.Image:
    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask)
    if mask.mode != 'L':
        mask = mask.convert('L')
    mask_array = np.array(mask)
    return Image.fromarray((mask_array > 128).astype(np.uint8) * 255)

def resize_for_inpainting(
    image: Image.Image,
    mask: Image.Image,
    target_size: Tuple[int, int] = (512, 512)
) -> Tuple[Image.Image, Image.Image, Tuple[int, int]]:
    original_size = image.size
    resized_image = image.resize(target_size, Image.Resampling.LANCZOS)
    resized_mask = mask.resize(target_size, Image.Resampling.NEAREST)
    return resized_image, resized_mask, original_size

def sd_inpaint(
    image: Image.Image,
    mask: Union[Image.Image, np.ndarray],
    inpaint_prompt: str,
    num_steps: int = 30,  # 이미지 생성 과정 반복 횟수 
    guidance_scale: float = 12.0,  # 프롬프트 충실도
    strength: float = 0.99  # 원본 이미지 변경 강도
) -> Image.Image:
    try:
        with cuda_memory_manager():
            device = get_device()
            
            # 마스크 전처리 개선
            mask = preprocess_mask(mask)
            mask = mask.filter(ImageFilter.GaussianBlur(radius=2))  # 경계 부드럽게 
        
            resized_image, resized_mask, original_size = resize_for_inpainting(image, mask)
            
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                use_safetensors=False
            ).to(device)
            
            if device == "mps":
                pipe.enable_attention_slicing()
                
            output = pipe(
                prompt=inpaint_prompt,
                image=resized_image,
                mask_image=resized_mask,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                strength=strength
            ).images[0]
            
            return output.resize(original_size, Image.Resampling.LANCZOS)
    except Exception as e:
        logger.error("인페인팅 실패", exc_info=e)
        raise RuntimeError(f"인페인팅 실패: {str(e)}")

def lama_cleaner(
    image: np.ndarray,
    mask: np.ndarray,
    device: Optional[str] = None
) -> Image.Image:
    try:
        with cuda_memory_manager():
            device = device or get_device()
            model = get_lama_cleaner()
            
            image = norm_img(image)
            mask = norm_img(mask)
            mask = (mask > 0) * 1
            
            image = torch.from_numpy(image).unsqueeze(0).to(device)
            mask = torch.from_numpy(mask).unsqueeze(0).to(device)
            
            inpainted = model(image, mask)
            result = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
            result = np.clip(result * 255, 0, 255).astype("uint8")
            
            return Image.fromarray(result)
    except Exception as e:
        logger.error("LaMa 클리닝 실패", exc_info=e)
        raise RuntimeError(f"이미지 클리닝 실패: {str(e)}")
