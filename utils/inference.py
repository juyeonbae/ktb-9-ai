import sys
from typing import Optional, Tuple, List, Union
from contextlib import contextmanager
import logging
import numpy as np
import torch
import cv2
import gc  # [추가] 가비지 컬렉션을 위한 import
from PIL import Image, ImageOps, ImageFilter
import tritonclient.http
from diffusers import StableDiffusionInpaintPipeline
from utils.prompt_processor import PromptProcessor
from utils.image_quality import ImageQualityManager  # [추가] 품질 관리자 import

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

# [개선] 상수 정의
DEFAULT_TARGET_SIZE = (768, 768)  # 기본 타겟 크기 통일
MAX_SIZE = 768  # 최대 처리 크기
QUALITY_THRESHOLD = 0.7  # 품질 임계값

# [개선] CUDA 메모리 관리
@contextmanager
def cuda_memory_manager():
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()  # [추가] 가비지 컬렉션 수행

# [추가] 이미지 컨텍스트 분석 함수
def analyze_image_context(image: Image.Image, mask: Image.Image) -> str:
    """이미지 컨텍스트 분석"""
    try:
        # 이미지 크기와 비율 분석
        width, height = image.size
        aspect_ratio = width / height
        
        # 마스크 위치 분석
        mask_array = np.array(mask.convert('L'))
        mask_indices = np.where(mask_array > 128)
        if len(mask_indices[0]) == 0:
            return ""
            
        # 마스크 중심점 계산
        center_y = np.mean(mask_indices[0]) / height
        center_x = np.mean(mask_indices[1]) / width
        
        # 위치 기반 컨텍스트 생성
        position_contexts = []
        if center_y < 0.3:
            position_contexts.append("in the upper part")
        elif center_y > 0.7:
            position_contexts.append("in the lower part")
            
        if center_x < 0.3:
            position_contexts.append("on the left side")
        elif center_x > 0.7:
            position_contexts.append("on the right side")
        else:
            position_contexts.append("in the center")
        
        # 크기 분석
        mask_area = len(mask_indices[0]) / (width * height)
        if mask_area < 0.1:
            size_context = "small"
        elif mask_area > 0.3:
            size_context = "large"
        else:
            size_context = "medium-sized"
            
        context = f"{size_context} object {', '.join(position_contexts)} of the scene"
        logger.debug(f"Generated context: {context}")
        
        return context
        
    except Exception as e:
        logger.error(f"Context analysis failed: {str(e)}")
        return ""

# [개선] 모델 선택 로직을 별도 함수로 분리
def select_model(
    prompt: str,
    is_removal: bool,
    is_generation: bool
) -> str:
    """작업 유형에 따른 적절한 모델 선택"""
    if is_removal:
        return "lama"
    elif is_generation:
        return "sd2"
    return "sd1"

# [개선] 이미지 전처리 함수들 통합 및 개선
def preprocess_image_and_mask(
    image: Image.Image,
    mask: Union[Image.Image, np.ndarray],
    target_size: Optional[Tuple[int, int]] = None,
    for_generation: bool = False
) -> Tuple[Image.Image, Image.Image, Tuple[int, int]]:
    """이미지와 마스크 전처리 통합 함수"""
    try:
        # [추가] 품질 관리자 초기화
        quality_manager = ImageQualityManager()
        
        # 원본 크기 저장
        original_size = image.size
        target_size = target_size or DEFAULT_TARGET_SIZE
        
        # 이미지 품질 체크 및 처리
        image = quality_manager.process_image(image)
        
        # 마스크 전처리
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)
            
        if mask.mode != 'L':
            mask = mask.convert('L')
            
        # 마스크 이진화 및 경계 처리
        mask_array = np.array(mask)
        binary_mask = (mask_array > 128).astype(np.uint8) * 255
        mask = Image.fromarray(binary_mask)
        
        if for_generation:
            # 생성을 위한 추가 마스크 처리
            mask = mask.filter(ImageFilter.MaxFilter(5))
            blur_radius = 3
        else:
            # 일반 편집을 위한 마스크 처리
            blur_radius = 2
            
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # 크기 조정 로직
        if target_size:
            # 타겟 크기가 지정된 경우만 리사이즈
            temp_image = image.resize(target_size, Image.Resampling.LANCZOS)
            temp_mask = mask.resize(target_size, Image.Resampling.NEAREST)
        else:
            # 타겟 크기가 없으면 원본 크기 유지
            temp_image = image
            temp_mask = mask

        logger.debug(f"Processed image size: {temp_image.size}")
        
        return temp_image, temp_mask, original_size
        
    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        raise RuntimeError(f"Preprocessing error: {str(e)}")

# [개선] 생성 함수 개선
def generate_and_add(
    image: Image.Image,
    mask: Union[Image.Image, np.ndarray],
    prompt: str,
    num_steps: int = 75,
    guidance_scale: float = 12.0,
) -> Image.Image:
    try:
        with cuda_memory_manager():
            logger.info("Starting image generation process")
            device = get_device()
            
            # 전처리
            image, mask, original_size = preprocess_image_and_mask(
                image, mask, DEFAULT_TARGET_SIZE, for_generation=True
            )
            
            # 컨텍스트 분석 및 프롬프트 강화
            context = analyze_image_context(image, mask)
            enhanced_prompt = f"{prompt}, {context}, highly detailed, realistic lighting"
            logger.debug(f"Enhanced prompt: {enhanced_prompt}")
            
            # 모델 설정
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-inpainting",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                use_safetensors=False
            ).to(device)
            
            if device == "mps":
                pipe.enable_attention_slicing()
            
            # 네거티브 프롬프트
            negative_prompt = (
                "blurry, low quality, distorted, deformed, ugly, "
                "bad anatomy, watermark, signature, poorly drawn"
            )
            
            # 생성
            output = pipe(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=1
            ).images[0]
            
            # 후처리
            quality_manager = ImageQualityManager()
            output = quality_manager.process_image(output)
            
            logger.debug(f"Generated image size: {output.size}")
            logger.debug(f"Original size to restore: {original_size}")
            
            # 크기가 변경되었다면 원본 크기로 강제 복원
            if output.size != original_size:
                logger.info(f"Resizing from {output.size} to {original_size}")
                output = output.resize(original_size, Image.Resampling.LANCZOS)
            
            return output
            
    except Exception as e:
        logger.error("Generation failed", exc_info=e)
        raise RuntimeError(f"Generation error: {str(e)}")

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


# [개선] LaMa Cleaner 함수 개선
def lama_cleaner(
    image: np.ndarray,
    mask: np.ndarray,
    device: Optional[str] = None
) -> Image.Image:
    try:
        with cuda_memory_manager():
            logger.info("Starting LaMa cleaning process")
            device = device or get_device()
            model = get_lama_cleaner()
            
            # 이미지 크기 관리
            h, w = image.shape[:2]
            original_size = (w, h)
            
            # 크기 제한 및 조정
            if max(h, w) > MAX_SIZE:
                scale = MAX_SIZE / max(h, w)
                new_h = int(h * scale) - int(h * scale) % 8  # 8의 배수로 조정
                new_w = int(w * scale) - int(w * scale) % 8
                
                image = np.array(Image.fromarray(image).resize((new_w, new_h), Image.Resampling.LANCZOS))
                mask = np.array(Image.fromarray(mask).resize((new_w, new_h), Image.Resampling.NEAREST))
                
            logger.debug(f"Processing size: {image.shape}")
            
            # 이미지 정규화 및 텐서 변환
            image = norm_img(image)
            mask = norm_img(mask)
            mask = (mask > 0) * 1
            
            image = torch.from_numpy(image).unsqueeze(0).to(device)
            mask = torch.from_numpy(mask).unsqueeze(0).to(device)
            
            # 인페인팅 수행
            with torch.no_grad():
                inpainted = model(image, mask)
                
            result = inpainted[0].permute(1, 2, 0).cpu().numpy()
            result = np.clip(result * 255, 0, 255).astype("uint8")
            
            # 품질 관리 및 크기 복원
            quality_manager = ImageQualityManager()
            result_image = Image.fromarray(result)
            result_image = quality_manager.process_image(result_image)
            
            if result.shape[:2][::-1] != original_size:
                result_image = result_image.resize(original_size, Image.Resampling.LANCZOS)
            
            return result_image
            
    except Exception as e:
        logger.error("LaMa cleaning failed", exc_info=e)
        raise RuntimeError(f"Cleaning error: {str(e)}")

# [개선] 메인 인페인팅 함수 개선
def sd_inpaint(
    image: Image.Image,
    mask: Union[Image.Image, np.ndarray],
    inpaint_prompt: str,
    num_steps: int = 50,
    guidance_scale: float = 9.0,
    strength: float = 0.99
) -> Image.Image:
    try:
        with cuda_memory_manager():
            # 작업 유형 확인
            is_removal = any(word in inpaint_prompt for word in ["지워", "지워줘", "제거", "삭제", "없애"])
            is_generation = any(word in inpaint_prompt for word in ["추가", "생성", "넣어", "바꿔"])
            
            # 모델 선택
            model_type = select_model(inpaint_prompt, is_removal, is_generation)
            logger.info(f"Selected model type: {model_type}")
            
            # 이미지와 마스크 전처리
            preprocessed_image, preprocessed_mask, _ = preprocess_image_and_mask(
                image, mask, for_generation=(model_type == "sd2")
            )
            
            # 모델별 처리
            if model_type == "lama":
                logger.info("Using LaMa Cleaner for removal")
                return lama_cleaner(
                    image=np.array(preprocessed_image),
                    mask=np.array(preprocessed_mask),
                    device=get_device()
                )
            elif model_type == "sd2":
                logger.info("Using SD2 for generation")
                prompt_processor = PromptProcessor()
                enhanced_prompt = prompt_processor.process_generation_prompt(inpaint_prompt)
                return generate_and_add(preprocessed_image, preprocessed_mask, enhanced_prompt)
            else:
                logger.info("Using SD1 for general editing")
                return process_with_sd1(
                    preprocessed_image, preprocessed_mask, inpaint_prompt,
                    num_steps, guidance_scale, strength
                )
                
    except Exception as e:
        logger.error("Inpainting failed", exc_info=e)
        raise RuntimeError(f"Inpainting error: {str(e)}")
    
    
# [추가] SD1 처리를 위한 별도 함수
def process_with_sd1(
    image: Image.Image,
    mask: Union[Image.Image, np.ndarray],
    prompt: str,
    num_steps: int,
    guidance_scale: float,
    strength: float
) -> Image.Image:
    try:
        device = get_device()
        
        # 프롬프트 처리
        prompt_processor = PromptProcessor()
        processed_prompt = prompt_processor.process(prompt)
        logger.debug(f"Processed prompt: {processed_prompt}")
        
        # 이미지 및 마스크 전처리
        image, mask, original_size = preprocess_image_and_mask(
            image, mask, DEFAULT_TARGET_SIZE
        )
        
        # 모델 설정
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_safetensors=False
        ).to(device)
        
        if device == "mps":
            pipe.enable_attention_slicing()
        
        # 이미지 생성
        output = pipe(
            prompt=processed_prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            num_images_per_prompt=1
        ).images[0]
        
        # 품질 관리 및 크기 복원
        quality_manager = ImageQualityManager()
        output = quality_manager.process_image(output)
        return output.resize(original_size, Image.Resampling.LANCZOS)
        
    except Exception as e:
        logger.error("SD1 processing failed", exc_info=e)
        raise RuntimeError(f"SD1 processing error: {str(e)}")

# [추가] 유틸리티 함수들
def log_image_info(image: Image.Image, stage: str):
    """이미지 정보 로깅"""
    logger.debug(f"[{stage}] Image size: {image.size}, Mode: {image.mode}")

def check_inputs(
    image: Image.Image,
    mask: Union[Image.Image, np.ndarray],
    prompt: str
) -> None:
    """입력 유효성 검사"""
    if not isinstance(image, Image.Image):
        raise ValueError("Invalid image input")
    if not prompt:
        raise ValueError("Empty prompt")
    if image.size[0] < 64 or image.size[1] < 64:
        raise ValueError("Image too small")