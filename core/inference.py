import sys
from typing import Optional, Tuple, List, Union
from contextlib import contextmanager
import logging
import numpy as np
import torch
import cv2
import gc
from PIL import Image
import tritonclient.http

from core.image_processing import ImageProcessor, ProcessingConfig, ModelType
from utils.lama_cleaner_helper import norm_img
from core.model_setup import (
    get_triton_client,
    get_sd_inpaint,
    get_lama_cleaner
)
from utils.device_utils import get_device
from utils.agent import PromptOptimizationAgent

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('image_editor.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ImageEditPipeline:
    def __init__(self):
        self.config = ProcessingConfig()
        self.image_processor = ImageProcessor(self.config)
        self.prompt_agent = PromptOptimizationAgent()
        self.device = get_device()

    @contextmanager
    def _cuda_memory_manager(self):
        """CUDA 메모리 관리를 위한 컨텍스트 매니저"""
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

    
    def _lama_cleaner(self, image: np.ndarray, mask: np.ndarray) -> Image.Image:
        """LaMa Cleaner를 사용한 객체 제거"""
        try:
            # CUDA 메모리 관리 컨텍스트 외부에서 이미지 전처리
            if image is None or mask is None:
                raise ValueError("Image and mask must not be None")
                
            # 이미지와 마스크의 형상 검증
            if len(image.shape) != 3 or image.shape[2] != 3:
                raise ValueError(f"Invalid image shape: {image.shape}, expected (H, W, 3)")
            if len(mask.shape) != 2:
                raise ValueError(f"Invalid mask shape: {mask.shape}, expected (H, W)")
                
            # 원본 크기 저장
            h, w = image.shape[:2]
            original_size = (w, h)
            
            with self._cuda_memory_manager():
                logger.info("Starting LaMa cleaning process")
                model = get_lama_cleaner()
                
                # 8의 배수로 패딩
                new_h = ((h + 7) // 8) * 8
                new_w = ((w + 7) // 8) * 8
                
                if new_h != h or new_w != w:
                    # 새 크기로 패딩된 이미지 생성
                    padded_image = np.zeros((new_h, new_w, 3), dtype=np.uint8)
                    padded_mask = np.zeros((new_h, new_w), dtype=np.uint8)
                    
                    # 원본 이미지 복사
                    padded_image[:h, :w] = image
                    padded_mask[:h, :w] = mask
                    
                    # 가장자리 픽셀로 패딩
                    if h < new_h:
                        padded_image[h:, :w] = image[-1:, :, :]
                    if w < new_w:
                        padded_image[:h, w:] = image[:, -1:, :]
                    if h < new_h and w < new_w:
                        padded_image[h:, w:] = image[-1:, -1:, :]
                    
                    image = padded_image
                    mask = padded_mask
                
                # 크기 제한 처리
                if max(new_h, new_w) > self.config.max_size:
                    scale = self.config.max_size / max(new_h, new_w)
                    scaled_h = int(new_h * scale) - int(new_h * scale) % 8
                    scaled_w = int(new_w * scale) - int(new_w * scale) % 8
                    
                    image = np.array(Image.fromarray(image).resize((scaled_w, scaled_h), Image.Resampling.LANCZOS))
                    mask = np.array(Image.fromarray(mask).resize((scaled_w, scaled_h), Image.Resampling.NEAREST))
                
                logger.debug(f"Processing size: {image.shape}")
                
                # 입력 정규화
                image = norm_img(image)
                mask = norm_img(mask)
                mask = (mask > 0) * 1
                
                # 텐서 변환
                image = torch.from_numpy(image).unsqueeze(0).to(self.device)
                mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)
                
                # 모델 추론
                with torch.no_grad():
                    inpainted = model(image, mask)
                    
                # 후처리
                result = inpainted[0].permute(1, 2, 0).cpu().numpy()
                result = np.clip(result * 255, 0, 255).astype("uint8")
                
                result_image = Image.fromarray(result)
                
                # 원본 크기로 크롭
                if result.shape[:2] != (h, w):
                    result_image = result_image.crop((0, 0, w, h))
                
                return self.image_processor.quality_manager.process_image(result_image)
            
        except Exception as e:
            logger.error("LaMa cleaning failed", exc_info=True)
            raise RuntimeError(f"Cleaning error: {str(e)}")

    def remove_object(
        self,
        image: Union[np.ndarray, Image.Image],
        mask: Union[np.ndarray, Image.Image],
    ) -> Image.Image:
        """객체 제거 파이프라인"""
        try:
            # 입력 정규화
            if isinstance(image, Image.Image):
                image = np.array(image)
            if isinstance(mask, Image.Image):
                mask = np.array(mask)
            
            # 마스크 전처리 - 단순 이진화만 수행
            mask = (mask > 127).astype(np.uint8) * 255
            
            # LaMa 모델로 객체 제거
            return self._lama_cleaner(image, mask)
                
        except Exception as e:
            logger.error("Object removal failed", exc_info=e)
            raise RuntimeError(f"Removal error: {str(e)}")

    def edit_object(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        num_steps: int = 70,
        guidance_scale: float = 8.5,
    ) -> Image.Image:
        """객체 편집 파이프라인"""
        try:
            with self._cuda_memory_manager():
                # 마스크 영역 분석
                mask_np = np.array(mask)
                y_indices, x_indices = np.nonzero(mask_np)
                
                if len(y_indices) == 0:
                    raise ValueError("No masked area found")
                
                # 패딩을 포함한 크롭 영역 계산
                padding = self.config.padding
                x_min = max(0, np.min(x_indices) - padding)
                x_max = min(mask_np.shape[1], np.max(x_indices) + padding)
                y_min = max(0, np.min(y_indices) - padding)
                y_max = min(mask_np.shape[0], np.max(y_indices) + padding)
                
                # 8의 배수로 조정
                x_min = (x_min // 8) * 8
                x_max = ((x_max + 7) // 8) * 8
                y_min = (y_min // 8) * 8
                y_max = ((y_max + 7) // 8) * 8
                
                # 영역 추출
                crop_area = (x_min, y_min, x_max, y_max)
                cropped_image = image.crop(crop_area)
                cropped_mask = mask.crop(crop_area)
                
                # SD 모델 입력 크기에 맞게 리사이징
                target_size = (512, 512)
                resized_image = cropped_image.resize(target_size, Image.Resampling.LANCZOS)
                resized_mask = cropped_mask.resize(target_size, Image.Resampling.NEAREST)
                
                # 프롬프트 최적화 및 모델 실행
                optimized_prompt = self.prompt_agent.optimize_prompt(prompt, resized_image, resized_mask)
                model = get_sd_inpaint()
                
                edited = model(
                    prompt=optimized_prompt,
                    image=resized_image,
                    mask_image=resized_mask,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                ).images[0]
                
                # 원본 크기로 복원
                edited = edited.resize(cropped_image.size, Image.Resampling.LANCZOS)
                
                # 결과 합성
                result = image.copy()
                result.paste(edited, (x_min, y_min))
                
                return result
                
        except Exception as e:
            logger.error("Object editing failed", exc_info=e)
            raise RuntimeError(f"Editing error: {str(e)}")

    def process_image(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: Optional[str] = None,
        **kwargs
    ) -> Image.Image:
        """통합 처리 인터페이스"""
        if prompt:
            return self.edit_object(image, mask, prompt, **kwargs)
        else:
            return self.remove_object(image, mask)


# Deprecated 함수들
def generate_and_add(*args, **kwargs):
    logger.warning("Deprecated: Use ImageEditPipeline.edit_object instead")
    raise NotImplementedError

def sd_inpaint(*args, **kwargs):
    logger.warning("Deprecated: Use ImageEditPipeline.process_image instead")
    raise NotImplementedError