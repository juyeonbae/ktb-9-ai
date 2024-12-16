import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Optional, Union, Any
import logging
from enum import Enum
from dataclasses import dataclass

from utils.image_quality import ImageQualityManager, ProcessingConfig

logger = logging.getLogger(__name__)

class ModelType(Enum):
    LAMA = "lama"
    SD2 = "sd2"

class ImageProcessor:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.quality_manager = ImageQualityManager(config)

    def preprocess_mask_for_removal(self, mask: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """객체 제거를 위한 간단한 마스크 전처리"""
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
            
        if len(mask.shape) == 3:
            mask = mask[..., 0]  # 첫 번째 채널만 사용
            
        # 단순 이진화
        return (mask > 127).astype(np.uint8) * 255

    def preprocess_mask_for_generation(
        self,
        mask: Image.Image,
        feather_amount: int = 3
    ) -> Image.Image:
        """객체 생성을 위한 마스크 전처리"""
        if mask.mode != 'L':
            mask = mask.convert('L')
            
        # 마스크 부드럽게 처리
        mask_np = np.array(mask)
        kernel = np.ones((3, 3), np.uint8)
        
        # 노이즈 제거 및 확장
        denoised = cv2.medianBlur(mask_np, 3)
        dilated = cv2.dilate(denoised, kernel, iterations=2)
        
        # 경계 스무딩
        smoothed = cv2.GaussianBlur(
            dilated,
            (feather_amount*2+1, feather_amount*2+1),
            sigmaX=feather_amount,
            sigmaY=feather_amount
        )
        
        return Image.fromarray(smoothed)

    def get_crop_area(
        self,
        mask: Union[np.ndarray, Image.Image],
        padding: int = 32
    ) -> Dict[str, int]:
        """마스크 영역 기준으로 크롭 영역 계산"""
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
            
        # 마스크 영역 찾기
        y_indices, x_indices = np.nonzero(mask)
        
        if len(y_indices) == 0:
            return {"cropped": False}
            
        # 패딩을 포함한 영역 계산
        x_min = max(0, np.min(x_indices) - padding)
        x_max = min(mask.shape[1], np.max(x_indices) + padding)
        y_min = max(0, np.min(y_indices) - padding)
        y_max = min(mask.shape[0], np.max(y_indices) + padding)
        
        # 8의 배수로 조정
        x_min = (x_min // 8) * 8
        x_max = ((x_max + 7) // 8) * 8
        y_min = (y_min // 8) * 8
        y_max = ((y_max + 7) // 8) * 8
        
        return {
            "cropped": True,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max
        }

    def blend_images(
        self,
        original: Union[np.ndarray, Image.Image],
        edited: Union[np.ndarray, Image.Image],
        mask: Union[np.ndarray, Image.Image],
        feather_amount: int = 5
    ) -> Image.Image:
        """편집된 영역 블렌딩"""
        # 입력 정규화
        if isinstance(original, Image.Image):
            original = np.array(original)
        if isinstance(edited, Image.Image):
            edited = np.array(edited)
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
            
        # 마스크 페더링
        mask_float = mask.astype(float) / 255.0
        feathered = cv2.GaussianBlur(
            mask_float,
            (feather_amount*2+1, feather_amount*2+1),
            sigmaX=feather_amount,
            sigmaY=feather_amount
        )
        
        # 색상 일관성 맞추기
        edited_lab = cv2.cvtColor(edited, cv2.COLOR_RGB2LAB)
        original_lab = cv2.cvtColor(original, cv2.COLOR_RGB2LAB)
        
        # 마스크 주변 영역의 컬러 통계 계산
        kernel = np.ones((7,7), np.uint8)  # 영역 확장
        boundary = cv2.dilate(mask, kernel) - mask
        
        # 각 채널별 정규화 및 매칭
        for i in range(3):
            if np.any(boundary):
                # 주변 영역의 통계
                ref_mean = np.mean(original_lab[boundary > 0, i])
                ref_std = np.std(original_lab[boundary > 0, i])
                
                # 편집 영역의 통계
                target_mean = np.mean(edited_lab[mask > 0, i])
                target_std = np.std(edited_lab[mask > 0, i])
                
                if target_std > 0:
                    # 색상 매칭 강화
                    edited_lab[..., i] = (
                        (edited_lab[..., i] - target_mean)
                        * (ref_std / target_std) + ref_mean
                    ).clip(0, 255)
        
        # 최종 블렌딩
        edited_rgb = cv2.cvtColor(edited_lab, cv2.COLOR_LAB2RGB)
        
        # Poisson 블렌딩 적용
        mask_bool = (mask > 127).astype(np.uint8) * 255
        center = (mask.shape[1]//2, mask.shape[0]//2)
        
        try:
            blended = cv2.seamlessClone(
                edited_rgb.astype(np.uint8),
                original.astype(np.uint8),
                mask_bool,
                center,
                cv2.MIXED_CLONE
            )
        except:
            # Poisson 블렌딩 실패시 일반 블렌딩으로 폴백
            blended = original.copy()
            for c in range(3):
                blended[..., c] = (
                    original[..., c] * (1 - feathered) +
                    edited_rgb[..., c] * feathered
                )

        return Image.fromarray(blended.astype(np.uint8))

    def determine_model(self, prompt: Optional[str] = None) -> ModelType:
        """작업 유형에 따른 모델 결정"""
        if not prompt:
            return ModelType.LAMA
            
        removal_keywords = ["지워", "지워줘", "제거", "삭제", "없애"]
        if any(keyword in prompt for keyword in removal_keywords):
            return ModelType.LAMA
            
        return ModelType.SD2
    
    def analyze_image_style(self, image: Image.Image) -> Dict[str, Any]:
        """이미지의 전체적인 스타일을 분석합니다."""
        try:
            img_array = np.array(image)
            
            # 1. 전체적인 톤 분석
            img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l_channel = img_lab[:,:,0]
            tone_mean = np.mean(l_channel)
            tone_std = np.std(l_channel)
            
            # 2. 주요 색상 추출
            img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            hist_hue = cv2.calcHist([img_hsv], [0], None, [180], [0,180])
            dominant_hue = np.argmax(hist_hue)
            
            # 3. 질감 특성 추출
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            gabor_responses = []
            for theta in [0, 45, 90, 135]:
                kernel = cv2.getGaborKernel((21, 21), 5.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
                filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
                gabor_responses.append(np.mean(filtered))
            
            return {
                "tone": {
                    "mean": float(tone_mean),
                    "std": float(tone_std)
                },
                "color": {
                    "dominant_hue": float(dominant_hue)
                },
                "texture": {
                    "gabor_features": gabor_responses
                }
            }
            
        except Exception as e:
            logger.error(f"Style analysis failed: {str(e)}")
            return {}