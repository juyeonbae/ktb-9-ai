from PIL import Image, ImageEnhance
import numpy as np
from typing import Tuple, Optional

class ImageQualityManager:
    def __init__(self):
        self.min_resolution = 512
        self.max_resolution = 2048
        self.quality_threshold = 0.5

    def check_image_quality(self, image: Image.Image) -> float:
        """이미지 품질 점수 계산"""
        # 해상도 체크
        width, height = image.size
        resolution_score = min(1.0, (width * height) / (self.min_resolution ** 2))
        
        # 선명도 체크
        gray = image.convert('L')
        array = np.array(gray)
        sharpness = np.average(np.abs(np.diff(array)))
        sharpness_score = min(1.0, sharpness / 100)
        
        # 최종 품질 점수
        return (resolution_score + sharpness_score) / 2

    def enhance_image(self, image: Image.Image) -> Image.Image:
        """이미지 품질 향상"""
        # 선명도 향상
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.2)
        
        # 대비 향상
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)
        
        return image

    def resize_if_needed(
        self, 
        image: Image.Image,
        target_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[Image.Image, bool]:
        """필요한 경우에만 리사이징"""
        width, height = image.size
        was_resized = False

        if target_size:
            target_width, target_height = target_size
            if (width, height) != (target_width, target_height):
                image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
                was_resized = True
        else:
            # 너무 큰 이미지 크기 제한
            if width > self.max_resolution or height > self.max_resolution:
                ratio = min(self.max_resolution / width, self.max_resolution / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                was_resized = True

        return image, was_resized

    def process_image(
        self, 
        image: Image.Image, 
        target_size: Optional[Tuple[int, int]] = None
    ) -> Image.Image:
        """이미지 처리 파이프라인"""
        # 품질 체크
        initial_quality = self.check_image_quality(image)
        
        # 리사이징 (필요한 경우)
        image, was_resized = self.resize_if_needed(image, target_size)
        
        # 리사이징된 경우 품질 향상
        if was_resized:
            image = self.enhance_image(image)
        
        # 최종 품질 체크
        final_quality = self.check_image_quality(image)
        
        # 품질이 떨어진 경우 추가 향상
        if final_quality < initial_quality:
            image = self.enhance_image(image)
        
        return image