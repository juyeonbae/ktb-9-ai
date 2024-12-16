from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path
import logging

from utils.inference import ImageEditPipeline
from utils.image_processing import ImageProcessor
from utils.image_quality import ProcessingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_mask(image_size: tuple) -> Image.Image:
    """테스트용 마스크 생성"""
    mask = Image.new('L', image_size, 0)  # 검은색 마스크
    draw = ImageDraw.Draw(mask)
    
    # 이미지 중앙에 흰색 사각형 그리기
    width, height = image_size
    rect_size = min(width, height) // 4
    left = (width - rect_size) // 2
    top = (height - rect_size) // 2
    right = left + rect_size
    bottom = top + rect_size
    
    draw.rectangle([left, top, right, bottom], fill=255)  # 흰색 영역 (마스크)
    return mask

def test_image_pipeline():
    # 파이프라인 초기화
    pipeline = ImageEditPipeline()
    
    try:
        # 1. 이미지 로드
        image_path = "에펠탑.jpg"
        image = Image.open(image_path)
        logger.info(f"Loaded image size: {image.size}")
        
        # 2. 테스트용 마스크 생성
        mask = create_test_mask(image.size)
        logger.info(f"Created mask size: {mask.size}")
        
        # 마스크 저장 (디버깅용)
        mask.save("test_mask.png")
        logger.info("Saved test mask to test_mask.png")
        
        # 3. 객체 제거 테스트
        logger.info("Testing object removal...")
        removal_result = pipeline.remove_object(image, mask)
        logger.info(f"Removal result size: {removal_result.size}")
        
        # 결과 저장
        removal_result.save("removal_test_result.png")
        logger.info("Saved removal result to removal_test_result.png")
        
        # 4. 객체 편집 테스트
        logger.info("Testing object editing...")
        edit_prompt = "red sports car"
        edit_result = pipeline.edit_object(image, mask, edit_prompt)
        logger.info(f"Edit result size: {edit_result.size}")
        
        # 결과 저장
        edit_result.save("edit_test_result.png")
        logger.info("Saved edit result to edit_test_result.png")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())  # 상세한 에러 정보 출력
        return False

if __name__ == "__main__":
    print("Starting image pipeline test...")
    success = test_image_pipeline()
    print(f"Test {'succeeded' if success else 'failed'}")