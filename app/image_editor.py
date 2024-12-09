import base64
from io import BytesIO
from PIL import Image
import numpy as np
import requests
from urllib.parse import urljoin
from typing import Optional, Dict, Any

class ImageEditorAPI:
    """이미지 편집 API 클라이언트"""
    
    def __init__(self, base_url: str):
        """
        Args:
            base_url (str): 백엔드 API 서버의 기본 URL
        """
        self.base_url = base_url
        self.endpoints = {
            "edit_image": "/api/edit-image"
        }
    
    def _encode_image(self, image: Image.Image) -> str:
        """이미지를 base64로 인코딩"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def _encode_mask(self, mask: np.ndarray) -> str:
        """마스크를 base64로 인코딩"""
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        buffered = BytesIO()
        mask_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def edit_image(
    self, 
    image: Image.Image, 
    prompt: str, 
    mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        try:
            # 요청 데이터 준비
            payload = {
                "image_data": self._encode_image(image),
                "prompt": prompt
            }
            
            # 디버깅을 위한 로그 추가
            print(f"Sending request with prompt: {prompt}")
            print(f"Image size: {image.size}")
            
            if mask is not None and mask.any():
                payload["mask_data"] = self._encode_mask(mask)
                print(f"Mask shape: {mask.shape}")
            
            # API 요청
            response = requests.post(
                urljoin(self.base_url, self.endpoints["edit_image"]),
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30  # 타임아웃 설정
            )
            
            # 응답 상세 로깅
            print(f"Response status code: {response.status_code}")
            if response.status_code != 200:
                print(f"Error response content: {response.text}")
                
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {str(e)}")
            raise Exception(f"API 요청 실패: {str(e)}")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            raise Exception(f"처리 중 에러 발생: {str(e)}")
        """
        이미지 편집 요청을 보내는 메서드
        
        Args:
            image (PIL.Image): 편집할 원본 이미지
            prompt (str): 편집 지시사항
            mask (numpy.ndarray, optional): 편집 영역 마스크 (0과 1로 구성된 2D array)
            
        Returns:
            dict: API 응답 데이터 (edited_image_url 등 포함)
            
        Raises:
            RequestException: API 요청 실패시
            ValueError: 잘못된 입력값
        """

# 사용 예시
"""
# API 클라이언트 초기화
api_client = ImageEditorAPI("http://your-backend-url")

# 이미지 편집 요청
try:
    # 이미지와 마스크 준비
    image = Image.open("example.jpg")
    mask = np.zeros((image.height, image.width), dtype=np.uint8)
    mask[100:200, 100:200] = 1  # 예시 마스크
    
    # 편집 요청
    result = api_client.edit_image(
        image=image,
        prompt="Remove the background",
        mask=mask
    )
    
    # 결과 처리
    edited_image_url = result["edited_image_url"]
    print(f"편집된 이미지 URL: {edited_image_url}")
    
except Exception as e:
    print(f"에러 발생: {str(e)}")
"""