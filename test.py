    # test_api.py
import requests
import base64
from PIL import Image
import numpy as np
import io

def create_test_mask(width=704, height=704):
    """테스트용 마스크 이미지 생성"""
    # 흰색 원형 마스크 생성 (중앙에 위치)
    mask = Image.new('L', (width, height), 0)
    for x in range(width):
        for y in range(height):
            # 중앙에 반지름 100픽셀의 원 그리기
            if (x - width/2)**2 + (y - height/2)**2 < 100**2:
                mask.putpixel((x, y), 255)
    return mask

def mask_to_base64(mask_image):
    """마스크 이미지를 base64로 변환"""
    buffer = io.BytesIO()
    mask_image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()

def test_image_edit_api():
    """이미지 편집 API 테스트"""
    # API 엔드포인트
    url = "http://localhost:8000/api/edit-image"
    
    # 테스트 이미지 URL
    test_images = {
        "landscape": "https://images.unsplash.com/photo-1506744038136-46273834b3fb",
        "cat": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba",
        "dog": "https://images.unsplash.com/photo-1517849845537-4d257902454a",
        "room": "https://images.unsplash.com/photo-1554995207-c18c203602cb"
    }
    
    # 테스트할 이미지 선택
    test_image_url = test_images["landscape"]  # 원하는 이미지로 변경 가능
    
    # 테스트 데이터 준비
    test_data = {
        "image_url": test_image_url,
        "mask_data": mask_to_base64(create_test_mask()),
        "prompt": "make this area blue with clouds",  # 원하는 프롬프트로 변경 가능
        "edit_type": "inpaint"
    }

    try:
        print(f"Testing with image: {test_image_url}")
        print(f"Using prompt: {test_data['prompt']}")
        
        # API 호출
        print("Making API request...")
        response = requests.post(url, json=test_data)
        
        # 응답 확인
        if response.status_code == 200:
            result = response.json()
            
            # 결과 이미지 저장
            if result["status"] == "success":
                img_data = base64.b64decode(result["edited_image"])
                result_image = Image.open(io.BytesIO(img_data))
                result_image.save("test_result.png")
                print("Success! Result saved as test_result.png")
            else:
                print(f"Error: {result.get('message', 'Unknown error')}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Error during test: {str(e)}")

if __name__ == "__main__":
    test_image_edit_api() 