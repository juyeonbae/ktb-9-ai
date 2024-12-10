from typing import List, Optional, Dict, Any
from PIL import Image
import numpy as np

class ImageStateManager:
    """
    이미지 상태 관리 클래스
    Streamlit의 session_state를 대체하여 이미지 편집 상태를 관리
    """
    def __init__(self, initial_image: Optional[Image.Image] = None):
        self.state = {
            "inference_image": [initial_image] if initial_image else [],
            "image_state": 0,
            "num_coord": 0,
            "sam_image": None,
            "mask": None,
            "coord": False,
            "freedraw": False,
            "canvas": None,
            "text": ""
        }

    def backward_inference_image(self) -> Dict[str, Any]:
        """
        이전 이미지 상태로 되돌리는 함수
        Returns:
            Dict: 현재 상태 정보를 담은 딕셔너리
        """
        if self.state["image_state"] > 0:
            self.state["image_state"] -= 1
            self.state["num_coord"] = 0
            return {"status": "success", "message": "Moved to previous image"}
        return {"status": "warning", "message": "This is First Image!"}

    def forward_inference_image(self) -> Dict[str, Any]:
        """
        다음 이미지 상태로 이동하는 함수
        Returns:
            Dict: 현재 상태 정보를 담은 딕셔너리
        """
        if len(self.state["inference_image"]) - 1 > self.state["image_state"]:
            self.state["image_state"] += 1
            self.state["num_coord"] = 0
            
            if self.state["canvas"] is not None:
                try:
                    if 'raw' in self.state["canvas"]:
                        self.state["canvas"]['raw']["objects"] = []
                except Exception as e:
                    return {"status": "error", "message": f"Error handling canvas state: {str(e)}"}
            return {"status": "success", "message": "Moved to next image"}
        return {"status": "warning", "message": "This is Last Image!"}

    def reset_inference_image(self) -> Dict[str, Any]:
        """
        이미지 상태를 초기 상태로 리셋하는 함수
        Returns:
            Dict: 현재 상태 정보를 담은 딕셔너리
        """
        if len(self.state["inference_image"]) > 0:
            self.state["inference_image"] = [self.state["inference_image"][0]]
            self.state["image_state"] = 0
            self.state["num_coord"] = 0
            return {"status": "success", "message": "Image state reset successfully"}
        return {"status": "warning", "message": "No images to reset"}

    def reset_text(self) -> Dict[str, Any]:
        """
        텍스트 입력을 초기화하는 함수
        Returns:
            Dict: 현재 상태 정보를 담은 딕셔너리
        """
        self.state["text"] = ""
        return {"status": "success", "message": "Text reset successfully"}

    def reset_coord(self) -> Dict[str, Any]:
        """
        좌표 및 관련 상태를 초기화하는 함수
        Returns:
            Dict: 현재 상태 정보를 담은 딕셔너리
        """
        if "canvas" in self.state:
            self.state["canvas"] = None

        self.state["num_coord"] = 0
        self.state["sam_image"] = None
        self.state["mask"] = None
        self.state["coord"] = False
        self.state["freedraw"] = False
        return {"status": "success", "message": "Coordinates reset successfully"}

    def get_current_image(self) -> Optional[Image.Image]:
        """
        현재 상태의 이미지를 반환하는 함수
        Returns:
            Optional[Image.Image]: 현재 이미지 또는 None
        """
        if self.state["inference_image"] and len(self.state["inference_image"]) > self.state["image_state"]:
            return self.state["inference_image"][self.state["image_state"]]
        return None

    def add_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        새로운 이미지를 상태에 추가하는 함수
        Args:
            image (Image.Image): 추가할 이미지
        Returns:
            Dict: 현재 상태 정보를 담은 딕셔너리
        """
        self.state["inference_image"].append(image)
        return {"status": "success", "message": "Image added successfully"}

    def get_state(self) -> Dict[str, Any]:
        """
        현재 전체 상태를 반환하는 함수
        Returns:
            Dict: 현재 상태 정보를 담은 딕셔너리
        """
        return self.state