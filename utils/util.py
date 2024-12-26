import os
import cv2
from PIL import Image
import re
import numpy as np
import random
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path  # [추가] 안전한 경로 처리

def save_uploaded_image(directory: str, uploaded_images: List) -> None:
    """업로드된 이미지들을 지정된 디렉토리에 저장"""
    # [추가] 확장자 검증
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
    directory_path = Path(directory)
    directory_path.mkdir(exist_ok=True)
    
    for image in uploaded_images:
        ext = Path(image.name).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {ext}")
            
        file_path = directory_path / image.name
        with open(file_path, 'wb') as f:
            f.write(image.getbuffer())

def save_uploaded_file(directory: str, file: object) -> None:
    """단일 업로드 파일을 저장"""
    directory_path = Path(directory)
    directory_path.mkdir(exist_ok=True)
    
    file_path = directory_path / file.name
    with open(file_path, 'wb') as f:
        f.write(file.getbuffer())

def save_dataframe(directory: str, file_name: str, df: 'pandas.DataFrame') -> None:
    """DataFrame을 CSV 파일로 저장"""
    directory_path = Path(directory)
    directory_path.mkdir(exist_ok=True)
    
    file_path = directory_path / file_name
    df.to_csv(file_path, index=False)

def box_label(
    image: np.ndarray,
    box: List[int],
    label: str = '',
    color: Tuple[int, int, int] = (128, 128, 128),
    txt_color: Tuple[int, int, int] = (255, 255, 255)
) -> None:
    """이미지에 바운딩 박스와 레이블을 그리는 함수"""
    try:
        lw = max(round(sum(image.shape) / 2 * 0.003), 2)
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        
        if label:
            tf = max(lw - 1, 1)
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]
            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)
            cv2.putText(
                image,
                label,
                (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                lw / 3,
                txt_color,
                thickness=tf,
                lineType=cv2.LINE_AA
            )
    except Exception as e:
        raise ValueError(f"Error drawing box label: {str(e)}")

def closest_multiple_of_8(number: int) -> int:
    """주어진 숫자에서 가장 가까운 8의 배수를 찾는 함수"""
    remainder = number % 8
    return number - remainder if remainder < 4 else number + (8 - remainder)

def resize_image(
    image: Image.Image,
    max_width: int,
    max_height: int
) -> Tuple[Image.Image, float]:
    """이미지를 최대 크기에 맞게 리사이즈"""
    try:
        width, height = image.size
        
        if width > max_width or height > max_height:
            width_ratio = max_width / width
            height_ratio = max_height / height
            
            resize_ratio = min(width_ratio, height_ratio)
            new_width = closest_multiple_of_8(int(width * resize_ratio))
            new_height = closest_multiple_of_8(int(height * resize_ratio))
            
            resized_img = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            return resized_img, resize_ratio
            
        return image, 1.0
    except Exception as e:
        raise ValueError(f"Error resizing image: {str(e)}")

def plot_bboxes(
    image: np.ndarray,
    box: List[int],
    score: float,
    label: str,
    label_index: int
) -> None:
    """객체 탐지 결과를 이미지에 시각화"""
    colors = [(89, 161, 197), (67, 161, 255), (128, 128, 128)]  # 기본 색상
    label = f"{label} {score}%"
    color = colors[label_index % len(colors)]
    box_label(image, box, label, color)

def string_to_dictionary(text: str) -> Dict[str, str]:
    """텍스트를 딕셔너리로 변환"""
    try:
        sections = text.split('\n\n')
        result_dict = {}
        
        for section in sections:
            parts = section.split(':')
            if len(parts) >= 2:
                key = parts[0].strip()
                value = ':'.join(parts[1:]).strip()
                result_dict[key] = value
                
        return result_dict
    except Exception as e:
        raise ValueError(f"Error converting string to dictionary: {str(e)}")

def label_select(text: str) -> str:
    """'start:'와 ':end' 사이의 텍스트를 추출"""
    try:
        result = re.search(r'start:(.*?):end', text)
        if result:
            return result.group(1)
        raise ValueError("Pattern 'start:...:end' not found in text")
    except Exception as e:
        raise ValueError(f"Error selecting label: {str(e)}")

def xywh2xyxy(xywh: np.ndarray) -> np.ndarray:
    """[x,y,width,height] 형식을 [x1,y1,x2,y2] 형식으로 변환"""
    try:
        xyxy = xywh.copy()
        xyxy[:,2] = xywh[:, 0] + xywh[:, 2]
        xyxy[:,3] = xywh[:, 1] + xywh[:, 3]
        return xyxy
    except Exception as e:
        raise ValueError(f"Error converting coordinates: {str(e)}")

def combine_masks(mask_list: List[np.ndarray]) -> np.ndarray:
    """여러 마스크를 하나로 결합"""
    if not mask_list:
        raise ValueError("At least one mask required")
    
    try:
        combined_mask = mask_list[0]
        for mask in mask_list[1:]:
            combined_mask = np.logical_or(combined_mask, mask)
        
        return combined_mask[np.newaxis, :, :]
    except Exception as e:
        raise ValueError(f"Error combining masks: {str(e)}")

def random_hex_color() -> str:
    """랜덤 색상 선택"""
    color_list = ["#caff70", "#07ccff", "#fa0087", "#f88379", "#7d37ff"]
    return random.choice(color_list)

def dilate_mask(
    mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1
) -> np.ndarray:
    """마스크 팽창"""
    try:
        if kernel_size % 2 == 0:
            kernel_size += 1  # 커널 크기는 홀수여야 함
            
        mask = mask.astype(np.uint8).squeeze()
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.dilate(mask, kernel, iterations=iterations)
    except Exception as e:
        raise ValueError(f"Error dilating mask: {str(e)}")