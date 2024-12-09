import os
import cv2
from PIL import Image
import re
import numpy as np
import random

def save_uploaded_image(directory, uploaded_images):
    """
    업로드된 이미지들을 지정된 디렉토리에 저장
    
    Args:
        directory (str): 저장할 디렉토리 경로
        uploaded_images (list): 업로드된 이미지 파일 리스트
    """
    os.makedirs(directory, exist_ok=True)
    
    for image in uploaded_images:
        with open(os.path.join(directory, image.name), 'wb') as f:
            f.write(image.getbuffer())

def save_uploaded_file(directory, file):
    """
    단일 업로드 파일을 저장
    
    Args:
        directory (str): 저장할 디렉토리 경로
        file: 업로드된 파일 객체
    """
    os.makedirs(directory, exist_ok=True)
    
    with open(os.path.join(directory, file.name), 'wb') as f:
        f.write(file.getbuffer())

def save_dataframe(directory, file_name, df):
    """
    DataFrame을 CSV 파일로 저장
    
    Args:
        directory (str): 저장할 디렉토리 경로
        file_name (str): 저장할 파일명
        df (pandas.DataFrame): 저장할 DataFrame
    """
    os.makedirs(directory, exist_ok=True)
    df.to_csv(os.path.join(directory, file_name), index=False)

def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    """
    이미지에 바운딩 박스와 레이블을 그리는 함수
    
    Args:
        image (numpy.ndarray): 대상 이미지
        box (list): 박스 좌표 [x1, y1, x2, y2]
        label (str): 표시할 레이블
        color (tuple): 박스 색상 (B,G,R)
        txt_color (tuple): 텍스트 색상 (B,G,R)
    """
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)  # 선 두께 계산
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))  # 박스 좌표
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    
    if label:
        tf = max(lw - 1, 1)  # 폰트 두께
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # 텍스트 크기
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # 텍스트 배경
        cv2.putText(image,
                   label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                   0,
                   lw / 3,
                   txt_color,
                   thickness=tf,
                   lineType=cv2.LINE_AA)

def closest_multiple_of_8(number):
    """
    주어진 숫자에서 가장 가까운 8의 배수를 찾는 함수
    
    Args:
        number (int): 입력 숫자
    Returns:
        int: 가장 가까운 8의 배수
    """
    remainder = number % 8
    
    if remainder < 4:
        result = number - remainder
    else:
        result = number + (8 - remainder)
    
    return result

def resize_image(image, max_width, max_height):
    """
    이미지를 최대 크기에 맞게 리사이즈하는 함수
    
    Args:
        image (PIL.Image): 원본 이미지
        max_width (int): 최대 너비
        max_height (int): 최대 높이
        
    Returns:
        tuple: (리사이즈된 이미지, 리사이즈 비율)
    """
    width, height = image.size
    
    if width > max_width or height > max_height:
        width_ratio = max_width / width
        height_ratio = max_height / height
        
        resize_ratio = min(width_ratio, height_ratio)
        new_width = closest_multiple_of_8(int(width * resize_ratio))
        new_height = closest_multiple_of_8(int(height * resize_ratio))        
        
        resized_img = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    else:
        resized_img = image
        resize_ratio = 0
        
    return resized_img, resize_ratio

def plot_bboxes(image, box, score, label, label_index):
    """
    객체 탐지 결과를 이미지에 시각화하는 함수
    
    Args:
        image: 대상 이미지
        box: 바운딩 박스 좌표
        score: 탐지 점수
        label: 객체 레이블
        label_index: 레이블 인덱스 (색상 선택용)
    """
    # 미리 정의된 색상 리스트
    colors = [(89, 161, 197), (67, 161, 255), ...]  # 색상 리스트 축약

    label = label + " " + str(score) + "%"
    color = colors[label_index]
    box_label(image, box, label, color)

def string_to_dictionary(text):
    """
    텍스트를 딕셔너리로 변환하는 함수
    
    Args:
        text (str): 변환할 텍스트
        
    Returns:
        dict: 변환된 딕셔너리
    """
    sections = text.split('\n\n')
    result_dict = {}

    for section in sections:
        parts = section.split(':')
        key = parts[0].strip()
        value = parts[1].strip()
        result_dict[key] = value

    return result_dict

def label_select(text):
    """
    'start:'와 ':end' 사이의 텍스트를 추출하는 함수
    
    Args:
        text (str): 대상 텍스트
    Returns:
        str: 추출된 텍스트
    """
    result = re.search(r'start:(.*?):end', text)
    extracted_text = result.group(1)
    return extracted_text

def xywh2xyxy(xywh):
    """
    [x,y,width,height] 형식을 [x1,y1,x2,y2] 형식으로 변환
    
    Args:
        xywh (numpy.ndarray): [x,y,width,height] 형식의 좌표 배열
    Returns:
        numpy.ndarray: [x1,y1,x2,y2] 형식의 좌표 배열
    """
    xyxy = xywh.copy()
    xyxy[:,2] = xywh[:, 0] + xywh[:, 2]  # x2 = x + width
    xyxy[:,3] = xywh[:, 1] + xywh[:, 3]  # y2 = y + height
    return xyxy

def combine_masks(mask_list):
    """
    여러 마스크를 하나로 결합하는 함수
    
    Args:
        mask_list (list): 결합할 마스크 리스트
    Returns:
        numpy.ndarray: 결합된 마스크
    """
    if not mask_list:
        raise ValueError("적어도 하나 이상의 배열이 필요합니다.")
    
    combined_mask = mask_list[0]
    for mask in mask_list[1:]:
        combined_mask = np.logical_or(combined_mask, mask)
    
    return combined_mask[np.newaxis, :, :]

def random_hex_color():
    """
    미리 정의된 색상 리스트에서 랜덤하게 색상을 선택하는 함수
    
    Returns:
        str: 선택된 HEX 색상 코드
    """
    color_list = ["#caff70", "#07ccff", "#fa0087", "#f88379", "#7d37ff"]
    hex_color = random.choice(color_list)
    return hex_color

def dilate_mask(mask, kernel_size=3, iterations=1):
    """
    마스크를 팽창시키는 함수
    
    Args:
        mask (numpy.ndarray): 원본 마스크
        kernel_size (int): 커널 크기
        iterations (int): 반복 횟수
        
    Returns:
        numpy.ndarray: 팽창된 마스크
    """
    mask = mask.astype(np.uint8).squeeze()
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=iterations)
    return dilated_mask