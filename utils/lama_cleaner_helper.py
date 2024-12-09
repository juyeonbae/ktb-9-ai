import io
import os
import sys
from typing import List, Optional
from urllib.parse import urlparse
import cv2
from PIL import Image, ImageOps, PngImagePlugin
import numpy as np
import torch
from loguru import logger
from torch.hub import download_url_to_file, get_dir
import hashlib

def md5sum(filename):
    """
    파일의 MD5 해시값을 계산하는 함수
    
    Args:
        filename (str): 해시값을 계산할 파일 경로
    
    Returns:
        str: 파일의 MD5 해시값
    """
    md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(128 * md5.block_size), b""):
            md5.update(chunk)
    return md5.hexdigest()

def get_cache_path_by_url(url):
    """
    URL에 해당하는 로컬 캐시 경로를 반환하는 함수
    
    Args:
        url (str): 모델 다운로드 URL
    
    Returns:
        str: 로컬 캐시 파일 경로
    """
    parts = urlparse(url)
    hub_dir = get_dir()
    model_dir = os.path.join(hub_dir, "checkpoints")
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    return cached_file

def download_model(url, model_md5: str = None):
    """
    모델 파일을 다운로드하고 MD5 검증을 수행하는 함수
    
    Args:
        url (str): 모델 다운로드 URL
        model_md5 (str, optional): 기대되는 MD5 해시값
    
    Returns:
        str: 다운로드된 모델의 로컬 경로
        
    Raises:
        SystemExit: MD5 검증 실패 시
    """
    cached_file = get_cache_path_by_url(url)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        download_url_to_file(url, cached_file, hash_prefix, progress=True)
        
        # MD5 검증 수행
        if model_md5:
            _md5 = md5sum(cached_file)
            if model_md5 == _md5:
                logger.info(f"Download model success, md5: {_md5}")
            else:
                try:
                    os.remove(cached_file)
                    logger.error(
                        f"Model md5: {_md5}, expected md5: {model_md5}, wrong model deleted. Please restart lama-cleaner."
                        f"If you still have errors, please try download model manually first https://lama-cleaner-docs.vercel.app/install/download_model_manually.\n"
                    )
                except:
                    logger.error(
                        f"Model md5: {_md5}, expected md5: {model_md5}, please delete {cached_file} and restart lama-cleaner."
                    )
                exit(-1)

    return cached_file

def handle_error(model_path, model_md5, e):
    """
    모델 로딩 에러 처리 함수
    
    Args:
        model_path (str): 모델 파일 경로
        model_md5 (str): 기대되는 MD5 해시값
        e (Exception): 발생한 예외
        
    Raises:
        SystemExit: 항상 프로그램 종료
    """
    _md5 = md5sum(model_path)
    if _md5 != model_md5:
        try:
            os.remove(model_path)
            logger.error(
                f"Model md5: {_md5}, expected md5: {model_md5}, wrong model deleted. Please restart lama-cleaner."
                f"If you still have errors, please try download model manually first https://lama-cleaner-docs.vercel.app/install/download_model_manually.\n"
            )
        except:
            logger.error(
                f"Model md5: {_md5}, expected md5: {model_md5}, please delete {model_path} and restart lama-cleaner."
            )
    else:
        logger.error(
            f"Failed to load model {model_path},"
            f"please submit an issue at https://github.com/Sanster/lama-cleaner/issues and include a screenshot of the error:\n{e}"
        )
    exit(-1)

def load_jit_model(url_or_path, device, model_md5: str):
    """
    TorchScript 모델을 로드하는 함수
    
    Args:
        url_or_path (str): 모델 URL 또는 로컬 경로
        device (str): 사용할 디바이스 ('cpu' 또는 'cuda')
        model_md5 (str): 기대되는 MD5 해시값
    
    Returns:
        torch.jit.ScriptModule: 로드된 모델
        
    Raises:
        SystemExit: 모델 로딩 실패 시
    """
    if os.path.exists(url_or_path):
        model_path = url_or_path
    else:
        model_path = download_model(url_or_path, model_md5)

    logger.info(f"Loading model from: {model_path}")
    try:
        model = torch.jit.load(model_path, map_location="cpu").to(device)
    except Exception as e:
        handle_error(model_path, model_md5, e)
    model.eval()
    return model

def norm_img(np_img):
    """
    이미지 배열을 정규화하는 함수
    
    Args:
        np_img (numpy.ndarray): 입력 이미지 배열
    
    Returns:
        numpy.ndarray: 정규화된 이미지 배열 (채널 우선, 0-1 범위)
    """
    if len(np_img.shape) == 2:
        np_img = np_img[:, :, np.newaxis]
    np_img = np.transpose(np_img, (2, 0, 1))  # CHW 형식으로 변환
    np_img = np_img.astype("float32") / 255   # 0-1 범위로 정규화
    return np_img