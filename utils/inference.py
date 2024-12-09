import sys
sys.path.append('Grounded-Segment-Anything')
sys.path.append('Grounded-Segment-Anything/EfficientSAM')

import streamlit as st
import numpy as np
import torch
from PIL import Image, ImageOps

from diffusers import StableDiffusionInpaintPipeline

from utils.lama_cleaner_helper import norm_img
from utils.model_setup import (
    get_triton_client, 
    get_sd_inpaint, 
    get_lama_cleaner, 
    get_instruct_pix2pix
    )
import tritonclient.http

from utils.device_utils import get_device

# 디버깅 로깅 함수 추가
def log_debug(message):
    """디버깅 메시지를 출력하는 헬퍼 함수"""
    st.write(f"Debug: {message}")

def instruct_pix2pix(image, prompt):
    try:
        log_debug("Starting pix2pix transformation...")
        # GPU 메모리 정리 추가
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        pipe = get_instruct_pix2pix()
        image = ImageOps.exif_transpose(image)
        images = pipe(
            prompt, 
            image=image, 
            num_inference_steps=20, 
            image_guidance_scale=1.5, 
            guidance_scale=7
        ).images
        
        log_debug("Pix2pix transformation completed")
        return images
    except Exception as e:
        st.error(f"Error in instruct_pix2pix: {str(e)}")
        import traceback
        st.error(f"Full error: {traceback.format_exc()}")
        return None

def sam(image, neg_coords, pos_coords, labels):
    """SAM 모델을 사용한 세그멘테이션 함수"""
    try:
        log_debug("Initializing SAM...")
        triton_client = get_triton_client()
        image = np.array(image).copy()
        
        # 입력 설정
        try:
            pos_coords_in = tritonclient.http.InferInput("pos_coords", pos_coords.shape, "INT64")
            neg_coords_in = tritonclient.http.InferInput("neg_coords", neg_coords.shape, "INT64")
            labels_in = tritonclient.http.InferInput("labels", labels.shape, "INT64")
            image_in = tritonclient.http.InferInput("input_image", image.shape, "UINT8")
            
            pos_coords_in.set_data_from_numpy(pos_coords.astype(np.int64))
            neg_coords_in.set_data_from_numpy(neg_coords.astype(np.int64))
            labels_in.set_data_from_numpy(labels.astype(np.int64))
            image_in.set_data_from_numpy(image.astype(np.uint8))
            
            inputs = [pos_coords_in, neg_coords_in, labels_in, image_in]
        except Exception as e:
            st.error(f"Error preparing SAM inputs: {str(e)}")
            return None, None, None
            
        # 출력 설정
        outputs = [
            tritonclient.http.InferRequestedOutput(name="mask", binary_data=False),
            tritonclient.http.InferRequestedOutput(name="segmented_image", binary_data=False)
        ]
        
        log_debug("Processing SAM inference...")
        response = triton_client.infer(
            model_name="sam", 
            model_version="1", 
            inputs=inputs, 
            outputs=outputs
        )
        
        mask = response.as_numpy("mask")
        segmented_image = response.as_numpy("segmented_image")
        
        log_debug("SAM processing completed")
        return image, mask, Image.fromarray(segmented_image)
        
    except Exception as e:
        st.error(f"Error in SAM: {str(e)}")
        import traceback
        st.error(f"Full error: {traceback.format_exc()}")
        return None, None, None

def sd_inpaint(image: Image.Image, mask: Image.Image, inpaint_prompt):
    """Stable Diffusion을 사용한 인페인팅 함수"""
    try:
        log_debug("Starting inpainting process...")
        
        def resize_images(original_image, mask_image, target_size=(512, 512)):
            """이미지와 마스크를 목표 크기로 리사이징"""
            original_size = original_image.size
            resized_image = original_image.resize(target_size, Image.Resampling.LANCZOS)
            resized_mask = mask_image.resize(target_size, Image.Resampling.NEAREST)
            return resized_image, resized_mask, original_size
            
        device = get_device()
        log_debug(f"Using device: {device}")
        
        # GPU 메모리 정리
        if device == "cuda":
            torch.cuda.empty_cache()
            
        # 모델 로드 및 디바이스 설정
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        
        # 이미지 리사이징
        log_debug("Resizing images...")
        resized_image, resized_mask, original_size = resize_images(image, mask)
        
        # MPS 디바이스 최적화
        if device == "mps":
            pipe.enable_attention_slicing()
            
        log_debug("Performing inpainting...")
        output = pipe(
            prompt=inpaint_prompt,
            image=resized_image,
            mask_image=resized_mask,
            num_inference_steps=50,
            guidance_scale=7.5
        ).images[0]
        
        # 원본 크기로 복원
        log_debug("Restoring original size...")
        inpainted_image = output.resize(original_size, Image.Resampling.LANCZOS)
        
        log_debug("Inpainting completed successfully")
        return inpainted_image
        
    except Exception as e:
        st.error(f"Error in sd_inpaint: {str(e)}")
        import traceback
        st.error(f"Full error: {traceback.format_exc()}")
        return None

def lama_cleaner(image, mask, device):
    """LaMa 모델을 사용한 이미지 클리닝 함수"""
    try:
        log_debug("Starting LaMa cleaner operation...")
        device = get_device()  # 디바이스 자동 감지
        
        # GPU 메모리 정리
        if device == "cuda":
            torch.cuda.empty_cache()
            
        model = get_lama_cleaner()
        
        # 이미지 전처리
        image = norm_img(image)
        mask = norm_img(mask)
        mask = (mask > 0) * 1
        
        # 텐서 변환 및 디바이스 할당
        image = torch.from_numpy(image).unsqueeze(0).to(device)
        mask = torch.from_numpy(mask).unsqueeze(0).to(device)
        
        log_debug("Processing image with LaMa...")
        inpainted_image = model(image, mask)
        
        # 결과 후처리
        cur_res = inpainted_image[0].permute(1, 2, 0).detach().cpu().numpy()
        cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
        
        log_debug("LaMa cleaning completed successfully")
        return Image.fromarray(cur_res)
        
    except Exception as e:
        st.error(f"Error in lama_cleaner: {str(e)}")
        import traceback
        st.error(f"Full error: {traceback.format_exc()}")
        return None

