from lama_cleaner import LamaCleaner
from diffusers import StableDiffusionInpaintPipeline
import cv2

# lama-cleaner 및 Stable Diffusion 모델 초기화
lama_model = LamaCleaner()
sd_inpaint_model = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")

# 객체 제거 함수
def remove_object(image_path, mask):
    image = cv2.imread(image_path)
    # lama-cleaner로 객체 제거
    inpainted_image = lama_model.inpaint(image, mask)
    # Stable Diffusion으로 배경 추가 복원
    final_image = sd_inpaint_model(image=inpainted_image, mask=mask).images[0]
    return final_image
