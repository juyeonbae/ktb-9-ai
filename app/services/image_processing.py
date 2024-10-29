from langchain.prompts import PromptTemplate
from app.services.model_loader import load_stable_diffusion_inpaint  # 모델 로드 함수 가져오기
from app.services.inpainting import remove_object, inpaint_background  # inpainting 관련 함수 가져오기
from app.services.mask_generator import generate_mask  # 마스크 생성 함수 가져오기

# LangChain 프롬프트 템플릿 설정
prompt_template = PromptTemplate(
    input_variables=["command"],
    template="이미지에서 {command} 작업을 수행하세요."
)

# 명령어에 따라 적절한 모델을 선택하여 호출하는 함수
def select_model_and_process(command, image):
    # LangChain 프롬프트를 사용해 명령어를 해석합니다.
    formatted_prompt = prompt_template.format(command=command)

    if "객체 제거" in formatted_prompt:
        mask = generate_mask(image, command)  # 객체 제거를 위한 마스크 생성
        result = remove_object(image, mask)  # 객체 제거 및 배경 복원
    elif "배경 복원" in formatted_prompt:
        mask = generate_mask(image, command)  # 배경 복원을 위한 마스크 생성
        result = inpaint_background(image, mask)  # 배경 복원
    elif "이미지 변형" in formatted_prompt:
        sd_inpaint = load_stable_diffusion_inpaint()
        result = sd_inpaint(image=image, prompt=formatted_prompt).images[0]  # Stable Diffusion Inpainting 실행
    else:
        result = image  # 기본적으로 원본 이미지를 반환

    return result

from instruct_pix2pix import InstructPix2Pix  # Instruct Pix2Pix 모델이 제공된다고 가정

# Instruct Pix2Pix로 이미지 변형 함수
def modify_image_with_pix2pix(image, command):
    pix2pix_model = InstructPix2Pix()
    modified_image = pix2pix_model.edit_image(image, command)
    return modified_image
