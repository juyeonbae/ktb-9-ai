from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from PIL import Image
import numpy as np
import cv2
from typing import Optional, Dict, Any
import os
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromptOptimizationAgent:
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1):
        """
        프롬프트 최적화 에이전트 초기화
        
        Args:
            model_name: 사용할 OpenAI 모델 이름 (기본값: gpt-4o-mini)
            temperature: 생성 다양성 조절 (0에 가까울수록 결정적)
        """
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY in .env file"
            )
        
        try:
            self.llm = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                api_key=api_key
            )
            logger.info(f"Successfully initialized PromptOptimizationAgent with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize ChatOpenAI: {str(e)}")
            raise

    def _analyze_image_style(self, image: Image.Image) -> Dict[str, Any]:
        """이미지의 스타일 특성을 분석합니다."""
        try:
            # PIL Image를 RGB로 변환
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            img_array = np.array(image)
            
            # 채널 수 확인 및 처리
            if len(img_array.shape) == 2:  # 흑백 이미지
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA 이미지
                img_array = img_array[:, :, :3]
                
            # 이후 처리
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            hist_hue = cv2.calcHist([img_hsv], [0], None, [180], [0,180])
            dominant_hue = np.argmax(hist_hue)
            saturation = np.mean(img_hsv[:, :, 1])
            
            texture_features = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            return {
                "lighting": {
                    "brightness": float(brightness),
                    "contrast": float(contrast)
                },
                "color": {
                    "dominant_hue": float(dominant_hue),
                    "saturation": float(saturation)
                },
                "texture": float(texture_features)
            }
            
        except Exception as e:
            logger.error(f"Style analysis failed: {str(e)}")
            return {}
            
    def _get_image_analysis(self, image: Image.Image, mask: Image.Image) -> Dict[str, Any]:
        """
        이미지와 마스크의 세부 정보를 분석합니다.
        """
        try:
            # 기본 분석
            img_width, img_height = image.size
            img_aspect_ratio = img_width / img_height
            
            # 마스크 영역 분석
            mask_array = np.array(mask)
            mask_indices = np.where(mask_array > 128)
            
            if len(mask_indices[0]) == 0:
                return {}
                
            # 마스크 경계 상자 계산
            y_min, y_max = mask_indices[0].min(), mask_indices[0].max()
            x_min, x_max = mask_indices[1].min(), mask_indices[1].max()
            
            # 마스크 크기와 위치 정보
            mask_width = x_max - x_min
            mask_height = y_max - y_min
            mask_center_x = (x_min + x_max) / 2 / img_width
            mask_center_y = (y_min + y_max) / 2 / img_height
            mask_area_ratio = len(mask_indices[0]) / (img_width * img_height)
            
            # 스타일 분석 추가
            style_analysis = self._analyze_image_style(image)
            
            # 마스크 주변 영역 분석
            kernel = np.ones((5,5), np.uint8)
            dilated_mask = cv2.dilate(mask_array, kernel) - mask_array
            context_image = image.crop((x_min, y_min, x_max, y_max))
            context_style = self._analyze_image_style(context_image)
            
            return {
                "image_size": (img_width, img_height),
                "aspect_ratio": img_aspect_ratio,
                "mask_position": {
                    "center_x": mask_center_x,
                    "center_y": mask_center_y,
                    "relative_width": mask_width / img_width,
                    "relative_height": mask_height / img_height
                },
                "mask_area_ratio": mask_area_ratio,
                "image_style": style_analysis,
                "context_style": context_style
            }
        except Exception as e:
            logger.error(f"Image analysis failed: {str(e)}")
            return {}

    def _generate_position_description(self, position: Dict[str, float]) -> str:
        """
        마스크 위치에 대한 자연어 설명을 생성합니다.
        """
        try:
            x, y = position["center_x"], position["center_y"]
            
            v_pos = "upper" if y < 0.33 else "middle" if y < 0.66 else "lower"
            h_pos = "left" if x < 0.33 else "center" if x < 0.66 else "right"
            
            return f"{v_pos} {h_pos}"
        except Exception as e:
            logger.error(f"Position description generation failed: {str(e)}")
            return "center"

    def _generate_style_description(self, style: Dict[str, Any]) -> str:
        """스타일 특성에 대한 자연어 설명을 생성합니다."""
        try:
            lighting = style.get("lighting", {})
            color = style.get("color", {})
            
            brightness_desc = "bright" if lighting.get("brightness", 0) > 128 else "dim"
            contrast_desc = "high contrast" if lighting.get("contrast", 0) > 50 else "soft"
            saturation_desc = "vibrant" if color.get("saturation", 0) > 128 else "muted"
            texture_desc = "detailed" if style.get("texture", 0) > 100 else "smooth"
            
            return f"{brightness_desc}, {contrast_desc}, {saturation_desc} colors, {texture_desc} texture"
        except Exception as e:
            logger.error(f"Style description generation failed: {str(e)}")
            return ""

    def optimize_prompt(self, 
                       user_prompt: str, 
                       image: Optional[Image.Image] = None,
                       mask: Optional[Image.Image] = None) -> str:
        """
        사용자 프롬프트를 SD Inpainting 모델에 최적화된 형태로 변환합니다.
        
        Args:
            user_prompt: 사용자가 입력한 원본 프롬프트
            image: 원본 이미지 (선택사항)
            mask: 마스크 이미지 (선택사항)
            
        Returns:
            최적화된 프롬프트 문자열
        """
        try:
            context = ""
            if image is not None and mask is not None:
                analysis = self._get_image_analysis(image, mask)
                if analysis:
                    position = self._generate_position_description(analysis["mask_position"])
                    size = "small" if analysis["mask_area_ratio"] < 0.1 else \
                           "medium" if analysis["mask_area_ratio"] < 0.3 else "large"
                    
                    image_style = self._generate_style_description(analysis["image_style"])
                    context_style = self._generate_style_description(analysis["context_style"])
                    
                    context = f"""
                    편집 영역: 이미지의 {position} 부분에 위치한 {size} 크기의 영역
                    이미지 스타일: {image_style}
                    주변 영역 특성: {context_style}
                    """
            
            system_prompt = """
            당신은 Stable Diffusion Inpainting 모델을 위한 프롬프트 최적화 전문가입니다.
            사용자의 편집 요청을 SD 모델이 잘 이해할 수 있는 프롬프트로 변환해주세요.

            1. 조화를 위한 구체적 키워드:
                - 주변 조명과 동일한 lighting, identical lighting condition
                - 일관된 색조 유지 matching color tone, consistent color palette
                - 동일한 텍스처 스타일 same texture quality, matching details

            2. 스타일 통합 키워드:
                - seamlessly integrated, perfectly blended
                - matching the surrounding style
                - coherent with the scene

            3. 품질 관련 키워드 강화:
                - ultra high quality, sharp details
                - professional photography
                - masterful composition

            4. 기술적 세부사항:
                - correct perspective
                - accurate shadows and highlights
                - proper depth integration
            
            최적화 규칙:
            1. 명확한 동작 지시어 포함 (replace with, transform into, change to)
            2. 대상의 구체적 설명 추가 (색상, 재질, 스타일)
            3. 공간적 관계 명시 (위치, 방향, 크기)
            4. 품질 관련 키워드 포함 (photorealistic, high quality, detailed)
            5. 주변 환경과의 조화를 위한 컨텍스트 설명
            6. 조명, 색조, 질감의 일관성 강조
            7. 부정적 프롬프트 제거 및 긍정적 표현으로 변환

            출력 형식:
            - 한 문장으로 된 명확한 프롬프트
            - 불필요한 설명이나 주석 제외
            """
            
            user_message = f"""
            컨텍스트: {context}
            사용자 요청: {user_prompt}

            위 요청을 SD 2 Inpainting 모델용 프롬프트로 최적화해주세요.
            주변 환경의 스타일, 조명, 색조와 자연스럽게 조화되도록 해주세요.
            """
            
            response = self.llm([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
            ])
            
            optimized_prompt = response.content.strip()
            logger.info(f"Successfully optimized prompt: {optimized_prompt}")
            
            return optimized_prompt
            
        except Exception as e:
            logger.error(f"Prompt optimization failed: {str(e)}")
            return user_prompt  # 실패 시 원본 프롬프트 반환