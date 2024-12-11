import logging
from typing import Dict, Optional
from googletrans import Translator
import time

logger = logging.getLogger(__name__)

class TranslationManager:
    def __init__(self):
        self.translator = Translator()
        self.retry_count = 3
        self.delay = 1
    
    def translate(self, text: str, src: str = 'ko', dest: str = 'en') -> Optional[str]:
        for attempt in range(self.retry_count):
            try:
                if not text or all(ord(c) < 128 for c in text):
                    return text
                    
                result = self.translator.translate(text, src=src, dest=dest)
                return result.text
                
            except Exception as e:
                logger.warning(f"번역 시도 {attempt + 1} 실패: {str(e)}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.delay)
                continue
        
        logger.error("번역 최종 실패")
        return text

class PromptProcessor:
    def __init__(self):
        self.translation_manager = TranslationManager()
        self.action_map = self._create_action_map()
        self.object_map = self._create_object_map()
        
        self.object_details = {
            "비행기": "a commercial airplane flying in the blue sky, clear detailed view, correct scale and perspective",
            "자동차": "a detailed modern car on the road, matching the scene perspective",
            "사람": "a person standing naturally in the scene, correct proportion",
            # 다른 객체들 추가
        }
        
    def _create_action_map(self) -> Dict[str, str]:
        return {
            # 제거 관련
            "지워": "remove this completely and fill with natural background",
            "지워줘": "remove this completely and fill with natural background",
            "제거": "remove this completely and fill with natural background",
            "삭제": "remove this completely and fill with natural background",
            "없애": "remove this completely and fill with natural background",
            
            # 변경 관련
            "변경": "transform this into",
            "바꿔": "transform this into",
            "바꿔줘": "transform this into",
            "수정해줘": "modify to",
            
            # 추가 관련 
            "추가해줘": "add",
            "넣어줘": "insert",
            
            # 스타일 관련
            "스타일로": "in the style of",
            "느낌으로": "in the style of",
            "분위기로": "in the atmosphere of",
            "같이": "similar to"
        }
    
    def _create_object_map(self) -> Dict[str, str]:
        return {
            "탑": "tower",
            "건물": "building",
            "나무": "tree",
            "사람": "person",
            "하늘": "sky",
            "구름": "the clouds",
            "산": "mountain",
            "자동차": "car",
            "강": "river",
            "바다": "ocean"
        }
    
    def _enhance_prompt(self, prompt: str) -> str:
        # 품질 관련 키워드
        quality_keywords = (
            ", best quality, high resolution, highly detailed, "
            "professional photo retouching, seamless editing, "
            "perfect composition"
        )
        
        # 작업 유형별 키워드
        if any(word in prompt.lower() for word in ["지워", "제거", "삭제", "없애"]):
            context_keywords = (
                ", seamlessly blend with surroundings, "
                "maintain perspective and lighting, "
                "natural background restoration, "
                "perfect composition"
            )
        else:
            context_keywords = (
                ", maintain consistency with surroundings, "
                "preserve lighting and perspective, "
                "natural integration"
            )
        
        return f"{prompt}{context_keywords}{quality_keywords}"
    
    def process(self, prompt: str) -> str:
        try:
            # 1. 소문자 변환 및 기본 전처리
            processed_prompt = prompt.lower().strip()
            
            # 2. 액션 매핑
            for ko_action, en_action in self.action_map.items():
                if ko_action in processed_prompt:
                    processed_prompt = processed_prompt.replace(ko_action, en_action)
                    break
            
            # 3. 객체 매핑
            for ko_obj, en_obj in self.object_map.items():
                if ko_obj in prompt:
                    processed_prompt = processed_prompt.replace(ko_obj, en_obj)
            
            # 4. 프롬프트 강화
            enhanced_prompt = self._enhance_prompt(processed_prompt)
            
            # 5. 번역
            final_prompt = self.translation_manager.translate(enhanced_prompt)
            
            logger.info(f"원본 프롬프트: {prompt}")
            logger.info(f"처리된 프롬프트: {final_prompt}")
            
            return final_prompt
            
        except Exception as e:
            logger.error(f"프롬프트 처리 실패: {str(e)}")
            return prompt
        
    def process_generation_prompt(self, prompt: str) -> str:
        # 객체 종류 파악
        for obj_key, obj_detail in self.object_details.items():
            if obj_key in prompt:
                return (
                    f"{obj_detail}, photorealistic, perfect integration with surroundings, "
                    "professional photography, masterpiece quality"
                )
        return prompt    
