# LangChain 관련 임포트
from langchain.chat_models import ChatOpenAI  # OpenAI의 채팅 모델 사용
from langchain.chains.conversation.memory import ConversationBufferWindowMemory  # 대화 기록 관리
from langchain.agents import initialize_agent, AgentType  # 에이전트 초기화 및 타입 정의

# 커스텀 이미지 편집 도구 임포트
from utils.custom_tools import (
    ImageTransformTool,    # 이미지 변환 도구
    ObjectEraseTool        # 객체 제거 도구
)

def image_editor_agent():
    """
    이미지 편집을 위한 AI 에이전트를 초기화하고 반환하는 함수
    
    Returns:
        agent: 초기화된 LangChain 에이전트 객체
    
    사용되는 도구:
    - ImageTransformTool: 이미지 변환 기능
    - ObjectEraseTool: 객체 제거 기능
    """
    # 사용할 도구들을 리스트로 초기화
    tools = [ImageTransformTool(), ObjectEraseTool()]
    
    # 에이전트 초기화 및 반환
    return initialize_agent(
        # 에이전트 타입 설정 (OpenAI Functions 사용)
        agent=AgentType.OPENAI_FUNCTIONS,
        
        # 사용할 도구들 설정
        tools=tools,
        
        # LLM(Language Learning Model) 설정
        llm=ChatOpenAI(
            temperature=0,          # 창의성 낮게 설정 (결정적인 응답 위해)
            model_name="gpt-4", 
            request_timeout=300
        ),
        
        # 최대 반복 횟수를 1로 제한 (한 번의 시도만 허용)
        max_iterations=1,
        
        # 상세한 로그 출력 활성화
        verbose=True
    ) 