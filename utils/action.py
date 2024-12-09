import streamlit as st

def backward_inference_image():
    """
    이전 이미지 상태로 되돌리는 함수
    - 현재 이미지 상태가 0보다 크면 이전 상태로 이동
    - 첫 번째 이미지인 경우 토스트 메시지 표시
    """
    if st.session_state["image_state"] > 0:
        # 이미지 상태 인덱스를 1 감소
        st.session_state["image_state"] -= 1
        # 좌표 정보 초기화
        st.session_state["num_coord"] = 0
    else:
        # 첫 번째 이미지일 때 메시지 표시
        st.toast('This is First Image!')
    
def forward_inference_image():
    """
    다음 이미지 상태로 이동하는 함수
    - 다음 상태가 존재하는 경우 다음 상태로 이동
    - 마지막 이미지인 경우 토스트 메시지 표시
    - 캔버스 상태도 함께 초기화
    """
    if len(st.session_state["inference_image"]) - 1 > st.session_state["image_state"]:
        # 이미지 상태 인덱스를 1 증가
        st.session_state["image_state"] += 1
        # 좌표 정보 초기화
        st.session_state["num_coord"] = 0
        
        # 캔버스 객체가 존재하는 경우 캔버스 초기화
        if "canvas" in st.session_state and st.session_state["canvas"] is not None:
            try:
                if 'raw' in st.session_state["canvas"]:
                    st.session_state["canvas"]['raw']["objects"] = []
            except Exception as e:
                st.error(f"Error handling canvas state: {str(e)}")
    else:
        # 마지막 이미지일 때 메시지 표시
        st.toast('This is Last Image!')
            
def reset_inference_image():
    """
    이미지 상태를 초기 상태로 리셋하는 함수
    - 이미지 리스트를 첫 번째 이미지만 남기고 초기화
    - 이미지 상태와 좌표 정보도 초기화
    """
    # 이미지 리스트를 첫 번째 이미지만 남기고 초기화
    st.session_state["inference_image"] = [st.session_state["inference_image"][0]]
    # 이미지 상태를 0으로 초기화
    st.session_state["image_state"] = 0
    # 좌표 정보 초기화
    st.session_state["num_coord"] = 0
    
def reset_text():
    """
    텍스트 입력을 초기화하는 함수
    - 텍스트 상태를 빈 문자열로 초기화
    """
    st.session_state["text"] = ""
    
def reset_coord():
    """
    좌표 및 관련 상태를 초기화하는 함수
    - 캔버스 상태 삭제
    - 세그멘테이션 관련 상태 초기화
    - 좌표 정보 초기화
    """
    # 캔버스 상태가 존재하면 삭제
    if "canvas" in st.session_state:
        del st.session_state["canvas"]
    
    # 관련 상태 변수들 초기화
    st.session_state["num_coord"] = 0          # 좌표 개수 초기화
    st.session_state["sam_image"] = None       # 세그멘테이션 이미지 초기화
    st.session_state["mask"] = None            # 마스크 초기화
    st.session_state["coord"] = False          # 좌표 존재 여부 초기화
    st.session_state["freedraw"] = False       # 자유 그리기 모드 상태 초기화