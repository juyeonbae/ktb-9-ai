# 1. 제거된 코드
- Streamlit UI 관련 모든 코드 (st.image, st.button 등)
- 캔버스 관련 코드 (st_canvas)
- 세션 상태 관리 코드 (st.session_state)
- 로컬 모델 호출 코드

# 2. 새로 추가된 코드
- API 클라이언트 클래스
- HTTP 요청/응답 처리
- 데이터 인코딩/디코딩
- 에러 처리 로직

# 3. 유지된 코드
- 이미지 처리 로직
- 마스크 처리 로직
- 기본적인 유틸리티 함수

----
# 수정 전: 로컬 처리
def process_image(image, mask, prompt):
    result = local_model(image, mask, prompt)
    return result

# 주요 기능
1. Streamlit UI 구현
   - 이미지 업로드
   - 캔버스 드로잉
   - 프롬프트 입력
   - 결과 표시

2. 세션 상태 관리
   - 이미지 상태 (original, current)
   - 마스크 상태
   - 캔버스 상태

3. 로컬 모델 처리
   - 직접 모델 호출
   - 이미지 처리 로직
   - 결과 표시

# 수정 후: API 요청
def process_image(image, mask, prompt):
    encoded_image = encode_base64(image)
    encoded_mask = encode_base64(mask)
    response = api_request(encoded_image, encoded_mask, prompt)
    return response.json()

# 주요 기능
1. API 통신 로직
   - 이미지 전송
   - 마스크 전송
   - 프롬프트 전송
   - 결과 URL 수신

2. 데이터 처리
   - base64 인코딩
   - 응답 처리
   - 에러 처리