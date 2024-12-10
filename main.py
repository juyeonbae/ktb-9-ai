from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
import io
from PIL import Image
import numpy as np
from utils.inference import sd_inpaint  

app = FastAPI(title="Image Editor API")

app.add_middleware(
   CORSMiddleware,
   allow_origins=["http://localhost:3000"],
   allow_credentials=True,
   allow_methods=["*"], 
   allow_headers=["*"],
)

@app.post("/api/edit-image")
async def edit_image(
   image: UploadFile = File(...),
   prompt: str = Form(...),
   mask_data: str = Form(...)
):
    try:
        print("=== 요청 시작 ===")
        print(f"받은 프롬프트: {prompt}")
        print(f"파일 이름: {image.filename}")

       # 1. 이미지 처리
        print("이미지 처리 시작")
        image_content = await image.read()
        img = Image.open(io.BytesIO(image_content))
        print(f"이미지 크기: {img.size}")

        # 2. 마스크 처리 - 그레이스케일로 변환 추가
        print("마스크 처리 시작")
        try:
            mask_bytes = base64.b64decode(mask_data)
            mask = Image.open(io.BytesIO(mask_bytes))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"마스크 데이터 처리 실패: {str(e)}")

        # 3. 이미지와 마스크 크기 검증
        if img.size != mask.size:
            raise HTTPException(
                status_code=400, 
                detail=f"마스크 크기({mask.size})가 이미지 크기({img.size})와 일치하지 않습니다"
            )

        # 마스크 이진화 처리 추가
        mask_array = np.array(mask)
        mask_binary = Image.fromarray((mask_array > 128).astype(np.uint8) * 255)  # 수정: 마스크 이진화

       # 4. SD Inpainting으로 이미지 편집
        print("이미지 편집 시작")
        edited_image = sd_inpaint(
            image=img,
            mask=mask_binary,  # 수정: 이진화된 마스크 사용
            inpaint_prompt=prompt
           )
           
        if edited_image is None:
            raise HTTPException(status_code=500, detail="이미지 편집 실패")

        # 5. 결과 반환
        print("결과 이미지 반환 준비")
        output = io.BytesIO()
        edited_image.save(output, format='PNG')
        output.seek(0)
       
        print("=== 처리 완료 ===")
        return StreamingResponse(output, media_type="image/png")
       
    except Exception as e:
        print(f"=== 에러 발생 ===")
        print(f"에러 타입: {type(e)}")
        print(f"에러 메시지: {str(e)}")
        import traceback
        print(f"상세 에러: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Server is running"}

if __name__ == "__main__":
   import uvicorn
   uvicorn.run(app, host="0.0.0.0", port=8000)