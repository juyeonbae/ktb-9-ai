from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
import io
from PIL import Image
import numpy as np
from utils.inference import sd_inpaint  # 추가된 import - Stable Diffusion Inpainting

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

       # 디버깅용 마스크 저장
       mask_bytes = base64.b64decode(mask_data)
       with open("debug_mask.png", "wb") as f:
           f.write(mask_bytes)
       print("마스크 이미지가 debug_mask.png로 저장되었습니다.")

       # 1. 이미지 처리
       print("이미지 처리 시작")
       image_content = await image.read()
       img = Image.open(io.BytesIO(image_content))
       print(f"이미지 크기: {img.size}")

       # 2. 마스크 처리
       print("마스크 처리 시작")
       try:
           mask_bytes = base64.b64decode(mask_data)
           mask = Image.open(io.BytesIO(mask_bytes))
           print(f"마스크 크기: {mask.size}")
       except Exception as mask_error:
           print(f"마스크 처리 중 에러: {str(mask_error)}")
           raise HTTPException(status_code=400, detail=f"마스크 처리 실패: {str(mask_error)}")

       # 3. 이미지 크기 확인 및 조정 
       print("이미지 크기 확인")
       if img.size != mask.size:
           print(f"크기 불일치 - 이미지: {img.size}, 마스크: {mask.size}")
           mask = mask.resize(img.size)
           print("마스크 크기 조정됨")

       # 4. SD Inpainting으로 이미지 편집 (수정된 부분)
       print("이미지 편집 시작")
       try:
           # SD Inpainting 모델로 이미지 편집
           edited_image = sd_inpaint(
               image=img,
               mask=mask, 
               inpaint_prompt=prompt
           )
           
           if edited_image is None:
               raise Exception("이미지 편집 실패")
               
           result_image = edited_image
           print("이미지 편집 완료")
                   
       except Exception as edit_error:
           print(f"이미지 편집 중 에러: {str(edit_error)}")
           raise HTTPException(status_code=500, detail=f"이미지 편집 실패: {str(edit_error)}")

       # 5. 결과 반환
       print("결과 이미지 반환 준비")
       output = io.BytesIO()
       result_image.save(output, format='PNG')
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