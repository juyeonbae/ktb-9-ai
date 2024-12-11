from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
import io, traceback
from PIL import Image
import numpy as np
from utils.inference import sd_inpaint 
from utils.image_quality import ImageQualityManager 

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
        
        quality_manager = ImageQualityManager()  # 인스턴스 생성

        # 1. 이미지 처리
        print("이미지 처리 시작")
        image_content = await image.read()
        img = Image.open(io.BytesIO(image_content))
        
        original_size = img.size  # 원본 크기 저장
        print(f"원본 이미지 크기: {original_size}")
        
        # RGB 모드 확인 
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 초기 품질 처리
        img = quality_manager.process_image(img)
        print(f"처리된 이미지 크기: {img.size}")

        # 2. 마스크 처리
        print("마스크 처리 시작")
        try:
            mask_bytes = base64.b64decode(mask_data)
            mask = Image.open(io.BytesIO(mask_bytes))
            mask = quality_manager.process_image(mask, target_size=img.size)

            
            # 마스크를 이미지 크기에 맞게 조정
            if mask.size != img.size:
                mask = mask.resize(img.size, Image.Resampling.NEAREST)
                
            # 마스크를 그레이스케일로 변환
            if mask.mode != 'L':
                mask = mask.convert('L')
            
            # 마스크 이진화 처리 개선
            mask_array = np.array(mask)
            threshold = 128
            mask_binary = Image.fromarray(
                ((mask_array > threshold).astype(np.uint8) * 255),
                mode='L'
            )
            
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"마스크 데이터 처리 실패: {str(e)}"
            )

        # 3. SD Inpainting으로 이미지 편집
        print("이미지 편집 시작")
        edited_image = sd_inpaint(
            image=img,
            mask=mask_binary,
            inpaint_prompt=prompt
        )
        
        # 최종 품질 향상
        edited_image = quality_manager.process_image(edited_image)
        
        if edited_image is None:
            raise HTTPException(status_code=500, detail="이미지 편집 실패")

        # 4. 결과 이미지 저장 전 원본 크기로 복원
        print(f"편집된 이미지 크기: {edited_image.size}")
        if edited_image.size != original_size:
            print(f"이미지 크기 복원: {edited_image.size} -> {original_size}")
            edited_image = edited_image.resize(original_size, Image.Resampling.LANCZOS)


        # 4. 결과 이미지 저장 및 반환
        print("결과 이미지 반환 준비")
        output = io.BytesIO()
        
        # 고품질 PNG 저장 설정
        edited_image.save(
            output, 
            format='PNG',
            quality=100,
            optimize=False,
            subsampling=0
        )
        output.seek(0)
        
        print("=== 처리 완료 ===")
        
        # 캐시 제어 헤더 추가
        headers = {
            "Content-Disposition": "attachment; filename=edited_image.png",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
        
        return StreamingResponse(
            output, 
            media_type="image/png",
            headers=headers
        )
       
    except Exception as e:
        print(f"=== 에러 발생 ===")
        print(f"에러 타입: {type(e)}")
        print(f"에러 메시지: {str(e)}")
        print(f"상세 에러: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Server is running"}

if __name__ == "__main__":
   import uvicorn
   uvicorn.run(app, host="0.0.0.0", port=8000)