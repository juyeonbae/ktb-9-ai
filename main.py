from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import io
import traceback
from PIL import Image
import numpy as np
import logging

from core.image_processing import ImageProcessor, ProcessingConfig
from core.inference import ImageEditPipeline


REMOVE_ACTION = "remove"
EDIT_ACTION = "edit"

app = FastAPI(title="Image Editor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://client.zero-dang.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)

# 이미지 프로세서 인스턴스
image_processor = ImageProcessor(ProcessingConfig())
pipeline = ImageEditPipeline()


@app.post("/api/edit-image")
async def edit_image(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    type: str = Form(...),
    prompt: Optional[str] = Form(None)
):

    try:
        logger.info("Received files:")
        logger.info(f"Image: {image.filename}, Content-Type: {image.content_type}")
        logger.info(f"Mask: {mask.filename}, Content-Type: {mask.content_type}")
        logger.info(f"Action type: {type}")  # type 값 확인용 로그

        # 1. 이미지 로드
        image_content = await image.read()
        
        logger.debug(f"Image content length: {len(image_content)}")
                
        try:
            img = Image.open(io.BytesIO(image_content))
            if img.mode == 'RGBA':
                logger.info("Converting image from RGBA to RGB mode")
                img = img.convert('RGB')
        except Exception as e:
            logger.error(f"Image open error: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid image format")

        # 2. 마스크 로드
        mask_content = await mask.read()
        try:
            mask_img = Image.open(io.BytesIO(mask_content))
            if mask_img.mode != 'L':
                mask_img = mask_img.convert('L')
        except Exception as e:
            logger.error(f"Mask loading error: {str(e)}")
            raise ValueError(f"마스크 로드 실패: {str(e)}")

        # 3. 작업 타입별 처리
        if type == REMOVE_ACTION:
            edited = pipeline.remove_object(
                image=img,
                mask=mask_img
            )
        elif type == EDIT_ACTION:
            if not prompt:
                raise ValueError("Edit 작업에는 프롬프트가 필요합니다")
            edited = pipeline.edit_object(
                image=img,
                mask=mask_img,
                prompt=prompt
            )
        else:
            raise ValueError(f"지원하지 않는 작업 타입: {type}")

        # 4. 결과 반환
        output = io.BytesIO()
        edited.save(output, format='PNG', quality=100, optimize=False)
        output.seek(0)

        return StreamingResponse(
            output,
            media_type="image/png",
            headers={
                "Content-Disposition": "attachment; filename=edited_image.png",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )

    except ValueError as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"에러 발생: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Server is running"}


if __name__ == "__main__":
   import uvicorn
   uvicorn.run(app, host="0.0.0.0", port=5001)