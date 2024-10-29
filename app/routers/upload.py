from fastapi import APIRouter, UploadFile, File
from app.utils.file_utils import save_file

router = APIRouter()

@router.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    file_path = save_file(file)  # utils에서 파일 저장 함수 호출
    return {"file_path": file_path}
