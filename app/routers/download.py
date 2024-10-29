from fastapi import APIRouter, FileResponse

router = APIRouter()

@router.get("/download/")
async def download_image(file_path: str):
    return FileResponse(file_path, media_type="image/png")
