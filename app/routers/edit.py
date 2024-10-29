from fastapi import APIRouter
from app.services.inpainting import remove_object, inpaint_background

router = APIRouter()

@router.post("/edit/")
async def edit_image(command: str, file_path: str):
    if command == "객체 제거":
        result = remove_object(file_path)
    elif command == "배경 복원":
        result = inpaint_background(file_path)
    return {"edited_image": result}
