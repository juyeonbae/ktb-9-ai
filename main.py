from fastapi import FastAPI, UploadFile, File

app = FastAPI()

@app.post("/process/")
async def process_image(file: UploadFile = File(...), command: str = "객체 제거"):
    image = load_image(file.file)  # 이미지 로드 함수 구현 필요
    mask = create_mask(image, command)
    
    if command == "객체 제거":
        result = remove_object(image, mask)
    elif command == "이미지 변형":
        result = modify_image_with_pix2pix(image, command)
    
    return {"message": "처리 완료", "result_image": result}  # result는 실제 이미지 경로로 반환
