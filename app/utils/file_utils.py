import os

def save_file(upload_file):
    file_path = f"./uploads/{upload_file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(upload_file.file.read())
    return file_path
