from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_upload_image():
    response = client.post("/upload/", files={"file": ("test.png", b"test data")})
    assert response.status_code == 200
