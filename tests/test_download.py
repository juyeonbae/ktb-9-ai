def test_download_image():
    response = client.get("/download/?file_path=test.png")
    assert response.status_code == 200
