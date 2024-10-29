def test_edit_image():
    response = client.post("/edit/", json={"command": "객체 제거", "file_path": "test.png"})
    assert response.status_code == 200
