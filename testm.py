import requests

def test_valid_input():
    response = requests.post("http://localhost:5000/predict", json={
        "features": [5.1, 3.5, 1.4, 0.2]
    })
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_invalid_input():
    response = requests.post("http://localhost:5000/predict", json={})
    assert response.status_code == 400
