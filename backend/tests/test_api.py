from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Lunar Localization Platform API is running"}

def test_get_data_status():
    response = client.get("/data/status")
    assert response.status_code == 200
    data = response.json()
    assert "mode" in data
    assert "sector" in data
    assert "real_local_available" in data

def test_set_data_mode_demo():
    response = client.post(
        "/data/mode",
        json={"mode": "DEMO"}
    )
    assert response.status_code == 200
    assert response.json()["current_mode"] == "DEMO"

def test_get_terrain():
    client.post("/data/mode", json={"mode": "DEMO"})
    
    response = client.get("/terrain?max_grid_size=10")
    assert response.status_code == 200
    data = response.json()
    
    assert "grid" in data
    assert data["grid"]["width"] <= 10
    assert data["grid"]["height"] <= 10
    
    assert "elevation" in data
    assert len(data["elevation"]) == data["grid"]["width"] * data["grid"]["height"]
    
    assert "slope" in data
    assert "roughness" in data
    assert "hazards" in data

def test_plan_path():
    client.post("/data/mode", json={"mode": "DEMO"})
    
    response = client.post(
        "/plan_path",
        json={"start": [10, 10], "goal": [20, 20]}
    )
    assert response.status_code == 200
    data = response.json()
    assert "path" in data
    assert len(data["path"]) > 0
    assert data["path"][0] == [10, 10]
    assert data["path"][-1] == [20, 20]
