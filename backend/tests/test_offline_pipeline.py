import pytest
from fastapi.testclient import TestClient
from pathlib import Path

from app.main import app, data_manager
from lunar_platform.core.data_manager import DataMode

client = TestClient(app)

def test_offline_demo_pipeline_no_raw_access(monkeypatch):
    """
    Tests that the DEMO mode pipeline works end-to-end without accessing data/raw.
    We patch Path.exists to throw an exception if data/raw is checked.
    """
    original_exists = Path.exists
    
    def mocked_exists(self):
        if "data/raw" in str(self.absolute()):
            raise PermissionError("Test attempted to access data/raw during runtime!")
        return original_exists(self)
        
    monkeypatch.setattr(Path, "exists", mocked_exists)
    
    # 1. Force DEMO mode
    response = client.post("/data/mode", json={"mode": "DEMO"})
    assert response.status_code == 200
    
    # 2. Verify status
    response = client.get("/data/status")
    assert response.status_code == 200
    assert response.json()["mode"] == "DEMO"
    
    # 3. Run path planning using DEMO data
    response = client.post(
        "/plan_path",
        json={"start": [10, 10], "goal": [20, 20]}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "path" in data
    assert len(data["path"]) > 0
    assert data["path"][0] == [10, 10]
    
def test_fallback_to_demo_on_startup():
    """
    Tests that starting the DataManager without a valid sector_001
    gracefully defaults to DEMO.
    """
    from lunar_platform.core.data_manager import DataManager
    
    # Create a fresh manager, sector_001 might exist but lacks full data,
    # so it should default to DEMO
    manager = DataManager()
    
    assert manager.active_mode == DataMode.DEMO
    
def test_explicit_real_local_fails_if_missing():
    """
    Tests that explicitly requesting REAL_LOCAL when files are missing
    raises a clear error instead of silently falling back.
    """
    from lunar_platform.core.data_manager import DataManager, DataMode
    
    manager = DataManager()
    
    # Force the check to fail by using a non-existent sector
    with pytest.raises(FileNotFoundError):
        manager.set_mode(DataMode.REAL_LOCAL, sector_id="does_not_exist")
