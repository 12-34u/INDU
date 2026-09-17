import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
import yaml
import json

def generate_demo_data(base_dir="data/demo"):
    """Generates synthetic demo data for the 1km x 1km lunar sector."""
    
    # Ensure directories exist
    os.makedirs(f"{base_dir}/reference", exist_ok=True)
    os.makedirs(f"{base_dir}/input", exist_ok=True)
    os.makedirs(f"{base_dir}/dem", exist_ok=True)
    os.makedirs(f"{base_dir}/craters", exist_ok=True)

    # Sector configuration (1km x 1km at 1m/px GSD)
    width, height = 1000, 1000
    gsd = 1.0

    # 1. Generate synthetic DEM (Perlin-like noise using sum of sines for simplicity)
    x = np.linspace(0, 10, width)
    y = np.linspace(0, 10, height)
    xv, yv = np.meshgrid(x, y)
    
    # Base terrain variation (elevation in meters, e.g., -50 to +50)
    dem = 20 * np.sin(xv) * np.cos(yv) + 10 * np.sin(2.5 * xv + 1) * np.cos(1.5 * yv)
    
    # Add some "craters"
    craters_info = []
    for _ in range(5):
        cx = np.random.randint(100, 900)
        cy = np.random.randint(100, 900)
        radius = np.random.randint(30, 80)
        depth = radius * 0.2
        
        # Create crater depression
        y_grid, x_grid = np.ogrid[-cy:height-cy, -cx:width-cx]
        dist_sq = x_grid**2 + y_grid**2
        mask = dist_sq < radius**2
        
        # Parabolic crater shape
        dem[mask] -= depth * (1 - dist_sq[mask] / radius**2)
        
        craters_info.append({
            "id": f"crater_{len(craters_info)}",
            "x": cx * gsd,
            "y": cy * gsd,
            "diameter_m": radius * 2 * gsd
        })

    # Save DEM
    transform = from_origin(0, height, gsd, gsd)
    dem_path = f"{base_dir}/dem/demo_dem.tif"
    with rasterio.open(
        dem_path, 'w', driver='GTiff',
        height=height, width=width,
        count=1, dtype=dem.dtype,
        crs='+proj=ortho +lat_0=0 +lon_0=0',
        transform=transform
    ) as dst:
        dst.write(dem, 1)

    # 2. Generate synthetic reference image based on DEM hillshade
    # Simple gradient for illumination
    dy, dx = np.gradient(dem, gsd, gsd)
    slope = np.pi/2. - np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dx, dy)
    
    sun_azimuth = np.radians(315)
    sun_elevation = np.radians(45)
    
    shaded = np.sin(sun_elevation) * np.sin(slope) + \
             np.cos(sun_elevation) * np.cos(slope) * np.cos(sun_azimuth - aspect)
             
    # Normalize to 0-255 uint8
    shaded = np.clip(shaded, 0, 1)
    ref_image = (shaded * 255).astype(np.uint8)

    # Save Reference Image
    ref_path = f"{base_dir}/reference/demo_ref.tif"
    with rasterio.open(
        ref_path, 'w', driver='GTiff',
        height=height, width=width,
        count=1, dtype=ref_image.dtype,
        crs='+proj=ortho +lat_0=0 +lon_0=0',
        transform=transform
    ) as dst:
        dst.write(ref_image, 1)

    # 3. Generate input image (a cropped, scaled, and rotated version of the reference)
    # E.g., representing a Chandrayaan image at 0.5m/px
    input_gsd = 0.5
    scale_factor = gsd / input_gsd  # 2.0
    
    # We'll just take a 400x400 crop of the 1m/px image, which corresponds to 400m x 400m
    # Then resize it to 800x800 to simulate 0.5m/px
    import cv2
    start_y, start_x = 300, 300
    crop_size = 400
    crop = ref_image[start_y:start_y+crop_size, start_x:start_x+crop_size]
    
    # Simulate different illumination/noise
    input_image = cv2.convertScaleAbs(crop, alpha=1.1, beta=10)
    # Resize to simulate finer GSD
    input_image = cv2.resize(input_image, (crop_size * 2, crop_size * 2), interpolation=cv2.INTER_LINEAR)
    
    # Save Input Image (we save it as a simple PNG or TIF without georeferencing to simulate needing registration)
    input_path = f"{base_dir}/input/demo_input.png"
    cv2.imwrite(input_path, input_image)
    
    # Save input metadata
    metadata = {
        "image_id": "demo_input_001",
        "sensor": "synthetic_ohrc",
        "instrument": "synthetic",
        "width": crop_size * 2,
        "height": crop_size * 2,
        "gsd_m_per_px": input_gsd,
        "source_path": input_path
    }
    with open(f"{base_dir}/input/demo_input_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save crater catalogue
    with open(f"{base_dir}/craters/demo_craters.json", "w") as f:
        json.dump({"craters": craters_info}, f, indent=2)
        
    print("Demo data generation complete.")

if __name__ == "__main__":
    generate_demo_data()
