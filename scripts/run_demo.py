import cv2
import json
from pathlib import Path

from lunar_platform.core.metadata import ImageMetadata, RegistrationQuality, RegistrationResult
from lunar_platform.scaling.resampler import resample_image
from lunar_platform.registration.sift import SiftMatcher
from lunar_platform.registration.ransac import estimate_transform
from lunar_platform.registration.geometry import compute_reprojection_rmse, calculate_spatial_coverage
from lunar_platform.registration.footprint import estimate_footprint

def run_registration_demo():
    print("--- LUNAR LOCALIZATION MVP DEMO ---")
    
    # 1. Load configuration (hardcoded for demo)
    working_gsd_m = 1.0
    
    # 2. Load demo data
    ref_path = "data/demo/reference/demo_ref.tif"
    input_path = "data/demo/input/demo_input.png"
    meta_path = "data/demo/input/demo_input_metadata.json"
    
    if not Path(ref_path).exists() or not Path(input_path).exists():
        print("Demo data not found. Please run scripts/create_demo.py first.")
        return
        
    print(f"Loading reference map: {ref_path}")
    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    
    print(f"Loading input image: {input_path}")
    input_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    
    with open(meta_path, "r") as f:
        meta_dict = json.load(f)
    
    metadata = ImageMetadata(
        native_gsd_m=meta_dict["gsd_m_per_px"],
        **meta_dict
    )
    
    # 3. GSD Harmonization
    print(f"Harmonizing GSD... Native: {metadata.native_gsd_m}m/px -> Working: {working_gsd_m}m/px")
    input_resampled, scale_factor = resample_image(input_img, metadata.native_gsd_m, working_gsd_m)
    print(f"Scale factor applied: {scale_factor:.2f}")
    
    # 4. Feature Matching
    print("Running SIFT Feature Matching...")
    matcher = SiftMatcher()
    pts_input, pts_ref, scores, matcher_name = matcher.match(input_resampled, ref_img)
    print(f"Found {len(pts_input)} raw matches.")
    
    # 5. Robust Geometry Estimation
    print("Estimating Geometry (Similarity Transform + RANSAC)...")
    transform, inliers = estimate_transform(pts_input, pts_ref, model_type="similarity")
    num_inliers = inliers.sum()
    inlier_ratio = num_inliers / len(pts_input) if len(pts_input) > 0 else 0
    print(f"Inliers: {num_inliers} ({inlier_ratio*100:.1f}%)")
    
    # Filter points to inliers for RMSE
    inlier_pts_input = pts_input[inliers]
    inlier_pts_ref = pts_ref[inliers]
    
    rmse_px = compute_reprojection_rmse(inlier_pts_input, inlier_pts_ref, transform)
    coverage = calculate_spatial_coverage(inlier_pts_input, input_resampled.shape[1], input_resampled.shape[0])
    
    print(f"Reprojection RMSE: {rmse_px:.2f} px")
    print(f"Spatial Coverage: {coverage*100:.1f}%")
    
    # 6. Footprint Estimation
    footprint = estimate_footprint(input_resampled.shape[1], input_resampled.shape[0], transform, working_gsd_m)
    print(f"Estimated Footprint in Local Sector Coordinates (meters):")
    print(f"  Top-Left: ({footprint.top_left.x:.1f}, {footprint.top_left.y:.1f})")
    print(f"  Bottom-Right: ({footprint.bottom_right.x:.1f}, {footprint.bottom_right.y:.1f})")
    
    # 7. Visualization
    # Draw matches
    vis_img = cv2.drawMatches(
        input_resampled, matcher.sift.detect(input_resampled), 
        ref_img, matcher.sift.detect(ref_img), 
        # Fake DMatch objects for visualization
        [cv2.DMatch(i, i, 0) for i in range(len(inlier_pts_input))], 
        None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite("outputs/registration/demo_matches.png", vis_img)
    print("Saved match visualization to outputs/registration/demo_matches.png")

if __name__ == "__main__":
    run_registration_demo()
