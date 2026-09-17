import numpy as np

def generate_cost_map(slope: np.ndarray, roughness: np.ndarray, hazard: np.ndarray, config: dict) -> np.ndarray:
    """
    Generates a normalized cost map [0, 1] for A* pathfinding.
    Cost is a weighted sum of normalized inputs.
    Any cell exceeding max_slope is given infinite cost (impassable).
    """
    weights = config.get("weights", {"slope": 0.45, "roughness": 0.25, "crater": 0.30})
    max_slope = config.get("max_slope_deg", 20.0)
    
    w_s = weights.get("slope", 0.33)
    w_r = weights.get("roughness", 0.33)
    w_c = weights.get("crater", 0.33)
    
    # Normalize inputs for fusion
    norm_slope = np.clip(slope / max_slope, 0.0, 1.0)
    
    # Assuming roughness of 0.5m is very high
    norm_roughness = np.clip(roughness / 0.5, 0.0, 1.0)
    
    # Hazard is already [0, 1]
    
    cost_map = (w_s * norm_slope) + (w_r * norm_roughness) + (w_c * hazard)
    
    # Apply hard constraints
    cost_map[slope > max_slope] = np.inf
    
    # Base cost to prefer shorter paths when terrain is flat
    base_cost = 0.05
    cost_map = cost_map + base_cost
    
    return cost_map
