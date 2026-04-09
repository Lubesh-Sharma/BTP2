import os
import scipy.io as sio
import numpy as np
from pyFM.mesh import TriMesh

def get_quality_label(error):
    """Standard DV-Matcher / Princeton scale for sqrt(Area) normalization."""
    if error < 0.02: return "Excellent (SOTA)"
    elif error < 0.05: return "Very good"
    elif error < 0.1: return "Good"
    elif error < 0.2: return "Moderate"
    else: return "Poor"

# ====== PATH CONFIGURATION ======
off_dir = './input/SHREC/off/'
mat_dir = './input/SHREC/mat/'
map_dir = './input/SHREC/map/'
p2p_dir = './p2p_results_st_te_SHREC/'

# 1. Get all .off files and sort them
all_off_files = sorted([f for f in os.listdir(off_dir) if f.endswith('.off')])
mesh_names = [os.path.splitext(f)[0] for f in all_off_files]

results_log = []

print(f"{'Source':<8} | {'Target':<8} | {'Geo Error':<10}")
print("-" * 45)

# ====== 44x44 NESTED LOOP ======
for src_name in mesh_names:
    for tgt_name in mesh_names:
        if src_name == tgt_name:
            continue
            
        # 1. Define filenames based on your specified logic:
        # p2p_{TARGET}_to_{SOURCE}.txt contains map: SOURCE -> TARGET
        map_filename = f"{src_name}_{tgt_name}.map"
        p2p_filename = f"p2p_{tgt_name}_to_{src_name}.txt"
        
        map_path = os.path.join(map_dir, map_filename)
        p2p_path = os.path.join(p2p_dir, p2p_filename)

        # Skip if Ground Truth or Prediction doesn't exist for this pair
        if not os.path.exists(map_path) or not os.path.exists(p2p_path):
            continue

        try:
            # 2. Load Target Mesh & Calculate Area Normalization
            # Error is measured on the surface of the Target mesh
            mesh_target = TriMesh(os.path.join(off_dir, f"{tgt_name}.off"), area_normalize=False)
            norm_factor = np.sqrt(mesh_target.area)

            # 3. Load Distance Matrix of Target
            dist_matrix = sio.loadmat(os.path.join(mat_dir, f"{tgt_name}.mat"))['dist']
            num_tgt_v = dist_matrix.shape[0]

            # 4. Load Ground Truth Map (Source -> Target)
            gt_map = np.loadtxt(map_path, dtype=np.int32)
            if gt_map.min() == 1: gt_map -= 1 # Convert 1-based to 0-based
            
            # 5. Load Predicted P2P (Source -> Target)
            txt_data = np.genfromtxt(p2p_path, dtype=int)
            p2p_preds = txt_data[:, 1]
            if p2p_preds.min() == 1: p2p_preds -= 1

            # --- RESOLVING OUT-OF-BOUNDS ERROR ---
            # Align arrays: find how many vertices we can compare
            num_compare = min(len(gt_map), len(p2p_preds))
            
            gt_indices = gt_map[:num_compare]
            pred_indices = p2p_preds[:num_compare]

            # Create a mask to ensure both GT and Pred indices are within the Target distance matrix
            # If target has 5199 vertices, indices must be 0 to 5198
            valid_mask = (gt_indices < num_tgt_v) & (pred_indices < num_tgt_v) & \
                         (gt_indices >= 0) & (pred_indices >= 0)
            
            if not np.any(valid_mask):
                print(f"{src_name:<8} | {tgt_name:<8} | ERROR: No valid indices found")
                continue

            # 6. Geodesic Calculation
            # Index into dist_matrix[GT_on_target, Prediction_on_target]
            raw_errors = dist_matrix[gt_indices[valid_mask], pred_indices[valid_mask]]
            final_error = raw_errors.mean() / norm_factor
            
            results_log.append({
                'source': src_name,
                'target': tgt_name,
                'error': final_error,
                'mat': f"{tgt_name}.mat"
            })
            
            print(f"{src_name:<8} | {tgt_name:<8} | {final_error:.6f}")

        except Exception as e:
            print(f"{src_name:<8} | {tgt_name:<8} | ERROR: {str(e)}")

# ====== FINAL AGGREGATION ======
if results_log:
    all_errs = [r['error'] for r in results_log]
    avg_total = np.mean(all_errs)
    sorted_results = sorted(results_log, key=lambda x: x['error'])

    print("\n" + "="*55)
    print(f"SHREC '19 SUMMARY ({len(results_log)} valid GT pairs)")
    print(f"OVERALL MEAN GEODESIC ERROR: {avg_total:.6f}")
    print(f"Overall Quality Category:   {get_quality_label(avg_total)}")
    print("="*55)

    print("\nTOP 5 BEST MATCHES (Min Error):")
    for i, res in enumerate(sorted_results[:5]):
        print(f"#{i+1} | {res['error']:.6f} | {res['source']} -> {res['target']}")
else:
    print("\n[!] No matching .map and .txt pairs found.")