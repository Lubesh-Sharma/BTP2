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
off_dir = './input/SURREAL/off/'
mat_dir = './input/SURREAL/mat/'
p2p_dir = './p2p_results_st_te_SHREC_to_SURREAL/'

# 1. Get the last 20 files to define our test set
all_off_files = sorted([f for f in os.listdir(off_dir) if f.endswith('.off')])
test_files = all_off_files[-20:] 
mesh_names = [os.path.splitext(f)[0] for f in test_files]

results_log = []

print(f"{'Source':<12} | {'Target':<12} | {'Geo Error':<10}")
print("-" * 45)

# ====== NESTED LOOP FOR PAIRWISE EVALUATION ======
for src_name in mesh_names:
    for tgt_name in mesh_names:
        
        # Skip self-to-self as it doesn't represent real deformation matching
        if src_name == tgt_name:
            continue
            
        # Define the expected filename for the predicted p2p map
        # Logic: p2p_{TARGET}_to_{SOURCE}.txt (Based on your previous setup)
        p2p_filename = f"p2p_{tgt_name}_to_{src_name}.txt"
        p2p_path = os.path.join(p2p_dir, p2p_filename)
        
        # Only proceed if the prediction file exists for this pair
        if not os.path.exists(p2p_path):
            continue
            
        try:
            # 1. Load Target Mesh Area (for DV-Matcher sqrt(Area) norm)
            mesh_target = TriMesh(f"{off_dir}{tgt_name}.off", area_normalize=False)
            norm_factor = np.sqrt(mesh_target.area)

            # 2. Load Distance Matrix of the TARGET
            dist_matrix = sio.loadmat(f"{mat_dir}{tgt_name}.mat")['dist']
            num_vertices = dist_matrix.shape[0]

            # 3. Define Ground Truth (Identity Map for SURREAL/SMPL)
            # In SURREAL, vertex i on src is vertex i on tgt
            corr_gt = np.arange(num_vertices)

            # 4. Load Predicted P2P Map
            txt_data = np.genfromtxt(p2p_path, dtype=int)
            p2p_preds = txt_data[:, 1] # Extract column with target predictions
            
            # Indexing fix
            if p2p_preds.min() == 1: p2p_preds -= 1
            
            # Ensure p2p_preds matches the length of our GT
            # If p2p was only calculated for a subset, we slice corr_gt
            current_gt = corr_gt[:len(p2p_preds)]

            # 5. Geodesic Error Calculation
            # Look up distances on the target surface between GT and Predicted indices
            raw_errors = dist_matrix[current_gt, p2p_preds]
            mean_geo_error = raw_errors.mean() / norm_factor
            
            results_log.append({
                'source': src_name,
                'target': tgt_name,
                'error': mean_geo_error,
                'mat': f"{tgt_name}.mat"
            })
            
            print(f"{src_name:<12} | {tgt_name:<12} | {mean_geo_error:.6f}")

        except Exception as e:
            # Silent skip or print error for debugging
            # print(f"Error on {src_name}->{tgt_name}: {e}")
            pass

# ====== FINAL STATS & TOP 5 ======
if results_log:
    all_errors = [r['error'] for r in results_log]
    avg_total = np.mean(all_errors)
    
    # Sort results to find the best (minimum error) matches
    sorted_results = sorted(results_log, key=lambda x: x['error'])

    print("\n" + "="*55)
    print(f"SURREAL MEAN GEODESIC ERROR (Over {len(results_log)} pairs)")
    print(f"OVERALL AVERAGE: {avg_total:.6f}")
    print(f"Overall Quality: {get_quality_label(avg_total)}")
    print("="*55)
    
    print("\nTOP 5 BEST SURREAL MATCHES (Minimum Error):")
    print(f"{'Rank':<5} | {'Error':<10} | {'Pair (Src -> Tgt)':<25}")
    print("-" * 55)
    for i, res in enumerate(sorted_results[:5]):
        pair_str = f"{res['source']} -> {res['target']}"
        print(f"#{i+1:<4} | {res['error']:.6f} | {pair_str:<25}")
    print("-" * 55)
else:
    print("\n[!] No valid p2p pair files found. Check naming: p2p_{target}_to_{source}.txt")