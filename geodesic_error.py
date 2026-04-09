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
off_dir = './input/SCAPE/off/'
mat_dir = './input/SCAPE/mat/'
vts_dir = './input/SCAPE/corres/'
p2p_dir = './p2p_results_st_te_SCAPE/'

# 1. Get the last 20 files for the test set
all_off_files = sorted([f for f in os.listdir(off_dir) if f.endswith('.off')])
test_files = all_off_files[-20:] 
mesh_names = [os.path.splitext(f)[0] for f in test_files]

results_log = []

print(f"{'Source':<12} | {'Target':<12} | {'Geo Error':<10}")
print("-" * 45)

# ====== NESTED LOOP FOR PAIRWISE EVALUATION (400 possible pairs) ======
for src_name in mesh_names:
    for tgt_name in mesh_names:
        
        # Skip self-matching (e.g., 085 to 085) as it's not a valid test of deformation
        if src_name == tgt_name:
            continue
            
        # Naming convention: p2p_{TARGET}_to_{SOURCE}.txt
        p2p_filename = f"p2p_{tgt_name}_to_{src_name}.txt"
        p2p_path = os.path.join(p2p_dir, p2p_filename)
        
        # Only process if the prediction file exists
        if not os.path.exists(p2p_path):
            continue
            
        try:
            # 1. Load Target Mesh Area (for DV-Matcher sqrt(Area) norm)
            mesh_target = TriMesh(f"{off_dir}{tgt_name}.off", area_normalize=False)
            norm_factor = np.sqrt(mesh_target.area)

            # 2. Load Distance Matrix of the TARGET
            dist_matrix = sio.loadmat(f"{mat_dir}{tgt_name}.mat")['dist']

            # 3. Load Ground Truth Correspondences
            # .vts files map vertices to a common template; subtracting 1 for 0-based
            corr_target = np.loadtxt(f"{vts_dir}{tgt_name}.vts", dtype=np.int32) - 1
            corr_source = np.loadtxt(f"{vts_dir}{src_name}.vts", dtype=np.int32) - 1

            # 4. Load Predicted P2P Map
            txt_data = np.genfromtxt(p2p_path, dtype=int)
            p2p_preds = txt_data[:, 1] # Extract target predictions column
            
            if p2p_preds.min() == 1: p2p_preds -= 1

            # 5. Geodesic Error Calculation
            # predictions: what the model says target vertex is for each source GT point
            predictions = p2p_preds[corr_source]
            
            # raw_errors: distance on target mesh between GT target and Prediction
            raw_errors = dist_matrix[corr_target, predictions]
            mean_geo_error = raw_errors.mean() / norm_factor
            
            results_log.append({
                'source': src_name,
                'target': tgt_name,
                'error': mean_geo_error,
                'mat_file': f"{tgt_name}.mat"
            })
            
            print(f"{src_name:<12} | {tgt_name:<12} | {mean_geo_error:.6f}")

        except Exception as e:
            # print(f"Error on {src_name}->{tgt_name}: {e}")
            pass

# ====== FINAL ANALYSIS ======
if results_log:
    all_errors = [r['error'] for r in results_log]
    total_avg = np.mean(all_errors)
    
    # Sort by error (Ascending)
    sorted_results = sorted(results_log, key=lambda x: x['error'])
    
    print("\n" + "="*55)
    print(f"FAUST MEAN GEODESIC ERROR (Over {len(results_log)} pairs)")
    print(f"OVERALL AVERAGE: {total_avg:.6f}")
    print(f"Overall Quality: {get_quality_label(total_avg)}")
    print("="*55)
    
    print("\nTOP 5 BEST FAUST MATCHES (Minimum Error):")
    print(f"{'Rank':<5} | {'Error':<10} | {'Pair (Src -> Tgt)':<20} | {'Target .mat'}")
    print("-" * 65)
    for idx, res in enumerate(sorted_results[:5]):
        pair_str = f"{res['source']} -> {res['target']}"
        print(f"#{idx+1:<4} | {res['error']:.6f} | {pair_str:<20} | {res['mat_file']}")
    print("-" * 65)
else:
    print("\n[!] No valid p2p files found. Ensure naming is: p2p_{target}_to_{source}.txt")