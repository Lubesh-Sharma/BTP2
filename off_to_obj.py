import os
import yaml
import torch
import argparse
import numpy as np
from utils.training_utils import set_seed
from utils.mesh import load_off, save_obj, build_knn_edges

def main():
    parser = argparse.ArgumentParser(description="OFF to OBJ Conversion")
    parser.add_argument('--config', type=str, default='config/off_to_obj.yaml', help='Path to config file')
    args = parser.parse_args()
    
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    set_seed(config)
    
    # Simple device selection as in other training scripts
    device = torch.device('cuda' if torch.cuda.is_available() and config.get('n_gpu', 0) > 0 else 'cpu')
    print(f"Using device: {device}")
    
    input_folder = config['input_folder']
    output_base = config['output_folder_path']
    k = config['k']
    train_size = config['train_size']
    test_size = config['test_size']
    
    # Final path logic: output_folder_path + /k_{k}
    output_folder = os.path.join(output_base, f"k_{k}")
    os.makedirs(output_folder, exist_ok=True)
    
    if not os.path.exists(input_folder):
        print(f"Error: Input folder {input_folder} not found.")
        return

    off_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.off')])
    limit = min(train_size + test_size, len(off_files))
    off_files = off_files[:limit]
    
    print(f"Processing {len(off_files)} files from {input_folder}")
    print(f"Saving results to {output_folder}")
    
    for off_file in off_files:
        in_path = os.path.join(input_folder, off_file)
        try:
            # Use utility functions from utils.mesh
            VPos, VColors, ITris = load_off(in_path)
            
            # Build kNN edge-based connectivity graph (Algorithm from coe_old_embedding/export_embedding_mesh.py)
            V, Edges = build_knn_edges(VPos, k=k)
            
            out_name = os.path.splitext(off_file)[0] + ".obj"
            out_path = os.path.join(output_folder, out_name)
            
            # Save as OBJ with lines (l)
            save_obj(out_path, V, np.array([]), Edges)
            print(f"  ✓ {off_file} processed")
            
        except Exception as e:
            print(f"  ✗ Error processing {off_file}: {e}")

if __name__ == '__main__':
    main()