import os
import yaml
import torch
import argparse
import numpy as np
from utils.training_utils import set_seed
from utils.mesh import load_off, save_obj, build_knn_edges


def convert_off_file(input_path, output_path, k):
    """Replace one OBJ graph while retaining original vertex coordinates/order."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    vertices, _, _ = load_off(input_path)
    vertices, edges = build_knn_edges(vertices, k=k)
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                       shape=(len(vertices), len(vertices)))
    components = connected_components(graph, directed=False, return_labels=False)
    if components != 1:
        raise ValueError(f'{input_path}: k={k} gives {components} disconnected components; '
                         'check the points or increase the graph neighbor count before training.')
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    temporary = str(output_path) + '.tmp'
    try:
        save_obj(temporary, vertices, np.array([]), edges)
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return output_path


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
        out_name = os.path.splitext(off_file)[0] + ".obj"
        out_path = os.path.join(output_folder, out_name)
        convert_off_file(in_path, out_path, k)
        print(f"  ✓ {off_file} processed")

if __name__ == '__main__':
    main()
