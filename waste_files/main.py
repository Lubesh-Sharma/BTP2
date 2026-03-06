import argparse
import os
import torch
import yaml
from core.preprocessing import process_geometry
from core.trainer import train_asmae
from core.inference import run_inference_pipeline
from models.asmae import ASMAE

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description="Full Pipeline: HKS -> ASMAE -> Correspondence")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Paths from config
    input_dir = config.get('input_dir', 'input')
    output_dir = config.get('output_dir', 'output')
    shape1_path = os.path.join(input_dir, config['shape1_file'])
    shape2_path = os.path.join(input_dir, config['shape2_file'])
    
    os.makedirs(output_dir, exist_ok=True)
    
    device = config.get('device', 'cpu')
    if device == 'cuda' and not torch.cuda.is_available(): device = 'cpu'
    
    # 1. Processing
    # Need inputs in config: k, t
    k = config.get('k', 30)
    t = config.get('t', 100.0)
    
    VPos1, El1, Feat1, _ = process_geometry(shape1_path, k, t)
    VPos2, El2, Feat2, _ = process_geometry(shape2_path, k, t)
    
    feature_dim = Feat1.shape[1]
    
    # Initialize Model
    model = ASMAE(
        feature_dim=feature_dim,
        embed_dim=config.get('embed_dim', 128),
        depth=config.get('depth', 4),
        num_heads=config.get('num_heads', 4),
        decoder_embed_dim=config.get('decoder_embed_dim', 64),
        decoder_depth=config.get('decoder_depth', 2),
        decoder_num_heads=config.get('decoder_num_heads', 4),
        mlk_ratio=config.get('mlp_ratio', 2.)
    ).to(device)
    
    # 2. Training
    model, np1, np2 = train_asmae(model, VPos1, Feat1, VPos2, Feat2, config)
    
    # 3. Viz
    name1 = os.path.splitext(config['shape1_file'])[0]
    name2 = os.path.splitext(config['shape2_file'])[0]
    
    run_inference_pipeline(model, VPos1, Feat1, VPos2, Feat2, El1, El2, device, name1, name2, config)

if __name__ == "__main__":
    main()
