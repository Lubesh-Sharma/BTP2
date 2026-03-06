import torch
import numpy as np
import os
from utils.mesh import save_obj
from utils.files import save_masked_matrix
from utils.matching import compute_nearest_neighbor_match, compute_sinkhorn_match
from utils.coloring import generate_colors_from_position, transfer_colors_by_indices
from .preprocessing import normalize_pc

def run_inference_pipeline(model, pos_s, feat_s, pos_t, feat_t, el1, el2, device, name1, name2, config):
    """
    Executes the full inference pipeline:
    1. Feature Extraction
    2. Matching (NN & Sinkhorn)
    3. Color Transfer
    4. Saving Results
    """
    model.eval()
    
    # Normalize for Model Inference
    pos_s_norm = normalize_pc(pos_s.copy())
    pos_t_norm = normalize_pc(pos_t.copy())
    
    # Prepare Inputs
    p_s = torch.tensor(pos_s_norm).float().unsqueeze(0).to(device)
    f_s = torch.tensor(feat_s).float().unsqueeze(0).to(device)
    p_t = torch.tensor(pos_t_norm).float().unsqueeze(0).to(device)
    f_t = torch.tensor(feat_t).float().unsqueeze(0).to(device)
    
    print("Running Inference Feature Extraction...")
    with torch.no_grad():
        # Extract features (Clean)
        emb_s = model.extract_features(f_s, p_s) # [1, N, C]
        emb_t = model.extract_features(f_t, p_t) # [1, M, C]
        
        # ---------------------------------------------------------
        # 0. Save Masked & Reconstructed Features (Debug/Viz)
        # ---------------------------------------------------------
        feature_ratio = config.get('feature_mask_group_ratio', 0.2)
        mask_ratio_pts = config.get('mask_ratio', 0.4)
        output_dir = config.get('output_dir', 'output')
        
        # Forward Pass S -> T
        pred_s, _, _, f_s_corr, _, final_mask_s = model(f_s, p_s, f_t, p_t, mask_ratio=mask_ratio_pts, feature_ratio=feature_ratio)
        # Forward Pass T -> S
        pred_t, _, _, f_t_corr, _, final_mask_t = model(f_t, p_t, f_s, p_s, mask_ratio=mask_ratio_pts, feature_ratio=feature_ratio)
        
        # Blend: Reconstructed = Original (Unmasked) + Prediction (Masked)
        # We want to show what the network "Regenerated". 
        # But if the user wants "rest of points remains same", we overwrite ONLY masked parts.
        rec_s_blended = f_s.clone()
        if final_mask_s is not None:
             rec_s_blended[final_mask_s] = pred_s[final_mask_s]
             
        rec_t_blended = f_t.clone()
        if final_mask_t is not None:
             rec_t_blended[final_mask_t] = pred_t[final_mask_t]
        
        # Save S
        save_masked_matrix(f_s_corr.squeeze(0).cpu().numpy(), os.path.join(output_dir, f"masked_matrix_{name1}.txt"))
        np.savetxt(os.path.join(output_dir, f"reconstructed_matrix_{name1}.txt"), rec_s_blended.squeeze(0).cpu().numpy())
        
        # Save T
        save_masked_matrix(f_t_corr.squeeze(0).cpu().numpy(), os.path.join(output_dir, f"masked_matrix_{name2}.txt"))
        np.savetxt(os.path.join(output_dir, f"reconstructed_matrix_{name2}.txt"), rec_t_blended.squeeze(0).cpu().numpy())
        
    # ---------------------------------------------------------
    # 1. Matching Algorithms
    # ---------------------------------------------------------
    
    # A. Nearest Neighbor (NN)
    # We need mappings in both directions potentially, but primarily T->S for coloring (Texture Pull)
    # User asked: "matching from source to target". So S->T.
    print("Computing Nearest Neighbor Matching (Source -> Target)...")
    matches_s2t_nn = compute_nearest_neighbor_match(emb_s, emb_t) # [N] (Indices of T)
    
    # For coloring Target, matching T->S is robust guarantees every target vertex has color
    matches_t2s_nn = compute_nearest_neighbor_match(emb_t, emb_s) # [M] (Indices of S)
    
    # B. Sinkhorn
    print("Computing Sinkhorn Matching...")
    matches_s2t_sink = compute_sinkhorn_match(emb_s, emb_t)
    matches_t2s_sink = compute_sinkhorn_match(emb_t, emb_s)
    
    # ---------------------------------------------------------
    # 2. Saving Matches
    # ---------------------------------------------------------
    output_dir = config.get('output_dir', 'output')
    
    np.savetxt(os.path.join(output_dir, f"matches_nn_{name1}_to_{name2}.txt"), matches_s2t_nn, fmt='%d')
    np.savetxt(os.path.join(output_dir, f"matches_sinkhorn_{name1}_to_{name2}.txt"), matches_s2t_sink, fmt='%d')
    
    # Also save reverse for debugging? Maybe useful.
    # np.savetxt(os.path.join(output_dir, f"matches_nn_{name2}_to_{name1}.txt"), matches_t2s_nn, fmt='%d')
    
    # ---------------------------------------------------------
    # 3. Coloring & Transfer
    # ---------------------------------------------------------
    print("Generating Colors...")
    # Generate Source Colors (Gradient based on position)
    colors_s = generate_colors_from_position(pos_s)
    
    # Transfer 1: NN (Using T->S map for dense coverage)
    colors_t_nn = transfer_colors_by_indices(colors_s, matches_t2s_nn)
    
    # Transfer 2: Sinkhorn (Using T->S map)
    colors_t_sink = transfer_colors_by_indices(colors_s, matches_t2s_sink)
    
    # Save Colored Meshes
    save_obj(os.path.join(output_dir, f"colored_{name1}.obj"), pos_s, colors_s, el1)
    save_obj(os.path.join(output_dir, f"colored_{name2}_nn.obj"), pos_t, colors_t_nn, el2)
    save_obj(os.path.join(output_dir, f"colored_{name2}_sinkhorn.obj"), pos_t, colors_t_sink, el2)
    
    print("Inference Complete. Matches and Colored shapes saved.")
