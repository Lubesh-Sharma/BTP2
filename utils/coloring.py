import numpy as np

def generate_colors_from_position(pos):
    """
    Generates RGB colors based on 3D position (normalized coordinates).
    
    Args:
        pos: [N, 3] Numpy array of positions.
        
    Returns:
        colors: [N, 3] Numpy array of colors in [0, 1].
    """
    pos_min = np.min(pos, axis=0)
    pos_max = np.max(pos, axis=0)
    
    # Normalize X, Y, Z to 0..1 range
    # Add epsilon to avoid div by zero
    colors = (pos - pos_min) / (pos_max - pos_min + 1e-8)
    return colors

def transfer_colors_by_indices(source_colors, matches):
    """
    Transfers colors from Source to Target based on matching indices.
    
    Args:
        source_colors: [N_s, 3] Colors of source points.
        matches: [N_t] Index array where matches[j] is the index of the Source point that matches Target point j.
                 WAIT! The matching logic in utils/matching.py returned 'matches' as [N_source] mapping S -> T.
                 To color Target, we usually need "For each target point, who is my source match?".
                 
                 If matching is Source -> Target (i.e., matches[i] = j implies S[i] maps to T[j]),
                 then we have a forward map.
                 But to color Target[j], we need to know which S[i] maps to it.
                 
                 Since Embedding comparison is symmetric, usually we compute matches for Target query:
                 "For each T point, find best S point".
                 
                 Implementation Check in matching.py:
                 matches = argmax(sim_matrix, dim=2)
                 sim_matrix is [B, N_s, N_t].
                 matches[i] returns the index j in Target that is closest to Source i.
                 So this is Source -> Target map.
                 
                 To color the Target, we need Target -> Source map.
                 "For each pixel in Target, what is its color from Source?"
                 
                 So we need to Run Matching in REVERSE (Target -> Source) or invert the map (scatter).
                 Ideally, we compute matches `compute_nearest_neighbor_match(emb_t, emb_s)`.
    """
    # If matches represents T -> S indices (length N_t, values in range 0..N_s-1)
    target_colors = source_colors[matches]
    return target_colors
