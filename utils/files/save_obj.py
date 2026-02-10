import os
import numpy as np
from utils.mesh import save_obj

def save_masked_obj(pos, elements, mask, filename):
    """
    Saves a visualization of the masked shape.
    """
    if not filename.startswith("output/"):
         filename = os.path.join("output", os.path.basename(filename))
         
    visible_indices = np.where(mask == 0)[0]
    VPos_sub = pos[visible_indices]
    ITris_empty = np.zeros((0, 3), dtype=np.int32)
    VColors_sub = np.zeros_like(VPos_sub) + 0.5 
    save_obj(filename, VPos_sub, VColors_sub, ITris_empty)
    print(f"Saved masked geometry to {filename}")
