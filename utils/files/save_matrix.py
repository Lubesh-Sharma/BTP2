import os
import numpy as np

def save_masked_matrix(features, filename):
    """
    Saves the feature matrix (some values zeroed out) to a text file.
    """
    if not filename.startswith("output/"):
         filename = os.path.join("output", os.path.basename(filename))
         
    np.savetxt(filename, features)
    print(f"Saved masked matrix to {filename}")
