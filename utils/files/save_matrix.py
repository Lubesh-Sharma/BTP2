import os
import numpy as np

def save_masked_matrix(features, filename):
    """
    Saves the feature matrix (some values zeroed out) to a text file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
         
    np.savetxt(filename, features)
    print(f"Saved masked matrix to {filename}")
