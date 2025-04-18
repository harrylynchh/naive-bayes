import numpy as np
from pathlib import Path
import sys

def load_likelihoods(path: Path) -> np.ndarray:
    """Return an array of two rows from likelihood.txt. """
    arr = np.loadtxt(path)
    if arr.shape[0] != 2:
        sys.exit("likelihood.txt must have exactly 2 rows (bird, airplane)")
    return arr


def load_tracks(path: Path) -> np.ndarray:
    """ Return array with rows of tracks (each row is 600 data points long) """
    data = np.loadtxt(path)
    if data.shape[1] != 600:
        sys.exit(f"{path} must have 600 samples per track (found {data.shape[1]})")
    return data