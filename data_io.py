'''
data_io.py
4/19/2025
Harry Lynch
Utilities to read files into respective likelihoods and tracks.  Each takes in
a Path object and outputs a numpy array containing the requested information.
'''

from pathlib import Path
import numpy as np
import sys

def load_likelihoods(path: Path) -> np.ndarray:
    """Return an array of two rows from likelihood.txt. """
    arr = np.loadtxt(path)
    if arr.shape[0] != 2:
        sys.exit("likelihood.txt must have exactly 2 rows (bird, airplane)")
    return arr

'''
load_sigma_likelihood
Given the formatted output from stdev_likelihood_gen.py, extract the min stdev
from the distribution and the two rows of observed probabilities by class
top row is birds, bottom row is planes
'''
def load_sigma_likelihood(path: Path) -> tuple[np.ndarray, float]:
    # Read the first line containing the min value
    with open(path) as f:
        line = f.readline().strip()
    sig_min = float(line)

    # Load the remaining two rows of probabilities into a numpy array
    arr = np.loadtxt(path, skiprows=1)
    # Make sure the shape of the data is okay
    if arr.ndim == 1 or arr.shape[0] != 2:
        sys.exit(f"Error: {path} must have two probability rows after the header.")
    return arr, sig_min

'''
load_tracks
Given the path to the testset file, load each row into a multidimensional np array and return.
Each track should have 600 data points
'''
def load_tracks(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.shape[1] != 600:
        sys.exit(f"{path} must have 600 samples per track (found {data.shape[1]})")
    return data