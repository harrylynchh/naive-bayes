'''
utils.py
4/19/2025
Harry Lynch
Utilities to aide in the classification of data tracks.  Those included here take
observed values and map to an index in a likelihood distribution.
'''

from globals import BIN_WIDTH_SIGMA, BIN_WIDTH_SPEED
import numpy as np

'''
speed_to_bin
Given a speed value, associate it with an index in the speed likelihood distribution
If the provided value is NaN, return a sentinel of -1
'''
def speed_to_bin(speed: float, n_bins: int) -> int:
    # If a NaN speed is given, return a sentinel
    if np.isnan(speed):
        return -1
    
    # Calculate which index of the likelihood distribution the velocity is associated with
    idx = int(round(speed / BIN_WIDTH_SPEED))
    # Ensure the idx is bound within the distribution
    return max(0, min(idx, n_bins - 1))

'''
sigma_to_bin
Given a rolling stdev value, associate it with an index in the stdev likelihood distribution
If the provided value is NaN, return a sentinel of -1
'''
def sigma_to_bin(sig: float, sig_min: float, n_bins: int) -> int:
    # If a NaN stdev is given, return a sentinel
    if np.isnan(sig):
        return -1
    # Calculate which index of the likelihood distribution the stdev is associated with
    idx = int(round((sig - sig_min) / BIN_WIDTH_SIGMA))
    # Ensure the idx is bound within the distribution
    return max(0, min(idx, n_bins - 1))