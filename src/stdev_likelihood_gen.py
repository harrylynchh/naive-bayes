'''
stdev_likelihood_gen.py
4/19/2025
Harry Lynch
Facility to generate a likelihood distribution of rolling stdevs.  Takes a dataset
containing 20 total tracks, 10 bird and 10 plane in that order and executes standard
deviation calculations over a sliding window w/ pandas.  Writes to stdev_likelihood.txt
with the final distribution for both classes (top row birds bottom row planes) and a header
with the minimum observed value important for classification later.
'''

from data_io import load_tracks
from globals import BIN_WIDTH_SIGMA, ROLL_WIN

from pathlib import Path
import numpy as np
import pandas as pd
import sys 

"""
build_sigma_likelihoods
Uses a rolling window of size ROLL_WIN (see globals) to calculate stdevs over 
various intervals in the dataset.  Assumes that track indexes 0-9 are birds and 10-19 are planes.
Return a likelihood table for rolling stddev and the minimum value over a provided dataset
"""
def build_sigma_likelihoods(dataset: np.ndarray) -> tuple[np.ndarray, float]:
    
    bird_values: list[np.ndarray] = []
    plane_values: list[np.ndarray] = []

    for i, track in enumerate(dataset):
        # Compute "rolling std‑dev" by taking the stdev of a windows of size ROLL_WIN
        # Pandas ensures that if the window is too small NaN is added (which we eliminate later)
        sigma_series = pd.Series(track).rolling(ROLL_WIN, min_periods=ROLL_WIN).std().to_numpy()
      
        # Add to the proper class distribution based on track index in dataset.txt
        if i < 10:
            bird_values.append(sigma_series)
        else:
            plane_values.append(sigma_series)
          
    # Combine all the stddev data for all class tracks together
    bird_sig = np.concatenate(bird_values)
    plane_sig = np.concatenate(plane_values)

    # Get rid of all the NaNs
    bird_sig = bird_sig[~np.isnan(bird_sig)]
    plane_sig = plane_sig[~np.isnan(plane_sig)]

    # Find the bounds of the distribution (min/max stddevs over both class distributions)
    sig_min = float(min(bird_sig.min(), plane_sig.min()))
    sig_max = float(max(bird_sig.max(), plane_sig.max()))
    n_bins = int(round((sig_max - sig_min) / BIN_WIDTH_SIGMA)) + 1

    # Create histograms using numpy and the stdev calculations from above
    bird_hist, _ = np.histogram(
        bird_sig,
        bins=n_bins,
        range=(sig_min - BIN_WIDTH_SIGMA / 2, sig_max + BIN_WIDTH_SIGMA / 2),
    )
    plane_hist, _ = np.histogram(
        plane_sig,
        bins=n_bins,
        range=(sig_min - BIN_WIDTH_SIGMA / 2, sig_max + BIN_WIDTH_SIGMA / 2),
    )

    print("SIG MIN IS: ", sig_min)

    # "Laplace Smoothing" (SEE README) to prevent empty data points from
    # Tanking probability calculations later by multiplying P(bird) or P(plane)
    # by 0, next two lines add 1.0 to each "bin"
    bird_hist = bird_hist.astype(float) + 1.0
    plane_hist = plane_hist.astype(float) + 1.0

    # Normalize both histograms to generate the final distributions
    # (ensure all entries sum to 1)
    bird_like = bird_hist / bird_hist.sum()
    plane_like = plane_hist / plane_hist.sum()
    
    # Create a final np array which contains 2 rows,
    # bird stdev likelihoods, and plane stdev likelihoods
    like_table = np.vstack([bird_like, plane_like])
    return like_table, sig_min

args = sys.argv

if len(args) != 2:
    print("Please provide a path to dataset.txt")
    print("Usage: python ../data/stdev_likelihood_gen.py")
    sys.exit(1)

dataset_path = args[1]
dataset = load_tracks(dataset_path)
likelihoods, sig_min = build_sigma_likelihoods(dataset)

# Write out the likelihood distributions to stdev_likelihood.txt
# NOTE: This produces a "header" containing the min value of the distro
out = Path("../data/stdev_likelihood.txt")
with open(out, "w") as f:
    f.write(f"{sig_min}\n")
    np.savetxt(f, likelihoods, fmt="%.6e", delimiter=" ")
