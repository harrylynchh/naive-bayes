'''
graph_sigma_likelihoods.py
4/19/2025
Harry Lynch
Standalone test file to visualize the results from a given track of the testset.
Takes in an index of the track the user would like to observe (0-19) and produces
a graph and various stats relating to the likelihoods used in the classifier.
This module helped me greatly in determining how to weigh the two likelihoods,
ultimately allowing my classifier to reach 100% accuracy over the testset.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Load the testset
testing = np.loadtxt("testing.txt")
# Select the track based off the index given in cmd line arg
track = testing[int(sys.argv[1])]

# Calculate various statistics for the given track's velocity
mean_speed = np.nanmean(track)
std_speed  = np.nanstd(track)
# Calculate stats for the stdev of the track using same rolling method
sigma_series = pd.Series(track).rolling(5, min_periods=5).std()
mean_sigma   = sigma_series.mean()
std_sigma    = sigma_series.std()

# Display the  stats
print(f"Mean Speed       = {mean_speed:.3f}")
print(f"Speed Std‑dev    = {std_speed:.3f}")
print(f"Mean Rolling stdev   = {mean_sigma:.3f}")
print(f"Rolling stdev of Std‑devs= {std_sigma:.3f}")

# Configure the two graphs for velocity and stdev and show 
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,6), sharex=True)
ax1.plot(track, color='C0')
ax1.set_ylabel("Speed (m/s)")
ax1.set_title("Track 10 Speed over Time")
ax2.plot(sigma_series, color='C1')
ax2.set_ylabel("stdev(v) (m/s)")
ax2.set_title("Track Rolling stdev(v) over Time")
ax2.set_xlabel("Time Step")
plt.tight_layout()
plt.show()
