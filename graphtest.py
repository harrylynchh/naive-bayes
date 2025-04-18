import numpy as np
import matplotlib.pyplot as plt

sigma_file = "stdev_likelihood.txt"

with open(sigma_file) as f:
    sig_min = float(f.readline().strip())
like_sig = np.loadtxt(sigma_file, skiprows=1)

n_bins = like_sig.shape[1]
print ("Number of bins: ", n_bins)
BIN_WIDTH_SIGMA = 0.1
sig_axis = sig_min + np.arange(n_bins) * BIN_WIDTH_SIGMA

plt.figure(figsize=(8, 4))
plt.plot(sig_axis, like_sig[0], label="Birds")
plt.plot(sig_axis, like_sig[1], label="Airplanes")
plt.xlabel("Rolling σ(v) bin center [m/s]")
plt.ylabel("P(σ-bin | class)")
plt.title("Likelihood curves for rolling speed standard deviation")
plt.legend()
plt.tight_layout()
plt.show()
