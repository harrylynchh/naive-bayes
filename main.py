'''
main.py
4/19/2025
Harry Lynch
Entry point for the classifier, handles command line arguments and the I/O of
track/likelihood plaintext files into numpy arrays for further processing.
Outputs the results of the classification by sending a summary of the final determination
and expected class of the test set to stdout and the full report of iterations and final decisions
to a file called "classifier.out".
USAGE:
   python main.py {testset.txt} {speed_likelihood.txt} [OPTIONALLY: stdev_likelihoods.txt]
'''
from pathlib import Path
from classifier import classify_track
import sys
import data_io as io

args = sys.argv

if len(args) < 3 or len(args) > 4:
    print("Improper arguments: ")
    print("Usage: python main.py {testset.txt} {speed_likelihoods.txt} [stdev_likelihoods.txt]")

# Load test-set data into the tracks
test_tracks = io.load_tracks(Path(args[1]))

# Load likelihoods into arrays (only load sigmas if provided)
speed_likes = io.load_likelihoods(Path(args[2]))

# If the stdev likelihoods are provided
if len(args) == 4:
    sigma_likes, min_sig = io.load_sigma_likelihood(Path(args[3]))
else:
    sigma_likes = None
    min_sig = None

results = []
sols = ['b', 'b', 'b', 'a', 'a', 'b', 'a', 'a', 'a', 'b']
# Classify each track in the data and append to the results array to write out later
# NOTE: Prints final determinations to stdout and full iteration-by-iteration classifications
#       to "classifer.out"
out = Path("classifier.out")
for idx, track in enumerate(test_tracks):
    iters, final = classify_track(track, speed_likes, sigma_likes, min_sig)
    results.append(f"FINAL: {final}, ITERS: {iters}")
    print(f"---- Track {idx + 1} determined to be: {final} | ACTUAL: {sols[idx]} ----")

# Write the result of each track to it's own row in classifier.out
out.write_text('\n'.join(results) + '\n')

print("\n---- SEE 'classifier.out' for full iteration-by-iteration decisions ----")
