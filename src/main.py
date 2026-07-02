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
   python main.py {../data/testset.txt} {../data/speed_likelihoods.txt} [../data/stdev_likelihoods.txt]
'''
from pathlib import Path
from classifier import classify_track
import sys
import data_io as io

args = sys.argv

if len(args) < 3 or len(args) > 4:
    print("Improper arguments: ")
    print("Usage: python main.py {../data/testset.txt} {../data/speed_likelihoods.txt} [../data/stdev_likelihoods.txt]")
    sys.exit()
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
finals = []
# Load the answer key if present; if it's missing, skip the ACTUAL column
# and accuracy report and just print determinations and probabilities
sols_path = Path("../data/solutions.txt")
sols = sols_path.read_text().split() if sols_path.exists() else None
# Classify each track in the data and append to the results array to write out later
# NOTE: Prints final determinations to stdout and full iteration-by-iteration classifications
#       to "classifer.out"
out = Path("../output/classifier.out")
for idx, track in enumerate(test_tracks):
    iters, final, prob = classify_track(track, speed_likes, sigma_likes, min_sig)
    results.append(f"FINAL: {final}, ITERS: {iters}")
    finals.append(final)
    if sols is not None:
        print(f"---- Track {idx + 1} determined to be: [{final}] with probability {prob:.3f} --- ACTUAL: [{sols[idx]}] ----")
    else:
        print(f"---- Track {idx + 1} determined to be: [{final}] with probability {prob:.3f} ----")

# Write the result of each track to it's own row in classifier.out
out.write_text('\n'.join(results) + '\n')

# Calculate accuracy and output (only if an answer key was loaded)
if sols is not None:
    correct = 0
    for idx, res in enumerate(sols):
        if res == finals[idx]:
            correct += 1

    print (f"\n---- ACCURACY: {correct / len(sols) * 100:.2f} % ----")

print("\n---- SEE 'classifier.out' for full iteration-by-iteration decisions ----")
