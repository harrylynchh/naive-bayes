'''
classifier.py
4/19/2025
Harry Lynch
File containing the primary algorithm for this naive bayesian classifier.  Uses
an implementation of recursive bayesian estimation to predict the identity of
an unknown flying object (either a bird or a plane) using velocity likelihoods
and optionally standard deviation of velocity likelihoods.  For each track,
returns an array of classifications and a final decision. 
'''
import numpy as np
import utils as utils
from globals import *

'''
Initialize a transition matrix which represents the probabilty that the next
classification is the same as the previous which is 0.9 per spec

Matrix looks like the following:
0.9. 0.1
0.1  0.9
'''
TRANS_MATRIX = np.array([[STAY_PROB, SWITCH_PROB], 
                        [SWITCH_PROB, STAY_PROB]])

"""
classify_track
Using a naive bayesian model, run a recursive bayesian estimation algorithm on one track
of a provided testset.  Assumes an equal prior at the beginning of each track [0.5, 0.5]
and uses a transition matrix with a transition probability of 0.9 as well as velocity
and (optionally) rolling standard deviation likelihoods to calculate a posterior probability
on every iteration formatted as [p(bird), p(plane)].  At the end of each iteration,
the object with the most classifications is determined as the identity of the track.
NOTES: 
If like_sig is None, only speed feature is used.
If both likelihoods are used, speed is weighed 0.7 to stdev's 0.3 (found through experimentation)
Returns per-sample labels (600 chars) and final label.
"""
def classify_track(track: np.ndarray,
                   like_speed: np.ndarray,
                   like_sig: np.ndarray | None,
                   sig_min: float | None) -> tuple[str, str]:
    
    # only use the weight if both are provided
    weight = SPEED_WEIGHT if like_sig is not None else 1.0

    # Determine how many measurements are in each likelihood distro
    n_speed_bins = like_speed.shape[1]
    n_sig_bins   = like_sig.shape[1] if like_sig is not None else 0

    # Initialize the prior at [p(bird) = 0.5, p(plane) = 0.5] per spec
    post = np.array([0.5, 0.5], dtype=float)
    # per keeps track of the per-iteration classifications
    per  = []
    # buf stores the 5 most recent speed entries and calculates the rolling
    # stdev at the 5th entry-- we then reference the track's stdev to the stdev
    # likelihoods calculated from the dataset if that file is provided at runtime
    buf  = []

    for speed in track:
        # Use the transition matrix to calculate probabilities given current
        # belief of bird/plane
        post = post @ TRANS_MATRIX
        # Start both likelihoods at 1.0 [bird, plane]
        like_vec = np.array([1.0, 1.0])
        # Get the current track entry for speed as a bin index
        idx_s = utils.speed_to_bin(speed, n_speed_bins)
        if idx_s >= 0:
            # If the index returned is within the distribution, multiply the
            # likelihood vector by the P(speed | bird) and P(speed | plane) respectively
            # NOTE: this likelihood is weighed higher than stdev
            like_vec *= (like_speed[:, idx_s] ** weight)
        # If a stdev likelihood is provided, contribute that update to the likelihoods
        if like_sig is not None:
            # Add the most recent speed to the buffer
            buf.append(speed)
            # Remove the oldest speed from the window
            if len(buf) > ROLL_WIN:
                buf.pop(0)
            # If we have 5 valid, non-NaN speeds, calculate the stdev over the 5
            if len(buf) == ROLL_WIN and not any(np.isnan(buf)):
                # calculate stdev and associate it to an idx on the distribution of likelihoods
                sigma = float(np.std(buf))
                idx_sig = utils.sigma_to_bin(sigma, sig_min, n_sig_bins)
                if idx_sig >= 0:
                    # Apply the weighted probabilities to the final likelihood vector same as line 50
                    like_vec *= (like_sig[:, idx_sig] ** (1 - weight))
        
        # Update posterior probability in light of the new likelihoods generated
        # by this iteration's measurements 
        post *= like_vec
        # Normalize so we have probabilities
        post /= post.sum()
        # Add the classification of whichever probability is larger after factoring
        # in likelihoods
        per.append(CLASSES[post.argmax()])
    
    # Tally per classifications and make a final decision-- tie goes to most recent observation
    b_cnt = per.count('b')
    a_cnt = per.count('a')
    if b_cnt > a_cnt:
        final = 'b'
    elif a_cnt > b_cnt:
        final = 'a'
    else:
        final = CLASSES[post.argmax()]
    
    # Return the per-iteration results and the final determination 
    return ''.join(per), final