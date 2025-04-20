# README.md
## Harry Lynch
### 4/19/2025
### Naive Bayesian Classifier for Identifying UFOs
---
# About:

    This program is designed to classify unidentified flying objects as either
    birds or airplanes based off of radar data on the velocity of these objects
    over a 600 second interval.  This is done with a Naive Bayesian Classifer
    which hinges on a recursive bayesian estimation algorithm as well as 
    likelihood distributions for speeds of birds and planes as well as a second
    feature derived from the first-- rolling standard deviation likelihoods.

    At it's core, this program takes "tracks" of object data (600 measurements)
    of the velocity of the object over time.  Over each track, we establish a
    probability matrix of [p(bird), p(plane)] which is initialized at [0.5, 0.5]

    On each new measurement, we first apply a transition matrix that biases the
    current posterior by the previous measurements-- so, if the current state
    of the vector is [0.73, 0.27] (BIRD), the transition matrix would further 
    increase the probability of the object being a BIRD.  This probability is
    0.9 in my implementation and is applied on every measurement.  After this,
    we refer the current velocity measurement to the distribution of likelihoods
    returning a matrix of [P(speed | bird), P(speed | plane)] from the likelihood.
    We multiply our posterior by this matrix to alter how likely we think a UFO is
    either object based on the probability of it appearing at that speed. This same
    process is applied to the standard deviation over the past 5 measurements.
    We then normalize the posterior and make a classification based on which of
    [p(bird), p(plane)] is higher.  Save the decision and repeat for all measurents.
    
    NOTE: In initial tests, just using the speed likelihood netted an accuracy of
    90% with the accuracy of both likelihoods also being 90%.  To improve the system,
    I attempted a few things.  I adjusted the bin width of the stdevs to hopefully
    improve the curve to no avail and I also tried shifting transition probabilities.
    None of these worked so I added weights to the likelihoods.  The weight that got me
    to 100% accuracy on the test set was weighing the speed likelihood 0.7 and 
    the stdev 0.3.  I believe this was because the track that was getting confused (10)
    had an abnormally small stdev for a bird and weighing it less tuned the algorithm
    enough to make the right decision.

    In determining which feature to add aside from speed, I landed on stdev after
    looking at the second figure in the spec of the test set data.  The birds wildly
    vary their speeds compared to the planes which remain constant for most of the measurement.
    This pointed me toward standard deviation as a way to differentiate the two objects further.
    These suspicions were further confirmed when I graphed each track using graph_track.py and observed
    the difference in the statistics.

---
# Usage:

    - This program relies on a small handful of dependencies for data visualization,
      preparation, and linear algebra calculations (matrix multiplication) before running,
      start up a virtual environment of your choosing and run:
              ** pip install < requirements.txt ** 
      This will install all dependencies listed in requirements.txt
    
    - After the environment is set up, to view the classifier run the following
              ** python main.py testing.txt likelihood.txt [stdev_likelihood.txt] **
      inclusion of the standard deviation likelihood is optional such that the difference
      in performance can be observed between the two.
    
    - The program will analyze the dataset and return, for each track, 
        - A determination
        - The probability of that determination (how sure the program is)
        - The actual solution of the given testset ('b' is bird, 'a' is airplane)
        - The accuracy over the testset comapred to the given solutions as a percent
    - NOTE: In-depth iteration-by-iteration output is written to a "classifier.out" file
            featuring the final determination and each data point's determination

# Resources:
    
    - I referenced StackOverflow a few times for some help with MatPlotLib and I heard
      of the library through Derrick Chaney's Piazza post as well as some data science hobbying I do
    - I extensively referenced the numpy and pandas documentation throughout this project as
      most of the math and manipulation is done through numpy data structures or pandas statistics
      calculations.  They're linked here [Pandas Rolling Stdev](https://pandas.pydata.org/docs/reference/api/pandas.core.window.rolling.Rolling.std.html), [NumPy](https://numpy.org/doc/stable/reference/arrays.ndarray.html), [MatPlotLib](https://matplotlib.org/stable/tutorials/pyplot.html)

**External packages required to run this program are listed in `requirement.txt`**