'''
globals.py
4/19/2025
Harry Lynch
Global variables for use across the I/O, likelihood generator, and classifier 
pertaining to distribution properties and algorithmic variables.
'''
# Distribution vars
BIN_WIDTH_SPEED = 0.5
BIN_WIDTH_SIGMA = 0.1
ROLL_WIN = 5

# Algorithm vars
SPEED_WEIGHT = 0.7
STAY_PROB = 0.9
SWITCH_PROB = 1.0 - STAY_PROB

CLASSES = ['b', 'a']
