# %%
from collections import defaultdict
from scipy import stats
from scipy.stats import erlang, expon, norm 
from itertools import product, combinations
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import time # benchmarking

# User Defined Modules
import Graph

# %%
data = Graph.erdos_renyi_simulation_trial(5, 1, '0', '3', iters=10)

# %%
data

# %%



