# #### FOR MAC OSX USERS
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.impute import KNNImputer
from imblearn.under_sampling import RandomUnderSampler

class MissingValuesFiller():
    def __init__(self):
        pass



# To test
raw_df = pd.read_csv("src_data/train.csv")
print(raw_df)