import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from aux import one_day_row

def load_model(model_path):

    with open(model_path, 'rb') as f:
        model_artifact = pickle.load(f)

    return model_artifact

