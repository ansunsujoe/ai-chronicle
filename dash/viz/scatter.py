import seaborn as sns
import pandas as pd
import numpy as np

def scatter2d(array, hue_vals):
    data = np.concatenate((array, hue_vals), axis=1)
    df = pd.DataFrame(data, columns=["x1", "x2", "y"])
    sns.scatterplot(data=df, x="x1", y="x2", hue="y")