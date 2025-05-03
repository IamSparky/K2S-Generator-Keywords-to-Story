import pandas as pd
import numpy as np

df = pd.read_pickle("../../../outputs/stories_with_keywords_valid.pkl")

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df['Fold'] = np.tile(np.arange(5), int(np.ceil(len(df) / 5)))[:len(df)]