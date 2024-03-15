from pickle import load
import pandas as pd

with open('model.pkl', 'rb') as f:
    model = load(f)

data = pd.read_csv("open_data.csv", index_col=0)

result = model.predict(data)

result = pd.Series(result)
result.to_csv("submission.csv", header=False) # или sample_submission.csv