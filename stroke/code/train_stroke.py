# %%
import tensorflow as tf
import numpy as np
import pylab as plt
import pandas as pd

# %%
# read train and test data
train_data = pd.read_csv('../data/train_2v.csv')
train_data = train_data.drop(['id'], axis=1)

test_data = pd.read_csv('../data/test_2v.csv')
test_data = test_data.drop(['id'], axis=1)

# %%
# do some data analysis before training and testing data
stroke_data = train_data[train_data['stroke'] == 1]
# stroke vs gender
count_female_stroke = len(stroke_data[stroke_data['gender'] == '0'])
count_male_stroke = len(stroke_data[stroke_data['gender'] == '1'])

# stroke vs age
count_age_stroke_map = {}
for i in range(10):
    min_age = i*10
    max_age = (i+1)*10
    count_age_stroke_map[str(i)] = len(
        stroke_data[(stroke_data['age'] >= min_age) & (stroke_data['age'] < max_age)])

# %%
# stroke vs
count_hypertension_stroke = len(
    stroke_data[stroke_data['hypertension'] == '1'])
print(count_hypertension_stroke)

ß  # %%
