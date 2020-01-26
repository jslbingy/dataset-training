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
# stroke vs gender
count_female_stroke = pd.DataFrame(train_data, gender == 1 & stroke == 1)
print(count_female_stroke)

# %%
