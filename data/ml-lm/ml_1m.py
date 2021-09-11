#download data from :http://files.grouplens.org/datasets/movielens/ml-1m.zip
import sys
sys.path.append("../../")
import os
import pandas as pd
import numpy as np
from script.data import time_based_split_v4

data_path='ml-1m'
# ratings
r_names = ['uidx', 'iidx', 'rating', 'ts']
r_dtype = {'uidx':int, 'iidx':int, 'rating':float, 'ts':float}
ratings = pd.read_csv(os.path.join(data_path, 'ratings.dat'),
sep='::',
names=r_names,
dtype=r_dtype)

print(ratings.shape)
ratings.uidx = ratings.uidx - 1
ratings.iidx = ratings.iidx - 1
print(ratings.head())

# user
u_names = ['uidx', 'gender', 'age', 'occupation', 'zipcode']
u_dtype = {'uidx': int, 'gender': str, 'age': int, 'occupation': int, 'zipcode': str}
users = pd.read_csv(os.path.join(data_path, 'users.dat'),
sep='::',
names=u_names,
dtype=u_dtype)

print(users.shape)
users.uidx = users.uidx - 1
print(users.head())


# item
i_names = ['iidx', 'title', 'genre']
i_dtype = {'iidx': int, 'title': str, 'genre': str} # genre could be better encoded 
items = pd.read_csv(os.path.join(data_path, 'movies.dat'),
sep='::',
names=i_names,
dtype=i_dtype)

print(items.shape)
items.iidx = items.iidx - 1
print(items.head())

all_data = users.join(ratings.set_index('uidx'), on = 'uidx').join(items.set_index('iidx'), on = 'iidx')

# convert strings to categories
all_data.gender = all_data.gender.astype('category').cat.codes
all_data.occupation = all_data.occupation.astype('category').cat.codes
all_data.zipcode = all_data.zipcode.astype('category').cat.codes
all_data.title = all_data.title.astype('category').cat.codes
all_data.genre = all_data.genre.astype('category').cat.codes
print(all_data.head())

all_names = ["uidx", "gender", "age", "occupation", "zipcode", "iidx", "rating", "ts", "title", "genre"]
all_dtype = {"uidx": int, "gender": int, "age": int, "occupation": int, "zipcode": int, "iidx": int, "rating": float, "ts": float, "title": int, "genre": int}
all_data.to_csv(os.path.join(data_path, 'all_data.csv'), sep = ";", header = all_names, index = False)
time_based_split_v4(all_data, data_path, 20, 50)
