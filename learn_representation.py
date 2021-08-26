import numpy as np
from RBM import RBM
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import argparse
import os

parser = argparse.ArgumentParser(
    description='Ridle, learning a representation for entities using a target distributions over the usage of relations.',
)
parser.add_argument('--dataset', nargs='?', default='DBp_2016-04', type=str)
parser = parser.parse_args()

print('Learning Ridle Representations on', parser.dataset)
# Loading Files
df = pd.read_pickle('./dataset/{}/dataset.pkl'.format(parser.dataset))[['S', 'P']].drop_duplicates()

# Learning Representation
mlb = MultiLabelBinarizer()
mlb.fit([df['P'].unique()])
df_distr_s = df.groupby('S')['P'].apply(list).reset_index(name='Class')
X = mlb.transform(df_distr_s['Class'])
rbm = RBM(n_hidden=50, n_iterations=100, batch_size=100, learning_rate=0.01)
rbm.fit(X)


## Save Entity Representation
r = pd.DataFrame(rbm.compress(X), index=df_distr_s['S']).reset_index()
r.to_csv('./dataset/{}/embedding.csv'.format(parser.dataset), index=False)
