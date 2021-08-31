import numpy as np
from RBM import RBM
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from pathlib import Path
import wget
import argparse
import os

parser = argparse.ArgumentParser(
    description='Ridle, learning a representation for entities using a target distributions over the usage of relations.',
)
parser.add_argument('--dataset', nargs='?', default='DBp_2016-04', type=str)
parser = parser.parse_args()


# https://www.dropbox.com/sh/szvuv79ubfqgmn5/AACHxl_eC0frcGrZpVy0VDQPa?dl=0


links = {}
links['dblp'] = 'https://www.dropbox.com/s/78srst5bjt2tta1/dataset.pkl?dl=1'
links['dbp_type_mapping'] = 'https://www.dropbox.com/s/2ec6dyr90pmjfm9/dbp_type_mapping.json?dl=1'
links['umls'] = 'https://www.dropbox.com/s/madbrirjc3yjtru/dataset.pkl?dl=1'
links['Person_DBpedia'] = 'https://www.dropbox.com/s/1omj2btnoj8g4xa/dataset.pkl?dl=1'
links['DBp_2016-04'] = 'https://www.dropbox.com/s/z38exis1ah3q5ze/dataset.pkl?dl=1'
links['Company_DBpedia'] = 'https://www.dropbox.com/s/bft3hmk2m6ecrkl/dataset.pkl?dl=1'
links['Songs_DBpedia'] = 'https://www.dropbox.com/s/u9k6qaydqowckae/dataset.pkl?dl=1'
links['Books_DBpedia'] = 'https://www.dropbox.com/s/wdqhov2g4bvwzr9/dataset.pkl?dl=1'
links['ChemicalCompounds_DBpedia'] = 'https://www.dropbox.com/s/fyyqgtwwf2pnj3b/dataset.pkl?dl=1'
links['Universities_DBpedia'] = 'https://www.dropbox.com/s/0g2moh3puz09uoy/dataset.pkl?dl=1'


if not os.path.isfile('./dataset/dbp_type_mapping.json'):
    print("Downloading dbp_type_mapping data.")
    data_url = links['dbp_type_mapping']
    wget.download(data_url, './dataset/dbp_type_mapping.json')


if not os.path.isfile('./dataset/{}/dataset.pkl'.format(parser.dataset)):
    print("Downloading {} data.".format(parser.dataset))
    data_url = links[parser.dataset]
    Path('./dataset/{}'.format(parser.dataset)).mkdir(parents=True, exist_ok=True)
    wget.download(data_url, './dataset/{}/dataset.pkl'.format(parser.dataset))



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
