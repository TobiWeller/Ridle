from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from keras import backend as K
import argparse
import os

parser = argparse.ArgumentParser(
    description='Instance Type Prediction using Ridle',
)
parser.add_argument('--dataset', nargs='?', default='DBp_2016-04', type=str)
parser = parser.parse_args()


# GELU Activation function
def gelu(x):
    return 0.5 * x * (1 + K.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))

# Load Representations
print('Reading Data...')
df = pd.read_csv('./dataset/{}/embedding.csv'.format(parser.dataset))


# Load mapping
if 'dbp' in parser.dataset.lower():
	mapping = pd.read_json('./dataset/dbp_type_mapping.json')
elif 'wd' in parser.dataset.lower() or 'wikidata' in parser.dataset.lower():
	mapping = pd.read_json('./dataset/wd_mapping_type.json')
else:
	mapping = pd.read_json('./dataset/{}/type_mapping.json'.format(parser.dataset))


# merge them
print('Processing Data...')
r = pd.merge(df, mapping, on='S')

K_FOLD = 10
mlb = MultiLabelBinarizer()
fold_no = 1
loss_per_fold, f1_macro, f1_micro, f1_weighted = [], [], [], []
kfold = KFold(n_splits=K_FOLD, shuffle=True, random_state=42)
targets = mlb.fit_transform(r['Class'])
inputs = r.drop(['S', 'Class'], axis=1).values
for train, test in kfold.split(inputs, targets):
	model = Sequential()
	model.add(Dense(inputs[train].shape[1], input_dim=inputs[train].shape[1]))
	model.add(Activation(gelu, name='Gelu'))
	model.add(Dense(targets[train].shape[1], activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	print('Training...')
	history = model.fit(inputs[train], targets[train], batch_size=64, validation_data=(inputs[test], targets[test]), epochs=100)
	y_pred = model.predict(inputs[test])
	y_pred[y_pred>=0.5]=1
	y_pred[y_pred<0.5]=0
	
	# Generate F1 scores
	scores = model.evaluate(inputs[test], targets[test], verbose=0)
	f1_macro.append(f1_score(targets[test], y_pred, average='macro', zero_division=1))
	f1_micro.append(f1_score(targets[test], y_pred, average='micro', zero_division=1))
	f1_weighted.append(f1_score(targets[test], y_pred, average='weighted', zero_division=1))

	print('Score for fold', fold_no, ':', model.metrics_names[0], 'of', scores[0], ';', 'F1-Macro:', f1_macro[-1], 'F1-Micro:', f1_micro[-1])
	loss_per_fold.append(scores[0])

	fold_no += 1
	
# Provide average scores
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(loss_per_fold)):
	print('------------------------------------------------------------------------')
	print('> Fold', i+1, ' - Loss:', loss_per_fold[i], '- F1-Macro:', f1_macro[i], '%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print('> F1-Macro:', np.mean(f1_macro), '(+-', np.std(f1_macro), ')')
print('> F1-Micro:', np.mean(f1_micro), '(+-', np.std(f1_micro), ')')
print('> Loss:', np.mean(loss_per_fold))
print('------------------------------------------------------------------------')

# Save results to file
result = {}
f1_macro = np.array(f1_macro)
f1_micro = np.array(f1_micro)
f1_weighted = np.array(f1_weighted)
result['F1-Macro'] = np.mean(f1_macro)
result['F1-Macro_std'] = np.std(f1_macro)
result['F1-Micro'] = np.mean(f1_micro)
result['F1-Micro_std'] = np.std(f1_micro)
result['F1-Weighted'] = np.mean(f1_weighted)
result['F1-Weighted_std'] = np.std(f1_weighted)
result['Dataset'] = parser.dataset
result['method'] = 'Ridle'
df_result = pd.DataFrame([result])
print(df_result)

if os.path.isfile('./evaluation_instance_type.csv'):
	df_result.to_csv('./evaluation_instance_type.csv', mode='a', header=False, index=False)
else:
	df_result.to_csv('./evaluation_instance_type.csv', index=False)

