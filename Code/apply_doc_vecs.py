import gensim
import numpy as np
from gensim.models import Doc2Vec
import time
import tqdm
import os
import re
import preProcess

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from correlation_pearson.code import CorrelationPearson

from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold



print("Loading model of DBoW para2vec:")
model = Doc2Vec.load('trained_models/doc_vectors_DBoW_expt100-100.d2v')


folders = ["EI-reg-En", "2018-EI-reg-En", "2018-EI-reg-En"]
datatypes = ["train", "dev", "test"]
emotions = ["anger", "fear", "joy", "sadness"]

data = []
vocabulary = []

for i,x in enumerate(folders):
    for j,y in enumerate(emotions):
        f = open(x + "-" + datatypes[i] +"/" + x + "-" + y + "-" + datatypes[i] + ".txt")
        raw = f.read()
        g = preProcess.getData(raw)
        data.append(g)

print(len(data))

train_anger_len = len(data[0][0])
train_fear_len = len(data[1][0])
train_joy_len = len(data[2][0])
train_sadness_len = len(data[3][0])

train_len = train_anger_len + train_fear_len + train_joy_len + train_sadness_len

dev_anger_len = len(data[4][0])
dev_fear_len = len(data[5][0])
dev_joy_len = len(data[6][0])
dev_sadness_len = len(data[7][0])

dev_len = dev_anger_len + dev_fear_len + dev_joy_len + dev_sadness_len

test_anger_len = len(data[8][0])
test_fear_len = len(data[9][0])
test_joy_len = len(data[10][0])
test_sadness_len = len(data[11][0])

test_len = test_anger_len + test_fear_len + test_joy_len + test_sadness_len

sm = 0
sum_arr = []
for j in range(len(data)):
    sm += len(data[j][0])
    sum_arr.append(sm)

print(sum_arr)

emotions = ['train_anger_', 'train_fear_', 'train_joy_', 'train_sadness_', 'dev_anger_', 'dev_fear_', 'dev_joy_', 'dev_sadness_']
for i in range(4):
	train_arrays = np.zeros((len(data[i][0]), 100))
	train_labels = np.zeros(len(data[i][0]))
	dev_arrays = np.zeros((len(data[i][0]), 100))
	dev_labels = np.zeros(len(data[i][0]))
	j = i + 4 # index for dev sets, i is index for training sets
	print("Creating training arrays for "+emotions[i])
	if i == 0:
		for k in tqdm.trange(0, sum_arr[i]):
			train_arrays[k] = model[emotions[i]+str(k)]
			train_labels[k] = data[i][1][k]
	else:
		for k in tqdm.trange(sum_arr[i-1], sum_arr[i]):
			train_arrays[k - sum_arr[i-1]] = model[emotions[i]+str(k)]
			train_labels[k - sum_arr[i-1]] = data[i][1][k - sum_arr[i-1]]
	print("Creating dev arrays for "+emotions[j])
	for k in tqdm.trange(sum_arr[j-1], sum_arr[j]):
		dev_arrays[k - sum_arr[j-1]] = model[emotions[j]+str(k)]
		dev_labels[k - sum_arr[j-1]] = data[j][1][k - sum_arr[j-1]]
	print("Training a Neural Network with ")
	mlp  = MLPRegressor(solver='sgd', alpha=1e-5, hidden_layer_sizes=(11, 6, 5), random_state=1, activation='relu', learning_rate='adaptive')
	mlp.fit(train_arrays, train_labels)
	mlp_predicted = mlp.predict(dev_arrays)
	# print(dev_labels - mlp_predicted)
	c = CorrelationPearson()
	print("pearson-coefficient for " + emotions[i] + ": ", c.result(dev_labels, mlp_predicted))

regMethods = [ "Neural Nets", "Decision Tree", "Random Forests", "K-NN", "ADA-Boost", "Gradient-Boost"]
regModels = [MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(9, 5, 7), random_state=1, activation='tanh', learning_rate='adaptive'),
DecisionTreeRegressor(random_state=0), RandomForestRegressor(max_depth=2, random_state=0), KNeighborsRegressor(n_neighbors=2),
AdaBoostRegressor(base_estimator=None, n_estimators=50, learning_rate=1.0),
GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, max_depth=3)
]

for z in tqdm.trange(6):
	finRes = []
	for i in range(4):
		X = np.zeros((len(data[i][0]), 100))
		y = np.zeros(len(data[i][0]))
		print("Creating training arrays for "+ emotions[i])
		if i == 0:
			for k in range(0, sum_arr[i]):
				X[k] = model[emotions[i]+str(k)]
				y[k] = data[i][1][k]
		else:
			for k in range(sum_arr[i-1], sum_arr[i]):
				X[k - sum_arr[i-1]] = model[emotions[i]+str(k)]
				y[k - sum_arr[i-1]] = data[i][1][k - sum_arr[i-1]]
		Res = []
		kf = KFold(n_splits=5)
		c = CorrelationPearson()
		for train_index, test_index in kf.split(X):
			#print("TRAIN:", train_index, "TEST:", test_index)
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]
			Rmodel = regModels[z]
			Rmodel.fit(X_train, y_train)
			Rmodel_predicted = Rmodel.predict(X_test)
			Res.append(c.result(y_test, Rmodel_predicted))
			print(regMethods[z] +"- Pearson Coefficient for "+ emotions[i] + ": ", c.result(y_test, Rmodel_predicted))
		
		print(regMethods[z] + ":Avg of pearson-coefficients for the " + emotions[i] + " : ", sum(Res)/5)
		finRes.append(sum(Res)/5)
	print("--------------------------------------------")
	print("Final PC for "+ regMethods[z] ,sum(finRes)/4)
	print("--------------------------------------------")


				