import gensim.models as gsm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import emot
import sys
import os
import numpy as np
import preProcess
import tqdm
import tensorflow as tf

from correlation_pearson.code import CorrelationPearson
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold


print("Loading models...")
w2v = gsm.KeyedVectors.load_word2vec_format('word2vec_twitter_model/word2vec_twitter_model.bin', binary=True, limit=500000)
e2v = gsm.KeyedVectors.load_word2vec_format('emoji2vec-master/pre-trained/emoji2vec.bin', binary=True)
print("Models Loaded.")

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


def produceWordEmbd(rawTweet):
	tweet = rawTweet

	# print(tweet)

	# Removing twitter handles' tags
	tweet = re.sub(r"@{1}[A-Za-z0-9_]+\s", ' ', tweet)

	# Removing web addresses
	tweet = re.sub(r"htt(p|ps)\S+", " ", tweet)

	# Removing email addresses
	emails = r'[a-zA-Z0-9._%-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}'
	tweet = re.sub(emails, " ", tweet)

	#Getting all emoticons together
	emojis_dict = emot.emoji(tweet)
	emojis = []
	for z in emojis_dict:
		emojis.append(z['value'])
		tweet = re.sub(z['value'], '', tweet)
	# print(tweet, emojis)
	# Tokenizing based on whitespaces
	tokens = word_tokenize(tweet)
	# print(tokens)

	# Getting hashtags intact
	newTokens = []
	for i,x in enumerate(tokens):
		if x == '#' and i < len(tokens)-1:
			y = x + tokens[i+1]
			newTokens.append(y)
		else:
			if i>0:
				if (tokens[i-1]!='#'):
					newTokens.append(x)
			else:
				newTokens.append(x)

	# Getting clitics intact
	finalTokens = []
	for j,x in enumerate(newTokens):
		S = ["'s", "'re", "'ve", "'d", "'m", "'em", "'ll", "n't"]
		if x in S:
			y = newTokens[j-1] + x
			finalTokens.append(y)
		else:
			if j<len(newTokens)-1:
				if newTokens[j+1] not in S:
					finalTokens.append(x)
			else:
				finalTokens.append(x)

	# Eliminate case sensitivity
	for i,z in enumerate(finalTokens):
		finalTokens[i] = z.lower()

	# Getting rid of stopwords
	stopwordSet = set(stopwords.words('english'))
	filteredFinalTokens = []
	for i,z in enumerate(finalTokens):
		if z not in stopwordSet:
			filteredFinalTokens.append(z)

	for x in filteredFinalTokens:
		u = re.split(r"\\n", x)
		for m in u:
			vocabulary.append(m)
	# print(filteredFinalTokens)

	words = filteredFinalTokens
	word_vecs = []
	for word in words:
		fr = np.zeros(400)
		if word in w2v.vocab:
			tr = w2v[word]
			for k in range(400):
				fr[k] = tr[k]
			word_vecs.append(fr)

	for emoji in emojis:
		yr = np.zeros(400)
		if emoji in e2v.vocab:
			zr = e2v[emoji]
			for k in range(300):
				yr[k] = zr[k]
			word_vecs.append(yr)

	return sum(word_vecs)/(len(word_vecs)+1)	
	pass

# print(word_vecs_tweet)

emotions = ['train_anger_', 'train_fear_', 'train_joy_', 'train_sadness_', 'dev_anger_', 'dev_fear_', 'dev_joy_', 'dev_sadness_', 'test_anger_', 'test_fear_', 'test_joy_', 'test_sadness_']
"""
# The following snippet if for tuning hyperparameters:
result=[]
for i in range(4):
	train_data = np.zeros((len(data[i][0]), 400))
	train_labels = np.zeros(len(data[i][0]))
	dev_data = np.zeros((len(data[i][0]), 400))
	dev_labels = np.zeros(len(data[i][0]))
	for j in tqdm.trange(len(data[i][0])):
		temp = produceWordEmbd(data[i][0][j])
		train_data[j] = temp
		train_labels[j] = data[i][1][j]
	h = i + 4 # index for dev sets, i is index for training sets
	
	for j in tqdm.trange(len(data[h][0])):
		temp = produceWordEmbd(data[h][0][j])
		dev_data[j] = temp
		dev_labels[j] = data[h][1][j]
	
	mlp  = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(9, 5, 7), random_state=1, activation='tanh', learning_rate='adaptive')
	mlp.fit(train_data, train_labels)
	mlp_predicted = mlp.predict(dev_data)
	# print(dev_labels - mlp_predicted)
	c = CorrelationPearson()
	result.append(c.result(dev_labels, mlp_predicted))
	print("pearson-coefficient for " + emotions[i] + ": ", c.result(dev_labels, mlp_predicted))

print("Avg of pearson-coefficients for all the four emotions: ", sum(result)/4)
"""


regMethods = [ "Neural Nets", "Decision Tree", "Random Forests", "K-NN", "ADA-Boost", "Gradient-Boost"]

regModels = [MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(9, 5, 7), random_state=1, activation='tanh', learning_rate='adaptive'),
DecisionTreeRegressor(random_state=0), RandomForestRegressor(max_depth=2, random_state=0), KNeighborsRegressor(n_neighbors=2),
AdaBoostRegressor(base_estimator=None, n_estimators=50, learning_rate=1.0),
GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, max_depth=3)
]

for z in range(6):
	finRes = []
	for i in range(4):
		X = np.zeros((len(data[i][0]), 400))
		y = np.zeros(len(data[i][0]))
		for j in tqdm.trange(len(data[i][0])):
			temp = produceWordEmbd(data[i][0][j])
			X[j] = temp
			y[j] = data[i][1][j]
		Res = []
		kf = KFold(n_splits=5)
		c = CorrelationPearson()
		for train_index, test_index in kf.split(X):
			#print("TRAIN:", train_index, "TEST:", test_index)
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]
			model = regModels[z]
			model.fit(X_train, y_train)
			model_predicted = model.predict(X_test)
			Res.append(c.result(y_test, model_predicted))
			print(regMethods[z] +"- Pearson Coefficient for "+ emotions[i] + ": ", c.result(y_test, model_predicted))
		
		print(regMethods[z] + ":Avg of pearson-coefficients for the " + emotions[i] + " : ", sum(Res)/5)
		finRes.append(sum(Res)/5)
	print("--------------------------------------------")
	print("Final PC for "+ regMethods[z] ,sum(finRes)/4)
	print("--------------------------------------------")



