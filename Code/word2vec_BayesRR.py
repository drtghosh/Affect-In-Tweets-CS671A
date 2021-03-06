# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 14:42:09 2018

@author: Debabrata
"""

import gensim.models as gsm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import emot
import sys
#import os
import numpy as np
import preProcess
import tqdm

from correlation_pearson.code import CorrelationPearson
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold

orig_stdout = sys.stdout
f = open('Word2vec_BayesRR_results.txt', 'w')
sys.stdout = f

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
		f = open(x + "-" + datatypes[i] +"/" + x + "-" + y + "-" + datatypes[i] + ".txt", encoding="utf8")
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

#Bayesian Ridge Reg model
regMethod = "Bayesian Ridge Regression"
regModel = BayesianRidge(compute_score=True)

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
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = regModel
        model.fit(X_train, y_train)
        model_predicted = model.predict(X_test)
        Res.append(c.result(y_test, model_predicted))
        print(regMethod +"- Pearson Coefficient for "+ emotions[i] + ": ", c.result(y_test, model_predicted))
		
    print(regMethod + ":Avg of pearson-coefficients for the " + emotions[i] + " : ", sum(Res)/5)
    finRes.append(sum(Res)/5)

print("--------------------------------------------")
print("Final PC for "+ regMethod ,sum(finRes)/4)
print("--------------------------------------------")
	
sys.stdout = orig_stdout
f.close()