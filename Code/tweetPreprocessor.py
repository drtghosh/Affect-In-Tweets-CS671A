import gensim.models as gsm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import emot
import sys
import numpy as np
import preProcess
import tqdm
import tensorflow as tf


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

print(len(data))

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

	# Treating emojis
	word_vecs = []
	for j,y in enumerate(filteredFinalTokens):
		z = emot.emoji(y)
		if z == []:
			if y in w2v.vocab:
				# print("Adding word vector for " + y)
				word_vecs.append(w2v[y])
		else:
			w = re.findall(r"[A-Za-z0-9]+", y)
			s1 = np.ndarray(shape=(300,1), dtype=float)
			s2 = np.ndarray(shape=(400,1), dtype=float)
			if w != []:
				w = w[0]
				t = re.sub(w,'',y)
				#s1 = np.ndarray(300, 1)
				#s2 = np.zeros(400, 1)
				if t in e2v.vocab:
					# print("Computing emoji vector for " + t)
					s1 = e2v[t]
					# print(type(s1))
					if w in w2v.vocab:
						# print("Computing vector for word-part " + w)
						s2 = w2v[w]
					#s1 = [s1, np.zeros(100,1)]
					
				N = 100
				s1 = np.pad(s1, (0, N), 'constant')
				word_vecs.append(s1 + s2)
	pass

for x in tqdm.trange(len(data)):
	s = data[x][0]
	for i in tqdm.trange(len(s)):
		print("Word embedding for ", i)
		produceWordEmbd(s[i])

L = len(vocabulary)

wordVectors = []

for l in tqdm.trange(L):
	y = vocabulary[l]
	z = emot.emoji(y)
	if z == []:
		if y in w2v.vocab:
			# print("Adding word vector for " + y)
			wordVectors.append(w2v[y])
	else:
		w = re.findall(r"[A-Za-z0-9]+", y)
		s1 = np.ndarray(shape=(300,1), dtype=float)
		s2 = np.ndarray(shape=(400,1), dtype=float)
		if w != []:
			w = w[0]
			t = re.sub(w,'',y)
			#s1 = np.ndarray(300, 1)
			#s2 = np.zeros(400, 1)
			if t in e2v.vocab:
				# print("Computing emoji vector for " + t)
				s1 = e2v[t]
				# print(type(s1))
				if w in w2v.vocab:
					# print("Computing vector for word-part " + w)
					s2 = w2v[w]
				#s1 = [s1, np.zeros(100,1)]
				
			N = 100
			s1 = np.pad(s1, (0, N), 'constant')
			wordVectors.append(s1 + s2)

print(vocabulary[100], wordVectors[100])

