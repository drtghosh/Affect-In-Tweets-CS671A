import re
import numpy as np
import gensim.models as gsm
import os
#import stop_words
import gensim
import tqdm
import time
from nltk.tokenize import RegexpTokenizer
#from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim.models.doc2vec import TaggedDocument
import preProcess
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


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
#print(train_anger_len)
train_fear_len = len(data[1][0])
#print(train_fear_len)
train_joy_len = len(data[2][0])
#print(train_joy_len)
train_sadness_len = len(data[3][0])
#print(train_sadness_len)

train_len = train_anger_len + train_fear_len + train_joy_len + train_sadness_len

dev_anger_len = len(data[4][0])
#print(dev_anger_len)
dev_fear_len = len(data[5][0])
#print(dev_fear_len)
dev_joy_len = len(data[6][0])
#print(dev_joy_len)
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

all_docs_list = []

for i in data:
    for j in range(len(i[0])):
        all_docs_list.append(i[0][j])


print("Total number of documents to be trained: ",len(all_docs_list))

def get_tag_str(index):
    t_idx = next(x[0] for x in enumerate(sum_arr) if x[1] > index)
    # Note, t_idx is in {0,1,2,..,11}
    if t_idx == 0:
        return "train_anger_" + str(index)
    if t_idx == 1:
        return "train_fear_" + str(index)
    if t_idx == 2:
        return "train_joy_" + str(index)
    if t_idx == 3:
        return "train_sadness_" + str(index)
    if t_idx == 4:
        return "dev_anger_" + str(index)
    if t_idx == 5:
        return "dev_fear_" + str(index)
    if t_idx == 6:
        return "dev_joy_" + str(index)
    if t_idx == 7:
        return "dev_sadness_" + str(index)
    if t_idx == 8:
        return "test_anger_" + str(index)
    if t_idx == 9:
        return "test_fear_" + str(index)
    if t_idx == 10:
        return "test_joy_" + str(index)
    if t_idx == 11:
        return "test_sadness_" + str(index)
    pass

def get_doc(doc_list):
 
    taggeddoc = []
 
    for index,i in enumerate(doc_list):
        tweet = i
        print("Processing document number: ", index)
        # Pre process the tweet

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
        for j,z in enumerate(finalTokens):
            finalTokens[j] = z.lower()

        # Getting rid of stopwords
        stopwordSet = set(stopwords.words('english'))
        filteredFinalTokens = []
        for j,z in enumerate(finalTokens):
            if z not in stopwordSet:
                filteredFinalTokens.append(z)

        TokenF = []
        for x in filteredFinalTokens:
            u = re.split(r"\\n", x)
            for m in u:
                TokenF.append(m)

        tag_idx = get_tag_str(index)
        td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(TokenF))).split(), [tag_idx])
        # for later versions, you may want to use: td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(stemmed_tokens))).split(),[str(index)])
        taggeddoc.append(td)
 
    return taggeddoc

docData = get_doc(all_docs_list)

from gensim.models import Doc2Vec
import gensim.models.doc2vec
from collections import OrderedDict
import multiprocessing
import time

cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

simple_models = [
    # PV-DBOW 
    Doc2Vec(dm=0, vector_size=100,  min_count = 3,window = 10, negative=5, hs=0, workers=cores),
    # PV-DM 
    Doc2Vec(dm=1, vector_size=300, window=10, negative=5, hs=0, min_count=2, workers=cores),
]

model = simple_models[0]
model.build_vocab(docData, update=False)

# start training
start = time.clock()
print("The training will begin now...")
for epoch in tqdm.trange(100):
    model.train(docData, total_examples = model.corpus_count, epochs=model.epochs)
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay
end = time.clock()

print("The running time was:", end-start)
model.save(os.path.join('trained_models', 'doc_vectors_DBoW_expt100-100.d2v'))