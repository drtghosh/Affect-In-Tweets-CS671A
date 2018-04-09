import re
import numpy as np
import gensim.models as gsm
import os
import stop_words
import gensim
import tqdm
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim.models.doc2vec import TaggedDocument

folders = ['aclImdb/train/pos', 'aclImdb/train/neg', 'aclImdb/test/pos', 'aclImdb/test/neg']

def sort_custom(files):
    l = len(files)
    file_ids = np.zeros(l)
    for i, x in enumerate(files):
        pattern = re.compile(r"\d{1,5}_")
        match = pattern.findall(x)
        match[0] = match[0][:-1]
        t = int(match[0])
        file_ids[i] = t
    X = files
    Y = file_ids
    Z = [x for _,x in sorted(zip(Y,X))]
    return Z
    pass

def get_doc_list(folder_name):
    doc_list = [] # Has the list of all documents (text data)
    # file_list has the names of all the files in the folder_name
    file_list = [folder_name+'/'+name for name in os.listdir(folder_name) if name.endswith('txt')]
    file_list = sort_custom(file_list)
    for file in file_list:
        st = open(file,'r').read()
        doc_list.append(st)
    print ('Found %s documents under the dir %s .....'%(len(file_list),folder_name))
    return doc_list

data = []
for j in tqdm.trange(4):
	t = get_doc_list(folders[j])
	for s in range(len(t)):
		data.append(t[s])

def split_doc_into_sent(doc):
    pattern = re.compile(r"([^A-Z]|I)[^A-Z\.](\.|\.'|\?|\?'|!|!'|-')\s+[A-Z'*][^\.]")
    match = pattern.finditer(doc)
    split_idx = []
    split_idx.append(0)
    for x in match:
        split_idx.append(x.span()[0]+3)
    parts = [doc[i:j] for i,j in zip(split_idx, split_idx[1:]+[None])]
    return parts
    pass

sentences = []
sentence_id = []
for z in data:
    y = split_doc_into_sent(z)
    if len(sentence_id) >= 1:
        h = sentence_id[len(sentence_id)-1]
    else:
        h = 0
    sentence_id.append(h + len(y)) # Number of sentences in given document
    for t in y:
        sentences.append(t)

print("Number of Sentences found:", len(sentences))

def get_doc(doc_list):
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')
    p_stemmer = PorterStemmer()
 
    taggeddoc = []
 
    texts = []
    for index,i in enumerate(doc_list):
        print("Number of Sentences processed:", index)
        # for tagged doc
        wordslist = []
        tagslist = []
 
        # clean and tokenize document string
        raw = i.lower()
        raw = raw.replace('<br />', ' ')
   		# Pad punctuation with spaces on both sides

        tokens = tokenizer.tokenize(raw)
 
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
 
        # remove numbers
        number_tokens = [re.sub(r'[\d]', ' ', i) for i in stopped_tokens]
        number_tokens = ' '.join(number_tokens).split()
 
        # stem tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in number_tokens]
        # remove empty
        length_tokens = [i for i in stemmed_tokens if len(i) > 1]
        # add tokens to list
        texts.append(length_tokens)
        for l,k in enumerate(sentence_id):
            if index in range(0, sentence_id[0]):
                h = 0
            elif index in range(sentence_id[l-1], sentence_id[l]):
                h = l
        #h = sentence_id[h] document number of the given sentence
        if h in range(0,12500):
            td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(stemmed_tokens))).split(), ["train_pos"+"doc_id" + str(h) +"sent_id"+ str(index)])
        if h in range(12500, 25000):
            td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(stemmed_tokens))).split(), ["train_neg"+"doc_id" + str(h) +"sent_id"+ str(index)])
        if h in range(25000, 37500):
            td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(stemmed_tokens))).split(), ["test_pos"+"doc_id" + str(h) +"sent_id"+ str(index)])
        if h in range(37500, 50000):
            td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(stemmed_tokens))).split(), ["test_neg"+"doc_id" + str(h) +"sent_id"+ str(index)]) 	    
        # for later versions, you may want to use: td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(stemmed_tokens))).split(),[str(index)])
        taggeddoc.append(td)
 
    return taggeddoc

docData = get_doc(sentences)

from gensim.models import Doc2Vec
import gensim.models.doc2vec
from collections import OrderedDict
import multiprocessing
import time

cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

simple_models = [
    # PV-DBOW 
    Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, workers=cores),
    # PV-DM 
    Doc2Vec(dm=1, vector_size=300, window=10, negative=5, hs=0, min_count=2, workers=cores),
]

model = simple_models[1]
model.build_vocab(docData, update=False)

# start training
start = time.clock()
print("The training will begin now...")
for epoch in tqdm.trange(20):
    if epoch % 10 == 0:
        print ('Now training epoch %s'%epoch)
    model.train(docData, total_examples = model.corpus_count, epochs=model.epochs)
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay
end = time.clock()

print("The running time was:", end-start)
model.save(os.path.join('trained_models', 'sent_vectors_dm.s2v'))