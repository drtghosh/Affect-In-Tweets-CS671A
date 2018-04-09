import re
import numpy as np
import gensim.models as gsm
import os
import stop_words
import gensim
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim.models.doc2vec import TaggedDocument
import tqdm

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

def get_doc(doc_list):
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')
    p_stemmer = PorterStemmer()
 
    taggeddoc = []
 
    texts = []
    for index,i in enumerate(doc_list):
        print("Currently processing document number:", index)
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
        if index in range(0,12500):
            td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(stemmed_tokens))).split(), ["train_pos"+str(index)])
        if index in range(12500, 25000):
            td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(stemmed_tokens))).split(), ["train_neg"+str(index)])
        if index in range(25000, 37500):
            td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(stemmed_tokens))).split(), ["test_pos"+str(index)])
        if index in range(37500, 50000):
            td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(stemmed_tokens))).split(), ["test_neg"+str(index)])
        # td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(stemmed_tokens))).split(), [str(index)])
        # for later versions, you may want to use: td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(stemmed_tokens))).split(),[str(index)])
        taggeddoc.append(td)
 
    return taggeddoc

docData = get_doc(data)

from gensim.models import Doc2Vec
import gensim.models.doc2vec
from collections import OrderedDict
import multiprocessing
import time

cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1, "This is to fasten the process."

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
    model.train(docData, total_examples = model.corpus_count, epochs=model.epochs)
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay
end = time.clock()

print("The running time was:", end-start)
model.save(os.path.join('trained_models','para_vectors_dm.d2v'))