# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 02:20:30 2018

@author: Debabrata
"""

#import os
import sys
import pandas as pd
import numpy as np
from nltk.tokenize import TweetTokenizer
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from emoint.featurizers.emoint_featurizer import EmoIntFeaturizer
#from sklearn.preprocessing import StandardScaler, RobustScaler
from emoint.ensembles.blending import blend
from sklearn.linear_model import Ridge

orig_stdout = sys.stdout
f = open('Ensemble_results.txt', 'w')
sys.stdout = f

def get_xy(path, tokenizer, featurizer):
    df = pd.read_csv(path, header=None, sep='\t')
    tweets = df[1]
    intensities = df[3]
    X = []
    for t in tweets:
        tokens = tokenizer.tokenize(t)
        features = featurizer.featurize(tokens)
        X.append(features)
    X, y = np.array(X), np.array(intensities)
    print("Shapes X: {}, y: {}".format(X.shape, y.shape))
    return X, y

def metrics(y_pred, y, print_metrics=False):
    p1 = pearsonr(y_pred, y)[0]
    s1 = spearmanr(y_pred, y)[0]
    ind = np.where(y >= 0.5)
    ydt = np.take(y_pred, ind).reshape(-1)
    ydpt = np.take(y, ind).reshape(-1)
    p2 = pearsonr(ydt, ydpt)[0]
    s2 = spearmanr(ydt, ydpt)[0]
    if print_metrics:
        print("Validation Pearsonr: {}".format(p1))
        print("Validation Spearmanr: {}".format(s1))
        print("Validation Pearsonr >= 0.5: {}".format(p2))
        print("Validation Spearmanr >= 0.5: {}".format(s2))
    return np.array((p1, s1, p2, s2))

def train(train_path, dev_path, featurizer, tokenizer):
    X_train, y_train = get_xy(train_path, tokenizer, featurizer)
    X_dev, y_dev = get_xy(dev_path, tokenizer, featurizer)
    regr = GradientBoostingRegressor()
    regr.fit(X_train, y_train)
    y_dev_pred = regr.predict(X_dev)
    metrics(y_dev_pred, y_dev, True)
    return regr, featurizer.features

featurizer = EmoIntFeaturizer()
tokenizer = TweetTokenizer()

print("Data for Emotion: Anger")
anger_regr, anger_features = train('B:/CS671-NLP/Project/resources/emoint/anger-ratings-0to1.train.txt',
    'B:/CS671-NLP/Project/resources/emoint/anger-ratings-0to1.dev.gold.txt', featurizer, tokenizer)

print("Data for Emotion: Fear")
fear_regr, fear_features = train('B:/CS671-NLP/Project/resources/emoint/fear-ratings-0to1.train.txt',
    'B:/CS671-NLP/Project/resources/emoint/fear-ratings-0to1.dev.gold.txt', featurizer, tokenizer)

print("Data for Emotion: Joy")
joy_regr, joy_features = train('B:/CS671-NLP/Project/resources/emoint/joy-ratings-0to1.train.txt',
    'B:/CS671-NLP/Project/resources/emoint/joy-ratings-0to1.dev.gold.txt', featurizer, tokenizer)

print("Data for Emotion: Sadness")
sadness_regr, sadness_features = train('B:/CS671-NLP/Project/resources/emoint/sadness-ratings-0to1.train.txt',
    'B:/CS671-NLP/Project/resources/emoint/sadness-ratings-0to1.dev.gold.txt', featurizer, tokenizer)

def blend_train(train_path, dev_path, featurizer, tokenizer, clfs):
    featurizer = EmoIntFeaturizer()
    tokenizer = TweetTokenizer()
    X_train, y_train = get_xy(train_path, tokenizer, featurizer)
    X_dev, y_dev = get_xy(dev_path, tokenizer, featurizer)
    y_dev_pred = blend(X_train, y_train, X_dev, clfs, regr=True, blend_clf=Ridge())
    metrics(y_dev_pred, y_dev, True)
    
models = [RandomForestRegressor(), ExtraTreesRegressor(), BaggingRegressor(), GradientBoostingRegressor()]

print("Data for Emotion: Anger")
blend_train('B:/CS671-NLP/Project/resources/emoint/anger-ratings-0to1.train.txt',
    'B:/CS671-NLP/Project/resources/emoint/anger-ratings-0to1.dev.gold.txt', featurizer, tokenizer, models)

print("Data for Emotion: Fear")
blend_train('B:/CS671-NLP/Project/resources/emoint/fear-ratings-0to1.train.txt',
    'B:/CS671-NLP/Project/resources/emoint/fear-ratings-0to1.dev.gold.txt', featurizer, tokenizer, models)

print("Data for Emotion: Joy")
blend_train('B:/CS671-NLP/Project/resources/emoint/joy-ratings-0to1.train.txt',
    'B:/CS671-NLP/Project/resources/emoint/joy-ratings-0to1.dev.gold.txt', featurizer, tokenizer, models)

print("Data for Emotion: Sadness")
blend_train('B:/CS671-NLP/Project/resources/emoint/sadness-ratings-0to1.train.txt',
    'B:/CS671-NLP/Project/resources/emoint/sadness-ratings-0to1.dev.gold.txt', featurizer, tokenizer, models)

sys.stdout = orig_stdout
f.close()
