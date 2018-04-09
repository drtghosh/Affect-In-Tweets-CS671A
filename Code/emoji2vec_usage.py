import gensim.models as gsm
from nltk.tokenize import word_tokenize
import re

e2v = gsm.KeyedVectors.load_word2vec_format('emoji2vec.bin', binary=True)
happy_vector = e2v['🍱']    # Produces an embedding vector of length 300

print(happy_vector)

tweet = 'I need a 🍱 sushi date🍙 @AnzalduaG 🍝an olive guarded date🧀 @lexiereid369 and a 👊🏼Rockys date🍕'
print(word_tokenize(tweet))
encodedTweet = tweet.encode('utf-8')
print(encodedTweet)
print(word_tokenize(encodedTweet))
print(encodedTweet[2])
"""
p = re.compile("\\\\U0001f[0-9][0-9A-Za-z][0-9A-Za-z]")
m = p.findall(encodedTweet)
"""