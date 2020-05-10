#!/usr/bin/env python
# coding: utf-8

# In[1]:
from flask import Flask, redirect, url_for, request
from flask_cors import CORS, cross_origin
app = Flask(__name__)
CORS(app)

import pandas as pd
import numpy as np
import re


# In[2]:
@app.route('/success/<name>')
def success(name):
   return 'welcome %s' % name

def read_glove_vecs(file):
    with open(file, 'r',encoding="utf8") as f:
        words = set()
        word_to_vec_map = {}
        
        for line in f:
            line = line.strip().split()
            word = line[0]
            words.add(word)
            word_to_vec_map[word] = np.array(line[1:], dtype=np.float64)
            
    return words, word_to_vec_map


# In[3]:

print('dsdfsdf')
words, word_to_vec_map = read_glove_vecs("C:\\Users\\SANJANA\\Desktop\\projectFiles\\glove.6B.50d.txt")



# In[4]:


import math
def dot_product2(v1, v2):
    return sum(map(operator.mul, v1, v2))
def vector_cos5(v1, v2):
    prod = dot_product2(v1, v2)
    len1 = math.sqrt(dot_product2(v1, v1))
    len2 = math.sqrt(dot_product2(v2, v2))
    return prod / (len1 * len2)


# In[5]:


df = pd.read_excel("C:\\Users\\SANJANA\\Desktop\\projectFiles\\Disease_Symptoms.xlsx").drop('Count of Disease Occurrence', axis = 1).fillna(method = 'ffill')


# In[6]:


df.Symptom = df.Symptom.map(lambda x: re.sub('^.*_', '', x))
df.Disease = df.Disease.map(lambda x: re.sub('^.*_', '', x))


# In[7]:


df.Symptom = df.Symptom.map(lambda x: x.lower())
df.Disease = df.Disease.map(lambda x: x.lower())


# In[8]:


df.Symptom = df.Symptom.map(lambda x: re.sub(r'(.*)\/(.*)', r'\1 \2', x))
df.Disease = df.Disease.map(lambda x: re.sub(r'(.*)\/(.*)', r'\1 \2', x))


# In[9]:


df.Symptom = df.Symptom.map(lambda x: re.sub(r'(.*)\(.*\)(.*)', r'\1\2', x))
df.Disease = df.Disease.map(lambda x: re.sub(r'(.*)\(.*\)(.*)', r'\1\2', x))


# In[10]:


df.Symptom = df.Symptom.map(lambda x: re.sub('\'', '', x))
df.Disease = df.Disease.map(lambda x: re.sub('\'', '', x))
df.Disease = df.Disease.map(lambda x: re.sub('\\xa0', ' ', x))


# In[11]:


counts = {}
def remove(x):
    print(x)
    for i in x.split():
        if not i in word_to_vec_map.keys():
            counts[i] = counts.get(i, 0) + 1
            # print(x,counts[i])
df.Symptom.map(lambda x: remove(x))
df.Disease.map(lambda x: remove(x))


# In[12]:


unrepresented_words = pd.DataFrame()
unrepresented_words['Words'] = counts.keys()
unrepresented_words['No. of Occurences'] = counts.values()
unrepresented_words.to_csv('Unrepresented Words.csv')


# In[13]:


frame = pd.DataFrame(df.groupby(['Symptom', 'Disease']).size()).drop(0, axis = 1)

frame = frame.iloc[1:]


# In[14]:


frame = frame.reset_index().set_index('Symptom')


# In[15]:


counts = {}
for i in frame.index:
    counts[i] = counts.get(i, 0) + 1


# In[16]:


import operator
sym, ct = zip(*sorted(counts.items(), key = operator.itemgetter(1), reverse = True))
sym_count = pd.DataFrame()
sym_count['Symptom'] = sym
sym_count['Count'] =  ct


# In[17]:


[frame.drop(i, inplace = True) for i in frame.index if counts[i] < 2]


# In[18]:


diseaseLst = []
frame.Disease.map(lambda x: diseaseLst.append(x))
lst = diseaseLst


# In[19]:


frame.Disease.head(20)


# In[20]:


frame.index.unique()


# In[21]:


couples_and_labels = []

import random
# run through the symptoms
for i in frame.index.unique():
    # make a temporary list of the diseases associated with the symptom (actual context words)
    a = list(frame.Disease.loc[i].values)
    # loop through the context words
    for j in a:
        # randomly select a disease that isn't associated with the symptom, to set as a non-context word with label 0,
        # by using the XOR operator, that finds the uncommon elements in the 2 sets
        non_context = random.choice(list(set(lst) ^ set(a)))
        # add labels of 1 and 0 to context and non-context words repectively
        couples_and_labels.append((i, j, 1))
        couples_and_labels.append((i, non_context, 0))


# In[22]:


b = random.sample(couples_and_labels, len(couples_and_labels))
# Extract the symptoms, the diseases and the corresponding labels
symptom, disease, label = zip(*b)


# In[23]:


s1 = pd.Series(list(symptom))
s2 = pd.Series(list(disease))
dic = {}


# In[24]:


for i,j in enumerate(s1.append(s2).unique()):
    dic[j] = i
    # print(j)
# Now all the symptoms are represented by a number in the arrays 'symptoms', and 'diseases'
symptoms = np.array(s1.map(dic), dtype = 'int32')
diseases = np.array(s2.map(dic), dtype = 'int32')


# In[25]:


# Make the labels too into an array
labels = np.array(label, dtype = 'int32')

lst = []

# size of the vocabulary ,ie, no. of unique words in corpus
vocab_size = len(dic)

# dimension of word embeddings
vector_dim = 50

# create an array of zeros of shape (vocab_size, vector_dim) to store the new embedding matrix (word vector representations)
embedding_matrix = np.zeros((len(dic), 50))


# In[26]:


for word, index in dic.items():
    # split each symptom/disease into a list of constituent words
    for i in word.split():
        if i in word_to_vec_map.keys(): #only if in glove!
            lst.append(word_to_vec_map[i]) # add the embeddings of each word in symptoms and diseases to list 'lst'
    
    # make an array out of the list    
    arr = np.array(lst) 
    
    # sum the embeddings of all words in the sentence, to get an embedding of the entire sentence
    # if in the entire sentence, word embeddings weren't available in GloVe vectors, make that sentence into a
    # zero array of shape (50,), just as a precaution, as we have already removed such words
    arrsum = arr.sum(axis = 0)     
    
    # normalize the values
    arrsum = arrsum/np.sqrt((arrsum**2).sum()) 
    
    # add the embedding to the corresponding word index
    embedding_matrix[index,:] = arrsum


# In[27]:


import tensorflow as tf
from keras.preprocessing import sequence
from keras.layers import Dot, Reshape, Dense, Input, Embedding
from keras.models import Model


# In[28]:


input_target = Input((1,))
input_context = Input((1,))


# In[29]:


embedding = Embedding(
    input_dim = vocab_size,
    output_dim = vector_dim,
    input_length = 1,
    name='embedding',
    trainable = True)


# In[30]:


embedding.build((None,))
embedding.set_weights([embedding_matrix])


# In[31]:


context = embedding(input_context)
context = Reshape((vector_dim, 1))(context)

target = embedding(input_target)
target = Reshape((vector_dim, 1))(target)


# In[32]:


dot = Dot(axes = 1)([context, target])
dot = Reshape((1,))(dot)
# pass it through a 'sigmoid' activation neuron; this is then comapared with the value in 'label' generated from the skipgram
out = Dense(1, activation = 'sigmoid')(dot)

# create model instance
model = Model(input = [input_context, input_target], output = out)
model.compile(loss = 'binary_crossentropy', optimizer = 'adam')

# fit the model, default batch_size of 32
# running for 25 epochs seems to generate good enough results, although running for more iterations may improve performance further
model.fit(x = [symptoms, diseases], y = labels, epochs = 25,)


# In[33]:


new_vecs = model.layers[2].get_weights()[0]


# In[34]:


similarity_score = 0.6


# In[35]:


print(new_vecs[37])


# In[36]:


d = pd.read_csv("C:\\Users\\SANJANA\\Desktop\\projectFiles\\Dictionary.csv")
dic = {}
for i in d.index:
    dic[d.Key.loc[i]] = d.Values.loc[i]


# In[41]:


symp = input('Enter symptom for which similar symptoms are to be found: ')


# In[43]:


print ('\nThe similar symptoms are: ')
@app.route('/login',methods = ['POST', 'GET'])
def login():
     symp=request.get_json()["symptom"]
     r=[]
     
     
# loop through the symptoms in the data set and find the symptoms with cosine similarity greater than 'similarity_score'
     for i in set(symptom):
     
        if  symp in dic and i in dic:
          XA = new_vecs[dic[i]]
          XB = new_vecs[dic[symp]]
          similarity = vector_cos5(XA, XB)
          if (similarity) > 0.5:
          # remove the same symptom from the list of outputs
            if i != symp:
              r.append(i)
     return {"time": r}
if __name__ == '__main__':
   app.run(debug = True)

