#!/usr/bin/env python
# coding: utf-8

# In[225]:

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



# In[226]:


import csv
from collections import defaultdict

disease_list = []

def return_list(disease):
    disease_list = []
    match = disease.replace('^','_').split('_')
    ctr = 1
    for group in match:
        if ctr%2==0:
            disease_list.append(group)
        ctr = ctr + 1

    return disease_list

with open("C:\\Users\\SANJANA\\Desktop\\projectFiles\\dataset_uncleaned.csv") as csvfile:
    reader = csv.reader(csvfile)
    disease=""
    weight = 0
    disease_list = []
    dict_wt = {}
    dict_=defaultdict(list)
    for row in reader:

        if row[0]!="\xc2\xa0" and row[0]!="":
            disease = row[0]
            disease_list = return_list(disease)
            weight = row[1]

        if row[2]!="\xc2\xa0" and row[2]!="":
            symptom_list = return_list(row[2])

            for d in disease_list:
                for s in symptom_list:
                    dict_[d].append(s)
                dict_wt[d] = weight


# In[227]:


with open("dataset_clean.csv","w") as csvfile:
    writer = csv.writer(csvfile)
    for key,values in dict_.items():
        for v in values:
            #key = str.encode(key)
            key = str.encode(key).decode('utf-8')
            #.strip()
            #v = v.encode('utf-8').strip()
            #v = str.encode(v)
            writer.writerow([key,v,dict_wt[key]])


# In[228]:


columns = ['Source','Target','Weight']


# In[229]:


data = pd.read_csv("dataset_clean.csv",names=columns, encoding ="ISO-8859-1")


# In[230]:


data.head()


# In[231]:


data.to_csv("dataset_clean.csv",index=False)


# In[232]:


slist = []
dlist = []
with open("nodetable.csv","w") as csvfile:
    writer = csv.writer(csvfile)

    for key,values in dict_.items():
        for v in values:
            if v not in slist:
                writer.writerow([v,v,"symptom"])
                slist.append(v)
        if key not in dlist:
            writer.writerow([key,key,"disease"])
            dlist.append(key)


# In[233]:


nt_columns = ['Id','Label','Attribute']


# In[234]:


nt_data = pd.read_csv("nodetable.csv",names=nt_columns, encoding ="ISO-8859-1",)


# In[235]:


nt_data.head()


# In[236]:


nt_data.to_csv("nodetable.csv",index=False)


# In[237]:


data = pd.read_csv("dataset_clean.csv", encoding ="ISO-8859-1")


# In[238]:


data.head()


# In[239]:


len(data['Source'].unique())


# In[240]:


len(data['Target'].unique())


# In[241]:


df = pd.DataFrame(data)


# In[242]:


df_1 = pd.get_dummies(df.Target)


# In[243]:


df_1.head()


# In[244]:


df.head()


# In[245]:


df_s = df['Source']


# In[246]:


df_pivoted = pd.concat([df_s,df_1], axis=1)


# In[247]:


df_pivoted.drop_duplicates(keep='first',inplace=True)


# In[248]:


df_pivoted[:5]


# In[249]:


len(df_pivoted)


# In[250]:


cols = df_pivoted.columns


# In[251]:


cols = cols[1:]


# In[252]:


df_pivoted = df_pivoted.groupby('Source').sum()
df_pivoted = df_pivoted.reset_index()
df_pivoted[:5]


# In[253]:


len(df_pivoted)


# In[254]:


df_pivoted.to_csv("df_pivoted.csv")


# In[255]:


x = df_pivoted[cols]
y = df_pivoted['Source']


# In[256]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


# In[257]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# In[258]:


mnb = MultinomialNB()
mnb = mnb.fit(x_train, y_train)


# In[259]:


mnb.score(x_test, y_test)


# In[260]:


mnb_tot = MultinomialNB()
mnb_tot = mnb_tot.fit(x, y)


# In[261]:


mnb_tot.score(x, y)


# In[262]:


print(x)


# In[263]:


disease_pred = mnb_tot.predict(x)


# In[264]:


disease_real = y.values


# In[265]:


for i in range(0, len(disease_real)):
    if disease_pred[i]!=disease_real[i]:
        print ('Pred: {0} Actual:{1}'.format(disease_pred[i], disease_real[i]))


# These are the predicted versus actual diseases that our classifier misclassifies.

# In[266]:


l1=[ "Heberden's node", "Murphy's sign", "Stahli's line", 'abdomen acute', 'abdominal bloating', 'abdominal tenderness', 'abnormal sensation', 'abnormally hard consistency', 'abortion', 'abscess bacterial', 'absences finding', 'achalasia', 'ache', 'adverse effect', 'adverse reaction', 'agitation', 'air fluid level', 'alcohol binge episode', 'alcoholic withdrawal symptoms', 'ambidexterity', 'angina pectoris', 'anorexia', 'anosmia', 'aphagia', 'apyrexial', 'arthralgia', 'ascites', 'asterixis', 'asthenia', 'asymptomatic', 'ataxia', 'atypia', 'aura', 'awakening early', 'barking cough', 'bedridden', 'behavior hyperactive', 'behavior showing increased motor activity', 'blackout', 'blanch', 'bleeding of vagina', 'bowel sounds decreased', 'bradycardia', 'bradykinesia', 'breakthrough pain', 'breath sounds decreased', 'breath-holding spell', 'breech presentation', 'bruit', 'burning sensation', 'cachexia', 'cardiomegaly', 'cardiovascular event', 'cardiovascular finding', 'catatonia', 'catching breath', 'charleyhorse', 'chest discomfort', 'chest tightness', 'chill', 'choke', 'cicatrisation', 'clammy skin', 'claudication', 'clonus', 'clumsiness', 'colic abdominal', 'consciousness clear', 'constipation', 'coordination abnormal', 'cough', 'cushingoid facies', 'cushingoid habitus', 'cyanosis', 'cystic lesion', 'debilitation', 'decompensation', 'decreased body weight', 'decreased stool caliber', 'decreased translucency', 'diarrhea', 'difficulty', 'difficulty passing urine', 'disequilibrium', 'distended abdomen', 'distress respiratory', 'disturbed family', 'dizziness', 'dizzy spells', 'drool', 'drowsiness', 'dullness', 'dysarthria', 'dysdiadochokinesia', 'dysesthesia', 'dyspareunia', 'dyspnea', 'dyspnea on exertion', 'dysuria', 'ecchymosis', 'egophony', 'elation', 'emphysematous change', 'energy increased', 'enuresis', 'erythema', 'estrogen use', 'excruciating pain', 'exhaustion', 'extrapyramidal sign', 'extreme exhaustion', 'facial paresis', 'fall', 'fatigability', 'fatigue', 'fear of falling', 'fecaluria', 'feces in rectum', 'feeling hopeless', 'feeling strange', 'feeling suicidal', 'feels hot/feverish', 'fever', 'flare', 'flatulence', 'floppy', 'flushing', 'focal seizures', 'food intolerance', 'formication', 'frail', 'fremitus', 'frothy sputum', 'gag', 'gasping for breath', 'general discomfort', 'general unsteadiness', 'giddy mood', 'gravida 0', 'gravida 10', 'green sputum', 'groggy', 'guaiac positive', 'gurgle', 'hacking cough', 'haemoptysis', 'haemorrhage', 'hallucinations auditory', 'hallucinations visual', 'has religious belief', 'headache', 'heartburn', 'heavy feeling', 'heavy legs', 'hematochezia', 'hematocrit decreased', 'hematuria', 'heme positive', 'hemianopsia homonymous', 'hemiplegia', 'hemodynamically stable', 'hepatomegaly', 'hepatosplenomegaly', 'hirsutism', 'history of - blackout', 'hoard', 'hoarseness', 'homelessness', 'homicidal thoughts', 'hot flush', 'hunger', 'hydropneumothorax', 'hyperacusis', 'hypercapnia', 'hyperemesis', 'hyperhidrosis disorder', 'hyperkalemia', 'hypersomnia', 'hypersomnolence', 'hypertonicity', 'hyperventilation', 'hypesthesia', 'hypoalbuminemia', 'hypocalcemia result', 'hypokalemia', 'hypokinesia', 'hypometabolism', 'hyponatremia', 'hypoproteinemia', 'hypotension', 'hypothermia, natural', 'hypotonic', 'hypoxemia', 'immobile', 'impaired cognition', 'inappropriate affect', 'incoherent', 'indifferent mood', 'intermenstrual heavy bleeding', 'intoxication', 'irritable mood', 'jugular venous distention', 'labored breathing', 'lameness', 'large-for-dates fetus', 'left atrial hypertrophy', 'lesion', 'lethargy', 'lightheadedness', 'lip smacking', 'loose associations', 'low back pain', 'lung nodule', 'macerated skin', 'macule', 'malaise', 'mass in breast', 'mass of body structure', 'mediastinal shift', 'mental status changes', 'metastatic lesion', 'milky', 'moan', 'monoclonal', 'monocytosis', 'mood depressed', 'moody', 'motor retardation', 'muscle hypotonia', 'muscle twitch', 'myalgia', 'mydriasis', 'myoclonus', 'nasal discharge present', 'nasal flaring', 'nausea', 'nausea and vomiting', 'neck stiffness', 'neologism', 'nervousness', 'night sweat', 'nightmare', 'no known drug allergies', 'no status change', 'noisy respiration', 'non-productive cough', 'nonsmoker', 'numbness', 'numbness of hand', 'oliguria', 'orthopnea', 'orthostasis', 'out of breath', 'overweight', 'pain', 'pain abdominal', 'pain back', 'pain chest', 'pain foot', 'pain in lower limb', 'pain neck', 'painful swallowing', 'pallor', 'palpitation', 'panic', 'pansystolic murmur', 'para 1', 'para 2', 'paralyse', 'paraparesis', 'paresis', 'paresthesia', 'passed stones', 'patient non compliance', 'pericardial friction rub', 'phonophobia', 'photophobia', 'photopsia', 'pin-point pupils', 'pleuritic pain', 'pneumatouria', 'polydypsia', 'polymyalgia', 'polyuria', 'poor dentition', 'poor feeding', 'posterior rhinorrhea', 'posturing', 'presence of q wave', 'pressure chest', 'previous pregnancies 2', 'primigravida', 'prodrome', 'productive cough', 'projectile vomiting', 'prostate tender', 'prostatism', 'proteinemia', 'pruritus', 'pulse absent', 'pulsus paradoxus', 'pustule', 'qt interval prolonged', 'r wave feature', 'rale', 'rambling speech', 'rapid shallow breathing', 'red blotches', 'redness', 'regurgitates after swallowing', 'renal angle tenderness', 'rest pain', 'retch', 'retropulsion', 'rhd positive', 'rhonchus', 'rigor - temperature-associated observation', 'rolling of eyes', 'room spinning', 'satiety early', 'scar tissue', 'sciatica', 'scleral icterus', 'scratch marks', 'sedentary', 'seizure', 'sensory discomfort', 'shooting pain', 'shortness of breath', 'side pain', 'sinus rhythm', 'sleeplessness', 'sleepy', 'slowing of urinary stream', 'sneeze', 'sniffle', 'snore', 'snuffle', 'soft tissue swelling', 'sore to touch', 'spasm', 'speech slurred', 'splenomegaly', 'spontaneous rupture of membranes', 'sputum purulent', 'st segment depression', 'st segment elevation', 'stiffness', 'stinging sensation', 'stool color yellow', 'stridor', 'stuffy nose', 'stupor', 'suicidal', 'superimposition', 'sweat', 'sweating increased', 'swelling', 'symptom aggravating factors', 'syncope', 'systolic ejection murmur', 'systolic murmur', 't wave inverted', 'tachypnea', 'tenesmus', 'terrify', 'thicken', 'throat sore', 'throbbing sensation quality', 'tinnitus', 'tired', 'titubation', 'todd paralysis', 'tonic seizures', 'transaminitis', 'transsexual', 'tremor', 'tremor resting', 'tumor cell invasion', 'unable to concentrate', 'unconscious state', 'uncoordination', 'underweight', 'unhappy', 'unresponsiveness', 'unsteady gait', 'unwell', 'urge incontinence', 'urgency of micturition', 'urinary hesitation', 'urinoma', 'verbal auditory hallucinations', 'verbally abusive behavior', 'vertigo', 'vision blurred', 'vomiting', 'weepiness', 'weight gain', 'welt', 'wheelchair bound', 'wheezing', 'withdraw', 'worry', 'yellow sputum']
l2 = []
for x in range(0, len(l1)):
    l2.append(0)
print(l2)    


# In[267]:


from sklearn.tree import DecisionTreeClassifier, export_graphviz


# In[268]:


X = df_pivoted[cols]
Y = df_pivoted['Source']


# In[269]:


#print(X)


# In[270]:


def DecisionTree(inputsymptoms):
    print ("DecisionTree")
    dt = DecisionTreeClassifier()
    clf_dt=dt.fit(X,Y)
    print ("Acurracy: ", clf_dt.score(X,Y))
    for k in range(0, len(l1)):
        for z in inputsymptoms:
            if(z == l1[k]):
                l2[k] = 1

    inputtest = [l2]
    predict = clf_dt.predict(inputtest)
    predicted = predict[0]
    status = 'no'
    for a in range(0, len(disease_real)):
        if(predicted == disease_real[a]):
            status = 'yes'
            break

    if (status == 'yes'):
        print("Disease : " + disease_real[a])
    else:
        print("Not Found")


# In[271]:


X1 = df_pivoted[cols]
Y1 = df_pivoted['Source']
from sklearn.ensemble import RandomForestClassifier


# In[272]:

@app.route('/RandomForest',methods = ['POST', 'GET'])
def RandomForest():
    print ("Randomforest")
    inputsymptoms=request.get_json()["symptom"]
    dt1 =  RandomForestClassifier()
    clf_dt1=dt1.fit(X,Y)
    print ("Acurracy: ", clf_dt1.score(X1,Y1))
    for k in range(0, len(l1)):
        for z in inputsymptoms:
            if(z == l1[k]):
                l2[k] = 1

    inputtest = [l2]
    predict = clf_dt1.predict(inputtest)
    predicted = predict[0]
    status = 'no'
    for a in range(0, len(disease_real)):
        if(predicted == disease_real[a]):
            status = 'yes'
            break

    if (status == 'yes'):
        print("Disease : " + disease_real[a])
        return {"time":disease_real[a]}
    else:
        print("Not Found")


# In[273]:


X2 = df_pivoted[cols]
Y2 = df_pivoted['Source']
from sklearn.naive_bayes import MultinomialNB


# In[274]:


def NaiveBayes(inputsymptoms):
    print ("Multinomial NB")
    dt2 =  MultinomialNB()
    clf_dt2=dt2.fit(X,Y)
    print ("Acurracy: ", clf_dt2.score(X2,Y2))
    for k in range(0, len(l1)):
        for z in inputsymptoms:
            if(z == l1[k]):
                l2[k] = 1

    inputtest = [l2]
    predict = clf_dt2.predict(inputtest)
    predicted = predict[0]
    status = 'no'
    for a in range(0, len(disease_real)):
        if(predicted == disease_real[a]):
            status = 'yes'
            break

    if (status == 'yes'):
        print("Disease : " + disease_real[a])
    else:
        print("Not Found")


# In[275]:


lst = [] 
  
# number of elemetns as input 
n = int(input("Enter number of elements : ")) 
  
# iterating till the range 
# for i in range(0, n): 
#     ele = input() 
  
#     lst.append(ele)
# DecisionTree(lst)
# RandomForest(lst)
# NaiveBayes(lst)

if __name__ == '__main__':
   app.run(debug = True)
# <hr>

# In[ ]:




