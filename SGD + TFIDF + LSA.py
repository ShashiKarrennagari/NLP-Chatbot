
import nltk
from nltk.corpus import stopwords
import re
from string import *
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from nltk.stem.porter import PorterStemmer
from sklearn import metrics
import pickle,pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from scipy.spatial.distance import cosine
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,TfidfTransformer        
from sklearn.pipeline import Pipeline
from contractions import contractions
from nltk import word_tokenize,pos_tag
from sklearn import linear_model

stemmer = PorterStemmer()
wnl = WordNetLemmatizer()

#Reading the training data
data = pd.read_csv('HR_Questions.csv')

#Loding HC documents
all_hc_doc_dict=pickle.load(open('all_hc_doc_dict.pickle', 'rb'))

#combining all the sentences of the documents into a single list
comb_list = []
for x,val in all_hc_doc_dict.items():
    for x,val in val.items():
         temp=[(x+" ")+"XX"+" "+sent for sent in re.sub(r'\d+\.','',val).split('.') if sent!='']
         comb_list.append(temp)
flat_lis = [item for sublist in comb_list for item in sublist]


    
#Preproccesing the data 
data = data.iloc[:,0:1]
data = data.applymap(str)
corpus = []
for i in range(len(data)):
    review = re.sub('[^a-zA-Z0-9]', ' ', data['Question'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer() #stemming 
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] #removing stopwords
    review = ' '.join(review)
    #review = nltk.word_tokenize(review)
    corpus.append(review)
for i in range(len(flat_lis)):
    review = re.sub('[^a-zA-Z0-9]', ' ', flat_lis[i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    #review = nltk.word_tokenize(review)
    corpus.append(review)
    
    
#Building a sparse matrix with tf-idf weights
transformer = TfidfVectorizer(ngram_range=(1,3),lowercase=True)
tfidf = transformer.fit_transform(corpus)

#pipeline for performing LSA on tf-idf sparse matrix
#n_comp is the desired dimensionality of wordvector
n_comp = 300
svd = TruncatedSVD(n_comp)
lsa = make_pipeline(svd, Normalizer(copy=False))
X_lsa = lsa.fit_transform(tfidf)

#for normalizing the word vector
def normalize(vec):
    norm=np.linalg.norm(vec)
    if norm == 0: 
       return vec
    else:
        return vec/norm
        
#cleaning the query and doc before calculating cosine similarity
def cleaan(sent):
    sent = re.sub('[^a-zA-Z0-9+]', ' ',sent)
    sent = sent.lower()
    sent = sent.split()
    sent = [ps.stem(word) for word in sent if not word in set(stopwords.words('english'))]
    sent = ' '.join(sent)
    return sent
    
#cosine similarity between query and doc using tf-idf vectors    
def sen2_vec_similarity(quer,doc):
    return(cosine(transformer.transform([query]).toarray(),transformer.transform([doc]).toarray()))

#cosine similarity between query and doc using LSA vectors    
def sen2vec_similarity(query,doc):
    return (cosine(lsa.transform(transformer.transform([query])),
            lsa.transform(transformer.transform([doc]))))
    
#*********************** USING SGD FOR IDENTIFYING INTENT *************************************
    
exclude_stem_lem=[]
#data for classifying intent with SGD
full_data=pd.read_csv('Classification_data.csv')
extra_data = (pd.read_csv('new_test_data.csv')).iloc[:,0:2]
full_data = full_data.append(extra_data)

#including and excluding some specific stopwords
include_stop=set(['need','want','fractal','fractalites'])
exclude_stop = set(['not','none','us'])

#creating a final set of stopwords
stop = set(stopwords.words('english')).union(include_stop)-exclude_stop


#cleaning the query before classifying intent
def clean(doc,expand=True,stem=True,lemmatize=True,stopword=True,spell=False,lower=True,contraction=False):
    if lower:
        doc = doc.lower()
    doc = re.sub('[^a-zA-Z0-9 \n\.\']', '',doc)
    if contractions:
        doc = " ".join(contractions[word].split("/")[0].strip() if word in contractions.keys() else word for word in doc.split())
    if spell:
        doc=spell_checker(doc)  
    if lemmatize:
        lems = []
        for word, tag in pos_tag(word_tokenize(doc)):
            wntag = tag[0].lower()
            wntag = wntag if wntag in ['n'] else None
            if wntag is not None or word=='us':
                    lemma = word
            else:
                    lemma = wnl.lemmatize(word)
            lems.append(lemma)
        doc = " ".join(lems)
    if stem:
        doc = " ".join(stemmer.stem(word) if word!='us' else word for word in doc.split())
    if stopword:
        doc = " ".join(word for word in doc.split() if word not in stop)
    return doc.split(' ')

#pipeline for tf-idf weights and classifier model
ppl = Pipeline([
              ('vect', TfidfVectorizer(ngram_range=(1,4),lowercase=True,tokenizer=clean)),
              ('tfidf', TfidfTransformer()),
              ('clf',linear_model.SGDClassifier(loss='log',max_iter=2000,alpha=0.0001,n_jobs=-1))  
      ])
ppl.fit(full_data.Question,full_data.Intent)

# ********************TESTING THE ALGORITHM***********************

#Data for testing
queriees = pd.read_csv('100_test_data.csv',error_bad_lines=False)
quer = queriees['Query']
ans = queriees['ans']
ans = ans.values.T.tolist()

#count is the number of correct predictions out of 100 test cases
count = 0
for i in range(len(quer)):
    query = quer[i]
    topic = str(ppl.predict([query])) #intent prediction
    topic = topic.replace("[","")
    topic = topic.replace("]","")
    topic = topic.replace("'","")
    query = cleaan(query)
    dicts = {}
    doc = all_hc_doc_dict[topic] #obtaining the documents corresponding to predicted intent
    comb=[]
    for x,val in doc.items():
        temp=[(x+" ")+"XX"+" "+sent for sent in re.sub(r'\d+\.','',val).split('.') if sent!='']
        comb.append(temp)
    # creating a list of all sentences of the document
    flat_list = [item.replace(',','') for sublist in comb for item in sublist]
    
    # calculating cosine similarity between query and relevant documents
    for x in flat_list:
        g = cleaan(x)
        #average cosine distances from LSA and tf-idf vectotrs
        f = (sen2vec_similarity(query,g) + sen2_vec_similarity(query,g))/2
        dicts[f] = x
    #sorting the dictionary of cosine distance value and selecting least cosine distance
    sx = sorted(dicts.keys())
    a = sx[0]
    if dicts[a].strip() == ans[i].strip():
        count = count+1;
        #print(quer[i])
        print(a)
        #print(dicts[a])
    '''else:
        print(quer[i])
        print(i)
        print(topic)
        print(ans[i])
        print("-----------")
        print(a)
        print(dicts[a])'''
print(count)

#***********************ANSWER TO THE INPUT QUERY***************************************
#Enter your query
query = "How many leaves can I take"
topic = str(ppl.predict([query]))
topic = topic.replace("[","")
topic = topic.replace("]","")
topic = topic.replace("'","")
query = cleaan(query)
dicts = {}
doc = all_hc_doc_dict[topic] #obtaining the documents corresponding to predicted intent
comb=[]
for x,val in doc.items():
    temp=[(x+" ")+"XX"+" "+sent for sent in re.sub(r'\d+\.','',val).split('.') if sent!='']
    comb.append(temp)
# creating a list of all sentences of the document
flat_list = [item.replace(',','') for sublist in comb for item in sublist]
        
# calculating cosine similarity between query and relevant documents 
for x in flat_list:
    g = cleaan(x)
    f = (sen2vec_similarity(query,g) + sen2_vec_similarity(query,g))/2
    dicts[f] = x  #storing cosine distance values in a dictionary
    
#sorting the dictionary of cosine distance value and selecting least cosine distance
sx = sorted(dicts.keys())
a = sx[0]
print(dicts[a])














           
