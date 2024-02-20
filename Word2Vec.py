
import numpy as np
import matplotlib.pyplot as plt
import pickle,pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,TfidfTransformer        
from sklearn.pipeline import Pipeline
from contractions import contractions
from nltk import word_tokenize,pos_tag
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine

ps = PorterStemmer()
wnl = WordNetLemmatizer()
stemmer = PorterStemmer()

nltk.download('stopwords')
nltk.download("punkt")
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


#read train data
dataset = pd.read_csv('HR_Questions.csv')
data = dataset.iloc[:,0:1]

#load HC documents
all_hc_doc_dict=pickle.load(open('all_hc_doc_dict.pickle', 'rb'))

#combining all HC documents into a single list
comb = []
for x,val in all_hc_doc_dict.items():
    for x,val in val.items():
         temp=[(x+" ")+"XX"+" "+sent for sent in re.sub(r'\d+\.','',val).split('.') if sent!='']
         comb.append(temp)
flat_lis = [item for sublist in comb for item in sublist]

#preprocessing the data
corpus = []
for i in range(len(data)):
    review = re.sub('[^a-zA-Z]', ' ', data['Question'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    review = nltk.word_tokenize(review)
    corpus.append(review)
for i in range(len(flat_lis)):
    review = re.sub('[^a-zA-Z]', ' ', flat_lis[i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    review = nltk.word_tokenize(review)
    corpus.append(review)
    

   
#parameters for word2vec
   
size = 300 # dimension of word vector
window_size = 2 #size of the search for context
epochs = 1000 #number of times the algorithm runs on a training datset
min_count = 2 #Excludes words that appear less than min count
workers = 4

model = Word2Vec(corpus, sg=1,window=window_size,size=size,
                 min_count=min_count,workers=workers,iter=epochs,sample=0.01)
                 
model.train(data['Question'], total_examples=len(data['Question']), epochs=10)

#function to clean query and docs before calculating cosine distance
def cleaan(sent):
    sent = re.sub('[^a-zA-Z]', ' ',sent)
    sent = sent.lower()
    sent = sent.split()
    sent = [ps.stem(word) for word in sent if not word in set(stopwords.words('english'))]
    sent = ' '.join(sent)
    return sent

#For obtaining sentence vectors from corresponding word vectors
def sent_vectorizer(sent):
    sent_vec = np.zeros(size)
    numw = 0
    sent = cleaan(sent)
    for w in sent.split():
        #print (w)
        try:
            sent_vec = np.add(sent_vec,model.wv[w])
            numw+=1
        except:
            pass
    return sent_vec / np.sqrt(sent_vec.dot(sent_vec))

#cosine distance between query and document        
def sen2vec_similarity(query,faq):
    return (cosine(sent_vectorizer(query),sent_vectorizer(faq)))
    
#*********************** USING SGD FOR IDENTIFYING TOPIC *************************************
    
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
from sklearn import linear_model

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
        f = sen2vec_similarity(query,g)
        dicts[f] = x
    #sorting the dictionary of cosine distance value and selecting least cosine distance
    sx = sorted(dicts.keys())
    a = sx[0]
    if dicts[a].strip() == ans[i].strip():
        count = count+1;
        '''print(quer[i])
        print(a)
        print(dicts[a])
        print("*************")'''
    #for checking the predicted and actual answer
    '''else:
        print(quer[i])
        print(i)
        print(topic)
        print(ans[i])
        print("-----------")
        print(a)
        print(dicts[a])
        print("##########################################")'''
print(count)
