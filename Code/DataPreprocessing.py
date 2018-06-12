import sklearn
import pickle
from nltk.corpus import stopwords
import string
import numpy as np
import pandas as pd
import random
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score,f1_score,confusion_matrix

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

class youtube_data:
 def __int__():
  #constructor
  #self.raw_data = data
  pass
 def load_data(self,data,tokenize):
  #take rawdata and  
  stop_words = set(stopwords.words('english'))
  punct = string.punctuation
  X = []
  Y = []
  for videoId, details in data.items():
   if tokenize:
    preprocessed = []
   else:
    preprocessed = ""
   comments = details['comments']
   for comment in comments:
    if comment is None:
     continue
    comment = comment.replace("\n"," ")
    for ch in punct: # remove punctuations
     comment = comment.replace(ch,"")
    if tokenize:#tokenize the sentence to words
     words = comment.split(" ") #split comments into words
     for word in words:
      preprocessed.append(word)
    else: #not tokenizing comments to words
     preprocessed = preprocessed +" "+ comment
   X.append(preprocessed)
   Y.append(details['relevance'])
  print("Length of X : ", len(X))
  print("Length of Y : ", len(Y))
  return X,Y
 def predict_random(self,X):
  #predict one of the classes randomly
  r = len(X)
  Y = []
  for i in range(r):
   j = random.randint(0,2)
   if j==0:
    Y.append("Relevant")
   elif j==1:
    Y.append("Not Relevant")
   elif j==2:
    Y.append("Deceptive")
  return Y 
 def predict_dominant_class(self,X):
  #find the dominant class and predict it always
  r = len(X)
  Y = []
  for i in range(r):
   Y.append('Not Relevant')
  return Y

if __name__== '__main__':
    fname = 'actualData.pkl'
    raw_data = pickle.load(open(fname, 'rb'))
    yd = youtube_data()
    X, y = yd.load_data(raw_data,True)
    Y_rand = yd.predict_random(X)
    Y_dominant = yd.predict_dominant_class(X)
    pred_rand_accuracy = accuracy_score(y, Y_rand)
    print('pred_rand_accuracy : %0.3f'%pred_rand_accuracy)
    pred_dominant_class_accuracy = accuracy_score(y, Y_dominant)
    print('pred_dominant_class_accuracy : %0.3f'%pred_dominant_class_accuracy)

    #Fitting Multinomial NB
    nbX,nbY = yd.load_data(raw_data,False)

    vectorizer = CountVectorizer(min_df=1)
    nbX_word_count = vectorizer.fit_transform(nbX)

    vectorizer = TfidfVectorizer(min_df=1)
    nbX_tfidf = vectorizer.fit_transform(nbX)

    metrics = ["accuracy","precision_weighted","f1_weighted"]
    alphas = [0.1, 0.5, 1, 10]
    
    for metric in metrics:
     for alpha in alphas:
      nb_clf_word_count = MultinomialNB(alpha=alpha, fit_prior = True)
      nb_clf_word_count.fit(nbX_word_count, nbY)
      score = cross_val_score(nb_clf_word_count,nbX_word_count,nbY, scoring=metric,cv=3)
      print("{0} of Multinomial Navie Bayes with fit prior as True and ALPHA = {1} when using word counts : {2} ".format(metric.upper(),alpha,np.mean(score)))
	  
      nb_clf_tfidf = MultinomialNB(alpha=alpha)
      nb_clf_tfidf.fit(nbX_tfidf, nbY)
      score = cross_val_score(nb_clf_tfidf,nbX_tfidf,nbY, scoring=metric,cv=3)
      print("{0} of Multinomial Navie Bayes with fit prior as False and ALPHA = {1} when using tfidf : {2}\n".format(metric.upper(),alpha,np.mean(score)))


