import pickle
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import conlltags2tree, tree2conlltags
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from itertools import product
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from gensim.models import Word2Vec
import nltk
from nltk.corpus import brown
import string
from collections import Counter

def make_feature_dicts(data, w2v_model_brown, NER=True, w2v=True, pos=True, token=True, caps=True, context=True):
   """
   builds features :
      when NER=True, use NER labels as features
      when w2v=True, use w2v vectors to build features
      when parseTree=True, use features from parse tree of the sentence
   """
   listOfDicts = []
   nerList = []
   for commentList in data:
       videoList = {}
       for tokenList in commentList:
           sentenceList = []
           sentenceDict = {}
           for tokenTupple in tokenList:
               featDict = {}
               if(w2v):
                try:
                  val = w2v_model_brown[tokenTupple[0]]
                  for i in range(len(val)):
                    featDict["w2v_"+str(i)] = val[i]
                except KeyError:
                  pass

               if(NER):
                   nerfeat ='ner='+ tokenTupple[2]
                   featDict[nerfeat] = 1
               if(token):
                   tokfeat = 'tok=' + tokenTupple[0].lower()
                   featDict[tokfeat] = 1
                                           
               if(caps):
                   if(tokenTupple[0][0].isupper()):
                       capfeat = 'is_caps'
                       featDict[capfeat] = 1
               if(pos):
                   posfeat = 'pos='+tokenTupple[1]
                   featDict[posfeat] = 1
            
               if(context and sentenceList):
                  prevDict = sentenceList[-1]
                  curDict = {'prev_' +k: v for k, v in prevDict.items() if not (k.startswith('prev_') or k.startswith('next_'))}
                  nextDict = {'next_' +k: v for k, v in featDict.items() if not (k.startswith('prev_') or k.startswith('next_'))}
                  featDict.update(curDict)
                  prevDict.update(nextDict)
                  sentenceList[-1] = prevDict
               sentenceList.append(featDict)
               sentenceDict.update(featDict)
           videoList.update(sentenceDict)
       listOfDicts.append(videoList)
   sentenceList.clear()
   #numpya = np.array(nerList)
   return listOfDicts

def tag_sentence(list_of_sentence_per_video):
   tagged_sentence_list = []
   for video_comment in list_of_sentence_per_video:
       video_tagged_comment_list = []
       for sentence in video_comment:
           ne_tree = ne_chunk(pos_tag(word_tokenize(sentence)))
           iob_tagged = tree2conlltags(ne_tree)
           video_tagged_comment_list.append(iob_tagged)
       tagged_sentence_list.append(video_tagged_comment_list)
   return tagged_sentence_list

def load_data(raw_data):
   """
   load the raw data from pickle file
   dumps data as below:
      List of sentences in the comments (irrespective of which video comment it is)
      List of sentences in the comments per video
   """
   l_s = []
   l_s_v = []
   Y = []
   puncts = string.punctuation
   #raw_data = pickle.load(open(fname, 'rb'))
   for videoId, details in raw_data.items():
      Y.append(details['relevance'])
      comments = details['comments']
      if comments is None:
         continue
      sentenceList = []
      for comment in comments:
         if comment is None:
            continue
         sentences = comment.split('.')
         if len(sentences) > 0:
            for i in range(len(sentences) - 1):
               sentence = sentences[i]
               sentence = sentence.replace("\n"," ")#remove newline characters in the comment
               for ch in puncts: # remove punctuations
                  sentence = sentence.replace(ch,"")
               l_s.append(sentence)
               sentenceList.append(sentence)
         else:
           comment = comment.replace("\n"," ")#remove newline characters in the comment
           for ch in puncts: # remove punctuations
              comment = comment.replace(ch,"")
           l_s.append(comment)
           sentenceList.append(comment)
      l_s_v.append(sentenceList)
   #print("l_s : ", l_s)
   print("Length of l_s_v : ", len(l_s_v))
   pickle.dump(l_s, open('listOfSentences.pkl', 'wb'))
   pickle.dump(l_s_v, open('listOfSentences_per_video.pkl', 'wb'))
   return Y

def calculate_f1(confusion_matrix):
   sum_r = confusion_matrix.sum(0)
   sum_c = confusion_matrix.sum(1)

   evaluation_matrix = dict()
   keys = sorted(confusion_matrix.keys())
   for key in keys:
    tp = confusion_matrix[key][key]
    if sum_r[key] == 0:
     p = 0
    else:
     p = tp/sum_r[key]
    if sum_c[key] == 0:
     r = 0
    else:
     r = tp/sum_c[key]
    if (p + r) == 0:
     f1 = 0
    else:
     f1 = (2 * p * r)/(p + r)
   return f1

def evaluate_combinations(X_train, Y_train, X_test, Y_test, model, w2v_vector):
   combination_results = dict()
   accuracy = []
   n_params = []
   caps = []
   pos = []
   NER = []
   context = []
   w2v = []
   F1 = []
   comb_list = list(product([True, False], repeat = 5))
   for i,each_comb in enumerate(comb_list):
    caps.append(each_comb[0])
    pos.append(each_comb[1])
    NER.append(each_comb[2])
    context.append(each_comb[3])
    w2v.append(each_comb[4])
    
    print("\ncaps : ", each_comb[0], " POS : ", each_comb[1], " NER : ", each_comb[2], " context : ", each_comb[3], " w2v : ", each_comb[4])    

    train_dicts = make_feature_dicts(X_train, w2v_model_wv, token=True, caps=each_comb[0], pos=each_comb[1], NER=each_comb[2], context=each_comb[3], w2v=each_comb[4])
    
    vec = DictVectorizer()
    X_train_v = vec.fit_transform(train_dicts)

    clf = 0
    if model == "SVM":
       clf = LinearSVC(C=0.1,random_state=123,class_weight="balanced",max_iter=100,fit_intercept=True)
    elif model == "NN":
       clf = MLPClassifier(hidden_layer_sizes=(5,10),solver='sgd',learning_rate = 'adaptive',activation='logistic', max_iter=50, random_state=42)
    elif model == "MNB":
       clf = GaussianNB()

    #clf = LogisticRegression()
    clf.fit(X_train_v.toarray(), Y_train)

    """
    #checking feature weights
    for i, cls in enumerate(clf.classes_):
      print("\nFeature weights for class : ",cls,"\n")
      df = pd.DataFrame(data= {"Features" : vec.feature_names_, "weights" : clf.coef_[i]})
      df = df.sort_values(axis=0,by='weights',ascending=False)
      print(df)
    """    

    test_dicts = make_feature_dicts(X_test, w2v_model_wv, token=True, caps=each_comb[0], pos=each_comb[1], NER=each_comb[2], context=each_comb[3], w2v=each_comb[4])
    X_test_v = vec.transform(test_dicts)
    Y_preds = clf.predict(X_test_v)

    #confusion_matrix = confusion(test_labels, preds)
    #class_labels = ["Relevant","Not Relevant","Deceptive"]
    class_labels = ["Relevant", "Not Relevant", "Deceptive"]
    c_m = confusion_matrix(Y_test, Y_preds, labels = class_labels)
    cm_df = pd.DataFrame(c_m, index=class_labels, columns=class_labels)
    print('confusion matrix:\n%s\n' % str(cm_df))

    #evaluation_matrix = evaluate(confusion_matrix)
    #f1_score = average_f1s(evaluation_matrix)
    acc = accuracy_score(Y_test, Y_preds)
    print("Accuracy : ", acc)
    n_params.append(clf.coef_.size)
    accuracy.append(acc)

    f1 = calculate_f1(cm_df)
    print("f1: ", f1)
    F1.append(f1)

   df = pd.DataFrame(data={'F1' : F1, 'Accuracy' : accuracy, 'n_params' : n_params, 'caps' : caps, 'pos' : pos, 'NER' : NER, 'context' : context, 'w2v' : w2v})
   #df = pd.DataFrame(data={'Accuracy' : accuracy, 'caps' : caps, 'pos' : pos, 'NER' : NER, 'context' : context, 'w2v' : w2v})
   df = df[['F1','Accuracy', 'n_params','caps','pos','NER','context','w2v']]
   #df = df[['Accuracy','caps','pos','NER','context','w2v']]
   df = df.sort_values(axis=0,by='F1',ascending=False)
   return df
 
if __name__ == "__main__":

   #load the labeled raw data 
   data = pickle.load(open("actualData.pkl","rb"))

   #create pickle file of features and get the labels for the videos
   Y = np.array(load_data(data))
   #print("Labels : ", Y)
   count = Counter(Y)
   print(count)

   list_of_sents = []
   list_of_sents_raw = pickle.load(open('listOfSentences.pkl','rb'))
   
   for line in list_of_sents_raw:
      list_of_sents.append(line.split(" "))

   #building word2vec model on brown corpus
   size = 20
   window = 3
   print("Word2Vec parameters - Vector size : ",size,"Window size : ",window)
   sentences_brown = brown.sents()
   w2v_model_brown = Word2Vec(sentences_brown, size=size, window=window, min_count=1)
   w2v_model_wv = w2v_model_brown.wv
   del w2v_model_brown

   """
   #load the labeled raw data 
   data = pickle.load(open("actualData.pkl","rb"))

   #create pickle file of features and get the labels for the videos
   Y = np.array(load_data(data))
   #print("Labels : ", Y)
   count = Counter(Y)
   print(count)
   
   """  

   #load the pickle file of features
   list_of_sentence_per_video = pickle.load(open('listOfSentences_per_video.pkl','rb'))
   
   #get the POS tags, NER tags of the words in the sentences
   tagged_sentence_list = tag_sentence(list_of_sentence_per_video)
   #pickle.dump(tagged_sentence_list, open('NERtaggedSentences.pkl', 'wb'))

   #create feature vector using POS tags, NER tags, w2v vector and token itself
   #feature_vectors = make_feature_dicts(tagged_sentence_list, w2v_model_wv, NER=False, w2v=True, pos=False)


   #vec = DictVectorizer()
   #X = vec.fit_transform(feature_vectors)
   #print(X)
   #print('training data shape: %s\n' % str(X.shape))

   """
   kfold = KFold(n_splits=2, shuffle=True, random_state=1234)
   preds = []
   truths = []
   for train, test in kfold.split(X):
    print('%d training instances and %d testing instances' %(len(train), len(test)))
    clf_svc = LinearSVC(C=0.1,random_state=123,class_weight="balanced",max_iter=100,fit_intercept=True)
    clf_svc.fit(X[train], Y[train])
    Y_pred = clf_svc.predict(X[test])
    preds.extend(Y_pred)
    truths.extend(Y[test])
    acc = accuracy_score(Y[test], Y_pred)
    print(acc)
   class_labels=["Relevant","Not Relevant","Deceptive"]
   c_m = confusion_matrix(truths, preds,labels=class_labels)
   cm_df = pd.DataFrame(c_m,index=class_labels,columns=class_labels)
   print(cm_df)
   """

   #clf_LinearSVC = LinearSVC(C=0.1,random_state=123,class_weight="balanced",max_iter=100,fit_intercept=True)

   X_train, X_test, Y_train, Y_test = train_test_split(tagged_sentence_list, Y, test_size = 0.33, random_state = 42)

   """
   train_dicts = make_feature_dicts(X_train, w2v_model_wv, token=True, caps=False, pos=True, NER=False, context=False, w2v=False)

   vec = DictVectorizer()
   X_train_v = vec.fit_transform(train_dicts)

   test_dicts = make_feature_dicts(X_test, w2v_model_wv, token=True, caps=False, pos=True, NER=False, context=False, w2v=False)
   X_test_v = vec.transform(test_dicts)

   clf = LinearSVC(C=0.1,random_state=42,class_weight="balanced",max_iter=100,fit_intercept=True)
   clf.fit(X_train_v, Y_train)

   Y_pred = clf.predict(X_test_v)
   
   class_labels = ["Not Deceptive", "Deceptive"]
   c_m = confusion_matrix(Y_test, Y_pred, labels = class_labels)
   cm_df = pd.DataFrame(c_m, index=class_labels, columns=class_labels)
   print('confusion matrix:\n%s\n' % str(cm_df))

   #evaluation_matrix = evaluate(confusion_matrix)
   #f1_score = average_f1s(evaluation_matrix)
   acc = accuracy_score(Y_test, Y_pred)
   print("Accuracy : ", acc)
   print(clf.coef_.size)

   f1 = calculate_f1(cm_df)
   print("f1: ", f1)

   #for i, feature in enumerate(vec.feature_names_):
   #print("\nFeature weights for class : ",cls,"\n")
   df = pd.DataFrame(data= {"Features" : vec.feature_names_, "weights" : clf.coef_})
   df = df.sort_values(axis=0,by='weights',ascending=False)
   print(df[:20])
   """

   """
   clf_LinearSVC.fit(X_train, Y_train)

   Y_pred = clf_LinearSVC.predict(X_test)

   acc = accuracy_score(Y_test, Y_pred)
   print(acc)
  
   class_labels = ["Relevant","Not Relevant","Deceptive","NA"]
   c_m = confusion_matrix(Y_test, Y_pred,labels = class_labels)
   cm_df = pd.DataFrame(c_m,index=class_labels,columns=class_labels)
   print(cm_df)
   
   for i, cls in enumerate(clf_LinearSVC.classes_):
      print("\nFeature weights for class : ",cls,"\n")
      df = pd.DataFrame(data= {"Features" : vec.feature_names_, "weights" : clf_LinearSVC.coef_[i]})
      df = df.sort_values(axis=0,by='weights',ascending=False)
      print(df[:50])
   """

   """
   kfold = KFold(n_splits=5, shuffle=True, random_state=1234)
   preds = []
   truths = []
   #y = np.array(labels)
   for train, test in kfold.split(X):
      gnb = LogisticRegression()
      #C=0.001,fit_intercept=True,max_iter=20,solver="lbfgs",class_weight = 'balanced')
      #MLPClassifier(hidden_layer_sizes=(5,10),solver='sgd',learning_rate = 'adaptive',activation='logistic')
      clf = gnb.fit(X[train], Y[train])
      #print(clf.class_count_)
      preds.extend(clf.predict(X[test]))
      truths.extend(Y[test])
   acc = accuracy_score(truths, preds)
   print('accuracy : %0.3f'%acc)
   cnf_matrix = confusion_matrix(truths,preds,labels=['Not Relevant','Deceptive','Relevant'])
   print(cnf_matrix)
   """
  
   #evaluate_combinations(np.colum_stack(X_train, Y_train), np.colum_stack(X_test, Y_test)) 
   combo_results = evaluate_combinations(X_train, Y_train, X_test, Y_test, "SVM", w2v_model_wv) #SVM or NN or MNB 
   print('combination results:\n%s' % str(combo_results))
