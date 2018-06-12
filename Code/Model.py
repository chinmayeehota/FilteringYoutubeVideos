import pickle
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import conlltags2tree, tree2conlltags
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC,LinearSVC
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score,f1_score,confusion_matrix

def make_feature_dicts(data, NER=True, w2v=True, pos=True):
   """
   builds features :
      when NER=True, use NER labels as features
      when w2v=True, use w2v vectors to build features
      when parseTree=True, use features from parse tree of the sentence
   """
   ###TODO
   token = True
   caps = True
   context = True
   listOfDicts = []
   nerList = []
   for commentList in data:
       videoList = {}
       for tokenList in commentList:
           sentenceList = []
           sentenceDict = {}
           for tokenTupple in tokenList:
               featDict = {}
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
   load the data from file and prepare in the required format
   """
   l_s = []
   l_s_v = []
   Y = []
   #raw_data = pickle.load(open(fname, 'rb'))
   for videoId, details in raw_data.items():
      Y.append(details['relevance'])
      comments = details['comments']
      if comments is None:
         continue
      for comment in comments:
         if comment is None:
            continue
         sentences = comment.split('.')
         if len(sentences) > 0:
            for i in range(len(sentences) - 1):
               l_s.append(sentences[i])
         else:
           l_s.append(comment)
      l_s_v.append(l_s)
   #print("l_s : ", l_s)
   #print("l_s_v : ", l_s_v)
   pickle.dump(l_s, open('listOfSentences.pkl', 'wb'))
   pickle.dump(l_s_v, open('listOfSentences_per_video.pkl', 'wb'))
   return Y

def evaluate_combinations(train_data, test_data):
   combination_results = dict()
   f1 = []
   n_params = []
   caps = []
   pos = []
   chunk = []
   context = []
   comb_list = list(product([True, False], repeat = 4))
   for i,each_comb in enumerate(comb_list):
    caps.append(each_comb[0])
    pos.append(each_comb[1])
    chunk.append(each_comb[2])
    context.append(each_comb[3])
    train_data = read_data('train.txt')
    dicts, labels = make_feature_dicts(train_data,token=True,caps=each_comb[0],pos=each_comb[1],chunk=each_comb[2],context=each_comb[3])
    vec = DictVectorizer()
    X = vec.fit_transform(dicts)
    clf = LogisticRegression()
    clf.fit(X, labels)
    test_data = read_data('test.txt')
    test_dicts, test_labels = make_feature_dicts(test_data,token=True,caps=each_comb[0],pos=each_comb[1],chunk=each_comb[2],context=each_comb[3])
    X_test = vec.transform(test_dicts)
    preds = clf.predict(X_test)

    confusion_matrix = confusion(test_labels, preds)

    evaluation_matrix = evaluate(confusion_matrix)
    f1_score = average_f1s(evaluation_matrix)
    f1.append(f1_score)
    n_params.append(clf.coef_.size)

   df = pd.DataFrame(data={'f1' : f1, 'n_params' : n_params, 'caps' : caps, 'pos' : pos, 'chunk' : chunk, 'context' : context})
   df = df[['f1', 'n_params','caps','pos','chunk','context']]
   df = df.sort_values(axis=0,by='f1',ascending=False)
   return df
 
if __name__ == "__main__":
   data = pickle.load(open("actualData.pkl","rb"))
   Y = np.array(load_data(data))
   list_of_sentence_per_video = pickle.load(open('listOfSentences_per_video.pkl','rb'))
   tagged_sentence_list = tag_sentence(list_of_sentence_per_video)
   #pickle.dump(tagged_sentence_list, open('NERtaggedSentences.pkl', 'wb'))
   feature_vectors = make_feature_dicts(tagged_sentence_list, NER=True, w2v=True, pos=True)
   print(len(feature_vectors))
   print(len(Y))

   vec = DictVectorizer()
   X = vec.fit_transform(feature_vectors)
   print(X)
   """
   vectorizer = CountVectorizer(min_df=1)
   X = vectorizer.fit_transform(feature_vectors)
   """
   print('training data shape: %s\n' % str(X.shape))
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

   clf_LinearSVC = LinearSVC(C=0.1,random_state=123,class_weight="balanced",max_iter=100,fit_intercept=True)

   X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 42)

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


  #evaluate_combinations(X_train, Y_train) 
