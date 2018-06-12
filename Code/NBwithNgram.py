
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
import nltk
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB,MultinomialNB
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score
"""
    step 1 can be just comments and relevance
    step 2 search query and title and relevance
    
    
    """
def getCommentsAndRelevance(raw_data):
    commentList = []
    relevanceList = []
    for videoId,value in raw_data.items():
        comment = value['comments']
        commentString = ' '.join(str(e) for e in comment)
        commentList.append(commentString)
        relevanceList.append(value['relevance'])
    return commentList,relevanceList


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



if __name__ == "__main__":
    fname = 'NewactualData.pkl'
    raw_data = pickle.load(open(fname, 'rb'))
    commentList,relevanceList = getCommentsAndRelevance(raw_data)
    
    vec = TfidfVectorizer(min_df=1,ngram_range=(3, 3))
    X = vec.fit_transform(commentList)
    
    #vec = CountVectorizer(ngram_range=(2, 2))
    #X = vec.fit_transform(commentList)
    y = np.array(relevanceList)
    kfold = KFold(n_splits=5, shuffle=True, random_state=1234)
    preds = []
    truths = []
    XDense = X.toarray()
    for train, test in kfold.split(X):
        gnb = GaussianNB()
        clf = gnb.fit(XDense[train], y[train])
        #mnb = MultinomialNB(alpha=0.5, class_prior=None, fit_prior=True)
        #clf = mnb.fit(XDense[train], y[train])
        print(clf.class_count_)
        preds.extend(clf.predict(XDense[test]))
        truths.extend(y[test])
    #n_errors = np.sum(np.abs(np.array(preds) - np.array(truths)))
    #print('percent incorrect=%.2f' % (n_errors / len(y)))
    acc = accuracy_score(truths, preds)
    print('accuracy : %0.3f'%acc)
    cnf_matrix = confusion_matrix(truths,preds,labels=['Not Relevant','Deceptive','Relevant'])
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['Not Relevant','Deceptive','Relevant'],
                      title='Confusion matrix, without normalization')




#print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)
    
    """v = CountVectorizer(ngram_range=(2, 2))
    pprint(v.fit(["an apple a day keeps the doctor away"]).vocabulary_)"""
