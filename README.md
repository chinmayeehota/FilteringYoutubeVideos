For Executing the code please follow the below steps:

1). <b>Run Collect.py</b>: (This needs manual tagging and has already been done.)
    This file will collect all the data from youtube. Then we manually tag the data and then it takes the tagged file and creates a new pickle file called actualData.pkl. Running Collect.py with command line arguement 0 (python Collect.py 0) will collect the data from YouTube, search queries should be specified within Collect.py. Running with command line arguement 1(python Collect.py 1), will read the manually tagged excel file and create the pickle file of the data. 

2). <b>DataPreprocessing.py</b>:
    Here we have preprocessed the data and implemented our baseline methods like predict_random, predict_dominant_class and          MultinomialNB and used evalution parameters like F1, accuracy and precision.

3). <b>NBwithNgram.py</b>:
    Here we have implemented Naive Bayes classifier with tri-gram.

4). <b>Model_Latest.py</b>:
    This is the main file where we have created the additional pickle files as needed by different classifiers. Also we have       implemented SVM, Neural Network and Multinomial Naive Bayes.
