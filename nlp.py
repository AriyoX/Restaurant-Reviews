import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re 
import nltk 
from sklearn.feature_extraction.text import CountVectorizer
nltk.download("stopwords") 
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib

dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = "\t", quoting = 3)

corpus = []
for i in range(0, 1000):
        review = re.sub('[^a-zA-Z]', ' ', dataset.iloc[:, 0][i])     
        review = review.lower()                                       
        review = review.split()                                      
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)                                    
        corpus.append(review)

cv = CountVectorizer(max_features = 1566)                                                 
X = cv.fit_transform(corpus).toarray() 
np.save("vocab.npy", cv.vocabulary_)
y = dataset.iloc[:, -1].values 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

joblib.dump(classifier, 'svm_model.joblib')