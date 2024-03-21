import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import numpy as np
import joblib

def sentiment_classifier(sentence):
    nltk.download("stopwords")
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')

    # Tokenize the input sentence
    review = re.sub('[^a-zA-Z]', ' ', sentence)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)

    # Load the vocabulary
    vocabulary = np.load("vocab.npy", allow_pickle=True).item()

    # Instantiate the CountVectorizer with loaded vocabulary
    cv = CountVectorizer(decode_error="replace", vocabulary=vocabulary)

    # Transform the tokenized sentence
    X = cv.transform([review]).toarray()

    # Load the trained classifier
    classifier = joblib.load("svm_model.joblib")

    # Predict the sentiment
    prediction = classifier.predict(X)

    # Return the sentiment prediction
    if prediction[0] == 1:
        return "Positive Review"
    else:
        return "Negative Review"

# Example usage:
# sentence = "This movie was great, I loved it!"
# print(sentiment_classifier(sentence))

# Serialize the function along with necessary objects
joblib.dump(sentiment_classifier, "sentiment_classifier.joblib")
