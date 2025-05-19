import nltk
import random
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy

nltk.download('movie_reviews')
nltk.download('punkt')

def extract_features(words):
    return {word: True for word in words}

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

feature_sets = [(extract_features(words), sentiment) for (words, sentiment) in documents]

train_set, test_set = feature_sets[100:], feature_sets[:100]

classifier = NaiveBayesClassifier.train(train_set)

print("Model Accuracy:", accuracy(classifier, test_set))
classifier.show_most_informative_features(10)

def analyze_sentiment(text):
    words = word_tokenize(text.lower())
    features = extract_features(words)
    return classifier.classify(features)

sample_texts = [
    "I really loved this movie!",
    "It was okay, not the best.",
    "This was a terrible experience.",
    "The plot was boring and the characters were weak.",
    "Fantastic performance and brilliant direction!"
]

print("\nSentiment Analysis Results:")
for text in sample_texts:
    sentiment = analyze_sentiment(text)
    print(f"Text: {text}\nSentiment: {sentiment}\n")
