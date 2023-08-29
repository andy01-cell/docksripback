import os
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# wajib
# nltk.download('punkt')

# Membaca dataset dari file CSV
dataframe = pd.read_csv('dataset/1.csv', encoding='ISO-8859-1')

# Atribut judul, abstrak, dan class pada dataset
judul = dataframe['judul'].values
abstrak = dataframe['abstrak'].values
kelas = dataframe['class'].values

# Pra-pemrosesan data
def preprocess(text):
    # Case folding (mengubah teks menjadi huruf kecil)
    text = text.lower()

    # Tokenizing (mengubah teks menjadi token)
    tokens = nltk.word_tokenize(text)

    # Filtering (menghapus karakter yang tidak diperlukan)
    tokens = [token for token in tokens if token not in string.punctuation]

    # Stemming (mengubah kata-kata menjadi bentuk dasarnya)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    # Menggabungkan kembali token menjadi teks
    processed_text = ' '.join(tokens)

    return processed_text

# Menggabungkan judul, dan abstrak menjadi satu teks
texts = []
for i in range(len(judul)):
    text = judul[i] + ' ' + abstrak[i]
    preprocessed_text = preprocess(text)
    texts.append(preprocessed_text)

# Bagi data menjadi data latihan dan data pengujian
texts_train, texts_test, labels_train, labels_test = train_test_split(texts, kelas, test_size=0.2, random_state=42)

# Konversi teks menjadi vektor TF-IDF
vectorizer = TfidfVectorizer()
tfidf_train = vectorizer.fit_transform(texts_train)
tfidf_test = vectorizer.transform(texts_test)

# Inisialisasi model klasifikasi Naive Bayes, SVM dan KNN
classifier = MultinomialNB()
svm_classifier = SVC()
knn_classifier = KNeighborsClassifier()

# Melatih model NB, SVM, KNN
classifier.fit(tfidf_train, labels_train)
svm_classifier.fit(tfidf_train, labels_train)
knn_classifier.fit(tfidf_train, labels_train)

# Select a single test data point
single_test_text = texts_test[0]
single_label = labels_test[0]

# Convert the test text to TF-IDF vector
single_tfidf_test = vectorizer.transform([single_test_text])

# Predictions for each classifier
nb_prediction = classifier.predict(single_tfidf_test)
svm_prediction = svm_classifier.predict(single_tfidf_test)
knn_prediction = knn_classifier.predict(single_tfidf_test)

# Ensemble with voting
ensemble_votes = [nb_prediction[0], svm_prediction[0], knn_prediction[0]]
ensemble_prediction = np.unique(ensemble_votes)[np.argmax(np.unique(ensemble_votes, return_counts=True)[1])]

# Create a label encoder object
label_encoder = LabelEncoder()

# Fit the label encoder on all training and test labels
all_labels = np.concatenate((labels_train, labels_test, [single_label]), axis=None)  # Combine training, test, and single label
label_encoder.fit(all_labels)

# Transform the single label to numerical representation
numerical_single_label = label_encoder.transform([single_label])[0]
numerical_ensemble_prediction = label_encoder.transform([ensemble_prediction])[0]

# Calculating MAE for individual classifiers
mae_nb = mean_absolute_error([numerical_single_label], [label_encoder.transform([nb_prediction])[0]])
mae_svm = mean_absolute_error([numerical_single_label], [label_encoder.transform([svm_prediction])[0]])
mae_knn = mean_absolute_error([numerical_single_label], [label_encoder.transform([knn_prediction])[0]])

# Calculating MAE for ensemble classifier
mae_ensemble = mean_absolute_error([numerical_single_label], [numerical_ensemble_prediction])

print("Actual Label:", single_label)
print("Naive Bayes Prediction:", label_encoder.inverse_transform([nb_prediction])[0])
print("SVM Prediction:", label_encoder.inverse_transform([svm_prediction])[0])
print("KNN Prediction:", label_encoder.inverse_transform([knn_prediction])[0])
print("Ensemble Prediction:", label_encoder.inverse_transform([numerical_ensemble_prediction])[0])
print("Naive Bayes MAE:", mae_nb)
print("SVM MAE:", mae_svm)
print("KNN MAE:", mae_knn)
print("Ensemble MAE:", mae_ensemble)
