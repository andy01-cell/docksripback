import os
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

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

# Prediksi kelas untuk data pengujian
predictions = classifier.predict(tfidf_test)
svm_predictions = svm_classifier.predict(tfidf_test)
knn_predictions = knn_classifier.predict(tfidf_test)

# Ensemble dengan menggunakan voting (mengambil mayoritas hasil prediksi dari ketiga algoritma)
ensemble_predictions = []
for nb_pred, svm_pred, knn_pred in zip(predictions, svm_predictions, knn_predictions):
    votes = [nb_pred, svm_pred, knn_pred]
    # Menggunakan fungsi max untuk voting mayoritas
    # Jika terdapat lebih dari satu nilai mayoritas, ambil nilai pertama yang muncul
    ensemble_pred = max(set(votes), key=votes.count)
    ensemble_predictions.append(ensemble_pred)

# Create a label encoder object
label_encoder = LabelEncoder()

# Convert the string labels to numerical representations
numerical_labels_test = label_encoder.fit_transform(labels_test)
numerical_predictions = label_encoder.transform(predictions)
numerical_svm_predictions = label_encoder.transform(svm_predictions)
numerical_knn_predictions = label_encoder.transform(knn_predictions)
numerical_ensemble_predictions = label_encoder.transform(ensemble_predictions)

# Evaluating the individual algorithms
ensemble_accuracy = accuracy_score(numerical_labels_test, numerical_ensemble_predictions)
ensemble_mae_nb = mean_absolute_error(numerical_labels_test, numerical_predictions)
ensemble_mae_svm = mean_absolute_error(numerical_labels_test, numerical_svm_predictions)
ensemble_mae_knn = mean_absolute_error(numerical_labels_test, numerical_knn_predictions)

# Evaluating Naive Bayes
nb_accuracy = accuracy_score(labels_test, predictions)
print("Naive Bayes Accuracy:", nb_accuracy)
print("Naive Bayes Predictions:", predictions)

# Evaluating SVM
svm_accuracy = accuracy_score(labels_test, svm_predictions)
print("SVM Accuracy:", svm_accuracy)
print("SVM Predictions:", svm_predictions)

# Evaluating KNN
knn_accuracy = accuracy_score(labels_test, knn_predictions)
print("KNN Accuracy:", knn_accuracy)
print("KNN Predictions:", knn_predictions)

# Evaluating the Ensemble Classifier
ensemble_accuracy = accuracy_score(labels_test, ensemble_predictions)
print("Ensemble Accuracy:", ensemble_accuracy)
print("Ensemble Predictions:", ensemble_predictions)

print("Naive Bayes MAE:", ensemble_mae_nb)
print("SVM MAE:", ensemble_mae_svm)
print("KNN MAE:", ensemble_mae_knn)
