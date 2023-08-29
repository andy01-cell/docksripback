from flask import Flask, request, jsonify
import os
import pandas as pd
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.utils import shuffle
from sklearn.ensemble import VotingClassifier

app = Flask(__name__)

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

# Mengubah label string menjadi representasi numerik menggunakan LabelEncoder
label_encoder = LabelEncoder()
numerical_labels_train = label_encoder.fit_transform(labels_train)
numerical_labels_test = label_encoder.transform(labels_test)

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

# Shuffle the testing data indices
texts_test, labels_test = shuffle(texts_test, labels_test, random_state=60)

# Predictions for the entire shuffled testing set
nb_predictions = classifier.predict(tfidf_test)
svm_predictions = svm_classifier.predict(tfidf_test)
knn_predictions = knn_classifier.predict(tfidf_test)

# Calculate accuracies
nb_accuracy = accuracy_score(labels_test, nb_predictions)
svm_accuracy = accuracy_score(labels_test, svm_predictions)
knn_accuracy = accuracy_score(labels_test, knn_predictions)

# Convert the string labels to numerical representations using LabelEncoder
label_encoder = LabelEncoder()
numerical_labels_train = label_encoder.fit_transform(labels_train)
numerical_labels_test = label_encoder.transform(labels_test)
numerical_nb_predictions = label_encoder.transform(nb_predictions)
numerical_svm_predictions = label_encoder.transform(svm_predictions)
numerical_knn_predictions = label_encoder.transform(knn_predictions)

# Calculate MAE
mae_nb = mean_absolute_error(numerical_labels_test, numerical_nb_predictions)
mae_svm = mean_absolute_error(numerical_labels_test, numerical_svm_predictions)
mae_knn = mean_absolute_error(numerical_labels_test, numerical_knn_predictions)

# Buat objek ensemble classifier dengan model yang ada
ensemble_classifier = VotingClassifier(estimators=[
    ('nb', classifier),
    ('svm', svm_classifier),
    ('knn', knn_classifier)
], voting='hard')  # 'hard' untuk voting mayoritas

# Latih ensemble classifier dengan data latihan
ensemble_classifier.fit(tfidf_train, numerical_labels_train)

# Melakukan prediksi dengan ensemble classifier pada data uji
ensemble_test_prediction = ensemble_classifier.predict(tfidf_test)

# Menghitung akurasi ensemble
ensemble_accuracy = accuracy_score(numerical_labels_test, ensemble_test_prediction)

# Menghitung MAE ensemble
mae_ensemble = mean_absolute_error(numerical_labels_test, ensemble_test_prediction)

print("Actual Label:", single_label)
print("Naive Bayes Prediction:", nb_predictions)
print("SVM Prediction:", svm_predictions)
print("KNN Prediction:", knn_predictions)
print("Naive Bayes Accuracy:", nb_accuracy)
print("SVM Accuracy:", svm_accuracy)
print("KNN Accuracy:", knn_accuracy)
print("Naive Bayes MAE:", mae_nb)
print("SVM MAE:", mae_svm)
print("KNN MAE:", mae_knn)
print("Ensemble Prediction:", ensemble_test_prediction)
print("Ensemble Accuracy:", ensemble_accuracy)
print("Ensemble MAE:", mae_ensemble)


@app.route('/predict', methods=['POST'])
def predict():
    test_title = request.form.get('judul')
    test_abstract = request.form.get('abstrak')

    # Menggabungkan judul dan abstrak menjadi satu teks
    test_text = test_title + ' ' + test_abstract

    # Pra-pemrosesan teks uji
    preprocessed_test_text = preprocess(test_text)

    # Konversi teks uji menjadi vektor TF-IDF
    tfidf_test_text = vectorizer.transform([preprocessed_test_text])

    # Melakukan prediksi dengan ensemble classifier pada data uji
    ensemble_test_prediction = ensemble_classifier.predict(tfidf_test_text)

    # Mengubah prediksi menjadi label kelas
    predicted_class = label_encoder.inverse_transform(ensemble_test_prediction)

    # Menyiapkan hasil prediksi untuk respons API
    response = {
        "input_judul": test_title,
        "input_abstrak": test_abstract,
        "predicted_class": predicted_class[0],
        "ensemble_akurasi" : ensemble_accuracy,
        "ensemble_MAE": mae_ensemble,
        "Naive_Bayes_Accuracy": nb_accuracy,
        "SVM_Accuracy": svm_accuracy,
        "KNN_Accuracy": ensemble_accuracy,
        "Naive_Bayes_MAE": mae_nb,
        "SVM_MAE": mae_svm,
        "KNN_MAE": mae_knn,

    }

    return jsonify(response)

if __name__ == '__main__':
    # ... Kode yang ada sebelumnya ...

    app.run(debug=True)


