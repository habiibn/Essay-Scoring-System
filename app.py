from flask import Flask,request,render_template,url_for,jsonify
import site
import numpy as np
import pandas as pd
import pickle
import nltk
nltk.download('stopwords')
import re
import math
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,word_tokenize
from gensim.models import Word2Vec
from gensim.test.utils import datapath
from keras.layers import Embedding, LSTM, Dense, Dropout, Lambda, Flatten
from keras.models import Sequential, load_model, model_from_config
import keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import cohen_kappa_score
from gensim.models.keyedvectors import KeyedVectors
from keras import backend as K


def esai_ke_listkata(essay_v, remove_stopwords):
    essay_v = re.sub("[^a-zA-Z]", " ", essay_v)
    kata = essay_v.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        kata = [w for w in kata if not w in stops]
    return (kata)

def esai_ke_kalimat(essay_v, remove_stopwords):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(essay_v.strip())
    kalimat = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            kalimat.append(esai_ke_listkata(raw_sentence, remove_stopwords))
    return kalimat


def makeFeatureVec(kata, model, num_features):
    featureVec = np.zeros((num_features,),dtype="float32")
    num_words = 0.
    for word in kata:
        if word in model:
            num_words += 1
            featureVec = np.add(featureVec, model[word])       
    featureVec = np.divide(featureVec,num_words)
    return featureVec


def getAvgFeatureVecs(essays, model, num_features):
    counter = 0
    essayFeatureVecs = np.zeros((len(essays),num_features),dtype="float32")
    for essay in essays:
        essayFeatureVecs[counter] = makeFeatureVec(essay, model, num_features)
        counter = counter + 1
    return essayFeatureVecs


with open('saved_dictionary.pkl', 'rb') as f:
    data = pickle.load(f)


model = data

def final (text):
    if len(text) > 20:
        num_features = 50
        clean_test_essays = []
        clean_test_essays.append(esai_ke_listkata( text, remove_stopwords=True ))
        testDataVecs = getAvgFeatureVecs( clean_test_essays, model, num_features )
        testDataVecs = np.array(testDataVecs)
        testDataVecs = np.reshape(testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))

        lstm_model = load_model("model_lstm_50.h5")
        prediksi = lstm_model.predict(testDataVecs)

        if math.isnan(prediksi):
            prediksi = 0
        else:
            prediksi = np.round(prediksi)

        if prediksi < 0:
            prediksi = 0
    else:
        prediksi = 0
    return prediksi



app = Flask(__name__)



@app.get('/')
def index_get():
    return render_template('index.html')

@app.route("/")
def image():
    return render_template("habib.png","aidah.png","salma.png","elkiya.png")

@app.route('/sample', methods=["POST", "GET"])
def sample():
    if request.method == 'POST':
        sample_input = request.form.get('text1')
        sample_output = final(sample_input)
        final_output = sample_output[0]
        final1_output = final_output[0] 
        return render_template('index.html', sample_input=sample_input, sample_output=final1_output)
    elif request.method == 'GET':
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug='off')
