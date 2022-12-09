from flask import Flask,request,render_template,url_for,jsonify
from tensorflow import keras
from keras import backend as K
from keras.models import load_model
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import datapath
import nltk
from nltk.corpus import stopwords
import re
import numpy as np

# Normalizing content of texts
def sent2word(x):
    stop_words = set(stopwords.words('english')) 
    x=re.sub("[^A-Za-z]"," ",x)
    x.lower()
    filtered_sentence = [] 
    words=x.split()
    for w in words:
        if w not in stop_words: 
            filtered_sentence.append(w)
    return filtered_sentence

# Making word of given text to vector based on given reference model
def makeVec(words, model, num_features):
    vec = np.zeros((num_features,),dtype="float32")
    print(vec)
    noOfWords = 0
    # index2word_set = set(model.index_to_key)
    for i in words:
        if i in model:
            noOfWords += 1
            vec = np.add(vec,model[i])    
            print(vec)    
            print(vec.shape)    
    vec = np.divide(vec,noOfWords)
    return vec

# Constructing texts with vector as its content
def getVecs(essays, model, num_features):
    c=0
    essay_vecs = np.zeros((len(essays),num_features),dtype="float32")
    print(essay_vecs.shape)
    for i in essays:
        essay_vecs[c] = makeVec(i, model, num_features)
        c+=1
    return essay_vecs


def convertToVec(text):
    content=text
    if len(content) > 20:
        num_features = 200
        model = KeyedVectors.load_word2vec_format(datapath("word2vec_pre_kv_c"), binary=False)
        print(model)
        clean_test_essays = []
        clean_test_essays.append(sent2word(content))
        print(clean_test_essays)
        testDataVecs = getVecs(clean_test_essays, model, num_features)
        testDataVecs = np.array(testDataVecs)
        testDataVecs = np.reshape(testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))

        lstm_model = load_model("final_lstm.h5")
        preds = lstm_model.predict(testDataVecs)
        return str(round(preds[0][0]))



# Ceck using data dummy 
final_text = "I want to find job, do you have some? If so, please give me some :)"
score = convertToVec(final_text)
print(final_text + "/n" + score)
# app = Flask(__name__)

# @app.get('/')
# def index_get():
#     return render_template('index.html')

# @app.route("/")
# def image():
#     return render_template("habib.png","aidah.png","salma.png","elkiya.png")
    
# @app.route("/", methods=['POST'])
# def create_task():
#     K.clear_session()
#     final_text = request.get_json("text")["text"]
#     score = convertToVec(final_text)
#     print(final_text + "/n" + score)
#     K.clear_session()
#     return jsonify({score: score})

# if __name__=='__main__':
#     app.run(debug=True)