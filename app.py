from flask import Flask,request,render_template,url_for,jsonify
# from tensorflow.keras import backend as K
# from tensorflow.keras.models import load_model
from gensim.models.keyedvectors import KeyedVectors
from nltk.corpus import stopwords
import re
import numpy as np

def image():
    return render_template("habib.png","aidah.png","salma.png","elkiya.png")

# Normalizing content of texts
def sent2word(x):
    # stop_words = set(stopwords.words('english')) 
    # x=re.sub("[^A-Za-z]"," ",x)
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
    noOfWords = 0.
    index2word_set = set(model.wv.index2word)
    for i in words:
        if i in index2word_set:
            noOfWords += 1
            vec = np.add(vec,model[i])        
    vec = np.divide(vec,noOfWords)
    return vec

# Constructing texts with vector as its content
def getVecs(essays, model, num_features):
    c=0
    essay_vecs = np.zeros((len(essays),num_features),dtype="float32")
    for i in essays:
        essay_vecs[c] = makeVec(i, model, num_features)
        c+=1
    return essay_vecs


def convertToVec(text):
    content=text
    if len(content) > 20:
        num_features = 300
        model = KeyedVectors.load_word2vec_format("word2vecmodel.bin", binary=True)
        clean_test_essays = []
        clean_test_essays.append(sent2word(content))
        testDataVecs = getVecs(clean_test_essays, model, num_features )
        testDataVecs = np.array(testDataVecs)
        testDataVecs = np.reshape(testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))

        # lstm_model = load_model("final_lstm.h5")
        preds = lstm_model.predict(testDataVecs)
        return str(round(preds[0][0]))


app = Flask(__name__)
# model = load_model('final_lstm.h5')

@app.get('/')
def index_get():
    image()
    return render_template('index.html')

@app.route("/", methods=['POST'])
def create_task():
    # K.clear_session()
    final_text = request.get_json("text")["text"]
    score = convertToVec(final_text)
    # K.clear_session()
    return jsonify({'score': score}), 201

if __name__=='__main__':
    app.run(debug=True)