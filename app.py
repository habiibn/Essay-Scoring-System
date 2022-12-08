from flask import Flask, render_template

app = Flask(__name__)



@app.get('/')
def index_get():
    return render_template('index.html')

@app.route("/")
def image():
    return render_template("habib.png","aidah.png","salma.png","elkiya.png")