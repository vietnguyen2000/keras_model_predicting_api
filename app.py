from flask import Flask, render_template, request, jsonify
from google.protobuf.message import Error
import model
import time

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, myname is sang"

@app.route("/predict/", methods=['GET', 'POST'])
def post():
    execution_time = time.time()
    try:
        content = request.json
        url_list = content["urls"]
        label_list = []
        label_list = trained_model.predict(url_list)
        
        url_label = list(zip(url_list,label_list))
        return jsonify(label=url_label, time=time.time() - execution_time), 200
    except Error as e:
        print(e)
        return jsonify(label="Error"), 400

if __name__ == '__main__':
    trained_model = model.Model()
    app.run(debug=True)