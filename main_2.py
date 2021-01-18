from flask import Flask, render_template, request, jsonify
import predict_image

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, myname is sang"

@app.route("/predict/", methods=['GET', 'POST'])
def post():
    try:
        content = request.json
        url_list = content["urls"]
        label_list = []
        for i in range(0,len(url_list)):
            label_list.append(predict_image.predict(url_list[i]))
        
        url_label = list(zip(url_list,label_list))
        return jsonify(label=url_label), 200
    except:
        return jsonify(label="Error"), 400

if __name__ == '__main__':
    app.run(debug=True)