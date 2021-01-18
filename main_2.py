from flask import Flask, render_template, request, jsonify
import predict_image

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, myname is sang"

@app.route("/predict/")
def send():
    try:
        url=request.args.get("url")
        predicted_label = predict_image.predict(url)
        return jsonify(label=predicted_label), 200
    except:
        return jsonify(label="cannot predict"), 200

if __name__ == '__main__':
    app.run(debug=True)