from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2

app = Flask(__name__)

# Load the model once when the app starts
model = load_model("finalcnn.h5", compile=False)

def check(res):
    p1 = ["brown spot", "healthy", "leaf scald"]
    pred = model.predict(res)
    res_idx = np.argmax(pred)
    if res_idx >= len(p1):
        print(f"Prediction index {res_idx} out of range for list {p1}")
        return "Unknown"
    res = p1[res_idx]
    print(f"Predicted: {res} with confidence: {pred[0][res_idx]}")
    return res

def convert_img_to_tensor2(fpath):
    img = cv2.imread(fpath)
    if img is None:
        raise ValueError(f"Could not read image from {fpath}")
    img = cv2.resize(img, (224, 224))  # Resize to 224x224
    res = img_to_array(img)
    res = np.array(res, dtype=np.float32) / 255.0
    res = res.reshape(1, 224, 224, 3)
    return res

@app.route('/test', methods=['POST', 'GET'])
def test():
    if request.method == 'POST':
        img = request.files['img']
        img_path = 'static/h.jpg'
        img.save(img_path)
        try:
            res = convert_img_to_tensor2(img_path)
            msg = check(res)
        except Exception as e:
            msg = str(e)
        return render_template('result.html', res=msg)
    else:
        return render_template('rice.html', res="invalid input")

@app.route('/testown', methods=['POST', 'GET'])
def testown():
    if request.method == 'POST':
        img = request.files['img']
        img_path = 'static/h.jpg'
        img.save(img_path)
        try:
            res = convert_img_to_tensor2(img_path)
            msg = check(res)
        except Exception as e:
            msg = str(e)
        return render_template('result.html', res=msg)
    else:
        return render_template('rice.html', res="invalid input")

@app.route('/a', methods=['POST', 'GET'])
def choose():
    if request.method == 'POST':
        name = request.form.get("datasets")
        if name == "created":
            return render_template('riceown.html')
        else:
            return render_template('rice.html')
    return render_template('choosedataset.html')

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == "admin" and password == "1234":
            return render_template('choosedataset.html')
        else:
            return render_template('login.html', msg="Login failed")

    return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
