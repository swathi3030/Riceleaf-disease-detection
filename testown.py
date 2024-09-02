import tensorflow as tf
from flask import Flask,request,render_template
#from keras.utils import load_img
from tensorflow.keras.preprocessing.image import load_img
from keras.models import load_model
#from keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input
import numpy as np
import cv2
import flask
from tensorflow.keras.preprocessing.image import img_to_array


app = Flask(__name__)
@app.route('/testown',methods=['POST','GET'])
def index():
    model = load_model("finalcnn.h5",compile=False)
    # def check(res):
    #     p1 = ["bacterial blight", "brown spot","healthy"]
    #     path = p1
    #     pred = model.predict(res)
    #     res = np.argmax(pred)
    #     res = path[res]
    #     return (res)
    #
    # def convert_img_to_tensor2(fpath):
    #     img = cv2.imread(fpath)
    #     img = cv2.resize(img, (180, 180))
    #     res = img_to_array(img)
    #     res = np.array(res, dtype=np.float16) / 255.0
    #     res = res.reshape(-1, 180, 180, 3)
    #     res = res.reshape(1, 180, 180, 3)
    #     return res

    def check(res):
            p1 = ["brown spot", "healthy", "leaf scald"]
            path = p1
            pred = model.predict(res)
            res = np.argmax(pred)
            res = path[res]
            return res


    def convert_img_to_tensor2(fpath):
            img = cv2.imread(fpath)
            print(img)

            img = cv2.resize(img, (256, 256))
            print(img)
            res = img_to_array(img)
            res = np.array(res, dtype=np.float16) / 255.0
            res = res.reshape(-1, 256, 256, 3)
            res = res.reshape(1, 256, 256, 3)
            return res

    if request.method == 'POST':
        img=request.files['img']
        img.save('static/h.jpg')
        res = convert_img_to_tensor2("static/h.jpg")
        msg=check(res)
        return render_template('result.html', res=msg)

    else:
        return render_template('rice.html',res="invalid input")


if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0")