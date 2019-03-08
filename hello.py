from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

from keras.models import load_model
from keras.preprocessing import image
import numpy as np

import sys
import os
import glob
import re
import pickle


app = Flask(__name__)

MODEL_PATH = 'models/Experiment2.h5'


#loading the pretrained model
model = load_model(MODEL_PATH)
model._make_predict_function()

#loaddictionary
label_dictionary = open("dictionary/dict.pickle","rb")
labels = pickle.load(label_dictionary)
label_dictionary.close()

print('Model loaded. Start serving...')

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(299, 299))
    x=np.array(img)
    x=np.reshape(x,(1,299,299,3))
    preds = model.predict_classes(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname('uploads')
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result
        #red_class = preds.argmax(axis=-1)            # Simple argmax
        #result = str(pred_class[0][0][1])

        label_prediction = [labels[k] for k in preds]
        result = str(label_prediction)

        return result

    return None




if __name__ == "__main__":
    app.run()
