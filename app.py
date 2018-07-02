from flask import Flask, request, jsonify
from flask_cors import CORS
from io import BytesIO
from tensorflow import keras
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.python.keras._impl.keras.applications import imagenet_utils
import base64

app = Flask(__name__)
model = keras.applications.VGG19(include_top=True,weights=None)
model.load_weights("./keras/models/vgg19_weights.h5")
graph = tf.get_default_graph()

@app.route('/image-classify',methods =['POST'])
def image_classify():
    content = request.json
    image = content["image"]
    im = Image.open(BytesIO(base64.b64decode(image)))
    prediction = load_and_predict(im,model)
    result = {}
    for label in prediction:
        _,l,score =label
        result[l] = str(score)
    return jsonify(result)



def load_and_predict(image,model):
    image_array = np.asarray(image.resize(size=(224,224)))
    image_expanded = np.expand_dims(image_array,axis=0)
    with graph.as_default():
        prediction = model.predict(x=image_expanded)
    labeled_prediction = imagenet_utils.decode_predictions(prediction)
    return labeled_prediction[0]

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=5000)
