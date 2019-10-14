from django.shortcuts import render
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse, HttpResponse
from django.core import serializers
from django.conf import settings
import json

#library of predict
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array

import manage

from PIL import Image
import numpy
import requests
from io import BytesIO
# Create your views here.
graph = None
model = None

def load_model():
    global model
    mobile = tf.keras.applications.mobilenet.MobileNet()
    x = mobile.layers[-6].output

    # Create a new dense layer for predictions
    # 7 corresponds to the number of classes
    x = Dropout(0.25)(x)
    predictions = Dense(8, activation='softmax')(x)

    # inputs=mobile.input selects the input layer, outputs=predictions refers to the
    # dense layer we created above.

    model = Model(inputs=mobile.input, outputs=predictions)
    model.load_weights("./Predict/model _final.h5")
    print("AAAA")
    # global graph  
    # graph = tf.compat.v1.get_default_graph()
    # tf.keras.backend.clear_session()
# load_model()
@api_view(["POST"])
def predict_disease(link_image):
    tf.keras.backend.clear_session()
    try:
        image = json.loads(link_image.body)
    except ValueError as e:
        return Response(e.args[0],status.HTTP_400_BAD_REQUEST)

    load_model()

    response = requests.get(image["link_image"])
    im = Image.open(BytesIO(response.content))
    # im = Image.open("/content/download.png")
    im = im.resize((224,224), Image.ANTIALIAS)
    img = img_to_array(im)
    img = tf.keras.applications.mobilenet.preprocess_input(img)
    img_ex = numpy.expand_dims(img, axis=0)
    # with graph.as_default():
    result = model.predict(img_ex)
    result = result.ravel().tolist()
    return HttpResponse(json.dumps(result),content_type='application/json')