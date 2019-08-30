from django.shortcuts import render
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.core import serializers
from django.conf import settings
import json

#library of predict
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


import Image
import numpy
import requests
from io import BytesIO
# Create your views here.

@api_view(["POST"])
def predict_disease(link_image):
    try:
        image = json.loads(link_image.body)
    except ValueError as e:
        return Response(e.args[0],status.HTTP_400_BAD_REQUEST)
    mobile = tf.keras.applications.mobilenet.MobileNet()
    # CREATE THE MODEL ARCHITECTURE

    # Exclude the last 5 layers of the above model.
    # This will include all layers up to and including global_average_pooling2d_1
    x = mobile.layers[-6].output

    # Create a new dense layer for predictions
    # 7 corresponds to the number of classes
    x = Dropout(0.25)(x)
    predictions = Dense(7, activation='softmax')(x)

    # inputs=mobile.input selects the input layer, outputs=predictions refers to the
    # dense layer we created above.

    model = Model(inputs=mobile.input, outputs=predictions)
    model.load_weights("./Predict/model.h5")
    #print(image["link_image"])
    response = requests.get(image["link_image"])
    im = Image.open(BytesIO(response.content))
    # im = Image.open("/content/download.png")
    im = im.resize((224,224), Image.ANTIALIAS)
    np_im = numpy.array(im)
    # print(np_im.shape)
    np_im = np_im[:,:,:3]
    img_ex = numpy.expand_dims(np_im, axis=0)
    result = model.predict(img_ex)
    result = result.ravel().tolist()
    return JsonResponse(json.dumps(result),safe=False)