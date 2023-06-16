from django.shortcuts import render
# import necessary libraries
import joblib
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.conf import settings
import numpy as np
import json
import sys
import os

# load the model from the static folder
path_to_model = os.path.join(settings.BASE_DIR, 'static/model/')
loaded_model = joblib.load(open(path_to_model+'diabete_detector_model.pkl', 'rb'))

# Create your views here.

@api_view(['GET'])
def index(request):
    return_data = {
        "error_code": "0",
        "info": "success",
    }
    return Response(return_data)


@api_view(['POST'])
def predict_patient_status(request):
    try:

        # load the request data
        patient_json_info = request.data

        # retrieve all the values from the json data
        patient_info  = np.array(list(patient_json_info.values()))

        # make prediction
        patient_status = loaded_model.predict([patient_info])
        
        # model confidence score
        model_confidence_score = np.max(loaded_model.predict_proba([patient_info]))

        model_prediction = {
            'info': 'success',
            'patient_status': patient_status[0],
            'model_confidence_proba': float("{:.2f}".format(model_confidence_score*100))
        }

    except ValueError as ve:
        model_pradiction = {
            'error_code' : '-1',
            "info": str(ve)
        }


    return Response(model_prediction)
