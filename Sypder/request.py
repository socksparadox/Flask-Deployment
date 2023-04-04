# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 16:58:23 2023

@author: hp
"""

import requests

url = 'http://127.0.0.1:5000/predict_api'
r = requests.post(url, json={'fixed_acidity':7.4, 'volatile_acidity': 0.7, 'citric_acid':0,	
                             'residual_sugar':1.9, 'chlorides':0.076, 'free_sulfur_dioxide':11,	
                             'total_sulfur_dioxide':34, 'density':0.9978, 'pH':3.51,	
                             'sulphates':0.56, 'alcohol':9.4,'color':1})

print(r.json())