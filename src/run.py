# using flask_restful
from flask import Flask, jsonify, request, make_response
from flask_restful import Resource, Api
import product_recommendation_module as prm
import csv
import numpy as np
import json

model, helper_data = prm.getModelandHelperData()
# creating the flask app
app = Flask(__name__)
# creating an API object
api = Api(app)
# segmento renta ind_actividad_cliente cod_prov	
test_input = {
	'fecha_dato': '2016-06-28',
	'ncodpers': '15889',
	'ind_empleado': 'F',
	'pais_residencia': 'ES',
	'sexo': 'V',
	'age': '56',
	'fecha_alta': '1995-01-16',
	'antiguedad': '256',
	'tiprel_1mes': 'N',
	'cod_prov': '28',
	'ind_actividad_cliente': '1',
	'renta': '326124.90',
	'segmento': '01 - TOP'
}

class ndarrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def generate_prediction(users_dict):
	modelAnswer = prm.predict(model, helper_data, users_dict)
	return modelAnswer

# making a class for a particular resource
# the get, post methods correspond to get and post requests
# they are automatically mapped by flask_restful.
# other methods include put, delete, etc.
class Root(Resource):
	
	# Corresponds to POST request
	def post(self):	
		ans = generate_prediction(test_input)
		ansJson = json.dumps(ans, cls=ndarrayEncoder) 
		return make_response(jsonify({'data': ansJson}), 201)

# adding the defined resources along with their corresponding urls
api.add_resource(Root, '/')

# driver function
if __name__ == '__main__':
	app.run(debug = True)
