# using flask_restful
from flask import Flask, jsonify, request, make_response
from flask_restful import Resource, Api
import product_recommendation_module as prm
import json
import numpy as np
from itertools import zip_longest

model, helper_data = prm.getModelandHelperData()
# creating the flask app
app = Flask(__name__)
# creating an API object
api = Api(app)

class ndarrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def generate_prediction(users_dict):
	modelAnswer = prm.predict(model, helper_data, users_dict)
	return modelAnswer

def parse_params(args):
	input_params = ['fecha_dato', 'ncodpers', 'ind_empleado', 'pais_residencia', 'sexo', 'age', 'fecha_alta', 'antiguedad', 'tiprel_1mes', 'cod_prov', 'ind_actividad_cliente', 'renta', 'segmento']
	parsed = {}
	for param in input_params:
		parsed[param] = args.get(param).split(',') or []
	
	return [dict(zip(parsed, t)) for t in zip_longest(*parsed.values(), fillvalue='')]

def check_input_valid(params):
	result = True
	missing_params = []
	for row in params:
		if '' not in [row['fecha_dato'], row['ncodpers']]:
			continue
		else:
			if not row['fecha_dato']: 
				missing_params.append('fecha_dato')
			if not row['ncodpers']: 
				missing_params.append('ncodpers')			
			
	error_msg = 'Error: invalid input format. The following parameters are missing: ' + ', '.join(missing_params) if len(missing_params) else ''					
	return (False, error_msg) if error_msg else (True, '')

# making a class for a particular resource
# the get, post methods correspond to get and post requests
# they are automatically mapped by flask_restful.
# other methods include put, delete, etc.
class Root(Resource):
	
	# Corresponds to POST request
	def post(self):
		parsed_params = parse_params(request.args)
		input_valid, error_msg = check_input_valid(parsed_params)
		if input_valid:
			answer = generate_prediction(parsed_params)
			answer_json = json.dumps(answer, cls=ndarrayEncoder) 
			return make_response(jsonify({'data': answer_json}), 201)
		else:
			return make_response(jsonify({'message': error_msg}), 400)		

# adding the defined resources along with their corresponding urls
api.add_resource(Root, '/')

# driver function
if __name__ == '__main__':
	app.run(debug = True)
