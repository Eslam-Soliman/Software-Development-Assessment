# using flask_restful
from flask import Flask, jsonify, request, make_response
from flask_restful import Resource, Api
import product_recommendation_module as prm
import json

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
	parsed = {
		'fecha_dato': args.get('fecha_dato'),
		'ncodpers': args.get('ncodpers'),
		'ind_empleado': args.get('ind_empleado') or '',
		'pais_residencia': args.get('pais_residencia') or '',
		'sexo': args.get('sexo') or '',
		'age': args.get('age') or '',
		'fecha_alta': args.get('fecha_alta') or '',
		'antiguedad': args.get('antiguedad') or '',
		'tiprel_1mes': args.get('tiprel_1mes') or '',
		'cod_prov': args.get('cod_prov') or '',
		'ind_actividad_cliente': args.get('ind_actividad_cliente') or '',
		'renta': args.get('renta') or '',
		'segmento': args.get('segmento') or '',
	}
	return parsed

def check_input_valid(args):
	if args.get('fecha_dato') and args.get('ncodpers'):
		return True, ''
	else:
		missing_params = []
		if not args.get('fecha_dato'): 
			missing_params.append('fecha_dato')
		if not args.get('ncodpers'): 
			missing_params.append('ncodpers')			
		error_msg = 'Error: invalid input format. The following parameters are missing: ' + ', '.join(missing_params)
		return False, error_msg


# making a class for a particular resource
# the get, post methods correspond to get and post requests
# they are automatically mapped by flask_restful.
# other methods include put, delete, etc.
class Root(Resource):
	
	# Corresponds to POST request
	def post(self):
		input_valid, error_msg = check_input_valid(request.args)
		if input_valid:
			parsed_params = parse_params(request.args)
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
