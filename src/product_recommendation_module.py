"""
Code based on BreakfastPirate Forum post
__author__ : SRK
"""
import csv
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from sklearn import preprocessing, ensemble

# I need
# age	seniority 	segment		sex 	relationship income country of residence	activity 				regionNumber registration_date
# age 	antiguedad	segmento	sexo	tiprel_1mes	 renta	pais_residencia			ind_actividad_cliente	cod_prov	 fecha_alta	

mapping_dict = {
'ind_empleado'  : {-99:0, 'N':1, 'B':2, 'F':3, 'A':4, 'S':5},
'sexo'          : {'V':0, 'H':1, -99:2},
'tiprel_1mes'   : {-99:0, 'I':1, 'A':2, 'P':3, 'R':4, 'N':5},
'ind_actividad_cliente' : {'0':0, '1':1, -99:2},
'segmento'      : {'02 - PARTICULARES':0, '03 - UNIVERSITARIO':1, '01 - TOP':2, -99:2},
'pais_residencia' : {'LV': 102, 'BE': 12, 'BG': 50, 'BA': 61, 'BM': 117, 'BO': 62, 'JP': 82, 'JM': 116, 'BR': 17, 'BY': 64, 'BZ': 113, 'RU': 43, 'RS': 89, 'RO': 41, 'GW': 99, 'GT': 44, 'GR': 39, 'GQ': 73, 'GE': 78, 'GB': 9, 'GA': 45, 'GN': 98, 'GM': 110, 'GI': 96, 'GH': 88, 'OM': 100, 'HR': 67, 'HU': 106, 'HK': 34, 'HN': 22, 'AD': 35, 'PR': 40, 'PT': 26, 'PY': 51, 'PA': 60, 'PE': 20, 'PK': 84, 'PH': 91, 'PL': 30, 'EE': 52, 'EG': 74, 'ZA': 75, 'EC': 19, 'AL': 25, 'VN': 90, 'ET': 54, 'ZW': 114, 'ES': 0, 'MD': 68, 'UY': 77, 'MM': 94, 'ML': 104, 'US': 15, 'MT': 118, 'MR': 48, 'UA': 49, 'MX': 16, 'IL': 42, 'FR': 8, 'MA': 38, 'FI': 23, 'NI': 33, 'NL': 7, 'NO': 46, 'NG': 83, 'NZ': 93, 'CI': 57, 'CH': 3, 'CO': 21, 'CN': 28, 'CM': 55, 'CL': 4, 'CA': 2, 'CG': 101, 'CF': 109, 'CD': 112, 'CZ': 36, 'CR': 32, 'CU': 72, 'KE': 65, 'KH': 95, 'SV': 53, 'SK': 69, 'KR': 87, 'KW': 92, 'SN': 47, 'SL': 97, 'KZ': 111, 'SA': 56, 'SG': 66, 'SE': 24, 'DO': 11, 'DJ': 115, 'DK': 76, 'DE': 10, 'DZ': 80, 'MK': 105, -99: 1, 'LB': 81, 'TW': 29, 'TR': 70, 'TN': 85, 'LT': 103, 'LU': 59, 'TH': 79, 'TG': 86, 'LY': 108, 'AE': 37, 'VE': 14, 'IS': 107, 'IT': 18, 'AO': 71, 'AR': 13, 'AU': 63, 'AT': 6, 'IN': 31, 'IE': 5, 'QA': 58, 'MZ': 27},
}
cat_cols = list(mapping_dict.keys())

target_cols = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']
target_cols = target_cols[2:]

def getTarget(row):
	tlist = []
	for col in target_cols:
		if row[col].strip() in ['', 'NA']:
			target = 0
		else:
			target = int(float(row[col]))
		tlist.append(target)
	return tlist

def getIndex(row, col):
	val = row[col].strip()
	if val not in ['','NA']:
		ind = mapping_dict[col][val]
	else:
		ind = mapping_dict[col][-99]
	return ind

def getAge(row):
	mean_age = 40.
	min_age = 20.
	max_age = 90.
	range_age = max_age - min_age
	age = row['age'].strip()
	if age == 'NA' or age == '':
		age = mean_age
	else:
		age = float(age)
		if age < min_age:
			age = min_age
		elif age > max_age:
			age = max_age
	return round( (age - min_age) / range_age, 4)

def getCustSeniority(row):
	min_value = 0.
	max_value = 256.
	range_value = max_value - min_value
	missing_value = 0.
	cust_seniority = row['antiguedad'].strip()
	if cust_seniority == 'NA' or cust_seniority == '':
		cust_seniority = missing_value
	else:
		cust_seniority = float(cust_seniority)
		if cust_seniority < min_value:
			cust_seniority = min_value
		elif cust_seniority > max_value:
			cust_seniority = max_value
	return round((cust_seniority-min_value) / range_value, 4)

def getRent(row):
	min_value = 0.
	max_value = 1500000.
	range_value = max_value - min_value
	missing_value = 101850.
	rent = row['renta'].strip()
	if rent == 'NA' or rent == '':
		rent = missing_value
	else:
		rent = float(rent)
		if rent < min_value:
			rent = min_value
		elif rent > max_value:
			rent = max_value
	return round((rent-min_value) / range_value, 6)

def processRow(toPredict, cust_dict):
	x_vars_list = []
	y_vars_list = []

	for row in toPredict:
		if row['fecha_dato'] not in ['2015-05-28', '2015-06-28', '2016-05-28', '2016-06-28']:
			return x_vars_list

		cust_id = int(row['ncodpers'])
		if row['fecha_dato'] in ['2015-05-28', '2016-05-28']:	
			target_list = getTarget(row)
			cust_dict[cust_id] =  target_list[:]
			return x_vars_list

		x_vars = []
		for col in cat_cols:
			x_vars.append( getIndex(row, col) )
		x_vars.append( getAge(row) )
		x_vars.append( getCustSeniority(row) )
		x_vars.append( getRent(row) )

		if row['fecha_dato'] == '2016-06-28':
			prev_target_list = cust_dict.get(cust_id, [0]*22)
			x_vars_list.append(x_vars + prev_target_list)
		elif row['fecha_dato'] == '2015-06-28':
			prev_target_list = cust_dict.get(cust_id, [0]*22)
			target_list = getTarget(row)
			new_products = [max(x1 - x2,0) for (x1, x2) in zip(target_list, prev_target_list)]
			if sum(new_products) > 0:
				for ind, prod in enumerate(new_products):
					if prod>0:
						assert len(prev_target_list) == 22
						x_vars_list.append(x_vars+prev_target_list)
						y_vars_list.append(ind)

	return x_vars_list

def predict(model, helper_data, toPredict):
	global target_cols
	x_vars_list = processRow(toPredict, helper_data)
	test_X = np.array(x_vars_list)
	del x_vars_list
	xgtest = xgb.DMatrix(test_X)
	preds = model.predict(xgtest)
	del test_X, xgtest

	target_cols = np.array(target_cols)
	preds = np.argsort(preds, axis=1)
	preds = np.fliplr(preds)[:,:7]
	test_id = [d['ncodpers'] for d in toPredict]
	final_preds = [" ".join(list(target_cols[pred])) for pred in preds]
	out_df = {'ncodpers':test_id, 'added_products':final_preds}
	return out_df

def getModelandHelperData():
	model = xgb.Booster()
	model.load_model("product_recommendation_model.json")
	with open('cust_dict_save.pkl', 'rb') as f:
		cust_dict = pickle.load(f)
		return model, cust_dict
