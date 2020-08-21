import pickle
import pystan as stan
import os

def save_pickle(obj, file):
	with open(file,'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
	print('Dumped pickle to {}'.format(file))
	
def load_pickle(file):
	with open(file,'rb') as f:
		return pickle.load(f)
		
script_dir = os.path.dirname(os.path.realpath(__file__))		
		
# Load and compile Stan models
model_dict = {'drt_dZ_StanModel.pkl':'drt_dZ_modelcode.txt',
	'drt_dZ_outliers_StanModel.pkl':'drt_dZ_outliers_modelcode.txt',
	'drt_dZ_pos_StanModel.pkl':'drt_dZ_pos_modelcode.txt',
	'drt_dZ_pos_outliers_StanModel.pkl':'drt_dZ_pos_outliers_modelcode.txt',
	'drt_no-dZ_StanModel.pkl':'drt_no-dZ_modelcode.txt',
	'drt_no-dZ_outliers_StanModel.pkl':'drt_no-dZ_outliers_modelcode.txt',
	'drt_no-dZ_pos_StanModel.pkl':'drt_no-dZ_pos_modelcode.txt',
	'drt_no-dZ_pos_outliers_StanModel.pkl':'drt_no-dZ_pos_outliers_modelcode.txt',
	}


for pkl, code_file in model_dict.items():	
	if not os.path.exists(os.path.join(script_dir,'stan_model_files',pkl)):
		with open(os.path.join(script_dir,'stan_model_files',code_file)) as f:
			model_code = f.read()
		print(f'Compiling {code_file}...')
		model = stan.StanModel(model_code=model_code)
		save_pickle(model,os.path.join(script_dir,'stan_model_files',pkl))