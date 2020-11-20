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
model_dict = {
	# Pure series models
	'Series_StanModel.pkl':'Series_modelcode.txt',
	'Series_outliers_StanModel.pkl':'Series_outliers_modelcode.txt',
	'Series_pos_StanModel.pkl':'Series_pos_modelcode.txt',
	'Series_pos_outliers_StanModel.pkl':'Series_pos_outliers_modelcode.txt',
	# Pure parallel models
	'Parallel_StanModel.pkl':'Parallel_modelcode.txt',
	'Parallel_outliers_StanModel.pkl':'Parallel_outliers_modelcode.txt',
	'Parallel_SA_StanModel.pkl':'Parallel_SA_modelcode.txt',
	'Parallel_fitY_StanModel.pkl':'Parallel_fitY_modelcode.txt',
	'Parallel_fitY_SA_StanModel.pkl':'Parallel_fitY_SA_modelcode.txt',
	# Mixed series-parallel models
	'Series-Parallel_StanModel.pkl':'Series-Parallel_modelcode.txt',
	'Series-Parallel_pos_StanModel.pkl':'Series-Parallel_pos_modelcode.txt',
	'Series-Parallel_pos_outliers_StanModel.pkl':'Series-Parallel_pos_outliers_modelcode.txt',
	'Series-Parallel_outliers_StanModel.pkl':'Series-Parallel_outliers_modelcode.txt',
	'Series-2Parallel_StanModel.pkl':'Series-2Parallel_modelcode.txt',
	'Series-2Parallel_pos_StanModel.pkl':'Series-2Parallel_pos_modelcode.txt',
	'MultiDist_StanModel.pkl':'MultiDist_modelcode.txt'
	# 'Series-Parallel_pos_StanModel_qsum.pkl':'Series-Parallel_pos_modelcode_qsum.txt'
	}
     
for pkl, code_file in model_dict.items():	
	if not os.path.exists(os.path.join(script_dir,'stan_model_files',pkl)):
		with open(os.path.join(script_dir,'stan_model_files',code_file)) as f:
			model_code = f.read()
		print(f'Compiling {code_file}...')
		model = stan.StanModel(model_code=model_code)
		save_pickle(model,os.path.join(script_dir,'stan_model_files',pkl))