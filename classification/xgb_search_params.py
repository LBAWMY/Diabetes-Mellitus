#coding=utf-8

import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd
import copy
import numpy as np
from config import *
import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--offline_test', type=int, default=1)

args = parser.parse_args()
OFFLINE_BUTTON = args.offline_test
#loading feature file and merge
if OFFLINE_BUTTON == 1:
    origin_train = pd.read_csv('../data/feature/pure_train_feat.csv')
else:
    origin_train = pd.read_csv('../data/feature/all_train_feat.csv')

draft_param_feature = list(origin_train.columns)
draft_param_feature.remove('血糖')
draft_param_feature.remove('id')

print("all",origin_train.shape)
train = origin_train[origin_train['血糖']<=19]
highValue = np.percentile(train['血糖'],95)
print('<'+str(19),train.shape)
print('<'+str(highValue),train[train['血糖']>=highValue].shape)
if OFFLINE_BUTTON == 1:
    origin_test = pd.read_csv('../data/feature/pure_test_feat.csv')
else:
    origin_test = pd.read_csv('../data/feature/all_test_feat.csv')
test_y = origin_test['血糖']
origin_test['血糖'] = -999
train_test = pd.concat([train,origin_test],axis=0)
train_test.dropna(axis='columns',how='all',inplace=True)
# delete specify params and the remaining params without sugar
train_test.drop(del_params,axis=1,inplace=True)
train = train_test[train_test['血糖']!=-999]
test = train_test[train_test['血糖']==-999]

# regression predict result
predict_result = pd.read_csv('../regression/predict_result.csv')
reg_result_y = predict_result['score_all']

#generate label for each product_no
train['血糖'] = train['血糖'].apply(lambda x: 1 if x>=highValue else 0)
train_y = train['血糖']
train_x = train.drop(['血糖','id'],axis=1)
test_id = test.id
test_x = test.drop(['id','血糖'],axis=1)

# #'p1_p2' is a new feature
train_x['acid-soda'] = train_x['嗜酸细胞%'] - train_x['嗜碱细胞%']
test_x['acid-soda'] = test_x['嗜酸细胞%'] - test_x['嗜碱细胞%']
train_x['acid+soda'] = train_x['嗜酸细胞%'] + train_x['嗜碱细胞%']
test_x['acid+soda'] = test_x['嗜酸细胞%'] + test_x['嗜碱细胞%']
predictors = train_x.columns

#training xgboost
dtrain = xgb.DMatrix(train_x,label=train_y)
dtest = xgb.DMatrix(test_x)

params={
    'booster':'gbtree',
	'objective': 'binary:logistic',
	'scale_pos_weight':float(len(train_y)-sum(train_y))/sum(train_y),
	'eval_metric': 'auc',
	'max_depth':6,
	'lambda':70,
	'subsample':0.6,
	'colsample_bytree':0.6,
	'eta': 0.001,
	'seed':1024,
	'nthread':8
}

#通过cv找最佳的nround
cv_log = xgb.cv(params,dtrain,num_boost_round=25000,nfold=5,metrics='auc',early_stopping_rounds=50,seed=1024)#num_boost_round=25000
bst_auc= cv_log['test-auc-mean'].max()
cv_log['nb'] = cv_log.index
cv_log.index = cv_log['test-auc-mean']
bst_nb = cv_log.nb.to_dict()[bst_auc]

#xgboost parameters
params_set = {'max_depth':range(4, 7),
			  'lamda':[0.2,0.4,0.6,0.8,1],
			  'eta':[0.001,0.01,0.1],
			  'colsample_bytree':[0.7,0.8],
			  'learning_rate':[0.05, 0.1],
			  'seed':range(1,5)
			  }
# params_set = {'max_depth':range(4, 7),
# 			  'lamda':range(1, 20, 10),
# 			  'eta':[0.001,0.01],
# 			  'colsample_bytree':[0.7],
# 			  'learning_rate':[0.05],
# 			  'seed':range(1,2)
# 			  }
constant_set = {'booster':'gbtree',
			  'objective': 'binary:logistic',
	          'scale_pos_weight':float(len(train_y)-sum(train_y))/sum(train_y),
	          'eval_metric': 'auc'}


def combine_xgb_params(const_dict, variable_dict):
	all_xgb_params = []
	keys = list(variable_dict.keys())
	index = range(len(keys))
	for param0 in variable_dict[keys[index[0]]]:
		xgb_params = copy.deepcopy(const_dict)
		xgb_params[keys[index[0]]] = param0
		for param1 in variable_dict[keys[index[1]]]:
			xgb_params = copy.deepcopy(xgb_params)
			xgb_params[keys[index[1]]] = param1
			for param2 in variable_dict[keys[index[2]]]:
				xgb_params = copy.deepcopy(xgb_params)
				xgb_params[keys[index[2]]] = param2
				for param3 in variable_dict[keys[index[3]]]:
					xgb_params = copy.deepcopy(xgb_params)
					xgb_params[keys[index[3]]] = param3
					for param4 in variable_dict[keys[index[4]]]:
						xgb_params = copy.deepcopy(xgb_params)
						xgb_params[keys[index[4]]] = param4
						for param5 in variable_dict[keys[index[5]]]:
							xgb_params = copy.deepcopy(xgb_params)
							xgb_params[keys[index[5]]] = param5
							all_xgb_params.append(xgb_params)
	return all_xgb_params

def gridSearch(all_xgb_params,dtrain,dtest,test_y,bst_nb,reg_result_y,highValue):
	# record = []
	# log_dataframe = pd.DataFrame([])
	flag = 0
	for params in all_xgb_params:
		log = {}
		log = copy.deepcopy(params)
		model = xgb.train(params, dtrain, num_boost_round=bst_nb + 50)
		# predict test set
		predict_test_y = model.predict(dtest)
		test_result = pd.DataFrame(test_id, columns=["id"])
		test_result["xgb_hp95_prob"] = predict_test_y
		test_result["true sugar"] = test_y
		test_result["regress sugar"] = reg_result_y

		xgb_hp95_results = test_result.sort_values(by='xgb_hp95_prob', axis=0, ascending=False)

		# top5
		top5_prob = xgb_hp95_results['xgb_hp95_prob'].values[0:5]
		top5_reg_value = xgb_hp95_results['regress sugar'].values[0:5]
		top5_true_value = xgb_hp95_results['true sugar'].values[0:5]

		value_mean5 = np.mean(top5_true_value)
		top5_improve = np.mean(np.square(top5_true_value-top5_reg_value))/2.0 - np.mean(np.square(top5_true_value-highValue))/2.0

		# top10
		top10_prob = xgb_hp95_results['xgb_hp95_prob'].values[0:10]
		top10_reg_value = xgb_hp95_results['regress sugar'].values[0:10]
		top10_true_value = xgb_hp95_results['true sugar'].values[0:10]

		value_mean10 = np.mean(top10_true_value)
		top10_improve = np.mean(np.square(top10_true_value-top10_reg_value))/2.0 - np.mean(np.square(top10_true_value-highValue))/2.0

		# top15
		top15_prob = xgb_hp95_results['xgb_hp95_prob'].values[0:15]
		top15_reg_value = xgb_hp95_results['regress sugar'].values[0:15]
		top15_true_value = xgb_hp95_results['true sugar'].values[0:15]

		value_mean15 = np.mean(top15_true_value)
		top15_improve = np.mean(np.square(top15_true_value-top15_reg_value))/2.0 - np.mean(np.square(top15_true_value-highValue))/2.0

		log['value-mean5'] = value_mean5
		log['improve5'] = top5_improve
		log['value-mean10'] = value_mean10
		log['improve10'] = top10_improve
		log['value-mean15'] = value_mean15
		log['improve15'] = top15_improve

		# col = list(pd.DataFrame(list(log.items())).T.iloc[0])
		# key = pd.DataFrame(list(log.items())).T.iloc[1].T
		# if flag == 0:
		# 	log_dataframe = pd.concat([log_dataframe,pd.DataFrame(list(log.items())).T])
		# 	flag = 1
		# else:
		# 	log_dataframe = pd.concat([log_dataframe, pd.DataFrame(np.array(list(log.items()))[:,1].reshape(1,-1))])
		if flag == 0:
			log_dataframe = pd.DataFrame(np.array(list(log.items()))[:, 1].reshape(1, -1), columns=list(np.array(list(log.items()))[:, 0].reshape(1,-1)))
			flag = 1
		else:
			log_dataframe1 = pd.DataFrame(np.array(list(log.items()))[:, 1].reshape(1, -1),
										 columns=list(np.array(list(log.items()))[:, 0].reshape(1, -1)))
			log_dataframe = pd.concat([log_dataframe,log_dataframe1])
			# record.append(log)
	return log_dataframe


all_xgb_params = combine_xgb_params(constant_set, params_set)
grid_log = gridSearch(all_xgb_params,dtrain,dtest,test_y,bst_nb,reg_result_y=reg_result_y,highValue=highValue)
grid_log['improve5'] = grid_log['improve5'].astype(float)
grid_log['improve10'] = grid_log['improve10'].astype(float)
grid_log['improve15'] = grid_log['improve15'].astype(float)
grid_log = grid_log.reset_index(drop=True)
# improve5_frame = grid_log.sort_values(by='improve5', axis=0, ascending=False)
# top5 = improve5_frame['improve5'][0:5]
# improve10_frame = grid_log.sort_values(by='improve10', axis=0, ascending=False)
# top10 = improve10_frame['improve10'][0:10]
# improve15_frame = grid_log.sort_values(by='improve15', axis=0, ascending=False)
# top15 = improve15_frame['improve15'][0:15]
# print('**************TOP 5****************')
# print(top5)
# print('**************TOP 10****************')
# print(top10)
# print('**************TOP 15****************')
# print(top15)
grid_log.to_csv("./best-params.csv", index=None, encoding='utf-8')
# print(grid_log)

# watchlist = [(dtrain,'train')]
# max_depth = range(4,7)
# lamda = np.arange(1,100,10)
# subsample = [0.6,0.7,0.8]
# colsample_bytree = [0.6,0.7,0.8]
# eta = [0.001,0.01,0.1]
# learning_rate = [0.05, 0.1, 0.25, 0.5, 1.0]
# for params in all_xgb_params:
# 	model = xgb.train(params,dtrain,num_boost_round=bst_nb+50)

