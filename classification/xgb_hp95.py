#coding=utf-8

import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd
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
elif OFFLINE_BUTTON == 2:
    origin_train = pd.read_csv('../data/feature/offline_train_feat.csv')
else:
    origin_train = pd.read_csv('../data/feature/all_train_feat.csv')

draft_param_feature = list(origin_train.columns)
draft_param_feature.remove('血糖')
draft_param_feature.remove('id')

print("all",origin_train.shape)
train = origin_train[origin_train['血糖']<=19]
highValue = np.percentile(train['血糖'],50)
print('<'+str(19),train.shape)
print('<'+str(highValue),train[train['血糖']>=highValue].shape)
if OFFLINE_BUTTON == 1:
    origin_test = pd.read_csv('../data/feature/pure_test_feat.csv')
    test_ori_value = origin_test['血糖']
elif OFFLINE_BUTTON == 2:
	origin_test = pd.read_csv('../data/feature/offline_test_feat.csv')
	test_ori_value = origin_test['血糖']
else:
    origin_test = pd.read_csv('../data/feature/all_test_feat.csv')
origin_test['血糖'] = -999
train_test = pd.concat([train,origin_test],axis=0)
train_test.dropna(axis='columns',how='all',inplace=True)
# delete specify params and the remaining params without sugar
train_test.drop(del_params,axis=1,inplace=True)
train = train_test[train_test['血糖']!=-999]
test = train_test[train_test['血糖']==-999]

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
	'max_depth':4,
	'lambda':0.2,
	'subsample':0.6,
	'colsample_bytree':0.7,
	'eta': 0.001,
	'seed':1024,
	'nthread':8
}

#通过cv找最佳的nround
cv_log = xgb.cv(params,dtrain,num_boost_round=250,nfold=5,metrics='auc',early_stopping_rounds=50,seed=1024)#num_boost_round=25000
bst_auc= cv_log['test-auc-mean'].max()
cv_log['nb'] = cv_log.index
cv_log.index = cv_log['test-auc-mean']
bst_nb = cv_log.nb.to_dict()[bst_auc]

#train
watchlist = [(dtrain,'train')]
model = xgb.train(params,dtrain,num_boost_round=bst_nb+50,evals=watchlist)

#predict test set
test_y = model.predict(dtest)
test_result = pd.DataFrame(test_id,columns=["id"])
test_result["hp95_prob"] = test_y
test_result["true suger"] = test_ori_value
# regression predict result
predict_result = pd.read_csv('../regression/predict_result.csv')
xgb_result = pd.read_csv('../regression/xgb_preds.csv')
rf_result = pd.read_csv('../regression/rf_preds.csv')
lgdm_result = pd.read_csv('../regression/lgdm_preds.csv')
dart_result = pd.read_csv('../regression/dart_preds.csv')
test_result["reg_result"] = predict_result['score_all']
test_result["rf_result"] = rf_result['score_all']
test_result["xgb_result"] = xgb_result['score_all']
test_result["lgdm_result"] = lgdm_result['score_all']
test_result["dart_result"] = dart_result['score_all']
test_result.to_csv("hp95_prob.csv",index=None,encoding='utf-8')

print('high value:  '+str(highValue))
print(bst_nb,bst_auc)

#Plot feature importance
myfont = FontProperties(fname="simhei.ttf")
fig,ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model,height=0.8,ax=ax)
ax.set_yticklabels(predictors,rotation='horizontal',fontproperties=myfont)
# plt.show()
