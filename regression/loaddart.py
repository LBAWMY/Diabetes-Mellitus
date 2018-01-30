#coding=utf-8

import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from config import *
import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--offline_test', type=int, default=0)

args = parser.parse_args()
OFFLINE_BUTTON = args.offline_test

#loading feature file and merge
if OFFLINE_BUTTON == 1:
    origin_train = pd.read_csv('../data/feature/pure_train_feat.csv')
elif OFFLINE_BUTTON == 2:
	origin_train = pd.read_csv('../data/feature/offline_train_feat1.csv')
else:
    origin_train = pd.read_csv('../data/feature/all_train_feat.csv')

train_param_feature = list(origin_train.columns)
train_param_feature.remove('id')
train_param_feature.remove('血糖')

print("all",origin_train.shape)
train = origin_train[origin_train['血糖']<=19]
print('<=19',train.shape)

if OFFLINE_BUTTON == 1:
    origin_test = pd.read_csv('../data/feature/pure_test_feat.csv')
elif OFFLINE_BUTTON == 2:
    origin_test = pd.read_csv('../data/feature/offline_test_feat1.csv')
else:
    origin_test = pd.read_csv('../data/feature/all_test_feat_B.csv')

#drop nan columns
origin_test['血糖'] = -999
train_test = pd.concat([train,origin_test],axis=0)
train_test.dropna(axis='columns',how='all',inplace=True)
# delete specify params and the remaining params without sugar
train_test.drop(del_params,axis=1,inplace=True)
train = train_test[train_test['血糖']!=-999]
test = train_test[train_test['血糖']==-999]

train_y = train['血糖']
train_x = train.drop(['血糖','id'],axis=1)
test_id = test['id']
test_x = test.drop(['id','血糖'],axis=1)

# #'p1_p2' is a new feature
train_x['acid_soda'] = train_x['嗜酸细胞%'] - train_x['嗜碱细胞%']
test_x['acid_soda'] = test_x['嗜酸细胞%'] - test_x['嗜碱细胞%']
predictors = train_x.columns

#training xgboost
dtrain = xgb.DMatrix(train_x,label=train_y)
dtest = xgb.DMatrix(test_x)


watchlist = [(dtrain,'train')]

dart_reg = xgb.Booster({'nthread':8})
dart_reg.load_model('../model/dart_reg.model')

#predict test set
test_y = dart_reg.predict(dtest)
test_result = pd.DataFrame(test_id,columns=["id"])
test_result["score_all"] = test_y
test_result.to_csv("dart_preds1.csv",index=None,encoding='utf-8')

