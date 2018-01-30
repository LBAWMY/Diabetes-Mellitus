#coding=utf-8

import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import h5py
import tensorflow as tf
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

draft_param_feature = list(origin_train.columns)
draft_param_feature.remove('血糖')
draft_param_feature.remove('id')

print("all",origin_train.shape)
train = origin_train[origin_train['血糖']<=19]
highValue = np.percentile(train['血糖'],98)
print('<'+str(19),train.shape)
print('<'+str(highValue),train[train['血糖']>=highValue].shape)
if OFFLINE_BUTTON == 1:
    origin_test = pd.read_csv('../data/feature/pure_test_feat.csv')
    test_ori_value = origin_test['血糖']
elif OFFLINE_BUTTON == 2:
	origin_test = pd.read_csv('../data/feature/offline_test_feat.csv')
	test_ori_value = origin_test['血糖']
else:
    origin_test = pd.read_csv('../data/feature/all_test_feat_Bx.csv')
    # origin_test = pd.read_csv('../data/feature/all_test_feat.csv')
    test_ori_value = 1
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

# loading a series of model
xgb_p50_model = xgb.Booster({'nthread':8})
xgb_p50_model.load_model('../model/classification/xgb_p50.model')

xgb_p98_model = xgb.Booster({'nthread':8})
xgb_p98_model.load_model('../model/classification/xgb_p98.model')

xgb_p85_model = xgb.Booster({'nthread':8})
xgb_p85_model.load_model('../model/classification/xgb_p75.model')

#predict test set
test_y = xgb_p50_model.predict(dtest)
test_result = pd.DataFrame(test_id,columns=["id"])
test_result["hp50_prob"] = test_y
test_result["true suger"] = test_ori_value
# regression predict result
# predict_result = pd.read_csv('../regression/predict_result1.csv')
# xgb_result = pd.read_csv('../regression/xgb_preds1.csv')
# rf_result = pd.read_csv('../regression/rf_preds1.csv')
# lgdm_result = pd.read_csv('../regression/lgdm_preds1.csv')
# dart_result = pd.read_csv('../regression/dart_preds1.csv')
# test_result["reg_result"] = predict_result['score_all']
# test_result["rf_result"] = rf_result['score_all']
# test_result["xgb_result"] = xgb_result['score_all']
# test_result["lgdm_result"] = lgdm_result['score_all']
# test_result["dart_result"] = dart_result['score_all']
feat_prob_model1 = pd.concat([test_result,test_x],axis=1)

feat_prob_model1 = feat_prob_model1.sort_values(by='hp50_prob', axis=0, ascending=False)
topk = int(len(test)*0.5)
topk_prob = feat_prob_model1[0:topk]
# test_result.to_csv("hp95_prob1.csv",index=None,encoding='utf-8')
topk_prob.to_csv("topk1.csv",index=None,encoding='utf-8')

# second model
test_x2 = topk_prob[predictors]
dtest2 = xgb.DMatrix(test_x2)

test_y2 = xgb_p98_model.predict(dtest2)
test_id2 = topk_prob['id']
test_result2 = pd.DataFrame(test_id2,columns=["id"])
test_result2["true suger"] = test_ori_value
test_result2["hp98_prob"] = test_y2
feat_prob_model2 = pd.concat([test_result2,test_x2],axis=1)

feat_prob_model2 = feat_prob_model2.sort_values(by='hp98_prob', axis=0, ascending=False)
topkk = int(len(test_x2))
topkk_prob = feat_prob_model2[0:topkk]
# test_result.to_csv("hp95_prob1.csv",index=None,encoding='utf-8')
topkk_prob.to_csv("topk2.csv",index=None,encoding='utf-8')

# # third model
# test_x3 = topkk_prob[predictors]
# dtest3 = xgb.DMatrix(test_x3)
#
# test_y3 = xgb_p85_model.predict(dtest3)
# test_id3 = topkk_prob['id']
# test_result3 = pd.DataFrame(test_id3,columns=["id"])
# test_result3["true suger"] = test_ori_value
# test_result3["hp85_prob"] = test_y3
# feat_prob_model3 = pd.concat([test_result3,test_x3],axis=1)
#
# feat_prob_model3 = feat_prob_model3.sort_values(by='hp85_prob', axis=0, ascending=False)
# topkkk = int(len(test_x3)*0.6)
# topkkk_prob = feat_prob_model3[0:topkkk]
# # test_result.to_csv("hp95_prob1.csv",index=None,encoding='utf-8')
# topkkk_prob.to_csv("topkkk.csv",index=None,encoding='utf-8')


### rf model
rf_p50_model = joblib.load('../model/classification/rf_p98.pkl')
test_x3 = topkk_prob[predictors]
test_y3 = rf_p50_model.predict_proba(test_x3.values)[:,1]
# rf_p50_model = joblib.load('../model/rf_reg.pkl')
# test_x3 = topkk_prob[predictors]
# test_y3 = rf_p50_model.predict(test_x3.values)

test_id3 = topkk_prob['id']
test_result3 = pd.DataFrame(test_id3,columns=["id"])
test_result3["true suger"] = test_ori_value
test_result3["hp50_prob_rf"] = test_y3
feat_prob_model3 = pd.concat([test_result3,test_x3],axis=1)

feat_prob_model3 = feat_prob_model3.sort_values(by='hp50_prob_rf', axis=0, ascending=False)
topkkk = int(len(test_x3))
topkkk_prob = feat_prob_model3[0:topkkk]
# test_result.to_csv("hp95_prob1.csv",index=None,encoding='utf-8')
topkkk_prob.to_csv("topk3.csv",index=None,encoding='utf-8')

# load dart model
dart_reg = xgb.Booster({'nthread':8})
dart_reg.load_model('../model/dart_reg.model')

test_x4 = topkkk_prob[predictors]
dtest4 = xgb.DMatrix(test_x4)

test_y4 = dart_reg.predict(dtest4)
test_id4 = topkkk_prob['id']
test_result4 = pd.DataFrame(test_id4,columns=["id"])
test_result4["true suger"] = test_ori_value
test_result4["hp50_prob"] = test_y4

feat_prob_model4 = pd.concat([test_result4,test_x4],axis=1)

feat_prob_model4 = feat_prob_model4.sort_values(by='hp50_prob', axis=0, ascending=False)
topkkkk = int(len(test_x4))
topkkkk_prob = feat_prob_model4[0:topkkkk]
# test_result.to_csv("hp95_prob1.csv",index=None,encoding='utf-8')
topkkkk_prob.to_csv("topk4.csv",index=None,encoding='utf-8')

### mlp model
file = h5py.File('../model/classification/MLP_model_P60.h5', 'r')

test_x5 = topkkkk_prob[predictors]
# trainlabel = train_y.apply(lambda x: np.array([1,0]) if x==1 else np.array([0,1]))
# test_y5 = train_y.values

# Standardize data
scaler_data_max = file['scaler_data_max']
scaler_data_min = file['scaler_data_min']
scaler_data_scaler = file['scaler_data_scaler']

std_test_x5 = (test_x5-scaler_data_min)*scaler_data_scaler

def fully_connected(input_layer, weights, biases):
    layer = tf.add(tf.matmul(input_layer, weights), biases)
    return(tf.nn.relu(layer))

x_data = tf.placeholder(shape=[None, std_test_x5.shape[1]], dtype=tf.float32)

# create hidden layer
weight_1 = file['weight_1'][:]
bias_1 = file['bias_1'][:]
layer_1 = fully_connected(x_data, weight_1, bias_1)

weight_2 = file['weight_2'][:]
bias_2 = file['bias_2'][:]
layer_2 = fully_connected(layer_1, weight_2, bias_2)

weight_3 = file['weight_3'][:]
bias_3 = file['bias_3'][:]
layer_3 = fully_connected(layer_2, weight_3, bias_3)

# create output layer
weight_4 = file['weight_4'][:]
bias_4 = file['bias_4'][:]
final_output = tf.matmul(layer_3, weight_4) + bias_4

# Create a Session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

test_y5 = sess.run(tf.nn.softmax(final_output), feed_dict={x_data: std_test_x5})[:,1]
test_id5 = topkkkk_prob['id']
test_result5 = pd.DataFrame(test_id5,columns=["id"])
test_result5["true suger"] = test_ori_value
test_result5["hp50_prob"] = test_y5

feat_prob_model5 = pd.concat([test_result5,test_x5],axis=1)

feat_prob_model5 = feat_prob_model5.sort_values(by='hp50_prob', axis=0, ascending=False)
topkkkkk = int(len(test_x5)*0.30)
topkkkkk_prob = feat_prob_model5[0:topkkkkk]
# test_result.to_csv("hp95_prob1.csv",index=None,encoding='utf-8')
topkkkkk_prob.to_csv("topk5.csv",index=None,encoding='utf-8')

