# coding=utf-8
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier
###
# load data
###
origin_train = pd.read_csv('../data/feature/offline_train_feat.csv')

draft_param_feature = list(origin_train.columns)
draft_param_feature.remove('血糖')
draft_param_feature.remove('id')

print("all",origin_train.shape)
train = origin_train[origin_train['血糖']<=19]
highValue = np.percentile(train['血糖'],95)
# highValue = np.percentile(train['血糖'],80)
print('<'+str(19),train.shape)
print('<'+str(highValue),train[train['血糖']>=highValue].shape)

origin_test = pd.read_csv('../data/feature/offline_test_feat.csv')
test_ori_value = origin_test['血糖']# 保存一下， 统一处理完后恢复
origin_test['血糖'] = -999
train_test = pd.concat([train,origin_test],axis=0)
train_test.dropna(axis='columns',how='all',inplace=True)
train = train_test[train_test['血糖']!=-999]
test = train_test[train_test['血糖']==-999]
test['血糖'] = test_ori_value
#generate label for each product_no
train['血糖'] = train['血糖'].apply(lambda x: 1 if x>=highValue else 0)
train_y = train['血糖']
train_x = train.drop(['id', '血糖', '乙肝表面抗原','乙肝表面抗体','乙肝e抗原','乙肝e抗体','乙肝核心抗体'],axis=1)# drop some high vacancy param
test_id = test.id
test['血糖'] = test['血糖'].apply(lambda x: 1 if x>=highValue else 0)
test_y = test['血糖']
test_x = test.drop(['id', '血糖', '乙肝表面抗原','乙肝表面抗体','乙肝e抗原','乙肝e抗体','乙肝核心抗体'],axis=1)

###
# select classification
###

# # 1.rf
rf_best1 = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=10,
                                 min_samples_leaf=35, max_features='sqrt', oob_score=True, random_state=10)
rf_best2= RandomForestClassifier(n_estimators=78, max_depth=15, min_samples_split=10,
                                 min_samples_leaf=35, max_features=23, oob_score=True, random_state=10)
rf_best3 = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=10,
                                 min_samples_leaf=35, max_features='sqrt', oob_score=True, random_state=10)
# rf_best4= RandomForestClassifier(n_estimators=78, max_depth=15, min_samples_split=10,
#                                  min_samples_leaf=35, max_features=23, oob_score=True, random_state=10)
# rf_best5 = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=10,
#                                  min_samples_leaf=35, max_features='sqrt', oob_score=True, random_state=10)
# rf_best6= RandomForestClassifier(n_estimators=78, max_depth=15, min_samples_split=10,
#                                  min_samples_leaf=35, max_features=23, oob_score=True, random_state=10)
#
# # 2.svc
# svm_clf = SVC()
#
# # 3.xgb
# xgb_params={
#     'booster':'gbtree',
# 	'objective': 'binary:logistic',
# 	'scale_pos_weight':float(len(train_y)-sum(train_y))/sum(train_y),
# 	'eval_metric': 'auc',
# 	'max_depth':6,
# 	'reg_lambda':70,
# 	'subsample':0.6,
# 	'colsample_bytree':0.6,
# 	'learning_rate': 0.001,
# 	'seed':1024,
# 	'nthread':-1
# }

# xgb1_params={
#     'booster':'gbtree',
# 	'n_estimators': 1000,
# 	'objective': 'binary:logistic',
# 	'eval_metric': 'auc',
# 	'max_depth':4,
# 	'reg_lambda':0.8,
# 	'subsample':0.7,
# 	'colsample_bytree':0.7,
# 	'learning_rate': 0.6,
# 	'seed':1024,
# 	'nthread': 16
# }
# xgb2_params={
#     'booster':'gbtree',
# 	'n_estimators': 1000,
# 	'objective': 'binary:logistic',
# 	'eval_metric': 'auc',
# 	'max_depth':4,
# 	'reg_lambda':0.8,
# 	'subsample':0.7,
# 	'colsample_bytree':0.7,
# 	'learning_rate': 0.6,
# 	'seed':1025,
# 	'nthread': 16
# }
# xgb3_params={
#     'booster':'gbtree',
# 	'n_estimators': 1000,
# 	'objective': 'binary:logistic',
# 	'eval_metric': 'auc',
# 	'max_depth':4,
# 	'reg_lambda':0.8,
# 	'subsample':0.7,
# 	'colsample_bytree':0.7,
# 	'learning_rate': 0.6,
# 	'seed':1028,
# 	'nthread': 16
# }
# xgb4_params={
#     'booster':'gbtree',
# 	'n_estimators': 1000,
# 	'objective': 'binary:logistic',
# 	'eval_metric': 'auc',
# 	'max_depth':4,
# 	'reg_lambda':0.8,
# 	'subsample':0.7,
# 	'colsample_bytree':0.7,
# 	'learning_rate': 0.6,
# 	'seed':1000,
# 	'nthread': 16
# }
# xgb5_params={
#     'booster':'gbtree',
# 	'n_estimators': 1000,
# 	'objective': 'binary:logistic',
# 	'eval_metric': 'auc',
# 	'max_depth':4,
# 	'reg_lambda':0.8,
# 	'subsample':0.7,
# 	'colsample_bytree':0.7,
# 	'learning_rate': 0.6,
# 	'seed':1010,
# 	'nthread': 16
# }


# xgb1_clf = XGBClassifier(**xgb1_params)
# xgb2_clf = XGBClassifier(**xgb2_params)
# xgb3_clf = XGBClassifier(**xgb3_params)
# xgb4_clf = XGBClassifier(**xgb4_params)
# xgb5_clf = XGBClassifier(**xgb5_params)



svc_clf = SVC(probability=True)
rnd_clf = RandomForestClassifier()

# voting_clf = VotingClassifier(
#         estimators = [ ('rf1', rf_best1), ('rf2', rf_best2),('rf3', rf_best3),('rf4', rf_best4),('rf5', rf_best5),('rf6', rf_best6)],
#         voting='soft'
# )
voting_clf = VotingClassifier(
        estimators = [ ('rf1', rf_best1), ('rf2', rf_best2),('rf3', rf_best3)],
        voting='soft'
)
# voting_clf = VotingClassifier(
#         estimators = [ ('rf', rnd_clf),('svc', svc_clf)],
#         voting='soft'
# )

voting_clf.fit(train_x, train_y)
voting_hp95 = voting_clf.predict_proba(test_x)
voting_hp95 = voting_hp95[:, 1]
test_result = pd.DataFrame(test_id, columns=["id"])
test_result["blood suger"] = test_ori_value
test_result["label"] = test_y
test_result["voting_hp95_prob"] = voting_hp95

xgb_hp95_prob = test_result.sort_values(by='voting_hp95_prob', axis=0, ascending=False)

top10_prob = xgb_hp95_prob[0:11]
print(top10_prob)

xgb_hp95_prob.to_csv("./output/voting_hp95_fr+xgb.csv", index=None, encoding='utf-8')
# xgb_blood = test_result.sort_values(by='blood suger', axis=0, ascending=False)
# xgb_blood.to_csv("./output/voting_blood_null.csv", index=None, encoding='utf-8')

