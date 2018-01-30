import xgboost as xgb
import pandas as pd
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from dateutil.parser import parse
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pylab as plot
import time
import numpy as np
from config import *

import argparse
parser = argparse.ArgumentParser(description='manual to this script')
# 0为训练全部数据并测试线上结果，1为最小规模训练及测试线下结果，2为线下训练数据集及测试线下测试集
parser.add_argument('--offline_test', type=int, default=2)
args = parser.parse_args()
OFFLINE_BUTTON = args.offline_test
#loading feature file and merge
if OFFLINE_BUTTON == 1:
    origin_train = pd.read_csv('../data/feature/pure_train_feat.csv')
elif OFFLINE_BUTTON == 2:
    origin_train = pd.read_csv('../data/feature/offline_train_feat.csv')
    # origin_train = pd.read_csv('../data/feature/all_train_feat_aug5.csv')d_train_feat_aug.csv
    # origin_train = pd.read_csv('../data/feature/d_train_feat_aug.csv')
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
    origin_test = pd.read_csv('../data/feature/offline_test_feat.csv')
    # origin_test = pd.read_csv('../data/feature/all_test_feat.csv')
else:
    origin_test = pd.read_csv('../data/feature/all_test_feat_B.csv')

#drop nan columns
origin_test['血糖'] = -999
train_test = pd.concat([train,origin_test],axis=0)
train_test.dropna(axis='columns',how='all',inplace=True)
# delete specify params and the remaining params without sugar
train_test.drop(del_params,axis=1,inplace=True)
# predictors = [f for f in train_test.columns if f not in ['血糖','id']]

train = train_test[train_test['血糖']!=-999]
test = train_test[train_test['血糖']==-999]

train_y = train['血糖']
train_x = train.drop(['血糖','id'],axis=1)
test_id = test['id']
test_x = test.drop(['id','血糖'],axis=1)

# 人为尝试组合新特征
# train_x['acid-soda'] = train_x['嗜酸细胞%'] - train_x['嗜碱细胞%']
# test_x['acid-soda'] = test_x['嗜酸细胞%'] - test_x['嗜碱细胞%']
train_x['acid+soda'] = train_x['嗜酸细胞%'] + train_x['嗜碱细胞%']
test_x['acid+soda'] = test_x['嗜酸细胞%'] + test_x['嗜碱细胞%']
train_x['白细胞计数-红细胞计数'] = train_x['白细胞计数'] - train_x['红细胞计数']
test_x['白细胞计数-红细胞计数'] = test_x['白细胞计数'] - test_x['红细胞计数']
# train_x['白细胞计数+红细胞计数'] = train_x['白细胞计数'] + train_x['红细胞计数']
# test_x['白细胞计数+红细胞计数'] = test_x['白细胞计数'] + test_x['红细胞计数']
# train_x['甘油三酯**2'] = train_x['甘油三酯'] * train_x['甘油三酯']
# test_x['甘油三酯**2'] = test_x['甘油三酯'] * test_x['甘油三酯']
# train_x['高-低'] = train_x['高密度脂蛋白胆固醇'] - train_x['低密度脂蛋白胆固醇']
# test_x['高-低'] = test_x['高密度脂蛋白胆固醇'] - test_x['低密度脂蛋白胆固醇']
train_x['高+低'] = train_x['高密度脂蛋白胆固醇'] + train_x['低密度脂蛋白胆固醇']
test_x['高+低'] = test_x['高密度脂蛋白胆固醇'] + test_x['低密度脂蛋白胆固醇']
# train_x['尿酸-尿素'] = train_x['尿酸'] - train_x['尿素']
# test_x['尿酸-尿素'] = test_x['尿酸'] - test_x['尿素']
predictors = train_x.columns

def evalerror(pred, df):
    label = df.get_label().values.copy()
    score = mean_squared_error(label, pred) * 0.5
    return ('mse', score, False)


print('开始训练...')
params = {
    'learning_rate': 0.085,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
    'sub_feature': 0.80,
    'num_leaves': 60,
    'colsample_bytree': 0.70,
    'feature_fraction': 0.72,
    'min_data': 20,
    'min_hessian': 1,
    'verbose': 8,
}

print('开始CV 5折训练...')
scores = []
t0 = time.time()
train_preds = np.zeros(train.shape[0])
n_folder = 5
test_preds = np.zeros((test.shape[0], n_folder))
kf = KFold(len(train), n_folds=n_folder, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    print('第{}次训练...'.format(i))
    lgb_train1 = lgb.Dataset(train_x.iloc[train_index], train_y.iloc[train_index])
    lgb_train2 = lgb.Dataset(train_x.iloc[test_index], train_y.iloc[test_index])
    gbm = lgb.train(params,
                    lgb_train1,
                    num_boost_round=30000,
                    valid_sets=lgb_train2,
                    verbose_eval=300,
                    feval=evalerror,
                    early_stopping_rounds=100)
    feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
    train_preds[test_index] += gbm.predict(train_x.iloc[test_index])
    test_preds[:, i] = gbm.predict(test_x[predictors])
print('线下得分：    {}'.format(mean_squared_error(train['血糖'], train_preds) * 0.5))
print('CV训练用时{}秒'.format(time.time() - t0))

test_result = pd.DataFrame(test_id,columns=["id"])
test_result["score_all"] = test_preds.mean(axis=1)
test_result.to_csv("lgdm_preds_aug.csv",index=None,encoding='utf-8')

#Plot feature importance
myfont = FontProperties(fname="simhei.ttf")
featureImportance = gbm.feature_importance()
#scale by max importance
featureImportance = featureImportance/featureImportance.max()
sorted_idx = np.argsort(featureImportance)
barPos = np.arange(sorted_idx.shape[0]) + 0.5
plot.barh(barPos, featureImportance[sorted_idx], align='center')
plot.yticks(barPos, predictors[sorted_idx],fontproperties=myfont)
plot.xlabel('Variable Importance')
# plot.show()
# #Plot feature importance
# myfont = FontProperties(fname="simhei.ttf")
# fig,ax = plt.subplots(figsize=(12,18))
# lgb.plot_importance(gbm,height=0.8,max_num_features=50,ax=ax)
# ax.set_yticklabels(predictors,rotation='horizontal',fontproperties=myfont)
# plt.show()

if OFFLINE_BUTTON == 1:
    origin_test1 = pd.read_csv('../data/feature/pure_test_feat.csv')
    mse = mean_squared_error(origin_test1['血糖'], test_result["score_all"]) * 0.5
    print('lgdm MSE: '+str(mse))
elif OFFLINE_BUTTON == 2:
    origin_test1 = pd.read_csv('../data/feature/offline_test_feat.csv')
    # origin_test1 = pd.read_csv('../data/feature/all_test_feat.csv')
    mse = mean_squared_error(origin_test1['血糖'], test_result["score_all"]) * 0.5
    print('lgdm MSE: '+str(mse))
