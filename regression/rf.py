from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import pylab as plot
from matplotlib.font_manager import FontProperties
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
else:
    origin_train = pd.read_csv('../data/feature/all_train_feat.csv')

train = origin_train[origin_train['血糖']<=19]
# train_param_features = list(train.columns)
# train_param_features.pop(0)#del id
# train_param_features.pop(-3)#del sugur
# param_features = np.array(train_param_features)

if OFFLINE_BUTTON == 1:
    origin_test = pd.read_csv('../data/feature/pure_test_feat.csv')
elif OFFLINE_BUTTON == 2:
    origin_test = pd.read_csv('../data/feature/offline_test_feat.csv')
    # origin_test = pd.read_csv('../data/feature/all_test_feat.csv')
else:
    origin_test = pd.read_csv('../data/feature/all_test_feat.csv')
#fill nan with train_test.median
origin_test['血糖'] = -999
train_test = pd.concat([train,origin_test],axis=0)
train_test.dropna(axis='columns',how='all',inplace=True)
train_test.fillna(train_test.median(),inplace=True)
# delete specify params and the remaining params without sugar
train_test.drop(del_params,axis=1,inplace=True)
# predictors = [f for f in train_test.columns if f not in ['血糖','id']]
# predictors = np.array(predictors)

train = train_test[train_test['血糖']!=-999]
test = train_test[train_test['血糖']==-999]

train_y = train['血糖']
train_x = train.drop(['血糖','id'],axis=1)
test_id = test.id
test_x = test.drop(['id','血糖'],axis=1)

# 人为尝试组合新特征
# train_x['acid-soda'] = train_x['嗜酸细胞%'] - train_x['嗜碱细胞%']
# test_x['acid-soda'] = test_x['嗜酸细胞%'] - test_x['嗜碱细胞%']
train_x['acid+soda'] = train_x['嗜酸细胞%'] + train_x['嗜碱细胞%']
test_x['acid+soda'] = test_x['嗜酸细胞%'] + test_x['嗜碱细胞%']
train_x['白细胞计数-红细胞计数'] = train_x['白细胞计数'] - train_x['红细胞计数']
test_x['白细胞计数-红细胞计数'] = test_x['白细胞计数'] - test_x['红细胞计数']
predictors = train_x.columns

model = RandomForestRegressor(n_estimators=1000,criterion='mse',max_depth=6,max_features=0.6,min_samples_leaf=5,n_jobs=8,random_state=777)#min_samples_leaf: 5~10
scores = cross_val_score(model,train_x.values,train_y.values,cv=5,scoring='mean_squared_error')
print(np.sqrt(-scores),np.mean(np.sqrt(-scores)))

model.fit(train_x.values,train_y.values)
# 保存rf模型
joblib.dump(model, '../model/regression/xgb_reg.model')
test_result = pd.DataFrame(test_id,columns=["id"])
test_result["score_all"] = model.predict(test_x.values)
test_result.to_csv("rf_preds.csv",index=None,encoding='utf-8')
print(test_result.describe())

#Plot feature importance
myfont = FontProperties(fname="simhei.ttf")
featureImportance = model.feature_importances_
#scale by max importance
featureImportance = featureImportance/featureImportance.max()
sorted_idx = np.argsort(featureImportance)
barPos = np.arange(sorted_idx.shape[0]) + 0.5
plot.barh(barPos, featureImportance[sorted_idx], align='center')
plot.yticks(barPos, predictors[sorted_idx],fontproperties=myfont)
plot.xlabel('Variable Importance')
# plot.show()

if OFFLINE_BUTTON == 1:
    origin_test1 = pd.read_csv('../data/feature/pure_test_feat.csv')
    exact_value = origin_test1['血糖']
    mse = np.mean(np.square(test_result["score_all"] - exact_value))/2.
    print('random forrest MSE:'+str(mse))
elif OFFLINE_BUTTON == 2:
    origin_test1 = pd.read_csv('../data/feature/offline_test_feat.csv')
    # origin_test1 = pd.read_csv('../data/feature/all_test_feat.csv')
    exact_value = origin_test1['血糖']
    mse = np.mean(np.square(test_result["score_all"] - exact_value))/2.
    print('random forrest MSE:'+str(mse))