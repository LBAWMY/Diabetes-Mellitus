#coding=utf-8
import numpy as np
import pandas as pd
import time
import os
from multiprocessing import Pool
from dateutil.parser import parse

FEATURE_PATH = './data/feature'
ORIGIN_TRAINDATA_PATH = './data/d_train_20180102.csv'
ORIGIN_TESTDATA_PATH = './data/d_test_A_20180102.csv'

OUTPUT_TRAINDATA_PATH = './data/feature/all_train_feat.csv'
OUTPUT_TESTDATA_PATH = './data/feature/all_test_feat.csv'

if not os.path.lexists(FEATURE_PATH):
    os.mkdir(FEATURE_PATH)

############     draft表 类别特征编码   ##############
######################################################
train = pd.read_csv(ORIGIN_TRAINDATA_PATH, encoding='gb2312')
test = pd.read_csv(ORIGIN_TESTDATA_PATH, encoding='gb2312')
# 统计训练集与测试集的数据缺失情况
check_train_null = train.isnull().sum(axis=0).sort_values(ascending=False)/float(len(train))
check_test_null = test.isnull().sum(axis=0).sort_values(ascending=False)/float(len(test))
# print('***************************************')
# print(check_train_null[check_train_null < 0.2])
# print('***************************************')
# print(check_test_null[check_test_null < 0.2])
print('*****************TRAIN DATA**********************')
print(check_train_null)
print('******************TEST DATA**********************')
print(check_test_null)
test['血糖'] = -999
draft_train_test = pd.concat([train,test],axis=0)
# category_var = list(train.columns)
# category_var存放需要进行one-hot标签化的类别名称
category_var = ['性别']

draft_train_test['性别'] = draft_train_test['性别'].map({'男': 1, '女': 0})
draft_train_test['体检日期'] = (pd.to_datetime(draft_train_test['体检日期']) - parse('2017-10-09')).dt.days
draft_train_test.fillna(draft_train_test.median(axis=0),inplace=True)

# add the dimension of kinds variables
for var in category_var:
    var_dummies = pd.get_dummies(draft_train_test[var])
    var_dummies.columns = [var+'_'+str(i) for i in range(var_dummies.shape[1])]
    if var in ['性别']:# delete some orign variables
        draft_train_test.drop(var,axis=1,inplace=True)
    draft_train_test = pd.concat([draft_train_test,var_dummies],axis=1)

draft_train = draft_train_test[draft_train_test['血糖']!=-999]
draft_test = draft_train_test[draft_train_test['血糖']==-999]
draft_test.drop('血糖',axis=1,inplace=True)

# choose the 95 and 30 percentile in train data
# keyindex95 = np.percentile(draft_train['血糖'],95)
# keyindex30 = np.percentile(draft_train['血糖'],30)

draft_train.to_csv(OUTPUT_TRAINDATA_PATH,index=None)
draft_test.to_csv(OUTPUT_TESTDATA_PATH,index=None)