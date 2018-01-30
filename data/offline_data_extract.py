#coding=utf-8
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

ALLTRAINFEAT_PATH = './feature/all_train_feat.csv'
OFFLINE_TRAINFEAT_PATH = './feature/offline_train_feat.csv'
OFFLINE_TESTFEAT_PATH = './feature/offline_test_feat.csv'

# 此函数返回dataFrames类数据中指定param在[minvalue, maxvalue)所有的id('id'在dataFrame类对象中的列名中)的集合
def rt_idset(minvalue, maxvalue, param, dataFrame):
    mid_dataFrame = dataFrame[(dataFrame[param] >= minvalue)]
    return mid_dataFrame[mid_dataFrame[param] < maxvalue]['id'].values

train = pd.read_csv(ALLTRAINFEAT_PATH)
train_param_features = list(train.columns)

# key_max = np.max(train['血糖'].values)
key_max = 18
key_min = np.min(train['血糖'].values)
# 血糖以1为分布来取
step = 1
# 取出0.9作为线上测试集。0.1作为线下测试集
split_rate = 0.90
df_train = pd.DataFrame([])
df_test = pd.DataFrame([])
for index, i in enumerate(np.arange(key_min,key_max,step=step)):
    low = i
    high = i + step
    id_set = rt_idset(low, high, '血糖', train)
    # create offline train and test sets
    np.random.seed(index)
    train_indices = np.random.choice(id_set.shape[0], int(split_rate*id_set.shape[0]),replace=False)
    test_indices = np.array(list(set(range(id_set.shape[0])) - set(train_indices)))
    x_vals_train = train[train.id.isin(id_set[train_indices])]
    x_vals_test = train[train.id.isin(id_set[test_indices])]
    df_train = pd.concat([df_train, x_vals_train])
    df_test = pd.concat([df_test, x_vals_test])

df_train.sort_values(by='id',inplace=True)
df_test.sort_values(by='id',inplace=True)
df_train.to_csv(OFFLINE_TRAINFEAT_PATH,index=None)
df_test.to_csv(OFFLINE_TESTFEAT_PATH,index=None)

# # 1、直方图
# fig = plt.figure()
# ax1 = fig.add_subplot(121)
# a = ax1.hist(df_train['血糖'], range=[0,40], bins=40)
# ax2 = fig.add_subplot(122)
# ax2.hist(df_test['血糖'], range=[0,40], bins=40)
# plt.title('test distribution')
# plt.xlabel('test')
# plt.ylabel('count')
# plt.show()