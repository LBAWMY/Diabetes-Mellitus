#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import random
import datetime


def augment_train_data_two_side(data):
    """前后各放大10%
    """
    data_low10 = data.iloc[data.index // 10, :]
    data_top10 = data.iloc[-(len(data.index)) // 10, :]
    return pd.concat([data_low10, data, data_top10], axis=0, ignore_index=True)

data = pd.read_csv("./data/d_train.csv", index_col=False, encoding='GBK')
data = data.sort_values(by=[u"血糖"])
data = augment_train_data_two_side(data)
data = data.sample(frac=1, random_state=618).reset_index(drop=True)  # 随机打乱数据


predictors = [f for f in data.columns if f not in ['血糖', 'id', '性别', "年龄", "体检日期", "乙肝表面抗原",
                                                    "乙肝表面抗体", "乙肝e抗原", "乙肝e抗体", "乙肝核心抗体"]]
# “乙肝...核心” 这五维特征缺失比较严重，在训练模型中均删除，故不考虑处理

new_data = pd.DataFrame(data=None, columns=data.columns)
for i in range(50):  # 30代表放大倍数
    print("第{}次加噪处理...".format(i))
    index_list = random.sample(range(data.shape[0]), data.shape[0])
    for j, index in enumerate(index_list):
        print("处理第{}个数...".format(j))
        create_data = data.loc[index]

        cur_random = random.randint(0, 9)  # 产生随机数，用于选择特征处理方法
        cur_len = random.randint(1, 12)  # 选择特征的个数

        select_feature_list = random.sample(range(len(predictors)), cur_len)

        # 随机选择剔除特征或修改特征值
        if cur_random <= 4:
            for feature_index in select_feature_list:
                feature = predictors[feature_index]
                create_data[feature] = np.nan
        elif cur_random <= 8:
            for feature_index in select_feature_list:
                feature = predictors[feature_index]
                if create_data[feature] == np.nan:
                    create_data[feature] = random.random(data[feature].mean() - data[feature].std(),
                                                         data[feature].mean() + data[feature].std())
                else:
                    create_data[feature] *= random.randint(93, 107) * 0.01
        else:
            print("保留原值")
        new_data = new_data.append(create_data)

new_data = new_data.sample(frac=1, random_state=618).reset_index(drop=True)  # 随机打乱数据
new_data.to_csv(r'd_train_augment_{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)