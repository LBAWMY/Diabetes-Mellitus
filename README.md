# 天池精准医疗大赛——人工智能辅助糖尿病遗传风险预测

这是天池平台上的一道关于精准医疗方面的赛题：[链接](https://tianchi.aliyun.com/competition/introduction.htm?raceId=231638)。

## 解决方案

[待写...]

## 代码说明
### data

存放原始的数据文件，包括：

- 训练数据，`draft_data_train.csv`
- A榜测试数据，`d_test_A_20180102`
- B榜测试数据，`d_test_B_20180128.csv`

### 数据分析与特征提取

- `feature_extract.py`,提取特征，在`data`目录下生成`feature`目录，存放特征文件
- `offline_data_extract.py`,提取线下训练与测试集

### classification

- `xgb_hp95.py`，训练xgboost分类器，判断'id'的`血糖`是否大于95百分位数
- `xgb_lp30.py`，训练xgboost分类器，判断'id'的`血糖`是否小于30百分位数
- `series_model.py`,训练一系列分类器，包括MLP,RF,XGB等，模型均保持在`model/classification`文件夹下
- `level_elimination.py`,加载一些列保存的分类模型，对test数据集进行瀑布流筛选，保留top150

### MLP_regression
- `yy_tensorflow.py`,训练MLP回归模型，并同时对测试集进行回归预测


### model

- `classification` 文件夹保存分类器的各类模型
- `regression` 文件夹保存回归其的各类模型(lightdm模型除外)


### regression

- `xgb.py` 训练xgboost回归器
- `dart.py`，训练dart回归器
- `lightdm.py`，训练lightdm回归器
- `rf.py`,训练lightdm回归器
- `gen_submission.py`，生成提交文件，融合了四个回归模型的结果。

- `post_process.py`，使用分类模型的预测结果，对回归预测的结果进行后处理

