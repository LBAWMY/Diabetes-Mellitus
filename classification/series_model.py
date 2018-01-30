#coding=utf-8

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import h5py
import pandas as pd
import numpy as np
from config import *
from sklearn.externals import joblib
import argparse
parser = argparse.ArgumentParser(description='manual to this script')
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

draft_param_feature = list(origin_train.columns)
draft_param_feature.remove('血糖')
draft_param_feature.remove('id')

print("all",origin_train.shape)
train = origin_train[origin_train['血糖']<=19]
highValue = np.percentile(train['血糖'],85)
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


# #####################################################################################################################################
# #training xgboost model
# dtrain = xgb.DMatrix(train_x,label=train_y)
# dtest = xgb.DMatrix(test_x)
#
# # params={
# #     'booster':'gbtree',
# # 	'objective': 'binary:logistic',
# # 	'scale_pos_weight':float(len(train_y)-sum(train_y))/sum(train_y),
# # 	'eval_metric': 'auc',
# # 	'max_depth':5,
# # 	'lambda':0.2,
# # 	'subsample':0.6,
# # 	'colsample_bytree':0.7,
# # 	'eta': 0.001,
# # 	'seed':10,
# # 	'nthread':8
# # }
# params={
#     'booster':'gbtree',
# 	'objective': 'binary:logistic',
# 	'scale_pos_weight':float(len(train_y)-sum(train_y))/sum(train_y),
# 	'eval_metric': 'auc',
# 	'max_depth':4,
# 	'lambda':0.2,
# 	'subsample':0.6,
# 	'colsample_bytree':0.7,
# 	'eta': 0.001,
# 	'seed':1024,
# 	'nthread':8
# }
#
# #通过cv找最佳的nround
# cv_log = xgb.cv(params,dtrain,num_boost_round=250,nfold=5,metrics='auc',early_stopping_rounds=50,seed=1024)#num_boost_round=25000
# bst_auc= cv_log['test-auc-mean'].max()
# cv_log['nb'] = cv_log.index
# cv_log.index = cv_log['test-auc-mean']
# bst_nb = cv_log.nb.to_dict()[bst_auc]
#
# #train
# watchlist = [(dtrain,'train')]
# model = xgb.train(params,dtrain,num_boost_round=bst_nb+50,evals=watchlist)
#
# # SAVE MODEL
# model.save_model('../model/classification/xgb_p98.model')
#
# # #predict test set
# # test_y = model.predict(dtest)
# # test_result = pd.DataFrame(test_id,columns=["id"])
# # test_result["hp95_prob"] = test_y
# # test_result["true suger"] = test_ori_value
# # # regression predict result
# # predict_result = pd.read_csv('../regression/predict_result1.csv')
# # xgb_result = pd.read_csv('../regression/xgb_preds1.csv')
# # rf_result = pd.read_csv('../regression/rf_preds1.csv')
# # lgdm_result = pd.read_csv('../regression/lgdm_preds1.csv')
# # dart_result = pd.read_csv('../regression/dart_preds1.csv')
# # test_result["reg_result"] = predict_result['score_all']
# # test_result["rf_result"] = rf_result['score_all']
# # test_result["xgb_result"] = xgb_result['score_all']
# # test_result["lgdm_result"] = lgdm_result['score_all']
# # test_result["dart_result"] = dart_result['score_all']
# # merged = pd.concat([test_result,train_x],axis=1)
# # # test_result.to_csv("hp95_prob1.csv",index=None,encoding='utf-8')
# # merged.to_csv("merged.csv",index=None,encoding='utf-8')
# print('high value:  '+str(highValue))
# # print(bst_nb,bst_auc)
# #
# # #Plot feature importance
# # myfont = FontProperties(fname="simhei.ttf")
# # fig,ax = plt.subplots(figsize=(12,18))
# # xgb.plot_importance(model,height=0.8,ax=ax)
# # ax.set_yticklabels(predictors,rotation='horizontal',fontproperties=myfont)
# # # plt.show()



# ####################################################################################################################################
# # train rf model
# X = train_x
# y = train_y
#
# rf_best3= RandomForestClassifier(n_estimators=1000, max_depth=5, min_samples_split=5,
#                                  min_samples_leaf=10, oob_score=True, random_state=8)
# rf_best3.fit(X, y)
# # SAVE MODEL
# joblib.dump(rf_best3, '../model/classification/rf_p98.pkl')

# ###################################################################################################################################
# train MLP model
traindata = train_x.values
# trainlabel = train_y.apply(lambda x: np.array([1,0]) if x==1 else np.array([0,1]))
trainlabel = train_y.values
OneHotEncoder().fit_transform(trainlabel.reshape((-1,1)))
train_label = OneHotEncoder().fit_transform(trainlabel.reshape((-1,1))).toarray()
# Binarizer(threshold=0).fit_transform(trainlabel)

# lala = np.array([])
# trainlabel = train_y.apply(lambda x: np.tile([1,0]) if x==1 else np.tile([0,1]))

# rbf_kernal_ovr = Pipeline((
#     ("scaler", StandardScaler()),
#     ("rbf_ovr", SVC(kernel="rbf",  decision_function_shape='ovr')),
# ))
# rbf_kernal_ovr.fit(traindata, trainlabel)
# # SAVE MODEL
# joblib.dump(rbf_kernal_ovr, '../model/classification/svm_p50.pkl')
# Standardize data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
g_data = scaler.fit_transform(train_x)
g_label = np.array(train_label)
# create train and test sets
train_indices = np.random.choice(g_data.shape[0], int(0.8*g_data.shape[0]), replace=False)
test_indices = np.array(list(set(range(g_data.shape[0])) - set(train_indices)))
x_vals_train = g_data[train_indices]
x_vals_test = g_data[test_indices]
y_vals_train = g_label[train_indices]
y_vals_test = g_label[test_indices]

# Plot the data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(g_data[:,0], g_data[:,1], g_data[:,2], c=g_label)
# plt.show()

# define the model
def init_weight(shape, st_dev):
    weight = tf.Variable(tf.random_normal(shape, stddev=st_dev))
    return(weight)

def init_bias(shape, st_dev):
    bias = tf.Variable(tf.random_normal(shape, stddev=st_dev))
    return(bias)

x_data = tf.placeholder(shape=[None, g_data.shape[1]], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, g_label.shape[1]], dtype=tf.float32)

learning_rate = tf.placeholder(dtype=tf.float32)

def fully_connected(input_layer, weights, biases):
    layer = tf.add(tf.matmul(input_layer, weights), biases)
    return(tf.nn.relu(layer))

# MLP Structure
InputLayer = g_data.shape[1]
# HiddenLayer = [512, 1024, 512]
HiddenLayer = [100, 50, 25]
Output_Layer = g_label.shape[1]

# create hidden layer
weight_1 = init_weight(shape=[InputLayer, HiddenLayer[0]], st_dev=1.0)
bias_1 = init_bias(shape=[HiddenLayer[0]], st_dev=1)
layer_1 = fully_connected(x_data, weight_1, bias_1)

weight_2 = init_weight(shape=[HiddenLayer[0], HiddenLayer[1]], st_dev=1.0)
bias_2 = init_bias(shape=[HiddenLayer[1]], st_dev=1.0)
layer_2 = fully_connected(layer_1, weight_2, bias_2)

weight_3 = init_weight(shape=[HiddenLayer[1], HiddenLayer[2]], st_dev=1.0)
bias_3 = init_bias(shape=[HiddenLayer[2]], st_dev=1.0)
layer_3 = fully_connected(layer_2, weight_3, bias_3)

# create output layer
weight_4 = init_weight(shape=[HiddenLayer[2], Output_Layer], st_dev=1.0)
bias_4 = init_bias(shape=[Output_Layer], st_dev=1.0)
final_output = tf.matmul(layer_3, weight_4) + bias_4
# final_output = fully_connected(layer_3, weight_4, bias_4)
# final_output = tf.nn.softmax(tf.matmul(layer_3, weight_4) + bias_4)

# Get the train accuracy
temp_prediction = tf.equal(tf.argmax(y_target, 1), tf.argmax(tf.nn.softmax(final_output), 1))
# temp_prediction = tf.equal(tf.argmax(y_target, 1), tf.argmax(final_output, 1))
accuracy = tf.reduce_mean(tf.cast(temp_prediction, tf.float32))

# define the loss
loss = tf.losses.softmax_cross_entropy(onehot_labels=y_target, logits=final_output)

# define an optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# Create a Session
config = tf.ConfigProto()
config.gpu_options.allow_growth =True
sess = tf.Session(config=config)
#sess = tf.Session()

# init the variables
initializer = tf.global_variables_initializer()
sess.run(initializer)

# Train the model
train_loss_vec = []
test_loss_vec = []
train_accuracy_vec = []
test_accuracy_vec = []
batch_size = 50
for i in range(50000):
    # choose random indices for batch selection
    rand_index = np.random.choice(x_vals_train.shape[0], size=batch_size)
    # Get random batch
    rand_x = x_vals_train[rand_index]
    rand_y = y_vals_train[rand_index] # attention
    # Learning rate control
    rate = 0.001
    if i > 5000 and i < 30000:
        rate = 0.0007
    elif i >=30000:
        rate = 0.001

    # Run the training step
    sess.run(optimizer, feed_dict={x_data: rand_x, y_target: rand_y, learning_rate: rate})

    if (i+1)%500 == 0:
        # Get and store the train loss
        temp_train_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        train_loss_vec.append(float(temp_train_loss))
        # Get and store the test loss
        temp_test_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: y_vals_test})
        train_loss_vec.append(float(temp_test_loss))

        # Get and store train accuracy
        temp_train_accuracy = sess.run(accuracy, feed_dict={x_data: rand_x, y_target: rand_y})
        train_accuracy_vec.append(float(temp_train_accuracy))

        # Get and store test accuracy
        temp_test_accuracy = sess.run(accuracy, feed_dict={x_data: x_vals_test, y_target: y_vals_test})
        test_accuracy_vec.append(float(temp_test_accuracy))

        print('Generation: ' + str(i+1) + '. Learning rate = ' + str(rate) + '. Loss = ' + str(temp_train_loss))
        print('Generation: ' + str(i + 1) + '. Train Accuracy = ' + str(temp_train_accuracy)+ '. Test Accuracy = ' + str(temp_test_accuracy))

# Get and save the model parameter
# file = h5py.File('MLP16_model.h5', 'w')
file = h5py.File('../model/classification/MLP_model_P60.h5', 'w')

file.create_dataset('weight_1', data=sess.run(weight_1))
file.create_dataset('bias_1', data=sess.run(bias_1))
file.create_dataset('weight_2', data=sess.run(weight_2))
file.create_dataset('bias_2', data=sess.run(bias_2))
file.create_dataset('weight_3', data=sess.run(weight_3))
file.create_dataset('bias_3', data=sess.run(bias_3))
file.create_dataset('weight_4', data=sess.run(weight_4))
file.create_dataset('bias_4', data=sess.run(bias_4))

file.create_dataset('scaler_data_max', data=scaler.data_max_)
file.create_dataset('scaler_data_min', data=scaler.data_min_)
file.create_dataset('scaler_data_scaler', data=scaler.scale_)

file.close()

# sess.run(tf.argmax(final_output, 1), feed_dict={x_data: g_data[0:1450]})

plt.plot(train_loss_vec, 'k-', label='Train Loss')
plt.plot(test_loss_vec, 'r-', label='Test Loss')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

plt.plot(train_accuracy_vec, 'k-', label='Train Accuracy')
plt.plot(test_accuracy_vec, 'r-', label='Test Accuracy')
plt.title('Accuracy per Generation')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='upper right')
plt.show()