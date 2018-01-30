import pandas as pd
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--offline_test', type=int, default=1)

args = parser.parse_args()
OFFLINE_BUTTON = args.offline_test

# 融合四个回归模型的结果
lgdm_preds = pd.read_csv('lgdm_preds.csv')
xgb_preds = pd.read_csv('xgb_preds.csv')
dart_preds = pd.read_csv('dart_preds.csv')
rf_preds = pd.read_csv('rf_preds.csv')
if OFFLINE_BUTTON == 1:
    origin_test1 = pd.read_csv('../data/feature/pure_test_feat.csv')
    exact_value = origin_test1['血糖']
    lgdm_mse = np.mean(np.square(lgdm_preds.score_all - exact_value))/2.
    print('lgdm model MSE:'+str(lgdm_mse))
    xgb_mse = np.mean(np.square(xgb_preds.score_all - exact_value))/2.
    print('xgb model MSE:'+str(xgb_mse))
    dart_mse = np.mean(np.square(dart_preds.score_all - exact_value))/2.
    print('dart model MSE:'+str(dart_mse))
    rf_mse = np.mean(np.square(rf_preds.score_all - exact_value))/2.
    print('rf model MSE:'+str(rf_mse))
elif OFFLINE_BUTTON == 2:
    origin_test1 = pd.read_csv('../data/feature/offline_test_feat.csv')
    exact_value = origin_test1['血糖']
    lgdm_mse = np.mean(np.square(lgdm_preds.score_all - exact_value))/2.
    print('lgdm model MSE:'+str(lgdm_mse))
    xgb_mse = np.mean(np.square(xgb_preds.score_all - exact_value))/2.
    print('xgb model MSE:'+str(xgb_mse))
    dart_mse = np.mean(np.square(dart_preds.score_all - exact_value))/2.
    print('dart model MSE:'+str(dart_mse))
    rf_mse = np.mean(np.square(rf_preds.score_all - exact_value))/2.
    print('rf model MSE:'+str(rf_mse))

w = [0.3,0.3,0.2,0.2] # acquire by tensorflow
xgb_preds.score_all = w[0]*lgdm_preds.score_all + w[1]*xgb_preds.score_all + w[2]*dart_preds.score_all + w[3]*rf_preds.score_all
# w = [0.4,0.3,0.3] # acquire by tensorflow
# xgb_preds.score_all = w[0]*lgdm_preds.score_all + w[1]*xgb_preds.score_all + w[2]*rf_preds.score_all

submission = xgb_preds

# submission.sort_values(by='score_all',inplace=True)
submission.to_csv('predict_result.csv',index=None)
print(submission.describe())

if OFFLINE_BUTTON == 1:
    fu_mse = np.mean(np.square(xgb_preds.score_all - exact_value))/2.
    print('fusion model MSE:'+str(fu_mse))
elif OFFLINE_BUTTON == 2:
    fu_mse = np.mean(np.square(xgb_preds.score_all - exact_value))/2.
    print('fusion model MSE:'+str(fu_mse))