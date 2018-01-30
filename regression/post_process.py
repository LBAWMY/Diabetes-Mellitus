import pandas as pd
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--offline_test', type=int, default=1)

args = parser.parse_args()
OFFLINE_BUTTON = args.offline_test

if OFFLINE_BUTTON == 1:
    origin_test1 = pd.read_csv('../data/feature/pure_test_feat.csv')
    exact_value = origin_test1['血糖']
#lp30 prob
lp30 = pd.read_csv('../classification/lp30_prob.csv')
lp30.sort_values(by='lp30_prob',inplace=True)
lp30_15 = lp30.tail(10)

#hp95 prob
hp95 = pd.read_csv('../classification/hp95_prob.csv')
hp95.sort_values(by='hp95_prob',inplace=True)
hp95_20 = hp95.tail(10)

# post process regression predict result
predict_result = pd.read_csv('predict_result.csv')

if OFFLINE_BUTTON == 1:
    pre_mse = np.mean(np.square(predict_result.score_all - exact_value))/2.
    print('pre MSE:'+str(pre_mse))

predict_result = pd.merge(predict_result,lp30_15,on='id',how='left')
predict_result.fillna(-999,inplace=True)
predict_result = predict_result[predict_result.lp30_prob==-999]
predict_result = predict_result[['id','score_all']]

predict_result = pd.merge(predict_result,hp95_20,on='id',how='left')
predict_result.fillna(-999,inplace=True)
predict_result = predict_result[predict_result.hp95_prob==-999]
predict_result = predict_result[['id','score_all']]

lp30_15['score_all'] = 4.85
lp30_15.drop('lp30_prob',axis=1,inplace=True)

hp95_20['score_all'] = 10.52
hp95_20.drop('hp95_prob',axis=1,inplace=True)

#generate submission file
submission = pd.concat([predict_result,lp30_15,hp95_20],axis=0)
submission.sort_values(by='id',inplace=True)
results = submission['score_all']
results.to_csv('final_predict_result2.csv',index=None, header=None, float_format='%.4f')
print(submission.describe())
if OFFLINE_BUTTON == 1:
    mse = np.mean(np.square(submission.score_all - exact_value))/2.
    print('final MSE:'+str(mse))
