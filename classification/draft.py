#coding=utf-8

import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd
import copy
import numpy as np
from config import *


grid_log = pd.read_csv('best-params.csv')
improve5_frame = grid_log.sort_values(by='improve5', axis=0, ascending=False)
top5 = improve5_frame['improve5'][0:5]
improve10_frame = grid_log.sort_values(by='improve10', axis=0, ascending=False)
top10 = improve10_frame['improve10'][0:10]
improve15_frame = grid_log.sort_values(by='improve15', axis=0, ascending=False)
top15 = improve15_frame['improve15'][0:15]
print('**************TOP 5****************')
print(top5)
print('**************TOP 10****************')
print(top10)
print('**************TOP 15****************')
print(top15)