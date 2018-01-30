'''
Set the config variable.
'''

import configparser as cp
import json

config = cp.RawConfigParser()
config.read('../data/config/config.cfg')

del_params = config.get("dropout_features","del_params").split(',')
print('delete these params: '+ str(del_params))