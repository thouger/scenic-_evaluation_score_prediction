# -*- coding: utf-8 -*-
# @Time    : 2018/11/28 18:04
# @Author  : thouger
# @Email   : 1030490158@qq.com
# @File    : with_cnn.py
# @Software: PyCharm

import pandas as pd
import jieba
import jieba.analyse

def participle(data):
    data['doc'] = data['Discuss'].map(lambda x: jieba.lcut(x))
jieba.analyse.set_stop_words('../input/scenic_score_prediction/stop.txt')
train = pd.read_csv('../input/scenic_score_prediction/train_first.csv')
test = pd.read_csv('../input/scenic_score_prediction/predict_first.csv')

participle(train)
participle(test)
print()