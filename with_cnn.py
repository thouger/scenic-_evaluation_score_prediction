# -*- coding: utf-8 -*-
# @Time    : 2018/11/28 18:04
# @Author  : thouger
# @Email   : 1030490158@qq.com
# @File    : with_cnn.py
# @Software: PyCharm

import pandas as pd
import jieba
import jieba.analyse
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

jieba.analyse.set_stop_words('../input/scenic_score_prediction/stop.txt')
train = pd.read_csv('../input/scenic_score_prediction/train_first.csv')
test = pd.read_csv('../input/scenic_score_prediction/predict_first.csv')

#分词
def participle(data):
    data['word'] = data['Discuss'].map(lambda x: jieba.lcut(x))
participle(train)
participle(test)

max_features = 80000  ## 词汇量
token = Tokenizer(num_words=max_features)
token.fit_on_texts(train.word.values)
train['Discuss_seq'] = token.texts_to_sequences(train.word.values)
test['Discuss_seq'] = token.texts_to_sequences(test.word.values)

maxlen =150
def get_keras_data(data):
    return {
        'Discuss_seq' : pad_sequences(data.Discuss_seq,maxlen=maxlen)
    }
x_train = get_keras_data(train)
x_test = get_keras_data(test)
y_train = train.Score.values
print()