# -*- coding: utf-8 -*-
# @Time    : 2018/11/28 18:04
# @Author  : thouger
# @Email   : 1030490158@qq.com
# @File    : with_cnn.py
# @Software: PyCharm

import pandas as pd
import jieba
import jieba.analyse
from keras import Input, Model, optimizers
from keras.callbacks import EarlyStopping
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, concatenate, Dropout, Dense
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from keras import backend as K

jieba.analyse.set_stop_words('../input/scenic_score_prediction/stop.txt')
train = pd.read_csv('../input/scenic_score_prediction/train_first.csv')
test = pd.read_csv('../input/scenic_score_prediction/predict_first.csv')


# 分词
def participle(data):
    data['word'] = data['Discuss'].map(lambda x: jieba.lcut(x))


participle(train)
participle(test)

max_features = 80000  ## 词汇量

token = Tokenizer(num_words=max_features)
token.fit_on_texts(train.word.values)
train['Discuss_seq'] = token.texts_to_sequences(train.word.values)
test['Discuss_seq'] = token.texts_to_sequences(test.word.values)

maxlen = 150


def get_keras_data(data):
    return {
        'Discuss_seq': pad_sequences(data.Discuss_seq, maxlen=maxlen)
    }


x_train = get_keras_data(train)
x_test = get_keras_data(test)
y_train = train.Score.values

early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=2)

embed_size = 200  # emb 长度


def score(y_true, y_pred):
    return 1.0 / (1.0 + K.sqrt(K.mean(K.square(y_true - y_pred), axis=-1)))


def cnn():
    comment_seq = Input(shape=[maxlen], name='Discuss_seq')
    emb_comment = Embedding(max_features, embed_size)(comment_seq)

    convs = []
    for i in range(2, 6):
        conv = Conv1D(filters=100, kernel_size=i)(emb_comment)
        pool = MaxPooling1D(maxlen - i + 1)(conv)
        pool = Flatten()(pool)
        convs.append(pool)
    merge = concatenate(convs, axis=1)
    out = Dropout(0.5)(merge)
    out = Dense(32, activation='relu')(out)
    out = Dense(units=1, activation='linear')(out)
    model = Model([comment_seq], out)
    # adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='mse', optimizer='adam', metrics=['mae', score])
    return model

batch_size=128
epochs = 20
model = cnn()
model.summary()
model.fit(x_train,y_train,validation_split=0.1,batch_size=batch_size,epochs=epochs,shuffle=True,callbacks=[early_stopping])
preds = model.predict(x_test)
submission = pd.DataFrame(test.Id.values,columns=['Id'])
submission['Score'] = preds
submission.to_csv('../input/scenic_score_prediction/cnn-baseline.csv',index=None,header=None)
print()
