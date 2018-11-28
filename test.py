from keras_preprocessing.text import Tokenizer

tok_raw = Tokenizer(num_words=10)
tok_raw.fit_on_texts(['i','like','china'])
train['Discuss_seq'] = tok_raw.texts_to_sequences(train.doc.values)
test['Discuss_seq'] = tok_raw.texts_to_sequences(test.doc.values)