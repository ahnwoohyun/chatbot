import pandas as pd
import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, Dropout, Conv1D, GlobalMaxPool1D, concatenate, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np

# 데이터 읽어오기
train_file = '../../train_data/intent_sp_train_utf8.csv'
test_file = '../../test_data/intent_sp_test_utf8.csv'
train_data = pd.read_csv(train_file, delimiter=',', encoding='utf-8')
test_data = pd.read_csv(test_file, delimiter=',', encoding='utf-8')

train_sentences = train_data['sentence'].tolist()
train_intents = train_data['intent_id'].tolist()
test_sentences = test_data['sentence'].tolist()
test_intents = test_data['intent_id'].tolist()

from utils.preprocess import Preprocess
p = Preprocess(word2index_dic='../../dict/chatbot_dict.bin')

# 단어 시퀀스 생성
train_sequences = []
for s in train_sentences:
    pos = p.pos(s)
    keywords = p.get_keywords(pos, include_tag=False)
    seq = p.get_wordidx_sequence(keywords)
    train_sequences.append(seq)

test_sequences = []
for s in test_sentences:
    pos = p.pos(s)
    keywords = p.get_keywords(pos, include_tag=False)
    seq = p.get_wordidx_sequence(keywords)
    test_sequences.append(seq)

# 단어 인덱스 시퀀스 벡터 생성
# 단어 시퀀스 벡터 크기
from config.GlobalParams import MAX_SEQ_LEN
X_train = preprocessing.sequence.pad_sequences(train_sequences, maxlen=MAX_SEQ_LEN, padding='post')
X_test = preprocessing.sequence.pad_sequences(test_sequences, maxlen=MAX_SEQ_LEN, padding='post')

y_train = np.array(train_intents)
y_test = np.array(test_intents)

# 하이퍼파라미터 설정
dropout_prob = 0.3
EMB_SIZE = 128
EPOCH = 10
VOCAB_SIZE = len(p.word_index) + 1  # 전체 단어 수

# CNN 모델 정의
input_layer = Input(shape=(MAX_SEQ_LEN,))
embedding_layer = Embedding(VOCAB_SIZE, EMB_SIZE, input_length=MAX_SEQ_LEN)(input_layer)
dropout_emb = Dropout(rate=dropout_prob)(embedding_layer)

conv1 = Conv1D(
    filters=128,
    kernel_size=2,
    padding='valid',
    activation=tf.nn.relu)(dropout_emb)
pool1 = GlobalMaxPool1D()(conv1)

conv2 = Conv1D(
    filters=128,
    kernel_size=3,
    padding='valid',
    activation=tf.nn.relu)(dropout_emb)
pool2 = GlobalMaxPool1D()(conv2)

conv3 = Conv1D(
    filters=128,
    kernel_size=4,
    padding='valid',
    activation=tf.nn.relu)(dropout_emb)
pool3 = GlobalMaxPool1D()(conv3)

# 3, 4, 5-gram 이후 합치기
concat = concatenate([pool1, pool2, pool3])

hidden = Dense(128, activation=tf.nn.relu)(concat)
dropout_hidden = Dropout(rate=dropout_prob)(hidden)
predictions = Dense(3, activation=tf.nn.softmax)(dropout_hidden)

# 모델 생성
model = Model(inputs=input_layer, outputs=predictions)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
mc = ModelCheckpoint('intent_model.h5', monitor='accuracy', mode='max', verbose=1, save_best_only=True)

# 모델 학습
model.fit(X_train, y_train, callbacks=[mc], batch_size=64, validation_split=0.2, epochs=EPOCH, verbose=1)

# test_data 정확도 측정
loaded_model = load_model('intent_model.h5')
print('\n 테스트 정확도: %.4f' % (loaded_model.evaluate(X_test, y_test)[1]))
