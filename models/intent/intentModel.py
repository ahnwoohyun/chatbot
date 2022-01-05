import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname('utils'))))))
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing import sequence
import argparse
from utils.preprocess import Preprocess






# 인텐트 분류 모델 모듈
class IntentModel:
    def __init__(self, model_name, preprocess):
        # 인텐트 클래스 label
        self.labels = {0: '문서검색', 1: '문서공유', 2: '기타'} 
        # 인텐트 분류 모델 load
        self.model = load_model(model_name)

        # 챗봇 Preprocess 객체
        self.p = preprocess

    # 인텐트 클래스 예측
    def predict_class(self, query):
        # 형태소 분석
        pos = self.p.pos(query)

        # 문장 내 키워드 추출
        keywords = self.p.get_keywords(pos, include_tag=False)
        print(keywords)
        sequences = [self.p.get_wordidx_sequence(keywords)]
        print(sequences)

        # 단어 시퀀스 벡터 크기
        from config.GlobalParams import MAX_SEQ_LEN

        # 패딩 처리
        padded_seqs = preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post')
        
        #predict = float(self.model.predict(padded_seqs))
        predict = self.model.predict(padded_seqs)
        predict_class = tf.math.argmax(predict, axis=1)
        #print(predict_class)
        return predict_class.numpy()[0]
        #print('==========result==========')
        #if(predict > 0.5):
        #    print("{:.2f}% 확률로 문서공유 의도로 분류됩니다.\n".format(predict * 100))
        #else:
        #    print("{:.2f}% 확률로 문서검색 의도로 분류됩니다.\n".format((1 - predict) * 100))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', default='predict') 
    parser.add_argument('-q', required=True, default=None) #query

    p = Preprocess(word2index_dic='../../dict/chatbot_dict.bin')
    intent = IntentModel(model_name='intent_model.h5', preprocess=p)
    args = parser.parse_args()
    
    predict = intent.predict_class(args.q)
    print(intent.labels[predict], "의도로 분류됩니다.")
    