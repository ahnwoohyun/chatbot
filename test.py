from utils.preprocess import Preprocess
from models.intent.intentModel import IntentModel
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', default='predict') 
    parser.add_argument('-q', required=True, default=None) #query

    p = Preprocess(word2index_dic='dict/chatbot_dict.bin', synonym_dic='dict/synonym_dict.csv')
    intent = IntentModel(model_name='models/intent/intent_model.h5', preprocess=p)
    args = parser.parse_args()

    predict = intent.predict_class(args.q)
    if predict == 2:
        print('============ result ============')
        print("죄송합니다. '문서검색' 혹은 '문서공유'라고 입력 해주세요")
    else:
        print('============ result ============')
        print(intent.labels[predict], "의도로 분류됩니다.")