from utils.preprocess import Preprocess
from models.intent.intentModel import IntentModel

p = Preprocess(word2index_dic='dict/chatbot_dict.bin')
intent = IntentModel(model_name='models/intent/intent_model.h5', preprocess=p)

query = '계약서 검색'

predict = intent.predict_class(query)
#intent_name = intent.labels[predict]

print(predict)