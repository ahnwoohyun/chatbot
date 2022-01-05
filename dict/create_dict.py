from utils.preprocess import Preprocess
from tensorflow.keras import preprocessing
import pickle
import pandas as pd


# 말뭉치 데이터 읽어오기
data = pd.read_csv('../train_data/intent_sp_train_utf8.csv', encoding='utf-8')

sentences = []
for i in data['sentence']:
    sentences2 = []
    i = i.strip()
    sentences2.append(i)
    sentences.append(sentences2)

# 말뭉치 데이터에서 키워드만 추출해서 사전 리스트 생성
p = Preprocess()
dict = []

for c in sentences:
    pos = p.pos(c[0])
    for k in pos:
        dict.append(k[0])
print(dict)

# 사전에 사용될 word2index 생성
# 사전의 첫 번째 인덱스에는 OOV 사용
tokenizer = preprocessing.text.Tokenizer(oov_token='OOV')
tokenizer.fit_on_texts(dict)
word_index = tokenizer.word_index

# 사전 파일 생성
f = open('chatbot_dict.bin', 'wb')
try:
    pickle.dump(word_index, f)
except Exception as e:
    print(e)
finally:
    f.close()