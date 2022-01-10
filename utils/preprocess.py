from konlpy.tag import Mecab
import pickle
import pandas as pd


class Preprocess:
    def __init__(self, word2index_dic='', synonym_dic=''):
        # 단어 인덱스 사전 불러오기
        if word2index_dic != '':
            f = open(word2index_dic, 'rb')
            self.word_index = pickle.load(f)
            f.close()
        else:
            self.word_index = None

        # 동의어 사전 불러오기
        if synonym_dic != '':
            synonym = pd.read_csv(synonym_dic, encoding='utf-8')
            self.synonym_dic = {name: value for name, value in zip(synonym['synonym'].tolist(), synonym['entry'].tolist())}
        else:
            self.synonym_dic = None

        # 형태소 분석기
        self.mecab = Mecab()

        # 추출할 형태소 
        self.include_tags = ['NNG', 'NNP', 'NNB', 'VV', 'SL', 'SN', 'EC', 'EF', 'VX']

    # 형태소 분석기 POS tagging
    def pos(self, sentence):
        return self.mecab.pos(sentence)

    # 불용어 제거 후 필요한 품사정보만 가져오기 및 동의어 변환
    def get_keywords(self, pos, include_tag=False):
        f = lambda x: x in self.include_tags
        word_list = []
        for p in pos:
            if f(p[1]) is True:
                word_list.append(p if include_tag is True else p[0])

        for idx in range(len(word_list)):
            if word_list[idx] in self.synonym_dic.keys():
                word_list[idx] = self.synonym_dic[word_list[idx]]
        return word_list

    # 키워드를 단어 인덱스 시퀀스로 변환
    def get_wordidx_sequence(self, keywords):
        if self.word_index is None:
            return []
        w2i = []
        for word in keywords:
            try:
                w2i.append(self.word_index[word])
            except KeyError:
                w2i.append(self.word_index['OOV'])
        return w2i

