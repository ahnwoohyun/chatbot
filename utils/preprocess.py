from konlpy.tag import Mecab
import pickle


class Preprocess:
    def __init__(self, word2index_dic=''):
        # 단어 인덱스 사전 불러오기
        if word2index_dic != '':
            f = open(word2index_dic, 'rb')
            self.word_index = pickle.load(f)
            f.close()
        else:
            self.word_index = None

        # 형태소 분석기
        self.mecab = Mecab()

        # 추출할 형태소 
        self.include_tags = ['NNG', 'NNP', 'VV', 'SL', 'SN']

    # 형태소 분석기 POS tagging
    def pos(self, sentence):
        return self.mecab.pos(sentence)

    # 불용어 제거 후 필요한 품사정보만 가져오기
    def get_keywords(self, pos, include_tag=False):
        f = lambda x: x in self.include_tags
        word_list = []
        for p in pos:
            if f(p[1]) is True:
                word_list.append(p if include_tag is True else p[0])
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
