import os
import random
import json
import jieba
import numpy as np
from flags import BOS, EOS, UNK, DROPOUT

def read_json(filename):
    with open(filename, 'r') as fp:
        data = json.load(fp)
    return data


class utils():
    def __init__(self,args):
        self.batch_size = args.batch_size
        self.data_dir = args.data_dir
        self.word_embd_path = os.path.join(self.data_dir, 'word_embd')
        self.sequence_length = args.sequence_length
        self.word_id_dict = read_json(os.path.join(self.data_dir, 'dict'))
        self.BOS_id = BOS
        self.EOS_id = EOS
        self.unknown_id =  UNK
        self.droptout_id = DROPOUT
        jieba.load_userdict(os.path.join(self.data_dir, 'word'))

        self.id_word_dict = [[]]*len(self.word_id_dict)
        print(len(self.id_word_dict))
        for word in self.word_id_dict:
            self.id_word_dict[self.word_id_dict[word]] = word

    def word_drop_out(self,sents,rate=0.3):
        sents = np.array(sents)
        for i in range(len(sents)):
            for j in range(len(sents[i])):
                if random.random()<=rate and sents[i][j]!=0:
                    sents[i][j] = self.word_id_dict['__DROPOUT__']
        return sents

    def sent2id(self, sent, l=None, sp=False):
        sent = sent.decode('utf-8').strip()
        if sp:
          sent_list = list(jieba.cut(sent, cut_all=False))
        else:
          sent_list = sent.split()
        vec = np.ones((self.sequence_length),dtype=np.int32)
        sent_len = len(sent_list)
        unseen = 0
        for i, word in enumerate(sent_list):
            if i==self.sequence_length:
                break
            if word in self.word_id_dict:
                vec[i] = self.word_id_dict[word]
            else:
                vec[i] = self.unknown_id
        if l:
            return vec, sent_len
        else:
            return vec


    def id2sent(self,ids):
        word_list = []
        for i in ids:
            if i == EOS:
              break
            word_list.append(self.id_word_dict[i])
        return ' '.join(word_list).encode('utf-8')


    def train_data_generator(self):
        while True:
            f = open(os.path.join(self.data_dir,'source_train'),'r')
            data = f.readlines()
            random.shuffle(data)

            batch_s = [];batch_t = [];
            for i in range(len(data)):
                s_sent,s_l = self.sent2id(data[i],1)
                t_sent,t_l = self.sent2id(data[i],1)
                if s_l<=30 and t_l<=30 and random.random()<0.8:
                    batch_s.append(s_sent)
                    batch_t.append(t_sent)
                    if len(batch_s)== self.batch_size:
                        yield batch_s,batch_t
                        batch_s = [];batch_t = [];

    def test_data_generator(self):
        f = open(os.path.join(self.data_dir,'source_test'),'r')
        data = f.readlines()

        batch_s = [];   batch_sen = []
        for i in range(len(data)):
            s_sent, s_l = self.sent2id(data[i], 1)
            if s_l < self.sequence_length :
                batch_s.append(s_sent)
                batch_sen.append(data[i])
                if len(batch_s)== self.batch_size:
                    yield batch_s, batch_sen
                    batch_s = [];   batch_sen = []

    def load_word_embedding(self):
        embd = []
        with open(self.word_embd_path,'r') as f:
            for index, line in enumerate(f.readlines()):
                row = line.strip().split(' ')
                embd.append(row[1:])
                if index == len(self.word_id_dict)-5:  #EOS BOS UNK DROPOUT
                    print('SIZE: ' + str(index+1))
                    break
            print('Word Embedding Loaded')
            embedding = np.asarray(embd,'f')
            return embedding



