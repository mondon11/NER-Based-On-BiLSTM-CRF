# coding = utf-8
import tqdm
import pickle
import random

input_file = 'ner_datas.txt'
train_file = 'train_data'
test_file = 'test_data'
word2id_pkl = 'word2id.pkl'

label_raw = []
label_new = ['O','B-LOC','I-LOC','B-ORG','I-ORG','B-PER','I-PER']

def raw2list(input_file):
    res = []
    with open(input_file,'r') as f:
        trunk = []
        for line in tqdm.tqdm(f.readlines()):
            char_list = []
            if len(line.strip()) > 0:
                for char in line.strip().split():
                    char_list.append(char)
                trunk.append(char_list)
            else:
                del trunk[0]
                res.append(trunk)
                if len(trunk[0])!=len(trunk[1]):
                    print('error!')
                trunk = []
    return  res

def build_vocab(vocab_path, input_list, min_count):
    word2id = {}
    for sent_, tag_ in input_list:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            if word not in word2id:
                word2id[word] = [len(word2id)+1, 1]
            else:
                word2id[word][1] += 1
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>':
            low_freq_words.append(word)
    for word in low_freq_words:  # 要过滤掉低频词
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id  # 那些没有在词典中出现的词的id，模型训练好之后，肯定会遇到一些不在词典中的词
    word2id['<PAD>'] = 0

    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)

def gen_train_test(input_list,train_file,test_file,train_prop = 0.8):
    random.shuffle(input_list)
    train_dict = {}
    test_dict = {}
    train_list = input_list[:int(len(input_list)*train_prop)]
    test_list = input_list[int(len(input_list)*train_prop):]
    with open(train_file,'w') as f:
        for word, label in train_list:
            for i in range(len(word)):
                if label[i] == 'B-PERSON':
                    label[i] = 'B-PER'
                elif label[i] == 'I-PERSON':
                    label[i] = 'I-PER'
                elif label[i] == 'B-GPE':
                    label[i] = 'B-LOC'
                elif label[i] == 'I-GPE':
                    label[i] = 'I-LOC'
                elif label[i] not in label_new:
                    label[i] = 'O'
                else:
                    pass
                f.write(word[i]+'\t'+label[i]+'\n')
                train_dict[label[i]] = train_dict.get(label[i],0)+1
            f.write('\n')
    with open(test_file,'w') as f:
        for word,label in test_list:
            for i in range(len(word)):
                if label[i] == 'B-PERSON':
                    label[i] = 'B-PER'
                elif label[i] == 'I-PERSON':
                    label[i] = 'I-PER'
                elif label[i] == 'B-GPE':
                    label[i] = 'B-LOC'
                elif label[i] == 'I-GPE':
                    label[i] = 'I-LOC'
                elif label[i] not in label_new:
                    label[i] = 'O'
                else:
                    pass
                f.write(word[i]+'\t'+label[i]+'\n')
                test_dict[label[i]] = test_dict.get(label[i], 0) + 1
            f.write('\n')
    print('train labels count:')
    print(train_dict)
    print('test labels count:')
    print(test_dict)


input_list = raw2list(input_file)
vocab = build_vocab(word2id_pkl,input_list,5)
gen_train_test(input_list,train_file,test_file)