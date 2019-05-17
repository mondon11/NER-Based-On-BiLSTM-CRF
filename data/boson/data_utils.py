# coding = utf-8
import re
import tqdm
import pickle
import random

label_new = ['O','B-LOC','I-LOC','B-ORG','I-ORG','B-PER','I-PER']

def origin2tag():
    with open('./origindata.txt','r') as inpt,open('./wordtag.txt','w') as outp:
        for line in inpt.readlines():
            line=line.strip()
            line=line.replace(' ','')
            i=0
            while i <len(line):

                if line[i] == '{':
                    i+=2
                    temp=""
                    while line[i]!='}':
                        temp+=line[i]
                        i+=1
                    i+=2
                    word=temp.split(':')
                    sen = word[1]
                    outp.write(sen[0]+"/B_"+word[0]+" ")
                    for j in sen[1:len(sen)]:
                        outp.write(j+"/I_"+word[0]+" ")
                else:
                    outp.write(line[i]+"/O ")
                    i+=1
            outp.write('\n')



def tagsplit():
    with open('./wordtag.txt','r') as inp,open('./wordtagsplit.txt','w') as outp:
        texts = inp.read()
        sentences = re.split('[。！？；]/[O]', texts)
        for sentence in sentences:
            if sentence != " ":
                outp.write(sentence.strip() + '\n')

def raw2list(input_file):
    res = []
    with open(input_file,'r') as f:
        for line in tqdm.tqdm(f.readlines()):
            word_li = []
            label_li = []
            trunk = []
            line = line.strip()
            line_li = re.split(' ',line)
            for item in line_li:
                word_li.append(item[0])
                label_li.append(item[2:])
            trunk.append(word_li)
            trunk.append(label_li)
            res.append(trunk)
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
                if label[i] == 'B_person_name':
                    label[i] = 'B-PER'
                elif label[i] == 'I_person_name':
                    label[i] = 'I-PER'
                elif label[i] == 'B_location':
                    label[i] = 'B-LOC'
                elif label[i] == 'I_location':
                    label[i] = 'I-LOC'
                elif label[i] == 'B_org_name':
                    label[i] = 'B-ORG'
                elif label[i] == 'I_org_name':
                    label[i] = 'I-ORG'
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
                if label[i] == 'B_person_name':
                    label[i] = 'B-PER'
                elif label[i] == 'I_person_name':
                    label[i] = 'I-PER'
                elif label[i] == 'B_location':
                    label[i] = 'B-LOC'
                elif label[i] == 'I_location':
                    label[i] = 'I-LOC'
                elif label[i] == 'B_org_name':
                    label[i] = 'B-ORG'
                elif label[i] == 'I_org_name':
                    label[i] = 'I-ORG'
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


def main():
    origin2tag()
    tagsplit()
    input_list = raw2list('./wordtagsplit.txt')
    vocab = build_vocab('./word2id.pkl', input_list, 5)
    gen_train_test(input_list, 'train_data.txt', 'test_data.txt')


if __name__ == '__main__':
    main()