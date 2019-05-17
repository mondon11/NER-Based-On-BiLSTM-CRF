# coding = utf-8
import tqdm
import re
import random
import pickle

def originHandle():
    '''
    [香港/ns  特别/a  行政区/n]ns => 香港特别行政区/ns
    江/nr  泽民/nr => 江泽民/nr
    '''
    with open('./renmin.txt','r') as inp,open('./renmin2.txt','w') as outp:
        for line in tqdm.tqdm(inp.readlines()):
            if line == '\n':
                continue
            line = line.split('  ')
            i = 1
            while i<len(line)-1:
                if line[i][0]=='[':
                    outp.write(line[i].split('/')[0][1:])
                    i+=1
                    while i<len(line)-1 and line[i].find(']')==-1:
                        if line[i]!='':
                            outp.write(line[i].split('/')[0])
                        i+=1
                    outp.write(line[i].split('/')[0].strip()+'/'+line[i].split('/')[1][-2:]+' ')
                elif line[i].split('/')[1]=='nr':
                    word = line[i].split('/')[0]
                    i+=1
                    if i<len(line)-1 and line[i].split('/')[1]=='nr':
                        outp.write(word+line[i].split('/')[0]+'/nr ')
                    else:
                        outp.write(word+'/nr ')
                        continue
                else:
                    outp.write(line[i]+' ')
                i+=1

            outp.write('\n')

def originHandle2():
    with open('./renmin2.txt','r') as inp,open('./renmin3.txt','w') as outp:
        for line in tqdm.tqdm(inp.readlines()):
            line = line.split(' ')
            i = 0
            while i<len(line)-1:
                if line[i]=='':
                    i+=1
                    continue
                word = line[i].split('/')[0]
                tag = line[i].split('/')[1]

                if tag=='nr' or tag=='ns' or tag=='nt':
                    _tag = {'nr':'PER','ns':'LOC','nt':'ORG'}
                    outp.write(word[0]+"/B-"+_tag[tag]+" ")
                    for j in word[1:len(word)]:
                        if j!=' ':
                            outp.write(j+"/I-"+_tag[tag]+" ")
                else:
                    for wor in word:
                        outp.write(wor+'/O ')

                i+=1
            outp.write('\n')

def originHandle3():
    with open('./renmin3.txt','r') as inp,open('./renmin4.txt','w') as outp:
        texts = inp.read()
        sentences = re.split('[。！？；]/[O]', texts)
        for sentence in sentences:
	        if sentence != " ":
		        outp.write(sentence.strip()+'\n')

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
                f.write(word[i]+'\t'+label[i]+'\n')
                train_dict[label[i]] = train_dict.get(label[i],0)+1
            f.write('\n')
    with open(test_file,'w') as f:
        for word,label in test_list:
            for i in range(len(word)):
                f.write(word[i]+'\t'+label[i]+'\n')
                test_dict[label[i]] = test_dict.get(label[i], 0) + 1
            f.write('\n')
    print('train labels count:')
    print(train_dict)
    print('test labels count:')
    print(test_dict)


def main():
    originHandle()
    originHandle2()
    originHandle3()
    input_list = raw2list('./renmin4.txt')
    vocab = build_vocab('./word2id.pkl', input_list, 5)
    gen_train_test(input_list, 'train_data.txt', 'test_data.txt')
if __name__ == '__main__':
    main()