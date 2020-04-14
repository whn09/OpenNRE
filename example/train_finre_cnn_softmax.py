import sys, json
import torch
import os
import numpy as np
import opennre
from opennre import encoder, model, framework
import json

def text2token(text):
    token = list(text)
    new_token = []
    skip = 0
    for i in range(len(token)):
        if skip > 0:
            skip -= 1
            continue
        # TODO how to deal with English words?
        if token[i] == '<' and i < len(token)-2 and token[i+1] == 'N' and token[i+2] == '>':
            new_token.append('<N>')
            skip = 2
        else:
            new_token.append(token[i])
            skip = 0
    return new_token

def get_pos(name, text):
    pos = [0, 0]
    index = text.find(name)
    n_cnt = 0
    for i in range(index):
        # TODO how to deal with English words?
        if text[i] == '<' and i < index-2 and text[i+1] == 'N' and text[i+2] == '>':
            n_cnt += 1
    index -= n_cnt*2
    length = len(name)
    pos = [index, index+length]
    return pos

def tsv2openre(ifilename, ofilename):
    fout = open(ofilename, 'w')
    with open(ifilename, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if line == '':
                continue
            params = line.split('\t')
            item = {}
            item['token'] = text2token(params[3])
            item['h'] = {}
            item['h']['name'] = params[0]
            item['h']['id'] = ''
            item['h']['pos'] = get_pos(params[0], params[3])
            item['t'] = {}
            item['t']['name'] = params[1]
            item['t']['id'] = ''
            item['t']['pos'] = get_pos(params[1], params[3])
            item['relation'] = params[2]
            fout.write(json.dumps(item, ensure_ascii=False)+'\n')
    fout.close()

def relation2id_txt2json(ifilename, ofilename):
    fout = open(ofilename, 'w')
    result = {}
    with open(ifilename, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            params = line.split(' ')
            result[params[0]] = int(params[1])
    json.dump(result, fout)
    fout.close()
    
# Some basic settings
root_path = '.'
if not os.path.exists('ckpt'):
    os.mkdir('ckpt')
ckpt = 'ckpt/finre_cnn_softmax.pth.tar'

# Transform data
relation2id_txt2json(os.path.join(root_path, 'benchmark/FinRE/relation2id.txt'), os.path.join(root_path, 'benchmark/FinRE/finre_rel2id.json'))
tsv2openre(os.path.join(root_path, 'benchmark/FinRE/train.txt'), os.path.join(root_path, 'benchmark/FinRE/finre_train.txt'))
tsv2openre(os.path.join(root_path, 'benchmark/FinRE/test.txt'), os.path.join(root_path, 'benchmark/FinRE/finre_test.txt'))
tsv2openre(os.path.join(root_path, 'benchmark/FinRE/valid.txt'), os.path.join(root_path, 'benchmark/FinRE/finre_valid.txt'))

# Check data
rel2id = json.load(open(os.path.join(root_path, 'benchmark/FinRE/finre_rel2id.json')))
# TODO need change to Chinese embedding
wordi2d = json.load(open(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_word2id.json')))
word2vec = np.load(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_mat.npy'))

# Define the sentence encoder
sentence_encoder = opennre.encoder.CNNEncoder(
    token2id=wordi2d,
    max_length=40,
    word_size=50,
    position_size=5,
    hidden_size=230,
    blank_padding=True,
    kernel_size=3,
    padding_size=1,
    word2vec=word2vec,
    dropout=0.5
)

# Define the model
model = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)

# Define the whole training framework
framework = opennre.framework.SentenceRE(
    train_path=os.path.join(root_path, 'benchmark/FinRE/finre_train.txt'),
    val_path=os.path.join(root_path, 'benchmark/FinRE/finre_valid.txt'),
    test_path=os.path.join(root_path, 'benchmark/FinRE/finre_test.txt'),
    model=model,
    ckpt=ckpt,
    batch_size=32,
    max_epoch=100,
    lr=0.1,
    weight_decay=1e-5,
    opt='sgd'
)

# Train the model
framework.train_model()

# Test the model
framework.load_state_dict(torch.load(ckpt)['state_dict'])
result = framework.eval_model(framework.test_loader)

# Print the result
print('Accuracy on test set: {}'.format(result['acc']))
