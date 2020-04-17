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
    
def get_word2id_and_word2vec(ifilename, ofilename1, ofilename2):
    fout = open(ofilename1, 'w')
    word2id = {}
    word2vec = []
    cnt = 0
    with open(ifilename, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            params = line.split()
            word = "".join(params[0: -200])
            if len(word) == 1:
                vector = list(map(np.float32, params[-200:]))
                word2id[word] = cnt
                word2vec.append(vector)
                cnt += 1
                if cnt % 10000 == 0:
                    print('cnt:', cnt)
    json.dump(word2id, fout, ensure_ascii=False)
    np.save(ofilename2, np.array(word2vec))
    fout.close()
    
# Some basic settings
#root_path = 'benchmark/FinRE/'
#pretrain_path = 'pretrain/tencent/'
#if not os.path.exists('ckpt'):
#    os.mkdir('ckpt')
#ckpt = 'ckpt/finre_cnn_softmax.pth.tar'

# Transform data
#relation2id_txt2json(os.path.join(root_path, 'benchmark/FinRE/relation2id.txt'), os.path.join(root_path, 'benchmark/FinRE/finre_rel2id.json'))
#tsv2openre(os.path.join(root_path, 'benchmark/FinRE/train.txt'), os.path.join(root_path, 'benchmark/FinRE/finre_train.txt'))
#tsv2openre(os.path.join(root_path, 'benchmark/FinRE/test.txt'), os.path.join(root_path, 'benchmark/FinRE/finre_test.txt'))
#tsv2openre(os.path.join(root_path, 'benchmark/FinRE/valid.txt'), os.path.join(root_path, 'benchmark/FinRE/finre_valid.txt'))

# Transform word embedding
#get_word2id_and_word2vec(os.path.join(root_path, 'pretrain/tencent/Tencent_AILab_ChineseEmbedding.txt'), 
#                         os.path.join(root_path, 'pretrain/tencent/Tencent_AILab_ChineseEmbedding_word2id.json'), 
#                         os.path.join(root_path, 'pretrain/tencent/Tencent_AILab_ChineseEmbedding_mat.npy'))

import argparse
import logging
import sagemaker_containers

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=100, metavar='E',
                    help='number of total epochs to run (default: 100)')
parser.add_argument('--batch_size', type=int, default=32, metavar='BS',
                    help='batch size (default: 32)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='initial learning rate (default: 0.1)')

env = sagemaker_containers.training_env()
parser.add_argument('--hosts', type=list, default=env.hosts)
parser.add_argument('--current-host', type=str, default=env.current_host)
parser.add_argument('--model-dir', type=str, default=env.model_dir)
parser.add_argument('--data-dir', type=str, default=env.channel_input_dirs.get('training'))
parser.add_argument('--pretrain-dir', type=str, default=env.channel_input_dirs.get('pretrain'))
parser.add_argument('--num-gpus', type=int, default=env.num_gpus)

args = parser.parse_args()

root_path = args.data_dir
pretrain_path = args.pretrain_dir
ckpt = os.path.join(args.model_dir, 'finre_cnn_softmax.pth.tar')

# Check data
rel2id = json.load(open(os.path.join(root_path, 'finre_rel2id.json')))
# TODO need change to Chinese embedding
wordi2d = json.load(open(os.path.join(pretrain_path, 'Tencent_AILab_ChineseEmbedding_word2id.json')))
word2vec = np.load(os.path.join(pretrain_path, 'Tencent_AILab_ChineseEmbedding_mat.npy'))

# Define the sentence encoder
sentence_encoder = opennre.encoder.CNNEncoder(
    token2id=wordi2d,
    max_length=40,
    word_size=200,
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
    train_path=os.path.join(root_path, 'finre_train.txt'),
    val_path=os.path.join(root_path, 'finre_valid.txt'),
    test_path=os.path.join(root_path, 'finre_test.txt'),
    model=model,
    ckpt=ckpt,
    batch_size=args.batch_size,
    max_epoch=args.epochs,
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
