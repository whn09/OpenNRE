import sys, json
import torch
import os
import numpy as np
try:
    import opennre
    from opennre import encoder, model, framework
except:
    print('pip install start')
    os.system('/opt/conda/bin/python -m pip -r /opt/ml/code/requirements.txt')
    print('pip install end')
    print('setup install start')
    os.system('/opt/conda/bin/python /opt/ml/code/setup.py install')
    print('setup install end')
    import opennre
    from opennre import encoder, model, framework

# Some basic settings
# root_path = '.'
# sys.path.append(root_path)
# if not os.path.exists('ckpt'):
#     os.mkdir('ckpt')
# ckpt = 'ckpt/finre_bertentity_softmax.pth.tar'

import argparse
import logging
import sagemaker_containers

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=10, metavar='E',
                    help='number of total epochs to run (default: 10)')
parser.add_argument('--batch_size', type=int, default=64, metavar='BS',
                    help='batch size (default: 64)')
parser.add_argument('--lr', type=float, default=2e-5, metavar='LR',
                    help='initial learning rate (default: 2e-5)')

try:
    env = sagemaker_containers.training_env()
    parser.add_argument('--hosts', type=list, default=env.hosts)
    parser.add_argument('--current-host', type=str, default=env.current_host)
    parser.add_argument('--model-dir', type=str, default=env.model_dir)
    parser.add_argument('--data-dir', type=str, default=env.channel_input_dirs.get('training'))
    parser.add_argument('--pretrain-dir', type=str, default=env.channel_input_dirs.get('pretrain'))
    parser.add_argument('--num-gpus', type=int, default=env.num_gpus)
except:
    parser.add_argument('--hosts', type=list, default=[])
    parser.add_argument('--current-host', type=str, default="")
    parser.add_argument('--model-dir', type=str, default="")
    parser.add_argument('--data-dir', type=str, default="")
    parser.add_argument('--pretrain-dir', type=str, default="")
    parser.add_argument('--num-gpus', type=int, default=0)

args = parser.parse_args()

root_path = args.data_dir
pretrain_path = args.pretrain_dir
ckpt = os.path.join(args.model_dir, 'finre_bertentity_softmax.pth.tar')

# Check data
rel2id = json.load(open(os.path.join(root_path, 'finre_rel2id.json')))

# Define the sentence encoder
sentence_encoder = opennre.encoder.BERTEntityEncoder(
    max_length=80, 
    # pretrain_path=os.path.join(root_path, 'pretrain/albert_base_zh')  # TODO cannot support now
    pretrain_path=pretrain_path
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
    batch_size=args.batch_size, # Modify the batch size w.r.t. your device
    max_epoch=args.epochs,
    lr=args.lr,
    opt='adamw'
)

# Train the model
framework.train_model()

# Test the model
framework.load_state_dict(torch.load(ckpt)['state_dict'])
result = framework.eval_model(framework.test_loader)

# Print the result
print('Accuracy on test set: {}'.format(result['acc']))
