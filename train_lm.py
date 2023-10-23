import os
import argparse
import time

import torch
from torch.utils.data import DataLoader

#from module.text import transformerLM
from modules.data import split_text_dataset
from modules.data import textDataset
from modules.data import collate_text_fn
#from module.loss import 
from modules.preprocess_text import preprocessing
from modules.losses import NLLLoss
from modules.utils import Optimizer, get_optimizer
from modules.AED import textEncoder

from modules.utils import learningRateScheduler

def save(path, epoch, model, optimizer):
    state = {
        'model': model.state_dict(),
        #'optimizer': optimizer.state_dict()
    }
    torch.save(state, os.path.join(path, 'model_%d.pt'%(epoch)))
    print('Model saved')

def load(path, model):
    state = torch.load(os.path.join(path, 'model_99.pt'))
    model.load_state_dict(state['model'])
    #if 'optimizer' in state and optimizer:
    #    optimizer.load_state_dict(state['optimizer'])
    print('Model loaded')

def main():

    args = argparse.ArgumentParser()

    # DONOTCHANGE
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--iteration', type=str, default='0')
    args.add_argument('--pause', type=int, default=0)


    # Parameters 
    args.add_argument('--use_cuda', type=bool, default=True)
    args.add_argument('--seed', type=int, default=777)
    args.add_argument('--num_epochs', type=int, default=30)
    args.add_argument('--batch_size', type=int, default=192)
    #args.add_argument('--batch_size', type=int, default=128)
    args.add_argument('--save_result_every', type=int, default=10)
    args.add_argument('--checkpoint_every', type=int, default=1)
    args.add_argument('--print_every', type=int, default=500)
    args.add_argument('--dataset', type=str, default='kspon')
    args.add_argument('--output_unit', type=str, default='character')
    args.add_argument('--num_workers', type=int, default=16)
    args.add_argument('--num_threads', type=int, default=16)
    args.add_argument('--init_lr', type=float, default=1e-06)
    args.add_argument('--final_lr', type=float, default=1e-06)
    args.add_argument('--peak_lr', type=float, default=1e-04)
    args.add_argument('--init_lr_scale', type=float, default=1e-02)
    args.add_argument('--final_lr_scale', type=float, default=5e-02)
    args.add_argument('--max_grad_norm', type=int, default=400)
    args.add_argument('--warmup_steps', type=int, default=1000)
    args.add_argument('--weight_decay', type=float, default=1e-01)
    args.add_argument('--reduction', type=str, default='mean')
    args.add_argument('--optimizer', type=str, default='adam')
    args.add_argument('--lr_scheduler', type=str, default='tri_stage_lr_scheduler')
    args.add_argument('--total_steps', type=int, default=200000)

    #args.add_argument('--architecture', type=str, default='deepspeech2')
    #args.add_argument('--architecture', type=str, default='ALoeAED')
    args.add_argument('--use_bidirectional', type=bool, default=True)
    args.add_argument('--dropout', type=float, default=3e-01)
    args.add_argument('--num_encoder_layers', type=int, default=3)
    args.add_argument('--hidden_dim', type=int, default=1024)
    args.add_argument('--rnn_type', type=str, default='gru')
    args.add_argument('--max_len', type=int, default=400)
    args.add_argument('--activation', type=str, default='hardtanh')
    args.add_argument('--teacher_forcing_ratio', type=float, default=1.0)
    args.add_argument('--teacher_forcing_step', type=float, default=0.0)
    args.add_argument('--min_teacher_forcing_ratio', type=float, default=1.0)
    args.add_argument('--joint_ctc_attention', type=bool, default=False)

    args.add_argument('--audio_extension', type=str, default='wav')
    args.add_argument('--transform_method', type=str, default='fbank')

    #preprocessing('/share/datas/ALoeProj/preText/all3.txt', os.getcwd())
    config = args.parse_args()

    #trainData = textDataset(os.path.join(os.getcwd(),'transcripts_train.txt'))
    trainData, valData, testData = split_text_dataset(os.path.join(os.getcwd(),'transcripts.txt'))
    #trainData, valData, testData = split_text_dataset(os.path.join(os.getcwd(),'transcripts_test.txt'))
    
    model = textEncoder()
    # prepare data
    optimizer = get_optimizer(model, config)
    optimizer = Optimizer(optimizer, None, None, config.max_grad_norm)
    nll = NLLLoss()

    num_epochs = config.num_epochs
    num_workers = config.num_workers

    train_begin_time = time.time()

    train_loader = DataLoader(
            trainData,
            batch_size=1024,
            shuffle=True,
            collate_fn=collate_text_fn,
            num_workers=8
    )
    valid_loader = DataLoader(
            valData,
            batch_size=512,
            shuffle=True,
            collate_fn=collate_text_fn,
            num_workers=8
    )
    test_loader = DataLoader(
            testData,
            batch_size=512,
            shuffle=True,
            collate_fn=collate_text_fn,
            num_workers=8
    )
    
    
    model.to('cuda:0')
    
    for epoch in range(30):
        model.train()
        epoch_loss_total = 0
        epoch_val_loss_total = 0
        epoch_test_loss_total = 0

        idx = 1
        print('train batchs',len(train_loader))

        for datas, data_length, targets in train_loader:

            numOfStep = optimizer.count #.optimizer.state#[optimizer.optimizer.param_groups[0]["params"]]#[-1]] #["step"]
            lr = learningRateScheduler(256,numOfStep,3700)
            optimizer.set_lr(lr)

            datas = datas.to('cuda:0')
            data_length = data_length.to('cuda:0')
            targets = targets.to('cuda:0')
            logits, output_mask = model(datas,data_length)
            logp = torch.nn.functional.log_softmax(logits,dim=1).transpose(1,2)#.squeeze(-1) # [B, T, D]

            loss = nll(
                logp,
                targets,
                output_mask
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(model)
            epoch_loss_total += loss.item()
            if idx % 100 == 0:
                print('running loss :', epoch_loss_total/idx)
            idx +=1 


        evg_train_loss = epoch_loss_total/len(train_loader)
        print('train loss :',evg_train_loss)
        save('./temp', epoch, model, optimizer)

        model.eval()
        for datas, data_length, targets in valid_loader:
            datas = datas.to('cuda:0')
            data_length = data_length.to('cuda:0')
            targets = targets.to('cuda:0')

            logits, output_mask = model(datas,data_length)
            logp = torch.nn.functional.log_softmax(logits,dim=1).transpose(1,2)#.squeeze(-1) # [B, T, D]

            loss = nll(
                logp,
                targets,
                output_mask
            )
            epoch_val_loss_total += loss.item()
        evg_val_loss = epoch_val_loss_total/len(valid_loader)

        print('eva val loss :',evg_val_loss)

        for datas, data_length, targets in test_loader:
            datas = datas.to('cuda:0')
            data_length = data_length.to('cuda:0')
            targets = targets.to('cuda:0')

            logits, output_mask = model(datas,data_length)
            logp = torch.nn.functional.log_softmax(logits,dim=1).transpose(1,2)#.squeeze(-1) # [B, T, D]

            loss = nll(
                logp,
                targets,
                output_mask
            )
            epoch_test_loss_total += loss.item()
        evg_test_loss = epoch_val_loss_total/len(test_loader)

        print('test loss :', evg_test_loss)


main()
