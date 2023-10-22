import torch
import torch.nn.functional as F
import numpy as np
import math
from dataclasses import dataclass
import time

from modules.utils import learningRateScheduler

#torch.set_printoptions(edgeitems=300)
def greedy_scoring(model, criterion, h, h_mask, targets, device, sos=1, eos=2):

    #batchSize, _, _ = h.size()
    batchSize, T = targets.size()
    tokens = torch.tensor([[sos]] * batchSize).to(device)
    token_lengths = torch.tensor([1]*batchSize).to(device)
    #out_logprobs = torch.zeros(batchSize,2000).to(device)
    loss_sum = 0
    #print('targets',targets)
    for i in range(T):
        out, out_mask = model.module.decode_one_step(tokens, token_lengths, h, h_mask)
        #print('out mask',out_mask.size())
        next_token = out[:,:,-1:].argmax(dim=1)
        logprobs = F.log_softmax(out[:,:,-1:],dim=1).squeeze(-1)
        #print(logprobs[:,2])
        #sum_logprobs += logprobs.squeeze(0)
        loss = criterion(logprobs,targets[:,i],out_mask.squeeze(1)[:,i])
        loss_sum +=loss.item()
        next_token[tokens[:, -1] == eos] = eos
        #print(tokens[:,-1] != eos)
        token_lengths[tokens[:,-1] != eos]+=1
        tokens = torch.cat([tokens,next_token],dim=-1)
        endOfDecode = (tokens[:,-1] == eos).all()
        if endOfDecode:
            break
    """
    for i in range(T):
        print(tokens[0])
        out, out_mask = model.module.decode_one_step(tokens, token_lengths, h, h_mask)
        print(out.size())
        print(out[:,:,-1:].size())
        logprobs = F.log_softmax(out[:,:,-1:],dim=1).squeeze(-1)
        print(logprobs.size())
        next_token = logprobs.argmax(dim=1)
        print('next',next_token[0],next_token.size())
        print('logp',logprobs,logprobs.size())
        print(logprobs.size(),targets[:,i].size(),out_mask.squeeze(1)[:,i])
        #sum_logprobs += logprobs.squeeze(0)
        #print(logprobs.size(),targets.size())
        loss = criterion(logprobs,targets[:,i],out_mask.squeeze(1)[:,i])
        loss_sum +=loss.item()
        next_token[tokens[:, -1] == eos] = eos
        print('next eos',next_token[0],next_token.size())
        print(tokens.size(),next_token.size())
        tokens = torch.cat([tokens,next_token.unsqueeze(-1)],dim=-1)
        endOfDecode = (tokens[:,-1] == eos).all()
        if endOfDecode:
            break
    """
    return tokens[:,1:], loss_sum/T

def trainer(epoch, mode, config, dataloader, optimizer, model, ctc, nll, metric, train_begin_time, tokenizer, device):

    log_format = "[INFO] step: {:4d}/{:4d}, loss: {:.6f}, " \
                              "cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h, lr: {:.6f}"
    total_num = 0
    epoch_loss_total = 0.
    print(f'[INFO] {mode} Start')
    epoch_begin_time = time.time()
    cnt = 0
    for inputs, targets, input_lengths, target_lengths in dataloader:
        begin_time = time.time()
        
        if epoch > 3:
            numOfStep = optimizer.count #.optimizer.state#[optimizer.optimizer.param_groups[0]["params"]]#[-1]] #["step"]
            lr = learningRateScheduler(256,numOfStep,3700)
            optimizer.set_lr(lr)
        optimizer.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)
        input_lengths = input_lengths.to(device)
        target_lengths = torch.as_tensor(target_lengths).to(device)
        model = model.to(device)

        if mode == 'train':
            #outputs, output_mask, output_lengths = model(inputs, input_lengths, targets, target_lengths, device)
            outputs, output_mask, ctc_outputs, output_lengths = model(inputs, input_lengths, targets, target_lengths, device)
            logp = torch.nn.functional.log_softmax(outputs,dim=1).transpose(1,2)#.squeeze(-1) # [B, T, D]

            #loss = nll(
            nllloss = nll(
                logp,
                targets,
                output_mask
            )
            
            ctcloss = ctc(
                ctc_outputs.transpose(1,2).transpose(0,1),
                targets,
                tuple(output_lengths),
                tuple(target_lengths)
            )
            
            loss = (0.2 * ctcloss) + (0.8 * nllloss)
            #print(logp.size()) # [B , T , D]
            y_hats = logp.max(2)[1]
            #print(y_hats,y_hats.size())

        elif mode == 'valid':
            outputs, output_mask = model.module.encoder_forward(inputs, input_lengths)
            y_hats, loss = greedy_scoring(model, nll, outputs, output_mask, targets, device)
            #print(tokenizer.ids_to_text(targets.tolist()))
            #print(tokenizer.ids_to_text(y_hats.tolist()))
            #outputs, output_mask = model(inputs, input_lengths, targets, target_lengths, device)
            #y_hats = outputs.max(1)[1]
            epoch_loss_total += loss

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(model)
            epoch_loss_total += loss.item()

        total_num += int(input_lengths.sum())

        torch.cuda.empty_cache()

        if cnt % config.print_every == 0:

            current_time = time.time()
            elapsed = current_time - begin_time
            epoch_elapsed = (current_time - epoch_begin_time) / 60.0
            train_elapsed = (current_time - train_begin_time) / 3600.0
            sum_cer = 0
            for i in range(len(y_hats)):
                cer = metric(targets[i][:target_lengths[i].item()].unsqueeze(0), y_hats[i][:target_lengths[i].item()].unsqueeze(0))
                sum_cer+=cer
            cer = sum_cer/len(y_hats)
            print('label',tokenizer.label_to_string(targets[0]),':','predict',tokenizer.label_to_string(y_hats[0][:target_lengths[0].item()]))
            print('label',tokenizer.label_to_string(targets[1]),':','predict',tokenizer.label_to_string(y_hats[1][:target_lengths[1].item()]))
            print('label',tokenizer.label_to_string(targets[2]),':','predict',tokenizer.label_to_string(y_hats[2][:target_lengths[2].item()]))
            #print('label',tokenizer.ids_to_text(targets[0].tolist()[:target_lengths[0].item()]),'\n','predict',tokenizer.ids_to_text(y_hats[0].tolist()[:target_lengths[0].item()]) )
            print(log_format.format(
                cnt, len(dataloader), loss,
                cer, elapsed, epoch_elapsed, train_elapsed,
                optimizer.get_lr(),
            ))
            del y_hats
        #if cnt == 5:
        #    break
        cnt += 1
    return model, epoch_loss_total/len(dataloader), metric(targets, y_hats)
