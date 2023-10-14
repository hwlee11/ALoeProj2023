import torch
import torch.nn.functional as F
import numpy as np
import math
from dataclasses import dataclass
import time
from nova import DATASET_PATH

def greedy_scoring(model, criterion, h, h_mask, targets, device, sos=1, eos=2):

    #batchSize, _, _ = h.size()
    batchSize, T = targets.size()
    tokens = torch.tensor([[sos]] * batchSize).to(device)
    token_lengths = torch.tensor([1]*batchSize).to(device)
    out_logprobs = torch.zeros(batchSize,2000).to(device)
    loss_sum = 0
    for i in range(T):
        out, out_mask = model.module.decode_one_step(tokens, token_lengths, h, h_mask)
        next_token = out[:,:,-1:].argmax(dim=1)
        logprobs = F.log_softmax(out[:,:,-1:],dim=1).squeeze(-1)
        #sum_logprobs += logprobs.squeeze(0)
        #print(logprobs.size(),targets.size())
        loss = criterion(logprobs,targets[:,i],out_mask.squeeze(1)[:,i])
        loss_sum +=loss.item()
        next_token[tokens[:, -1] == eos] = eos
        #print(tokens.size(),next_token.size())
        tokens = torch.cat([tokens,next_token],dim=-1)
        endOfDecode = (tokens[:,-1] == eos).all()
        if endOfDecode:
            break

    return tokens, loss_sum/T

def trainer(mode, config, dataloader, optimizer, model, criterion, metric, train_begin_time, device):

    log_format = "[INFO] step: {:4d}/{:4d}, loss: {:.6f}, " \
                              "cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h, lr: {:.6f}"
    total_num = 0
    epoch_loss_total = 0.
    print(f'[INFO] {mode} Start')
    epoch_begin_time = time.time()
    cnt = 0
    for inputs, targets, input_lengths, target_lengths in dataloader:
        begin_time = time.time()

        optimizer.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)
        input_lengths = input_lengths.to(device)
        target_lengths = torch.as_tensor(target_lengths).to(device)
        model = model.to(device)

        if mode == 'train':
            outputs, output_mask = model(inputs, input_lengths, targets, target_lengths, device)
            logp = torch.nn.functional.log_softmax(outputs,dim=1).transpose(1,2)#.squeeze(-1) # [B, T]
            loss = criterion(
                logp,
                targets,
                output_mask
            )
            y_hats = logp.argmax(dim=2).squeeze(-1)


        elif mode == 'valid':
            outputs, output_lengths = model.module.encoder_forward(inputs, input_lengths)
            y_hats, loss = greedy_scoring(model, criterion, outputs, output_lengths, targets, device)
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
            cer = metric(targets, y_hats)
            print(log_format.format(
                cnt, len(dataloader), loss,
                cer, elapsed, epoch_elapsed, train_elapsed,
                optimizer.get_lr(),
            ))
        #if cnt == 5:
        #    break
        cnt += 1
    return model, epoch_loss_total/len(dataloader), metric(targets, y_hats)
