import torch
import numpy as np
import math
from dataclasses import dataclass
import time
from nova import DATASET_PATH

def greedy_scoring(model, h, h_mask, device, sos=1, eos=2):

    batchSize, _, T = h.size()
    tokens = torch.tensor([sos] * batchSize).to(device)
    token_lengths = torch.tensor([1]*batchSize).to(device)
    sum_logprobs = torch.zeros(2,2000).to(device)
    for i in range(T):
        out, out_mask = model.decode_one_step(tokens, token_lengths, h, h_mask)
        next_token = out.argmax(dim=1)
        logprobs = F.log_softmax(out[:,:,-1:],dim=1).squeeze(-1)
        sum_logprobs += logprobs.squeeze(0)

        next_token[tokens[:, -1] == eos] = eos
        tokens = torch.cat([tokens,next_token],dim=-1)
        endOfDecode = (tokens[:,-1] == eos).all()
        if endOfDecode:
            break

    return tokens, sum_logprobs

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
            outputs, output_lengths = model(inputs, input_lengths)
            loss = criterion(
                outputs.transpose(0, 1),
                targets[:, 1:],
                tuple(output_lengths),
                tuple(target_lengths)
            )
            y_hats = outputs.max(-1)[1]

        elif mode == 'valid':
            outputs, output_lengths = model.encoder_forward(inputs, input_legnths)
            y_hats, loss = greedy_scoring(model, outputs, outputs_lengths, device)

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(model)
        elif mode == 'valid':

        total_num += int(input_lengths.sum())
        epoch_loss_total += loss.item()

        torch.cuda.empty_cache()

        if cnt % config.print_every == 0:

            current_time = time.time()
            elapsed = current_time - begin_time
            epoch_elapsed = (current_time - epoch_begin_time) / 60.0
            train_elapsed = (current_time - train_begin_time) / 3600.0
            cer = metric(targets[:, 1:], y_hats)
            print(log_format.format(
                cnt, len(dataloader), loss,
                elapsed, epoch_elapsed, train_elapsed,
                optimizer.get_lr(),
            ))
        cnt += 1
    return model, epoch_loss_total/len(dataloader), metric(targets[:, 1:], y_hats)
