import os
import numpy as np
import torchaudio

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from modules.vocab import KoreanSpeechVocabulary
from modules.data import load_audio
from modules.model import DeepSpeech2
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from nova import DATASET_PATH
import time

torch.set_printoptions(edgeitems=300)
def load(path, model):
    state = torch.load(os.path.join(path, 'model_99.pt'))
    model.load_state_dict(state['model'])
    #if 'optimizer' in state and optimizer:
    #    optimizer.load_state_dict(state['optimizer'])
    print('Model loaded')


    #def BeamSearchDecode(model, logits, beam_size, sos=1, eos=2):

class BeamSearchDecoder():
    def __init__(
        self,
        beam_size: int,
        eot: int,
        #inference: Inference,
        patience = None,
    ):
        self.beam_size = beam_size
        self.eot = eot
        #self.inference = inference
        self.patience = patience or 1.0
        self.max_candidates: int = round(beam_size * self.patience)
        self.finished_sequences = None

        assert (
            self.max_candidates > 0
        ), f"Invalid beam size ({beam_size}) or patience ({patience})"

    def reset(self):
        self.finished_sequences = None

    def update(
        self, tokens: Tensor, logits: Tensor, token_lengths, sum_logprobs: Tensor
    ) -> Tuple[Tensor, bool]:
        if tokens.shape[0] % self.beam_size != 0:
            raise ValueError(f"{tokens.shape}[0] % {self.beam_size} != 0")

        n_audio = tokens.shape[0] // self.beam_size
        if self.finished_sequences is None:  # for the first update
            self.finished_sequences = [{} for _ in range(n_audio)]
        logprobs = F.log_softmax(logits.float(), dim=1)[:,:,-1].squeeze(-1)#.transpose(1,2)
        #print('in update logpr',logprobs.size())
        next_tokens, finished_sequences = [], []
        for i in range(n_audio):
            scores, sources, finished = {}, {}, {}

            # STEP 1: calculate the cumulative log probabilities for possible candidates
            for j in range(self.beam_size):
                idx = i * self.beam_size + j
                prefix = tokens[idx].tolist()
                for logprob, token in zip(*logprobs[idx].topk(self.beam_size + 1)):
                    if prefix[-1] == self.eot:
                        sequence = tuple(prefix + [self.eot])
                        scores[sequence] = sum_logprobs[idx].item()
                    else:
                        new_logprob = (sum_logprobs[idx] + logprob).item()
                        sequence = tuple(prefix + [token.item()])
                        scores[sequence] = new_logprob


            # STEP 2: rank the candidates and keep the top beam_size sequences for each audio
            saved = 0
            for sequence in sorted(scores, key=scores.get, reverse=True):
                if sequence[-1] == self.eot:
                    finished[sequence] = scores[sequence]
                sum_logprobs[len(next_tokens)] = scores[sequence]
                token_lengths[len(next_tokens)]+=1
                next_tokens.append(sequence)
                #source_indices.append(sources[sequence])

                saved += 1
                if saved == self.beam_size:
                    break
                """
                else:
                    sum_logprobs[len(next_tokens)] = scores[sequence]
                    token_lengths[len(next_tokens)]+=1
                    next_tokens.append(sequence)
                    #source_indices.append(sources[sequence])

                    saved += 1
                    if saved == self.beam_size:
                        break
                """
            finished_sequences.append(finished)

        #next_tokens = torch.tensor(next_tokens)
        #print(next_tokens)
        #print(tokens.size())
        #next_tokens[tokens[:, -1] == self.eot] = self.eot
        tokens = torch.tensor(next_tokens, device=tokens.device)
        #self.inference.rearrange_kv_cache(source_indices)
        # add newly finished sequences to self.finished_sequences
        assert len(self.finished_sequences) == len(finished_sequences)
        for previously_finished, newly_finished in zip(
            self.finished_sequences, finished_sequences
        ):
            for seq in sorted(newly_finished, key=newly_finished.get, reverse=True):
                if len(previously_finished) >= self.max_candidates:
                    break  # the candidate list is full
                previously_finished[seq] = newly_finished[seq]

        # mark as completed if all audio has enough number of samples
        completed = all(
            len(sequences) >= self.max_candidates
            for sequences in self.finished_sequences
        )
        completed = (tokens[:,-1] == self.eot).all()

        return tokens, completed

    def finalize(self, preceding_tokens: Tensor, sum_logprobs: Tensor):
        # collect all finished sequences, including patience, and add unfinished ones if not enough
        sum_logprobs = sum_logprobs.cpu()
        for i, sequences in enumerate(self.finished_sequences):
            if (
                len(sequences) < self.beam_size
            ):  # when not enough sequences are finished
                for j in list(np.argsort(sum_logprobs[i]))[::-1]:
                    sequence = preceding_tokens[i, j].tolist() + [self.eot]
                    sequences[tuple(sequence)] = sum_logprobs[i][j].item()
                    if len(sequences) >= self.beam_size:
                        break

        tokens: List[List[Tensor]] = [
            [torch.tensor(seq) for seq in sequences.keys()]
            for sequences in self.finished_sequences
        ]
        sum_logprobs: List[List[float]] = [
            list(sequences.values()) for sequences in self.finished_sequences
        ]
        return tokens, sum_logprobs

def beamSearch(model, beam, beamSize, h, h_mask, device, sos=1, eos=2):

    batchSize, _, T = h.size()
    #tokens = torch.tensor([sos]).repeat(batchSize, 1)
    tokens = torch.tensor([[sos]] * (batchSize * beamSize)).to(device)
    token_lengths = torch.tensor([1] * (batchSize * beamSize)).to(device)
    #beam = BeamSearchDecoder(beamSize, eos)

    sum_logprobs = torch.zeros(batchSize*beamSize).to(device)
    #hyps = {'tokens':[],'socre':[]}
    h = h.repeat_interleave(beamSize,dim=0)
    h_mask = h_mask.repeat_interleave(beamSize, dim=0)

    while True:
        logits, out_mask = model.module.decode_one_step(tokens, token_lengths, h, h_mask)
        tokens, completed = beam.update(tokens, logits, token_lengths, sum_logprobs)
        if completed :
            break

    tokens = tokens.reshape(batchSize ,beamSize, -1)
    sum_logprobs = sum_logprobs.reshape(batchSize, beamSize)
    selectedIdx = torch.argmax(sum_logprobs, dim=1)
    results = list()
    #print('tokens shape',tokens.size())
    #print('slected',selectedIdx,selectedIdx.size())
    for b in range(batchSize):
        results.append(tokens[b][selectedIdx[b]].tolist())
    #print(results)
    tokens =torch.tensor(results)
    #print(tokens,tokens.size())
    #print(token_lengths)
    beam.reset()
    #tokens, sum_logprobs = beam.finalize(tokens,sum_logprobs)

    return tokens

def batch_greedy_scoring(model, h, h_mask, device, sos=1, eos=2):

    batchSize, _, _ = h.size()
    tokens = torch.tensor([[sos]] * batchSize).to(device)
    token_lengths = torch.tensor([1]*batchSize).to(device)
    while True:
        out, out_mask = model.module.decode_one_step(tokens, token_lengths, h, h_mask)
        #print('out mask',out_mask.size())
        next_token = out[:,:,-1:].argmax(dim=1)
        #logprobs = F.log_softmax(out[:,:,-1:],dim=1).squeeze(-1)
        #print(logprobs[:,2])
        #sum_logprobs += logprobs.squeeze(0)
        next_token[tokens[:, -1] == eos] = eos
        #print(tokens[:,-1] != eos)
        token_lengths[tokens[:,-1] != eos]+=1
        tokens = torch.cat([tokens,next_token],dim=-1)
        endOfDecode = (tokens[:,-1] == eos).all()
        if endOfDecode:
            break

    return tokens

def greedy_scoring(model, h, h_mask, device, sos=1, eos=2):

    #batchSize, T = targets.size()
    batchSize, _, _ = h.size()
    tokens = torch.tensor([[sos]] * batchSize).to(device)
    token_lengths = torch.tensor([1]*batchSize).to(device)

    while True:
        out, out_mask = model.module.decode_one_step(tokens, token_lengths, h, h_mask)
        next_token = out[:,:,-1:].argmax(dim=1)
        #next_token[tokens[:, -1] == eos] = eos
        #token_lengths[tokens[:,-1] != eos]+=1
        if next_token[:,-1] ==eos:
            break
        token_lengths+=1
        tokens = torch.cat([tokens,next_token],dim=-1)
        #endOfDecode = (tokens[:,-1] == eos).all()

    return tokens

def parse_audio(audio_path: str, del_silence: bool = False, audio_extension: str = 'pcm') -> Tensor:
    signal = load_audio(audio_path, del_silence, extension=audio_extension)
    feature = torchaudio.compliance.kaldi.fbank(
        waveform=Tensor(signal).unsqueeze(0),
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        window_type='hamming'
    ).transpose(0, 1).numpy()

    feature -= feature.mean()
    feature /= np.std(feature)
    return torch.FloatTensor(feature).transpose(0, 1)

def single_infer(model,audio_path,tokenizer):
    s = time.time()
    device = 'cuda'
    label_path = os.path.join(DATASET_PATH, 'train', 'train_label')
    # data preprocessing and build tokenizer
    #tokenizer = preprocessing_spe(label_path, os.getcwd())
    #tokenizer = KoreanSpeechVocabulary(os.path.join(os.getcwd(), 'labels.csv'), output_unit='character')
    feature = parse_audio(audio_path, del_silence=True).to(device)
    #feature = features[0].to(device)
    input_length = torch.LongTensor([len(feature)]).to(device)
    #vocab = KoreanSpeechVocabulary(os.path.join(os.getcwd(), 'labels.csv'), output_unit='character')

    #if isinstance(model, nn.DataParallel):
    #    model = model.module
    model.eval()
    #model.device = device
    outputs, output_lengths = model.module.encoder_forward(feature.unsqueeze(0), input_length)
    #sentence = beamSearch(model, beamSize, outputs, output_lengths, device)
    sentence = greedy_scoring(model, outputs, output_lengths, device)
    #text = tokenizer.ids_to_text(sentence[0].tolist())
    text = tokenizer.label_to_string(sentence[0][1:])
    t = time.time()
    print(text,t-s)

    return text

def infer_test(model,tokenizer):
    audio_path = '/share/datas/Validation/audio/노인남여_노인대화77_F_문XX_60_제주_실내/노인남여_노인대화77_F_문XX_60_제주_실내_84091.WAV'
    s = single_infer(model,audio_path)
    text = tokenizer.ids_to_text(s[0].tolist())
    print(text)

@torch.no_grad()
def test_infer(model, testLoader, tokenizer):

    model.eval()
    results = list()
    beamSize = 2
    beam = BeamSearchDecoder(beamSize, eot=2)
    for features, input_lengths, paths in testLoader:
        s = time.time()
        #print(features.size(),paths)
        outputs, output_mask = model.module.encoder_forward(features.to('cuda'), input_lengths.to('cuda'))
        #sentence = batch_greedy_scoring(model, outputs, output_mask, device='cuda').cpu().numpy()
        #sentence = batch_beam_scoring(model, outputs, output_lengths, device='cuda').cpu().numpy()
        sentence = beamSearch(model, beam, beamSize, outputs, output_mask, device='cuda').cpu().numpy()
        for i in range(len(sentence)):
            text = tokenizer.label_to_string(sentence[i][1:input_lengths[i].item()])
            #print(text)
            results.append(
                {
                    'filename': paths[i].split('/')[-1],
                    'text' : text
                }
            )
        t = time.time()
        #print(t-s)

    return results

