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
        self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor
    ) -> Tuple[Tensor, bool]:
        if tokens.shape[0] % self.beam_size != 0:
            raise ValueError(f"{tokens.shape}[0] % {self.beam_size} != 0")

        n_audio = tokens.shape[0] // self.beam_size
        if self.finished_sequences is None:  # for the first update
            self.finished_sequences = [{} for _ in range(n_audio)]
        logprobs = F.log_softmax(logits.float(), dim=1)[:,:,-1:].squeeze(-1)#.transpose(1,2)
        next_tokens, finished_sequences = [], []
        for i in range(n_audio):
            scores, sources, finished = {}, {}, {}

            # STEP 1: calculate the cumulative log probabilities for possible candidates
            for j in range(self.beam_size):
                idx = i * self.beam_size + j
                prefix = tokens[idx].tolist()
                for logprob, token in zip(*logprobs[idx].topk(self.beam_size + 1)):
                    new_logprob = (sum_logprobs[idx] + logprob).item()
                    sequence = tuple(prefix + [token.item()])
                    scores[sequence] = new_logprob
                    sources[sequence] = idx

            # STEP 2: rank the candidates and keep the top beam_size sequences for each audio
            saved = 0
            for sequence in sorted(scores, key=scores.get, reverse=True):
                if sequence[-1] == self.eot:
                    finished[sequence] = scores[sequence]
                else:
                    sum_logprobs[len(next_tokens)] = scores[sequence]
                    next_tokens.append(sequence)
                    #source_indices.append(sources[sequence])

                    saved += 1
                    if saved == self.beam_size:
                        break

            finished_sequences.append(finished)

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

def beamSearch(model, beamSize, h, h_mask, device, sos=1, eos=2):

    batchSize, _, T = h.size()
    #tokens = torch.tensor([sos]).repeat(batchSize, 1)
    tokens = torch.tensor([[sos]] * (batchSize * beamSize)).to(device)
    token_lengths = torch.tensor([1] * (batchSize * beamSize)).to(device)
    beam = BeamSearchDecoder(beamSize, eos)

    sum_logprobs = torch.zeros(batchSize*beamSize).to(device)
    #hyps = {'tokens':[],'socre':[]}
    h = h.squeeze(0).repeat(beamSize,1,1)
    h_mask = h_mask.squeeze(0).repeat(beamSize,1,1)

    while True:
        #print(h.size(),h_mask.size())
        #exit()
        #print(logits.size())
        #tokens = tokens.repeat_interleave(beamSize, dim=0).to(device)
        #print(tokens.size())
        logits, out_mask = model.module.decode_one_step(tokens, token_lengths, h, h_mask)
        tokens, completed = beam.update(tokens, logits, sum_logprobs)
        logprobs = F.log_softmax(logits.float(), dim=1)
        if completed:
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
        next_token[tokens[:, -1] == eos] = eos
        token_lengths[tokens[:,-1] != eos]+=1
        tokens = torch.cat([tokens,next_token],dim=-1)
        endOfDecode = (tokens[:,-1] == eos).all()
        if endOfDecode:
            break

    return tokens

def parse_audio(audio_path: str, del_silence: bool = False, audio_extension: str = 'pcm') -> Tensor:
    signal = load_audio(audio_path, del_silence, extension=audio_extension)
    feature = torchaudio.compliance.kaldi.fbank(
        waveform=Tensor(signal).unsqueeze(0),
        num_mel_bins=80,
        frame_length=20,
        frame_shift=10,
        window_type='hamming'
    ).transpose(0, 1).numpy()

    feature -= feature.mean()
    feature /= np.std(feature)
    return torch.FloatTensor(feature).transpose(0, 1)


#def single_infer(model, audio_path, beamSize, tokeniezr):
#def single_infer(model, features, beamSize, tokenizer):
def single_infer(model,audio_path):
    device = 'cuda'
    label_path = os.path.join(DATASET_PATH, 'train', 'train_label')
    # data preprocessing and build tokenizer
    tokenizer = preprocessing_spe(label_path, os.getcwd())
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
    text = tokenizer.ids_to_text(sentence[0].tolist())

    return text

def infer_test(model,tokenizer):
    audio_path = '/share/datas/Validation/audio/노인남여_노인대화77_F_문XX_60_제주_실내/노인남여_노인대화77_F_문XX_60_제주_실내_84091.WAV'
    s = single_infer(model,audio_path)
    text = tokenizer.ids_to_text(s[0].tolist())
    print(text)

