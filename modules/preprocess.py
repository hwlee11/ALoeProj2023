import re
import os
import pandas as pd

import json
from modules.process_asr_text_tokenizer import build_spe_model
from modules.sentencepiece_tokenizer import SentencePieceTokenizer



def bracket_filter(sentence, mode='phonetic'):
    new_sentence = str()

    if mode == 'phonetic':
        flag = False

        for ch in sentence:
            if ch == '(' and flag is False:
                flag = True
                continue
            if ch == '(' and flag is True:
                flag = False
                continue
            if ch != ')' and flag is False:
                new_sentence += ch

    elif mode == 'spelling':
        flag = True

        for ch in sentence:
            if ch == '(':
                continue
            if ch == ')':
                if flag is True:
                    flag = False
                    continue
                else:
                    flag = True
                    continue
            if ch != ')' and flag is True:
                new_sentence += ch

    else:
        raise ValueError("Unsupported mode : {0}".format(mode))

    return new_sentence


def special_filter(sentence, mode='phonetic', replace=None):
    SENTENCE_MARK = ['?', '!', '.']
    NOISE = ['o', 'n', 'u', 'b', 'l']
    EXCEPT = ['/', '+', '*', '-', '@', '$', '^', '&', '[', ']', '=', ':', ';', ',']

    new_sentence = str()
    for idx, ch in enumerate(sentence):
        if ch not in SENTENCE_MARK:
            if idx + 1 < len(sentence) and ch in NOISE and sentence[idx + 1] == '/':
                continue

        if ch == '#':
            new_sentence += '샾'

        elif ch == '%':
            if mode == 'phonetic':
                new_sentence += replace
            elif mode == 'spelling':
                new_sentence += '%'

        elif ch not in EXCEPT:
            new_sentence += ch

    pattern = re.compile(r'\s\s+')
    new_sentence = re.sub(pattern, ' ', new_sentence.strip())
    return new_sentence


def sentence_filter(raw_sentence, mode, replace=None):
    return special_filter(bracket_filter(raw_sentence, mode), mode, replace)


def load_label(filepath):
    char2id = dict()
    id2char = dict()

    ch_labels = pd.read_csv(filepath, encoding="utf-8")

    id_list = ch_labels["id"]
    char_list = ch_labels["char"]
    freq_list = ch_labels["freq"]

    for (id_, char, freq) in zip(id_list, char_list, freq_list):
        char2id[char] = id_
        id2char[id_] = char
    return char2id, id2char


def sentence_to_target(sentence, char2id):
    target = str()

    for ch in sentence:
        try:
            target += (str(char2id[ch]) + ' ')
        except KeyError:
            continue

    return target[:-1]


def generate_character_script(data_df, labels_dest):
    print('[INFO] create_script started..')
    char2id, id2char = load_label(os.path.join(labels_dest, "labels.csv"))

    with open(os.path.join(labels_dest,"transcripts.txt"), "w+") as f:
        for audio_path, transcript in data_df.values:
            char_id_transcript = sentence_to_target(transcript, char2id)
            f.write(f'{audio_path}\t{transcript}\t{char_id_transcript}\n')

def generate_script(data_df, labels_dest):
    print('[INFO] create_script started..')

    with open(os.path.join(labels_dest,"alltext.txt"), "w+") as f:
        for audio_path, transcript in data_df.values:
            f.write(f'{transcript}\n')

def generate_spe_script_pd(data_df, spe_model):
    print('[INFO] create_script started..')

    with open(os.path.join(os.getcwd(),"transcripts.txt"), "w+") as f:
        for audio_path, transcript in data_df.values:
            #char_id_transcript = spe_model(transcript, char2id)
            spe_id_list = spe_model.text_to_ids(transcript)
            spe_id_transcript = ' '.join(map(str,spe_id_list))
            f.write(f'{audio_path}\t{transcript}\t{spe_id_transcript}\n')

def preprocessing(transcripts_dest, labels_dest):
    transcript_df = pd.read_csv(transcripts_dest)
    generate_character_script(transcript_df, labels_dest)

    print('[INFO] Preprocessing is Done')

def preprocessing_spe(transcripts_dest, labels_dest):
    transcript_df = pd.read_csv(transcripts_dest)
    generate_script(transcript_df, labels_dest)
    # build spe
    build_spe_model(labels_dest,'alltext.txt')
    spe_model = SentencePieceTokenizer(os.path.join(os.getcwd(),'tokenizer_spe_unigram_v5000_bos_eos/tokenizer.model'))
    generate_spe_script_pd(transcript_df, spe_model)

    print('[INFO] Preprocessing is Done')
    #return tokenizer
    return spe_model

def json_parse(path):
    data_dict = dict()
    f = open(path)
    while True:
        line = f.readline()
        if line == "":
            break
        line = line.rstrip()
        pJ = open(line)
        json_data_dict = json.load(pJ)
        fileId = json_data_dict['발화정보']['scriptId']
        label = json_data_dict['발화정보']['stt']
        audioPath = json_data_dict['발화정보']['fileNm']
        if label.find("FP") != -1:
            continue
        data_dict[fileId] = (label,audioPath)
        pJ.close()

    print(os.path.join(os.getcwd(),"text.txt"))
    #with open(os.path.join(os.getcwd(),"transcripts.txt"), "w+") as ff:
    with open(os.path.join(os.getcwd(),"text.txt"), "w+") as ff:
        #for audio_path, transcript in data_df.values:
        fileIds = data_dict.keys()
        for i in fileIds:
            transcript, audio_path = data_dict[i]
            ff.write(f'{transcript}\n')
    f.close()

    return data_dict

def generate_spe_script_dict(data_dict, spe_model):
    print('[INFO] create_script started..')

    with open(os.path.join(os.getcwd(),"transcripts.txt"), "w+") as f:
        fileIds = data_dict.keys()
        for i in fileIds:
            transcript, audio_path = data_dict[i]
            spe_id_list = spe_model.text_to_ids(transcript)
            spe_id_transcript = ' '.join(map(str,spe_id_list))
            f.write(f'{audio_path}\t{transcript}\t{spe_id_transcript}\n')

def preprocessing_test(transcripts_dest, labels_dest):
    #transcript_df = pd.read_csv(transcripts_dest)
    # prepare data
    data_dict = json_parse(transcripts_dest)
    
    # build spe
    build_spe_model('/workspace/ALoeProj2023/temp','./text.txt')
    spe_model = SentencePieceTokenizer(os.path.join(os.getcwd(),'temp/tokenizer_spe_unigram_v5000_bos_eos/tokenizer.model'))
    generate_spe_script_dict(data_dict, spe_model)
    # gen label script

    #generate_character_script(transcript_df, labels_dest)

    print('[INFO] Preprocessing is Done')
    return spe_model



