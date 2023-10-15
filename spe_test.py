from modules.process_asr_text_tokenizer import build_spe_model
from modules.sentencepiece_tokenizer import SentencePieceTokenizer
import torch

#build_spe_model('/workspace/ALoeProj2023/temp','./test_text.txt')
spe_model = SentencePieceTokenizer('./temp/tokenizer_spe_unigram_v5000_bos_eos/tokenizer.model')

test = '이 두 소설은 줄거리가 유사해요.'
print('orin',test)
'''
tokens = spe_model.text_to_tokens(test)
print(tokens)
test = spe_model.tokens_to_text(tokens)
print(test)
'''
#print(spe_model.bos_token)
#print(spe_model.eos_token)
#exit()

vocab = spe_model.vocab
print(vocab)


ids = spe_model.text_to_ids(test)
print(ids)
text = spe_model.ids_to_text(ids)
print(text)
text = spe_model.ids_to_text(torch.tensor(ids).tolist())
print(text)
