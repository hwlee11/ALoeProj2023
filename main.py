import torch
import queue
import os
import random
import warnings
import time
import json
import argparse
from glob import glob

from modules.preprocess import preprocessing_spe, get_spe
from modules.preprocess import preprocessing
from modules.trainer import trainer
from modules.utils import (
    get_optimizer,
    get_criterion,
    get_lr_scheduler,
)
from modules.audio import (
    FilterBankConfig,
    MelSpectrogramConfig,
    MfccConfig,
    SpectrogramConfig,
)
from modules.model import build_model
from modules.vocab import KoreanSpeechVocabulary
from modules.data import split_dataset, collate_fn, testDataset, collate_test_fn
from modules.utils import Optimizer
from modules.metrics import get_metric
from modules.AEDInference import single_infer, test_infer


from torch.utils.data import DataLoader

import nova
from nova import DATASET_PATH


def bind_model(model, config, tokenizer, optimizer=None):
    def save(path, *args, **kwargs):
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, os.path.join(path, 'model.pt'))
        print('Model saved')

    def load(path, *args, **kwargs):
        state = torch.load(os.path.join(path, 'model.pt'))
        model.load_state_dict(state['model'])
        if 'optimizer' in state and optimizer:
            optimizer.load_state_dict(state['optimizer'])
        print('Model loaded')

    # 추론
    def infer(path, **kwargs):
        return inference(path, model, config, tokenizer)

    nova.bind(save=save, load=load, infer=infer)  # 'nova.bind' function must be called at the end.


def inference(path, model, config, tokenizer, **kwargs):
    model.eval()

    results = []
    test_datas = glob(os.path.join(path, '*'))
    test_set = testDataset(test_datas, config)
    test_loader = DataLoader(
                test_set,
                batch_size=32,
                shuffle=False,
                collate_fn=collate_test_fn,
                num_workers=8
    )
    results = test_infer(model, test_loader, tokenizer)

    return sorted(results, key=lambda x: x['filename'])



if __name__ == '__main__':

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
    args.add_argument('--feature_extract_by', type=str, default='kaldi')
    args.add_argument('--sample_rate', type=int, default=16000)
    args.add_argument('--frame_length', type=int, default=25)
    args.add_argument('--frame_shift', type=int, default=10)
    args.add_argument('--n_mels', type=int, default=80)
    args.add_argument('--freq_mask_para', type=int, default=27)
    args.add_argument('--time_mask_num', type=int, default=10)
    args.add_argument('--freq_mask_num', type=int, default=2)
    args.add_argument('--normalize', type=bool, default=True)
    args.add_argument('--del_silence', type=bool, default=True)
    args.add_argument('--spec_augment', type=bool, default=True)
    args.add_argument('--input_reverse', type=bool, default=False)

    config = args.parse_args()
    warnings.filterwarnings('ignore')

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    device = 'cuda' if config.use_cuda == True else 'cpu'
    if hasattr(config, "num_threads") and int(config.num_threads) > 0:
        torch.set_num_threads(config.num_threads)

    tokenizer = KoreanSpeechVocabulary(os.path.join(os.getcwd(), 'labels.csv'), output_unit='character')
    vocab_size = len(tokenizer) # 
    blank_id = tokenizer.blank_id
    model = build_model(config, vocab_size, device)
    print(model)
    print(vocab_size, blank_id)

    """
    tokenizer = get_spe()#preprocessing_spe(, os.getcwd())
    # build AED model
    vocab_size = 5000 + 1 #len(tokenizer) # 
    blank_id = 5000
    model = build_model(config, vocab_size, device)
    print(model)
    print(vocab_size, blank_id)
    #model = build_model(config, vocab_size, device)
    """
    optimizer = get_optimizer(model, config)
    bind_model(model, config, tokenizer,optimizer=optimizer)
    metric = get_metric(metric_name='CER', vocab=tokenizer)

    if config.pause:
        nova.paused(scope=locals())

    if config.mode == 'train':

        # prepare data
        config.dataset_path = os.path.join(DATASET_PATH, 'train', 'train_data')
        label_path = os.path.join(DATASET_PATH, 'train', 'train_label')
        preprocessing(label_path, os.getcwd())
        #preprocessing_spe(tokenizer, label_path, os.getcwd())
        # data preprocessing and build tokenizer
        #vocab_size = len(tokenizer.vocab)
    
        train_dataset, valid_dataset = split_dataset(config, os.path.join(os.getcwd(), 'transcripts.txt'))


        lr_scheduler = get_lr_scheduler(config, optimizer, len(train_dataset))
        optimizer = Optimizer(optimizer, None, None, config.max_grad_norm)
        #optimizer = Optimizer(optimizer, lr_scheduler, int(len(train_dataset)*config.num_epochs), config.max_grad_norm)
        ctc, nll = get_criterion(config, blank_id)

        num_epochs = config.num_epochs
        num_workers = config.num_workers

        train_begin_time = time.time()

        train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=config.num_workers
        )
        valid_loader = DataLoader(
                valid_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=config.num_workers
        )

        init_lr = 1e-3
        optimizer.set_lr(init_lr)
        #lr = init_lr

        for epoch in range(num_epochs):
            print('[INFO] Epoch %d start' % epoch)

            # train
            
            if epoch == 4:
                optimizer.count = 3700

            model.train()
            model, train_loss, train_cer = trainer(
                epoch,
                'train',
                config,
                train_loader,
                optimizer,
                model,
                ctc,
                nll,
                metric,
                train_begin_time,
                tokenizer,
                device
            )

            print('[INFO] Epoch %d (Training) Loss %0.4f CER %0.4f' % (epoch, train_loss, train_cer))
            
            #if epoch == 9 or epoch == 19:
            #    lr /=10
            #    optimizer.set_lr(lr)
            
            # valid
            model.eval()
            model, valid_loss, valid_cer = trainer(
                epoch,
                'valid',
                config,
                valid_loader,
                optimizer,
                model,
                ctc,
                nll,
                metric,
                train_begin_time,
                tokenizer,
                device
            )

            print('[INFO] Epoch %d (Validation) Loss %0.4f  CER %0.4f' % (epoch, valid_loss, valid_cer))

            nova.report(
                summary=True,
                epoch=epoch,
                train_loss=train_loss,
                train_cer=train_cer,
                val_loss=valid_loss,
                val_cer=valid_cer
            )

            if epoch % config.checkpoint_every == 0:
                nova.save(epoch)

            torch.cuda.empty_cache()
            print(f'[INFO] epoch {epoch} is done')
        print('[INFO] train process is done')
