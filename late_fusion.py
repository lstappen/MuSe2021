import argparse
import os
import glob
import sys
import torch
import numpy as np
import random
import pandas as pd
from datetime import datetime
from dateutil import tz
from torch.nn import CrossEntropyLoss

import config
from loss import CCCLoss
from utils import Logger
from train import train_model
from eval import evaluate
from model import Model
from dataset import MuSeDataset
from data_parser import get_data_partition, segment_sample


def parse_args():
    parser = argparse.ArgumentParser(description='Late Fusion')

    parser.add_argument('--task', type=str, required=True, choices=['wilder', 'sent', 'physio', 'stress'],
                        help='Specify the task (wilder, sent, physio or stress).')
    parser.add_argument('--preds_path', type=str, required=True,
                        help='Specify the directory that contains the predictions that are to be fused.')
    parser.add_argument('--emo_dim', default='arousal',
                        help='Specify the emotion dimension (default: arousal).')
    parser.add_argument('--win_len', type=int, default=200,
                        help='Specify the window length for segmentation (default: 200 frames).')
    parser.add_argument('--hop_len', type=int, default=100,
                        help='Specify the hop length to for segmentation (default: 100 frames).')
    parser.add_argument('--d_rnn', type=int, default=64,
                        help='Specify the number of hidden states in the RNN (default: 64).')
    parser.add_argument('--rnn_n_layers', type=int, default=1,
                        help='Specify the number of layers for the RNN (default: 1).')
    parser.add_argument('--rnn_bi', action='store_true',
                        help='Specify whether the RNN is bidirectional or not (default: False).')
    parser.add_argument('--d_fc_out', type=int, default=64,
                        help='Specify the number of hidden neurons in the output layer (default: 64).')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Specify the number of epochs (default: 100).')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Specify the batch size (default: 256).')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Specify initial learning rate (default: 0.0001).')
    parser.add_argument('--seed', type=int, default=101,
                        help='Specify the initial random seed (default: 101).')
    parser.add_argument('--n_seeds', type=int, default=5,
                        help='Specify number of random seeds to try (default: 5).')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Specify whether to use gpu for training (default: False).')
    parser.add_argument('--cache', action='store_true',
                        help='Specify whether to cache data as pickle file (default: False).')
    parser.add_argument('--predict', action='store_true',
                        help='Specify when no test labels are available; test predictions will be saved (default: False).')
    parser.add_argument('--save', action='store_true',
                        help='Specify whether to save trained model (default: False).')
    parser.add_argument('--eval_model', type=str, default=None,
                        help='Specify model which is to be evaluated; no training with this option (default: False).')

    args = parser.parse_args()
    return args


def load_data(task, paths, emo_dim, win_len=200, hop_len=100, apply_segmentation=True):
    label_path = paths['labels']
    preds_paths = glob.glob(paths['preds'] + '/*/csv/')
    print(preds_paths)

    data = {'train': {'feature': [], 'label': [], 'meta': []},
            'devel': {'feature': [], 'label': [], 'meta': []},
            'test': {'feature': [], 'label': [], 'meta': []}}
    vid2partition, partition2vid = get_data_partition(paths['partition'])
    feature_dims = [1] * len(preds_paths)
    feature_idx = 2  # first to columns are timestamp and segment_id, features start with the third column
    for partition, vids in partition2vid.items():
        for vid in vids:
            sample_data = []
            segment_ids_per_step = []  # necessary for MuSe-Sent

            # preds
            for i, path in enumerate(preds_paths):
                preds_file = os.path.join(path, vid + '.csv')
                assert os.path.exists(
                    preds_file), f'Error: no available "{path}" preds file for video "{vid}": "{preds_file}".'
                df = pd.read_csv(preds_file)
                if i == 0:
                    preds_data = df  # keep timestamp and segment id in 1st preds val
                    segment_ids_per_step = df.iloc[:, 1]
                else:
                    preds_data = df.iloc[:, feature_idx:]
                sample_data.append(preds_data)
            data[partition]['feature_dims'] = feature_dims

            # label
            label_file = os.path.join(label_path, emo_dim, vid + '.csv')
            assert os.path.exists(
                label_file), f'Error: no available "{emo_dim}" label file for video "{vid}": "{label_file}".'
            df = pd.read_csv(label_file)

            if task == 'sent':
                label = df['class_id'].values
                label_stretched = [label[s_id - 1] if not pd.isna(s_id) else pd.NA for s_id in segment_ids_per_step]
                label_data = pd.DataFrame(data=label_stretched, columns=[emo_dim])
            else:  # task == 'wilder'
                label_data = pd.DataFrame(data=df['value'].values, columns=[emo_dim])
            sample_data.append(label_data)

            # concat
            sample_data = pd.concat(sample_data, axis=1)
            if partition != 'test':
                sample_data = sample_data.dropna()

            # segment
            if apply_segmentation:
                if task == 'sent':
                    seg_type = 'by_segs_only' if partition != 'train' else 'by_segs'
                    samples = segment_sample(sample_data, win_len, hop_len, seg_type)
                elif task in ['wilder', 'physio', 'stress']:
                    if partition == 'train':
                        samples = segment_sample(sample_data, win_len, hop_len, 'normal')
                    else:
                        samples = [sample_data]
            else:
                if task == 'sent':
                    samples = segment_sample(sample_data, win_len, hop_len, 'by_segs_only')
                else:
                    samples = [sample_data]

            # store
            for i, segment in enumerate(samples):
                n_emo_dims = 1
                if len(segment.iloc[:, 2:-n_emo_dims].values) > 0:
                    meta = np.column_stack((np.array([int(vid)] * len(segment)),
                                            segment.iloc[:, :2].values))  # video_id, timestamp, segment_id
                    data[partition]['meta'].append(meta)
                    data[partition]['label'].append(segment.iloc[:, -n_emo_dims:].values)
                    data[partition]['feature'].append(segment.iloc[:, 2:-n_emo_dims].values)

    return data


def main(args):
    np.random.seed(10)
    random.seed(10)

    print('Loading data ...')
    data = load_data(args.task, args.paths, args.emo_dim, args.win_len, args.hop_len)
    data_loader = {}
    for partition in data.keys():
        set = MuSeDataset(data, partition)
        batch_size = args.batch_size if partition == 'train' else 1
        shuffle = True if partition == 'train' else False
        data_loader[partition] = torch.utils.data.DataLoader(set, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    args.d_in = data_loader['train'].dataset.get_feature_dim()

    if args.task == 'sent':
        args.n_targets = max([x[0, 0] for x in data['train']['label']]) + 1  # number of classes
        criterion = CrossEntropyLoss()
        score_str = 'Macro-F1'
    else:
        args.n_targets = 1
        criterion = CCCLoss()
        score_str = 'CCC'

    if args.eval_model is None:  # Train and validate for each seed
        seeds = range(args.seed, args.seed + args.n_seeds)
        val_losses, val_scores, best_model_files, test_scores = [], [], [], []

        for seed in seeds:
            torch.manual_seed(seed)

            model = Model(args)

            print('=' * 50)
            print('Training model... [seed {}]'.format(seed))

            val_loss, val_score, best_model_file = train_model(args.task, model, data_loader, args.epochs, args.lr,
                                                               args.paths['model'], seed, args.use_gpu, criterion)
            if not args.predict:
                test_loss, test_score = evaluate(args.task, model, data_loader['test'], criterion, args.use_gpu)
                test_scores.append(test_score)
            val_losses.append(val_loss)
            val_scores.append(val_score)
            best_model_files.append(best_model_file)

        best_idx = val_scores.index(max(val_scores))  # find best performing seed
        print('=' * 50)
        print(f'Best {score_str} on [Val] for seed {seeds[best_idx]}: '
              f'[Val {score_str}]: {val_scores[best_idx]:7.4f}'
              f"{f' | [Test {score_str}]: {test_scores[best_idx]:7.4f}' if not args.predict else ''}")
        print('=' * 50)
        model_file = best_model_files[best_idx]

    else:  # Evaluate existing model (No training)
        model_file = args.eval_model
        model = torch.load(model_file)
        _, valid_score = evaluate(args.task, model, data_loader['devel'], criterion, args.use_gpu)
        print(f'Evaluating {model_file}:')
        print(f'[Val {score_str}]: {valid_score:7.4f}')
        if not args.predict:
            _, test_score = evaluate(args.task, model, data_loader['test'], criterion, args.use_gpu)
            print(f'[Test {score_str}]: {test_score:7.4f}')

    if args.predict:
        print('Predicting test samples...')
        best_model = torch.load(model_file)
        evaluate(args.task, best_model, data_loader['test'], criterion, args.use_gpu, predict=True,
                 prediction_path=args.paths['predict'])

    if not args.save and not args.eval_model:
        if os.path.exists(model_file):
            os.remove(model_file)

    print('Done.')


if __name__ == '__main__':
    args = parse_args()

    args.log_file_name = 'FUSION_[{}]_[{}]_[{}_{}_{}]_[{}_{}]'.format(
        datetime.now(tz=tz.gettz()).strftime("%Y-%m-%d-%H-%M"), args.emo_dim,
        args.d_rnn, args.rnn_n_layers, args.rnn_bi, args.lr, args.batch_size)

    # adjust your paths in config.py
    args.paths = {'log': os.path.join(config.LOG_FOLDER, args.task),
                  'data': os.path.join(config.DATA_FOLDER, args.task),
                  'model': os.path.join(config.MODEL_FOLDER, args.task, args.log_file_name)}
    if args.predict:
        args.paths['predict'] = os.path.join(config.PREDICTION_FOLDER, args.task, args.log_file_name)
    for folder in args.paths.values():
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

    args.paths.update({'preds': os.path.join(args.preds_path, args.task),
                       'labels': config.PATH_TO_LABELS[args.task],
                       'partition': config.PARTITION_FILES[args.task]})

    sys.stdout = Logger(os.path.join(args.paths['log'], args.log_file_name + '.txt'))
    print(' '.join(sys.argv))

    main(args)
