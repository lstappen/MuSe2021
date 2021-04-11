import numpy as np
import os
import pandas as pd
import torch
from sklearn.metrics import f1_score

from loss import get_segment_wise_logits, get_segment_wise_labels


def calc_ccc(preds, labels):
    preds = np.row_stack(preds)[:, 0]
    labels = np.row_stack(labels)[:, 0]

    preds_mean, labels_mean = np.mean(preds), np.mean(labels)
    cov_mat = np.cov(preds, labels)
    covariance = cov_mat[0, 1]
    preds_var, labels_var = cov_mat[0, 0], cov_mat[1, 1]

    ccc = 2.0 * covariance / (preds_var + labels_var + (preds_mean - labels_mean) ** 2)
    return ccc


def write_predictions(full_metas, full_preds, prediction_path):
    assert prediction_path != ''

    csv_dir = os.path.join(prediction_path, 'csv')
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    columns = ['timestamp', 'segment_id', 'value']

    for i, (meta, pred) in enumerate(zip(full_metas, full_preds)):
        vid = int(meta[0, 0])
        sample_file_name = f'{vid}.csv'

        sample_data = np.column_stack([meta[:, 1], meta[:, 2], pred[:, 0]])  # 'timestamp', 'segment_id', 'value'
        df = pd.DataFrame(sample_data, columns=columns)
        df = df[df.segment_id != 0]  # remove any existing zero-padding
        df[['timestamp', 'segment_id']] = df[['timestamp', 'segment_id']].astype(np.int)

        df.to_csv(os.path.join(csv_dir, sample_file_name), index=False)


def write_predictions_for_sent(full_metas, full_preds, full_metas_stepwise, full_logits, prediction_path):
    assert prediction_path != ''

    csv_dir = os.path.join(prediction_path, 'csv_segment_wise')
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    columns = ['timestamp', 'segment_id', 'value']

    # Part 1: Write segment-wise predictions (for test predictions)
    df = pd.DataFrame(columns=columns)
    df_idx = 0
    vid = -1
    for i, (meta, pred) in enumerate(zip(full_metas, full_preds)):
        new_vid = int(meta[0])
        if vid == -1:
            vid = new_vid
            sample_file_name = f'{vid}.csv'
        if new_vid != vid:
            df[['timestamp', 'segment_id']] = df[['timestamp', 'segment_id']].astype(np.int)
            df.to_csv(os.path.join(csv_dir, sample_file_name), index=False)
            df = pd.DataFrame(columns=columns)
            vid = new_vid
            sample_file_name = f'{vid}.csv'
            df_idx = 0

        df.loc[df_idx] = [meta[1], meta[2], pred]
        df_idx += 1
    df[['timestamp', 'segment_id']] = df[['timestamp', 'segment_id']].astype(np.int)
    df.to_csv(os.path.join(csv_dir, sample_file_name), index=False)

    # Part 2: write timestep-wise logits, as needed for late fusion
    csv_dir = os.path.join(prediction_path, 'csv')
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    columns = ['timestamp', 'segment_id', 'c_0', 'c_1', 'c_2', 'c_3', 'c_4']

    df = pd.DataFrame(columns=columns)
    df_idx = 0
    vid = -1
    for i, (meta, pred) in enumerate(zip(full_metas_stepwise, full_logits)):
        new_vid = int(meta[0, 0])
        if vid == -1:
            vid = new_vid
            sample_file_name = f'{vid}.csv'
        if new_vid != vid:
            df[['timestamp', 'segment_id']] = df[['timestamp', 'segment_id']].astype(np.int)
            df.to_csv(os.path.join(csv_dir, sample_file_name), index=False)
            df = pd.DataFrame(columns=columns)
            vid = new_vid
            sample_file_name = f'{vid}.csv'
            df_idx = 0

        sample_data = pd.DataFrame(np.column_stack([meta[:, 1], meta[:, 2], pred]), columns=columns)
        sample_data = sample_data[sample_data.segment_id != 0]  # remove any existing zero-padding
        df = pd.concat([df, sample_data], axis=0)
        # df.loc[df_idx] = [meta[1], meta[2], pred]
        df_idx += 1
    df[['timestamp', 'segment_id']] = df[['timestamp', 'segment_id']].astype(np.int)
    df.to_csv(os.path.join(csv_dir, sample_file_name), index=False)


def evaluate(task, model, data_loader, criterion, use_gpu=False, predict=False, prediction_path=''):
    losses, sizes = 0, 0
    full_preds = []
    if predict:
        full_metas = []
    else:
        full_labels = []

    if task == 'sent':
        full_logits = []
        full_metas_stepwise = []

    model.eval()
    with torch.no_grad():
        for batch, batch_data in enumerate(data_loader, 1):
            features, feature_lens, labels, metas = batch_data
            batch_size = features.size(0)

            if use_gpu:
                model.cuda()
                features = features.cuda()
                feature_lens = feature_lens.cuda()
                labels = labels.cuda()

            preds = model(features, feature_lens)

            if task == 'sent':
                logits_stepwise = preds
                metas_stepwise = metas
                logits = get_segment_wise_logits(preds, feature_lens)
                preds = torch.argmax(logits, dim=1)
                metas = get_segment_wise_labels(metas)

            if predict:
                full_metas.append(metas.detach().squeeze(0).numpy())
                if task == 'sent':
                    full_metas_stepwise.append(metas_stepwise.detach().squeeze(0).numpy())
                    full_logits.append(logits_stepwise.cpu().detach().squeeze(0).numpy())
            else:
                if task == 'sent':
                    labels = get_segment_wise_labels(labels)
                    loss = criterion(logits, labels[:, 0])
                else:
                    loss = criterion(preds[:, :, 0], labels[:, :, 0], feature_lens)

                losses += loss.item() * batch_size
                sizes += batch_size

                full_labels.append(labels.cpu().detach().squeeze(0).numpy())
            full_preds.append(preds.cpu().detach().squeeze(0).numpy())

        if predict:
            if task == 'sent':
                # segment-wise predictions
                write_predictions_for_sent(full_metas, full_preds, full_metas_stepwise, full_logits, prediction_path)
            else:
                write_predictions(full_metas, full_preds, prediction_path)
            return
        else:
            if task == 'sent':
                score = f1_score(full_labels, full_preds, average='macro')
            else:
                score = calc_ccc(full_preds, full_labels)
            total_loss = losses / sizes
            return total_loss, score
