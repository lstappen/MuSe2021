import os
import numpy as np
import pandas as pd
import pickle


def get_data_partition(partition_file):
    vid2partition, partition2vid = {}, {}
    df = pd.read_csv(partition_file)

    for row in df.values:
        vid, partition = str(row[0]), row[-1]
        vid2partition[vid] = partition
        if partition not in partition2vid:
            partition2vid[partition] = []
        if vid not in partition2vid[partition]:
            partition2vid[partition].append(vid)

    return vid2partition, partition2vid


def segment_sample(sample, win_len, hop_len, segment_type='normal'):
    segmented_sample = []
    assert hop_len <= win_len and win_len >= 10

    segment_ids = sorted(set(sample['segment_id'].values))
    if segment_type in ['by_segs', 'by_segs_only']:
        for id in segment_ids:
            segment = sample[sample['segment_id'] == id]
            if segment_type == 'by_segs_only':
                segmented_sample.append(segment)
            else:
                for s_idx in range(0, len(segment), hop_len):
                    e_idx = min(s_idx + win_len, len(segment))
                    sub_segment = segment.iloc[s_idx:e_idx]
                    segmented_sample.append(sub_segment)
                    if e_idx == len(segment):
                        break
    elif segment_type == 'normal':
        for s_idx in range(0, len(sample), hop_len):
            e_idx = min(s_idx + win_len, len(sample))
            segment = sample.iloc[s_idx:e_idx]
            segmented_sample.append(segment)
            if e_idx == len(sample):
                break
    else:
        print('No such segmentation available.')
    return segmented_sample


def normalize_data(data, idx_list, column_name='feature'):
    train_data = np.row_stack(data['train'][column_name])
    train_mean = np.nanmean(train_data, axis=0)
    train_std = np.nanstd(train_data, axis=0)

    for partition in data.keys():
        for i in range(len(data[partition][column_name])):
            for s_idx, e_idx in idx_list:
                data[partition][column_name][i][:, s_idx:e_idx] = \
                    (data[partition][column_name][i][:, s_idx:e_idx] - train_mean[s_idx:e_idx]) / (
                            train_std[s_idx:e_idx] + 1e-6)  # standardize
                data[partition][column_name][i][:, s_idx:e_idx] = np.where(  # replace any nans with zeros
                    np.isnan(data[partition][column_name][i][:, s_idx:e_idx]), 0.0,
                    data[partition][column_name][i][:, s_idx:e_idx])

    return data


def load_data(task, paths, feature_set, emo_dim, normalize=True, norm_opts=None, win_len=200, hop_len=100, save=False,
              apply_segmentation=True):
    feature_path = paths['features']
    label_path = paths['labels']

    data_file_name = f'data_{task}_{"_".join(feature_set)}_{emo_dim}_{"norm_" if normalize else ""}{win_len}_' \
        f'{hop_len}{"_seg" if apply_segmentation else ""}.pkl'
    data_file = os.path.join(paths['data'], data_file_name)

    if os.path.exists(data_file):  # check if file of preprocessed data exists
        print(f'Find cached data "{os.path.basename(data_file)}".')
        data = pickle.load(open(data_file, 'rb'))
        return data

    print('Constructing data from scratch ...')
    data = {'train': {'feature': [], 'label': [], 'meta': []},
            'devel': {'feature': [], 'label': [], 'meta': []},
            'test': {'feature': [], 'label': [], 'meta': []}}
    vid2partition, partition2vid = get_data_partition(paths['partition'])
    feature_dims = [0] * len(feature_set)

    feature_idx = 2  # first to columns are timestamp and segment_id, features start with the third column

    for partition, vids in partition2vid.items():
        for vid in vids:
            sample_data = []
            segment_ids_per_step = []  # necessary for MuSe-Sent

            # parse features
            for i, feature in enumerate(feature_set):
                feature_file = os.path.join(feature_path, feature, vid + '.csv')
                assert os.path.exists(
                    feature_file), f'Error: no available "{feature}" feature file for video "{vid}": "{feature_file}".'
                df = pd.read_csv(feature_file)
                feature_dims[i] = df.shape[1] - feature_idx
                if i == 0:
                    feature_data = df  # keep timestamp and segment id in 1st feature val
                    segment_ids_per_step = df.iloc[:, 1]
                else:
                    feature_data = df.iloc[:, feature_idx:]
                sample_data.append(feature_data)
            data[partition]['feature_dims'] = feature_dims

            # parse labels
            label_file = os.path.join(label_path, emo_dim, vid + '.csv')
            assert os.path.exists(
                label_file), f'Error: no available "{emo_dim}" label file for video "{vid}": "{label_file}".'
            df = pd.read_csv(label_file)

            if task == 'sent':
                label = df['class_id'].values
                label_stretched = [label[s_id - 1] if not pd.isna(s_id) else pd.NA for s_id in segment_ids_per_step]
                label_data = pd.DataFrame(data=label_stretched, columns=[emo_dim])
            else:
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
            for i, segment in enumerate(samples):  # each segment has columns: timestamp, segment_id, features, labels
                n_emo_dims = 1
                if len(segment.iloc[:, feature_idx:-n_emo_dims].values) > 0:  # check if there are features
                    meta = np.column_stack((np.array([int(vid)] * len(segment)),
                                            segment.iloc[:, :feature_idx].values))  # video_id, timestamp, segment_id
                    data[partition]['meta'].append(meta)
                    data[partition]['label'].append(segment.iloc[:, -n_emo_dims:].values)
                    data[partition]['feature'].append(segment.iloc[:, feature_idx:-n_emo_dims].values)

    if normalize:
        idx_list = []

        assert norm_opts is not None and len(norm_opts) == len(feature_set)
        norm_opts = [True if norm_opt == 'y' else False for norm_opt in norm_opts]

        print(f'Feature dims: {feature_dims} ({feature_set})')
        feature_dims = np.cumsum(feature_dims).tolist()
        feature_dims = [0] + feature_dims

        norm_feature_set = []  # normalize data per feature and only if norm_opts is True
        for i, (s_idx, e_idx) in enumerate(zip(feature_dims[0:-1], feature_dims[1:])):
            norm_opt, feature = norm_opts[i], feature_set[i]
            if norm_opt:
                norm_feature_set.append(feature)
                idx_list.append([s_idx, e_idx])

        print(f'Normalized features: {norm_feature_set}')
        data = normalize_data(data, idx_list)

    if save:  # save loaded and preprocessed data
        print('Saving data...')
        pickle.dump(data, open(data_file, 'wb'))

    return data
