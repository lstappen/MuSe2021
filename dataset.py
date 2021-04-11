import torch
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence


class MuSeDataset(Dataset):
    def __init__(self, data, partition):
        super(MuSeDataset, self).__init__()
        self.partition = partition
        features, labels = data[partition]['feature'], data[partition]['label']
        metas = data[partition]['meta']
        self.feature_dim = features[0].shape[-1]
        self.n_samples = len(features)

        feature_lens = []
        for feature in features:
            feature_lens.append(len(feature))
        self.feature_lens = torch.tensor(feature_lens)
        if partition == 'train':
            self.features = pad_sequence([torch.tensor(feature, dtype=torch.float) for feature in features],
                                         batch_first=True)
            self.labels = pad_sequence([torch.tensor(label, dtype=torch.float) for label in labels], batch_first=True)
            self.metas = pad_sequence([torch.tensor(meta) for meta in metas], batch_first=True)

        else:
            self.features = [torch.tensor(feature, dtype=torch.float) for feature in features]
            self.labels = [torch.tensor(label, dtype=torch.float) for label in labels]
            self.metas = [torch.tensor(meta) for meta in metas]

    def get_feature_dim(self):
        return self.feature_dim

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        feature = self.features[idx]
        feature_len = self.feature_lens[idx]
        label = self.labels[idx]
        meta = self.metas[idx]

        sample = feature, feature_len, label, meta
        return sample
