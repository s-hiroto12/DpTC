import torch
import pandas as pd


def get_data(project,h, k, type):  
    root = 'data_cross/' + project + '/' + project +'_' + str(h) +'/'+str(k)+'/' + type + '/features.pkl'
    data = pd.read_pickle(root)

    features = data['code'].values
    labels = data['label'].values

    labels = torch.from_numpy(labels)
    return features, labels

# features, labels = get_data('ambari', 0, 0, 'train')
# print(features.shape)

data = pd.read_pickle('data_cross/ambari/ambari_0/0/train/features.pkl')
features = data['code'].values
labels = data['label'].values
# labels = torch.from_numpy(labels)
print(labels.shape)
print(labels)