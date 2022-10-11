import glob
import os
import cv2
import pandas
from collections import defaultdict
import pickle
import torch


# construct a .pkl file with information of all three sets
if __name__ == '__main__':
    total_dict = {}
    modes = ['train', 'dev', 'test']
    count = 0
    for mode in modes:
        total_dict[mode] = defaultdict(dict)
        with open(f'../data/{mode}.pkl', 'rb') as f:
            dd = pickle.load(f)
        for id, feature, label, signer in zip(dd['ids'], dd['features'], dd['label'], dd['signer']):
            total_dict[mode][id]['paths'] = feature
            total_dict[mode][id]['label'] = label
            total_dict[mode][id]['signer'] = signer

    with open('../data/data.pkl', 'wb') as f:
         pickle.dump(total_dict, f)
