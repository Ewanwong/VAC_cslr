import os
import pandas
from collections import defaultdict
import pickle
from sklearn import preprocessing


if __name__ == '__main__':
    sel = ['dev', 'test', 'train']

    gloss_list = []
    for i in sel:
        label_path = path = f"../data/annotations/manual/{i}.corpus.csv"
        f = pandas.read_csv(label_path, header=0, names=['data'])
        for entry in f['data']:
            idx, file_id, signer, glosses = entry.split('|')
            gloss_list += glosses.split(' ')

    gloss_list = [gloss for gloss in set(gloss_list) if len(gloss) > 0]
    gloss_list.insert(0, '<BLANK>')

    dict = defaultdict(int)
    for idx, gloss in enumerate(gloss_list):
        dict[gloss] = idx

    with open('../data/gloss_dict.pkl', 'wb') as f:
        pickle.dump(dict, f)
