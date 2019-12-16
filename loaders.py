from __future__ import print_function, division
import fasttext
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import math

print("Loading fasttext model...")
ft = fasttext.load_model("C:\\Users\\zarkopafilis\\Desktop\\implybot\\pretrained\\cc.en.300.bin")
vocab_size = len(ft.words)

SOS_EMBEDDING = ft['<sos>']
EOS_EMBEDDING = ft['<eos>']
EMB_LEN = ft.get_dimension()

print("Loaded {} words, with {} vector length encoding.".format(len(ft.words), EMB_LEN))


def get_emb_len():
    return EMB_LEN


def min_distance(emb):
    min_d = EMB_LEN
    min_w = 'yeet'
    for w in ft.words:
        w_e = ft[w]
        d = [(a - b) ** 2 for a, b in zip(emb, w_e)]
        d = math.sqrt(sum(d))
        if d < min_d:
            min_d = d
            min_w = w

    return min_w


def embeddings_to_text(embeddings):
    words = [i for i in map(min_distance, embeddings)]
    return words.join(' ')


def make_embedding(message, max_len):
    tokens = message[0].split()
    embs = [i for i in map(lambda x: ft[x], tokens)]
    embs.insert(0, SOS_EMBEDDING)

    l = len(embs)
    length_diff = max_len - l
    if length_diff > 0:
        embs = embs + [EOS_EMBEDDING] * length_diff
    else:
        embs = embs[:max_len]
        embs[-1] = EOS_EMBEDDING

    return embs


class DiscordDataset(Dataset):

    def __init__(self, txt_file, max_len):
        self.chat_log = pd.read_csv(txt_file, header=None)
        self.max_len = max_len
        print("Loaded {} messages".format(len(self.chat_log)))

    def __len__(self):
        return len(self.chat_log)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        else:
            if idx == len(self.chat_log):
                idx = idx-1

            idx = [idx, idx+1]

        msgs = self.chat_log.iloc[idx, :]
        msgs = msgs.values.tolist()
        # strip to max length
        msgs = list(map(lambda x: make_embedding(x, self.max_len), msgs))
        msgs = np.array([msgs])
        msgs = msgs.astype('float32').reshape(-1, EMB_LEN)

        sample = {'src': msgs[:-1], 'trg': msgs[1:]}

        # transform to vector
        return sample