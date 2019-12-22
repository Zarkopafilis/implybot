from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.data import BucketIterator

from implybot import layers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_nn(input_dim):
    INPUT_DIM = input_dim
    OUTPUT_DIM = INPUT_DIM
    ENC_HID_DIM = 512
    DEC_HID_DIM = 512
    ATTN_DIM = 64
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    enc = layers.Encoder(INPUT_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    attn = layers.Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)
    dec = layers.Decoder(OUTPUT_DIM, INPUT_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

    model = layers.Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)
    return model


def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


# def count_parameters(model: nn.Module):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
#
#
# print(f'The model has {count_parameters(model):,} trainable parameters')

def train(model: nn.Module, iterator,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          clip: float):

    model.train()

    epoch_loss = 0
    msg_count = 0
    for _, batch in enumerate(iterator):
        msg_count += 1
        if msg_count % 1000 == 0:
            print(f'Trained for {msg_count} messages')

        src = batch['src']
        trg = batch['trg']

        src = src.to(device, non_blocking=True)
        trg = trg.to(device, non_blocking=True)

        optimizer.zero_grad()

        output = model(src, trg)

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(output, trg.view(-1, 300))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model: nn.Module,
             iterator: BucketIterator,
             criterion: nn.Module):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for _, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0)  # turn off teacher forcing

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
