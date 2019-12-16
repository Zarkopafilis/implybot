from __future__ import unicode_literals, print_function, division
import time

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from implybot import train_util
from implybot import loaders

print("Starting...")

MAX_MSG_LEN = 20
BATCH_SIZE = 32

N_EPOCHS = 10
CLIP = 1

print("> Preparing Dataset Loader")
train_iterator = loaders.DiscordDataset("C:\\Users\\zarkopafilis\\Desktop\\implybot\\discord.txt", MAX_MSG_LEN)

train_iterator = DataLoader(dataset=train_iterator, batch_size=BATCH_SIZE, shuffle=True)

print("> Making Model")
model = train_util.make_nn(loaders.get_emb_len())
optimizer = optim.Adam(model.parameters())

criterion = nn.MSELoss()  # make sure this ignores the <sos> and <eos> tokens

print("> Training start")
for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss = train_util.train(model, train_iterator, optimizer, criterion, CLIP)
    end_time = time.time()

    epoch_mins, epoch_secs = train_util.epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {train_util.math.exp(train_loss):7.3f}')

print("> Training finished")

print("> Saving model")
train_util.torch.save(model, "C:\\Users\\zarkopafilis\\Desktop\\implybot\\torchmodel.pt")

model.eval()

test_inputs = ["hey bot whatsup?", "marios wyd", "F", "bsd > linux?", "php bad"]

print("\n> Running test inputs: \n\n")
for msg in test_inputs:
    src = train_util.make_embedding(msg, MAX_MSG_LEN)
    trg = [train_util.SOS_EMBEDDING]

    output = model(src, trg, 0)  # turn off teacher forcing

    output = output[1:].view(-1, output.shape[-1])
    trg = trg[1:].view(-1)

    o = train_util.embeddings_to_text(trg);
    print("Q: {} - A: {}".format(msg, o))