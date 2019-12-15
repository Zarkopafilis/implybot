import fasttext

# model = fasttext.train_supervised('./discord.txt', lr=0.2, epoch=18, thread=4, pretrainedVectors='./crawl-300d-2M.vec')
# model.save_model("./retrained/latest.bin")

import fasttext
model = fasttext.train_unsupervised('./discord.txt', "cbow")
model.save_model("./discord.bin")
