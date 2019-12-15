import pandas as pd
import fasttext

ft = fasttext.load_model("C:\\Users\\zarkopafilis\\Desktop\\implybot\\pretrained\\cc.en.300.bin")

with open('./discord.txt', 'w+') as output:
    df = pd.read_csv('./old_general.csv')

    df = df[df['from_bot'] != 1]
    df = df[df['is_attachment'] != 1]

    df = df['content']
    df = df[~df.str.startswith('>>')]

    df = df.str.strip()

    df = df.replace(r'\"*', '', regex=True)

    df = df.replace(r'\n*', '', regex=True)
    df = df.replace(r'\r*', '', regex=True)

    df = df.replace(r'[^\u0000-\u007F]+', '', regex=True)

    # save sentences
    df.to_csv('./discord.txt', header=None, index=None, mode='a')
