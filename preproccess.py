import pandas as pd
import csv
import os

with open('./discord.txt', 'w+') as output:
    df = pd.read_csv('./old_general.csv')
    df.style.set_properties(**{'text-align': 'left'})
    df = df[df['from_bot'] != 1]
    df = df[df['is_attachment'] != 1]

    df = df['content']
    df = df[~df.str.startswith('>>')]

    df = df.str.strip()

    df = df.str.replace('"', "")
    df = df.str.replace('\'', "")
    df = df.str.replace(r"\n*", "", regex=True)
    df = df.str.replace(r"\r*", "", regex=True)
    df = df.str.replace(r"\s+", " ", regex=True)

    df = df.str.replace(r'[^\u0000-\u007F]+', "", regex=True)

    df = df[(df != '')]

    # save sentences
    # df.to_csv('./discord.txt', header=None, index=None, mode='a', quoting=csv.QUOTE_NONE, escapechar='')
    with open(os.path.join(os.getcwd(), 'discord.txt'), 'w') as outfile:
        for index, row in df.iteritems():
            outfile.write(row + "\n")

