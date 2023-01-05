import argparse
import logging
import os
import numpy as np
import pandas as pd 
import pickle

from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser()

parser.add_argument('-embeddings', type=str)
parser.add_argument('-dir', type=str)
parser.add_argument('-files', type=str, nargs='+')
parser.add_argument('-out', type=str)

def main(args):
    with open(args.embeddings, 'rb') as f:
        embed = np.load(f)

    df = {'file':[], 'time(s)':[], 'steps':[],'avg_sim':[]}
    for file in args.files:
        logging.info(f'Currently Computing Similarity for {file}')
        with open(os.path.join(args.dir, file), 'rb') as f:
            idx, steps, end = pickle.load(f)

        df['file'].append(file)
        df['steps'].append(steps)
        df['time(s)'].append(end)
        idx = np.array(idx)
        tmp = embed[idx]
        df['avg_sim'].append(np.mean(cosine_similarity(tmp)))
    df = pd.DataFrame(df)
    df.to_csv(args.out, index=False)
    

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main(parser.parse_args())