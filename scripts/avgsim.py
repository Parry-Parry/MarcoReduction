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
    embed = np.load(args.embeddings)

    df = {'file':[], 'avg_sim':[], 'half_avg_sim':[], 'quarter_avg_sim':[]}
    

    dir_build = lambda x : os.path.join(args.dir, x)
    cut = lambda x, y : x[:y, :y]
    for file in args.files:
        with(dir_build(file), 'rb') as f:
            idx = pickle.load(f)

        df['file'].append(file)

        tmp_embeddings = embed[idx]
        half = len(idx) // 2
        quarter = len(idx) // 4
        quarter3 = half + quarter
        sim_matrix = cosine_similarity(tmp_embeddings)
        
        df['avg_sim'].append(np.mean(sim_matrix))
        df['half_avg_sim'].append(np.mean(cut(sim_matrix, half)))
        df['quarter_avg_sim'].append(np.mean(cut(sim_matrix, quarter)))
        df['quarter3_avg_sim'].append(np.mean(cut(sim_matrix, quarter3)))

    df = pd.DataFrame(df)
    df.to_csv(args.out, index=False)
    
    







    




if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info('--Initialising Candidate Choice Using Metropolis Hastings Process--')
    main(parser.parse_args())