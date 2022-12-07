import pickle 
import numpy as np
import pandas as pd
import argparse
import logging

parser = argparse.ArgumentParser()

parser.add_argument('-dataset', type=str)
parser.add_argument('-indices', type=str)
parser.add_argument('-out', type=str)

def main(args):
    cols = ['qid', 'pid+', 'pid-']
    types = {col : str for col in cols}
    logging.info('Reading Dataset...')
    
    df = pd.read_csv(args.dataset, sep='\t', header=None, index_col=False, names=cols, dtype=types)

    with open(args.indices, 'rb') as f:
        indices = pickle.load(f)
    
    new_df = df.loc[indices]
    logging.info('Writing New Dataset to Disk...')
    new_df.to_csv(args.out, sep='\t', header=False, index=False)
    return 0
    


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info('Retrieving Triples by Indices Array')
    main(parser.parse_args())