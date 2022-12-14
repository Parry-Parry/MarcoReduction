import numpy as np 
import pandas as pd
import argparse
import logging 
import time 
from typing import Any, NamedTuple
from sklearn.metrics.pairwise import cosine_similarity

class Config(NamedTuple):
    triples : Any
    k : int 
    t : float 


class Process:
    state_id = 0
    def __init__(self, config : Config) -> None:
        self.triples = config.triples
        self.index = np.arange(len(config.triples)) # Index for candidates
        self.k = config.k # Num samples for mean
        self.t = config.t # Threshold similarity
        self.c = None # Set of Candidates

    def _distance(self, x, mean):
        return np.mean(cosine_similarity(x.reshape(1, -1), mean))

    def _get_indices(self): # Check we have enough candidates to sample
        c = list(self.c)
        l_c = len(c)
        if l_c > self.k:
            return np.random.choice(c, self.k, replace=False)
        else:
            return c 
    
    def _get_candidates(self):
        idx = self._get_indices()

        if len(idx) > 1:
            return self.triples[idx] # Get random K from candidate set
        else:
            return self.triples[idx].reshape(1, -1)

    def _step(self):
        c_id = np.random.choice(self.index) 
        c = self.triples[c_id] # Get random candidate

        K = self._get_candidates()
        d = self._distance(c, K) # Cosine Similarity

        if d < self.t: # If candidate dissimilarity over threshold
            self.state_id = c_id # Accept Candidate

        return self.state_id
    
    def run(self, x0, k):
        self.state_id = x0
        t = 0 
        self.c = set() # Set allows for the compiler to ignore candidates we have already accepted
        logging.info(f'Retrieving {k} candidates with starting id: {x0}')
        assert self.c is not None
        self.c.add(x0)
        start = time.time()
        while len(self.c) < k:
            self.c.add(self._step())
            t += 1
            if t % 1000: logging.info(f'{t} steps complete, {len(self.c)} candidates found')
        end = time.time() - start 

        logging.info(f'Completed collection in {end} seconds')

        return list(self.c), t


parser = argparse.ArgumentParser()

parser.add_argument('-textsource', type=str)
parser.add_argument('-embedsource', type=str)
parser.add_argument('-k', type=int, default=100)
parser.add_argument('-t', type=float, default=0.0)
parser.add_argument('-c', type=int, default=1e5)
parser.add_argument('-out', type=str)
parser.add_argument('--start', type=int)


def main(args):
    cols = ['query', 'psg+', 'psg-']
    types = {col : str for col in cols}
    
    logging.info('Reading Text...')
    triples_df = pd.read_csv(args.textsource, sep='\t', header=None, index_col=False, names=cols, dtype=types)

    logging.info('Reading Embeddings...')
    with open(args.embedsource, 'rb') as f:
        array = np.load(f)
    
    config = Config(
        triples=array,
        k = args.k,
        t = args.t
    )

    model = Process(config)
    if args.start:
        start_id = args.start 
    else:
        start_id = np.random.randint(0, len(config.triples))

    idx, t = model.run(start_id, args.c)

    new_df = triples_df.loc[idx]

    new_df.to_csv(args.out, sep='\t', header=False, index=False)

    logging.info(f'{args.c} samples found in {t} steps, Saving...')

    return 0 

    

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info('--Initialising Candidate Choice Using Markov Process--')
    main(parser.parse_args())