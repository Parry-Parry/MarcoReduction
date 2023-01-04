import pickle
import numpy as np 
import pandas as pd
import argparse
import logging 
import time 
from sklearn.metrics.pairwise import cosine_similarity

class Process:
    state_id = 0
    def __init__(self, triples, k=100, target_t=0.65, min_t=None, max_steps_per_sample=10) -> None:
        self.triples = triples
        self.index = np.arange(len(triples)) # Index for candidates
        self.k = k # Num samples for mean
        self.c = None # Set of Candidates
        self.threshold = None
        self.t = target_t
        self.time = [0]
        if min_t:
            self.max_steps = max_steps_per_sample
            self.thresholds = [target_t, min_t]
        else:
            self.thresholds = None 
            self.max_steps = None
    
    def _set_t(self, step):
        if step > self.time[-1]: return None
        self.t = np.interp(step, self.time, self.thresholds)

    def _distance(self, x, mean):
        return np.mean(cosine_similarity(x.reshape(1, -1), mean))

    def _get_indices(self): # Check we have enough candidates to sample
        c = list(self.c)
        l_c = len(c)
        if l_c > self.k:
            choice = np.random.choice(c, self.k, replace=False)
            self.threshold = choice
            return choice
        else:
            return c 
    
    def _get_candidates(self):
        if self.threshold is not None:
            return self.triples[self.threshold]
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
        self.time.append(k * self.max_steps)
        step = 0 
        self.c = set() # Set allows for the compiler to ignore candidates we have already accepted
        logging.info(f'Retrieving {k} candidates with starting id: {x0}')
        assert self.c is not None
        self.c.add(x0)
        start = time.time()
        while len(self.c) < k:
            self.c.add(self._step())
            step += 1
            self._set_t(step)
            if step % 1000: logging.info(f'{step} steps complete, {len(self.c)} candidates found')
        end = time.time() - start 

        logging.info(f'Completed collection in {end} seconds')

        return list(self.c), step


parser = argparse.ArgumentParser()

parser.add_argument('-textsource', type=str)
parser.add_argument('-embedsource', type=str)
parser.add_argument('-k', type=int, nargs='+')
parser.add_argument('-t', type=float, nargs='+')
parser.add_argument('-c', type=int, default=1e5)
parser.add_argument('-out', type=str)
parser.add_argument('--min_t', type=float, nargs='*')
parser.add_argument('--idxout', type=str)
parser.add_argument('--max_step', type=int)
parser.add_argument('--start', type=int)


def main(args):
    cols = ['query', 'psg+', 'psg-']
    types = {col : str for col in cols}
    
    logging.info('Reading Text...')
    triples_df = pd.read_csv(args.textsource, sep='\t', header=None, index_col=False, names=cols, dtype=types)

    logging.info('Reading Embeddings...')
    with open(args.embedsource, 'rb') as f:
        array = np.load(f)
    
    for k in args.k:
        for t in args.t:

            model = Process(array, k, target_t=t, min_t=min_t)
            if args.start:
                start_id = args.start 
            else:
                start_id = np.random.randint(0, len(array))

            idx, steps = model.run(start_id, args.c)
            new_df = triples_df.loc[idx]

            if args.idxout:
                file = (idx, steps)
                with open(args.idxout + f'mhcosine.{k}.{t}.{args.c}.pkl', 'wb') as f:
                    pickle.dump(file, f)

            new_df.to_csv(args.out + f'mh.{k}.{t}.{args.c}.tsv', sep='\t', header=False, index=False)

            logging.info(f'{args.c} samples found in {steps} steps, Saving...')

    return 0 

    

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info('--Initialising Candidate Choice Using Metropolis Hastings Process--')
    main(parser.parse_args())