import pickle
import numpy as np 
import argparse
import logging 
import time 
from sklearn.metrics.pairwise import cosine_similarity

class Process:
    state_id = 0
    def __init__(self, triples, k=100, t=0.65, max_steps_per_sample=10) -> None:
        self.triples = triples
        self.index = np.arange(len(triples)) # Index for candidates
        self.k = k # Num samples for mean
        self.c = None # Set of Candidates
        self.threshold = None
        self.t = t
        self.max_steps = max_steps_per_sample
        self.current_best = (0, 1.)
        self.count = 0
    
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
    
    def _reset(self):
        self.count = 0
        self.current_best = (0, 1.)

    def _step(self):
        c_id = np.random.choice(self.index) 
        c = self.triples[c_id] # Get random candidate

        K = self._get_candidates()
        d = self._distance(c, K) # Cosine Similarity

        if d < self.t: # If candidate dissimilarity over threshold
            self._reset()
            self.state_id = c_id # Accept Candidate
        else:
            self.count += 1
            if d < self.current_best[1]:
                self.current_best = (c_id, d)
            if self.count >= self.max_steps:
                self.state_id = self.current_best[0]
                self._reset()
                

        return self.state_id
    
    def run(self, x0, k):
        self.state_id = x0
        step = 0 
        self.c = set() # Set allows for the compiler to ignore candidates we have already accepted
        logging.info(f'Retrieving {k} candidates with starting id: {x0}')
        assert self.c is not None
        self.c.add(x0)
        start = time.time()
        while len(self.c) < k:
            self.c.add(self._step())
            step += 1
            if step % 1000 == 0: logging.info(f'{step} steps complete, {len(self.c)} candidates found')
        end = time.time() - start 

        logging.info(f'Completed collection in {end} seconds')

        return list(self.c), step, end


parser = argparse.ArgumentParser()

parser.add_argument('-source', type=str)
parser.add_argument('-k', type=int, nargs='+')
parser.add_argument('-t', type=float, nargs='+')
parser.add_argument('-max_step', type=int, nargs='+')
parser.add_argument('-c', type=int, default=1e5)
parser.add_argument('-out', type=str)
parser.add_argument('--start', type=int)


def main(args):
    logging.info('Reading Embeddings...')
    with open(args.source, 'rb') as f:
        array = np.load(f)
    
    for k in args.k:
        for max_s in args.max_step:
            for t in args.t:
                model = Process(array, k, t, max_steps_per_sample=max_s)
                if args.start:
                    start_id = args.start 
                else:
                    start_id = np.random.randint(0, len(array))

                idx, steps, end = model.run(start_id, args.c)

                file = (idx, steps, end)
                with open(args.out + f'mhcosine.{k}.{t}.{max_s}.{args.c}.pkl', 'wb') as f:
                    pickle.dump(file, f)

                logging.info(f'{args.c} samples found in {steps} steps, Saving...')

    return 0 

    

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info('--Initialising Candidate Choice Using Metropolis Hastings Process--')
    main(parser.parse_args())