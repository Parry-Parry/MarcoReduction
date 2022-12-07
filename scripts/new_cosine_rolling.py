import argparse
import bz2
import pickle
import time
from typing import NamedTuple, Any
import faiss 
import numpy as np
import logging 
from functools import partial
import multiprocessing as mp

from queryreduce.utils.utils import time_output

class AcceptanceConfig(NamedTuple):
    states : Any
    metric : str 
    sub : int 
    alpha : float 
    update : int 
    compare : str 
    threshold : str
    threshold_init : float
    gpus : int

'''
Rejection Sampler Based on a Rolling Centroid

Parameters
----------

states : np.array -> Vectors representing discrete state soace
metric : str -> What type of metric to measure distance by
sub : int -> Subset of previous candidates for each comparison 
alpha : float -> Acceptance threshold for ratio
update : int -> How often to recompute centroid 
compare : str -> Take either maximum or mean for ratio
gpus : int -> Number of usable gpus

Generated Parameters
--------------------

id : np.array -> Seperate index array to allow removal of candidates without having to run through high dim state space
centroid : np.array -> Rolling centroid of candidate cluster for acceptance ratio

Runtime Parameters
------------------

x0 : int -> Starting index
k : int -> Desired number of samples
'''

class Sampler:
    def __init__(self, config : AcceptanceConfig) -> None:
        faiss.omp_set_num_threads(mp.cpu_count())
        compare = {
            'max' : self._compare_max,
            'mean' : self._compare_mean
        }
        threshold = {
            'random' : self._random_threshold,
            'set' : self._set_threshold
        }

        self.states = config.states
        faiss.normalize_L2(self.states)
        self.id = np.arange(len(config.states), dtype=np.int64)
        self.sub = config.sub
        self.update = config.update

        self.idx = []
        self.centroid = np.zeros((1, config.states.shape[-1]))
        self.subset = None
        self.threshold_val = config.threshold_init
        self.compare = compare[config.compare]
        self.threshold = threshold[config.threshold]
        
        self.distance = np.inner
    
    def _compare_max(self, x, xs) -> float:
        return np.max(np.inner(x, xs)) < self.threshold_val

    def _compare_mean(self, x, xs) -> float:
        return np.mean(np.inner(x, xs)) < self.threshold_val

    def _get_subset(self) -> np.array:
        if len(self.idx) > self.sub:
            logging.debug('More candidates than subset')
            indices = np.random.choice(self.idx, self.sub, replace=False)
        elif len(self.idx) == 0:
            logging.debug('No Candidates')
            return None
        else:
            logging.debug('Less candidates than subset')
            indices = self.idx
        return self.states[indices]

    def _update_centroid(self) -> None:
        candidates = self._get_subset()
        if candidates is None: return None
        self.subset = candidates
        self.centroid = np.expand_dims(np.mean(candidates, axis=0), axis=0)
        self.threshold_val = np.mean(self.distance(self.centroid, candidates))
    
    def _set_threshold(self, x):
        if self.subset is not None: return self.compare(x, self.subset)
        return self.compare(x, self.centroid)

    def _random_threshold(self, x):
        vecs = self._get_subset()

        if not vecs: return True
        return self.compare(x, vecs)

    def run(self, x0, k) -> np.array:
        x_init = self.states[x0]
        self.centroid = np.expand_dims(x_init, axis=0)
        ticker = 0 # Update Ticker
        t = 0 # Total Steps
        logging.info(f'Retrieving {k} candidates with starting id: {x0}')
        start = time.time()
        while len(self.idx) < k:
            x_cand = np.random.choice(self.id)
            np.delete(self.id, x_cand)
            threshold = self.threshold(np.expand_dims(self.states[x_cand], axis=0))
            logging.debug(f'Threshold value {threshold}')
            if threshold:
                ticker += 1
                self.idx.append(x_cand)
            
            if t % self.update == 0:
                logging.debug(f'Updating Centroid at step {t}')
                self._update_centroid()
                
            if t % 1000 == 0:
                diff = time.time() - start
                logging.info(f'Time Elapsed over {t} steps: {diff} | {len(self.idx)} candidates found')
            t += 1
        end = time.time()
        logging.info(time_output(end - start))
        
        return np.array(list(self.idx)), t

parser = argparse.ArgumentParser()

parser.add_argument('-source', type=str)
parser.add_argument('-sub', type=int, default=10)
parser.add_argument('-update', type=int, default=100)
parser.add_argument('-compare', type=str, default='mean')
parser.add_argument('-threshold', type=str, default='set')
parser.add_argument('-threshold_init', type=float, default=1.)
parser.add_argument('-samples', type=int, default=1)
parser.add_argument('-out', type=str, default='/')
parser.add_argument('-ngpu', type=int, default=0)
parser.add_argument('--start', type=int, default=None)
parser.add_argument('--compress', action='store_true')
parser.add_argument('--verbose', action='store_true')

def main(args):
    if args.compress:
        with bz2.open(args.source, 'rb') as f:
            array = pickle.load(f)
    else:
        with open(args.source, 'rb') as f:
            array = np.load(f)
   
    config = AcceptanceConfig(
        states=array,
        metric=None,
        sub=args.sub,
        alpha=None,
        update=args.update,
        compare=args.compare,
        threshold=args.threshold,
        threshold_init=args.threshold_init,
        gpus = args.ngpu
    )

    model = Sampler(config)
    if args.start:
        start_id = args.start 
    else:
        start_id = np.random.randint(0, len(config.states))

    I, t = model.run(start_id, args.samples)

    logging.info(f'{args.samples} samples found in {t} steps, Saving...')

    with open(args.out + f'samples.cosine.{args.update}.{args.sub}.{args.compare}.{args.threshold}.{args.samples}.pkl', 'wb') as f:
        pickle.dump(I, f)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.verbose: log_level = logging.DEBUG
    else: log_level = logging.INFO
    logging.basicConfig(format='%(asctime)s - %(message)s', level=log_level)
    logging.info('--Initialising Candidate Choice Using Acceptance Threshold Sampler--')
    main(args)