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
        compare = {
            'max' : self._compare_max,
            'mean' : self._compare_mean
        }
        distance = {
            'L2' : faiss.METRIC_L2,
            'IP' : faiss.METRIC_INNER_PRODUCT
        }

        self.states = config.states
        if config.metric == 'IP' : faiss.normalize_L2(self.states)
        self.metric = distance[config.metric]
        self.id = np.arange(len(config.states), dtype=np.int64)
        self.sub = config.sub
        self.alpha = config.alpha
        self.update = config.update

        self.idx = []
        self.centroid = np.zeros(config.states.shape[-1], dtype=np.int64)
        self.compare = compare[config.compare]

        if config.gpus:
            logging.debug('Using GPU')
            res = [faiss.StandardGpuResources() for _ in range(config.gpus)]
            if config.gpus == 1: res = res[0]
            self.distance = partial(faiss.pairwise_distance_gpu, res=res, metric=self.metric)
        elif not config.gpus and config.metric == 'IP':
            logging.error('Cannot use Inner Product on CPU Exiting...')
            exit
        else:
            logging.debug('Using CPU')
            self.distance = partial(faiss.pairwise_distances, mt=self.metric)
    
    def _compare_max(self, x, c) -> float:
        return np.max(x) / np.max(c)

    def _compare_mean(self, x, c) -> float:
        return np.mean(x) / np.mean(c)

    def _threshold(self, x):
        '''
        Case:
            * There are more candidates than the desired subset
            * There are no candidates
            * There are less candidates than the desired subset
        '''
        if len(self.idx) > self.sub:
            logging.debug('More candidates than subset')
            indices = np.random.choice(self.idx, self.sub)
        elif len(self.idx) == 0:
            logging.debug('No Candidates')
            return 10.
        else:
            logging.debug('Less candidates than subset')
            indices = self.idx
        
        vecs = self.states[indices]
        dist_x = self.distance(x, vecs)
        dist_c = self.distance(self.centroid, vecs)

        return self.compare(dist_x, dist_c)

    def run(self, x0, k) -> np.array:
        faiss.omp_set_num_threads(mp.cpu_count())
        x_init = self.states[x0]
        self.centroid = x_init
        ticker = 0 # Update Ticker
        t = 0 # Total Steps
        logging.info(f'Retrieving {k} candidates with starting id: {x0}')
        start = time.time()
        while len(self.idx) < k:
            x_cand = np.random.choice(self.id)
            np.delete(self.id, x_cand)
            threshold = self._threshold(np.expand_dims(self.states[x_cand], axis=0))
            logging.debug(f'Threshold value {threshold}')
            if threshold > self.alpha:
                ticker += 1
                self.idx.append(x_cand)
                if ticker % self.update == 0:
                    logging.debug(f'Updating Centroid at step {t}')
                    self.centroid = np.mean(self.states[self.idx])
                
            if t % 1000 == 0:
                diff = time.time() - start
                logging.info(f'Time Elapsed over {t} steps: {diff} | {len(self.idx)} candidates found')
            t += 1
        end = time.time()
        logging.info(time_output(end - start))
        
        return np.array(list(self.idx)), t

parser = argparse.ArgumentParser()

parser.add_argument('-source', type=str)
parser.add_argument('-metric', type=str, default='IP')
parser.add_argument('-sub', type=int, default=10)
parser.add_argument('-alpha', type=float, default=1.)
parser.add_argument('-update', type=int, default=100)
parser.add_argument('-compare', type=str, default='mean')
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
        metric=args.metric,
        sub=args.sub,
        alpha=args.alpha,
        update=args.update,
        compare=args.compare,
        gpus = args.ngpu
    )

    model = Sampler(config)
    if args.start:
        start_id = args.start 
    else:
        start_id = np.random.randint(0, len(config.states))

    I, t = model.run(start_id, args.samples)

    logging.info(f'{args.samples} samples found in {t} steps, Saving...')

    with open(args.out + f'samples.{args.samples}.pkl', 'wb') as f:
        pickle.dump(I, f)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.verbose: log_level = logging.DEBUG
    else: log_level = logging.INFO
    logging.basicConfig(format='%(asctime)s - %(message)s', level=log_level)
    logging.info('--Initialising Candidate Choice Using Acceptance Threshold Sampler--')
    main(args)



            
            
            









