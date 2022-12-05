import numpy as np 
import argparse
import logging 
import bz2
import pickle
import faiss
import time 
from typing import Dict, Tuple, Any, NamedTuple
from collections import defaultdict

def time_output(diff : int) -> str:
    seconds = diff
    minutes = seconds / 60
    hours = minutes / 60 

    if hours > 1:
        return f'Completed search in {hours} hours'
    elif minutes > 1:
        return f'Completed search in {minutes} minutes'
    else:
        return f'Completed search in {seconds} seconds'

def weight(array : np.array, dim : int, alpha : float, beta : float, equal : bool) -> np.array:
    
    if equal: 
        faiss.normalize_L2(array)
        return array

    gamma = np.max(1 - alpha - beta, 0)

    array[:, :dim] = alpha * array[:, :dim]
    array[:, dim:2*dim] = beta * array[:, dim:2*dim]
    array[:, 2*dim:3*dim] = gamma * array[:, 2*dim:3*dim]
    faiss.normalize_L2(array)
    return array

class MarkovConfig(NamedTuple):
    triples : Any
    dim : int
    alpha : float
    beta : float
    equal : bool
    n : int
    k : int 
    store : str 
    nprobe : int
    ngpu : int
    built : bool


class Process:
    '''
    Markov Process with single ergodic class

    Config Parameters
    -----------------
    alpha : float -> Weight of query embedding on distance 
    beta : float -> Weight of positive document on distance
    equal : bool -> If True apply no weighting to embedding space
    dim : int -> dimensionality of single embedding
    k : int -> number of clusters in Index
    n : int -> n-nearest neighbours in similarity search 
    triples : np.array -> Set of embeddings of shape [num_samples, num_embeddings * embed_dim] 

    Generated Parameters
    --------------------
    P : dict[np.array] -> Represents the transition probability matrix 
    prob_dim : int -> Derived from 3 * embed dim
    index : faiss.index -> Cosine Similarity Search

    run() Parameters
    ----------------
    x0 : int -> id of starting state
    k : int -> desired number of samples
    '''

    state_id = 0
    def __init__(self, config : MarkovConfig) -> None:
        self.triples = weight(config.triples, config.dim, config.alpha, config.beta, config.equal)
        self.P : Dict[int, np.array, np.array] = defaultdict(lambda : np.zeros(config.n))
        self.prob_dim = 3 * config.dim
        self.nprobe = config.nprobe
        self.ngpu = config.ngpu
        self.index = self._build_index(config.triples, config.k, config.store) if not config.built else self._load_index(config.store, config.k)
        self.n = config.n
        if self.ngpu > 0:
            logging.info('Using GPU, capping neighbours at 2048')
            self.n = min(2048, self.n)

    
    def set_nprobe(self, nprobe : int):
        self.nprobe = nprobe 
        self.index.nprobe = nprobe

    def _to_device(self, index):
        if self.ngpu == 1:
            res = faiss.StandardGpuResources() 
            index = faiss.index_cpu_to_gpu(res, 0, index)
        elif self.ngpu > 1:
            index = faiss.index_cpu_to_all_gpus(index)
    
    def _load_index(self, store : str, k : int):
        assert store is not None

        index = faiss.read_index(store + f'triples.{k}.index')
        index.nprobe = self.nprobe

        self._to_device(index)
            
        return index
    
    def _build_index(self, triples : np.array, k : int, store : str):
        assert store is not None
        logging.info('Building Index...')

        start = time.time()
        quantiser = faiss.IndexFlatL2(self.prob_dim) 
        index = faiss.IndexIVFFlat(quantiser, self.prob_dim, k, faiss.METRIC_INNER_PRODUCT)
        index.train(triples)
        index.add(triples)
        end = time.time()

        logging.info(time_output(end - start))

        logging.info('Storing Index to Disk...')
        faiss.write_index(index, store + f'triples.{k}.index')
        
        index.nprobe = self.nprobe

        self._to_device(index)

        return index

    def _distance(self, x):
        return self.index.search(-x, self.n)
    
    def _step(self):
        if np.all(self.P[self.state_id] == 0):
            _, I = self._distance(np.expand_dims(self.triples[self.state_id], axis=0))
            self.P[self.state_id] = I.ravel()
        
        I = self.P[self.state_id]    
        self.state_id = np.random.choice(I)

        return self.state_id
    
    def run(self, x0, k):
        self.state_id = x0
        t = 0 
        idx = set()
        logging.info(f'Retrieving {k} candidates with starting id: {x0}')
        start = time.time()
        while len(idx) < k:
            idx.add(self._step())
            t += 1
        end = time.time() 

        logging.info(time_output(end - start))

        return np.array(list(idx)), t

parser = argparse.ArgumentParser()

parser.add_argument('-source', type=str)
parser.add_argument('-alpha', type=float, default=1.)
parser.add_argument('-beta', type=float, default=1.)
parser.add_argument('-n', type=int, default=1)
parser.add_argument('-k', type=int, default=100)
parser.add_argument('-store', type=str)
parser.add_argument('-samples', type=int, default=1)
parser.add_argument('-out', type=str, default='/')
parser.add_argument('-nprobe', type=int, default=0)
parser.add_argument('-ngpu', type=int, default=0)
parser.add_argument('--start', type=int, default=None)
parser.add_argument('--eq', action='store_true')
parser.add_argument('--built', action='store_true')
parser.add_argument('--compress', action='store_true')

def main(args):
    if args.compress:
        with bz2.open(args.source, 'rb') as f:
            array = pickle.load(f)
    else:
        with open(args.source, 'rb') as f:
            array = np.load(f)
    
    nprobe = args.k // 10 if args.nprobe == 0 else args.nprobe
   
    config = MarkovConfig(
        triples=array,
        dim=array.shape[-1]//3,
        alpha=args.alpha,
        beta=args.beta,
        equal=True if args.eq else False,
        n = args.n,
        k = args.k,
        store = args.store,
        nprobe = nprobe, 
        ngpu = args.ngpu,
        built = True if args.built else False
    )

    model = Process(config)
    if args.start:
        start_id = args.start 
    else:
        start_id = np.random.randint(0, len(config.triples))

    I, t = model.run(start_id, args.samples)

    logging.info(f'{args.samples} samples found in {t} steps, Saving...')

    with open(args.out + f'samples.{args.samples}.pkl', 'wb') as f:
        pickle.dump(I, f)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info('--Initialising Candidate Choice Using Markov Process--')
    main(parser.parse_args())