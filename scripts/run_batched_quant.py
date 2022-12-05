import numpy as np 
import argparse
import logging 
import bz2
import pickle
import faiss
import time 
from typing import Dict, Tuple, Any, NamedTuple
from collections import defaultdict
import multiprocessing as mp

from queryreduce.models.config import MarkovConfig
from queryreduce.models.markov import Process

parser = argparse.ArgumentParser()

parser.add_argument('-source', type=str)
parser.add_argument('-alpha', type=float, default=1.)
parser.add_argument('-beta', type=float, default=1.)
parser.add_argument('-n', type=int, default=1)
parser.add_argument('-store', type=str)
parser.add_argument('-samples', type=int, default=1)
parser.add_argument('-batch', type=int, default=16)
parser.add_argument('-type', type=str, default='std')
parser.add_argument('-out', type=str, default='/')
parser.add_argument('-nprobe', type=int, default=3)
parser.add_argument('-ngpu', type=int, default=0)
parser.add_argument('--start', type=int, default=None)
parser.add_argument('--eq', action='store_true')
parser.add_argument('--compress', action='store_true')

def main(args):
    if args.compress:
        with bz2.open(args.source, 'rb') as f:
            array = pickle.load(f)
    else:
        with open(args.source, 'rb') as f:
            array = np.load(f)
   
    config = MarkovConfig(
        triples=array,
        dim=array.shape[-1]//3,
        alpha=args.alpha,
        beta=args.beta,
        equal=True if args.eq else False,
        batch = args.batch,
        batch_type = args.type,
        n = args.n,
        store = args.store,
        nprobe = args.nprobe, 
        ngpu = args.ngpu,
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