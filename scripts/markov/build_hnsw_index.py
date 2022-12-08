import numpy as np 
import faiss 
import logging 
import bz2
import argparse 
import pickle
import multiprocessing as mp

parser = argparse.ArgumentParser()

parser.add_argument('-source', type=str)
parser.add_argument('-k', type=int)
parser.add_argument('-out', type=str) 
parser.add_argument('-gpus', type=int, default=0) 
parser.add_argument('--compress', action="store_true")
parser.add_argument('--l2', action="store_true")

def main(args):
    faiss.omp_set_num_threads(mp.cpu_count())
    if args.compress:
        logging.info('Opening Compressed File')
        with bz2.open(args.source, 'rb') as f:
            triples = pickle.load(f)
    else:
        logging.info('Loading Numpy Array')
        with open(args.source, 'rb') as f:
            triples = np.load(f)

    prob_dim = triples.shape[-1]
    logging.info('Training Index')


    faiss.normalize_L2(triples)
    quantiser = faiss.IndexHNSWFlat(prob_dim, 64) 
    if not args.l2: index = faiss.IndexIVFFlat(quantiser, prob_dim, args.k)
    else: index = faiss.IndexIVFFlat(quantiser, prob_dim, args.k, faiss.METRIC_L2)

    index.cp.min_points_per_centroid = 5
    index.quantizer_trains_alone = 2

    if args.gpus > 0:
        if args.gpus == 1:
            res = faiss.StandardGpuResources() 
            index = faiss.index_cpu_to_gpu(res, 0, index)
        else:
            index = faiss.index_cpu_to_all_gpus(index)

    index.train(triples)
    index.add(triples)

    logging.info('Trained Index, Saving...')
    faiss.write_index(index, args.out + f'hnsw.{args.k}.index')

    return 0
    

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info('Building Faiss IVF HNSW Index')
    main(parser.parse_args())