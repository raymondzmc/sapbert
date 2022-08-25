import argparse
import logging
import os
import json
from tqdm import tqdm
import numpy as np
import sys
import pdb
sys.path.append("../") 

from utils import (
    evaluate,
)

from src.data_loader import (
    DictionaryDataset,
    QueryDataset,
    QueryDataset_custom,
    QueryDataset_COMETA,
)
from src.model_wrapper import (
    Model_Wrapper
)
LOGGER = logging.getLogger()

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='sapbert evaluation')

    # Required
    parser.add_argument('--model_dir', required=True, help='Directory for model')
    # parser.add_argument('--dictionary_path', type=str, required=True, help='dictionary path')
    # parser.add_argument('--data_dir', type=str, required=True, help='data set to evaluate')

    # Run settings
    parser.add_argument('--use_cuda',  action="store_true")
    parser.add_argument('--output_dir', type=str, default='./output/', help='Directory for output')
    parser.add_argument('--filter_composite', action="store_true", help="filter out composite mention queries")
    parser.add_argument('--filter_duplicate', action="store_true", help="filter out duplicate queries")
    parser.add_argument('--save_predictions', action="store_true", help="whether to save predictions")

    # Tokenizer settings
    parser.add_argument('--max_length', default=25, type=int)
    
    # options for COMETA
    parser.add_argument('--cometa', action="store_true", \
            help="whether to load full sentence from COMETA (or just use the mention)")
    parser.add_argument('--medmentions', action="store_true")
    parser.add_argument('--custom_query_loader', action="store_true")
    #parser.add_argument('--cased', action="store_true")

    parser.add_argument('--agg_mode', type=str, default="cls", help="{cls|mean_pool|nospec}")

    args = parser.parse_args()
    return args
    
def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)

def load_dictionary(dictionary_path): 
    dictionary = DictionaryDataset(
        dictionary_path = dictionary_path
    )
    return dictionary.data

def load_queries(data_dir, filter_composite, filter_duplicate):
    dataset = QueryDataset(
        data_dir=data_dir,
        filter_composite=filter_composite,
        filter_duplicate=filter_duplicate
    )
    return dataset.data

def load_queries_COMETA(data_dir, load_full_sentence, filter_duplicate):
    dataset = QueryDataset_COMETA(
        data_dir=data_dir,
        load_full_sentence=load_full_sentence,
        filter_duplicate=filter_duplicate
    )
    return dataset.data
                
def main(args):
    init_logging()
    print(args)

    model_wrapper = Model_Wrapper().load_model(
        path=args.model_dir,
        max_length=args.max_length,
        use_cuda=args.use_cuda,
    )
    

    path = '../training_data/hNLP-train-test-seen-unseen.json'

    with open(path, 'r') as f:
        data = json.load(f)
        id2synonyms = data['CONCEPT_TO_SYNS']
        # id2synonyms = {k: v for k, v in id2synonyms.items() if k in data['CONCEPTS']['SEEN']}
        # all_synonyms =

    id2vectors = {}
    for k, v in tqdm(id2synonyms.items()):
        vectors = model_wrapper.embed_dense(names=v, show_progress=False, batch_size=4096, agg_mode=args.agg_mode)
        id2vectors[k] = vectors



    test_set = data['DATASET']['TEST']
    recalls = [[0, 0, 0], [0, 0, 0]]
    n_examples = 0
    n_multi = 0
    for example in tqdm(test_set):
        text = example['text']
        mention_vector = model_wrapper.embed_dense(names=[text], show_progress=False, batch_size=4096, agg_mode=args.agg_mode)
        probs = []
        for k, v in id2vectors.items():
            dense_score_matrix = model_wrapper.get_score_matrix(
                query_embeds=mention_vector, 
                dict_embeds=v,
            )
            probs.append((k, dense_score_matrix.mean().item()))

        for entity in example['entities']:
            gt_concept = entity[0]
            multi_span = len(entity[1]) > 1 

            sorted_probs = sorted(probs, key=lambda x: x[1], reverse=True)
            sorted_ids = [prob[0] for prob in sorted_probs]

            if multi_span:
                n_multi += 1
                recall_idx = 1
            else:
                recall_idx = 0

            if gt_concept in sorted_ids[:1]:
                recalls[recall_idx][0] += 1
            if gt_concept in sorted_ids[:5]:
                recalls[recall_idx][1] += 1
            if gt_concept in sorted_ids[:10]:
                recalls[recall_idx][2] += 1

            n_examples += 1


    n_single = n_examples - n_multi
    print(f"Single Span: {n_single}/{n_examples}")
    test_summary = f"Top-1 Recall: {round(recalls[0][0]/n_single, 4)}, \
                     Top-5 Recall: {round(recalls[0][1]/n_single, 4)}, \
                     Top-10 Recall: {round(recalls[0][2]/n_single, 4)}"
    print(test_summary)
    print(f"Multi Span: {n_multi}/{n_examples}")
    test_summary = f"Top-1 Recall: {round(recalls[1][0]/n_multi, 4)}, \
                     Top-5 Recall: {round(recalls[1][1]/n_multi, 4)}, \
                     Top-10 Recall: {round(recalls[1][2]/n_multi, 4)}"
    print(test_summary)
    print(f"All: {n_examples}")
    test_summary = f"Top-1 Recall: {round((recalls[0][0] + recalls[1][0])/n_examples, 4)}, \
                     Top-5 Recall: {round((recalls[0][1] + recalls[1][1])/n_examples, 4)}, \
                     Top-10 Recall: {round((recalls[0][2] + recalls[1][2])/n_examples, 4)}"
    print(test_summary)

if __name__ == '__main__':
    args = parse_args()
    main(args)
