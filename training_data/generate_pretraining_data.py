import os
import json
import torch
import random
import itertools
import pdb

def generate_hnlp(path):
    with open(path, 'r') as f:
        data = json.load(f)
        id2synonyms = data['CONCEPT_TO_SYNS']
        id2synonyms = {k: v for k, v in id2synonyms.items() if k in data['CONCEPTS']['SEEN']}

    pos_pairs = []
    for concept_id, concept_synonyms in id2synonyms.items():
        pairs = list(itertools.combinations(concept_synonyms, r=2))

        if len(pairs)>50: # if >50 pairs, then trim to 50 pairs
            pairs = random.sample(pairs, 50)

        for p in pairs:
            line = str(concept_id) + "||" + p[0] + "||" + p[1]
            pos_pairs.append(line)

    with open('./training_file_hnlp_pairwise_pair_th50.txt', 'w') as f:
        for line in pos_pairs:
            f.write("%s\n" % line)

def generate_rfe(path):

    data = torch.load(path)
    name2synonyms = data['synonyms_names']

    pos_pairs = []
    for concept_name, concept_synonyms in name2synonyms.items():
        pairs = list(itertools.combinations(concept_synonyms, r=2))

        if len(pairs)>50: # if >50 pairs, then trim to 50 pairs
            pairs = random.sample(pairs, 50)

        for p in pairs:
            line = str(concept_name) + "||" + p[0] + "||" + p[1]
            pos_pairs.append(line)

    with open('./training_file_rfe_pairwise_pair_th50.txt', 'w') as f:
        for line in pos_pairs:
            f.write("%s\n" % line)


if __name__ == '__main__':
    # path = 'hNLP-train-test-seen-unseen.json'
    # generate_hnlp(path)
    path = 'CuRSA-FIXED-v0-processed-all.pth'
    generate_rfe(path)



    