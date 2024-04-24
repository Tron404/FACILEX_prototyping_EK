#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os

from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from functools import partial

def read_data():
    models = ["bert_uncased", "multiqa_mpnet_dot", "mpnet", "tfidf", "fasttext_facilex"]
    df_all = []
    path = "./data/caselaw_emb"

    # load in the individual jurisdictions and combine them into one dataframe
    for jurisdiction in os.listdir(path):
        aux = pd.read_pickle(f"{path}/{jurisdiction}/emb_{models[0]}.pickle")
        for model in models:
            read_df = pd.read_pickle(f"./data/caselaw_emb/{jurisdiction}/emb_{model}.pickle")
            if type(read_df) == pd.DataFrame:
                aux[read_df.columns[-1]] = read_df[read_df.columns[-1]]
            else:
                model = "embedding_" + model
                aux[model] = list(read_df)
        df_all.append(aux)

    df_all = pd.concat(df_all)
    df_all = df_all.reset_index(drop = True)
    return df_all

df_all = read_data()

def write_output(df_entry, file, **additional_information):
    if additional_information["output_type"] == "query":
        file.write(f"\n===================\n")
        file.write(f"({df_entry['jurisdiction']}) ({df_entry['celex']}) ({df_entry['title']})\n")
        file.write(df_entry["summaryEn"])
        file.write(f"\n===================\nRetrieved cases (top {additional_information['top_k']})\n------------\n")
    else:
        file.write(f"({df_entry['celex'] & additional_information['df_entry_retrieved']['retrieved_celex']}) ({additional_information['df_entry_retrieved']['jurisdiction']}) ({additional_information['df_entry_retrieved']['title']})\n")
        file.write(additional_information['df_entry_retrieved']["summaryEn"])
        file.write("\n\n")

def evaluate(retrieval_scores, missed_retrieval_scores, top_k):
    metrics = {}

    #### precision 
    metrics[f"precision@{top_k}"] = round(np.mean(np.sum(retrieval_scores, axis = 1)/top_k, 0), 3)

    #### map@5
    average_precision = np.sum([(np.sum(retrieval_scores[:, :k], axis = 1)/k) * retrieval_scores[:, k-1] for k in range(1, top_k+1)])/top_k
    metrics[f"map@{top_k}"] = round(average_precision/len(retrieval_scores), 3)
    
    #### MRR
    aux_retrieval_scores = np.hstack([np.zeros((retrieval_scores.shape[0], 1)), retrieval_scores])
    rank = np.argmax(aux_retrieval_scores, axis = 1) # get first non-zero rank
    running_mrr = np.sum(np.divide(1, rank, where = rank != 0)) # rank starts at 0 bcs. of this so rank + 1
    metrics[f"mrr@{top_k}"] = round(running_mrr/(len(retrieval_scores)), 3)

    return metrics

def prepare_search(df, model):
    aux = df.copy(True)

    embds_query = np.asarray(aux[model].tolist())
    embds_corpus = np.asarray(aux[model].tolist())

    return embds_query, embds_corpus


def cosine_search_output(query, search, query_type, df, top_k = 5, debug_output: int = 0, unit_testing_output: bool = False):
    if debug_output:
        texts = open(f"output_combined.txt", "w")
    
    cosine_scores = cosine_similarity(query, search)
    retrieval_scores = []
    missed_retrieval_scores = []
    idx_retrieved_cases = []
    
    for i in range(cosine_scores.shape[0]):
        all_matches = np.argsort(cosine_scores[i])[::-1]

        best_matches = all_matches[1:top_k+1]
        query_celex = df.iloc[i][query_type]

        idx_retrieved_cases += [[i, best_matches]]

        write_output(df.iloc[0], texts, top_k = top_k) if i < debug_output else 1

        # see which top_k retrieved documents were retrieved
        score = []
        for retrieved_case in best_matches:
            retrieved_celex = df.iloc[retrieved_case][query_type]
            score.append(int(len(query_celex & retrieved_celex) > 0))
            write_output(df.iloc[i], texts, top_k = top_k, df_entry_retrieved = df.iloc[retrieved_case]) if i < debug_output else 1

        ### find all relevant cases that were missed
        missed_matches = []
        # for retrieved_case in all_matches[1:]:
        for retrieved_case in all_matches[top_k+1:]:
            retrieved_celex = df.iloc[retrieved_case][query_type]
            missed_matches.append(int(len(query_celex & retrieved_celex) > 0))

        score = np.asarray(score)
        retrieval_scores.append(score)
        missed_retrieval_scores.append(missed_matches)
        
    texts.close() if debug_output else 1

    if unit_testing_output:
        return_items = (np.asarray(retrieval_scores), np.asarray(missed_retrieval_scores), cosine_scores, idx_retrieved_cases)
    else:
        return_items = (np.asarray(retrieval_scores), np.asarray(missed_retrieval_scores), None)

    return return_items

if __name__ == "__main__":

    # search_functions = [partial(cosine_search_output, query_type = "celex"), partial(cosine_search_output, query_type = "citation_article")]
    search_functions = [partial(cosine_search_output, query_type = "celex")]

    scores_search_function = defaultdict()

    model = "embedding_multi-qa-mpnet-base-dot-v1"
    for search_func in search_functions:
        embds_query, embds_corpus = prepare_search(df_all)

        retrieval_scores, missed_retrieval_scores, _ = search_func(query = embds_query, search = embds_corpus, df = df_all, top_k = 5)
        evaluation_results = evaluate(retrieval_scores, missed_retrieval_scores, top_k = 5)