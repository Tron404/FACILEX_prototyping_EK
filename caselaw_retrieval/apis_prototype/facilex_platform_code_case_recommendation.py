import json
import re
import numpy as np
import torch

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# assumption that data contains non-relevant elements of code/URL
def sanitize_data(text: str) -> str:
    text = re.sub(r"(?:https://)?www.[^\s<]+", "", text) # remove potential websites
    text = re.sub(r"<.*?>", "", text) # remove HTML elements
    text = re.sub(r"&nbsp;", "", text) # remove leftover formatting
    text = text.strip()

    return text

# given the embedded representation of a query case and the embeddings of the cases through which to search, alongside their respective CELEX IDs,
# return the ordered indices (from most similar to least) of retrieved cases
def cosine_similarity_search(query: np.ndarray, search_space: np.ndarray, query_celex: str, search_space_celex: list, top_k: int = 5) -> list:
    cosine_scores: list = cosine_similarity(query.reshape(1,-1), search_space)[0]

    # sort the similarity scores from most to least similar and select the top 5 most similar cases alongside their CELEX IDs
    all_matches: np.ndarray = np.argsort(cosine_scores)[::-1]
    best_matches: np.ndarray = all_matches[:top_k]
    search_space_celex: list = search_space_celex[best_matches]

    similar_cases: list = [] # indices of most similar cases
    for retrieved_case, retrieved_case_celex in zip(best_matches, search_space_celex):
        if query_celex == retrieved_case_celex: # only show the user cases which do have the same CELEX number; this can be removed to include semantically similar cases which differ in their CELEX
            similar_cases.append(retrieved_case)

    return similar_cases

if __name__ == "__main__":
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # for embedding using transformers
    embedding_model: SentenceTransformer = SentenceTransformer("../models/multi-qa-mpnet-base-dot-v1").to(device)

    query_data: dict = json.load(open("example_input_query.json", "r")) # json of case that facilitates query
    search_data: dict = json.load(open("example_input_search_corpus.json", "r")) # json of all available cases on which to search for similar ones

    query_data["summaryEn"] = sanitize_data(query_data["summaryEn"])
    for search_text in search_data:
        search_text["summaryEn"] = sanitize_data(search_text["summaryEn"])

    # extract the most important of information from the search json and store it in lists to later convert into json items
    search_celex = []
    search_embedded = []
    search_text = []
    search_jurisdiction = []
    for search_space_item in search_data:
        search_embedded.append(embedding_model.encode(search_space_item["summaryEn"]))
        search_celex.append(search_space_item["euProvisions"])
        search_text.append(search_space_item["summaryEn"])
        search_jurisdiction.append(search_space_item["jurisdiction"])

    query_embedded: np.ndarray = embedding_model.encode(query_data["summaryEn"])
    # convert lists to np.ndarray to facilitate index slicing
    search_celex = np.asarray(search_celex)
    search_embedded = np.asarray(search_embedded)
    search_text = np.asarray(search_text)
    search_jurisdiction = np.asarray(search_jurisdiction)

    recommended_idx: list = cosine_similarity_search(query_embedded, search_embedded, query_data["euProvisions"], search_celex)
    output_data = [{"summaryEn": search_text[idx], "euProvisions": search_celex[idx], "jurisdiction": search_jurisdiction[idx]} for idx in recommended_idx]

    json.dump(output_data, open("example_output.json", "w"))

    del embedding_model    