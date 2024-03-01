import json
import re
import numpy as np
import torch

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# assumption that data contains html elements
def sanitize_data(text: str) -> str:
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"&nbsp;", "", text)
    text = text.strip()

    return text

# given the embedded representation of a query case and the embeddings of the cases through which to search, alongside their respective CELEX IDs,
# return the ordered indices (from most similar to least) of retrieved cases
def cosine_similarity_search(query: np.ndarray, search_space: np.ndarray, query_celex: str, search_space_celex: list, top_k: int = 5) -> list:
    cosine_scores: list = cosine_similarity(query.reshape(1,-1), search_space)[0]

    all_matches: np.ndarray = np.argsort(cosine_scores)[::-1]
    best_matches: np.ndarray = all_matches[:top_k]
    search_space_celex: list = search_space_celex[best_matches]

    similar_cases: list = []
    for retrieved_case, retrieved_case_celex in zip(best_matches, search_space_celex):
        if query_celex == retrieved_case_celex:
            similar_cases.append(retrieved_case)

    return similar_cases

if __name__ == "__main__":
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # for embedding using transformers
    embedding_model: SentenceTransformer = SentenceTransformer("../models/multi-qa-mpnet-base-dot-v1").to(device)

    data: dict = json.load(open("example_input.json", "r"))
    data["query"]["summaryEn"] = sanitize_data(data["query"]["summaryEn"])
    for search_text in data["search_space"]:
        search_text["summaryEn"] = sanitize_data(search_text["summaryEn"])

    search_celex = []
    search_embedded = []
    search_text = []
    search_jurisdiction = []
    for search_space_item in data["search_space"]:
        search_embedded.append(embedding_model.encode(search_space_item["summaryEn"]))
        search_celex.append(search_space_item["celex"])
        search_text.append(search_space_item["summaryEn"])
        search_jurisdiction.append(search_space_item["jurisdiction"])

    query_embedded: np.ndarray = embedding_model.encode(data["query"]["summaryEn"])
    search_celex = np.asarray(search_celex)
    search_embedded = np.asarray(search_embedded)
    search_text = np.asarray(search_text)
    search_jurisdiction = np.asarray(search_jurisdiction)

    recommended_idx: list = cosine_similarity_search(query_embedded, search_embedded, data["query"]["celex"], search_celex)
    output_data = [{"summaryEn": search_text[idx], "celex": search_celex[idx], "jurisdiction": search_jurisdiction[idx]} for idx in recommended_idx]

    json.dump(output_data, open("example_output.json", "w"))

    del embedding_model    