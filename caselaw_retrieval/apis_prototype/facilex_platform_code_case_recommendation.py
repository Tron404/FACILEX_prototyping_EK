import json
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

# given the embedded representation of a query case and the embeddings of the cases through which to search, alongside their respective CELEX IDs,
# return the ordered indices (from most similar to least) of retrieved cases
def cosine_similarity_search(query: np.ndarray, search_space: np.ndarray, query_celex: str, search_space_celex: list, top_k: int = 5) -> list:
    cosine_scores: list = cosine_similarity(query.reshape(1,-1), search_space)[0]

    # sort the similarity scores from most to least similar and select the top 5 most similar cases alongside their CELEX IDs, while ignoring the query itself
    all_matches: np.ndarray = np.argsort(cosine_scores)[::-1][1:]
    best_matches: np.ndarray = all_matches[:top_k]
    search_space_celex: np.ndarray = search_space_celex[best_matches]

    # only show the user cases which do have the same CELEX number; this can be removed to include semantically similar cases which differ in their CELEX
    # if there are no EU provisions in the query case, then simply return the most semantically similar cases
    similar_cases: list = [retrieved_case for retrieved_case, retrieved_case_celex in zip(best_matches, search_space_celex) if query_celex == None or query_celex == retrieved_case_celex] # indices of most similar cases
    
    return similar_cases

if __name__ == "__main__":
    query_data: dict = json.load(open("input_query.json", "r")) # json of case that facilitates query
    search_embeddings_data: dict = json.load(open("corpus_embedded.json", "r")) # json of all available cases on which to search for similar ones
    search_text_data: dict = json.load(open("corpus.json", "r"))

    # extract the most important of information from the search json and store it in lists to later convert into json items
    uniqueId_to_idx: np.ndarray = np.asarray([id["uniqueId"] for id in search_text_data])
    search_celex: np.ndarray = np.asarray([search_space_item["euProvisions"] for search_space_item in search_text_data])
    search_embedded: np.ndarray = np.asarray([np.asarray(search_space_item["embedding"]) for search_space_item in search_embeddings_data])

    # get embedding of query
    query_embedded: np.ndarray = np.asarray([search_space_item["embedding"] for search_space_item in search_embeddings_data if search_space_item["uniqueId"] == query_data["uniqueId"]])

    recommended_idx: list = cosine_similarity_search(query_embedded, search_embedded, query_data["euProvisions"], search_celex)
    output_data: list = [{"uniqueId": uniqueId_to_idx[idx], "summaryEn": search_text_data[idx]["summaryEn"], "euProvisions": search_celex[idx], "jurisdiction": search_text_data[idx]["jurisdiction"]} for idx in recommended_idx]

    json.dump(output_data, open("example_output.json", "w"), indent = 2)