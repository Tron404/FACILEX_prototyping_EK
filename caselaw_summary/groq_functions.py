import json
import logging
import numpy as np
import os
import sys
import tiktoken
import math
import time

from functools import partial
from transformers import AutoTokenizer

from datasets import load_dataset
from env_utils import load_env_from_file
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential, RetryError
from tqdm import tqdm
from groq import Groq

# long=650; short=130; tiny=25
WORDS = int(sys.argv[-1])
# based on addage from OpenAI: 1 token ~ 0.75 words --> 1 word ~ 1.333 tokens; +10 words to complete things
MAX_OUTPUT_LENGTH = math.ceil(WORDS * 1.333 + 10)
SAFETY_MARGIN = 50
input_model_params = {
    "mixtral": {
        "max_context_length": 32768,
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "temperature": 0,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "max_tokens": MAX_OUTPUT_LENGTH
    },
    "mistral": {
        "max_context_length": 32768,
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "temperature": 0,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "max_tokens": MAX_OUTPUT_LENGTH
    },
    "gemma": {
        "max_context_length": 8192,
        "model": "google/gemma-1.1-7b-it",
        "temperature": 0,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "max_tokens": MAX_OUTPUT_LENGTH
    },
    "gemma2": {
        "max_context_length": 8192,
        "model": "google/gemma-2-9b-it",
        "temperature": 0,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "max_tokens": MAX_OUTPUT_LENGTH
    },
    "gpt-3.5": {
        "max_context_length": 16384,
        "model": "gpt-3.5-turbo-1106",
        "temperature": 0,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "max_tokens": MAX_OUTPUT_LENGTH
    },
    "llama3": {
        "max_context_length": 8192,
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "temperature": 0,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "max_tokens": MAX_OUTPUT_LENGTH
    }
}

input_api_map = {
    "openai": partial(OpenAI),
    "deepinfra": partial(OpenAI, base_url = "https://api.deepinfra.com/v1/openai"),
    "groq": partial(Groq)
}

SYSTEM_PROMPT = f"You are a legal expert. You are tasked with reading legal texts and creating summaries. Your summaries are truthful, relevant, and faithful to the source documents, using only facts and entities present in them, while also including as many as possible. Your summaries read like histories of cases, useful for other lawyers. Your summary must be around {WORDS} words long."
EXTRACT_TYPES = np.asarray(["random_selection_textrank", "first5last5_textrank", "random_selection_bert", "first5last5_bert"])

class llmResponse:
    def __init__(self, dataset_name, api, user_model, prompt_type, test_size, is_equal):
        np.random.seed(42)
        is_equal = bool(int(is_equal))
        self.api = api
        self.user_model  = user_model
        self.model = input_model_params[user_model]["model"]
        self.prompt_type  = prompt_type
        self.is_equal = is_equal
        self.test_size = int(test_size)
        self.dataset_name = dataset_name
        self.llm_params = {key:value for key,value in input_model_params[self.user_model].items() if key != "max_context_length"}
        self._load_utilities()

        if "eurlex" in dataset_name:
            global EXTRACT_TYPES 
            EXTRACT_TYPES = np.asarray(["random_selection_textrank", "random_selection_bert"])


        self.sentence_limit = 10 
        self.doc_limit = 10

        self.model = self.model.split("/")[-1]
        ##################################
        if "mixtral" in self.model.lower():
            self.model = "mixtral-8x7b-32768" 
        ##################################
        if is_equal:
            self.prompt_type = "equal_"+self.prompt_type
        if self.model not in os.listdir(f"answers/{self.dataset_name}"):
            os.mkdir(f"answers/{self.dataset_name}/{self.model}")
        if self.prompt_type not in os.listdir(f"answers/{self.dataset_name}/{self.model}"):
            os.mkdir(f"answers/{self.dataset_name}/{self.model}/{self.prompt_type}")

        self.logger = logging.getLogger(self.model)
        logging.basicConfig(filename=f"logs/log_{self.dataset_name}_{self.model}_{self.prompt_type}_{WORDS}.log", encoding="utf-8", level=logging.WARNING, filemode="w", format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')

    def __call__(self):
        self._response_from_llm()

    ##### LOADING
    def _limit_doc_sentences(self, current_length, doc, docs, docs_idx, start_sent, sent_limit):
        if start_sent > self.sentence_limit:
            return doc, start_sent, sent_limit

        sent_limit = self.sentence_limit
        while (len(self.tokenizer.encode(doc)) + current_length > input_model_params[self.user_model]["max_context_length"]) and len(docs) > 0:
            sent_limit = self.sentence_limit
            while (len(self.tokenizer.encode(doc)) + current_length > input_model_params[self.user_model]["max_context_length"]) and sent_limit > start_sent:
                doc = "".join(["".join(doc_sentences[start_sent:sent_limit]) for doc_sentences in docs])
                sent_limit -= 1

            if sent_limit <= start_sent and len(docs) < 2:
                break

            if (len(self.tokenizer.encode(doc)) + current_length > input_model_params[self.user_model]["max_context_length"]) and len(docs) > 1:
                random_doc_idx = np.random.randint(0, len(docs))
                docs.pop(random_doc_idx)
                docs_idx.pop(random_doc_idx)
        
        if len(self.tokenizer.encode(doc)) + current_length > input_model_params[self.user_model]["max_context_length"]:
            doc, start_sent, sent_limit = self._limit_doc_sentences(current_length, doc, docs, docs_idx, start_sent+1, self.sentence_limit)

        return doc, start_sent, sent_limit

    def _from_extracted(self, path, test_size, current_length):
        summs = []
        files = os.listdir(path)
        files = sorted(files, key = lambda x: int(x.split(".")[0]))
        doc_sent_limit = {}
        extract_type = path.split("/")[-2]
        file_eq = f"doc_limits/doc_sent_limit_gemma_{self.dataset_name}_{extract_type}_{self.prompt_type}_{WORDS}.json"
        if "equal" in self.prompt_type and self.user_model != "gemma":
            docs_limit_json = json.load(open(file_eq, "r"))
        for docket_idx, file in enumerate(files):
            if int(file.split(".")[0]) < test_size:
                docs = json.load(open(path + file, "r"))
                orig_size = len(docs)
                docs_idx = list(range(orig_size))

                if self.is_equal and self.user_model != "gemma":
                    docs_to_select, aux_sent_limit = docs_limit_json[str(docket_idx)]
                    docs = [doc for doc_idx, doc in enumerate(docs) if doc_idx in docs_to_select]
                    self.sentence_limit = aux_sent_limit
                    self.doc_limit = len(docs)

                    self.logger.warning(f"Changed number of documents from {orig_size} to {self.doc_limit} and number of sentences from 10 to {self.sentence_limit} for docket {docket_idx}.")

                while len(docs) > self.doc_limit:
                    random_doc_idx = np.random.randint(0, len(docs))
                    docs.pop(random_doc_idx)
                    docs_idx.pop(random_doc_idx)

                doc = "".join(["".join(doc_sentences[:self.sentence_limit]) for doc_sentences in docs])

                sent_limit = self.sentence_limit
                if len(self.tokenizer.encode(doc)) + current_length > input_model_params[self.user_model]["max_context_length"]:
                    doc, start_sent, sent_limit = self._limit_doc_sentences(current_length, doc, docs, docs_idx, 0, self.sentence_limit)
             
                    self.logger.warning(f"Number of documents and/or sentences changed due to being too big for file {file.split('.')[0]}: documents={len(docs)}/{orig_size} - sentences={sent_limit+1}/{self.sentence_limit} (starting from {start_sent}) - tokens={len(self.tokenizer.encode(doc)) + current_length}")

                doc_sent_limit[docket_idx] = (docs_idx, sent_limit)
                summs.append(doc)
        if self.user_model == "gemma":
            extract_type = path.split("/")[-2]
            json.dump(doc_sent_limit, open(f"doc_limits/doc_sent_limit_gemma_{self.dataset_name}_{extract_type}_{self.prompt_type}_{WORDS}.json", "w"), indent=2)
        return summs

    def _load_prompt(self, prompt_type):
        prompt = ""
        if self.dataset_name == "eurlexsum":
            prompt_type += "_eurlex"
        with open(f"prompts/prompt_{prompt_type}.json", "r") as file:
            prompt = json.load(file)["content"]

        return prompt

    def _load_utilities(self):
        self.user_prompt = self._load_prompt(self.prompt_type)

        load_env_from_file(".")
        self.client = input_api_map[self.api](api_key = os.environ[f"{self.api.upper()}_API_KEY"])

        if self.api == "openai":
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)

        if self.api == "groq":
            self.llm_params["model"] = self.llm_params["model"].split("/")[-1]

            if "gemma-2" in self.llm_params["model"]: # because Groq doesn't have the hyphen after gemma for gemma2
                self.llm_params["model"] = "gemma2-9b-it"

        self.length_system_prompt = len(self.tokenizer.encode(SYSTEM_PROMPT))
        self.user_prompt_length = len(self.tokenizer.encode(self.user_prompt))

    ##### RESPONSE
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
    def _completion_with_retry(self, file_path, **kwargs):
        # file_path: model/prompt_type/extract_sum_type/idx
        _, prompt_type, extract_sum_type, idx = file_path.split("/")[2:]
        success = False
        try:
            chat_completion = self.client.chat.completions.create(**kwargs)
            json.dump([*kwargs["messages"], {"role": "assistant", "content": chat_completion.choices[0].message.content, "usage": dict(chat_completion.usage)}], open(f"{file_path}_{WORDS}.json", "w"), indent=2)  
            success = True  
        except Exception as exc:
            self.logger.error(f"({type(exc).__name__} ({exc.args}) - {prompt_type} - {extract_sum_type} - {idx}")
            success = False  

        return success
    
    def _response_from_llm(self):
        for extract_sum_type in EXTRACT_TYPES:
            path = f"extracted_sums/{self.dataset_name}/extracted_sums_json_{extract_sum_type}/"
            os.makedirs(f"answers/{self.dataset_name}/{self.model}/{self.prompt_type}/{extract_sum_type}/", exist_ok=True)

            extracted_summaries = self._from_extracted(path, test_size=self.test_size, current_length=input_model_params[self.user_model]["max_tokens"]+self.user_prompt_length+self.length_system_prompt+SAFETY_MARGIN)
            errors = 0

            iterator = tqdm(extracted_summaries, desc=f"Model={self.model}, Prompt={self.prompt_type}, Selection={extract_sum_type}, Errors={errors}")
            for idx, summ in enumerate(iterator):
                if f"{idx}_{WORDS}.json" in os.listdir(f"answers/{self.dataset_name}/{self.model}/{self.prompt_type}/{extract_sum_type}"):
                    continue
                prompt = self.user_prompt.format(INPUT=summ)

                message_history = [
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
   
                chat_completion_success = self._completion_with_retry(
                    file_path=f"answers/{self.dataset_name}/{self.model}/{self.prompt_type}/{extract_sum_type}/{idx}",
                    messages=message_history,
                    **self.llm_params
                )

                if self.api == "groq":
                    time.sleep(7)           

                if not chat_completion_success:
                    errors += 1
                    iterator.set_description(f"Model={self.model}, Prompt={self.prompt_type}, Selection={extract_sum_type}, Errors={errors}, Last_error_file={idx}")

if __name__ == "__main__":
    llm = llmResponse(*sys.argv[1:-1])
    llm()