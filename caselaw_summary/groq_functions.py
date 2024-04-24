import json
import logging
import numpy as np
import os
import sys
import tiktoken
from functools import partial
from transformers import AutoTokenizer

from datasets import load_dataset
from env_utils import load_env_from_file
from groq import Groq
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential, RetryError
from tqdm import tqdm

MAX_OUTPUT_LENGTH = 250
SAFETY_MARGIN = 25
input_model_params = {
    "mixtral": {
        "max_context_length": 32768,
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
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

SYSTEM_PROMPT = "You are a legal expert. You must answer concisely and truthfully, including only information that is relevant to the conversation. Stay faithful to the original text and keep the exact wording as found in the text as closely as possible. Only include facts relevant to the text, without any filler words."
EXTRACT_TYPES = np.asarray(["random_selection", "first5last5", "random_selection_bert", "first5last5_bert"])

class llmResponse:
    def __init__(self, api, user_model, prompt_type, test_size, is_equal):
        np.random.seed(42)
        is_equal = bool(int(is_equal))
        self.api = api
        self.user_model  = user_model
        self.model = input_model_params[user_model]["model"]
        self.prompt_type  = prompt_type
        self.test_size = int(test_size)
        self.llm_params = {key:value for key,value in input_model_params[self.user_model].items() if key != "max_context_length"}
        self._load_utilities()

        if is_equal:
            self.sentence_limit = 1 
            self.doc_limit = 3
        else:
            self.sentence_limit = 10 
            self.doc_limit = 10

        self.model = self.model.split("/")[-1]
        ##################################
        if "mixtral" in self.model.lower():
            self.model = "mixtral-8x7b-32768" 
        if "gemma" in self.model.lower():
            self.model = "gemma-7b-it"
        ##################################
        if is_equal:
            self.prompt_type = "equal_"+self.prompt_type
        if self.model not in os.listdir("answers"):
            os.mkdir(f"answers/{self.model}")
        if self.prompt_type not in os.listdir(f"answers/{self.model}"):
            os.mkdir(f"answers/{self.model}/{self.prompt_type}")

        self.logger = logging.getLogger(self.model)
        logging.basicConfig(filename=f"logs/log_{self.model}_{self.prompt_type}.log", encoding="utf-8", level=logging.WARNING, filemode="w", format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')

    def __call__(self):
        self._response_from_llm()

    ##### LOADING
# 05:08:25 ERROR (BadRequestError (('Error code: 400 - {\'error\': {\'message\': \'{"object":"error","message":"This model\\\'s maximum context length is 8192 tokens. However, you requested 8197 tokens (7947 in the messages, 250 in the completion). Please reduce the length of the messages or completion.","type":"BadRequestError","param":null,"code":400}\'}}',)) - basic - random_selection - 62
    def _from_extracted(self, path, test_size, current_length):
        summs = []
        files = os.listdir(path)
        files = sorted(files, key = lambda x: int(x.split(".")[0]))
        for file in files:
            if int(file.split(".")[0]) < test_size:
                docs = json.load(open(path + file, "r"))
                orig_size = len(docs)

                while len(docs) > self.doc_limit:
                    random_doc_idx = np.random.randint(0, len(docs))
                    docs.pop(random_doc_idx)

                doc = "".join(["".join(doc_sentences[:self.sentence_limit]) for doc_sentences in docs])

                if len(self.tokenizer.encode(doc)) + current_length > input_model_params[self.user_model]["max_context_length"]:
                    sent_limit=self.sentence_limit-1
                    while (len(self.tokenizer.encode(doc)) + current_length > input_model_params[self.user_model]["max_context_length"]) and len(docs) > 0:
                        sent_limit = self.sentence_limit-1
                        while (len(self.tokenizer.encode(doc)) + current_length > input_model_params[self.user_model]["max_context_length"]) and sent_limit > 0:
                            doc = "".join(["".join(doc_sentences[:sent_limit]) for doc_sentences in docs])
                            sent_limit -= 1
                        random_doc_idx = np.random.randint(0, len(docs))
                        docs.pop(random_doc_idx)

                    if orig_size > len(docs)+1 or self.sentence_limit > sent_limit+2:
                        self.logger.warning(f"Number of documents and/or sentences changed due to being too big for file {file.split('.')[0]}: documents={len(docs)+1}/{orig_size} - sentences={sent_limit+1}/{self.sentence_limit} - tokens={len(self.tokenizer.encode(doc)) + current_length}")
                    if len(self.tokenizer.encode(doc)) + current_length > input_model_params[self.user_model]["max_context_length"]:
                        self.logger.error(f"Summary still too big - {len(self.tokenizer.encode(doc)) + current_length} > {input_model_params[self.user_model]['max_context_length']} - {file.split('.')[0]}")
                summs.append(doc)
        return summs

    def _load_original_data(self):
        multi_lexsum = load_dataset("allenai/multi_lexsum", name="v20230518")
        modified_dataset = multi_lexsum["test"].filter(lambda x: x["summary/short"] != None)

        return modified_dataset

    def _load_prompt(self, prompt_type):
        prompt = ""
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

        self.length_system_prompt = len(self.tokenizer.encode(SYSTEM_PROMPT))
        self.user_prompt_length = len(self.tokenizer.encode(self.user_prompt))

        self.dataset = self._load_original_data()

    ##### RESPONSE
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
    def _completion_with_retry(self, file_path, **kwargs):
        # file_path: model/prompt_type/extract_sum_type/idx
        _, prompt_type, extract_sum_type, idx = file_path.split("/")[1:]
        success = False
        try:
            chat_completion = self.client.chat.completions.create(**kwargs)
            json.dump([*kwargs["messages"], {"role": "assistant", "content": chat_completion.choices[0].message.content, "usage": dict(chat_completion.usage)}], open(f"{file_path}.json", "w"), indent=2)  
            success = True  
        except Exception as exc:
            self.logger.error(f"({type(exc).__name__} ({exc.args}) - {prompt_type} - {extract_sum_type} - {idx}")
            success = False  

        return success
    
    def _response_from_llm(self):
        for extract_sum_type in EXTRACT_TYPES:
            path = f"extracted_sums/extracted_sums_json_{extract_sum_type}/"
            if extract_sum_type not in os.listdir(f"answers/{self.model}/{self.prompt_type}/"):
                os.mkdir(f"answers/{self.model}/{self.prompt_type}/{extract_sum_type}/")

            extracted_summaries = self._from_extracted(path, test_size=self.test_size, current_length=input_model_params[self.user_model]["max_tokens"]+self.user_prompt_length+self.length_system_prompt+SAFETY_MARGIN)
            errors = 0
            iterator = tqdm(extracted_summaries, desc=f"Model={self.model}, Prompt={self.prompt_type}, Selection={extract_sum_type}, Errors={errors}")
            for idx, summ in enumerate(iterator):
                if f"{idx}.json" in os.listdir(f"answers/{self.model}/{self.prompt_type}/{extract_sum_type}"):
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
                    file_path=f"answers/{self.model}/{self.prompt_type}/{extract_sum_type}/{idx}",
                    messages=message_history,
                    **self.llm_params
                )

                if not chat_completion_success:
                    errors += 1
                    iterator.set_description(f"Model={self.model}, Prompt={self.prompt_type}, Selection={extract_sum_type}, Errors={errors}, Last_error_file={idx}")

if __name__ == "__main__":
    llm = llmResponse(*sys.argv[1:])
    llm()