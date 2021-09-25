import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

from . import config

class ArticleTransformer:
    def __init__(self):
        self.config = config.Config()
        self._create_model()

    def _create_model(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=self.tokenizer.eos_token_id)
    
    def run_fn(self, text):
        input_ids = self.tokenizer.encode(text, return_tensors='tf')
        min_length = self.config.max_len - int(self.config.max_len / 2)
        sample_output = self.model.generate(
            input_ids, 
            do_sample=True, 
            max_length=self.config.max_len,
            min_length=min_length,
            top_p=self.config.top_p, 
            top_k=self.config.top_k
        )
        return sample_output