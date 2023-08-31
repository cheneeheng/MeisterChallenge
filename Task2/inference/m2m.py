# Author: CHEN Ee Heng
# Date: 31.08.2023

from typing import List

from transformers import M2M100Tokenizer
from transformers import M2M100ForConditionalGeneration

DEFAULT_MODEL_NAME = "facebook/m2m100_418M"


class M2M:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        self.model = M2M100ForConditionalGeneration.from_pretrained(
            model_name).get_encoder()

    def tokenize(self, x: str) -> List[float]:
        return self.tokenizer.encode(x)

    def encode(self, x: str) -> List[float]:
        encoded_input = self.tokenizer(x, return_tensors='pt')
        output = self.model(**encoded_input)
        return output.last_hidden_state.mean(axis=1)[0].tolist()
