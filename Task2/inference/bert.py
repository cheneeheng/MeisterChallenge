# Author: CHEN Ee Heng
# Date: 31.08.2023

from typing import List

from transformers import BertModel
from transformers import BertTokenizer

DEFAULT_BERTMultilingual_NAME = "bert-base-multilingual-cased"
DEFAULT_BERTEnglish_NAME = "bert-base-cased"


class BERTBase:
    def __init__(self, model_name: str) -> None:
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def tokenize(self, x: str) -> List[float]:
        return self.tokenizer.encode(x)

    def encode(self, x: str) -> List[float]:
        encoded_input = self.tokenizer(x, return_tensors='pt')
        output = self.model(**encoded_input)
        return output.pooler_output[0].tolist()


class BERTMultilingual(BERTBase):
    def __init__(self, model_name: str = DEFAULT_BERTMultilingual_NAME) -> None:
        super().__init__(model_name)


class BERTEnglish(BERTBase):
    def __init__(self, model_name: str = DEFAULT_BERTEnglish_NAME) -> None:
        super().__init__(model_name)
