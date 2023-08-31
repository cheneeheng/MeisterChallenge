# Author: CHEN Ee Heng
# Date: 30.08.2023

from typing import List
import json
import pickle
import pandas as pd

from transformers import BertModel
from transformers import BertTokenizer
from transformers import M2M100Tokenizer
from transformers import M2M100ForConditionalGeneration

from googletrans import Translator

from .schema import Data
from .schema import RequestItem

LABEL_ID_PATH = "./data/label_id.json"
MODEL_PATH = "./result/classifier.pickle"


bert_en_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
bert_en_model = BertModel.from_pretrained("bert-base-cased")

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
bert_model = BertModel.from_pretrained("bert-base-multilingual-cased")

m2m_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
m2m_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")  # noqa
m2m_model = m2m_model.get_encoder()

translator = Translator()


def bert_en_encode(x):
    encoded_input = bert_en_tokenizer(x, return_tensors='pt')
    output = bert_en_model(**encoded_input)
    return output.pooler_output[0].tolist()


def bert_encode(x):
    encoded_input = bert_tokenizer(x, return_tensors='pt')
    output = bert_model(**encoded_input)
    return output.pooler_output[0].tolist()


def m2m_encode(x):
    encoded_input = m2m_tokenizer(x, return_tensors='pt')
    output = m2m_model(**encoded_input)
    return output.last_hidden_state.mean(axis=1)[0].tolist()


class Classifier:
    def __init__(self) -> None:
        with open(MODEL_PATH, 'rb') as f:
            self.model = pickle.load(f)
        with open(LABEL_ID_PATH) as f:
            label_id = json.load(f)
        self.label_id = {v: k for k, v in label_id.items()}

    @staticmethod
    def map_cell_list_to_columns(_df, column, length):
        return pd.DataFrame(_df[column].to_list(),
                            columns=[f'{column}_{i}' for i in range(length)],
                            index=_df.index)

    @staticmethod
    def idea_cell_list_to_columns(_df, column, length):
        _df_tmp = pd.DataFrame(
            _df[column].to_list(),
            columns=[f'{column}_{i}' for i in range(length)],
            index=_df.index
        )
        _df_tmp = pd.concat([_df['map_id'], _df_tmp], axis=1)
        _df_tmp = _df_tmp.groupby('map_id')[[f'{column}_{i}' for i in range(length)]].transform('max')  # noqa
        return _df_tmp

    @staticmethod
    def pad_to_512(_df, column):
        _df[f'{column}_pad512'] = _df[f'{column}'].apply(
            lambda x: (x + [0] * (512 - len(x)))[:512])
        return _df


    def preprocess(self, x: List[Data]):
        # TODO: this is a mess...

        for i, x_i in enumerate(x):
            assert isinstance(x_i.map_title, str), f"row {i} map_title is not str..."
            assert isinstance(x_i.idea_title, str), f"row {i} idea_title is not str..."

        df = pd.DataFrame([s.__dict__ for s in x])
        df = df.astype(str)
        df['map_id'] = pd.to_numeric(df['map_id'])

        df['map_title_en'] = df['map_title'].apply(lambda x: translator.translate(str(x), dest='en', src='auto').text)  # noqa
        df['map_title_en_tok_bert'] = df['map_title_en'].apply(lambda x: bert_en_tokenizer.encode(x))  # noqa
        df['map_title_en_emb_bert'] = df['map_title_en'].apply(bert_en_encode)
        df['map_title_tok_bert'] = df['map_title'].apply(lambda x: bert_tokenizer.encode(x))  # noqa
        df['map_title_emb_bert'] = df['map_title'].apply(bert_encode)
        df['map_title_tok_m2m'] = df['map_title'].apply(lambda x: m2m_tokenizer.encode(x))  # noqa
        df['map_title_emb_m2m'] = df['map_title'].apply(m2m_encode)

        df['idea_title_en'] = df['idea_title'].apply(lambda x: translator.translate(str(x), dest='en', src='auto').text)  # noqa
        df['idea_title_en_tok_bert'] = df['idea_title_en'].apply(lambda x: bert_en_tokenizer.encode(x))  # noqa
        df['idea_title_en_emb_bert'] = df['idea_title_en'].apply(bert_en_encode)  # noqa
        df['idea_title_tok_bert'] = df['idea_title'].apply(lambda x: bert_tokenizer.encode(x))  # noqa
        df['idea_title_emb_bert'] = df['idea_title'].apply(bert_encode)
        df['idea_title_tok_m2m'] = df['idea_title'].apply(lambda x: m2m_tokenizer.encode(x))  # noqa
        df['idea_title_emb_m2m'] = df['idea_title'].apply(m2m_encode)

        for col in list(df):
            if 'tok' in col:
                df = self.pad_to_512(df, col)

        df = pd.concat(
            [
                self.map_cell_list_to_columns(df, 'map_title_en_emb_bert', 768),  # noqa
                self.map_cell_list_to_columns(df, 'map_title_emb_bert', 768),  # noqa
                self.map_cell_list_to_columns(df, 'map_title_emb_m2m', 1024),  # noqa
                self.map_cell_list_to_columns(df, 'map_title_en_tok_bert_pad512', 512),  # noqa
                self.map_cell_list_to_columns(df, 'map_title_tok_bert_pad512', 512),  # noqa
                self.map_cell_list_to_columns(df, 'map_title_tok_m2m_pad512', 512),  # noqa
                self.idea_cell_list_to_columns(df, 'idea_title_en_emb_bert', 768),  # noqa
                self.idea_cell_list_to_columns(df, 'idea_title_emb_bert', 768),  # noqa
                self.idea_cell_list_to_columns(df, 'idea_title_emb_m2m', 1024),  # noqa
                self.idea_cell_list_to_columns(df, 'idea_title_en_tok_bert_pad512', 512),  # noqa
                self.idea_cell_list_to_columns(df, 'idea_title_tok_bert_pad512', 512),  # noqa
                self.idea_cell_list_to_columns(df, 'idea_title_tok_m2m_pad512', 512),  # noqa
            ],
            axis=1
        )

        df = df.drop_duplicates()

        return df

    def predict(self, x: pd.DataFrame) -> int:
        assert len(x) == 1, "API current only supports 1 unique map_id..."
        return self.model.predict(x)[0]

    def postprocess(self, x: int) -> str:
        return self.label_id[x]
