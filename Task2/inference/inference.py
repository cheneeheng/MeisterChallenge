# Author: CHEN Ee Heng
# Date: 30.08.2023

from typing import List
import json
import pickle
import pandas as pd

from ..schema.data_classes import Data

from .translator import GoogleTranslator
from .m2m import M2M
from .bert import BERTEnglish
from .bert import BERTMultilingual
from .df_utils import cell_list_to_columns
from .df_utils import cell_list_to_columns_and_max
from .df_utils import pad_cell_list_to_512

LABEL_ID_PATH = "./data/label_id.json"
MODEL_PATH = "./result/classifier.pickle"


class Classifier:
    def __init__(self) -> None:
        with open(MODEL_PATH, 'rb') as f:
            self.model = pickle.load(f)
        with open(LABEL_ID_PATH) as f:
            label_id = json.load(f)
        self.label_id = {v: k for k, v in label_id.items()}

        self.translator = GoogleTranslator()
        self.bert = BERTMultilingual()
        self.bert_en = BERTEnglish()
        self.m2m = M2M()

    def preprocess(self, x: List[Data]):
        # TODO: this is a mess...

        for i, x_i in enumerate(x):
            assert isinstance(
                x_i.map_title, str), f"row {i} map_title is not str..."
            assert isinstance(
                x_i.idea_title, str), f"row {i} idea_title is not str..."

        df = pd.DataFrame([s.__dict__ for s in x])
        df = df.astype(str)
        df['map_id'] = pd.to_numeric(df['map_id'])

        df['map_title_en'] = df['map_title'].apply(self.translator.translate_to_en)  # noqa
        df['map_title_en_tok_bert'] = df['map_title_en'].apply(self.bert_en.tokenize)  # noqa
        df['map_title_en_emb_bert'] = df['map_title_en'].apply(self.bert_en.encode)  # noqa
        df['map_title_tok_bert'] = df['map_title'].apply(self.bert.tokenize)
        df['map_title_emb_bert'] = df['map_title'].apply(self.bert.encode)
        df['map_title_tok_m2m'] = df['map_title'].apply(self.m2m.tokenize)
        df['map_title_emb_m2m'] = df['map_title'].apply(self.m2m.encode)

        df['idea_title_en'] = df['idea_title'].apply(self.translator.translate_to_en)  # noqa
        df['idea_title_en_tok_bert'] = df['idea_title_en'].apply(self.bert_en.tokenize)  # noqa
        df['idea_title_en_emb_bert'] = df['idea_title_en'].apply(self.bert_en.encode)  # noqa
        df['idea_title_tok_bert'] = df['idea_title'].apply(self.bert.tokenize)
        df['idea_title_emb_bert'] = df['idea_title'].apply(self.bert.encode)
        df['idea_title_tok_m2m'] = df['idea_title'].apply(self.m2m.tokenize)
        df['idea_title_emb_m2m'] = df['idea_title'].apply(self.m2m.encode)

        for col in list(df):
            if 'tok' in col:
                df = pad_cell_list_to_512(df, col)

        df = pd.concat(
            [
                cell_list_to_columns(df, 'map_title_en_emb_bert', 768),  # noqa
                cell_list_to_columns(df, 'map_title_emb_bert', 768),  # noqa
                cell_list_to_columns(df, 'map_title_emb_m2m', 1024),  # noqa
                cell_list_to_columns(df, 'map_title_en_tok_bert_pad512', 512),  # noqa
                cell_list_to_columns(df, 'map_title_tok_bert_pad512', 512),  # noqa
                cell_list_to_columns(df, 'map_title_tok_m2m_pad512', 512),  # noqa
                cell_list_to_columns_and_max(df, 'idea_title_en_emb_bert', 768),  # noqa
                cell_list_to_columns_and_max(df, 'idea_title_emb_bert', 768),  # noqa
                cell_list_to_columns_and_max(df, 'idea_title_emb_m2m', 1024),  # noqa
                cell_list_to_columns_and_max(df, 'idea_title_en_tok_bert_pad512', 512),  # noqa
                cell_list_to_columns_and_max(df, 'idea_title_tok_bert_pad512', 512),  # noqa
                cell_list_to_columns_and_max(df, 'idea_title_tok_m2m_pad512', 512),  # noqa
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
