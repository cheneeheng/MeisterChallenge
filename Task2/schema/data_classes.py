# Author: CHEN Ee Heng
# Date: 30.08.2023

from typing import List
from pydantic import BaseModel


class Data(BaseModel):
    map_id: int
    map_title: str
    idea_title: str


class RequestItem(BaseModel):
    input_data: List[Data]
    return_label_ids: bool = True
    convert_label_id_to_name: bool = True