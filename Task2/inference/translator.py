# Author: CHEN Ee Heng
# Date: 31.08.2023

from googletrans import Translator


class GoogleTranslator():
    def __init__(self) -> None:
        self.translator = Translator()

    def translate_to_en(self, x: str) -> str:
        return self.translator.translate(x, dest='en', src='auto').text

    def detect_language(self, x: str) -> str:
        return self.translator.detect(x).lang
