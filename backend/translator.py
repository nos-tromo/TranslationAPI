from langdetect import detect, LangDetectException

import nltk
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Translator:
    def __init__(self, model_name: str = "facebook/nllb-200-3.3B", local_only: bool = True):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.torch_dtype = torch.float16 if self.device.type in ["cuda", "mps"] else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            local_files_only=local_only
        ).to(self.device)

        self.iso_to_nllb = {
            "en": "eng_Latn",
            "fr": "fra_Latn",
            "de": "deu_Latn",
            "es": "spa_Latn",
            "ar": "arb_Arab",
            "ru": "rus_Cyrl",
            "tr": "tur_Latn"
        }

    def auto_detect(self, text: str) -> str:
        try:
            iso_lang = detect(text)
            return self.iso_to_nllb.get(iso_lang, "eng_Latn")
        except LangDetectException:
            return "eng_Latn"

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if source_lang == "auto":
            source_lang = self.auto_detect(text)

        if target_lang == "eng_Latn":
            return self._run_translation(text, source_lang, target_lang)
        else:
            # Step 1: to English
            intermediate = self._run_translation(text, source_lang, "eng_Latn")
            # Step 2: to target
            return self._run_translation(intermediate, "eng_Latn", target_lang)

    def _run_translation(self, text: str, source_lang: str, target_lang: str) -> str:
        self.tokenizer.src_lang = source_lang
        translated_text = []

        for sentence in nltk.sent_tokenize(text):
            inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(self.device)
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(target_lang),
                max_length=400
            )
            translated_text.append(self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True))

        return " ".join(translated_text)
