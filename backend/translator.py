from langdetect import detect, LangDetectException
import logging
from typing import Any

import nltk
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Translator:
    """
    A translation pipeline using the NLLB-200 model from Hugging Face Transformers.

    Supports sentence-wise translation, automatic language detection,
    and multistep translation through English as an intermediate language
    for improved output quality.
    """

    def __init__(self):
        """
        Initializes the Translator:
        - Loads the model, tokenizer, and device.
        - Sets up ISO-to-NLLB language code mapping.
        - Prepares logger for diagnostics.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.tokenizer, self.model, self.device = self._load_model()
        self.iso_to_nllb = {
            "en": "eng_Latn",
            "fr": "fra_Latn",
            "de": "deu_Latn",
            "es": "spa_Latn",
            "ar": "arb_Arab",
            "ru": "rus_Cyrl",
            "tr": "tur_Latn"
        }

    def _load_model(self, model_name: str = "facebook/nllb-200-3.3B", local_only: bool = True) -> tuple[Any, Any, torch.device] | None:
        """
        Loads the translation model and tokenizer.

        Tries to load from local cache first. If not found, downloads from Hugging Face Hub.

        Args:
            model_name (str): The name of the pretrained model.
            local_only (bool): Whether to restrict loading to local files only.

        Returns:
            tuple: (tokenizer, model, device) if successful.
        """
        try:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            )
            torch_dtype = torch.float16 if device.type in ["cuda", "mps"] else torch.float32

            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name, torch_dtype=torch_dtype, local_files_only=local_only
                ).to(device)
                self.logger.info("✅ Loaded model from local cache.")
            except FileNotFoundError:
                self.logger.info("⬇️ Model not in local cache — downloading from Hugging Face...")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name, torch_dtype=torch_dtype
                ).to(device)

            return tokenizer, model, device
        except Exception as e:
            logging.error(f"Error while loading model: {e}")

    def _detect_language(self, text: str) -> str:
        """
        Detects the language of a given text and maps it to an NLLB code.

        Args:
            text (str): The input text to analyze.

        Returns:
            str: The NLLB language code or "eng_Latn" as fallback.
        """
        try:
            iso_lang = detect(text)
            return self.iso_to_nllb.get(iso_lang, "eng_Latn")
        except LangDetectException:
            return "eng_Latn"

    def _model_inference(self, text: str, source_lang: str, target_lang: str) -> str | None:
        """
        Translates the given text sentence-by-sentence using the model.

        Args:
            text (str): Input text to translate.
            source_lang (str): NLLB code of source language.
            target_lang (str): NLLB code of target language.

        Returns:
            str: Translated text.
        """
        try:
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
        except Exception as e:
            self.logger.info(f"Error while running model inference: {e}")

    def translate(self, text: str, source_lang: str, target_lang: str) -> str | None:
        """
        Translates text between any supported NLLB languages.

        If source_lang is 'auto', attempts language detection.
        If translating between two non-English languages, routes via English.

        Args:
            text (str): Text to translate.
            source_lang (str): NLLB source language code or 'auto'.
            target_lang (str): NLLB target language code.

        Returns:
            str: Final translated output.
        """
        try:
            if source_lang == "auto":
                source_lang = self._detect_language(text)

            if target_lang == "eng_Latn":
                return self._model_inference(text, source_lang, target_lang)
            else:
                intermediate = self._model_inference(text, source_lang, "eng_Latn")
                return self._model_inference(intermediate, "eng_Latn", target_lang)
        except Exception as e:
            self.logger.info(f"Error during translation pipeline: {e}")
