import logging
from typing import Any

import flag
import pycountry
import torch
from langcodes import Language
from langdetect import detect
from nltk import sent_tokenize
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Translator:
    def __init__(self):
        """
        Initializes the Translator:
        - Prepares logger for diagnostics.
        - Loads the model, tokenizer, and device.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.tokenizer, self.model, self.device = self._load_model()

    def _load_model(
        self, model_name: str = "google/madlad400-3b-mt", local_only: bool = True
    ) -> tuple[Any, Any, torch.device] | None:
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
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
            torch_dtype = (
                torch.float16 if device.type in ["cuda", "mps"] else torch.float32
            )

            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, local_files_only=True
                )
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name, torch_dtype=torch_dtype, local_files_only=local_only
                ).to(device)
                self.logger.info("✅ Loaded model from local cache.")
            except FileNotFoundError:
                self.logger.info(
                    "⬇️ Model not in local cache — downloading from Hugging Face..."
                )
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name, torch_dtype=torch_dtype
                ).to(device)

            return tokenizer, model, device
        except Exception as e:
            self.logger.error(f"Error while loading model: {e}")
            raise

    def _get_country_flag(self, language_name: str) -> str:
        """
        Convert a language name to the corresponding country flag.

        Args:
            language (str): Language name (e.g. "French")

        Returns:
            str: Country flag (e.g. "🇫🇷")
        """
        try:
            lang = Language.find(language_name)
            country_code = lang.maximize().region
            return flag.flag(country_code) if country_code else ""
        except Exception as e:
            self.logger.error(f"Error converting language to country flag: {e}")
            return ""

    def detect_language(self, text: str) -> str:
        """
        Detect the language of a text.

        Args:
            text (str): Text to be translated.

        Returns:
            str: Detected language
        """
        try:
            lang = detect(text)
            lang_name = pycountry.languages.get(alpha_2=lang).name
            country_flag = self._get_country_flag(lang_name)
            lang_country = f"{lang_name} {country_flag}"
            return lang_country if lang_country else f"Unknown language code: {lang}"
        except Exception as e:
            self.logger.error(f"Error detecting language: {e}")

    def _model_inference(self, lang: str, text: str) -> str:
        """
        Use the model for inference on a given text input.

        Args:
            lang (str): Target language code.
            text (str): Text to translate.

        Returns:
            str: The translated text.
        """
        try:
            prompt = f"<2{lang}> {text}"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_length=400,
                num_beams=4,
                early_stopping=True,
            )
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        except Exception as e:
            self.logger.error(f"Error during model inference: {e}")

    def translate(self, target_lang: str, text: str) -> str:
        """
        Translates text sentence-wise between any supported MADLAD languages.

        Args:
            target_lang (str): Target language code.
            text (str): Text to translate.

        Returns:
            str: Final translated output.
        """
        try:
            return " ".join(
                [
                    self._model_inference(target_lang, sentence)
                    for sentence in sent_tokenize(text)
                ]
            )
        except Exception as e:
            self.logger.error(f"Error during translation pipeline: {e}")
            raise RuntimeError("Translation failed") from e
