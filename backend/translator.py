from langdetect import detect, LangDetectException
import logging

import nltk
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Translator:
    def __init__(self):
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

    def _load_model(self, model_name: str = "facebook/nllb-200-3.3B", local_only: bool = True) -> tuple:
        try:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            )
            torch_dtype = torch.float16 if device.type in ["cuda", "mps"] else torch.float32

            try:
                # Attempt to load the model and tokenizer from local files
                tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    local_files_only=local_only
                ).to(device)
                self.logger.info("✅ Loaded model from local cache.")
            except FileNotFoundError:
                # If files are not found locally, download them from Hugging Face
                self.logger.info("⬇️ Model not in local cache — downloading from Hugging Face...")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype
                ).to(device)

            return tokenizer, model, device
        except Exception as e:
            logging.error(f"Error while loading model: {e}")

    def _detect_language(self, text: str) -> str:
        try:
            iso_lang = detect(text)
            return self.iso_to_nllb.get(iso_lang, "eng_Latn")
        except LangDetectException:
            return "eng_Latn"

    def _model_inference(self, text: str, source_lang: str, target_lang: str) -> str:
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

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        try:
            if source_lang == "auto":
                source_lang = self._detect_language(text)

            if target_lang == "eng_Latn":
                return self._model_inference(text, source_lang, target_lang)
            else:
                # Step 1: to English
                intermediate = self._model_inference(text, source_lang, "eng_Latn")
                # Step 2: to target
                return self._model_inference(intermediate, "eng_Latn", target_lang)
        except Exception as e:
            self.logger.info(f"Error during translation pipeline: {e}")
