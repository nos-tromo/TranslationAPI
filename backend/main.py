import logging
from datetime import datetime
from pathlib import Path

# === Logging Configuration ===

logfile_name = f"translator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logs_dir = Path(".logs")
logs_dir.mkdir(parents=True, exist_ok=True)
logfile_path = logs_dir / logfile_name
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(logfile_path), logging.StreamHandler()],
)


import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from translator import Translator

# === FastAPI Setup ===

app = FastAPI(
    title="MADLAD-400 Translation API",
    description="Translate text using Google's MADLAD-400 model via Hugging Face Transformers.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
translator = Translator()


class TranslationRequest(BaseModel):
    """
    Request schema for translation endpoint.

    Attributes:
        target_lang (str): The language code of the target language.
        text (str): Input text to translate.
    """

    target_lang: str
    text: str


def _load_language_codes(
    filename: str = "language_codes.json",
) -> dict[str, str] | None:
    """
    Loads a mapping of language codes to human-readable names from a JSON file.

    Args:
        filename (str): Name of the JSON file to load (default: 'language_codes.json').

    Returns:
        dict[str, str] | None: A dictionary mapping language codes to names, or None if the file is missing.
    """
    try:
        language_path = Path(__file__).parent / filename
        with open(language_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as e:
        logging.error(f"Error loading language code file: {e}")
        raise


@app.post(
    "/translate",
    summary="Translate text",
    description="Translates input text to a target language using the MADLAD-400 model. ",
    tags=["Translation"],
)
def translate(req: TranslationRequest) -> dict[str, str] | None:
    """
    POST endpoint for translating text between languages.

    Accepts JSON payload with text and target language.
    If source language is 'auto', language detection will be used.

    Args:
        req (TranslationRequest): The translation input parameters.

    Returns:
        dict: A dictionary containing the translated text.
    """
    try:
        detected_lang = translator.detect_language(req.text)
        result = translator.translate(req.target_lang, req.text)
        return {
            "translation": result,
            "detected_language": detected_lang,
        }
    except Exception as e:
        logging.error(f"Error on /translate endpoint: {e}")
        raise HTTPException(status_code=500, detail="Translation failed.")


@app.get(
    "/languages",
    summary="List supported languages",
    description="Returns a list of supported MADLAD-400 language codes with human-readable names, "
    "based on the included `language_codes.json` file.",
    tags=["Metadata"],
)
def get_languages() -> list[dict[str, str]] | None:
    """
    GET endpoint for retrieving available translation languages.

    Returns:
        list[dict]: A list of objects with 'code' and 'name' for each language.
    """
    try:
        LANGUAGE_NAMES = _load_language_codes()
        return [
            {"code": code, "name": LANGUAGE_NAMES.get(code, code)}
            for code in LANGUAGE_NAMES
        ]
    except Exception as e:
        logging.error(f"Error on /languages endpoint: {e}")
        raise HTTPException(status_code=500, detail="Failed to load language list.")
