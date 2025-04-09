import logging
from datetime import datetime
from pathlib import Path

logfile_name = f"translator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logs_dir = Path(".logs")
logs_dir.mkdir(parents=True, exist_ok=True)
logfile_path = logs_dir / logfile_name
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(logfile_path),
        logging.StreamHandler()
    ]
)


import json
from pydantic import BaseModel

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from translator import Translator


app = FastAPI(title="NLLB-200 Translation API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
translator = Translator()


class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str


def _load_language_codes(filename: str = "language_codes.json") -> dict[str, str] | None:
    try:
        language_path = Path(__file__).parent / filename
        with open(language_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Error loading language code file: {e}")


@app.post("/translate")
def translate(req: TranslationRequest):
    try:
        result = translator.translate(req.text, req.source_lang, req.target_lang)
        return {"translation": result}
    except Exception as e:
        logging.error(f"Error on /translate endpoint: {e}")


@app.get("/languages")
def get_languages():
    try:
        LANGUAGE_NAMES = _load_language_codes()
        return [{"code": code, "name": LANGUAGE_NAMES.get(code, code)} for code in LANGUAGE_NAMES]
    except Exception as e:
        logging.error(f"Error on /languages endpoint: {e}")
