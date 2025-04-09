import json
from pathlib import Path
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

# Load languages
language_path = Path(__file__).parent / "language_codes.json"
with open(language_path, "r", encoding="utf-8") as f:
    LANGUAGE_NAMES = json.load(f)

# Initialize translator
translator = Translator()


class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str


@app.post("/translate")
def translate(req: TranslationRequest):
    result = translator.translate(req.text, req.source_lang, req.target_lang)
    return {"translation": result}


@app.get("/languages")
def get_languages():
    return [{"code": code, "name": LANGUAGE_NAMES.get(code, code)} for code in LANGUAGE_NAMES]
