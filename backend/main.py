import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langdetect import detect, LangDetectException
import nltk
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


app = FastAPI(title="NLLB-200 Translation API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load language name mappings
language_path = Path(__file__).parent / "language_codes.json"
with open(language_path, "r", encoding="utf-8") as f:
    LANGUAGE_NAMES = json.load(f)

# Model and tokenizer
model_name = "facebook/nllb-200-3.3B"
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
torch_dtype = torch.float16 if torch.cuda.is_available() or torch.backends.mps.is_available() else torch.float32

tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch_dtype, local_files_only=True).to(device)

# ISO-to-NLLB language mapping
iso_to_nllb = {
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "ar": "arb_Arab",
    "ru": "rus_Cyrl",
    "tr": "tur_Latn"
}


class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str


def run_translation(text: str, source_lang: str, target_lang: str) -> str:
    tokenizer.src_lang = source_lang
    translated_text = []
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang),
            max_length=400
        )
        translated_text.append(tokenizer.decode(generated_tokens[0], skip_special_tokens=True))
    return " ".join(translated_text)


@app.post("/translate")
def translate(req: TranslationRequest) -> dict[str, str]:
    source_lang = req.source_lang
    if source_lang == "auto":
        try:
            iso_lang = detect(req.text)
            source_lang = iso_to_nllb.get(iso_lang, "eng_Latn")
        except LangDetectException:
            source_lang = "eng_Latn"

    if req.target_lang == "eng_Latn":
        # Direct translation to English
        output = run_translation(req.text, source_lang, req.target_lang)
    else:
        # Step 1: Translate to English
        intermediate = run_translation(req.text, source_lang, "eng_Latn")
        # Step 2: Translate from English to target
        output = run_translation(intermediate, "eng_Latn", req.target_lang)

    return {"translation": output}


@app.get("/languages")
def get_languages():
    return [
        {"code": code, "name": LANGUAGE_NAMES.get(code, code)}
        for code in LANGUAGE_NAMES.keys()
    ]
