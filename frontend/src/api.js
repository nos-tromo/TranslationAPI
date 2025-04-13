import axios from "axios";

const BASE_URL = "http://localhost:8000";

export async function fetchLanguages() {
    const res = await axios.get(`${BASE_URL}/languages`);
    return res.data;
}

export async function translateText({ text, source_lang, target_lang }) {
    const res = await axios.post(`${BASE_URL}/translate`, {
        text,
        source_lang,
        target_lang
    });
    return res.data;
}
