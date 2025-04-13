import { useEffect, useState } from "react";
import { fetchLanguages, translateText } from "./api";

function App() {
    const [languages, setLanguages] = useState([]);
    const [text, setText] = useState("");
    const [targetLang, setTargetLang] = useState("en");
    const [detectedLang, setDetectedLang] = useState("");
    const [translation, setTranslation] = useState("");
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        fetchLanguages().then((langs) => {
            setLanguages(langs);
            setTargetLang("en"); // Default target language
        });
    }, []);

    const handleTranslate = async () => {
        if (!text || !targetLang) return;
        setLoading(true);
    
        const result = await translateText({
            text,
            target_lang: targetLang,
        });
    
        setTranslation(result.translation);
        setDetectedLang(result.detected_language);
        setLoading(false);
    };

    const handleFileUpload = (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (event) => setText(event.target.result);
        reader.readAsText(file);
    };

    return (
        <div style={{ padding: "2rem", fontFamily: "sans-serif", maxWidth: "800px", margin: "auto" }}>
            <h1>🌐 MADLAD-400 Translator</h1>

            {/* File Upload */}
            <div style={{ marginBottom: "1rem" }}>
                <input type="file" accept=".txt" onChange={handleFileUpload} />
            </div>

            {/* Input Text */}
            <div style={{ marginBottom: "1rem" }}>
                <textarea
                    rows="6"
                    style={{ width: "100%", fontSize: "1rem" }}
                    placeholder="Enter text to translate..."
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                />
            </div>

            {/* Target Language Selector */}
            <div style={{ marginBottom: "1rem" }}>
                <select value={targetLang} onChange={(e) => setTargetLang(e.target.value)}>
                    {languages.map(({ code, name }) => (
                        <option key={code} value={code}>{name}</option>
                    ))}
                </select>
            </div>

            {/* Translate Button */}
            <button onClick={handleTranslate} disabled={loading}>
                {loading ? "Translating..." : "Translate"}
            </button>

            {/* Output Box with Copy */}
            {translation && (
                <div
                    style={{
                        marginTop: "2rem",
                        padding: "1rem",
                        backgroundColor: "#1e1e1e",
                        color: "#f1f1f1",
                        borderRadius: "6px",
                    }}
                >
                    {detectedLang && (
                        <p style={{ marginBottom: "0.5rem" }}>
                            <strong>Source language:</strong> {detectedLang}
                        </p>
                    )}
                    <strong>Translation:</strong>
                    <button
                        onClick={() => navigator.clipboard.writeText(translation)}
                        style={{
                            marginLeft: "1rem",
                            fontSize: "0.9rem",
                            padding: "0.2rem 0.5rem",
                            cursor: "pointer",
                            backgroundColor: "#333",
                            color: "#f1f1f1",
                            border: "none",
                            borderRadius: "4px",
                        }}
                    >
                        📋 Copy
                    </button>
                    <button
                        onClick={() => {
                            setText("");
                            setTranslation("");
                            setDetectedLang("");
                        }}
                        style={{
                            marginTop: "1rem",
                            marginLeft: "1rem",
                            padding: "0.4rem 0.8rem",
                            fontSize: "0.9rem",
                            cursor: "pointer",
                            backgroundColor: "#555",
                            color: "white",
                            border: "none",
                            borderRadius: "4px",
                        }}
                    >
                        🔄 Reset
                    </button>
                    <p style={{ marginTop: "0.5rem", whiteSpace: "pre-wrap" }}>{translation}</p>
                </div>
            )}
        </div>
    );
}

export default App;
