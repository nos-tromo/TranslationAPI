# NLLB TranslationAPI

#### Backend setup
```bash
cd backend
uv sync
uvicorn main:app --reload
```
Access backend endpoint in the browser: `http://127.0.0.1:8000/docs`
#### Frontend setup
```bash
npm create vite@latest frontend -- --template react
cd frontend
npm install
npm install axios
npm run dev
```
Access frontend in the browser: `http://localhost:5173/`

#### Docker setup
```
docker compose up --build
```
