# NLLB TranslationAPI

#### Backend setup
```bash
cd backend
uv sync
uvicorn main:app --reload
```
Local browser access (backend): `http://127.0.0.1:8000/docs`
#### Frontend setup
```bash
npm create vite@latest frontend -- --template react
cd frontend
npm install
npm install axios
npm run dev
```
Local browser access (frontend): `http://localhost:5173/`

#### Docker setup
```
docker compose up --build
```
Local browser access: `http://localhost:8080/`
