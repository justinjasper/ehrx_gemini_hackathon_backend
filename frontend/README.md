# EHRX Frontend

React + Vite single-page app that satisfies the requirements in `FRONTEND_PRD.md`.

## Prerequisites
- Node.js 18 or later
- Backend API deployed (Cloud Run or local) exposing the `/upload`, `/documents`, `/documents/{id}/ontology`, `/documents/{id}/query` endpoints

## Configure
Create `.env` in this directory:
```bash
cp env.example .env
```
Edit `VITE_API_BASE_URL` to point at the backend base URL (e.g., `https://ehrx-backend-xxxx.a.run.app` or `http://localhost:8080`).

## Install & Run (Local)
```bash
cd frontend
npm install
npm run dev
```
Visit http://localhost:5173 . All API calls proxy to `VITE_API_BASE_URL`.

## Build
```bash
npm run build
npm run preview
```

## Containerization & Cloud Run Deployment

The included Dockerfile performs a multi-stage build and serves the static bundle with `serve`.

1. **Build container**
```bash
cd frontend
docker build \
  --build-arg VITE_API_BASE_URL=https://ehrx-backend-xxxx.a.run.app \
  -t us-central1-docker.pkg.dev/$PROJECT_ID/ehrx-frontend/app .
docker push us-central1-docker.pkg.dev/$PROJECT_ID/ehrx-frontend/app
```

2. **Deploy to Cloud Run**
```bash
gcloud run deploy ehrx-frontend \
  --image us-central1-docker.pkg.dev/$PROJECT_ID/ehrx-frontend/app \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated
```

3. **Configure CORS**
Ensure the backend allows the frontend origin. For FastAPI, add `CORSMiddleware` or configure an HTTPS ingress that permits the frontend domain.

## Usage Workflow
1. **Upload Tab** – select PDF, set optional page range, click “Upload & Process.” Backend response stores `document_id`.
2. **Ontology Tab** – choose a document from the dropdown to inspect the enhanced JSON (sub_docs/pages/elements).
3. **Query Tab** – type NL question, run query, see `answer_summary`, matched elements, and bounding boxes over the PDF (available for PDFs uploaded this session).

Refer to `FRONTEND_PRD.md` for detailed UX/feature requirements.

