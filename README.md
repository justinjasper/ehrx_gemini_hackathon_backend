# EHRX - AI-Powered EHR Data Liberation

**Transform massive, unstructured EHR PDFs into queryable, structured data in minutes.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Try%20Now-blue)](https://frontend-795204058658.europe-west1.run.app/)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Google Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-orange.svg)](https://ai.google.dev/)

---

## 🎯 Live Demo

**👉 [Try EHRX Now](https://frontend-795204058658.europe-west1.run.app/)**

Upload a clinical PDF or select from sample documents, then query the extracted data using natural language!

---

## 📋 Table of Contents

- [The Problem](#the-problem)
- [The Solution](#the-solution)
- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Deployment](#deployment)
- [Development](#development)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## 🚨 The Problem

When patients switch hospitals, their medical records should follow them seamlessly, but they don't.

Data interoperability laws require "smooth" data exchange between EHR systems. In practice, hospitals export **600+ page PDF dumps**—unstructured, unsearchable, and nearly useless. Clinicians spend hours manually hunting through these documents for critical patient information, risking errors and delays in care.

---

## ✨ The Solution

**EHRX transforms massive, unstructured EHR PDFs into queryable, structured data in minutes.**

Using Google's Gemini 2.5 vision-language models, EHRX:

- **Reads** 650+ page EHR documents with medical context understanding
- **Classifies** content into 19 semantic clinical categories (labs, medications, vitals, etc.)
- **Traces** every data point back to its exact source location for verification
- **Enables** natural language queries: *"What are the patient's current medications?"* → Instant, sourced answer

---

## 🎯 Features

### Core Capabilities

- **📄 Multi-Page PDF Processing**: Process documents with 650+ pages efficiently
- **🧠 Semantic Classification**: Automatically categorizes content into 19+ clinical element types
- **🔍 Natural Language Queries**: Ask questions in plain English and get instant answers
- **📍 Full Provenance**: Every data point traced to exact page and pixel coordinates
- **📊 Sub-Document Grouping**: Automatically organizes pages into clinical sections (labs, medications, notes, etc.)
- **✅ Confidence Scoring**: AI confidence metrics with automatic flags for human review
- **🔒 HIPAA-Compliant**: Full data lineage for auditing and verification

### Technical Features

- **Vision-Language Model Integration**: Google Gemini 2.5 Flash & Pro via Vertex AI
- **RESTful API**: FastAPI backend with comprehensive endpoints
- **Modern Frontend**: React + TypeScript with PDF visualization and bounding box overlays
- **Cloud-Native**: Deployed on Google Cloud Run for scalability
- **Docker Support**: Complete containerization for easy deployment

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (React)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │  Upload  │  │ Ontology │  │  Query   │                  │
│  │   Tab    │  │   Tab    │  │   Tab    │                  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                  │
└───────┼─────────────┼──────────────┼────────────────────────┘
        │             │              │
        └─────────────┴──────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │   FastAPI Backend (Cloud Run)│
        │  ┌────────────────────────┐ │
        │  │  /upload               │ │
        │  │  /sample-documents     │ │
        │  │  /documents/{id}/...   │ │
        │  └──────────┬─────────────┘ │
        └─────────────┼───────────────┘
                      │
        ┌─────────────┴───────────────┐
        │                             │
        ▼                             ▼
┌───────────────┐           ┌──────────────────┐
│  VLM Pipeline │           │  Query Agent     │
│  (Gemini 2.5) │           │  (Hybrid Filter) │
└───────┬───────┘           └──────────────────┘
        │
        ▼
┌─────────────────────────────┐
│   Vertex AI (Google Cloud)   │
│   - Gemini 2.5 Flash         │
│   - Gemini 2.5 Pro           │
└─────────────────────────────┘
```

### Key Components

1. **Frontend**: React + TypeScript SPA with three main tabs
2. **Backend API**: FastAPI service handling PDF processing and queries
3. **VLM Pipeline**: Document extraction using Gemini vision-language models
4. **Query Agent**: Hybrid filtering system for natural language queries
5. **Storage**: Temporary file storage (can be extended to Cloud Storage)

---

## 🚀 Quick Start

### Option 1: Use the Live Demo

Visit **[https://frontend-795204058658.europe-west1.run.app/](https://frontend-795204058658.europe-west1.run.app/)** and start processing documents immediately!

### Option 2: Local Development

```bash
# Clone the repository
git clone https://github.com/justinjasper/ehrx_gemini_hackathon.git
cd ehrx_gemini_hackathon

# Set up backend
pip install -r requirements.txt
export GCP_PROJECT_ID=your-project-id
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json

# Run backend
python app.py

# Set up frontend (in separate terminal)
cd frontend
npm install
cp env.example .env
# Edit .env: VITE_API_BASE_URL=http://localhost:8080
npm run dev
```

---

## 📦 Installation

### Prerequisites

- **Python 3.11+**
- **Node.js 18+** (for frontend)
- **Google Cloud Project** with Vertex AI API enabled
- **GCP Service Account** with Vertex AI permissions

### Backend Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0

# Configure environment
export GCP_PROJECT_ID=your-project-id
export GCP_LOCATION=us-central1
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

### Frontend Setup

```bash
cd frontend
npm install

# Configure API endpoint
cp env.example .env
# Edit .env and set VITE_API_BASE_URL to your backend URL
```

---

## 💻 Usage

### Web Interface

1. **Upload Tab**: Select a sample document or upload your own PDF
2. **Ontology Tab**: Browse the extracted document structure (sub-documents, pages, elements)
3. **Query Tab**: Ask natural language questions and see results with bounding box overlays

### Command Line

```bash
# Process a PDF
python run_mvp_pipeline.py

# Query existing results
python test_query_only.py

# Run tests
pytest tests/
```

### API Usage

```python
import requests

# Upload and process PDF
response = requests.post(
    "https://ehrx-gemini-hackathon-795204058658.europe-west1.run.app/upload",
    files={"file": open("document.pdf", "rb")},
    data={"page_range": "1-10"}
)
document_id = response.json()["document_id"]

# Query the document
query_response = requests.post(
    f"https://ehrx-gemini-hackathon-795204058658.europe-west1.run.app/documents/{document_id}/query",
    json={"question": "What medications is the patient taking?"}
)
print(query_response.json()["answer_summary"])
```

---

## 📚 API Documentation

### Base URL

```
https://ehrx-gemini-hackathon-795204058658.europe-west1.run.app
```

### Endpoints

#### `GET /`
API information and available endpoints.

#### `GET /health`
Health check endpoint.

#### `GET /sample-documents`
List available sample PDF documents.

**Response:**
```json
{
  "samples": [
    {
      "id": "Psychiatric evaluation.pdf",
      "filename": "Psychiatric evaluation.pdf",
      "display_name": "Psychiatric evaluation",
      "size_bytes": 123456
    }
  ]
}
```

#### `POST /upload`
Upload and process a PDF file.

**Request:**
- `file`: PDF file (multipart/form-data)
- `page_range`: Optional, default "all" (e.g., "1-10")
- `document_type`: Optional, default "Clinical EHR"

**Response:**
```json
{
  "document_id": "document_1234567890",
  "status": "complete",
  "total_pages": 22,
  "enhanced_json_url": "/documents/document_1234567890/ontology"
}
```

#### `POST /sample-documents/{filename}/process`
Process a bundled sample document.

**Request:**
- `page_range`: Optional, default "all"
- `document_type`: Optional, default "Clinical EHR"

**Response:** Same as `/upload`

#### `GET /documents/{id}/ontology`
Get the full enhanced JSON structure.

**Response:**
```json
{
  "document_id": "...",
  "sub_documents": [...],
  "pages": [
    {
      "page_number": 1,
      "elements": [
        {
          "element_id": "E_0001",
          "type": "medication_table",
          "content": "...",
          "bbox_pixel": [66.0, 90.0, 844.0, 308.0],
          "bbox_pdf": [...]
        }
      ]
    }
  ],
  "processing_stats": {...}
}
```

#### `POST /documents/{id}/query`
Query a processed document with natural language.

**Request:**
```json
{
  "question": "What medications is the patient taking?"
}
```

**Response:**
```json
{
  "answer_summary": "Patient is taking aspirin 81mg daily...",
  "matched_elements": [
    {
      "element_id": "E_0003",
      "type": "medication_table",
      "content": "...",
      "page_number": 7,
      "bbox_pixel": [66.0, 90.0, 844.0, 308.0],
      "relevance": "This table lists all medications..."
    }
  ],
  "reasoning": "Found medication information in table on page 7",
  "filter_stats": {"reduction_ratio": "1.3x"}
}
```

#### `GET /documents`
List all processed documents.

**Interactive API Documentation:**
Visit `https://ehrx-gemini-hackathon-795204058658.europe-west1.run.app/docs` for Swagger UI.

---

## ☁️ Deployment

### Google Cloud Run

#### Backend Deployment

```bash
# Set variables
export PROJECT_ID=$(gcloud config get-value project)
export REGION=europe-west1

# Build and push
gcloud builds submit \
  --tag ${REGION}-docker.pkg.dev/${PROJECT_ID}/ehrx-repo/ehrx-backend:latest

# Deploy
gcloud run deploy ehrx-gemini-hackathon \
  --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/ehrx-repo/ehrx-backend:latest \
  --region ${REGION} \
  --platform managed \
  --allow-unauthenticated \
  --memory 8Gi \
  --cpu 4 \
  --timeout 3600 \
  --set-env-vars "GCP_PROJECT_ID=${PROJECT_ID},CORS_ALLOW_ORIGINS=https://frontend-795204058658.europe-west1.run.app"
```

#### Frontend Deployment

```bash
cd frontend

# Build with backend URL
docker build \
  --build-arg VITE_API_BASE_URL=https://ehrx-gemini-hackathon-795204058658.europe-west1.run.app \
  -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/ehrx-repo/ehrx-frontend:latest .

# Push
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/ehrx-repo/ehrx-frontend:latest

# Deploy
gcloud run deploy ehrx-frontend \
  --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/ehrx-repo/ehrx-frontend:latest \
  --region ${REGION} \
  --allow-unauthenticated \
  --port 8080
```

### Docker Compose (Local)

```bash
docker-compose up --build
```

---

## 🛠️ Development

### Project Structure

```
ehrx_gemini_hackathon/
├── app.py                 # FastAPI backend server
├── run_mvp_pipeline.py    # CLI pipeline runner
├── requirements.txt       # Python dependencies
├── Dockerfile            # Backend container
├── ehrx/                # Core package
│   ├── vlm/            # Vision-language model integration
│   ├── agent/          # Query agent
│   ├── pdf/            # PDF processing
│   └── core/           # Configuration and utilities
├── frontend/            # React frontend
│   ├── src/
│   │   ├── components/ # React components
│   │   ├── api.ts      # API client
│   │   └── types.ts    # TypeScript types
│   └── Dockerfile      # Frontend container
├── configs/             # Configuration files
├── SampleEHR_docs/      # Sample PDF documents
└── tests/              # Test suite
```

### Running Tests

```bash
# Backend tests
pytest tests/

# With coverage
pytest --cov=ehrx tests/

# Frontend tests (if configured)
cd frontend
npm test
```

### Code Style

- **Python**: Follow PEP 8, use type hints
- **TypeScript**: ESLint + Prettier configuration
- **API**: Follow RESTful conventions

---

## 🧪 Testing

### Manual Testing

1. **Upload Flow**: Upload a PDF and verify processing completes
2. **Ontology View**: Check that document structure is correctly displayed
3. **Query Flow**: Test various natural language queries
4. **Bounding Boxes**: Verify PDF overlays match extracted elements

### Sample Queries

- "What medications is the patient taking?"
- "What were the patient's lab results?"
- "Show me all vital signs"
- "What are the patient's active problems?"
- "Were there any radiology studies?"

---

## 📊 Performance & Costs

### Processing Performance

- **Small documents (1-10 pages)**: ~30-60 seconds
- **Medium documents (50-100 pages)**: ~5-10 minutes
- **Large documents (500+ pages)**: ~30-60 minutes

### Cost Estimates

- **Gemini 2.5 Flash**: ~$0.10 per 1M input tokens, $0.40 per 1M output tokens
- **Processing 650 pages**: ~$1-2 per document
- **Query operations**: ~$0.01-0.05 per query

### Optimization Tips

- Use `page_range` parameter to process specific pages
- Cache processed documents for repeated queries
- Use Gemini Flash for extraction, Pro only for complex reasoning

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Write tests for new features
- Update documentation as needed
- Follow existing code style
- Add type hints to Python code
- Use meaningful commit messages

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Team

**Justin Jasper**  
Stanford CS (AI) Master's Student; prior experience in Healthcare AI development

**Xander Hnasko**  
Stanford CS (AI) Master's Student; prior experience in unstructured data ingestion within private financial markets

---

## 🙏 Acknowledgments

- Google Gemini 2.5 models via Vertex AI
- FastAPI for the excellent web framework
- React and TypeScript communities
- All contributors and testers

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/justinjasper/ehrx_gemini_hackathon/issues)
- **Live Demo**: [https://frontend-795204058658.europe-west1.run.app/](https://frontend-795204058658.europe-west1.run.app/)
- **API Docs**: [https://ehrx-gemini-hackathon-795204058658.europe-west1.run.app/docs](https://ehrx-gemini-hackathon-795204058658.europe-west1.run.app/docs)

---

## 🎯 Roadmap

- [ ] Cloud Storage integration for persistent document storage
- [ ] Batch processing API for multiple documents
- [ ] Enhanced visualization with interactive PDF viewer
- [ ] Export to FHIR format
- [ ] Multi-language support
- [ ] Advanced query capabilities with temporal reasoning

---

**Unlocking healthcare data, one PDF at a time.** 🏥✨

