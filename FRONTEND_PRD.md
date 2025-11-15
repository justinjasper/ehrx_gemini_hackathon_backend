# Frontend Product Requirements Document (PRD)

## 1. Purpose & Scope
- Build a single-page frontend that interacts with the Cloud Run EHRX backend.
- Adhere to best practice: **upload PDFs directly to the `/upload` API via multipart/form-data**, process immediately, and render JSON results; no persistent storage assumptions.
- Scope covers UI/UX, API integration, and visualization. All ML/extraction logic remains in the backend.

## 2. Personas & Use Cases
| Persona | Goals |
|---------|-------|
| Clinical Data Engineer | Upload clinical PDFs, inspect ontology, run natural-language queries. |
| Product Demo Stakeholder | Demonstrate the end-to-end pipeline without CLI usage. |

Use cases:
1. Upload a PDF → track processing completion.
2. Inspect the extracted ontology structure (sub-documents, pages, elements).
3. Ask natural-language questions → view summarized answers + provenance bounding boxes.

## 3. Information Architecture
```
┌────────────────────────────────────────────┐
│ Header: EHRX Pipeline UI                   │
├────────────────────────────────────────────┤
│ Tabs: [ Upload | Ontology | Query ]        │
├────────────────────────────────────────────┤
│ Tab Content (single column, desktop-first) │
└────────────────────────────────────────────┘
```

## 4. Tab Requirements

### 4.1 Upload Tab
**UI Components**
- File input (accept `.pdf` only) + upload button.
- Optional page range / document type inputs (text/select).
- Progress indicator (spinner within button or linear bar).
- Status card displaying `document_id`, `status`, `total_pages`, and deep-link to Ontology tab.

**Behavior**
- User selects PDF, clicks upload → frontend issues `POST /upload` with multipart form (`file`, `page_range`, `document_type`).
- Disable inputs while request inflight; show progress.
- On success, cache `document_id` in app state; display status.
- On failure, show inline error + retry button.

### 4.2 Ontology Tab
**Inputs**
- Document selector dropdown (populate via `GET /documents`, default to last uploaded ID).

**Data Fetching**
- On selection, call `GET /documents/{id}/ontology`.

**UI/Interaction**
- JSON tree viewer:
  - Root nodes: `patient_demographics`, `sub_documents`, `pages`, `processing_stats`.
  - Sub-documents: group by `type` (medications, labs, progress_notes, etc.) with badges for page ranges & confidence.
  - Pages: list of collapsible items; each shows `elements` with `element_id`, `type`, truncated `content`.
- Provide search/filter by element type.
- Show processing stats (total_pages, total_elements, cost).

### 4.3 Query Tab
**Inputs**
- Document selector (reuse state from Ontology tab).
- Textarea for natural-language question + submit button.

**API Integration**
- POST `/documents/{id}/query` with body `{"question": "..."}`.

**Outputs**
1. **Answer Summary** card (large typography).
2. **Reasoning** text block.
3. **Matched Elements table**: columns for element_id, type, page_number, content snippet, relevance. Each row has “View on Page” button.
4. **PDF Viewer with Bounding Boxes**:
   - Render PDF pages with `pdfjs-dist` to `<canvas>` or `<img>`.
   - Overlay bounding boxes using `bbox_pixel = [x_min, y_min, x_max, y_max]`.
   - Highlight selected matched element; allow toggling multiple boxes.
   - Scaling: maintain ratio between original page dimensions and rendered canvas (store original width/height from PDF.js).

**States**
- Loading spinner between request/response.
- Empty state message when no matches.
- Error message for backend failures.

## 5. API Contract (Frontend Perspective)
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/upload` | POST (multipart) | Upload & process PDF. Returns `{document_id, status, total_pages, enhanced_json_url}`. |
| `/documents` | GET | List processed documents for selectors. |
| `/documents/{id}/ontology` | GET | Retrieve enhanced ontology JSON (sub_documents, pages[].elements[], processing_stats). |
| `/documents/{id}/query` | POST | Submit NL questions; response contains `answer_summary`, `matched_elements[]`, `reasoning`, `filter_stats`. |

## 6. Technical Considerations
- Framework: React (Vite) or Next.js SPA.
- State management: React Context or Redux Toolkit for storing `document_id`, ontology, and query results.
- HTTP: `fetch` or `axios` with abort controllers.
- PDF rendering: `pdfjs-dist` (render to canvas); overlay `<div>` or `<canvas>` for rectangles.
- Performance: lazy-load ontology JSON; virtualize long element lists.
- Accessibility: keyboard-accessible tabs, ARIA attributes for tree view, high-contrast bounding boxes.
- Error handling: toast notifications for network/API errors; per-tab error banners.
- Security: sanitize JSON snippets before rendering; handle auth headers if Cloud Run protected (identity tokens).

## 7. UX States
| State | Upload | Ontology | Query |
|-------|--------|----------|-------|
| Loading | Disable inputs, show spinner | Show skeleton tree | Disable submit, show spinner |
| Empty | Prompt to upload PDF | Message “Select a document” | Message “Upload & select document” |
| Error | Inline alert w/ retry | Alert + retry button | Alert + preserved question |
| Success | Status card + link | JSON tree populated | Answer + bounding boxes |

## 8. Milestones
1. Scaffold layout + tab navigation.
2. Implement Upload tab (file picker, API integration, state storage).
3. Build Ontology viewer (document selector, JSON tree).
4. Build Query tab (API integration, matched elements list, PDF viewer w/ bbox overlays).
5. Polish (error states, responsive layout, documentation).

## 9. Open Questions
- Max PDF size? (Cloud Run limit is ~32 MB per request; consider warning in UI).
- Should ontology JSON be downloadable?
- Need authentication support (e.g., Google Identity token) before prod?
- Bounding box scaling: confirm extraction DPI (default 200) for accurate overlays.

---
**Reminder (Backend Best Practice):** Frontend uploads PDF directly to `/upload`; backend immediately processes and returns JSON. No shared storage or asynchronous polling required in the initial version.
{
  "cells": [],
  "metadata": {
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 2
}