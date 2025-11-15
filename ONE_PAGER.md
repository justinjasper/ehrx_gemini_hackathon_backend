# EHRX
### AI-Powered EHR Data Liberation

---

## Team

**Justin Jasper**
Stanford CS (AI) Master's Student; prior experience in Healthcare AI development

**Xander Hnasko**
Stanford CS (AI) Master's Student; prior experience in unstructured data ingestion within private financial markets. 

---

## The Problem

When patients switch hospitals, their medical records should follow them seamlessly, but they don't.

Data interoperability laws require "smooth" data exchange between EHR systems. In practice, hospitals export 600+ page PDF dumps—unstructured, unsearchable, and nearly useless. Clinicians spend hours manually hunting through these documents for critical patient information, risking errors and delays in care.

---

## The Solution

**EHRX transforms massive, unstructured EHR PDFs into queryable, structured data in minutes.**

Using Google's Gemini 2.5 vision-language models, EHRX:

- **Reads** 650+ page EHR documents with medical context understanding
- **Classifies** content into 19 semantic clinical categories (labs, medications, vitals, etc.)
- **Traces** every data point back to its exact source location for verification
- **Enables** natural language queries: *"What are the patient's current medications?"* → Instant, sourced answer

---

## How EHRX works

**Provenance-first AI extraction.**
Every piece of data EHRX surfaces is traceable to the exact page and pixel location in the original PDF. EHRX builds a semantic ontology of EHR data and _deterministically_ filters the records, answering questions aboutthem with an auditable trail, which is critical for healthcare compliance and clinician trust.

---

## Impact

- **Time:** Hours of manual review → seconds of natural language queries
- **Cost:** ~$1 to process 650 pages + 10 queries
- **Accuracy:** AI confidence scoring with automatic flags for human review
- **Compliance:** Full data lineage for HIPAA auditing and verification
---

## Built With

Google Gemini 2.5 Flash & Pro | Vertex AI | FastAPI | Cloud Run

---

*Unlocking healthcare data, one PDF at a time.*