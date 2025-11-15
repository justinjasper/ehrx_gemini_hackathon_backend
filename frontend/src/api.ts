import {
  DocumentSummary,
  OntologyDocument,
  QueryResponse,
  UploadResponse
} from "./types";

const API_BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(
    /\/$/,
    ""
  ) || "";

const withBase = (path: string) => `${API_BASE_URL}${path}`;

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const message = await response.text();
    throw new Error(
      message || `Request failed with status ${response.status}`
    );
  }
  return response.json() as Promise<T>;
}

export async function uploadDocument(
  file: File,
  pageRange: string,
  documentType: string
): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("page_range", pageRange);
  formData.append("document_type", documentType);

  const response = await fetch(withBase("/upload"), {
    method: "POST",
    body: formData
  });
  return handleResponse<UploadResponse>(response);
}

export async function fetchDocuments(): Promise<{
  documents: DocumentSummary[];
}> {
  const response = await fetch(withBase("/documents"));
  return handleResponse<{ documents: DocumentSummary[] }>(response);
}

export async function fetchOntology(
  documentId: string
): Promise<OntologyDocument> {
  const response = await fetch(
    withBase(`/documents/${encodeURIComponent(documentId)}/ontology`)
  );
  return handleResponse<OntologyDocument>(response);
}

export async function queryDocument(
  documentId: string,
  question: string
): Promise<QueryResponse> {
  const response = await fetch(
    withBase(`/documents/${encodeURIComponent(documentId)}/query`),
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ question })
    }
  );
  return handleResponse<QueryResponse>(response);
}

