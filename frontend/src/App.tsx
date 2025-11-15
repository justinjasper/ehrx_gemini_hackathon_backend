import { useEffect, useMemo, useState } from "react";
import "./App.css";
import {
  DocumentSummary,
  MatchedElement,
  OntologyDocument,
  QueryResponse
} from "./types";
import {
  fetchDocuments,
  fetchOntology,
  queryDocument,
  uploadDocument
} from "./api";
import UploadTab from "./components/UploadTab";
import OntologyTab from "./components/OntologyTab";
import QueryTab from "./components/QueryTab";

type TabKey = "upload" | "ontology" | "query";

function App() {
  const [activeTab, setActiveTab] = useState<TabKey>("upload");
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [selectedDocument, setSelectedDocument] = useState<string>("");
  const [ontology, setOntology] = useState<OntologyDocument | null>(null);
  const [loadingOntology, setLoadingOntology] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [queryLoading, setQueryLoading] = useState(false);
  const [queryResult, setQueryResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cachedPdfFile, setCachedPdfFile] = useState<File | null>(null);

  const loadDocuments = async () => {
    try {
      const response = await fetchDocuments();
      setDocuments(response.documents ?? []);
      if (!selectedDocument && response.documents?.length) {
        setSelectedDocument(response.documents[0].document_id);
      }
    } catch (err) {
      console.error(err);
      setError("Failed to fetch documents.");
    }
  };

  useEffect(() => {
    loadDocuments();
  }, []);

  useEffect(() => {
    if (!selectedDocument) {
      setOntology(null);
      return;
    }

    const loadOntology = async () => {
      setLoadingOntology(true);
      setError(null);
      try {
        const data = await fetchOntology(selectedDocument);
        setOntology(data);
      } catch (err) {
        console.error(err);
        setError("Failed to load ontology.");
        setOntology(null);
      } finally {
        setLoadingOntology(false);
      }
    };

    loadOntology();
  }, [selectedDocument]);

  const handleUpload = async (
    file: File,
    pageRange: string,
    documentType: string
  ) => {
    setUploading(true);
    setError(null);
    try {
      const response = await uploadDocument(file, pageRange, documentType);
      await loadDocuments();
      setSelectedDocument(response.document_id);
      setCachedPdfFile(file);
      setActiveTab("ontology");
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Upload failed.");
    } finally {
      setUploading(false);
    }
  };

  const handleQuery = async (question: string) => {
    if (!selectedDocument) return;
    setQueryLoading(true);
    setError(null);
    try {
      const response = await queryDocument(selectedDocument, question);
      setQueryResult(response);
      setActiveTab("query");
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Query failed.");
      setQueryResult(null);
    } finally {
      setQueryLoading(false);
    }
  };

  const pageInfoMap = useMemo(() => {
    const map = new Map<number, { width_px?: number; height_px?: number }>();
    ontology?.pages?.forEach((page) => {
      const pageNumber =
        page.page_number ?? page.pageNumber ?? page.page_num ?? 0;
      if (pageNumber && page.page_info) {
        map.set(pageNumber, {
          width_px: page.page_info.width_px,
          height_px: page.page_info.height_px
        });
      }
    });
    return map;
  }, [ontology]);

  return (
    <div className="app">
      <header className="app__header">
        <h1>EHRX Pipeline UI</h1>
        <p>Upload PDFs, explore ontology, and run natural language queries.</p>
      </header>

      <div className="tabs">
        {(["upload", "ontology", "query"] as TabKey[]).map((tab) => (
          <button
            key={tab}
            className={`tab ${activeTab === tab ? "tab--active" : ""}`}
            onClick={() => setActiveTab(tab)}
          >
            {tab.toUpperCase()}
          </button>
        ))}
      </div>

      {error && <div className="alert alert--error">{error}</div>}

      <section className="tab-content">
        {activeTab === "upload" && (
          <UploadTab onUpload={handleUpload} uploading={uploading} />
        )}

        {activeTab === "ontology" && (
          <OntologyTab
            documents={documents}
            selectedDocument={selectedDocument}
            onSelectDocument={(id) => setSelectedDocument(id)}
            ontology={ontology}
            loading={loadingOntology}
            onRefresh={loadDocuments}
          />
        )}

        {activeTab === "query" && (
          <QueryTab
            loading={queryLoading}
            onSubmit={handleQuery}
            queryResult={queryResult}
            documentId={selectedDocument}
            pdfFile={cachedPdfFile}
            pageInfoMap={pageInfoMap}
            ontologyAvailable={!!ontology}
          />
        )}
      </section>
    </div>
  );
}

export default App;

