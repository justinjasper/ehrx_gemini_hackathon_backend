import { useState } from "react";
import { MatchedElement, QueryResponse } from "../types";
import MatchedElementsList from "./MatchedElementsList";
import PDFViewer from "./PDFViewer";

interface QueryTabProps {
  loading: boolean;
  onSubmit: (question: string) => Promise<void> | void;
  queryResult: QueryResponse | null;
  documentId: string;
  pdfFile: File | null;
  pageInfoMap: Map<number, { width_px?: number; height_px?: number }>;
  ontologyAvailable: boolean;
}

const QueryTab = ({
  loading,
  onSubmit,
  queryResult,
  documentId,
  pdfFile,
  pageInfoMap,
  ontologyAvailable
}: QueryTabProps) => {
  const [question, setQuestion] = useState("");
  const [highlightedElementId, setHighlightedElementId] = useState<
    string | null
  >(null);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!question.trim() || !documentId) return;
    await onSubmit(question.trim());
    setHighlightedElementId(null);
  };

  const matchedElements: MatchedElement[] =
    queryResult?.matched_elements ?? [];

  return (
    <div className="grid">
      <div className="card">
        <form className="grid" onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="question">Question</label>
            <textarea
              id="question"
              placeholder="Example: What medications is the patient taking?"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              disabled={loading || !documentId}
            />
          </div>
          <button className="btn" type="submit" disabled={loading || !question}>
            {loading ? "Running Query…" : "Submit Query"}
          </button>
        </form>
      </div>

      {!documentId && (
        <div className="card">
          Select or upload a document before running queries.
        </div>
      )}

      {queryResult && (
        <div className="grid">
          <div className="card">
            <h3>Answer Summary</h3>
            <p>{queryResult.answer_summary || "No summary provided."}</p>
          </div>

          {queryResult.reasoning && (
            <div className="card">
              <h4>Reasoning</h4>
              <p>{queryResult.reasoning}</p>
            </div>
          )}

          <MatchedElementsList
            elements={matchedElements}
            highlightedElementId={highlightedElementId}
            onHighlight={(elementId, pageNumber) => {
              setHighlightedElementId(elementId);
            }}
          />
        </div>
      )}

      {ontologyAvailable ? (
        <PDFViewer
          file={pdfFile}
          matchedElements={matchedElements}
          highlightedElementId={highlightedElementId}
          onHighlightChange={(elementId) => setHighlightedElementId(elementId)}
          pageInfoMap={pageInfoMap}
        />
      ) : (
        <div className="card">
          PDF preview requires ontology data. Upload a document or switch to the
          Ontology tab first.
        </div>
      )}

      {!pdfFile && (
        <div className="card">
          <strong>Note:</strong> Bounding boxes require the PDF uploaded during
          this session. PDFs from previous sessions are not cached client-side.
        </div>
      )}
    </div>
  );
};

export default QueryTab;

