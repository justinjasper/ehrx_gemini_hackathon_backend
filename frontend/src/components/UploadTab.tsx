import { useState } from "react";

interface UploadTabProps {
  onUpload: (file: File, pageRange: string, documentType: string) => void;
  uploading: boolean;
}

const UploadTab = ({ onUpload, uploading }: UploadTabProps) => {
  const [file, setFile] = useState<File | null>(null);
  const [pageRange, setPageRange] = useState("all");
  const [documentType, setDocumentType] = useState("Clinical EHR");

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!file) return;
    onUpload(file, pageRange, documentType);
  };

  return (
    <form className="grid" onSubmit={handleSubmit}>
      <div className="form-group">
        <label htmlFor="file">PDF File</label>
        <input
          id="file"
          type="file"
          accept="application/pdf"
          onChange={(event) =>
            setFile(event.target.files ? event.target.files[0] : null)
          }
          required
          disabled={uploading}
        />
        <small>Maximum size ~32 MB (Cloud Run request limit).</small>
      </div>

      <div className="grid grid--two-columns">
        <div className="form-group">
          <label htmlFor="pageRange">Page Range</label>
          <input
            id="pageRange"
            value={pageRange}
            onChange={(event) => setPageRange(event.target.value)}
            placeholder='Example: "all" or "1-10"'
            disabled={uploading}
          />
        </div>
        <div className="form-group">
          <label htmlFor="documentType">Document Type</label>
          <input
            id="documentType"
            value={documentType}
            onChange={(event) => setDocumentType(event.target.value)}
            disabled={uploading}
          />
        </div>
      </div>

      <button
          className="btn"
          type="submit"
          disabled={!file || uploading}
        >
          {uploading ? "Uploading…" : "Upload & Process"}
        </button>

      {!file && (
        <div className="card">
          <strong>Tip:</strong> After upload completes, switch to the Ontology
          tab to explore results, then the Query tab for QA.
        </div>
      )}
    </form>
  );
};

export default UploadTab;

