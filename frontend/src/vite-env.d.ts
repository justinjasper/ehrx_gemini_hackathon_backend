/// <reference types="vite/client" />

declare module "pdfjs-dist/build/pdf.worker?url" {
  const workerUrl: string;
  export default workerUrl;
}

