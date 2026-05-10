declare module "pdf-parse" {
  interface PdfInfo {
    Title?: string;
    Author?: string;
    CreationDate?: string;
  }

  interface PdfParseResult {
    numpages: number;
    info?: PdfInfo;
    text: string;
  }

  interface PdfParseOptions {
    pagerender?: (pageData: unknown) => Promise<string>;
  }

  export default function pdf(
    dataBuffer: Buffer,
    options?: PdfParseOptions,
  ): Promise<PdfParseResult>;
}
