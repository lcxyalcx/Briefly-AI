/// <reference types="vite/client" />

import type {
  ApiProviderDraft,
  AskRequest,
  AskResponse,
  ChatRequest,
  ChatResponse,
  ImportApiProviderInput,
  LibraryState,
  NoteInput,
  PaperFilter,
  QueryRequest,
  QueryResponse,
  SettingsPatch,
  UpdatePaperInput,
} from "./shared/contracts";

declare global {
  interface Window {
    brieflyApi: {
      getState: (filter?: PaperFilter) => Promise<LibraryState>;
      importPdfs: () => Promise<LibraryState>;
      updatePaper: (input: UpdatePaperInput) => Promise<LibraryState>;
      addNote: (input: NoteInput) => Promise<LibraryState>;
      search: (request: QueryRequest) => Promise<QueryResponse>;
      ask: (request: AskRequest) => Promise<AskResponse>;
      chat: (request: ChatRequest) => Promise<ChatResponse>;
      regenerateBrief: (paperId: string) => Promise<LibraryState>;
      updateSettings: (patch: SettingsPatch) => Promise<LibraryState>;
      parseApiProviderDraft: (rawText: string) => Promise<ApiProviderDraft>;
      importApiProvider: (input: ImportApiProviderInput) => Promise<LibraryState>;
      refreshApiProviderModels: (providerId: string) => Promise<LibraryState>;
      removeApiProvider: (providerId: string) => Promise<LibraryState>;
      openPdf: (paperId: string) => Promise<boolean>;
    };
  }
}

export {};
