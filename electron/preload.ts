import { contextBridge, ipcRenderer } from "electron";
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
} from "../src/shared/contracts";

const api = {
  getState: (filter?: PaperFilter) =>
    ipcRenderer.invoke("library:get-state", filter) as Promise<LibraryState>,
  importPdfs: () =>
    ipcRenderer.invoke("library:import-pdfs") as Promise<LibraryState>,
  updatePaper: (input: UpdatePaperInput) =>
    ipcRenderer.invoke("library:update-paper", input) as Promise<LibraryState>,
  addNote: (input: NoteInput) =>
    ipcRenderer.invoke("library:add-note", input) as Promise<LibraryState>,
  removePaper: (paperId: string) =>
    ipcRenderer.invoke("library:remove-paper", paperId) as Promise<LibraryState>,
  search: (request: QueryRequest) =>
    ipcRenderer.invoke("library:search", request) as Promise<QueryResponse>,
  ask: (request: AskRequest) =>
    ipcRenderer.invoke("library:ask", request) as Promise<AskResponse>,
  chat: (request: ChatRequest) =>
    ipcRenderer.invoke("library:chat", request) as Promise<ChatResponse>,
  regenerateBrief: (paperId: string) =>
    ipcRenderer.invoke("library:regenerate-brief", paperId) as Promise<LibraryState>,
  updateSettings: (patch: SettingsPatch) =>
    ipcRenderer.invoke("library:update-settings", patch) as Promise<LibraryState>,
  parseApiProviderDraft: (rawText: string) =>
    ipcRenderer.invoke("library:parse-api-provider-draft", rawText) as Promise<ApiProviderDraft>,
  importApiProvider: (input: ImportApiProviderInput) =>
    ipcRenderer.invoke("library:import-api-provider", input) as Promise<LibraryState>,
  refreshApiProviderModels: (providerId: string) =>
    ipcRenderer.invoke("library:refresh-api-provider-models", providerId) as Promise<LibraryState>,
  removeApiProvider: (providerId: string) =>
    ipcRenderer.invoke("library:remove-api-provider", providerId) as Promise<LibraryState>,
  openPdf: (paperId: string) =>
    ipcRenderer.invoke("library:open-pdf", paperId) as Promise<LibraryState>,
  revealPdfInFolder: (paperId: string) =>
    ipcRenderer.invoke("library:reveal-pdf", paperId) as Promise<boolean>,
};

contextBridge.exposeInMainWorld("brieflyApi", api);
