export type ReadingStatus = "inbox" | "reading" | "summarized" | "archived";
export type InferenceMode = "heuristic" | "ollama" | "api";

export interface PaperSection {
  id: string;
  title: string;
  page: number;
  level: number;
}

export interface PaperReference {
  id: string;
  raw: string;
  year?: string;
  titleHint?: string;
}

export interface PaperChunk {
  id: string;
  paperId: string;
  text: string;
  page: number;
  sectionTitle?: string;
  embedding: number[];
  keywords: string[];
}

export interface PaperBrief {
  tldr: string;
  methods: string;
  innovations: string;
  experiments: string;
  limitations: string;
  reusableNotes: string[];
  groundedSections: string[];
  generatedAt: string;
  mode: InferenceMode;
}

export interface PaperNote {
  id: string;
  createdAt: string;
  content: string;
}

export interface ParsedPaper {
  id: string;
  title: string;
  authors: string[];
  abstract: string;
  year?: string;
  sourceFileName: string;
  storedPdfPath: string;
  importedAt: string;
  lastOpenedAt: string;
  researchArea: string;
  status: ReadingStatus;
  tags: string[];
  pageCount: number;
  wordCount: number;
  text: string;
  sections: PaperSection[];
  references: PaperReference[];
  chunks: PaperChunk[];
  brief: PaperBrief;
  notes: PaperNote[];
}

export interface QueryHit {
  chunkId: string;
  paperId: string;
  paperTitle: string;
  page: number;
  sectionTitle?: string;
  excerpt: string;
  score: number;
  keywordScore: number;
}

export interface QueryResponse {
  latencyMs: number;
  query: string;
  hits: QueryHit[];
  keywordBaselineTopScore: number;
  hybridTopScore: number;
}

export interface Citation {
  paperId: string;
  paperTitle: string;
  page: number;
  sectionTitle?: string;
  chunkId: string;
}

export interface AskResponse {
  question: string;
  answer: string;
  citations: Citation[];
  latencyMs: number;
  mode: "extractive" | "ollama" | "api";
  confidence: number;
}

export interface ProviderModelOption {
  id: string;
  ownedBy?: string;
}

export interface ApiProviderProfile {
  id: string;
  name: string;
  baseUrl: string;
  apiKey: string;
  type: "openai-compatible";
  defaultModel?: string;
  models: ProviderModelOption[];
  importedAt: string;
  lastSyncedAt?: string;
}

export interface ApiProviderDraft {
  name?: string;
  baseUrl?: string;
  apiKey?: string;
  defaultModel?: string;
  models?: string[];
}

export interface ModelRouting {
  summaryModel?: string;
  ragModel?: string;
  chatModel?: string;
}

export interface EvalMetrics {
  documentCount: number;
  indexedChunkCount: number;
  summaryGroundingRate: number;
  citationTraceabilityRate: number;
  qaHitRate: number;
  averageRetrievalLatencyMs: number;
  ragLiftVsKeyword: number;
}

export interface UserSettings {
  provider: InferenceMode;
  ollamaBaseUrl: string;
  ollamaModel: string;
  autoGenerateBriefs: boolean;
  apiProviders: ApiProviderProfile[];
  activeApiProviderId?: string;
  modelRouting: ModelRouting;
}

export interface LibraryState {
  papers: ParsedPaper[];
  selectedPaperId?: string;
  metrics: EvalMetrics;
  settings: UserSettings;
  availableTags: string[];
  availableResearchAreas: string[];
}

export interface PaperFilter {
  keyword?: string;
  status?: ReadingStatus | "all";
  tag?: string | "all";
}

export interface UpdatePaperInput {
  paperId: string;
  patch: Partial<
    Pick<ParsedPaper, "researchArea" | "status" | "tags" | "lastOpenedAt">
  >;
}

export interface NoteInput {
  paperId: string;
  content: string;
}

export interface QueryRequest {
  query: string;
  paperId?: string;
  topK?: number;
}

export interface AskRequest {
  question: string;
  paperId?: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: string;
  citations?: Citation[];
  model?: string;
}

export interface ChatRequest {
  messages: Array<Pick<ChatMessage, "role" | "content">>;
  paperId?: string;
  useRag?: boolean;
}

export interface ChatResponse {
  reply: ChatMessage;
  latencyMs: number;
  mode: InferenceMode;
  groundedBy: Citation[];
}

export interface ImportApiProviderInput {
  rawText?: string;
  name?: string;
  baseUrl?: string;
  apiKey?: string;
  defaultModel?: string;
}

export interface SettingsPatch {
  provider?: UserSettings["provider"];
  ollamaBaseUrl?: string;
  ollamaModel?: string;
  autoGenerateBriefs?: boolean;
  activeApiProviderId?: string;
  modelRouting?: Partial<ModelRouting>;
}
