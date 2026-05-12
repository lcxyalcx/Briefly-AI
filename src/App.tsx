import {
  startTransition,
  useDeferredValue,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type { KeyboardEvent as ReactKeyboardEvent, ReactNode } from "react";
import type {
  ApiProviderDraft,
  ApiProviderProfile,
  AskResponse,
  ChatMessage,
  LibraryState,
  ParsedPaper,
  QueryResponse,
  ReadingStatus,
  UserSettings,
  InferenceMode,
} from "./shared/contracts";

type WorkspaceTab =
  | "overview"
  | "ai"
  | "search"
  | "qa"
  | "chat"
  | "notes"
  | "settings"
  | "metrics";

type ScopeMode = "paper" | "library";
type FeedbackTone = "neutral" | "success" | "danger";

const CHAT_PENDING_MESSAGE_ID = "__briefly-chat-pending__";

interface ProviderPreset {
  label: string;
  name: string;
  baseUrl: string;
  helper: string;
}

interface ProviderFormState {
  name: string;
  baseUrl: string;
  apiKey: string;
  defaultModel: string;
  rawText: string;
}

const READING_STATUS_ORDER = [
  "inbox",
  "reading",
  "summarized",
  "archived",
] as const satisfies readonly ReadingStatus[];

const READING_STATUS_META: Record<ReadingStatus, { label: string; hint: string }> =
  {
    inbox: {
      label: "未读",
      hint: "尚未开始阅读；导入后默认为未读",
    },
    reading: {
      label: "阅读中",
      hint: "正在通读或使用 Insight / 检索辅助阅读",
    },
    summarized: {
      label: "已精读",
      hint: "已完成结构化提炼（如 TL;DR、方法拆解），可作写作素材",
    },
    archived: {
      label: "已归档",
      hint: "近期不再跟进，可随时改回其它状态",
    },
  };

const STATUS_LABELS: Record<ReadingStatus, string> = {
  inbox: READING_STATUS_META.inbox.label,
  reading: READING_STATUS_META.reading.label,
  summarized: READING_STATUS_META.summarized.label,
  archived: READING_STATUS_META.archived.label,
};

const MODE_LABELS: Record<InferenceMode, string> = {
  heuristic: "本地轻量模式",
  ollama: "Ollama 本地模型",
  api: "API Provider",
};

const PROVIDER_PRESETS: ProviderPreset[] = [
  {
    label: "SiliconFlow",
    name: "SiliconFlow",
    baseUrl: "https://api.siliconflow.cn/v1",
    helper: "硅基流动 · OpenAI-compatible（默认优先）",
  },
  {
    label: "OpenAI",
    name: "OpenAI",
    baseUrl: "https://api.openai.com/v1",
    helper: "官方 OpenAI 兼容接口",
  },
  {
    label: "OpenRouter",
    name: "OpenRouter",
    baseUrl: "https://openrouter.ai/api/v1",
    helper: "适合多模型选择",
  },
  {
    label: "DeepSeek",
    name: "DeepSeek",
    baseUrl: "https://api.deepseek.com/v1",
    helper: "适合中文科研问答",
  },
  {
    label: "Moonshot",
    name: "Moonshot",
    baseUrl: "https://api.moonshot.cn/v1",
    helper: "长上下文模型可选",
  },
];

function formatDate(value: string) {
  return new Intl.DateTimeFormat("zh-CN", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function compactNumber(value: number) {
  return new Intl.NumberFormat("zh-CN", {
    maximumFractionDigits: value >= 100 ? 0 : 1,
  }).format(value);
}

function maskApiKey(value: string) {
  if (!value) {
    return "未配置";
  }

  if (value.length <= 10) {
    return `${value.slice(0, 3)}***`;
  }

  return `${value.slice(0, 4)}***${value.slice(-4)}`;
}

function getProviderById(settings: UserSettings | null) {
  if (!settings) {
    return null;
  }

  return (
    settings.apiProviders.find(
      (provider) => provider.id === settings.activeApiProviderId,
    ) ?? settings.apiProviders[0] ?? null
  );
}

function getModelLabel(
  settings: UserSettings | null,
  slot: "summaryModel" | "ragModel" | "chatModel",
) {
  if (!settings) {
    return "未选择";
  }

  if (settings.provider === "ollama") {
    return settings.ollamaModel || "未配置 Ollama 模型";
  }

  if (settings.provider === "api") {
    const provider = getProviderById(settings);
    return (
      settings.modelRouting[slot] ??
      provider?.defaultModel ??
      provider?.models[0]?.id ??
      "未选择 API 模型"
    );
  }

  return slot === "chatModel" ? "自由对话未启用模型" : "Heuristic";
}

function buildProviderRoutingDraft(
  settings: UserSettings,
  provider: ApiProviderProfile | null,
) {
  const fallback = provider?.defaultModel ?? provider?.models[0]?.id;

  return {
    summaryModel: settings.modelRouting.summaryModel ?? fallback,
    ragModel: settings.modelRouting.ragModel ?? fallback,
    chatModel: settings.modelRouting.chatModel ?? fallback,
  };
}

function createEmptyProviderForm(): ProviderFormState {
  return {
    name: "",
    baseUrl: "",
    apiKey: "",
    defaultModel: "",
    rawText: "",
  };
}

function providerDraftToForm(draft: ApiProviderDraft): Partial<ProviderFormState> {
  return {
    name: draft.name ?? "",
    baseUrl: draft.baseUrl ?? "",
    apiKey: draft.apiKey ?? "",
    defaultModel: draft.defaultModel ?? draft.models?.[0] ?? "",
  };
}

function describeError(error: unknown) {
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return "发生了一个未预期的错误。";
}

function buildScopePaperId(
  scope: ScopeMode,
  selectedPaper: ParsedPaper | undefined,
) {
  if (scope === "paper") {
    return selectedPaper?.id;
  }
  return undefined;
}

function StatusBadge({ status }: { status: ReadingStatus }) {
  return <span className={`status-badge status-${status}`}>{STATUS_LABELS[status]}</span>;
}

function ChatTypingIndicator({ useRag }: { useRag: boolean }) {
  return (
    <div className="chat-typing" aria-live="polite" aria-busy="true">
      <span className="chat-typing-dots" aria-hidden="true">
        <span />
        <span />
        <span />
      </span>
      <span className="chat-typing-label">
        {useRag ? "正在检索相关片段并生成回复…" : "正在生成回复…"}
      </span>
    </div>
  );
}

function ToggleSwitchField({
  id,
  title,
  description,
  checked,
  disabled,
  onChange,
  className,
}: {
  id: string;
  title: string;
  description?: string;
  checked: boolean;
  disabled?: boolean;
  onChange: (next: boolean) => void;
  className?: string;
}) {
  const rootClass = [
    "toggle-field",
    disabled ? "toggle-field-disabled" : "",
    className ?? "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <label className={rootClass} htmlFor={id}>
      <span className="toggle-field-body">
        <span className="toggle-field-title">{title}</span>
        {description ? <span className="toggle-field-desc">{description}</span> : null}
      </span>
      <span className="toggle-switch">
        <input
          id={id}
          type="checkbox"
          role="switch"
          className="toggle-switch-input"
          checked={checked}
          disabled={disabled}
          aria-checked={checked}
          onChange={(event) => onChange(event.target.checked)}
        />
        <span className="toggle-switch-track" aria-hidden />
      </span>
    </label>
  );
}

function MetricTile({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint: string;
}) {
  return (
    <div className="metric-tile">
      <span className="metric-label">{label}</span>
      <strong className="metric-value">{value}</strong>
      <span className="metric-hint">{hint}</span>
    </div>
  );
}

function SectionCard({
  title,
  body,
  hint,
}: {
  title: string;
  body: string;
  hint?: string;
}) {
  return (
    <article className="panel-card">
      <div className="section-header">
        <h3>{title}</h3>
        {hint ? <span>{hint}</span> : null}
      </div>
      <p>{body || "等待生成内容。"}</p>
    </article>
  );
}

function InsightCard({
  title,
  eyebrow,
  children,
}: {
  title: string;
  eyebrow?: string;
  children: ReactNode;
}) {
  return (
    <article className="insight-card">
      <div className="insight-head">
        <div>
          {eyebrow ? <span className="eyebrow">{eyebrow}</span> : null}
          <h3>{title}</h3>
        </div>
      </div>
      <div className="insight-body">{children}</div>
    </article>
  );
}

function PromptChip({
  label,
  onClick,
}: {
  label: string;
  onClick: () => void;
}) {
  return (
    <button type="button" className="prompt-chip" onClick={onClick}>
      {label}
    </button>
  );
}

const UI_PREFS_STORAGE_KEY = "briefly.ai.ui";

interface UiPrefs {
  groundingScope?: ScopeMode;
  chatUseRag?: boolean;
  customBackground?: string;
}

function safeReadPrefs(): UiPrefs | null {
  try {
    const raw = localStorage.getItem(UI_PREFS_STORAGE_KEY);
    if (!raw) {
      return null;
    }
    return JSON.parse(raw) as UiPrefs;
  } catch {
    return null;
  }
}

function saveUiPrefs(prefs: UiPrefs) {
  try {
    const merged = {
      ...(safeReadPrefs() ?? {}),
      ...prefs,
    };
    localStorage.setItem(UI_PREFS_STORAGE_KEY, JSON.stringify(merged));
  } catch {
    /* ignore quota / privacy mode */
  }
}

const PAPER_LIST_LAYOUT_KEY = "briefly.ai.paperListLayout";
const RESIZER_GAP = 4;

interface PaperListLayoutPrefs {
  colArea: number;
  colStatus: number;
  colRetrieval: number;
  libraryWidth: number;
}

const DEFAULT_PAPER_LIST_LAYOUT: PaperListLayoutPrefs = {
  colArea: 118,
  colStatus: 88,
  colRetrieval: 58,
  libraryWidth: 400,
};

const COL_CLAMP = {
  area: [72, 260] as const,
  status: [68, 180] as const,
  retr: [48, 130] as const,
  library: [300, 640] as const,
};

function clampNum(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function readPaperListLayout(): PaperListLayoutPrefs {
  try {
    const raw = localStorage.getItem(PAPER_LIST_LAYOUT_KEY);
    if (!raw) {
      return { ...DEFAULT_PAPER_LIST_LAYOUT };
    }
    const p = JSON.parse(raw) as Partial<PaperListLayoutPrefs>;
    return {
      colArea: clampNum(
        Number(p.colArea) || DEFAULT_PAPER_LIST_LAYOUT.colArea,
        COL_CLAMP.area[0],
        COL_CLAMP.area[1],
      ),
      colStatus: clampNum(
        Number(p.colStatus) || DEFAULT_PAPER_LIST_LAYOUT.colStatus,
        COL_CLAMP.status[0],
        COL_CLAMP.status[1],
      ),
      colRetrieval: clampNum(
        Number(p.colRetrieval) || DEFAULT_PAPER_LIST_LAYOUT.colRetrieval,
        COL_CLAMP.retr[0],
        COL_CLAMP.retr[1],
      ),
      libraryWidth: clampNum(
        Number(p.libraryWidth) || DEFAULT_PAPER_LIST_LAYOUT.libraryWidth,
        COL_CLAMP.library[0],
        COL_CLAMP.library[1],
      ),
    };
  } catch {
    return { ...DEFAULT_PAPER_LIST_LAYOUT };
  }
}

function savePaperListLayout(layout: PaperListLayoutPrefs) {
  try {
    localStorage.setItem(PAPER_LIST_LAYOUT_KEY, JSON.stringify(layout));
  } catch {
    /* ignore */
  }
}

function redistributePair(
  snapL: number,
  snapR: number,
  delta: number,
  limL: readonly [number, number],
  limR: readonly [number, number],
): [number, number] {
  let nextL = clampNum(snapL + delta, limL[0], limL[1]);
  let nextR = snapR + snapL - nextL;
  if (nextR < limR[0]) {
    nextR = limR[0];
    nextL = clampNum(snapL + snapR - nextR, limL[0], limL[1]);
  } else if (nextR > limR[1]) {
    nextR = limR[1];
    nextL = clampNum(snapL + snapR - nextR, limL[0], limL[1]);
  }
  return [nextL, nextR];
}

function useWidePaperTable() {
  const query = `(min-width: 1241px) and (pointer: fine)`;
  const [ok, setOk] = useState(
    typeof window !== "undefined" ? window.matchMedia(query).matches : false,
  );
  useEffect(() => {
    const mq = window.matchMedia(query);
    const update = () => setOk(mq.matches);
    update();
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, []);
  return ok;
}

function buildPaperGridColumnsFull(layout: PaperListLayoutPrefs) {
  const { colArea: a, colStatus: s, colRetrieval: r } = layout;
  const g = RESIZER_GAP;
  return `minmax(0, 1fr) ${g}px ${a}px ${g}px ${s}px ${g}px ${r}px`;
}

function buildPaperGridColumnsCompact(layout: PaperListLayoutPrefs) {
  const { colArea: a, colStatus: s, colRetrieval: r } = layout;
  return `minmax(0, 1fr) ${a}px ${s}px ${r}px`;
}

function trySubmitMultilineArea(
  event: ReactKeyboardEvent<HTMLTextAreaElement>,
  busy: boolean,
  onSubmit: () => void | Promise<void>,
) {
  if (event.key !== "Enter") {
    return;
  }
  if (event.shiftKey || event.altKey) {
    return;
  }
  if (event.nativeEvent.isComposing) {
    return;
  }
  event.preventDefault();
  if (busy) {
    return;
  }
  void onSubmit();
}

function App() {
  const [library, setLibrary] = useState<LibraryState | null>(null);
  const [selectedPaperId, setSelectedPaperId] = useState<string>();
  const [activeTab, setActiveTab] = useState<WorkspaceTab>("overview");
  const [filterKeyword, setFilterKeyword] = useState("");
  const [statusFilter, setStatusFilter] = useState<ReadingStatus | "all">("all");
  const [tagFilter, setTagFilter] = useState<string | "all">("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResult, setSearchResult] = useState<QueryResponse | null>(null);
  const [askQuestion, setAskQuestion] = useState("");
  const [askResult, setAskResult] = useState<AskResponse | null>(null);
  const [noteDraft, setNoteDraft] = useState("");
  const [tagDraft, setTagDraft] = useState("");
  const [chatDraft, setChatDraft] = useState("");
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatUseRag, setChatUseRag] = useState(() => {
    const prefs = safeReadPrefs();
    return typeof prefs?.chatUseRag === "boolean" ? prefs.chatUseRag : true;
  });
  const [customBackground, setCustomBackground] = useState(() => {
    const prefs = safeReadPrefs();
    return prefs?.customBackground?.trim() ?? "";
  });
  const [groundingScope, setGroundingScope] = useState<ScopeMode>(() => {
    const prefs = safeReadPrefs();
    return prefs?.groundingScope === "library" ? "library" : "paper";
  });
  const [settingsDraft, setSettingsDraft] = useState<UserSettings | null>(null);
  const [providerForm, setProviderForm] = useState<ProviderFormState>(createEmptyProviderForm());
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [openingPdfId, setOpeningPdfId] = useState<string | null>(null);
  const [paperListLayout, setPaperListLayout] = useState<PaperListLayoutPrefs>(() =>
    readPaperListLayout(),
  );
  const [banner, setBanner] = useState("导入 PDF；悬停列表项可打开或删除。");
  const [bannerTone, setBannerTone] = useState<FeedbackTone>("neutral");
  const deferredKeyword = useDeferredValue(filterKeyword);
  const chatStreamRef = useRef<HTMLDivElement | null>(null);
  const widePaperTable = useWidePaperTable();

  function requireApi() {
    const api = window.brieflyApi;
    if (!api) {
      throw new Error("当前运行在浏览器预览模式，请通过 Electron 启动 Briefly AI 桌面应用。");
    }
    return api;
  }

  useEffect(() => {
    void (async () => {
      setBusyAction("loading");
      try {
        if (!window.brieflyApi) {
          setBanner("当前是浏览器预览模式。桌面 API 未注入，所以这里主要用于看布局，不用于真实导入和问答。");
          setBannerTone("neutral");
          return;
        }

        const next = await window.brieflyApi.getState();
        startTransition(() => {
          setLibrary(next);
          setSettingsDraft(next.settings);
          setSelectedPaperId(next.selectedPaperId);
        });
      } catch (error) {
        setBanner(describeError(error));
        setBannerTone("danger");
      } finally {
        setBusyAction(null);
      }
    })();
  }, []);

  useEffect(() => {
    if (!library) {
      return;
    }

    const exists = library.papers.some((paper) => paper.id === selectedPaperId);
    if (!exists) {
      setSelectedPaperId(library.papers[0]?.id);
    }
  }, [library, selectedPaperId]);

  useEffect(() => {
    setTagDraft("");
  }, [selectedPaperId]);

  useLayoutEffect(() => {
    if (!chatStreamRef.current) {
      return;
    }
    chatStreamRef.current.scrollTop = chatStreamRef.current.scrollHeight;
  }, [chatMessages]);

  useEffect(() => {
    saveUiPrefs({ groundingScope, chatUseRag, customBackground });
  }, [groundingScope, chatUseRag, customBackground]);

  useEffect(() => {
    const bg = customBackground.trim();
    if (!bg) {
      document.documentElement.style.removeProperty("--app-bg");
      return;
    }
    document.documentElement.style.setProperty("--app-bg", bg);
  }, [customBackground]);

  useEffect(() => {
    savePaperListLayout(paperListLayout);
  }, [paperListLayout]);

  const paperGridColumns = useMemo(
    () =>
      widePaperTable
        ? buildPaperGridColumnsFull(paperListLayout)
        : buildPaperGridColumnsCompact(paperListLayout),
    [widePaperTable, paperListLayout],
  );

  const filteredPapers =
    library?.papers.filter((paper) => {
      const matchesKeyword = deferredKeyword
        ? `${paper.title} ${paper.abstract} ${paper.tags.join(" ")} ${paper.authors.join(" ")}`
            .toLowerCase()
            .includes(deferredKeyword.toLowerCase())
        : true;
      const matchesStatus =
        statusFilter === "all" ? true : paper.status === statusFilter;
      const matchesTag = tagFilter === "all" ? true : paper.tags.includes(tagFilter);
      return matchesKeyword && matchesStatus && matchesTag;
    }) ?? [];

  const selectedPaper =
    filteredPapers.find((paper) => paper.id === selectedPaperId) ??
    library?.papers.find((paper) => paper.id === selectedPaperId) ??
    filteredPapers[0];

  const activeApiProvider = getProviderById(settingsDraft);
  const activeApiModels = activeApiProvider?.models ?? [];
  const latestAssistantMessage =
    [...chatMessages]
      .reverse()
      .find(
        (message) =>
          message.role === "assistant" && message.id !== CHAT_PENDING_MESSAGE_ID,
      ) ?? null;
  const scopedPaperId = buildScopePaperId(groundingScope, selectedPaper);

  const electronAvailable = Boolean(window.brieflyApi);
  const libraryIsEmpty = Boolean(library && library.papers.length === 0);
  const blockImport =
    !electronAvailable || busyAction === "loading" || busyAction === "import";

  function startLibrarySplitDrag(event: React.MouseEvent) {
    event.preventDefault();
    const snapshot = { ...paperListLayout };
    const startX = event.clientX;
    const onMove = (ev: MouseEvent) => {
      const dx = ev.clientX - startX;
      setPaperListLayout((piece) => ({
        ...piece,
        libraryWidth: clampNum(
          snapshot.libraryWidth + dx,
          COL_CLAMP.library[0],
          COL_CLAMP.library[1],
        ),
      }));
    };
    const onUp = () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      document.body.style.removeProperty("cursor");
      document.body.style.removeProperty("user-select");
    };
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }

  function startPaperColDrag(
    kind: "areaFromTitle" | "areaStatus" | "statusRetr",
    event: React.MouseEvent,
  ) {
    event.preventDefault();
    event.stopPropagation();
    const snapshot = { ...paperListLayout };
    const startX = event.clientX;
    const onMove = (ev: MouseEvent) => {
      const dx = ev.clientX - startX;
      setPaperListLayout((piece) => {
        if (kind === "areaFromTitle") {
          return {
            ...piece,
            colArea: clampNum(snapshot.colArea + dx, COL_CLAMP.area[0], COL_CLAMP.area[1]),
          };
        }
        if (kind === "areaStatus") {
          const [colArea, colStatus] = redistributePair(
            snapshot.colArea,
            snapshot.colStatus,
            dx,
            COL_CLAMP.area,
            COL_CLAMP.status,
          );
          return { ...piece, colArea, colStatus };
        }
        const [colStatus, colRetrieval] = redistributePair(
          snapshot.colStatus,
          snapshot.colRetrieval,
          dx,
          COL_CLAMP.status,
          COL_CLAMP.retr,
        );
        return { ...piece, colStatus, colRetrieval };
      });
    };
    const onUp = () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      document.body.style.removeProperty("cursor");
      document.body.style.removeProperty("user-select");
    };
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }

  function navigatePaperSelection(delta: number) {
    if (filteredPapers.length === 0) {
      return;
    }

    let index = filteredPapers.findIndex((paper) => paper.id === selectedPaperId);
    if (index === -1) {
      index = delta > 0 ? -1 : 0;
    }

    const nextIndex = Math.min(
      filteredPapers.length - 1,
      Math.max(0, index + delta),
    );
    const nextPaper = filteredPapers[nextIndex];
    if (!nextPaper) {
      return;
    }

    setSelectedPaperId(nextPaper.id);
    setSearchResult(null);
    setAskResult(null);
  }

  function handlePaperListKeyDown(event: ReactKeyboardEvent<HTMLElement>) {
    const narrowTarget = event.target;
    if (narrowTarget instanceof HTMLInputElement || narrowTarget instanceof HTMLTextAreaElement) {
      return;
    }
    if (narrowTarget instanceof HTMLSelectElement) {
      return;
    }

    if (event.altKey || event.metaKey || event.ctrlKey) {
      return;
    }
    if (event.key === "ArrowDown") {
      event.preventDefault();
      navigatePaperSelection(1);
      return;
    }
    if (event.key === "ArrowUp") {
      event.preventDefault();
      navigatePaperSelection(-1);
    }
  }

  function handleLibraryFilterKeyDown(event: ReactKeyboardEvent<HTMLInputElement>) {
    if (event.key !== "Escape") {
      return;
    }
    event.preventDefault();
    setFilterKeyword("");
  }

  function handleSearchKeyDown(event: ReactKeyboardEvent<HTMLTextAreaElement>) {
    trySubmitMultilineArea(event, busyAction === "search", handleSearch);
  }

  function handleAskKeyDown(event: ReactKeyboardEvent<HTMLTextAreaElement>) {
    trySubmitMultilineArea(event, busyAction === "ask", handleAsk);
  }

  function handleChatKeyDown(event: ReactKeyboardEvent<HTMLTextAreaElement>) {
    trySubmitMultilineArea(event, busyAction === "chat", handleChatSend);
  }

  function setFeedback(message: string, tone: FeedbackTone) {
    setBanner(message);
    setBannerTone(tone);
  }

  async function syncState(promise: Promise<LibraryState>, message: string) {
    const next = await promise;
    startTransition(() => {
      setLibrary(next);
      setSettingsDraft(next.settings);
      setSelectedPaperId((current) => current ?? next.selectedPaperId ?? next.papers[0]?.id);
    });
    setFeedback(message, "success");
  }

  function updateSettingsDraft(patch: Partial<UserSettings>) {
    if (!settingsDraft) {
      return;
    }

    setSettingsDraft({
      ...settingsDraft,
      ...patch,
      modelRouting: {
        ...settingsDraft.modelRouting,
        ...patch.modelRouting,
      },
    });
  }

  function updateProviderForm(patch: Partial<ProviderFormState>) {
    setProviderForm((current) => ({
      ...current,
      ...patch,
    }));
  }

  function fillProviderPreset(preset: ProviderPreset) {
    updateProviderForm({
      name: preset.name,
      baseUrl: preset.baseUrl,
    });
    setFeedback(`已填入 ${preset.label} 模板，现在只需要补 API Key 和模型名。`, "neutral");
  }

  function applyParsedProviderDraft(draft: ApiProviderDraft) {
    updateProviderForm(providerDraftToForm(draft));
  }

  async function handleImport() {
    setBusyAction("import");
    try {
      await syncState(
        requireApi().importPdfs(),
        "PDF 已导入。你现在可以在中间列表选文献，在右侧 Insight Card 直接进入检索、问答和聊天。",
      );
    } catch (error) {
      setFeedback(`导入失败：${describeError(error)}`, "danger");
    } finally {
      setBusyAction(null);
    }
  }

  async function patchPaper(
    paperId: string,
    patch: Partial<Pick<ParsedPaper, "researchArea" | "status" | "tags">>,
    message = "文献信息已更新。",
  ) {
    setBusyAction("patch");
    try {
      await syncState(requireApi().updatePaper({ paperId, patch }), message);
    } catch (error) {
      setFeedback(`更新失败：${describeError(error)}`, "danger");
    } finally {
      setBusyAction(null);
    }
  }

  async function commitTagDraft() {
    if (!selectedPaper) {
      return;
    }
    const nextTag = tagDraft.trim();
    if (!nextTag) {
      return;
    }
    const nextTags = [...selectedPaper.tags, nextTag].filter(
      (value, index, list) => value && list.indexOf(value) === index,
    );
    setTagDraft("");
    await patchPaper(selectedPaper.id, { tags: nextTags }, "标签已更新。");
  }

  async function removeTag(tag: string) {
    if (!selectedPaper) {
      return;
    }
    const nextTags = selectedPaper.tags.filter((item) => item !== tag);
    await patchPaper(selectedPaper.id, { tags: nextTags }, "标签已更新。");
  }

  async function handleRegenerateBrief(paperId: string) {
    setBusyAction("brief");
    try {
      await syncState(
        requireApi().regenerateBrief(paperId),
        "Insight Card 已更新，新的阅读摘要和方法提炼已经生成。",
      );
    } catch (error) {
      setFeedback(`摘要生成失败：${describeError(error)}`, "danger");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleRemovePaper(paperId: string, title: string) {
    const confirmed = window.confirm(`确认删除《${title}》吗？这会同时移除本地复制的 PDF。`);
    if (!confirmed) {
      return;
    }

    setBusyAction("paper-remove");
    try {
      await syncState(
        requireApi().removePaper(paperId),
        "文档已从 Briefly AI 文献库中删除，对应的本地 PDF 副本也已清理。",
      );
    } catch (error) {
      setFeedback(`删除文档失败：${describeError(error)}`, "danger");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleAddNote() {
    if (!selectedPaper || !noteDraft.trim()) {
      setFeedback("先选择一篇论文，再写一条想复用的笔记。", "neutral");
      return;
    }

    setBusyAction("note");
    try {
      await syncState(
        requireApi().addNote({
          paperId: selectedPaper.id,
          content: noteDraft.trim(),
        }),
        "笔记已保存到这篇论文下，后续写综述和做 related work 会更顺手。",
      );
      setNoteDraft("");
    } catch (error) {
      setFeedback(`保存笔记失败：${describeError(error)}`, "danger");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleSearch() {
    if (!searchQuery.trim()) {
      setFeedback("先输入一个你想检索的研究问题或术语。", "neutral");
      return;
    }

    setBusyAction("search");
    try {
      const result = await requireApi().search({
        query: searchQuery.trim(),
        paperId: scopedPaperId,
      });
      setSearchResult(result);
      setActiveTab("search");
      setFeedback(
        groundingScope === "paper"
          ? "已在当前论文中完成检索，右侧 Insight Card 会显示最值得回看的命中片段。"
          : "已在整个文献库中完成检索，适合做跨论文概念定位和比较。",
        "success",
      );
    } catch (error) {
      setFeedback(`检索失败：${describeError(error)}`, "danger");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleAsk() {
    if (!askQuestion.trim()) {
      setFeedback("先写下问题，我再帮你做带出处的 RAG 问答。", "neutral");
      return;
    }

    setBusyAction("ask");
    try {
      const result = await requireApi().ask({
        question: askQuestion.trim(),
        paperId: scopedPaperId,
      });
      setAskResult(result);
      setActiveTab("qa");
      setFeedback(
        "问答已完成。现在即使模型没能稳定返回 JSON，也会尽量给出可追溯答案，而不是直接沉默。",
        "success",
      );
    } catch (error) {
      setFeedback(`问答失败：${describeError(error)}`, "danger");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleChatSend() {
    if (!chatDraft.trim()) {
      setFeedback("先写一句话，再让我继续聊。", "neutral");
      return;
    }

    const userMessage: ChatMessage = {
      id: `chat-user-${Date.now()}`,
      role: "user",
      content: chatDraft.trim(),
      createdAt: new Date().toISOString(),
    };
    const baseThread = chatMessages.filter((message) => message.id !== CHAT_PENDING_MESSAGE_ID);
    const threadForModel = [...baseThread, userMessage];
    const pendingAssistant: ChatMessage = {
      id: CHAT_PENDING_MESSAGE_ID,
      role: "assistant",
      content: "",
      createdAt: new Date().toISOString(),
      model: "pending",
    };

    setChatMessages([...threadForModel, pendingAssistant]);
    setChatDraft("");
    setBusyAction("chat");

    try {
      const result = await requireApi().chat({
        messages: threadForModel.map((message) => ({
          role: message.role,
          content: message.content,
        })),
        paperId: scopedPaperId,
        useRag: chatUseRag,
      });

      setChatMessages([...threadForModel, result.reply]);
      setActiveTab("chat");
      setFeedback(
        chatUseRag
          ? groundingScope === "paper"
            ? "已基于当前论文的检索证据生成回复。"
            : "已基于整个文献库的检索证据生成回复。"
          : "已完成本轮自由对话回复。",
        "success",
      );
    } catch (error) {
      const errorMessage = describeError(error);
      setChatMessages([
        ...threadForModel,
        {
          id: `chat-error-${Date.now()}`,
          role: "assistant",
          content: `这次回复没有成功：${errorMessage}`,
          createdAt: new Date().toISOString(),
          model: "system",
        },
      ]);
      setFeedback(`聊天失败：${errorMessage}`, "danger");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleOpenPaperPdf(paperId: string) {
    setOpeningPdfId(paperId);
    setBusyAction("paper-open");
    try {
      await syncState(
        requireApi().openPdf(paperId),
        "已使用本机默认应用打开本地 PDF。若状态为「未读」，会自动记为「阅读中」并更新最近打开时间。",
      );
    } catch (error) {
      setFeedback(`打开 PDF 失败：${describeError(error)}`, "danger");
    } finally {
      setOpeningPdfId(null);
      setBusyAction(null);
    }
  }

  async function handleOpenPdf() {
    if (!selectedPaper) {
      setFeedback("先在中间文献列表里选择一篇论文。", "neutral");
      return;
    }

    await handleOpenPaperPdf(selectedPaper.id);
  }

  async function handleRevealPaperPdf(paperId: string) {
    try {
      await requireApi().revealPdfInFolder(paperId);
      setFeedback("已在文件管理器中定位到本地 PDF。", "success");
    } catch (error) {
      setFeedback(`无法打开所在文件夹：${describeError(error)}`, "danger");
    }
  }

  async function handleSaveSettings() {
    if (!settingsDraft) {
      return;
    }

    setBusyAction("settings");
    try {
      await syncState(
        requireApi().updateSettings(settingsDraft),
        `${MODE_LABELS[settingsDraft.provider]} 已保存，新的模型路由现在会用于摘要、RAG 和聊天。`,
      );
    } catch (error) {
      setFeedback(`保存设置失败：${describeError(error)}`, "danger");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleParseProviderDraft() {
    if (!providerForm.rawText.trim()) {
      setFeedback("先粘贴一段配置，我再帮你自动拆成表单字段。", "neutral");
      return;
    }

    setBusyAction("provider-parse");
    try {
      const draft = await requireApi().parseApiProviderDraft(providerForm.rawText.trim());
      applyParsedProviderDraft(draft);
      setFeedback("已从粘贴内容里识别出 Provider 字段，你可以检查后直接保存。", "success");
    } catch (error) {
      setFeedback(`解析配置失败：${describeError(error)}`, "danger");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleImportApiProvider() {
    if (!providerForm.baseUrl.trim() || !providerForm.apiKey.trim()) {
      setFeedback("至少需要 Base URL 和 API Key，才能完成 Provider 接入。", "neutral");
      return;
    }

    setBusyAction("provider-import");
    try {
      await syncState(
        requireApi().importApiProvider({
          name: providerForm.name.trim() || undefined,
          baseUrl: providerForm.baseUrl.trim(),
          apiKey: providerForm.apiKey.trim(),
          defaultModel: providerForm.defaultModel.trim() || undefined,
        }),
        "Provider 已保存并完成连接测试。现在可以在下方为摘要、RAG 和聊天分别指定模型。",
      );
      setProviderForm(createEmptyProviderForm());
      setActiveTab("settings");
    } catch (error) {
      setFeedback(`接入 Provider 失败：${describeError(error)}`, "danger");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleRefreshProviderModels() {
    if (!activeApiProvider) {
      setFeedback("当前还没有可刷新的 Provider。", "neutral");
      return;
    }

    setBusyAction("provider-sync");
    try {
      await syncState(
        requireApi().refreshApiProviderModels(activeApiProvider.id),
        "模型列表已刷新。如果你接的是 OpenRouter 或多模型服务，现在可以重新选择更合适的模型。",
      );
    } catch (error) {
      setFeedback(`刷新模型失败：${describeError(error)}`, "danger");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleRemoveProvider() {
    if (!activeApiProvider) {
      return;
    }

    setBusyAction("provider-remove");
    try {
      await syncState(
        requireApi().removeApiProvider(activeApiProvider.id),
        "当前 Provider 已移除。",
      );
    } catch (error) {
      setFeedback(`删除 Provider 失败：${describeError(error)}`, "danger");
    } finally {
      setBusyAction(null);
    }
  }

  function switchActiveProvider(providerId: string) {
    if (!settingsDraft) {
      return;
    }

    const provider =
      settingsDraft.apiProviders.find((item) => item.id === providerId) ?? null;

    setSettingsDraft({
      ...settingsDraft,
      activeApiProviderId: providerId,
      modelRouting: buildProviderRoutingDraft(settingsDraft, provider),
    });
  }

  const suggestedPrompts = selectedPaper
    ? [
        `这篇论文的核心方法链路是什么？`,
        `作者的实验设计是否足以支撑结论？`,
        `如果我要复现这篇论文，第一步该看哪些章节？`,
      ]
    : [
        "帮我比较当前文献库里关于 RAG 的常见思路。",
        "有哪些论文更适合先精读？",
        "帮我找和长上下文检索相关的文献。",
      ];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="card brand-card">
          <div className="brand-topline">
            <span className="eyebrow">Briefly AI</span>
          </div>
          <h1>文献工作台</h1>
          <p className="brand-tagline">
            Insight / 检索 / 问答，数据在本地。
          </p>
          <button
            type="button"
            className="primary-button"
            onClick={handleImport}
            disabled={blockImport}
          >
            {busyAction === "import"
              ? "正在导入…"
              : busyAction === "loading" && electronAvailable
                ? "加载文献库…"
                : !electronAvailable
                  ? "桌面版导入"
                  : "导入 PDF"}
          </button>
        </div>

        <div className="card sidebar-card">
          <div className="sidebar-section-head">
            <h2>我的文献库</h2>
            <span className="mono">{library?.metrics.documentCount ?? 0} papers</span>
          </div>
          <div className="collection-list">
            <button
              className={`collection-button ${statusFilter === "all" ? "active" : ""}`}
              onClick={() => setStatusFilter("all")}
            >
              全部文献
              <span>{library?.papers.length ?? 0}</span>
            </button>
            {(["inbox", "reading", "summarized", "archived"] as const).map((status) => (
              <button
                key={status}
                className={`collection-button ${statusFilter === status ? "active" : ""}`}
                onClick={() => setStatusFilter(status)}
              >
                {STATUS_LABELS[status]}
                <span>
                  {library?.papers.filter((paper) => paper.status === status).length ?? 0}
                </span>
              </button>
            ))}
          </div>
          <div className="sidebar-divider" />
          <div className="sidebar-section-head">
            <h2>热门标签</h2>
            <button className="text-button" onClick={() => setTagFilter("all")}>
              清空
            </button>
          </div>
          <div className="tag-cloud">
            {(library?.availableTags ?? []).slice(0, 12).map((tag) => (
              <button
                key={tag}
                className={`tag-chip ${tagFilter === tag ? "active" : ""}`}
                onClick={() => setTagFilter(tag)}
              >
                {tag}
              </button>
            ))}
          </div>
        </div>

        <div className="card sidebar-card">
          <div className="sidebar-section-head">
            <h2>模型</h2>
            <button type="button" className="text-button" onClick={() => setActiveTab("settings")}>
              设置
            </button>
          </div>
          <div className="mini-stack">
            <div className="mini-row">
              <span>当前模式</span>
              <strong>{settingsDraft ? MODE_LABELS[settingsDraft.provider] : "加载中"}</strong>
            </div>
            <div className="mini-row">
              <span>摘要模型</span>
              <strong>{getModelLabel(settingsDraft, "summaryModel")}</strong>
            </div>
            <div className="mini-row">
              <span>RAG 模型</span>
              <strong>{getModelLabel(settingsDraft, "ragModel")}</strong>
            </div>
            <div className="mini-row">
              <span>聊天模型</span>
              <strong>{getModelLabel(settingsDraft, "chatModel")}</strong>
            </div>
          </div>
          <div className="sidebar-cta-row">
            <button type="button" className="secondary-button" onClick={() => setActiveTab("search")}>
              检索
            </button>
            <button type="button" className="secondary-button" onClick={() => setActiveTab("chat")}>
              对话
            </button>
          </div>
        </div>
      </aside>

      <main className="workspace">
        {!electronAvailable ? (
          <div className="preview-mode-strip card" role="note">
            <div>
              <strong className="preview-mode-title">浏览器预览模式</strong>
              <p className="preview-mode-copy">
                Electron 未注入，仅能预览布局。桌面端请运行
                <span className="mono preview-mode-shell"> npm run dev </span>
                。
              </p>
            </div>
          </div>
        ) : null}

        <div className={`banner card banner-${bannerTone} banner-compact`}>
          <p className="banner-line">{banner}</p>
          <span className="mono banner-metrics">
            {library
              ? `${library.metrics.documentCount} docs · ${library.metrics.indexedChunkCount} ch`
              : "…"}
          </span>
        </div>

        <div className="desktop-grid">
          <section
            className="library-pane card"
            style={{ width: paperListLayout.libraryWidth, flexShrink: 0 }}
          >
            <div className="pane-head">
              <div>
                <h2 className="pane-title-plain">列表</h2>
              </div>
              <button
                type="button"
                className="secondary-button"
                onClick={handleImport}
                disabled={blockImport}
              >
                {busyAction === "import" ? "导入中…" : "导入"}
              </button>
            </div>

            <div className="library-toolbar">
              <input
                placeholder="标题、摘要、标签…"
                value={filterKeyword}
                onChange={(event) => setFilterKeyword(event.target.value)}
                onKeyDown={handleLibraryFilterKeyDown}
                aria-label="筛选文献列表"
              />
              <p className="toolbar-hint toolbar-hint-tight">
                {widePaperTable ? (
                  <>
                    表头竖线为列宽分割线，可拖拽；条目悬停出操作。<kbd>↑</kbd>
                    <kbd>↓</kbd> 切换 · Esc 清空搜索。
                  </>
                ) : (
                  <>
                    <kbd>↑</kbd>
                    <kbd>↓</kbd> 切换条目 · Esc 清空。
                  </>
                )}
              </p>
              <div className="toolbar-row">
                <select
                  value={statusFilter}
                  onChange={(event) =>
                    setStatusFilter(event.target.value as ReadingStatus | "all")
                  }
                  aria-label="按阅读状态筛选"
                >
                  <option value="all">全部状态</option>
                  {READING_STATUS_ORDER.map((value) => (
                    <option
                      key={value}
                      value={value}
                      title={READING_STATUS_META[value].hint}
                    >
                      {STATUS_LABELS[value]}
                    </option>
                  ))}
                </select>
                <select
                  value={tagFilter}
                  onChange={(event) => setTagFilter(event.target.value)}
                >
                  <option value="all">全部标签</option>
                  {(library?.availableTags ?? []).map((tag) => (
                    <option key={tag} value={tag}>
                      {tag}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div
              className="paper-table-head"
              style={{ gridTemplateColumns: paperGridColumns }}
            >
              {widePaperTable ? (
                <>
                  <span className="paper-head-main">文献</span>
                  <button
                    type="button"
                    className="col-resize-handle"
                    aria-label="拖动：加宽或缩窄研究方向列"
                    tabIndex={-1}
                    onMouseDown={(event) => startPaperColDrag("areaFromTitle", event)}
                  />
                  <span className="paper-head-slot paper-head-area">方向</span>
                  <button
                    type="button"
                    className="col-resize-handle"
                    aria-label="拖动：在研究方向与阅读状态之间分配宽度"
                    tabIndex={-1}
                    onMouseDown={(event) => startPaperColDrag("areaStatus", event)}
                  />
                  <span className="paper-head-slot paper-head-meta">状态</span>
                  <button
                    type="button"
                    className="col-resize-handle"
                    aria-label="拖动：在阅读状态与检索列之间分配宽度"
                    tabIndex={-1}
                    onMouseDown={(event) => startPaperColDrag("statusRetr", event)}
                  />
                  <span className="paper-head-slot paper-head-meta paper-head-end">
                    检索
                  </span>
                </>
              ) : (
                <>
                  <span className="paper-head-main">文献</span>
                  <span className="paper-head-slot paper-head-area">方向</span>
                  <span className="paper-head-slot paper-head-meta">状态</span>
                  <span className="paper-head-slot paper-head-meta paper-head-end">
                    检索
                  </span>
                </>
              )}
            </div>

            <div
              className="paper-table-shell"
              role="region"
              aria-label="文献列表。在当前列表内可按上、下方向键切换选中的论文。"
              onKeyDownCapture={handlePaperListKeyDown}
            >
              <div className="paper-table">
              {filteredPapers.length === 0 ? (
                <div className="empty-state">
                  <h3>{libraryIsEmpty ? "文献库还是空的" : "当前筛选下没有文献"}</h3>
                  <p>
                    {libraryIsEmpty
                      ? "使用上方「导入 PDF」或右侧「新增导入」加入第一篇文献。"
                      : "试着清空左侧标签或状态筛选，也可以调整上方关键词。"}
                  </p>
                </div>
              ) : (
                filteredPapers.map((paper) => (
                  <div
                    key={paper.id}
                    id={`paper-row-${paper.id}`}
                    className={`paper-row ${paper.id === selectedPaper?.id ? "active" : ""}`}
                    role="presentation"
                  >
                    <button
                      type="button"
                      className="paper-row-body"
                      style={{ gridTemplateColumns: paperGridColumns }}
                      aria-current={paper.id === selectedPaper?.id ? "true" : undefined}
                      onClick={() => {
                        setSelectedPaperId(paper.id);
                        setSearchResult(null);
                        setAskResult(null);
                      }}
                    >
                      <div className="paper-main-cell">
                        <strong>{paper.title}</strong>
                        {paper.abstract ? (
                          <span className="paper-excerpt">{paper.abstract}</span>
                        ) : (
                          <span className="paper-excerpt paper-excerpt-empty">暂无摘要</span>
                        )}
                      </div>
                      {widePaperTable ? <span className="paper-col-gutter" aria-hidden /> : null}
                      <span className="paper-area-cell">{paper.researchArea}</span>
                      {widePaperTable ? <span className="paper-col-gutter" aria-hidden /> : null}
                      <div className="paper-status-cell">
                        <StatusBadge status={paper.status} />
                      </div>
                      {widePaperTable ? <span className="paper-col-gutter" aria-hidden /> : null}
                      <div
                        className="paper-chunk-cell"
                        title={`${paper.chunks.length} 个检索片段 · ${paper.references.length} 条参考文献`}
                      >
                        <strong>{paper.chunks.length}</strong>
                        <span className="chunk-slash">/</span>
                        <span className="chunk-count-secondary">{paper.references.length}</span>
                      </div>
                    </button>
                    <div
                      className="paper-actions-cell"
                      role="toolbar"
                      aria-label={`「${paper.title}」：打开 PDF、定位文件、删除文献`}
                      onClick={(event) => event.stopPropagation()}
                    >
                      <button
                        type="button"
                        className="paper-action"
                        disabled={blockImport || busyAction === "paper-open"}
                        title="使用系统默认应用打开本地复制的 PDF"
                        onClick={(event) => {
                          event.stopPropagation();
                          if (!electronAvailable) {
                            setFeedback(
                              "当前为预览模式，请在 Electron 桌面版中打开本地 PDF。",
                              "neutral",
                            );
                            return;
                          }
                          void handleOpenPaperPdf(paper.id);
                        }}
                      >
                        {openingPdfId === paper.id ? "⋯" : "打开"}
                      </button>
                      <button
                        type="button"
                        className="paper-action"
                        title="访达或资源管理器"
                        disabled={!electronAvailable}
                        onClick={(event) => {
                          event.stopPropagation();
                          void handleRevealPaperPdf(paper.id);
                        }}
                      >
                        定位
                      </button>
                      <button
                        type="button"
                        className="paper-action paper-action-delete"
                        title="从文献库移除并删除本地 PDF 副本"
                        disabled={busyAction === "paper-remove"}
                        onClick={(event) => {
                          event.stopPropagation();
                          void handleRemovePaper(paper.id, paper.title);
                        }}
                      >
                        删除
                      </button>
                    </div>
                  </div>
                ))
              )}
              </div>
            </div>
          </section>

          <div
            className="pane-splitter-vertical"
            role="separator"
            aria-orientation="vertical"
            aria-label="拖动调整列表区与阅读区宽度"
            onMouseDown={startLibrarySplitDrag}
          />

          <section className="reader-pane card">
            {libraryIsEmpty ? (
              <div className="empty-state large">
                <h2>从这里搭建你的文献库</h2>
                <p className="empty-lead">
                  Briefly AI 会把 PDF 元数据与正文片段留在本地索引里，让你在检索、问答和聊天里都能回溯到出处。
                </p>
                <button
                  type="button"
                  className="primary-button empty-primary"
                  onClick={handleImport}
                  disabled={blockImport}
                >
                  {busyAction === "import" ? "正在导入…" : "导入第一批 PDF"}
                </button>
                {!electronAvailable ? (
                  <p className="empty-footnote">
                    当前为预览模式：请改用桌面端启动后才能真正导入文献。
                  </p>
                ) : null}
              </div>
            ) : selectedPaper ? (
              <>
                <div className="reader-head">
                  <div className="reader-copy">
                    <span className="eyebrow">Reading Workspace</span>
                    <h2>{selectedPaper.title}</h2>
                    <p>
                      {selectedPaper.authors.join(", ") || "作者未识别"} · {selectedPaper.pageCount} 页 ·{" "}
                      {compactNumber(selectedPaper.wordCount)} 词 · 导入于{" "}
                      {formatDate(selectedPaper.importedAt)}
                      {selectedPaper.lastOpenedAt
                        ? ` · 最近打开 ${formatDate(selectedPaper.lastOpenedAt)}`
                        : ""}
                    </p>
                  </div>
                  <div className="reader-actions">
                    <button
                      type="button"
                      className="secondary-button"
                      onClick={() => void handleOpenPdf()}
                      disabled={
                        !electronAvailable || busyAction === "paper-open" || busyAction === "loading"
                      }
                    >
                      {openingPdfId === selectedPaper.id ? "打开中…" : "打开本地 PDF"}
                    </button>
                    <button
                      type="button"
                      className="secondary-button"
                      onClick={() => void handleRevealPaperPdf(selectedPaper.id)}
                      disabled={!electronAvailable || busyAction === "loading"}
                      title="访达 / 资源管理器"
                    >
                      在文件夹中显示
                    </button>
                    <button
                      type="button"
                      className="secondary-button"
                      onClick={() => handleRegenerateBrief(selectedPaper.id)}
                    >
                      {busyAction === "brief" ? "生成中…" : "刷新 Insight"}
                    </button>
                    <button
                      type="button"
                      className="secondary-button danger-button"
                      onClick={() => handleRemovePaper(selectedPaper.id, selectedPaper.title)}
                    >
                      {busyAction === "paper-remove" ? "删除中…" : "删除文献"}
                    </button>
                  </div>
                </div>

                <div className="editor-strip">
                  <label className="field inline-field">
                    <span>研究方向</span>
                    <input
                      defaultValue={selectedPaper.researchArea}
                      key={`${selectedPaper.id}-area`}
                      onBlur={(event) =>
                        void patchPaper(selectedPaper.id, {
                          researchArea: event.target.value.trim() || "未分类",
                        })
                      }
                    />
                  </label>
                  <label className="field inline-field">
                    <span>阅读状态</span>
                    <select
                      value={selectedPaper.status}
                      title={READING_STATUS_META[selectedPaper.status].hint}
                      onChange={(event) =>
                        void patchPaper(selectedPaper.id, {
                          status: event.target.value as ReadingStatus,
                        })
                      }
                      aria-label="标准阅读状态"
                    >
                      {READING_STATUS_ORDER.map((value) => (
                        <option key={value} value={value} title={READING_STATUS_META[value].hint}>
                          {STATUS_LABELS[value]}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="field inline-field wide-field">
                    <span>关键词标签</span>
                    <div className="tag-editor">
                      <div className="tag-editor-list">
                        {selectedPaper.tags.map((tag) => (
                          <span key={tag} className="tag-token">
                            {tag}
                            <button
                              type="button"
                              className="tag-token-remove"
                              onClick={() => void removeTag(tag)}
                              aria-label={`删除标签 ${tag}`}
                            >
                              ×
                            </button>
                          </span>
                        ))}
                      </div>
                      <input
                        value={tagDraft}
                        onChange={(event) => setTagDraft(event.target.value)}
                        onKeyDown={(event) => {
                          if (event.key === "Enter" && !event.shiftKey) {
                            event.preventDefault();
                            void commitTagDraft();
                          }
                        }}
                        placeholder="输入标签后按回车新增"
                      />
                    </div>
                  </label>
                </div>

                <div className="tab-row">
                  {[
                    ["overview", "概览"],
                    ["ai", "AI 阅读"],
                    ["search", "语义检索"],
                    ["qa", "RAG 问答"],
                    ["chat", "聊天机器人"],
                    ["notes", "阅读笔记"],
                    ["settings", "模型设置"],
                    ["metrics", "评估"],
                  ].map(([value, label]) => (
                    <button
                      key={value}
                      className={`tab-button ${activeTab === value ? "active" : ""}`}
                      onClick={() => setActiveTab(value as WorkspaceTab)}
                    >
                      {label}
                    </button>
                  ))}
                </div>

                <div className="reader-layout">
                  <div className="reader-main">
                    {activeTab === "overview" ? (
                      <div className="panel-grid two-columns">
                        <SectionCard title="原始摘要" body={selectedPaper.abstract} />
                        <SectionCard
                          title="章节结构"
                          hint={`${selectedPaper.sections.length} sections`}
                          body={
                            selectedPaper.sections
                              .slice(0, 16)
                              .map((section) => `p.${section.page} · ${section.title}`)
                              .join("\n") || "未能稳定提取章节结构。"
                          }
                        />
                        <SectionCard
                          title="参考文献"
                          hint={`${selectedPaper.references.length} refs`}
                          body={
                            selectedPaper.references
                              .slice(0, 12)
                              .map(
                                (reference, index) =>
                                  `${index + 1}. ${reference.titleHint || reference.raw}`,
                              )
                              .join("\n") || "尚未在文档中识别到参考文献。"
                          }
                        />
                        <SectionCard
                          title="检索准备情况"
                          body={`当前已切出 ${selectedPaper.chunks.length} 个可检索片段，覆盖 ${selectedPaper.pageCount} 页原文。你可以在右侧 Insight Card 判断这篇论文更适合“快速过一遍”还是“立刻精读”。`}
                        />
                      </div>
                    ) : null}

                    {activeTab === "ai" ? (
                      <div className="panel-grid two-columns">
                        <SectionCard
                          title="TL;DR"
                          hint={`${selectedPaper.brief.mode} · ${getModelLabel(
                            settingsDraft,
                            "summaryModel",
                          )}`}
                          body={selectedPaper.brief.tldr}
                        />
                        <SectionCard title="方法总结" body={selectedPaper.brief.methods} />
                        <SectionCard title="创新点" body={selectedPaper.brief.innovations} />
                        <SectionCard title="实验设置" body={selectedPaper.brief.experiments} />
                        <SectionCard title="局限性" body={selectedPaper.brief.limitations} />
                        <article className="panel-card">
                          <div className="section-header">
                            <h3>可复用笔记</h3>
                            <span>{selectedPaper.brief.groundedSections.join(" · ")}</span>
                          </div>
                          <ul className="bullet-list">
                            {selectedPaper.brief.reusableNotes.map((item) => (
                              <li key={item}>{item}</li>
                            ))}
                          </ul>
                        </article>
                      </div>
                    ) : null}

                    {activeTab === "search" ? (
                      <div className="stack-panel">
                        <p className="rag-scope-hint">
                          语义检索只返回与查询相关的<strong>正文段落摘录</strong>
                          （PDF 文本层，非表格结构化还原）。需要核对表格或精确数字时请打开对应 PDF 页面。
                        </p>
                        <div className="action-card">
                          <textarea
                            rows={3}
                            placeholder="例如：作者如何缓解长上下文检索中的噪声片段问题？Enter 提交 · Shift+Enter 换行"
                            value={searchQuery}
                            disabled={!electronAvailable || busyAction === "search"}
                            onChange={(event) => setSearchQuery(event.target.value)}
                            onKeyDown={handleSearchKeyDown}
                          />
                          <div className="action-footer">
                            <div className="inline-controls">
                              <span className="mono">Retrieval Scope</span>
                              <select
                                value={groundingScope}
                                disabled={busyAction === "search"}
                                onChange={(event) =>
                                  setGroundingScope(event.target.value as ScopeMode)
                                }
                              >
                                <option value="paper">当前论文</option>
                                <option value="library">整个文献库</option>
                              </select>
                            </div>
                            <button
                              type="button"
                              className="primary-button"
                              disabled={!electronAvailable || busyAction === "search"}
                              aria-busy={busyAction === "search"}
                              onClick={handleSearch}
                            >
                              {busyAction === "search" ? "检索中…" : "执行语义检索"}
                            </button>
                          </div>
                        </div>

                        {busyAction === "search" ? (
                          <div className="rag-loading-panel" aria-busy="true">
                            <div className="rag-loading-strip">
                              <span className="skeleton-block skeleton-line wide" />
                              <span className="skeleton-block skeleton-line narrow" />
                              <span className="skeleton-block skeleton-line medium" />
                            </div>
                            <div className="rag-loading-cards">
                              <article className="panel-card result-card skeleton-card">
                                <span className="skeleton-block skeleton-line wide" />
                                <span className="skeleton-block skeleton-line full" />
                                <span className="skeleton-block skeleton-line full short" />
                              </article>
                              <article className="panel-card result-card skeleton-card">
                                <span className="skeleton-block skeleton-line wide" />
                                <span className="skeleton-block skeleton-line full" />
                              </article>
                            </div>
                          </div>
                        ) : null}

                        {busyAction !== "search" &&
                          (searchResult ? (
                            <>
                              <div className="comparison-strip">
                                <MetricTile
                                  label="混合检索最高分"
                                  value={compactNumber(searchResult.hybridTopScore)}
                                  hint="向量相似度 + 关键词覆盖"
                                />
                                <MetricTile
                                  label="关键词基线"
                                  value={compactNumber(searchResult.keywordBaselineTopScore)}
                                  hint="只看字面命中"
                                />
                                <MetricTile
                                  label="响应时间"
                                  value={`${compactNumber(searchResult.latencyMs)} ms`}
                                  hint="本机检索耗时"
                                />
                              </div>
                            <div className="result-list">
                              {searchResult.hits.map((hit, index) => (
                                <article key={hit.chunkId} className="panel-card result-card">
                                  <div className="result-meta">
                                    <strong>段落 {index + 1}</strong>
                                    <span className="mono">
                                      {hit.paperTitle} · p.{hit.page}
                                    </span>
                                  </div>
                                  <h3>{hit.sectionTitle || "相关正文段落"}</h3>
                                  <p>{hit.excerpt}</p>
                                </article>
                              ))}
                              </div>
                            </>
                          ) : (
                            <div className="empty-state secondary">
                              <h3>还没有检索结果</h3>
                              <p>可以先用问题或方法关键词试一下，看看哪几个段落最值得回看。</p>
                            </div>
                          ))}
                      </div>
                    ) : null}

                    {activeTab === "qa" ? (
                      <div className="stack-panel">
                        <p className="rag-scope-hint">
                          RAG 问答会依据检索到的<strong>正文摘录</strong>作答，侧重帮你理解文意；涉及表格或精确数值时，请结合引用页码对照
                          PDF。
                        </p>
                        <div className="action-card">
                          <textarea
                            rows={4}
                            placeholder="例如：这篇论文的创新点和实验设计是否足以支撑结论？Enter 提交 · Shift+Enter 换行"
                            value={askQuestion}
                            disabled={!electronAvailable || busyAction === "ask"}
                            onChange={(event) => setAskQuestion(event.target.value)}
                            onKeyDown={handleAskKeyDown}
                          />
                          <div className="action-footer">
                            <div className="inline-controls">
                              <span className="mono">
                                RAG · {getModelLabel(settingsDraft, "ragModel")}
                              </span>
                              <select
                                value={groundingScope}
                                disabled={busyAction === "ask"}
                                onChange={(event) =>
                                  setGroundingScope(event.target.value as ScopeMode)
                                }
                              >
                                <option value="paper">当前论文</option>
                                <option value="library">整个文献库</option>
                              </select>
                            </div>
                            <button
                              type="button"
                              className="primary-button"
                              disabled={!electronAvailable || busyAction === "ask"}
                              aria-busy={busyAction === "ask"}
                              onClick={handleAsk}
                            >
                              {busyAction === "ask" ? "回答中…" : "开始问答"}
                            </button>
                          </div>
                        </div>

                        {busyAction === "ask" ? (
                          <article className="panel-card qa-loading-card" aria-busy="true">
                            <div className="result-meta">
                              <strong>回答</strong>
                              <span className="mono">生成中…</span>
                            </div>
                            <div className="qa-loading-body">
                              <span className="skeleton-block skeleton-line full" />
                              <span className="skeleton-block skeleton-line full" />
                              <span className="skeleton-block skeleton-line medium" />
                            </div>
                          </article>
                        ) : null}

                        {askResult ? (
                          <article className="panel-card answer-card">
                            <div className="result-meta">
                              <strong>回答</strong>
                              <span className="mono">
                                {askResult.mode} · {compactNumber(askResult.confidence * 100)}%
                              </span>
                            </div>
                            <p className="answer-body">{askResult.answer}</p>
                            <div className="citation-row">
                              {askResult.citations.map((citation) => (
                                <span key={citation.chunkId} className="citation-pill">
                                  {citation.paperTitle} · p.{citation.page}
                                  {citation.sectionTitle ? ` · ${citation.sectionTitle}` : ""}
                                </span>
                              ))}
                            </div>
                          </article>
                        ) : busyAction !== "ask" ? (
                          <div className="empty-state secondary">
                            <h3>还没有问答结果</h3>
                            <p>
                              可以直接问方法、对比、局限性或实验可信度，系统会尽量给出带出处回答。
                            </p>
                          </div>
                        ) : null}
                      </div>
                    ) : null}

                    {activeTab === "chat" ? (
                      <div className="stack-panel">
                        <p className="rag-scope-hint">
                          开启 RAG 时，助手依据<strong>检索到的正文段落</strong>
                          解读论文；若片段疑似表格提取错乱，会提示你到 PDF 原页核对，而不是复述乱码。
                        </p>
                        <article className="panel-card chat-panel-card">
                          <div className="section-header">
                            <h3>聊天机器人</h3>
                            <span>{getModelLabel(settingsDraft, "chatModel")}</span>
                          </div>
                          <div className="chat-toolbar">
                            <ToggleSwitchField
                              id="toggle-chat-rag-grounding"
                              title="RAG Grounding"
                              description={
                                chatUseRag
                                  ? "依据检索到的正文片段作答，并尽量标注引用"
                                  : "关闭后不绑定检索结果，可做自由闲聊"
                              }
                              checked={chatUseRag}
                              disabled={busyAction === "chat"}
                              onChange={setChatUseRag}
                            />
                            <div className="inline-controls">
                              <span className="mono">Scope</span>
                              <select
                                value={groundingScope}
                                disabled={!chatUseRag || busyAction === "chat"}
                                onChange={(event) =>
                                  setGroundingScope(event.target.value as ScopeMode)
                                }
                              >
                                <option value="paper">当前论文</option>
                                <option value="library">整个文献库</option>
                              </select>
                              <button
                                type="button"
                                className="secondary-button"
                                disabled={busyAction === "chat"}
                                onClick={() => setChatMessages([])}
                              >
                                清空会话
                              </button>
                            </div>
                          </div>
                          <div className="chat-stream" ref={chatStreamRef}>
                            {chatMessages.length === 0 ? (
                              <div className="empty-state secondary chat-empty">
                                <h3>还没有会话</h3>
                                <p>
                                  你可以让机器人解释论文方法、比较实验设置，或者关闭 RAG
                                  做自由学术讨论。
                                </p>
                              </div>
                            ) : (
                              chatMessages.map((message) => (
                                <div
                                  key={message.id}
                                  className={`chat-bubble ${message.role === "assistant" ? "assistant" : "user"} ${message.id === CHAT_PENDING_MESSAGE_ID ? "pending" : ""}`}
                                >
                                  <div className="result-meta">
                                    <strong>{message.role === "assistant" ? "Briefly" : "你"}</strong>
                                    <span className="mono">
                                      {message.id === CHAT_PENDING_MESSAGE_ID
                                        ? "生成中"
                                        : (message.model ?? formatDate(message.createdAt))}
                                    </span>
                                  </div>
                                  {message.id === CHAT_PENDING_MESSAGE_ID ? (
                                    <ChatTypingIndicator useRag={chatUseRag} />
                                  ) : (
                                    <p>{message.content}</p>
                                  )}
                                  {message.citations?.length ? (
                                    <div className="citation-row">
                                      {message.citations.map((citation) => (
                                        <span key={citation.chunkId} className="citation-pill">
                                          {citation.paperTitle} · p.{citation.page}
                                        </span>
                                      ))}
                                    </div>
                                  ) : null}
                                </div>
                              ))
                            )}
                          </div>
                          <textarea
                            rows={4}
                            placeholder="例如：请把这篇论文的方法讲成一个可复现的实验流程。Enter 发送 · Shift+Enter 换行"
                            value={chatDraft}
                            disabled={!electronAvailable || busyAction === "chat"}
                            onChange={(event) => setChatDraft(event.target.value)}
                            onKeyDown={handleChatKeyDown}
                          />
                          <div className="action-footer">
                            <span className="mono">
                              {chatUseRag
                                ? groundingScope === "paper"
                                  ? "Grounded in current paper"
                                  : "Grounded in full library"
                                : "Free chat mode"}
                            </span>
                            <button
                              type="button"
                              className="primary-button"
                              disabled={!electronAvailable || busyAction === "chat"}
                              aria-busy={busyAction === "chat"}
                              onClick={handleChatSend}
                            >
                              {busyAction === "chat" ? "回复中…" : "发送消息"}
                            </button>
                          </div>
                        </article>
                      </div>
                    ) : null}

                    {activeTab === "notes" ? (
                      <div className="panel-grid two-columns">
                        <article className="panel-card">
                          <div className="section-header">
                            <h3>复用笔记</h3>
                            <span>面向综述、写作和实验复现</span>
                          </div>
                          <textarea
                            rows={8}
                            placeholder="记录你想复用的观点、方法对比、未来实验灵感或写作素材。"
                            value={noteDraft}
                            onChange={(event) => setNoteDraft(event.target.value)}
                          />
                          <button className="primary-button" onClick={handleAddNote}>
                            {busyAction === "note" ? "保存中…" : "加入笔记"}
                          </button>
                        </article>
                        <article className="panel-card">
                          <div className="section-header">
                            <h3>已保存笔记</h3>
                            <span>{selectedPaper.notes.length} entries</span>
                          </div>
                          {selectedPaper.notes.length ? (
                            <div className="note-list">
                              {selectedPaper.notes.map((note) => (
                                <div key={note.id} className="note-card">
                                  <span className="mono">{formatDate(note.createdAt)}</span>
                                  <p>{note.content}</p>
                                </div>
                              ))}
                            </div>
                          ) : (
                            <p>这篇论文还没有复用笔记，可以先记录方法框架、数据集、基线和局限性。</p>
                          )}
                        </article>
                      </div>
                    ) : null}

                    {activeTab === "settings" ? settingsDraft ? (
                      <div className="settings-layout">
                        <article className="panel-card">
                          <div className="section-header">
                            <h3>模型模式</h3>
                            <span>桌面端推理配置</span>
                          </div>
                          <label className="field field-tight">
                            <span>当前模式</span>
                            <select
                              value={settingsDraft.provider}
                              onChange={(event) =>
                                updateSettingsDraft({
                                  provider: event.target.value as InferenceMode,
                                })
                              }
                            >
                              <option value="heuristic">本地轻量模式</option>
                              <option value="ollama">Ollama 本地模型</option>
                              <option value="api">API Provider</option>
                            </select>
                          </label>

                          {settingsDraft.provider === "ollama" ? (
                            <>
                              <label className="field field-tight">
                                <span>Ollama 地址</span>
                                <input
                                  value={settingsDraft.ollamaBaseUrl}
                                  onChange={(event) =>
                                    updateSettingsDraft({
                                      ollamaBaseUrl: event.target.value,
                                    })
                                  }
                                />
                              </label>
                              <label className="field field-tight">
                                <span>Ollama 模型</span>
                                <input
                                  value={settingsDraft.ollamaModel}
                                  onChange={(event) =>
                                    updateSettingsDraft({
                                      ollamaModel: event.target.value,
                                    })
                                  }
                                />
                              </label>
                            </>
                          ) : null}

                          <ToggleSwitchField
                            id="toggle-auto-brief-after-import"
                            title="导入后自动生成摘要"
                            description="新 PDF 入库后尝试生成 Insight；也可在文献卡片里手动刷新"
                            checked={settingsDraft.autoGenerateBriefs}
                            className="toggle-field-setting"
                            onChange={(next) =>
                              updateSettingsDraft({
                                autoGenerateBriefs: next,
                              })
                            }
                          />

                          <label className="field field-tight">
                            <span>自定义背景（CSS 背景值）</span>
                            <input
                              placeholder="例如：linear-gradient(135deg, #ffd89b 0%, #19547b 100%)"
                              value={customBackground}
                              onChange={(event) => setCustomBackground(event.target.value)}
                            />
                          </label>
                          <div className="button-row">
                            <button
                              type="button"
                              className="secondary-button"
                              onClick={() => setCustomBackground("")}
                            >
                              恢复默认背景
                            </button>
                          </div>

                          <button className="secondary-button full-width" onClick={handleSaveSettings}>
                            {busyAction === "settings" ? "保存中…" : "保存模式配置"}
                          </button>
                        </article>

                        <article className="panel-card">
                          <div className="section-header">
                            <h3>快速添加 API</h3>
                            <span>表单优先，更接近真实使用习惯</span>
                          </div>
                          <div className="preset-grid">
                            {PROVIDER_PRESETS.map((preset) => (
                              <button
                                key={preset.label}
                                className="preset-card"
                                onClick={() => fillProviderPreset(preset)}
                              >
                                <strong>{preset.label}</strong>
                                <span>{preset.helper}</span>
                              </button>
                            ))}
                          </div>
                          <div className="form-grid two-up">
                            <label className="field field-tight">
                              <span>Provider 名称</span>
                              <input
                                value={providerForm.name}
                                onChange={(event) =>
                                  updateProviderForm({ name: event.target.value })
                                }
                              />
                            </label>
                            <label className="field field-tight">
                              <span>默认模型</span>
                              <input
                                placeholder="例如 gpt-4o-mini / deepseek-chat"
                                value={providerForm.defaultModel}
                                onChange={(event) =>
                                  updateProviderForm({ defaultModel: event.target.value })
                                }
                              />
                            </label>
                          </div>
                          <label className="field field-tight">
                            <span>Base URL</span>
                            <input
                              placeholder="https://api.example.com/v1"
                              value={providerForm.baseUrl}
                              onChange={(event) =>
                                updateProviderForm({ baseUrl: event.target.value })
                              }
                            />
                          </label>
                          <label className="field field-tight">
                            <span>API Key</span>
                            <input
                              placeholder="sk-..."
                              value={providerForm.apiKey}
                              onChange={(event) =>
                                updateProviderForm({ apiKey: event.target.value })
                              }
                            />
                          </label>
                          <div className="action-footer action-footer-wrap">
                            <span className="subtle-copy">
                              保存时会自动测试连接，并尽量拉取模型列表。
                            </span>
                            <button className="primary-button" onClick={handleImportApiProvider}>
                              {busyAction === "provider-import" ? "连接中…" : "保存并测试连接"}
                            </button>
                          </div>
                        </article>

                        <article className="panel-card">
                          <div className="section-header">
                            <h3>粘贴配置导入</h3>
                            <span>兼容 JSON / URL / ENV 片段</span>
                          </div>
                          <textarea
                            rows={8}
                            placeholder={`例如：
{
  "name": "OpenRouter",
  "baseUrl": "https://openrouter.ai/api/v1",
  "apiKey": "sk-...",
  "model": "openai/gpt-4o-mini"
}`}
                            value={providerForm.rawText}
                            onChange={(event) =>
                              updateProviderForm({ rawText: event.target.value })
                            }
                          />
                          <div className="action-footer action-footer-wrap">
                            <button className="secondary-button" onClick={handleParseProviderDraft}>
                              {busyAction === "provider-parse" ? "解析中…" : "自动填充表单"}
                            </button>
                            <span className="subtle-copy">
                              解析后你仍然可以手动修正字段，再点击上面的“保存并测试连接”。
                            </span>
                          </div>
                        </article>

                        <article className="panel-card">
                          <div className="section-header">
                            <h3>Provider 管理与模型路由</h3>
                            <span>{settingsDraft.apiProviders.length} profiles</span>
                          </div>

                          {settingsDraft.apiProviders.length > 0 ? (
                            <>
                              <label className="field field-tight">
                                <span>当前 Provider</span>
                                <select
                                  value={settingsDraft.activeApiProviderId ?? settingsDraft.apiProviders[0]?.id}
                                  onChange={(event) => switchActiveProvider(event.target.value)}
                                >
                                  {settingsDraft.apiProviders.map((provider) => (
                                    <option key={provider.id} value={provider.id}>
                                      {provider.name}
                                    </option>
                                  ))}
                                </select>
                              </label>

                              {activeApiProvider ? (
                                <div className="provider-summary">
                                  <div className="provider-badge-row">
                                    <span className="provider-badge">{activeApiProvider.name}</span>
                                    <span className="mono">
                                      {activeApiProvider.models.length} models
                                    </span>
                                  </div>
                                  <p>{activeApiProvider.baseUrl}</p>
                                  <p>Key: {maskApiKey(activeApiProvider.apiKey)}</p>
                                  <p>
                                    最近同步：
                                    {activeApiProvider.lastSyncedAt
                                      ? ` ${formatDate(activeApiProvider.lastSyncedAt)}`
                                      : " 尚未同步"}
                                  </p>
                                </div>
                              ) : null}

                              <div className="button-row">
                                <button className="secondary-button" onClick={handleRefreshProviderModels}>
                                  {busyAction === "provider-sync" ? "同步中…" : "刷新模型"}
                                </button>
                                <button className="secondary-button danger-button" onClick={handleRemoveProvider}>
                                  {busyAction === "provider-remove" ? "移除中…" : "删除当前"}
                                </button>
                              </div>

                              <div className="form-grid three-up">
                                <label className="field field-tight">
                                  <span>摘要模型</span>
                                  <select
                                    disabled={settingsDraft.provider !== "api" || activeApiModels.length === 0}
                                    value={
                                      settingsDraft.modelRouting.summaryModel ??
                                      activeApiProvider?.defaultModel ??
                                      ""
                                    }
                                    onChange={(event) =>
                                      updateSettingsDraft({
                                        modelRouting: {
                                          summaryModel: event.target.value,
                                        },
                                      })
                                    }
                                  >
                                    {activeApiModels.length === 0 ? (
                                      <option value="">当前没有可用模型</option>
                                    ) : (
                                      activeApiModels.map((model) => (
                                        <option key={model.id} value={model.id}>
                                          {model.id}
                                        </option>
                                      ))
                                    )}
                                  </select>
                                </label>

                                <label className="field field-tight">
                                  <span>RAG 模型</span>
                                  <select
                                    disabled={settingsDraft.provider !== "api" || activeApiModels.length === 0}
                                    value={
                                      settingsDraft.modelRouting.ragModel ??
                                      activeApiProvider?.defaultModel ??
                                      ""
                                    }
                                    onChange={(event) =>
                                      updateSettingsDraft({
                                        modelRouting: {
                                          ragModel: event.target.value,
                                        },
                                      })
                                    }
                                  >
                                    {activeApiModels.length === 0 ? (
                                      <option value="">当前没有可用模型</option>
                                    ) : (
                                      activeApiModels.map((model) => (
                                        <option key={model.id} value={model.id}>
                                          {model.id}
                                        </option>
                                      ))
                                    )}
                                  </select>
                                </label>

                                <label className="field field-tight">
                                  <span>聊天模型</span>
                                  <select
                                    disabled={settingsDraft.provider !== "api" || activeApiModels.length === 0}
                                    value={
                                      settingsDraft.modelRouting.chatModel ??
                                      activeApiProvider?.defaultModel ??
                                      ""
                                    }
                                    onChange={(event) =>
                                      updateSettingsDraft({
                                        modelRouting: {
                                          chatModel: event.target.value,
                                        },
                                      })
                                    }
                                  >
                                    {activeApiModels.length === 0 ? (
                                      <option value="">当前没有可用模型</option>
                                    ) : (
                                      activeApiModels.map((model) => (
                                        <option key={model.id} value={model.id}>
                                          {model.id}
                                        </option>
                                      ))
                                    )}
                                  </select>
                                </label>
                              </div>
                            </>
                          ) : (
                            <p>还没有导入 API Provider。你可以先从上面的模板开始，或者直接粘贴一段配置。</p>
                          )}
                        </article>
                      </div>
                    ) : null : null}

                    {activeTab === "metrics" ? (
                      <div className="panel-grid two-columns">
                        <article className="panel-card">
                          <div className="section-header">
                            <h3>单篇可观测指标</h3>
                            <span>帮助判断是否值得精读</span>
                          </div>
                          <div className="metric-grid">
                            <MetricTile
                              label="结构化覆盖"
                              value={`${compactNumber(
                                (selectedPaper.sections.length / Math.max(selectedPaper.pageCount, 1)) * 100,
                              )}%`}
                              hint="章节抽取密度"
                            />
                            <MetricTile
                              label="引用条目"
                              value={compactNumber(selectedPaper.references.length)}
                              hint="参考文献识别数量"
                            />
                            <MetricTile
                              label="检索片段"
                              value={compactNumber(selectedPaper.chunks.length)}
                              hint="可供 RAG 命中的 chunk"
                            />
                            <MetricTile
                              label="摘要 grounding"
                              value={compactNumber(selectedPaper.brief.groundedSections.length)}
                              hint="摘要引用到的章节数"
                            />
                          </div>
                        </article>
                        <article className="panel-card">
                          <div className="section-header">
                            <h3>库级评估</h3>
                            <span>普通关键词搜索 vs RAG</span>
                          </div>
                          <div className="metric-grid">
                            <MetricTile
                              label="摘要准确性"
                              value={`${compactNumber(library?.metrics.summaryGroundingRate ?? 0)}%`}
                              hint="越高说明摘要越有来源支撑"
                            />
                            <MetricTile
                              label="问答命中率"
                              value={`${compactNumber(library?.metrics.qaHitRate ?? 0)}%`}
                              hint="问题更容易命中关键片段"
                            />
                            <MetricTile
                              label="引用可追溯"
                              value={`${compactNumber(library?.metrics.citationTraceabilityRate ?? 0)}%`}
                              hint="越高越容易回到原文"
                            />
                            <MetricTile
                              label="RAG 增益"
                              value={`${compactNumber(library?.metrics.ragLiftVsKeyword ?? 0)} pts`}
                              hint="混合检索相对关键词基线的提升"
                            />
                          </div>
                        </article>
                      </div>
                    ) : null}
                  </div>

                  <aside className="insight-rail">
                    <InsightCard title="Insight Card" eyebrow="Worth Reading">
                      <p className="insight-lead">{selectedPaper.brief.tldr}</p>
                      <div className="mini-stack">
                        <div className="mini-row">
                          <span>当前状态</span>
                          <strong>{STATUS_LABELS[selectedPaper.status]}</strong>
                        </div>
                        <div className="mini-row">
                          <span>研究方向</span>
                          <strong>{selectedPaper.researchArea}</strong>
                        </div>
                        <div className="mini-row">
                          <span>推荐动作</span>
                          <strong>
                            {selectedPaper.status === "inbox" ? "先看方法和实验" : "继续深挖细节"}
                          </strong>
                        </div>
                      </div>
                    </InsightCard>

                    <InsightCard title="检索快照" eyebrow="Retrieval">
                      {searchResult?.hits[0] ? (
                        <>
                          <p className="insight-lead">
                            {searchResult.hits[0].sectionTitle || "相关正文段落"} · p.{searchResult.hits[0].page}
                          </p>
                          <p>{searchResult.hits[0].excerpt}</p>
                        </>
                      ) : (
                        <p>还没有做检索。先问一个方法问题，或者搜一个术语，右侧会常驻展示最强命中片段。</p>
                      )}
                    </InsightCard>

                    <InsightCard title="问答快照" eyebrow="Grounded QA">
                      {askResult ? (
                        <>
                          <p className="insight-lead">{askResult.answer.slice(0, 140)}</p>
                          <div className="mini-stack">
                            <div className="mini-row">
                              <span>回答模式</span>
                              <strong>{askResult.mode}</strong>
                            </div>
                            <div className="mini-row">
                              <span>置信度</span>
                              <strong>{compactNumber(askResult.confidence * 100)}%</strong>
                            </div>
                            <div className="mini-row">
                              <span>引用数量</span>
                              <strong>{askResult.citations.length}</strong>
                            </div>
                          </div>
                        </>
                      ) : (
                        <p>还没有问答结果。系统现在会优先给可用答案，失败时也会明确告诉你问题出在哪里。</p>
                      )}
                    </InsightCard>

                    <InsightCard title="聊天状态" eyebrow="Assistant">
                      {busyAction === "chat" ? (
                        <>
                          <p className="insight-lead insight-live">正在生成回复…</p>
                          <div className="mini-stack">
                            <div className="mini-row">
                              <span>会话消息</span>
                              <strong>{chatMessages.length}</strong>
                            </div>
                            <div className="mini-row">
                              <span>模式</span>
                              <strong>{chatUseRag ? "RAG Grounding" : "自由对话"}</strong>
                            </div>
                          </div>
                        </>
                      ) : latestAssistantMessage ? (
                        <>
                          <p className="insight-lead">
                            {(latestAssistantMessage.content || "—").slice(0, 140)}
                          </p>
                          <div className="mini-stack">
                            <div className="mini-row">
                              <span>会话消息</span>
                              <strong>{chatMessages.length}</strong>
                            </div>
                            <div className="mini-row">
                              <span>引用片段</span>
                              <strong>{latestAssistantMessage.citations?.length ?? 0}</strong>
                            </div>
                            <div className="mini-row">
                              <span>模型</span>
                              <strong>{latestAssistantMessage.model ?? "未标记"}</strong>
                            </div>
                          </div>
                        </>
                      ) : (
                        <p>机器人还没开聊。开启 RAG 后，它会优先依据当前论文或整库检索结果来回答。</p>
                      )}
                    </InsightCard>

                    <InsightCard title="快速追问" eyebrow="Prompt Ideas">
                      <div className="prompt-list">
                        {suggestedPrompts.map((prompt) => (
                          <PromptChip
                            key={prompt}
                            label={prompt}
                            onClick={() => {
                              setChatDraft(prompt);
                              setAskQuestion(prompt);
                              setSearchQuery(prompt);
                            }}
                          />
                        ))}
                      </div>
                    </InsightCard>

                    <InsightCard title="当前模型编排" eyebrow="Routing">
                      <div className="mini-stack">
                        <div className="mini-row">
                          <span>摘要</span>
                          <strong>{getModelLabel(settingsDraft, "summaryModel")}</strong>
                        </div>
                        <div className="mini-row">
                          <span>RAG</span>
                          <strong>{getModelLabel(settingsDraft, "ragModel")}</strong>
                        </div>
                        <div className="mini-row">
                          <span>聊天</span>
                          <strong>{getModelLabel(settingsDraft, "chatModel")}</strong>
                        </div>
                        <div className="mini-row">
                          <span>Provider</span>
                          <strong>{activeApiProvider?.name ?? "本地模式 / 未配置"}</strong>
                        </div>
                      </div>
                    </InsightCard>
                  </aside>
                </div>
              </>
            ) : (
              <div className="empty-state large">
                <h2>{electronAvailable ? "还没有选中文献" : "需要桌面完整环境"}</h2>
                <p className="empty-lead">
                  {electronAvailable
                    ? "从中间文献列表点选一篇即可开始编辑元数据；若列表为空，可先发起导入。"
                    : "预览模式仅能查看界面。请使用 npm run dev 启动 Electron 窗口以加载本地文献与模型。"}
                </p>
              </div>
            )}
          </section>
        </div>
      </main>
    </div>
  );
}

export { App };
export default App;
