import {
  startTransition,
  useDeferredValue,
  useEffect,
  useState,
} from "react";
import type {
  AskResponse,
  ChatMessage,
  LibraryState,
  ParsedPaper,
  QueryResponse,
  ReadingStatus,
  UserSettings,
  InferenceMode,
  ApiProviderProfile,
} from "./shared/contracts";

type WorkspaceTab =
  | "overview"
  | "ai"
  | "search"
  | "qa"
  | "chat"
  | "notes"
  | "metrics";

const STATUS_LABELS: Record<ReadingStatus, string> = {
  inbox: "待阅读",
  reading: "阅读中",
  summarized: "已提炼",
  archived: "已归档",
};

const MODE_LABELS: Record<InferenceMode, string> = {
  heuristic: "本地轻量模式",
  ollama: "Ollama 本地模型",
  api: "API Provider",
};

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

  return slot === "chatModel" ? "无模型模式" : "Heuristic";
}

function buildProviderRoutingDraft(
  settings: UserSettings,
  provider: ApiProviderProfile | null,
) {
  const fallback =
    provider?.defaultModel ??
    provider?.models[0]?.id;

  return {
    summaryModel: settings.modelRouting.summaryModel ?? fallback,
    ragModel: settings.modelRouting.ragModel ?? fallback,
    chatModel: settings.modelRouting.chatModel ?? fallback,
  };
}

function StatusBadge({ status }: { status: ReadingStatus }) {
  return <span className={`status-badge status-${status}`}>{STATUS_LABELS[status]}</span>;
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
    <article className="section-card">
      <div className="section-header">
        <h3>{title}</h3>
        {hint ? <span>{hint}</span> : null}
      </div>
      <p>{body || "等待生成内容。"}</p>
    </article>
  );
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
  const [chatDraft, setChatDraft] = useState("");
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatUseRag, setChatUseRag] = useState(true);
  const [providerImportText, setProviderImportText] = useState("");
  const [settingsDraft, setSettingsDraft] = useState<UserSettings | null>(null);
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [banner, setBanner] = useState(
    "本地模式已开启，文献会被复制到应用私有目录并在本机完成索引。",
  );
  const deferredKeyword = useDeferredValue(filterKeyword);

  useEffect(() => {
    void (async () => {
      setBusyAction("loading");
      try {
        const next = await window.brieflyApi.getState();
        startTransition(() => {
          setLibrary(next);
          setSettingsDraft(next.settings);
          setSelectedPaperId(next.selectedPaperId);
        });
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

  const filteredPapers =
    library?.papers.filter((paper) => {
      const matchesKeyword = deferredKeyword
        ? `${paper.title} ${paper.abstract} ${paper.tags.join(" ")}`
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

  async function syncState(promise: Promise<LibraryState>, message: string) {
    const next = await promise;
    startTransition(() => {
      setLibrary(next);
      setSettingsDraft(next.settings);
      setSelectedPaperId((current) => current ?? next.selectedPaperId ?? next.papers[0]?.id);
    });
    setBanner(message);
  }

  async function handleImport() {
    setBusyAction("import");
    try {
      await syncState(
        window.brieflyApi.importPdfs(),
        "PDF 已导入，索引和摘要已经同步更新。",
      );
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
      await syncState(window.brieflyApi.updatePaper({ paperId, patch }), message);
    } finally {
      setBusyAction(null);
    }
  }

  async function handleRegenerateBrief(paperId: string) {
    setBusyAction("brief");
    try {
      await syncState(
        window.brieflyApi.regenerateBrief(paperId),
        "AI 阅读摘要已重新生成。",
      );
    } finally {
      setBusyAction(null);
    }
  }

  async function handleAddNote() {
    if (!selectedPaper || !noteDraft.trim()) {
      return;
    }

    setBusyAction("note");
    try {
      await syncState(
        window.brieflyApi.addNote({
          paperId: selectedPaper.id,
          content: noteDraft.trim(),
        }),
        "笔记已加入复用区。",
      );
      setNoteDraft("");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleSearch() {
    if (!selectedPaper || !searchQuery.trim()) {
      return;
    }

    setBusyAction("search");
    try {
      const result = await window.brieflyApi.search({
        query: searchQuery.trim(),
        paperId: selectedPaper.id,
      });
      setSearchResult(result);
      setBanner("语义检索完成，可以对比关键词命中和混合检索效果。");
      setActiveTab("search");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleAsk() {
    if (!askQuestion.trim()) {
      return;
    }

    setBusyAction("ask");
    try {
      const result = await window.brieflyApi.ask({
        question: askQuestion.trim(),
        paperId: selectedPaper?.id,
      });
      setAskResult(result);
      setBanner("RAG 问答已生成，回答中的出处都来自可追溯片段。");
      setActiveTab("qa");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleChatSend() {
    if (!chatDraft.trim()) {
      return;
    }

    const userMessage: ChatMessage = {
      id: `chat-user-${Date.now()}`,
      role: "user",
      content: chatDraft.trim(),
      createdAt: new Date().toISOString(),
    };
    const nextMessages = [...chatMessages, userMessage];

    setChatMessages(nextMessages);
    setChatDraft("");
    setBusyAction("chat");

    try {
      const result = await window.brieflyApi.chat({
        messages: nextMessages.map((message) => ({
          role: message.role,
          content: message.content,
        })),
        paperId: chatUseRag ? selectedPaper?.id : undefined,
        useRag: chatUseRag,
      });

      setChatMessages([...nextMessages, result.reply]);
      setBanner(
        chatUseRag
          ? `聊天机器人已基于 ${selectedPaper?.title ?? "当前文献库"} 的检索上下文继续回答。`
          : "聊天机器人已返回新的自由对话回复。",
      );
      setActiveTab("chat");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleOpenPdf() {
    if (!selectedPaper) {
      return;
    }

    await window.brieflyApi.openPdf(selectedPaper.id);
  }

  async function handleSaveSettings() {
    if (!settingsDraft) {
      return;
    }

    setBusyAction("settings");
    try {
      await syncState(
        window.brieflyApi.updateSettings(settingsDraft),
        `${MODE_LABELS[settingsDraft.provider]} 已保存。`,
      );
    } finally {
      setBusyAction(null);
    }
  }

  async function handleImportApiProvider() {
    if (!providerImportText.trim()) {
      setBanner("先粘贴一段 API 配置，再执行导入。");
      return;
    }

    setBusyAction("provider-import");
    try {
      await syncState(
        window.brieflyApi.importApiProvider({
          rawText: providerImportText.trim(),
        }),
        "API Provider 已导入并同步模型列表。你现在可以为摘要、RAG 和聊天分别选模型。",
      );
      setProviderImportText("");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleRefreshProviderModels() {
    if (!activeApiProvider) {
      return;
    }

    setBusyAction("provider-sync");
    try {
      await syncState(
        window.brieflyApi.refreshApiProviderModels(activeApiProvider.id),
        "Provider 模型列表已刷新。",
      );
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
        window.brieflyApi.removeApiProvider(activeApiProvider.id),
        "当前 Provider 已移除。",
      );
    } finally {
      setBusyAction(null);
    }
  }

  function updateSettingsDraft(patch: Partial<UserSettings>) {
    if (!settingsDraft) {
      return;
    }

    setSettingsDraft({
      ...settingsDraft,
      ...patch,
    });
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

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-card card">
          <span className="eyebrow">Briefly AI</span>
          <h1>研究生的本地文献工作台</h1>
          <p>
            把导入、结构化解析、TL;DR、问答式阅读、引用追踪和笔记复用收进同一个桌面软件。
          </p>
          <button
            className="primary-button"
            onClick={handleImport}
            disabled={busyAction === "import"}
          >
            {busyAction === "import" ? "正在导入…" : "批量导入 PDF"}
          </button>
          <div className="flow-strip">
            <span>导入</span>
            <span>解析</span>
            <span>理解</span>
            <span>检索</span>
            <span>复用</span>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h2>效果评估</h2>
            <span className="mono">MVP Proxy</span>
          </div>
          <div className="metric-grid compact">
            <MetricTile
              label="摘要准确性"
              value={`${compactNumber(library?.metrics.summaryGroundingRate ?? 0)}%`}
              hint="由摘要与章节 grounding 估算"
            />
            <MetricTile
              label="问答命中率"
              value={`${compactNumber(library?.metrics.qaHitRate ?? 0)}%`}
              hint="按近次检索与回答置信估算"
            />
            <MetricTile
              label="引用可追溯"
              value={`${compactNumber(library?.metrics.citationTraceabilityRate ?? 0)}%`}
              hint="看参考文献与片段出处是否可回溯"
            />
            <MetricTile
              label="RAG 响应"
              value={`${compactNumber(library?.metrics.averageRetrievalLatencyMs ?? 0)} ms`}
              hint="本机检索平均耗时"
            />
          </div>
        </div>

        <div className="card settings-card">
          <div className="card-header">
            <h2>模型接入中心</h2>
            <span className="mono">CC Switch Inspired</span>
          </div>
          {settingsDraft ? (
            <>
              <label className="field">
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
                  <label className="field">
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
                  <label className="field">
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

              <div className="subsection">
                <div className="subsection-head">
                  <strong>导入 API Provider</strong>
                  <span className="mono">JSON / URL / ENV</span>
                </div>
                <p className="subtle-copy">
                  支持粘贴 JSON、类似 deep link 的 URL，或 `.env` 片段。导入后会自动尝试拉取模型列表。
                </p>
                <textarea
                  className="provider-import-box"
                  rows={7}
                  placeholder={`例如：
{
  "name": "OpenRouter",
  "baseUrl": "https://openrouter.ai/api/v1",
  "apiKey": "sk-...",
  "model": "openai/gpt-4o-mini"
}`}
                  value={providerImportText}
                  onChange={(event) => setProviderImportText(event.target.value)}
                />
                <div className="button-row">
                  <button className="primary-button" onClick={handleImportApiProvider}>
                    {busyAction === "provider-import" ? "导入中…" : "导入 Provider"}
                  </button>
                </div>
              </div>

              <div className="subsection">
                <div className="subsection-head">
                  <strong>Provider 管理</strong>
                  <span className="mono">
                    {settingsDraft.apiProviders.length} profiles
                  </span>
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
                  </>
                ) : (
                  <p className="subtle-copy">
                    还没有 API Provider。你可以像 CC Switch 一样先导入一个 OpenAI-compatible 配置。
                  </p>
                )}
              </div>

              <div className="subsection">
                <div className="subsection-head">
                  <strong>模型路由</strong>
                  <span className="mono">
                    {settingsDraft.provider === "api"
                      ? activeApiProvider?.name ?? "未选 Provider"
                      : MODE_LABELS[settingsDraft.provider]}
                  </span>
                </div>
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
                          ...settingsDraft.modelRouting,
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
                          ...settingsDraft.modelRouting,
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
                          ...settingsDraft.modelRouting,
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

              <label className="checkbox-field">
                <input
                  type="checkbox"
                  checked={settingsDraft.autoGenerateBriefs}
                  onChange={(event) =>
                    updateSettingsDraft({
                      autoGenerateBriefs: event.target.checked,
                    })
                  }
                />
                <span>导入后自动生成摘要</span>
              </label>

              <button className="secondary-button settings-submit" onClick={handleSaveSettings}>
                {busyAction === "settings" ? "保存中…" : "保存当前接入配置"}
              </button>
            </>
          ) : null}
        </div>
      </aside>

      <main className="workspace">
        <div className="banner card">
          <div>
            <strong>状态</strong>
            <p>{banner}</p>
          </div>
          <span className="mono">
            {library
              ? `${library.metrics.documentCount} papers / ${library.metrics.indexedChunkCount} chunks`
              : "准备中"}
          </span>
        </div>

        <div className="workspace-grid">
          <section className="library-pane card">
            <div className="card-header">
              <div>
                <h2>文献库</h2>
                <p>按研究方向、状态、关键词管理你的论文集合。</p>
              </div>
            </div>
            <div className="filter-group">
              <input
                placeholder="搜索标题、摘要或标签"
                value={filterKeyword}
                onChange={(event) => setFilterKeyword(event.target.value)}
              />
              <div className="triple-fields">
                <select
                  value={statusFilter}
                  onChange={(event) =>
                    setStatusFilter(event.target.value as ReadingStatus | "all")
                  }
                >
                  <option value="all">全部状态</option>
                  <option value="inbox">待阅读</option>
                  <option value="reading">阅读中</option>
                  <option value="summarized">已提炼</option>
                  <option value="archived">已归档</option>
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
                <button className="secondary-button" onClick={handleImport}>
                  新增导入
                </button>
              </div>
            </div>

            <div className="paper-list">
              {filteredPapers.length === 0 ? (
                <div className="empty-state">
                  <h3>还没有可浏览的文献</h3>
                  <p>先导入一批 PDF，Briefly 会自动提取摘要、结构、引用和可检索片段。</p>
                </div>
              ) : (
                filteredPapers.map((paper) => (
                  <button
                    key={paper.id}
                    className={`paper-card ${paper.id === selectedPaper?.id ? "active" : ""}`}
                    onClick={() => {
                      setSelectedPaperId(paper.id);
                      setSearchResult(null);
                      setAskResult(null);
                    }}
                  >
                    <div className="paper-card-top">
                      <StatusBadge status={paper.status} />
                      <span className="mono">{paper.year ?? "N/A"}</span>
                    </div>
                    <strong>{paper.title}</strong>
                    <p>{paper.abstract.slice(0, 150) || "暂无摘要，建议打开原文查看。"}</p>
                    <div className="paper-meta-row">
                      <span>{paper.researchArea}</span>
                      <span>{paper.references.length} refs</span>
                    </div>
                    <div className="tag-row">
                      {paper.tags.slice(0, 4).map((tag) => (
                        <span key={tag} className="tag-pill">
                          {tag}
                        </span>
                      ))}
                    </div>
                  </button>
                ))
              )}
            </div>
          </section>

          <section className="detail-pane card">
            {selectedPaper ? (
              <>
                <div className="detail-header">
                  <div className="detail-copy">
                    <span className="eyebrow">Paper Workspace</span>
                    <h2>{selectedPaper.title}</h2>
                    <p>
                      {selectedPaper.authors.join(", ") || "作者未识别"} · {selectedPaper.pageCount} 页 ·{" "}
                      {compactNumber(selectedPaper.wordCount)} 词 · 导入于 {formatDate(selectedPaper.importedAt)}
                    </p>
                  </div>
                  <div className="detail-actions">
                    <button className="secondary-button" onClick={handleOpenPdf}>
                      打开原文
                    </button>
                    <button
                      className="secondary-button"
                      onClick={() => handleRegenerateBrief(selectedPaper.id)}
                    >
                      {busyAction === "brief" ? "生成中…" : "刷新摘要"}
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
                      onChange={(event) =>
                        void patchPaper(selectedPaper.id, {
                          status: event.target.value as ReadingStatus,
                        })
                      }
                    >
                      <option value="inbox">待阅读</option>
                      <option value="reading">阅读中</option>
                      <option value="summarized">已提炼</option>
                      <option value="archived">已归档</option>
                    </select>
                  </label>
                  <label className="field inline-field wide-field">
                    <span>关键词标签</span>
                    <input
                      defaultValue={selectedPaper.tags.join(", ")}
                      key={`${selectedPaper.id}-tags`}
                      onBlur={(event) =>
                        void patchPaper(
                          selectedPaper.id,
                          {
                            tags: event.target.value
                              .split(",")
                              .map((item) => item.trim())
                              .filter(Boolean),
                          },
                          "标签已更新。",
                        )
                      }
                    />
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

                <div className="tab-panel">
                  {activeTab === "overview" ? (
                    <div className="panel-grid two-columns">
                      <SectionCard title="摘要" body={selectedPaper.abstract} />
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
                        title="引用追踪"
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
                        title="解析摘要"
                        body={`已提取 ${selectedPaper.chunks.length} 个可检索片段，覆盖 ${selectedPaper.pageCount} 页原文。你现在可以直接问细节问题，或用语义检索比对不同术语表达。`}
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
                      <article className="section-card">
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
                      <div className="action-box">
                        <textarea
                          rows={3}
                          placeholder="例如：作者如何处理长上下文检索中的噪声片段？"
                          value={searchQuery}
                          onChange={(event) => setSearchQuery(event.target.value)}
                        />
                        <div className="action-box-footer">
                          <span className="mono">Hybrid Retrieval: vector + keyword</span>
                          <button className="primary-button" onClick={handleSearch}>
                            {busyAction === "search" ? "检索中…" : "执行语义检索"}
                          </button>
                        </div>
                      </div>

                      {searchResult ? (
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
                              hint="只看命中词，不看语义"
                            />
                            <MetricTile
                              label="响应时间"
                              value={`${compactNumber(searchResult.latencyMs)} ms`}
                              hint="本机索引命中耗时"
                            />
                          </div>
                          <div className="result-list">
                            {searchResult.hits.map((hit, index) => (
                              <article key={hit.chunkId} className="result-card">
                                <div className="result-meta">
                                  <strong>Top {index + 1}</strong>
                                  <span className="mono">
                                    p.{hit.page} · {compactNumber(hit.score)} / kw {compactNumber(hit.keywordScore)}
                                  </span>
                                </div>
                                <h3>{hit.sectionTitle || "相关段落"}</h3>
                                <p>{hit.excerpt}</p>
                              </article>
                            ))}
                          </div>
                        </>
                      ) : (
                        <div className="empty-state secondary">
                          <h3>还没有检索结果</h3>
                          <p>先给一个研究问题或方法关键词，看看 RAG 比纯关键词搜索多抓到了什么。</p>
                        </div>
                      )}
                    </div>
                  ) : null}

                  {activeTab === "qa" ? (
                    <div className="stack-panel">
                      <div className="action-box">
                        <textarea
                          rows={4}
                          placeholder="例如：这篇论文的创新点和实验设计是否支持它的结论？"
                          value={askQuestion}
                          onChange={(event) => setAskQuestion(event.target.value)}
                        />
                        <div className="action-box-footer">
                          <span className="mono">
                            RAG · {getModelLabel(settingsDraft, "ragModel")}
                          </span>
                          <button className="primary-button" onClick={handleAsk}>
                            {busyAction === "ask" ? "回答中…" : "开始问答"}
                          </button>
                        </div>
                      </div>

                      {askResult ? (
                        <article className="answer-card">
                          <div className="result-meta">
                            <strong>回答</strong>
                            <span className="mono">
                              {askResult.mode} · {compactNumber(askResult.confidence * 100)}% confidence
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
                      ) : (
                        <div className="empty-state secondary">
                          <h3>还没有问答结果</h3>
                          <p>你可以直接问“方法相比基线强在哪”“实验设置是否充分”“局限性是否被作者承认”等问题。</p>
                        </div>
                      )}
                    </div>
                  ) : null}

                  {activeTab === "chat" ? (
                    <div className="stack-panel">
                      <article className="section-card">
                        <div className="section-header">
                          <h3>聊天机器人</h3>
                          <span>{getModelLabel(settingsDraft, "chatModel")}</span>
                        </div>
                        <div className="chat-toolbar">
                          <label className="checkbox-field compact-checkbox">
                            <input
                              type="checkbox"
                              checked={chatUseRag}
                              onChange={(event) => setChatUseRag(event.target.checked)}
                            />
                            <span>
                              基于
                              {selectedPaper ? `《${selectedPaper.title}》` : "当前文献库"}
                              做 RAG
                            </span>
                          </label>
                          <button
                            className="secondary-button"
                            onClick={() => setChatMessages([])}
                          >
                            清空会话
                          </button>
                        </div>
                        <div className="chat-stream">
                          {chatMessages.length === 0 ? (
                            <div className="empty-state secondary chat-empty">
                              <h3>还没有会话</h3>
                              <p>
                                你可以让机器人解释论文方法、比较实验设置，或者在关闭 RAG 后直接进行自由学术讨论。
                              </p>
                            </div>
                          ) : (
                            chatMessages.map((message) => (
                              <div
                                key={message.id}
                                className={`chat-bubble ${message.role === "assistant" ? "assistant" : "user"}`}
                              >
                                <div className="result-meta">
                                  <strong>{message.role === "assistant" ? "Briefly" : "你"}</strong>
                                  <span className="mono">
                                    {message.model ?? formatDate(message.createdAt)}
                                  </span>
                                </div>
                                <p>{message.content}</p>
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
                          placeholder="例如：请帮我把这篇论文的方法讲成一个可以复现的实验流程。"
                          value={chatDraft}
                          onChange={(event) => setChatDraft(event.target.value)}
                        />
                        <div className="action-box-footer">
                          <span className="mono">
                            {chatUseRag ? "Grounded Chat" : "Free Chat"}
                          </span>
                          <button className="primary-button" onClick={handleChatSend}>
                            {busyAction === "chat" ? "回复中…" : "发送消息"}
                          </button>
                        </div>
                      </article>
                    </div>
                  ) : null}

                  {activeTab === "notes" ? (
                    <div className="panel-grid two-columns">
                      <article className="section-card">
                        <div className="section-header">
                          <h3>复用笔记</h3>
                          <span>面向后续写作与综述整理</span>
                        </div>
                        <textarea
                          rows={8}
                          placeholder="记录你要复用的观点、方法对比、未来实验灵感或写作素材。"
                          value={noteDraft}
                          onChange={(event) => setNoteDraft(event.target.value)}
                        />
                        <button className="primary-button" onClick={handleAddNote}>
                          {busyAction === "note" ? "保存中…" : "加入笔记"}
                        </button>
                      </article>
                      <article className="section-card">
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
                          <p>这篇论文还没有复用笔记，可以先记录方法框架、数据集和局限性。</p>
                        )}
                      </article>
                    </div>
                  ) : null}

                  {activeTab === "metrics" ? (
                    <div className="panel-grid two-columns">
                      <article className="section-card">
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
                      <article className="section-card">
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
                            hint="越高说明问题更容易命中关键片段"
                          />
                          <MetricTile
                            label="引用可追溯"
                            value={`${compactNumber(library?.metrics.citationTraceabilityRate ?? 0)}%`}
                            hint="越高越容易回到原文与参考文献"
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
              </>
            ) : (
              <div className="empty-state large">
                <h2>从一批论文开始</h2>
                <p>
                  这个桌面应用会在本机完成 PDF 解析、分块索引、摘要生成和问答检索，不需要把文献先传到在线平台。
                </p>
                <button className="primary-button" onClick={handleImport}>
                  选择 PDF 开始导入
                </button>
              </div>
            )}
          </section>
        </div>
      </main>
    </div>
  );
}

export { App };
