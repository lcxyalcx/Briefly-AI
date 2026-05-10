import fs from "node:fs/promises";
import path from "node:path";
import { app } from "electron";
import type {
  ApiProviderDraft,
  ApiProviderProfile,
  AskRequest,
  AskResponse,
  ChatRequest,
  ChatResponse,
  Citation,
  EvalMetrics,
  LibraryState,
  NoteInput,
  PaperBrief,
  PaperChunk,
  PaperFilter,
  ParsedPaper,
  QueryHit,
  QueryRequest,
  QueryResponse,
  SettingsPatch,
  UpdatePaperInput,
  UserSettings,
  ImportApiProviderInput,
} from "../../src/shared/contracts";
import { parsePdfDocument, embedText } from "./pdf";
import { generateJson, generateText, isOllamaAvailable } from "./ollama";
import {
  fetchOpenAICompatibleModels,
  generateJsonWithOpenAICompatible,
  importOpenAICompatibleProvider,
  parseImportedProviderConfig,
  runOpenAICompatibleChat,
} from "./openai-compatible";

interface QueryAnalytics {
  latencyMs: number;
  hybridTopScore: number;
  keywordBaselineTopScore: number;
  confidence: number;
}

interface PersistedLibrary {
  papers: ParsedPaper[];
  settings: UserSettings;
  analytics: QueryAnalytics[];
}

const DEFAULT_SETTINGS: UserSettings = {
  provider: "heuristic",
  ollamaBaseUrl: "http://127.0.0.1:11434",
  ollamaModel: "qwen3:4b",
  autoGenerateBriefs: true,
  apiProviders: [],
  activeApiProviderId: undefined,
  modelRouting: {},
};

function uid(prefix: string) {
  return `${prefix}-${Math.random().toString(36).slice(2, 10)}`;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function normalizeWhitespace(text: string) {
  return text.replace(/[ \t]+/g, " ").replace(/\n{3,}/g, "\n\n").trim();
}

function tokenize(text: string) {
  return normalizeWhitespace(text)
    .toLowerCase()
    .replace(/[^a-z0-9\u4e00-\u9fa5\s]/g, " ")
    .split(/\s+/)
    .filter((token) => token.length > 1);
}

function sentenceSplit(text: string) {
  return normalizeWhitespace(text)
    .split(/(?<=[。！？.!?])\s+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function extractSentences(text: string, maxSentences: number, fallbackLength = 380) {
  const sentences = sentenceSplit(text).slice(0, maxSentences);
  if (sentences.length > 0) {
    return sentences.join(" ");
  }
  return text.slice(0, fallbackLength).trim();
}

function cosineSimilarity(left: number[], right: number[]) {
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;

  for (let index = 0; index < left.length; index += 1) {
    const a = left[index] ?? 0;
    const b = right[index] ?? 0;
    dot += a * b;
    leftNorm += a * a;
    rightNorm += b * b;
  }

  const denominator = Math.sqrt(leftNorm) * Math.sqrt(rightNorm);
  if (!denominator) {
    return 0;
  }

  return dot / denominator;
}

function keywordScore(queryTokens: string[], chunk: PaperChunk) {
  if (queryTokens.length === 0) {
    return 0;
  }

  const chunkText = chunk.text.toLowerCase();
  let hits = 0;
  for (const token of queryTokens) {
    if (chunkText.includes(token)) {
      hits += 1;
    }
  }
  return hits / queryTokens.length;
}

function topSentencesByQuery(text: string, query: string, maxSentences = 2) {
  const queryTokens = new Set(tokenize(query));
  const ranked = sentenceSplit(text)
    .map((sentence) => {
      const score = tokenize(sentence).reduce(
        (sum, token) => sum + (queryTokens.has(token) ? 1 : 0),
        0,
      );
      return { sentence, score };
    })
    .sort((a, b) => b.score - a.score);

  return ranked
    .filter((item) => item.score > 0)
    .slice(0, maxSentences)
    .map((item) => item.sentence)
    .join(" ");
}

function formatContextHits(hits: QueryHit[], limit = 5) {
  return hits
    .slice(0, limit)
    .map(
      (hit, index) =>
        `[${index + 1}] ${hit.paperTitle} | p.${hit.page} | ${hit.sectionTitle ?? "Section"}\n${hit.excerpt}`,
    )
    .join("\n\n");
}

function getSectionDigest(paper: ParsedPaper, keywords: string[]) {
  const lowerKeywords = keywords.map((keyword) => keyword.toLowerCase());

  const matched = paper.chunks
    .filter((chunk) =>
      lowerKeywords.some((keyword) =>
        `${chunk.sectionTitle ?? ""} ${chunk.text}`.toLowerCase().includes(keyword),
      ),
    )
    .slice(0, 3);

  if (matched.length === 0) {
    return {
      text: extractSentences(paper.abstract || paper.text, 2),
      sections: ["Abstract"],
    };
  }

  return {
    text: matched.map((chunk) => extractSentences(chunk.text, 1, 240)).join(" "),
    sections: [...new Set(matched.map((chunk) => chunk.sectionTitle ?? `P${chunk.page}`))],
  };
}

function rankQueryHits(
  query: string,
  papers: ParsedPaper[],
  topK = 6,
): QueryHit[] {
  const queryEmbedding = embedText(query);
  const queryTokens = tokenize(query);

  return papers
    .flatMap((paper) =>
      paper.chunks.map((chunk) => {
        const vectorScore = cosineSimilarity(queryEmbedding, chunk.embedding);
        const lexical = keywordScore(queryTokens, chunk);
        const sectionBoost = chunk.sectionTitle ? 0.06 : 0;
        const hybrid = vectorScore * 0.7 + lexical * 0.24 + sectionBoost;

        return {
          chunkId: chunk.id,
          paperId: paper.id,
          paperTitle: paper.title,
          page: chunk.page,
          sectionTitle: chunk.sectionTitle,
          excerpt: chunk.text.slice(0, 480),
          score: Number(hybrid.toFixed(4)),
          keywordScore: Number(lexical.toFixed(4)),
        } satisfies QueryHit;
      }),
    )
    .sort((left, right) => right.score - left.score)
    .slice(0, topK);
}

function summarizeSearchScores(hits: QueryHit[]) {
  const keywordBaselineTopScore = hits
    .slice()
    .sort((left, right) => right.keywordScore - left.keywordScore)[0]?.keywordScore ?? 0;
  const hybridTopScore = hits[0]?.score ?? 0;

  return {
    keywordBaselineTopScore,
    hybridTopScore,
  };
}

function getActiveApiProvider(settings: UserSettings): ApiProviderProfile | null {
  const active =
    settings.apiProviders.find(
      (provider) => provider.id === settings.activeApiProviderId,
    ) ?? settings.apiProviders[0];

  return active ?? null;
}

function resolveApiModel(
  settings: UserSettings,
  slot: "summaryModel" | "ragModel" | "chatModel",
  provider: ApiProviderProfile | null,
) {
  if (!provider) {
    return undefined;
  }

  return (
    settings.modelRouting[slot] ??
    provider.defaultModel ??
    provider.models[0]?.id
  );
}

function buildCitation(hit: QueryHit): Citation {
  return {
    paperId: hit.paperId,
    paperTitle: hit.paperTitle,
    page: hit.page,
    sectionTitle: hit.sectionTitle,
    chunkId: hit.chunkId,
  };
}

function formatConversation(messages: ChatRequest["messages"]) {
  return messages
    .map((message) => `${message.role === "user" ? "User" : "Assistant"}: ${message.content}`)
    .join("\n\n");
}

function buildFallbackCitations(hits: QueryHit[], limit = 2) {
  return hits.slice(0, limit).map((hit) => buildCitation(hit));
}

function extractCitationIndexesFromText(text: string, max: number) {
  const matches = [...text.matchAll(/\[(\d+)\]/g)]
    .map((match) => Number(match[1]) - 1)
    .filter((index) => Number.isFinite(index) && index >= 0 && index < max);

  return [...new Set(matches)];
}

function buildGroundedChatPrompt(
  request: ChatRequest,
  hits: QueryHit[],
) {
  return [
    "你是 Briefly AI 的论文阅读助手。",
    "你的回答必须优先依据给定论文片段，不要编造未出现的信息。",
    "如果证据不足，请明确说“当前片段不足以支持这个结论”。",
    "如果引用了片段，请在句末使用 [1] [2] 这样的编号标记。",
    "对话历史：",
    formatConversation(request.messages),
    "论文片段：",
    formatContextHits(hits, 4),
  ].join("\n\n");
}

function hydrateSettings(settings?: Partial<UserSettings>) {
  const merged: UserSettings = {
    ...DEFAULT_SETTINGS,
    ...settings,
    apiProviders: settings?.apiProviders ?? [],
    modelRouting: {
      ...DEFAULT_SETTINGS.modelRouting,
      ...settings?.modelRouting,
    },
  };

  if (
    merged.activeApiProviderId &&
    !merged.apiProviders.some((provider) => provider.id === merged.activeApiProviderId)
  ) {
    merged.activeApiProviderId = merged.apiProviders[0]?.id;
  }

  if (!merged.activeApiProviderId && merged.apiProviders[0]) {
    merged.activeApiProviderId = merged.apiProviders[0].id;
  }

  if (merged.provider === "api" && merged.apiProviders.length === 0) {
    merged.provider = "heuristic";
  }

  return merged;
}

function sanitizeModelRouting(settings: UserSettings) {
  const provider = getActiveApiProvider(settings);
  if (!provider) {
    settings.modelRouting = {};
    return settings;
  }

  const available = new Set(provider.models.map((model) => model.id));
  const nextRouting = { ...settings.modelRouting };

  for (const key of ["summaryModel", "ragModel", "chatModel"] as const) {
    const current = nextRouting[key];
    if (current && !available.has(current)) {
      nextRouting[key] = undefined;
    }

    if (!nextRouting[key]) {
      nextRouting[key] = provider.defaultModel ?? provider.models[0]?.id;
    }
  }

  settings.modelRouting = nextRouting;
  settings.activeApiProviderId = provider.id;
  return settings;
}

async function generateHeuristicBrief(paper: ParsedPaper): Promise<PaperBrief> {
  const method = getSectionDigest(paper, ["method", "methods", "approach", "framework", "model"]);
  const innovation = getSectionDigest(paper, ["contribution", "novel", "introdu", "motivation"]);
  const experiment = getSectionDigest(paper, ["experiment", "evaluation", "dataset", "results"]);
  const limitation = getSectionDigest(paper, ["limitation", "discussion", "future work", "conclusion"]);
  const abstract = extractSentences(paper.abstract || paper.text, 2, 320);

  return {
    tldr: abstract,
    methods: method.text,
    innovations: innovation.text,
    experiments: experiment.text,
    limitations: limitation.text,
    reusableNotes: [
      `研究问题：${extractSentences(paper.abstract || paper.text, 1, 180)}`,
      `方法抓手：${method.text.slice(0, 180) || "等待更详细的方法段落。"}`,
      `实验焦点：${experiment.text.slice(0, 180) || "建议检查实验章节与数据集描述。"}`,
      `复现提醒：优先关注 ${paper.sections
        .slice(0, 4)
        .map((section) => section.title)
        .join(" / ") || "章节结构"}。`,
    ],
    groundedSections: [
      "Abstract",
      ...method.sections,
      ...innovation.sections,
      ...experiment.sections,
      ...limitation.sections,
    ].filter((value, index, values) => values.indexOf(value) === index),
    generatedAt: new Date().toISOString(),
    mode: "heuristic",
  };
}

async function maybeGenerateApiBrief(
  paper: ParsedPaper,
  settings: UserSettings,
): Promise<PaperBrief | null> {
  if (settings.provider !== "api") {
    return null;
  }

  const provider = getActiveApiProvider(settings);
  const model =
    resolveApiModel(settings, "summaryModel", provider) ??
    resolveApiModel(settings, "ragModel", provider);

  if (!provider || !model) {
    return null;
  }

  const context = `${paper.abstract}\n\n${paper.text.slice(0, 16000)}`;
  const payload = await generateJsonWithOpenAICompatible<{
    tldr: string;
    methods: string;
    innovations: string;
    experiments: string;
    limitations: string;
    reusableNotes: string[];
  }>({
    provider,
    model,
    prompt: [
      "请基于下面的论文内容，输出一个 JSON 对象，字段为：",
      'tldr, methods, innovations, experiments, limitations, reusableNotes。',
      "要求：输出中文；每个字段内容要具体、克制、可复用；reusableNotes 返回字符串数组。",
      "论文内容：",
      context,
    ].join("\n"),
  });

  if (!payload) {
    return null;
  }

  return {
    tldr: payload.tldr,
    methods: payload.methods,
    innovations: payload.innovations,
    experiments: payload.experiments,
    limitations: payload.limitations,
    reusableNotes: payload.reusableNotes?.slice(0, 6) ?? [],
    groundedSections: ["Abstract", "Method", "Evaluation", "Conclusion"],
    generatedAt: new Date().toISOString(),
    mode: "api",
  };
}

async function maybeGenerateOllamaBrief(
  paper: ParsedPaper,
  settings: UserSettings,
): Promise<PaperBrief | null> {
  if (settings.provider !== "ollama") {
    return null;
  }

  const available = await isOllamaAvailable(settings);
  if (!available) {
    return null;
  }

  const context = `${paper.abstract}\n\n${paper.text.slice(0, 16000)}`;
  const payload = await generateJson<{
    tldr: string;
    methods: string;
    innovations: string;
    experiments: string;
    limitations: string;
    reusableNotes: string[];
  }>(
    settings,
    [
      "请基于下面的论文内容，输出一个 JSON 对象，字段为：",
      'tldr, methods, innovations, experiments, limitations, reusableNotes。',
      "要求：输出中文；每个字段内容要具体、克制、可复用；reusableNotes 返回字符串数组。",
      "论文内容：",
      context,
    ].join("\n"),
  );

  if (!payload) {
    return null;
  }

  return {
    tldr: payload.tldr,
    methods: payload.methods,
    innovations: payload.innovations,
    experiments: payload.experiments,
    limitations: payload.limitations,
    reusableNotes: payload.reusableNotes?.slice(0, 6) ?? [],
    groundedSections: ["Abstract", "Method", "Evaluation", "Conclusion"],
    generatedAt: new Date().toISOString(),
    mode: "ollama",
  };
}

async function maybeGenerateApiAnswer(
  request: AskRequest,
  hits: QueryHit[],
  settings: UserSettings,
): Promise<AskResponse | null> {
  if (settings.provider !== "api") {
    return null;
  }

  const provider = getActiveApiProvider(settings);
  const model = resolveApiModel(settings, "ragModel", provider);

  if (!provider || !model || hits.length === 0) {
    return null;
  }

  const payload = await generateJsonWithOpenAICompatible<{
    answer: string;
    citationIndexes: number[];
    confidence?: number;
  }>({
    provider,
    model,
    prompt: [
      "根据给定论文片段回答问题。",
      "只允许依据上下文作答，信息不足时明确说不知道。",
      "输出 JSON，字段：answer, citationIndexes, confidence。",
      "问题：",
      request.question,
      "上下文：",
      formatContextHits(hits),
    ].join("\n\n"),
  }).catch(() => null);

  if (payload?.answer) {
    const indexes = payload.citationIndexes
      ?.map((value) => value - 1)
      .filter((value) => value >= 0 && value < hits.length) ?? [0];

    return {
      question: request.question,
      answer: payload.answer,
      citations: indexes.length
        ? indexes.map((index) => buildCitation(hits[index]))
        : buildFallbackCitations(hits),
      latencyMs: 0,
      mode: "api",
      confidence: clamp(payload.confidence ?? 0.78, 0.1, 0.99),
    };
  }

  const fallback = await runOpenAICompatibleChat({
    provider,
    model,
    messages: [
      {
        role: "system",
        content:
          "你是严谨的论文问答助手。只依据提供的论文片段回答，不足时明确说当前片段不足。引用请使用 [1] [2] 标记。",
      },
      {
        role: "user",
        content: [
          `问题：${request.question}`,
          "上下文：",
          formatContextHits(hits),
        ].join("\n\n"),
      },
    ],
  }).catch(() => null);

  if (!fallback?.text) {
    return null;
  }

  const indexes = extractCitationIndexesFromText(fallback.text, hits.length);

  return {
    question: request.question,
    answer: fallback.text,
    citations: indexes.length
      ? indexes.map((index) => buildCitation(hits[index]))
      : buildFallbackCitations(hits),
    latencyMs: 0,
    mode: "api",
    confidence: clamp(hits[0]?.score * 0.88 + 0.1, 0.18, 0.96),
  };
}

async function maybeGenerateOllamaAnswer(
  request: AskRequest,
  hits: QueryHit[],
  settings: UserSettings,
): Promise<AskResponse | null> {
  if (settings.provider !== "ollama") {
    return null;
  }

  const available = await isOllamaAvailable(settings);
  if (!available || hits.length === 0) {
    return null;
  }

  const payload = await generateJson<{
    answer: string;
    citationIndexes: number[];
    confidence?: number;
  }>(
    settings,
    [
      "根据给定论文片段回答问题。",
      "只允许依据上下文作答，信息不足时明确说不知道。",
      "输出 JSON，字段：answer, citationIndexes, confidence。",
      "问题：",
      request.question,
      "上下文：",
      formatContextHits(hits),
    ].join("\n\n"),
  );

  if (payload?.answer) {
    const indexes = payload.citationIndexes
      ?.map((value) => value - 1)
      .filter((value) => value >= 0 && value < hits.length) ?? [0];

    return {
      question: request.question,
      answer: payload.answer,
      citations: indexes.length
        ? indexes.map((index) => buildCitation(hits[index]))
        : buildFallbackCitations(hits),
      latencyMs: 0,
      mode: "ollama",
      confidence: clamp(payload.confidence ?? 0.76, 0.1, 0.99),
    };
  }

  const fallback = await generateText(
    settings,
    [
      "你是严谨的论文问答助手。只依据提供的论文片段回答，不足时明确说当前片段不足。引用请使用 [1] [2] 标记。",
      `问题：${request.question}`,
      "上下文：",
      formatContextHits(hits),
    ].join("\n\n"),
  );

  if (!fallback) {
    return null;
  }

  const indexes = extractCitationIndexesFromText(fallback, hits.length);

  return {
    question: request.question,
    answer: fallback,
    citations: indexes.length
      ? indexes.map((index) => buildCitation(hits[index]))
      : buildFallbackCitations(hits),
    latencyMs: 0,
    mode: "ollama",
    confidence: clamp(hits[0]?.score * 0.86 + 0.1, 0.18, 0.95),
  };
}

function buildExtractiveAnswer(request: AskRequest, hits: QueryHit[]): AskResponse {
  const selected = hits.slice(0, 3);
  const answerBody = selected
    .map((hit, index) => {
      const focused = topSentencesByQuery(hit.excerpt, request.question, 2) || hit.excerpt;
      return `${index + 1}. ${focused}`;
    })
    .join("\n\n");

  const answer = selected.length
    ? `根据已检索到的论文片段，和问题最相关的信息如下：\n\n${answerBody}\n\n如果你要继续精读，优先回看这些引用段落的上下文。`
    : "当前没有检索到足够相关的片段，建议换一个更具体的问题，或者先限定到单篇论文后再问。";

  const confidence = selected.length
    ? clamp(selected[0].score * 0.92 + 0.08, 0.18, 0.93)
    : 0.12;

  return {
    question: request.question,
    answer,
    citations: selected.map((hit) => buildCitation(hit)),
    latencyMs: 0,
    mode: "extractive",
    confidence,
  };
}

async function maybeGenerateApiChat(
  request: ChatRequest,
  hits: QueryHit[],
  settings: UserSettings,
): Promise<ChatResponse | null> {
  if (settings.provider !== "api") {
    return null;
  }

  const provider = getActiveApiProvider(settings);
  const model = resolveApiModel(settings, "chatModel", provider);

  if (!provider || !model) {
    return null;
  }

  if (request.useRag) {
    if (hits.length === 0) {
      return {
        reply: {
          id: uid("chat"),
          role: "assistant",
          content:
            "这次没有检索到足够相关的论文片段，所以我不想假装自己已经读到了证据。你可以换一个更具体的问题、切到“全库检索”，或者先关闭 RAG 做自由讨论。",
          createdAt: new Date().toISOString(),
          citations: [],
          model,
        },
        latencyMs: 0,
        mode: "api",
        groundedBy: [],
      };
    }

    const result = await runOpenAICompatibleChat({
      provider,
      model,
      messages: [
        {
          role: "system",
          content:
            "你是 Briefly AI 的论文聊天机器人。回答必须依据提供的论文片段，不要编造。引用请用 [1] [2] 标记；如果证据不足，明确说明当前片段不足。",
        },
        {
          role: "system",
          content: `可用论文片段：\n\n${formatContextHits(hits, 4)}`,
        },
        ...request.messages.map((message) => ({
          role: message.role,
          content: message.content,
        })),
      ],
    }).catch(() => null);

    if (!result?.text) {
      return null;
    }

    const indexes = extractCitationIndexesFromText(result.text, hits.length);
    const groundedBy = indexes.length
      ? indexes.map((index) => buildCitation(hits[index]))
      : buildFallbackCitations(hits);

    return {
      reply: {
        id: uid("chat"),
        role: "assistant",
        content: result.text,
        createdAt: new Date().toISOString(),
        citations: groundedBy,
        model: result.model,
      },
      latencyMs: 0,
      mode: "api",
      groundedBy,
    };
  }

  const result = await runOpenAICompatibleChat({
    provider,
    model,
    messages: [
      {
        role: "system",
        content:
          "你是 Briefly AI 的学术阅读助手，回答要清晰、具体、不过度编造；如果用户没有给论文上下文，也要诚实说明你的假设。",
      },
      ...request.messages.map((message) => ({
        role: message.role,
        content: message.content,
      })),
    ],
  });

  return {
    reply: {
      id: uid("chat"),
      role: "assistant",
      content: result.text,
      createdAt: new Date().toISOString(),
      citations: [],
      model: result.model,
    },
    latencyMs: 0,
    mode: "api",
    groundedBy: [],
  };
}

async function maybeGenerateOllamaChat(
  request: ChatRequest,
  hits: QueryHit[],
  settings: UserSettings,
): Promise<ChatResponse | null> {
  if (settings.provider !== "ollama") {
    return null;
  }

  const available = await isOllamaAvailable(settings);
  if (!available) {
    return null;
  }

  if (request.useRag) {
    if (hits.length === 0) {
      return {
        reply: {
          id: uid("chat"),
          role: "assistant",
          content:
            "当前没有检索到足够相关的片段，我先不把自由生成内容伪装成论文结论。你可以改写问题、扩大到全库检索，或者关闭 RAG 继续聊。",
          createdAt: new Date().toISOString(),
          citations: [],
          model: settings.ollamaModel,
        },
        latencyMs: 0,
        mode: "ollama",
        groundedBy: [],
      };
    }

    const text = await generateText(settings, buildGroundedChatPrompt(request, hits));
    if (!text) {
      return null;
    }

    const indexes = extractCitationIndexesFromText(text, hits.length);
    const groundedBy = indexes.length
      ? indexes.map((index) => buildCitation(hits[index]))
      : buildFallbackCitations(hits);

    return {
      reply: {
        id: uid("chat"),
        role: "assistant",
        content: text,
        createdAt: new Date().toISOString(),
        citations: groundedBy,
        model: settings.ollamaModel,
      },
      latencyMs: 0,
      mode: "ollama",
      groundedBy,
    };
  }

  const text = await generateText(
    settings,
    [
      "你是 Briefly AI 的学术阅读助手，回答要清晰、具体、不过度编造；如果用户没有给论文上下文，也要诚实说明你的假设。",
      "对话历史：",
      formatConversation(request.messages),
    ].join("\n\n"),
  );

  if (!text) {
    return null;
  }

  return {
    reply: {
      id: uid("chat"),
      role: "assistant",
      content: text,
      createdAt: new Date().toISOString(),
      citations: [],
      model: settings.ollamaModel,
    },
    latencyMs: 0,
    mode: "ollama",
    groundedBy: [],
  };
}

function buildHeuristicChat(
  request: ChatRequest,
  hits: QueryHit[],
): ChatResponse {
  if (request.useRag && hits.length > 0) {
    const latestUserMessage =
      [...request.messages].reverse().find((message) => message.role === "user")?.content ??
      "当前问题";
    const answer = buildExtractiveAnswer(
      {
        question: latestUserMessage,
        paperId: request.paperId,
      },
      hits,
    );

    return {
      reply: {
        id: uid("chat"),
        role: "assistant",
        content: answer.answer,
        createdAt: new Date().toISOString(),
        citations: answer.citations,
        model: "heuristic-rag",
      },
      latencyMs: 0,
      mode: "heuristic",
      groundedBy: answer.citations,
    };
  }

  if (request.useRag) {
    return {
      reply: {
        id: uid("chat"),
        role: "assistant",
        content:
          "我没有从当前检索范围里找到足够相关的论文片段，所以暂时不给出带依据的 RAG 回答。你可以换个更具体的问题、切到全库检索，或者先关闭 RAG 做自由讨论。",
        createdAt: new Date().toISOString(),
        citations: [],
        model: "heuristic-rag",
      },
      latencyMs: 0,
      mode: "heuristic",
      groundedBy: [],
    };
  }

  return {
    reply: {
      id: uid("chat"),
      role: "assistant",
      content:
        "当前聊天机器人还处在无模型模式。你可以先导入一个 OpenAI-compatible API，或者切换到本地 Ollama，再继续进行自由对话。",
      createdAt: new Date().toISOString(),
      citations: [],
      model: "heuristic",
    },
    latencyMs: 0,
    mode: "heuristic",
    groundedBy: [],
  };
}

function computeMetrics(
  papers: ParsedPaper[],
  analytics: QueryAnalytics[],
): EvalMetrics {
  const summaryGroundingRate =
    papers.length === 0
      ? 0
      : (papers.reduce((sum, paper) => {
          const ratio = paper.sections.length
            ? paper.brief.groundedSections.length / Math.max(paper.sections.length, 4)
            : 0.4;
          return sum + clamp(ratio, 0, 1);
        }, 0) /
          papers.length) *
        100;

  const citationTraceabilityRate =
    papers.length === 0
      ? 0
      : (papers.reduce((sum, paper) => {
          if (paper.references.length === 0) {
            return sum + 0.45;
          }

          const identifiable = paper.references.filter(
            (reference) => reference.year || reference.titleHint,
          ).length;
          return sum + identifiable / paper.references.length;
        }, 0) /
          papers.length) *
        100;

  const averageRetrievalLatencyMs =
    analytics.length === 0
      ? 0
      : analytics.reduce((sum, item) => sum + item.latencyMs, 0) / analytics.length;

  const qaHitRate =
    analytics.length === 0
      ? papers.length
        ? (papers.reduce(
            (sum, paper) => sum + clamp(paper.chunks.length / 40, 0.2, 1),
            0,
          ) /
            papers.length) *
          100
        : 0
      : (analytics.reduce((sum, item) => sum + item.confidence, 0) / analytics.length) *
        100;

  const ragLiftVsKeyword =
    analytics.length === 0
      ? 0
      : (analytics.reduce(
          (sum, item) => sum + (item.hybridTopScore - item.keywordBaselineTopScore),
          0,
        ) /
          analytics.length) *
        100;

  return {
    documentCount: papers.length,
    indexedChunkCount: papers.reduce((sum, paper) => sum + paper.chunks.length, 0),
    summaryGroundingRate: Number(summaryGroundingRate.toFixed(1)),
    citationTraceabilityRate: Number(citationTraceabilityRate.toFixed(1)),
    qaHitRate: Number(qaHitRate.toFixed(1)),
    averageRetrievalLatencyMs: Number(averageRetrievalLatencyMs.toFixed(1)),
    ragLiftVsKeyword: Number(ragLiftVsKeyword.toFixed(1)),
  };
}

function filterPapers(papers: ParsedPaper[], filter?: PaperFilter) {
  if (!filter) {
    return papers;
  }

  return papers.filter((paper) => {
    const matchesKeyword = filter.keyword
      ? `${paper.title} ${paper.abstract} ${paper.tags.join(" ")}`
          .toLowerCase()
          .includes(filter.keyword.toLowerCase())
      : true;
    const matchesStatus =
      !filter.status || filter.status === "all" ? true : paper.status === filter.status;
    const matchesTag =
      !filter.tag || filter.tag === "all" ? true : paper.tags.includes(filter.tag);

    return matchesKeyword && matchesStatus && matchesTag;
  });
}

export class LibraryService {
  private constructor(
    private readonly rootDir: string,
    private readonly libraryPath: string,
    private readonly documentDir: string,
    private state: PersistedLibrary,
  ) {}

  static async bootstrap() {
    const rootDir = path.join(app.getPath("userData"), "briefly-ai");
    const libraryPath = path.join(rootDir, "library.json");
    const documentDir = path.join(rootDir, "documents");

    await fs.mkdir(documentDir, { recursive: true });

    let state: PersistedLibrary;
    try {
      const file = await fs.readFile(libraryPath, "utf8");
      const parsed = JSON.parse(file) as Partial<PersistedLibrary>;
      state = {
        papers: parsed.papers ?? [],
        settings: sanitizeModelRouting(hydrateSettings(parsed.settings)),
        analytics: parsed.analytics ?? [],
      };
    } catch {
      state = {
        papers: [],
        settings: sanitizeModelRouting(hydrateSettings(DEFAULT_SETTINGS)),
        analytics: [],
      };
      await fs.writeFile(libraryPath, JSON.stringify(state, null, 2), "utf8");
    }

    return new LibraryService(rootDir, libraryPath, documentDir, state);
  }

  private async persist() {
    this.state.settings = sanitizeModelRouting(hydrateSettings(this.state.settings));
    await fs.mkdir(this.rootDir, { recursive: true });
    await fs.writeFile(this.libraryPath, JSON.stringify(this.state, null, 2), "utf8");
  }

  getPaperById(paperId: string) {
    return this.state.papers.find((paper) => paper.id === paperId);
  }

  private getScopedPapers(paperId?: string) {
    return paperId
      ? this.state.papers.filter((paper) => paper.id === paperId)
      : this.state.papers;
  }

  private async buildBriefForPaper(paper: ParsedPaper) {
    return (
      (await maybeGenerateApiBrief(paper, this.state.settings)) ??
      (await maybeGenerateOllamaBrief(paper, this.state.settings)) ??
      (await generateHeuristicBrief(paper))
    );
  }

  async getState(filter?: PaperFilter): Promise<LibraryState> {
    const papers = filterPapers(this.state.papers, filter).sort(
      (left, right) =>
        new Date(right.importedAt).getTime() - new Date(left.importedAt).getTime(),
    );

    const availableTags = [
      ...new Set(this.state.papers.flatMap((paper) => paper.tags).filter(Boolean)),
    ].sort((left, right) => left.localeCompare(right));
    const availableResearchAreas = [
      ...new Set(this.state.papers.map((paper) => paper.researchArea).filter(Boolean)),
    ].sort((left, right) => left.localeCompare(right));

    return {
      papers,
      metrics: computeMetrics(this.state.papers, this.state.analytics),
      settings: sanitizeModelRouting(hydrateSettings(this.state.settings)),
      availableTags,
      availableResearchAreas,
      selectedPaperId: papers[0]?.id,
    };
  }

  async importPdfs(filePaths: string[]) {
    for (const filePath of filePaths) {
      const parsed = await parsePdfDocument(filePath, this.documentDir);
      if (this.state.settings.autoGenerateBriefs) {
        parsed.brief = await this.buildBriefForPaper(parsed);
      }
      this.state.papers.unshift(parsed);
    }

    await this.persist();
  }

  async regenerateBrief(paperId: string) {
    const paper = this.getPaperById(paperId);
    if (!paper) {
      throw new Error("Paper not found.");
    }

    paper.brief = await this.buildBriefForPaper(paper);
    await this.persist();
  }

  async updatePaper(input: UpdatePaperInput) {
    const paper = this.getPaperById(input.paperId);
    if (!paper) {
      throw new Error("Paper not found.");
    }

    Object.assign(paper, input.patch);
    await this.persist();
  }

  async addNote(input: NoteInput) {
    const paper = this.getPaperById(input.paperId);
    if (!paper) {
      throw new Error("Paper not found.");
    }

    paper.notes.unshift({
      id: uid("note"),
      createdAt: new Date().toISOString(),
      content: input.content.trim(),
    });

    paper.status = paper.status === "inbox" ? "reading" : paper.status;
    await this.persist();
  }

  async removePaper(paperId: string) {
    const paper = this.getPaperById(paperId);
    if (!paper) {
      throw new Error("Paper not found.");
    }

    this.state.papers = this.state.papers.filter((item) => item.id !== paperId);

    try {
      await fs.rm(paper.storedPdfPath, { force: true });
    } catch {
      // If the copied PDF is already missing, we still remove the library entry.
    }

    await this.persist();
  }

  async updateSettings(patch: SettingsPatch) {
    this.state.settings = sanitizeModelRouting(
      hydrateSettings({
        ...this.state.settings,
        ...patch,
        modelRouting: {
          ...this.state.settings.modelRouting,
          ...patch.modelRouting,
        },
      }),
    );
    await this.persist();
  }

  parseApiProviderDraft(rawText: string): ApiProviderDraft {
    return parseImportedProviderConfig(rawText);
  }

  async importApiProvider(input: ImportApiProviderInput) {
    const provider = await importOpenAICompatibleProvider(input);
    const existingIndex = this.state.settings.apiProviders.findIndex(
      (item) => item.baseUrl === provider.baseUrl || item.name === provider.name,
    );

    if (existingIndex >= 0) {
      provider.id = this.state.settings.apiProviders[existingIndex].id;
      provider.importedAt = this.state.settings.apiProviders[existingIndex].importedAt;
      this.state.settings.apiProviders[existingIndex] = provider;
    } else {
      this.state.settings.apiProviders.unshift(provider);
    }

    this.state.settings.provider = "api";
    this.state.settings.activeApiProviderId = provider.id;
    this.state.settings = sanitizeModelRouting(hydrateSettings(this.state.settings));
    await this.persist();
  }

  async refreshApiProviderModels(providerId: string) {
    const provider = this.state.settings.apiProviders.find((item) => item.id === providerId);
    if (!provider) {
      throw new Error("Provider not found.");
    }

    const synced = await fetchOpenAICompatibleModels(provider.baseUrl, provider.apiKey);
    provider.baseUrl = synced.resolvedBaseUrl;
    provider.models = synced.models;
    provider.lastSyncedAt = new Date().toISOString();
    provider.defaultModel = provider.defaultModel ?? synced.models[0]?.id;

    this.state.settings = sanitizeModelRouting(hydrateSettings(this.state.settings));
    await this.persist();
  }

  async removeApiProvider(providerId: string) {
    this.state.settings.apiProviders = this.state.settings.apiProviders.filter(
      (provider) => provider.id !== providerId,
    );

    if (this.state.settings.activeApiProviderId === providerId) {
      this.state.settings.activeApiProviderId = this.state.settings.apiProviders[0]?.id;
    }

    if (this.state.settings.apiProviders.length === 0 && this.state.settings.provider === "api") {
      this.state.settings.provider = "heuristic";
    }

    this.state.settings = sanitizeModelRouting(hydrateSettings(this.state.settings));
    await this.persist();
  }

  async search(request: QueryRequest): Promise<QueryResponse> {
    const startedAt = performance.now();
    const hits = rankQueryHits(
      request.query,
      this.getScopedPapers(request.paperId),
      request.topK ?? 6,
    );
    const { keywordBaselineTopScore, hybridTopScore } = summarizeSearchScores(hits);
    const latencyMs = performance.now() - startedAt;

    this.state.analytics.unshift({
      latencyMs,
      hybridTopScore,
      keywordBaselineTopScore,
      confidence: clamp(hybridTopScore, 0.05, 0.99),
    });
    this.state.analytics = this.state.analytics.slice(0, 60);
    await this.persist();

    return {
      latencyMs: Number(latencyMs.toFixed(1)),
      query: request.query,
      hits,
      keywordBaselineTopScore,
      hybridTopScore,
    };
  }

  async ask(request: AskRequest): Promise<AskResponse> {
    const hits = rankQueryHits(
      request.question,
      this.getScopedPapers(request.paperId),
      6,
    );
    const { keywordBaselineTopScore, hybridTopScore } = summarizeSearchScores(hits);
    const startedAt = performance.now();

    const answer =
      (await maybeGenerateApiAnswer(request, hits, this.state.settings)) ??
      (await maybeGenerateOllamaAnswer(request, hits, this.state.settings)) ??
      buildExtractiveAnswer(request, hits);

    answer.latencyMs = Number((performance.now() - startedAt).toFixed(1));

    this.state.analytics.unshift({
      latencyMs: answer.latencyMs,
      hybridTopScore,
      keywordBaselineTopScore,
      confidence: answer.confidence,
    });
    this.state.analytics = this.state.analytics.slice(0, 60);
    await this.persist();

    return answer;
  }

  async chat(request: ChatRequest): Promise<ChatResponse> {
    const latestUserMessage =
      [...request.messages].reverse().find((message) => message.role === "user")?.content ?? "";
    const hits =
      request.useRag && latestUserMessage
        ? rankQueryHits(latestUserMessage, this.getScopedPapers(request.paperId), 4)
        : [];

    const startedAt = performance.now();
    const response =
      (await maybeGenerateApiChat(request, hits, this.state.settings)) ??
      (await maybeGenerateOllamaChat(request, hits, this.state.settings)) ??
      buildHeuristicChat(request, hits);

    response.latencyMs = Number((performance.now() - startedAt).toFixed(1));
    return response;
  }
}
