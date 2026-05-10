import type {
  ApiProviderProfile,
  ChatRequest,
  ProviderModelOption,
} from "../../src/shared/contracts";

interface ModelsResponse {
  data?: Array<{
    id?: string;
    owned_by?: string;
  }>;
}

interface ChatCompletionsResponse {
  model?: string;
  choices?: Array<{
    message?: {
      content?:
        | string
        | Array<{
            type?: string;
            text?: string;
          }>;
    };
  }>;
}

interface ParsedImportDraft {
  name?: string;
  baseUrl?: string;
  apiKey?: string;
  defaultModel?: string;
  models?: string[];
}

function stripCodeFence(raw: string) {
  const fenced = raw.match(/```(?:json|toml|txt)?\s*([\s\S]*?)```/i);
  return fenced?.[1]?.trim() ?? raw.trim();
}

function normalizeBaseUrl(baseUrl: string) {
  return baseUrl.trim().replace(/\/+$/, "");
}

function candidateBaseUrls(baseUrl: string) {
  const normalized = normalizeBaseUrl(baseUrl);
  if (/\/v1$/i.test(normalized)) {
    return [normalized];
  }
  return [`${normalized}/v1`, normalized];
}

function textFromContent(
  content: string | Array<{ type?: string; text?: string }> | undefined,
) {
  if (typeof content === "string") {
    return content.trim();
  }

  if (Array.isArray(content)) {
    return content
      .map((item) => item.text ?? "")
      .join("")
      .trim();
  }

  return "";
}

function extractJsonBlock(value: string): string {
  const fenced = value.match(/```json\s*([\s\S]*?)```/i);
  if (fenced?.[1]) {
    return fenced[1].trim();
  }

  const objectStart = value.indexOf("{");
  const objectEnd = value.lastIndexOf("}");
  if (objectStart >= 0 && objectEnd > objectStart) {
    return value.slice(objectStart, objectEnd + 1);
  }

  return value;
}

function inferNameFromBaseUrl(baseUrl: string) {
  try {
    const url = new URL(baseUrl);
    return url.hostname.replace(/^api\./, "") || "Imported Provider";
  } catch {
    return "Imported Provider";
  }
}

function parseJsonDraft(raw: string): ParsedImportDraft | null {
  try {
    const parsed = JSON.parse(stripCodeFence(raw)) as Record<string, unknown>;
    const models = Array.isArray(parsed.models)
      ? parsed.models.map((item) => String(item))
      : undefined;

    return {
      name: String(
        parsed.name ?? parsed.provider ?? parsed.providerName ?? "",
      ).trim() || undefined,
      baseUrl: String(
        parsed.baseUrl ?? parsed.base_url ?? parsed.endpoint ?? parsed.url ?? "",
      ).trim() || undefined,
      apiKey: String(
        parsed.apiKey ?? parsed.api_key ?? parsed.key ?? parsed.token ?? "",
      ).trim() || undefined,
      defaultModel: String(
        parsed.defaultModel ??
          parsed.mainModel ??
          parsed.main_model ??
          parsed.model ??
          "",
      ).trim() || undefined,
      models,
    };
  } catch {
    return null;
  }
}

function parseUrlDraft(raw: string): ParsedImportDraft | null {
  try {
    const parsed = new URL(raw.trim());
    const read = (...keys: string[]) =>
      keys
        .map((key) => parsed.searchParams.get(key))
        .find((value) => value && value.trim().length > 0)
        ?.trim();
    const models = parsed.searchParams
      .getAll("models")
      .map((value) => value.trim())
      .filter(Boolean);

    return {
      name: read("name", "provider", "providerName"),
      baseUrl: read("baseUrl", "base_url", "endpoint", "url"),
      apiKey: read("apiKey", "api_key", "key", "token"),
      defaultModel: read("defaultModel", "mainModel", "main_model", "model"),
      models: models.length > 0 ? models : undefined,
    };
  } catch {
    return null;
  }
}

function parseEnvDraft(raw: string): ParsedImportDraft | null {
  const lines = stripCodeFence(raw)
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (lines.length === 0 || !lines.some((line) => line.includes("="))) {
    return null;
  }

  const entries = new Map<string, string>();
  for (const line of lines) {
    const index = line.indexOf("=");
    if (index === -1) {
      continue;
    }
    const key = line.slice(0, index).trim().toLowerCase();
    const value = line.slice(index + 1).trim().replace(/^['"]|['"]$/g, "");
    entries.set(key, value);
  }

  const read = (...keys: string[]) => keys.map((key) => entries.get(key)).find(Boolean);
  const models = read("models")
    ?.split(",")
    .map((value) => value.trim())
    .filter(Boolean);

  return {
    name: read("name", "provider", "provider_name"),
    baseUrl: read("base_url", "baseurl", "endpoint", "url"),
    apiKey: read("api_key", "apikey", "key", "token"),
    defaultModel: read("default_model", "main_model", "mainmodel", "model"),
    models: models && models.length > 0 ? models : undefined,
  };
}

export function parseImportedProviderConfig(raw: string): ParsedImportDraft {
  const content = raw.trim();
  if (!content) {
    return {};
  }

  return (
    parseJsonDraft(content) ??
    parseUrlDraft(content) ??
    parseEnvDraft(content) ?? {
      apiKey: content.startsWith("sk-") ? content : undefined,
    }
  );
}

async function requestOpenAICompatible(
  baseUrl: string,
  path: string,
  init?: RequestInit,
) {
  let lastError: string | null = null;

  for (const candidate of candidateBaseUrls(baseUrl)) {
    try {
      const response = await fetch(`${candidate}${path}`, init);
      if (response.ok) {
        return {
          response,
          resolvedBaseUrl: candidate,
        };
      }

      const body = await response.text();
      const message = body.slice(0, 280) || response.statusText;

      if (response.status === 404) {
        lastError = message;
        continue;
      }

      throw new Error(message);
    } catch (error) {
      lastError = error instanceof Error ? error.message : "Unknown request error";
    }
  }

  throw new Error(lastError ?? "Provider request failed.");
}

export async function fetchOpenAICompatibleModels(
  baseUrl: string,
  apiKey: string,
): Promise<{ models: ProviderModelOption[]; resolvedBaseUrl: string }> {
  const { response, resolvedBaseUrl } = await requestOpenAICompatible(
    baseUrl,
    "/models",
    {
      headers: {
        Authorization: `Bearer ${apiKey}`,
      },
    },
  );

  const payload = (await response.json()) as ModelsResponse;
  const models =
    payload.data
      ?.map((item) => ({
        id: item.id?.trim() ?? "",
        ownedBy: item.owned_by?.trim() || undefined,
      }))
      .filter((item) => item.id.length > 0)
      .sort((left, right) => left.id.localeCompare(right.id)) ?? [];

  return {
    models,
    resolvedBaseUrl,
  };
}

export async function importOpenAICompatibleProvider(input: {
  rawText?: string;
  name?: string;
  baseUrl?: string;
  apiKey?: string;
  defaultModel?: string;
}): Promise<ApiProviderProfile> {
  const parsed = input.rawText ? parseImportedProviderConfig(input.rawText) : {};
  const baseUrl = normalizeBaseUrl(
    input.baseUrl ?? parsed.baseUrl ?? "",
  );
  const apiKey = (input.apiKey ?? parsed.apiKey ?? "").trim();
  const name =
    (input.name ?? parsed.name ?? "").trim() ||
    (baseUrl ? inferNameFromBaseUrl(baseUrl) : "Imported Provider");
  const preferredModel =
    (input.defaultModel ?? parsed.defaultModel ?? "").trim() || undefined;

  if (!baseUrl) {
    throw new Error("缺少 API Base URL。");
  }

  if (!apiKey) {
    throw new Error("缺少 API Key。");
  }

  let resolvedBaseUrl = baseUrl;
  let models: ProviderModelOption[] = [];

  try {
    const synced = await fetchOpenAICompatibleModels(baseUrl, apiKey);
    resolvedBaseUrl = synced.resolvedBaseUrl;
    models = synced.models;
  } catch {
    models = [];
  }

  if (models.length === 0) {
    const fallbackModels = [
      preferredModel,
      ...(parsed.models ?? []),
    ].filter((value, index, values): value is string => Boolean(value) && values.indexOf(value) === index);

    models = fallbackModels.map((id) => ({ id }));
  }

  return {
    id: `provider-${Math.random().toString(36).slice(2, 10)}`,
    name,
    baseUrl: resolvedBaseUrl,
    apiKey,
    type: "openai-compatible",
    defaultModel:
      preferredModel ??
      models[0]?.id,
    models,
    importedAt: new Date().toISOString(),
    lastSyncedAt: models.length > 0 ? new Date().toISOString() : undefined,
  };
}

export async function runOpenAICompatibleChat(options: {
  provider: ApiProviderProfile;
  model: string;
  messages: Array<{ role: "system" | "user" | "assistant"; content: string }>;
  temperature?: number;
}) {
  const { response } = await requestOpenAICompatible(
    options.provider.baseUrl,
    "/chat/completions",
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${options.provider.apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: options.model,
        messages: options.messages,
        temperature: options.temperature ?? 0.2,
      }),
    },
  );

  const payload = (await response.json()) as ChatCompletionsResponse;
  const content = textFromContent(payload.choices?.[0]?.message?.content);

  if (!content) {
    throw new Error("Provider returned an empty assistant message.");
  }

  return {
    text: content,
    model: payload.model ?? options.model,
  };
}

export async function generateJsonWithOpenAICompatible<T>(options: {
  provider: ApiProviderProfile;
  model: string;
  prompt: string;
}) {
  const result = await runOpenAICompatibleChat({
    provider: options.provider,
    model: options.model,
    messages: [
      {
        role: "system",
        content: "你是一名严谨的论文阅读助手。只输出合法 JSON，不要输出解释，不要带 markdown。",
      },
      {
        role: "user",
        content: options.prompt,
      },
    ],
  });

  try {
    return JSON.parse(extractJsonBlock(result.text)) as T;
  } catch {
    return null;
  }
}

export function buildChatMessagesFromHistory(request: ChatRequest) {
  return request.messages.map((message) => ({
    role: message.role,
    content: message.content,
  }));
}
