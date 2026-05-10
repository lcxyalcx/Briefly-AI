import type { UserSettings } from "../../src/shared/contracts";

interface GenerateResponse {
  response?: string;
}

export async function generateText(
  settings: UserSettings,
  prompt: string,
): Promise<string | null> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 12000);

  try {
    const response = await fetch(`${settings.ollamaBaseUrl}/api/generate`, {
      method: "POST",
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: settings.ollamaModel,
        stream: false,
        prompt,
      }),
    });

    if (!response.ok) {
      return null;
    }

    const payload = (await response.json()) as GenerateResponse;
    return payload.response?.trim() ?? null;
  } catch {
    return null;
  } finally {
    clearTimeout(timeout);
  }
}

export async function isOllamaAvailable(
  settings: UserSettings,
): Promise<boolean> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 2000);

  try {
    const response = await fetch(`${settings.ollamaBaseUrl}/api/tags`, {
      signal: controller.signal,
    });
    return response.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timeout);
  }
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

export async function generateJson<T>(
  settings: UserSettings,
  prompt: string,
): Promise<T | null> {
  const output = await generateText(
    settings,
    [
      "你是一名严谨的论文阅读助手。",
      "只输出合法 JSON，不要输出解释，不要带 markdown。",
      prompt,
    ].join("\n\n"),
  );

  if (!output) {
    return null;
  }

  try {
    return JSON.parse(extractJsonBlock(output)) as T;
  } catch {
    return null;
  }
}
