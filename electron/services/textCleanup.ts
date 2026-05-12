/** PDF 文本层的表格、双栏等常被拆成乱序片段；在此做轻量清理供检索与展示。 */

export function normalizePdfUnicode(text: string): string {
  return text
    .replace(/\u0000/g, " ")
    .replace(/[\u00a0\u200b-\u200d\ufeff]/g, " ")
    .replace(/[ \t\f\v]{2,}/g, " ");
}

function lineLooksLikeBrokenTableOrNoise(line: string): boolean {
  const t = line.trim();
  if (t.length < 28) {
    return false;
  }

  const tokens = t.split(/\s+/).filter(Boolean);
  if (tokens.length < 10) {
    return false;
  }

  const letters = (t.match(/[a-zA-Z\u4e00-\u9fa5]/g) ?? []).length;
  const letterRatio = letters / Math.max(t.length, 1);
  const numericLike = tokens.filter((tok) => {
    return /^[\d.,:%\-+/]+$/.test(tok) || /^[\d.]+%$/.test(tok);
  }).length;

  if (numericLike / tokens.length >= 0.5 && letterRatio < 0.14) {
    return true;
  }

  const veryShort = tokens.filter((tok) => tok.length <= 2).length;
  if (veryShort / tokens.length >= 0.48 && tokens.length >= 14 && letterRatio < 0.25) {
    return true;
  }

  return false;
}

function softenTableGarbageLines(text: string): string {
  const rawLines = text.split("\n");
  const out: string[] = [];
  let garbageStreak = 0;
  const marker =
    "〔表格或多列排版区域：PDF 文本层提取可能错位，请到原文 PDF 该页核对〕";

  for (const raw of rawLines) {
    const line = raw.trim();
    if (!line) {
      continue;
    }

    if (lineLooksLikeBrokenTableOrNoise(line)) {
      garbageStreak += 1;
      if (garbageStreak === 1) {
        out.push(marker);
      }
      continue;
    }

    garbageStreak = 0;
    out.push(line);
  }

  return out.join("\n");
}

/**
 * 单页拼接成行文本后的清理入口（在分块 / embedding 之前调用）。
 */
export function cleanExtractedPageText(raw: string): string {
  let t = normalizePdfUnicode(raw);
  t = softenTableGarbageLines(t);
  return t
    .replace(/\u0000/g, " ")
    .replace(/([A-Za-z])-\n([A-Za-z])/g, "$1$2")
    .replace(/([A-Za-z])\n([a-z])/g, "$1$2")
    .replace(/[ \t]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .replace(/[^\S\r\n]+\n/g, "\n")
    .trim();
}

/**
 * 生成检索命中摘录：再次规整并在句号等处截断，避免半截乱码。
 */
export function excerptForRetrieval(text: string, maxChars: number): string {
  const cleaned = cleanExtractedPageText(text);
  if (cleaned.length <= maxChars) {
    return cleaned;
  }

  const slice = cleaned.slice(0, maxChars);
  const cutPoints = [
    slice.lastIndexOf("。"),
    slice.lastIndexOf("．"),
    slice.lastIndexOf("？"),
    slice.lastIndexOf("！"),
    slice.lastIndexOf(". "),
    slice.lastIndexOf("\n"),
  ];
  const lastBoundary = Math.max(...cutPoints);

  if (lastBoundary > maxChars * 0.52) {
    return slice.slice(0, lastBoundary + 1).trim();
  }

  return `${slice.trim()}…`;
}
