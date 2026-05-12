import fs from "node:fs/promises";
import path from "node:path";
import pdf from "pdf-parse";
import type {
  PaperChunk,
  PaperReference,
  PaperSection,
  ParsedPaper,
} from "../../src/shared/contracts";
import { cleanExtractedPageText } from "./textCleanup";

const KNOWN_SECTION_TITLES = [
  "abstract",
  "introduction",
  "background",
  "related work",
  "method",
  "methods",
  "approach",
  "model",
  "experiments",
  "experiment",
  "evaluation",
  "results",
  "discussion",
  "limitations",
  "future work",
  "conclusion",
  "references",
];

function normalizeWhitespace(text: string) {
  return text
    .replace(/\u0000/g, " ")
    .replace(/([A-Za-z])-\n([A-Za-z])/g, "$1$2")
    .replace(/([A-Za-z])\n([a-z])/g, "$1$2")
    .replace(/[ \t]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .replace(/[^\S\r\n]+\n/g, "\n")
    .trim();
}

function slugify(input: string) {
  return input
    .toLowerCase()
    .replace(/[^a-z0-9\u4e00-\u9fa5]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80);
}

function uid(prefix: string) {
  return `${prefix}-${Math.random().toString(36).slice(2, 10)}`;
}

function tokenize(text: string) {
  return normalizeWhitespace(text)
    .toLowerCase()
    .replace(/[^a-z0-9\u4e00-\u9fa5\s]/g, " ")
    .split(/\s+/)
    .filter((token) => token.length > 1);
}

function topKeywords(text: string, count = 8) {
  const scores = new Map<string, number>();
  for (const token of tokenize(text)) {
    scores.set(token, (scores.get(token) ?? 0) + 1);
  }

  return [...scores.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, count)
    .map(([token]) => token);
}

function hashString(input: string) {
  let hash = 2166136261;

  for (let index = 0; index < input.length; index += 1) {
    hash ^= input.charCodeAt(index);
    hash +=
      (hash << 1) + (hash << 4) + (hash << 7) + (hash << 8) + (hash << 24);
  }

  return hash >>> 0;
}

export function embedText(text: string, dimensions = 128) {
  const vector = Array.from({ length: dimensions }, () => 0);
  const tokens = tokenize(text);

  for (const token of tokens) {
    const hash = hashString(token);
    const indexA = hash % dimensions;
    const indexB = (hash >> 7) % dimensions;
    const sign = hash % 2 === 0 ? 1 : -1;
    vector[indexA] += sign * (1 + Math.min(token.length, 10) / 10);
    vector[indexB] += sign * 0.5;
  }

  const norm = Math.sqrt(vector.reduce((sum, item) => sum + item ** 2, 0)) || 1;
  return vector.map((item) => Number((item / norm).toFixed(6)));
}

function sentenceSplit(text: string) {
  return normalizeWhitespace(text)
    .split(/(?<=[。！？.!?])\s+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean);
}

function extractAbstract(fullText: string) {
  const abstractMatch = fullText.match(
    /(?:^|\n)abstract[:\s]*([\s\S]{120,2200}?)(?:\n(?:1\.?\s+)?introduction|\nkeywords|\nindex terms|\n[1I]\s+introduction)/i,
  );

  if (abstractMatch?.[1]) {
    return normalizeWhitespace(abstractMatch[1]);
  }

  return sentenceSplit(fullText).slice(0, 4).join(" ");
}

function isSectionHeading(line: string) {
  const candidate = line.trim();
  if (candidate.length < 4 || candidate.length > 90) {
    return false;
  }

  if (KNOWN_SECTION_TITLES.includes(candidate.toLowerCase())) {
    return true;
  }

  if (/^\d+(\.\d+){0,2}\s+[A-Z][A-Za-z0-9 ,:()/\-]{2,80}$/.test(candidate)) {
    return true;
  }

  if (/^[A-Z][A-Z0-9 ,:()/\-]{3,80}$/.test(candidate)) {
    return true;
  }

  return false;
}

function normalizeHeading(line: string) {
  return line
    .replace(/^\d+(\.\d+){0,2}\s+/, "")
    .replace(/\s+/g, " ")
    .trim();
}

function detectSections(pageTexts: string[]) {
  const sections: PaperSection[] = [];

  pageTexts.forEach((pageText, pageIndex) => {
    const lines = pageText
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);

    for (const line of lines) {
      if (!isSectionHeading(line)) {
        continue;
      }

      const title = normalizeHeading(line);
      const previous = sections[sections.length - 1];
      if (previous?.title.toLowerCase() === title.toLowerCase()) {
        continue;
      }

      sections.push({
        id: uid("section"),
        title,
        page: pageIndex + 1,
        level: line.match(/^\d+\.\d+/) ? 2 : 1,
      });
    }
  });

  return sections;
}

function detectReferences(fullText: string) {
  const referencesIndex = fullText.toLowerCase().lastIndexOf("\nreferences");
  if (referencesIndex === -1) {
    return [] as PaperReference[];
  }

  const tail = fullText.slice(referencesIndex);
  const rawLines = tail
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .slice(1);

  const references: string[] = [];
  let current = "";

  for (const line of rawLines) {
    if (/^(\[\d+\]|\d+\.)\s+/.test(line) || (/^[A-Z][a-zA-Z-]+,/.test(line) && current)) {
      if (current) {
        references.push(current.trim());
      }
      current = line;
    } else {
      current = `${current} ${line}`.trim();
    }
  }

  if (current) {
    references.push(current.trim());
  }

  return references.slice(0, 80).map((raw) => {
    const yearMatch = raw.match(/\b(19|20)\d{2}\b/);
    const titleMatch =
      raw.match(/[“"]([^“”"]{8,160})[”"]/u) ??
      raw.match(/\.\s([^.]{12,160})\.\s(?:[A-Z][a-z]+|Proc|In\b)/);

    return {
      id: uid("ref"),
      raw,
      year: yearMatch?.[0],
      titleHint: titleMatch?.[1]?.trim(),
    };
  });
}

function locateSectionTitle(sections: PaperSection[], page: number) {
  const candidates = sections.filter((section) => section.page <= page);
  return candidates[candidates.length - 1]?.title;
}

function splitPageBySection(pageText: string, page: number, sections: PaperSection[]) {
  const lines = pageText
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  const blocks: Array<{ title?: string; text: string }> = [];
  let currentTitle: string | undefined;
  let currentLines: string[] = [];

  for (const line of lines) {
    if (isSectionHeading(line)) {
      if (currentLines.length > 0 && currentTitle) {
        blocks.push({
          title: currentTitle,
          text: normalizeWhitespace(currentLines.join("\n")),
        });
      }

      currentTitle = normalizeHeading(line);
      currentLines = [];
      continue;
    }

    currentLines.push(line);
  }

  if (currentLines.length > 0 && currentTitle) {
    blocks.push({
      title: currentTitle,
      text: normalizeWhitespace(currentLines.join("\n")),
    });
  }

  return blocks.filter((block) => block.text.length > 20);
}

function chunkPageText(
  paperId: string,
  pageText: string,
  page: number,
  sections: PaperSection[],
) {
  const chunks: PaperChunk[] = [];
  const blocks = splitPageBySection(pageText, page, sections);

  for (const block of blocks) {
    const sentences = sentenceSplit(block.text);
    let current = "";

    for (const sentence of sentences) {
      if ((current + sentence).length > 900 && current) {
        chunks.push({
          id: uid("chunk"),
          paperId,
          page,
          text: current.trim(),
          sectionTitle: block.title,
          embedding: embedText(current),
          keywords: topKeywords(current),
        });

        const overlap = current.split(" ").slice(-36).join(" ");
        current = `${overlap} ${sentence}`.trim();
      } else {
        current = `${current} ${sentence}`.trim();
      }
    }

    if (current) {
      chunks.push({
        id: uid("chunk"),
        paperId,
        page,
        text: current.trim(),
        sectionTitle: block.title,
        embedding: embedText(current),
        keywords: topKeywords(current),
      });
    }
  }

  return chunks;
}

function extractTitle(
  fullText: string,
  fileName: string,
  metadataTitle?: string,
) {
  if (metadataTitle && metadataTitle.trim().length > 6) {
    return normalizeWhitespace(metadataTitle);
  }

  const lines = fullText
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  const candidate =
    lines.find((line) => line.length > 18 && line.length < 180) ?? fileName;
  return candidate.replace(/\s+/g, " ").trim();
}

function extractAuthors(fullText: string, metadataAuthor?: string) {
  if (metadataAuthor) {
    return metadataAuthor
      .split(/[,;]+/)
      .map((author) => author.trim())
      .filter(Boolean);
  }

  const lines = fullText
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .slice(0, 12);

  const authorLine = lines.find(
    (line) =>
      /^[A-Z][A-Za-z.\- ]+(,\s*[A-Z][A-Za-z.\- ]+)+$/.test(line) ||
      /^[A-Z][A-Za-z.\- ]+\s+(and|&)\s+[A-Z][A-Za-z.\- ]+$/i.test(line),
  );

  if (!authorLine) {
    return [];
  }

  return authorLine
    .split(/,| and | & /i)
    .map((author) => author.trim())
    .filter(Boolean);
}

function inferYear(fullText: string, metadataCreationDate?: string) {
  const metadataYear = metadataCreationDate?.match(/\b(19|20)\d{2}\b/);
  if (metadataYear?.[0]) {
    return metadataYear[0];
  }

  const header = fullText.slice(0, 1200);
  return header.match(/\b(19|20)\d{2}\b/)?.[0];
}

export async function parsePdfDocument(
  filePath: string,
  outputDir: string,
) {
  const buffer = await fs.readFile(filePath);
  const pageTexts: string[] = [];
  const fileName = path.basename(filePath);

  const parsed = await pdf(buffer, {
    pagerender: async (pageData: any) => {
      const content = await pageData.getTextContent({
        normalizeWhitespace: true,
      });

      let lastY: number | undefined;
      const lines: string[] = [];
      let currentLine = "";

      for (const item of content.items as any[]) {
        const value = typeof item.str === "string" ? item.str : "";
        if (!value) {
          continue;
        }

        const y = item.transform?.[5];
        if (lastY !== undefined && Math.abs(y - lastY) > 5) {
          if (currentLine.trim()) {
            lines.push(currentLine.trim());
          }
          currentLine = value;
        } else {
          currentLine = `${currentLine} ${value}`.trim();
        }
        lastY = y;
      }

      if (currentLine.trim()) {
        lines.push(currentLine.trim());
      }

      const pageText = cleanExtractedPageText(lines.join("\n"));
      pageTexts.push(pageText);
      return pageText;
    },
  });

  const fullText = normalizeWhitespace(pageTexts.join("\n\n"));
  const title = extractTitle(fullText, fileName, parsed.info?.Title);
  const paperId = slugify(`${title}-${Date.now()}`) || uid("paper");
  const storedPdfPath = path.join(outputDir, `${paperId}.pdf`);

  await fs.copyFile(filePath, storedPdfPath);

  const sections = detectSections(pageTexts);
  const chunks = pageTexts.flatMap((pageText, index) =>
    chunkPageText(paperId, pageText, index + 1, sections),
  );

  const document: ParsedPaper = {
    id: paperId,
    title,
    authors: extractAuthors(fullText, parsed.info?.Author),
    abstract: extractAbstract(fullText),
    year: inferYear(fullText, parsed.info?.CreationDate),
    sourceFileName: fileName,
    storedPdfPath,
    importedAt: new Date().toISOString(),
    lastOpenedAt: new Date().toISOString(),
    researchArea: "未分类",
    status: "inbox",
    tags: topKeywords(`${title} ${extractAbstract(fullText)}`, 5),
    pageCount: pageTexts.length || parsed.numpages || 1,
    wordCount: tokenize(fullText).length,
    text: fullText,
    sections,
    references: detectReferences(fullText),
    chunks,
    brief: {
      tldr: "",
      methods: "",
      innovations: "",
      experiments: "",
      limitations: "",
      reusableNotes: [],
      groundedSections: [],
      generatedAt: new Date(0).toISOString(),
      mode: "heuristic",
    },
    notes: [],
  };

  return document;
}
