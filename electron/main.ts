import { app, BrowserWindow, dialog, ipcMain, shell } from "electron";
import path from "node:path";
import { LibraryService } from "./services/library";
import type {
  ApiProviderDraft,
  AskRequest,
  ChatRequest,
  ImportApiProviderInput,
  LibraryState,
  NoteInput,
  PaperFilter,
  QueryRequest,
  SettingsPatch,
  UpdatePaperInput,
} from "../src/shared/contracts";

let mainWindow: BrowserWindow | null = null;
let libraryService: LibraryService | null = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1480,
    height: 960,
    minWidth: 1180,
    minHeight: 760,
    backgroundColor: "#09111f",
    titleBarStyle: "hiddenInset",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  const devUrl = process.env.VITE_DEV_SERVER_URL;

  if (devUrl) {
    void mainWindow.loadURL(devUrl);
  } else {
    void mainWindow.loadFile(path.join(__dirname, "../../dist/index.html"));
  }

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    void shell.openExternal(url);
    return { action: "deny" };
  });

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

async function getService() {
  if (!libraryService) {
    libraryService = await LibraryService.bootstrap();
  }
  return libraryService;
}

async function refreshState(filter?: PaperFilter): Promise<LibraryState> {
  const service = await getService();
  return service.getState(filter);
}

app.whenReady().then(async () => {
  libraryService = await LibraryService.bootstrap();
  createWindow();

  ipcMain.handle("library:get-state", async (_event, filter?: PaperFilter) =>
    refreshState(filter),
  );

  ipcMain.handle("library:import-pdfs", async () => {
    const window = BrowserWindow.getFocusedWindow() ?? mainWindow;
    const result = await dialog.showOpenDialog(window!, {
      properties: ["openFile", "multiSelections"],
      filters: [{ name: "PDF", extensions: ["pdf"] }],
    });

    if (result.canceled || result.filePaths.length === 0) {
      return refreshState();
    }

    const service = await getService();
    await service.importPdfs(result.filePaths);
    return refreshState();
  });

  ipcMain.handle(
    "library:update-paper",
    async (_event, input: UpdatePaperInput) => {
      const service = await getService();
      await service.updatePaper(input);
      return refreshState();
    },
  );

  ipcMain.handle("library:add-note", async (_event, input: NoteInput) => {
    const service = await getService();
    await service.addNote(input);
    return refreshState();
  });

  ipcMain.handle("library:search", async (_event, request: QueryRequest) => {
    const service = await getService();
    return service.search(request);
  });

  ipcMain.handle("library:ask", async (_event, request: AskRequest) => {
    const service = await getService();
    return service.ask(request);
  });

  ipcMain.handle("library:chat", async (_event, request: ChatRequest) => {
    const service = await getService();
    return service.chat(request);
  });

  ipcMain.handle(
    "library:regenerate-brief",
    async (_event, paperId: string) => {
      const service = await getService();
      await service.regenerateBrief(paperId);
      return refreshState();
    },
  );

  ipcMain.handle(
    "library:update-settings",
    async (_event, patch: SettingsPatch) => {
      const service = await getService();
      await service.updateSettings(patch);
      return refreshState();
    },
  );

  ipcMain.handle(
    "library:parse-api-provider-draft",
    async (_event, rawText: string): Promise<ApiProviderDraft> => {
      const service = await getService();
      return service.parseApiProviderDraft(rawText);
    },
  );

  ipcMain.handle(
    "library:import-api-provider",
    async (_event, input: ImportApiProviderInput) => {
      const service = await getService();
      await service.importApiProvider(input);
      return refreshState();
    },
  );

  ipcMain.handle(
    "library:refresh-api-provider-models",
    async (_event, providerId: string) => {
      const service = await getService();
      await service.refreshApiProviderModels(providerId);
      return refreshState();
    },
  );

  ipcMain.handle(
    "library:remove-api-provider",
    async (_event, providerId: string) => {
      const service = await getService();
      await service.removeApiProvider(providerId);
      return refreshState();
    },
  );

  ipcMain.handle("library:open-pdf", async (_event, paperId: string) => {
    const service = await getService();
    const paper = service.getPaperById(paperId);
    if (!paper) {
      throw new Error("Paper not found.");
    }

    await shell.openPath(paper.storedPdfPath);
    return true;
  });

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});
