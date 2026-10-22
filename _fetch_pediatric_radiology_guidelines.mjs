import { writeFile } from "node:fs/promises";
import { createRequire } from "node:module";

const require = createRequire(
  "file:///C:/Users/TimothyW/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/",
);
const { chromium } = require("playwright-core");

const browser = await chromium.launch({
  headless: true,
  executablePath: "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
  args: [
    "--no-sandbox",
    "--disable-gpu",
    "--disable-dev-shm-usage",
    "--disable-extensions",
    "--no-first-run",
  ],
});

const page = await browser.newPage({ viewport: { width: 1440, height: 1200 } });
await page.goto(
  "https://link.springer.com/journal/247/submission-guidelines",
  { waitUntil: "domcontentloaded", timeout: 90000 },
);
await page.waitForTimeout(15000);

const html = await page.content();
const text = await page.locator("body").innerText();
await writeFile(
  "C:/Users/TimothyW/.codex/visualizations/2026/09/15/01a0a3dc-efbe-7b63-af76-3a94cd06cb6a/pediatric_radiology_guidelines_dom.html",
  html,
  "utf8",
);
await writeFile(
  "C:/Users/TimothyW/.codex/visualizations/2026/09/15/01a0a3dc-efbe-7b63-af76-3a94cd06cb6a/pediatric_radiology_guidelines.txt",
  text,
  "utf8",
);
console.log(`title=${await page.title()}`);
console.log(`text_length=${text.length}`);
console.log(text.slice(0, 30000));

await browser.close();
