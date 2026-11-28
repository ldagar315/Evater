// Actual auth UI and SDK. Intercept EVERY backend request. No real sign-in/signup/email.
import puppeteer from "puppeteer";
import { createServer } from "node:http";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { join, resolve, extname } from "node:path";
const output = resolve(
  process.env.QA_EVIDENCE || "/downloads/evater-auth-hosted-evidence",
);
await mkdir(output, { recursive: true });
const server = process.env.QA_ORIGIN
  ? null
  : createServer(async (req, res) => {
      const path = new URL(req.url, "http://local").pathname;
      let data;
      try {
        data = await readFile(join(process.cwd(), "dist", path));
      } catch {
        data = await readFile("dist/index.html");
      }
      res.setHeader(
        "Content-Type",
        {
          ".js": "application/javascript",
          ".css": "text/css",
          ".png": "image/png",
          ".svg": "image/svg+xml",
          ".html": "text/html",
        }[extname(path)] || "text/html",
      );
      res.end(data);
    });
if (server) await new Promise((r) => server.listen(0, "127.0.0.1", r));
const origin =
  process.env.QA_ORIGIN || `http://127.0.0.1:${server.address().port}`;
const browser = await puppeteer.launch({
  executablePath: "/usr/bin/google-chrome",
  headless: true,
  args: ["--no-sandbox", "--disable-dev-shm-usage"],
});
const records = [],
  errors = [],
  requests = [],
  unexpected = [];
const assert = (x, message) => {
  if (!x) throw Error(message);
};
try {
  for (const width of [320, 390, 1280]) {
    const context = await browser.createBrowserContext();
    const page = await context.newPage();
    await page.setViewport({ width, height: 900 });
    let scenario = "confirmation";
    page.on("pageerror", (e) => errors.push({ width, message: e.message }));
    await page.setRequestInterception(true);
    page.on("request", async (r) => {
      const u = new URL(r.url());
      if (u.hostname.endsWith(".supabase.co")) {
        if (u.pathname === "/favicon.ico") return r.respond({ status: 204 });
        requests.push({
          width,
          scenario,
          path: u.pathname,
          method: r.method(),
        });
        const headers = {
          "access-control-allow-origin": "*",
          "access-control-allow-headers": "*",
          "access-control-allow-methods": "GET,POST,OPTIONS",
        };
        if (r.method() === "OPTIONS")
          return r.respond({ status: 204, headers });
        if (u.pathname === "/auth/v1/authorize" && r.method() === "GET") {
          assert(
            u.searchParams.get("provider") === "google",
            "wrong OAuth provider",
          );
          assert(
            u.searchParams.get("redirect_to") === origin + "/home",
            "OAuth return changed",
          );
          return r.respond({
            status: 200,
            contentType: "text/html",
            body: "<h1>Intercepted Google handoff</h1>",
          });
        }
        if (u.pathname === "/auth/v1/signup" && r.method() === "POST")
          return r.respond({
            status: 200,
            headers,
            contentType: "application/json",
            body: JSON.stringify({
              id: "00000000-0000-4000-8000-000000000099",
              email: "fixture@example.invalid",
              identities: [],
              aud: "authenticated",
              role: "authenticated",
              created_at: new Date().toISOString(),
              app_metadata: { provider: "email" },
              user_metadata: {},
            }),
          });
        if (u.pathname === "/auth/v1/token" && r.method() === "POST")
          return r.respond({
            status: 400,
            headers,
            contentType: "application/json",
            body: JSON.stringify({
              code: "invalid_credentials",
              msg: "Invalid login credentials",
            }),
          });
        unexpected.push({ method: r.method(), path: u.pathname });
        return r.abort();
      }
      if (u.hostname === "app.netlify.com" && u.pathname === "/cdp/")
        return r.abort(); // Preview toolbar, not the app
      if (u.origin === origin && r.method() === "GET") return r.continue();
      if (
        u.protocol === "data:" ||
        [
          "fonts.googleapis.com",
          "fonts.gstatic.com",
          "images.pexels.com",
        ].includes(u.hostname)
      )
        return r.continue();
      unexpected.push({ method: r.method(), host: u.host, path: u.pathname });
      return r.abort();
    });
    const click = async (text) => {
      const handle = await page.evaluateHandle(
        (t) =>
          [...document.querySelectorAll("button")].find((b) =>
            b.textContent.includes(t),
          ),
        text,
      );
      assert(handle.asElement(), "Missing button " + text);
      await handle.asElement().click();
    };
    const open = async (mode) => {
      await page.goto(
        origin + "/auth" + (mode === "signup" ? "?mode=signup" : ""),
        { waitUntil: "networkidle0" },
      );
      await page.waitForSelector("input[type=email]");
    };
    const snap = async (name) =>
      page.screenshot({
        path: join(output, `${width}-${name}.png`),
        fullPage: true,
      });
    const test = async (name, run) => {
      try {
        await run();
        records.push({ width, name, status: "PASS" });
      } catch (e) {
        records.push({ width, name, status: "FAIL", reason: e.message });
      }
    };
    const overflow = async () =>
      assert(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
        "Horizontal overflow",
      );
    await test("sign-in-layout-labels-keyboard", async () => {
      await open("signin");
      await overflow();
      assert(
        await page.$eval("h1", (e) => e.textContent === "Welcome back."),
        "Wrong title",
      );
      assert(
        await page.$eval(
          "input[type=email]",
          (e) => !!document.querySelector(`label[for="${e.id}"]`),
        ),
        "Unlabelled email",
      );
      assert(
        await page.$eval(
          "input[type=password]",
          (e) => e.autocomplete === "current-password",
        ),
        "Missing password hint",
      );
      await page.keyboard.press("Tab");
      assert(
        await page.evaluate(
          () =>
            document.activeElement.getAttribute("aria-label") === "Evater home",
        ),
        "Keyboard start",
      );
      await snap("signin");
    });
    await test("signup-mobile-layout", async () => {
      await open("signup");
      await overflow();
      assert(
        (await page.$$("input[type=password]")).length === 2,
        "Missing confirm field",
      );
      await snap("signup");
    });
    await test("email-confirmation-gate-and-recovery", async () => {
      await page.type("input[type=email]", "fixture@example.invalid");
      for (const field of await page.$$("input[type=password]"))
        await field.type("fixture-password-only");
      await click("Create Your Account");
      await page.waitForSelector("[role=status]");
      await overflow();
      assert(new URL(page.url()).pathname === "/auth", "Premature redirect");
      assert(
        (await page.$$("input[type=password]")).length === 0,
        "Password retained",
      );
      await snap("confirmation");
      await click("Back to sign in");
      await page.waitForSelector("input[type=email]");
      assert(
        (await page.$$("input[type=password]")).length === 1,
        "Not back to sign in",
      );
    });
    await test("credentials-error-stays-visible", async () => {
      scenario = "credentials";
      await open("signin");
      await page.type("input[type=email]", "fixture@example.invalid");
      await page.type("input[type=password]", "fixture-password-only");
      await click("Sign In to Evater");
      await page.waitForSelector("[role=alert]");
      assert(
        await page.$eval("[role=alert]", (e) =>
          e.textContent.includes("New here?"),
        ),
        "Guidance lost",
      );
      await overflow();
      await snap("error");
    });
    await test("google-handoff-only-no-real-provider", async () => {
      scenario = "google";
      await open("signin");
      await click("Continue with Google");
      await page.waitForFunction(() =>
        document.body.innerText.includes("Intercepted Google handoff"),
      );
      assert(
        new URL(page.url()).pathname === "/auth/v1/authorize",
        "Wrong handoff",
      );
    });
    await context.close();
  }
} finally {
  await browser.close();
  if (server) await new Promise((r) => server.close(r));
  const result = {
    origin,
    records,
    errors,
    requests,
    unexpected,
    scope:
      "Actual app/router/SDK with all backend requests intercepted. Synthetic fixtures only; no real user, email, OAuth provider or profile write.",
  };
  await writeFile(
    join(output, "results.json"),
    JSON.stringify(result, null, 2),
  );
  console.log(JSON.stringify(result, null, 2));
  if (
    records.some((x) => x.status !== "PASS") ||
    errors.length ||
    unexpected.length
  )
    process.exitCode = 1;
}
