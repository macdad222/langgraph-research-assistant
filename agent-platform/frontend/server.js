const http = require("node:http");
const https = require("node:https");
const fs = require("node:fs");
const path = require("node:path");
const querystring = require("node:querystring");

const port = Number(process.env.PORT || 8080);
const frontendPassword = process.env.FRONTEND_PASSWORD || "";
const apiProxyTarget = process.env.API_PROXY_TARGET || "http://agent.lab.internal:8001";
const apiProxyTimeoutMs = Number(process.env.API_PROXY_TIMEOUT_MS || 0);
const authCookie = "fortigate_frontend_auth=1";
const indexPath = path.join(__dirname, "index.html");
const fortigatePath = path.join(__dirname, "fortigate.html");
const networkDesignPath = path.join(__dirname, "network-design.html");
const inventoryPath = path.join(__dirname, "inventory.html");

function isAuthenticated(req) {
  return String(req.headers.cookie || "").split(";").map((item) => item.trim()).includes(authCookie);
}

function renderLogin(error = "") {
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>FortiGate Design Agent Login</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; min-height: 100vh; display: grid; place-items: center; background: radial-gradient(circle at top, #1e293b, #020617 58%); color: #e5e7eb; }
    form { width: min(420px, calc(100vw - 32px)); padding: 28px; border: 1px solid #334155; border-radius: 18px; background: rgba(15, 23, 42, 0.92); box-shadow: 0 24px 80px rgba(0, 0, 0, 0.35); }
    h1 { margin: 0 0 8px; font-size: 24px; }
    p { margin: 0 0 18px; color: #94a3b8; line-height: 1.5; }
    label { display: block; margin-bottom: 8px; color: #cbd5e1; font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; }
    input { width: 100%; box-sizing: border-box; border: 1px solid #475569; border-radius: 12px; padding: 13px 14px; background: #020617; color: #e5e7eb; font-size: 16px; }
    button { width: 100%; margin-top: 14px; border: 0; border-radius: 12px; padding: 13px 14px; background: #38bdf8; color: #082f49; font-weight: 800; cursor: pointer; }
    .error { margin-top: 12px; color: #fecaca; }
  </style>
</head>
<body>
  <form method="post" action="/login">
    <h1>FortiGate Design Agent</h1>
    <p>Enter the shared password to access the design tools.</p>
    <label for="password">Password</label>
    <input id="password" name="password" type="password" autocomplete="current-password" autofocus />
    <button type="submit">Unlock</button>
    ${error ? `<div class="error">${error}</div>` : ""}
  </form>
</body>
</html>`;
}

function proxyApi(req, res) {
  const target = new URL(apiProxyTarget);
  const upstreamPath = req.url.slice("/api".length) || "/";
  const upstreamUrl = new URL(upstreamPath, target);
  const client = upstreamUrl.protocol === "https:" ? https : http;
  const headers = { ...req.headers, host: upstreamUrl.host };
  delete headers.connection;
  delete headers["content-length"];
  let upstreamResponded = false;

  const proxyReq = client.request(
    upstreamUrl,
    {
      method: req.method,
      headers,
    },
    (proxyRes) => {
      upstreamResponded = true;
      res.writeHead(proxyRes.statusCode || 502, proxyRes.headers);
      proxyRes.pipe(res);
    },
  );

  proxyReq.on("error", (error) => {
    if (res.writableEnded || upstreamResponded) return;
    res.writeHead(502, { "content-type": "application/json" });
    res.end(JSON.stringify({ detail: `API proxy failed: ${error.message}` }));
  });

  if (apiProxyTimeoutMs > 0) {
    proxyReq.setTimeout(apiProxyTimeoutMs, () => {
      proxyReq.destroy(new Error(`API proxy timed out after ${apiProxyTimeoutMs}ms`));
    });
  }

  res.on("close", () => {
    if (!res.writableEnded) {
      proxyReq.destroy(new Error("Client disconnected before upstream response completed"));
    }
  });

  req.pipe(proxyReq);
}

const server = http.createServer((req, res) => {
  if (req.url === "/health") {
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ status: "ok" }));
    return;
  }

  if (req.url === "/login" && req.method === "GET") {
    res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
    res.end(renderLogin());
    return;
  }

  if (req.url === "/login" && req.method === "POST") {
    let body = "";
    req.on("data", (chunk) => {
      body += chunk.toString();
      if (body.length > 4096) req.destroy();
    });
    req.on("end", () => {
      const form = querystring.parse(body);
      if (frontendPassword && form.password === frontendPassword) {
        res.writeHead(302, {
          "set-cookie": `${authCookie}; Path=/; HttpOnly; SameSite=Lax`,
          location: "/network-design",
        });
        res.end();
        return;
      }
      res.writeHead(401, { "content-type": "text/html; charset=utf-8" });
      res.end(renderLogin("Incorrect password."));
    });
    return;
  }

  if (!isAuthenticated(req)) {
    res.writeHead(302, { location: "/login" });
    res.end();
    return;
  }

  if (req.url.startsWith("/api/") || req.url === "/api") {
    proxyApi(req, res);
    return;
  }

  if (req.url === "/" || req.url === "/index.html") {
    const html = fs.readFileSync(networkDesignPath, "utf8");
    res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
    res.end(html);
    return;
  }

  if (req.url === "/research") {
    const html = fs.readFileSync(indexPath, "utf8");
    res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
    res.end(html);
    return;
  }

  if (req.url === "/fortigate" || req.url === "/fortigate.html") {
    const html = fs.readFileSync(fortigatePath, "utf8");
    res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
    res.end(html);
    return;
  }

  if (req.url === "/network-design" || req.url === "/network-design.html") {
    const html = fs.readFileSync(networkDesignPath, "utf8");
    res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
    res.end(html);
    return;
  }

  if (req.url === "/inventory" || req.url === "/inventory.html") {
    const html = fs.readFileSync(inventoryPath, "utf8");
    res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
    res.end(html);
    return;
  }

  res.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
  res.end("Not found");
});

server.listen(port, "0.0.0.0", () => {
  console.log(`FortiGate design frontend listening on ${port}`);
});
