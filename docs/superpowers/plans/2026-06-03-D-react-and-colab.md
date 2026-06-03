# Sub-project D — React frontend + Colab notebook bridge

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Streamlit demo at `003-deepCab-website/` with a Vite + React + TypeScript SPA (three pages: Predict, Explain, Runs). Add a Colab notebook (`notebooks/colab-train-and-push.ipynb`) that exposes a Colab GPU kernel via ngrok so VS Code can attach and run training cells against Colab's hardware.

**Architecture:** React app fetches typed schemas from `/openapi.json` (script — not yet wired into CI). Dev: Vite on `:5173` fronted by Traefik at `app.deepcab.localhost`, HMR over WS. Prod: built to `dist/`, published to `gh-pages` branch by a GitHub Actions workflow. Colab notebook is a 6-cell script: setup → auth → ngrok+jupyter → train → GCS push → trigger Cloud Run deploy.

**Tech Stack:** Vite 5, React 18, TypeScript 5, Tailwind CSS 3, `openapi-typescript`, nginx (prod static), `pyngrok`, `jupyter_server`.

**Reference:** [Design spec §6, §7](../specs/2026-06-03-deepcab-gcp-infra-and-audit-design.md).

**Prerequisite:** Sub-projects A + B + C landed (Traefik routes `app.deepcab.localhost`; CI variables set).

---

## File map

### Part 1 — React frontend (`003-deepCab-website/`)

| Action | Path | Purpose |
|---|---|---|
| Delete | `003-deepCab-website/app.py` | Streamlit app removed |
| Delete | `003-deepCab-website/requirements.txt` | Streamlit deps removed |
| Delete | `003-deepCab-website/Dockerfile` (if Streamlit-specific) | replaced |
| Create | `003-deepCab-website/package.json` | npm manifest |
| Create | `003-deepCab-website/vite.config.ts` | Vite + React + Tailwind |
| Create | `003-deepCab-website/tsconfig.json` | strict TS |
| Create | `003-deepCab-website/tsconfig.node.json` | for vite.config |
| Create | `003-deepCab-website/tailwind.config.ts` | Tailwind |
| Create | `003-deepCab-website/postcss.config.js` | Tailwind plugin |
| Create | `003-deepCab-website/index.html` | SPA entry |
| Create | `003-deepCab-website/src/main.tsx` | mount |
| Create | `003-deepCab-website/src/App.tsx` | router |
| Create | `003-deepCab-website/src/index.css` | Tailwind directives |
| Create | `003-deepCab-website/src/api/client.ts` | typed fetch |
| Create | `003-deepCab-website/src/api/schemas.ts` | generated (placeholder, script regenerates) |
| Create | `003-deepCab-website/src/pages/Predict.tsx` | form → POST /predict |
| Create | `003-deepCab-website/src/pages/Explain.tsx` | bar chart of 5 SHAP groups |
| Create | `003-deepCab-website/src/pages/Runs.tsx` | runs list |
| Create | `003-deepCab-website/src/components/{Card,Field,StatusBadge}.tsx` | reusables |
| Create | `003-deepCab-website/src/lib/format.ts` | small utils |
| Create | `003-deepCab-website/Dockerfile` | multi-stage: node:20-alpine → nginx:alpine |
| Create | `003-deepCab-website/nginx.conf` | SPA fallback |
| Create | `003-deepCab-website/.env.example` | `VITE_API_BASE_URL=https://api.deepcab.localhost` |
| Create | `003-deepCab-website/.gitignore` | dist/, node_modules/ |
| Modify | `003-deepCab-website/README.md` | rewrite |
| Modify | `003-deepCab-website/Makefile` | rewrite |

| Modify | `001-deepCab-api/infra/compose/docker-compose.dev.yml` | add `react-dev` service |
| Create | `.github/workflows/deploy-frontend-gh-pages.yml` | publish dist/ to gh-pages |

### Part 2 — Colab notebook (`001-deepCab-api/notebooks/`)

| Action | Path | Purpose |
|---|---|---|
| Create | `001-deepCab-api/notebooks/colab-train-and-push.ipynb` | 6-cell notebook |
| Modify | `001-deepCab-api/notebooks/README.md` | document the new notebook + VS Code attach steps |
| Modify | `001-deepCab-api/Makefile` | add `make colab_kernel` (prints the steps) |
| Modify | `001-deepCab-api/pyproject.toml` | add `pyngrok` to optional `[project.optional-dependencies] notebook` |

### Part 3 — Cross-cutting

| Modify | `001-deepCab-api/CLAUDE.md` | React swap + notebook bridge |
| Modify | `005-products/CLAUDE.md` | 003-deepCab-website is now React (one-liner) |
| Modify | `005-products/DOCS.md` | same one-liner if file exists |

---

## Task D1: Scaffold React app (Vite + React + TS + Tailwind)

**Files:** see "File map → Part 1" above.

- [ ] **Step 1: Remove Streamlit artifacts**

```bash
cd 003-deepCab-website
rm -f app.py requirements.txt
# Keep Makefile + README.md; we'll overwrite them next.
git add -A
git status --short
```

- [ ] **Step 2: Write `package.json`**

```json
{
  "name": "deepcab-website",
  "private": true,
  "version": "0.0.1",
  "type": "module",
  "scripts": {
    "dev": "vite --host 0.0.0.0 --port 5173",
    "build": "tsc -b && vite build",
    "preview": "vite preview --host 0.0.0.0 --port 4173",
    "lint": "tsc -b --noEmit",
    "gen:types": "openapi-typescript ${VITE_API_BASE_URL:-http://localhost:8000}/openapi.json -o src/api/schemas.ts"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-router-dom": "^6.26.2",
    "recharts": "^2.13.0"
  },
  "devDependencies": {
    "@types/react": "^18.3.10",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.1",
    "autoprefixer": "^10.4.20",
    "openapi-typescript": "^7.4.1",
    "postcss": "^8.4.47",
    "tailwindcss": "^3.4.13",
    "typescript": "^5.6.2",
    "vite": "^5.4.8"
  }
}
```

- [ ] **Step 3: Write `vite.config.ts`**

```ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    hmr: { clientPort: 443 },          // HMR through Traefik TLS
    strictPort: true,
  },
});
```

- [ ] **Step 4: Write `tsconfig.json` and `tsconfig.node.json`**

`tsconfig.json`:
```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "baseUrl": "src",
    "paths": { "@/*": ["./*"] }
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

`tsconfig.node.json`:
```json
{
  "compilerOptions": {
    "composite": true,
    "skipLibCheck": true,
    "module": "ESNext",
    "moduleResolution": "bundler",
    "allowSyntheticDefaultImports": true
  },
  "include": ["vite.config.ts"]
}
```

- [ ] **Step 5: Write `tailwind.config.ts` and `postcss.config.js`**

`tailwind.config.ts`:
```ts
import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: { extend: {} },
  plugins: [],
} satisfies Config;
```

`postcss.config.js`:
```js
export default {
  plugins: { tailwindcss: {}, autoprefixer: {} },
};
```

- [ ] **Step 6: Write `index.html`**

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>deepCab</title>
  </head>
  <body class="bg-slate-50 text-slate-900">
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 7: Write `src/main.tsx`, `src/App.tsx`, `src/index.css`**

`src/index.css`:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

`src/main.tsx`:
```tsx
import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route, NavLink, Navigate } from "react-router-dom";
import "./index.css";
import App from "./App";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>
);
```

`src/App.tsx`:
```tsx
import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import Predict from "@/pages/Predict";
import Explain from "@/pages/Explain";
import Runs from "@/pages/Runs";

const navLink = "px-3 py-2 rounded-md text-sm font-medium";
const active = "bg-slate-900 text-white";
const inactive = "text-slate-700 hover:bg-slate-200";

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b bg-white">
        <nav className="max-w-5xl mx-auto px-6 py-3 flex items-center gap-4">
          <span className="font-bold tracking-tight">deepCab</span>
          <NavLink end to="/predict" className={({ isActive }) => `${navLink} ${isActive ? active : inactive}`}>Predict</NavLink>
          <NavLink to="/explain" className={({ isActive }) => `${navLink} ${isActive ? active : inactive}`}>Explain</NavLink>
          <NavLink to="/runs" className={({ isActive }) => `${navLink} ${isActive ? active : inactive}`}>Runs</NavLink>
          <span className="ml-auto text-xs text-slate-500">
            api: {import.meta.env.VITE_API_BASE_URL}
          </span>
        </nav>
      </header>
      <main className="flex-1 max-w-5xl w-full mx-auto px-6 py-8">
        <Routes>
          <Route path="/" element={<Navigate to="/predict" replace />} />
          <Route path="/predict" element={<Predict />} />
          <Route path="/explain" element={<Explain />} />
          <Route path="/runs" element={<Runs />} />
        </Routes>
      </main>
    </div>
  );
}
```

- [ ] **Step 8: Write `src/api/client.ts` + `src/api/schemas.ts` placeholder**

`src/api/client.ts`:
```ts
const BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(public status: number, public body: string) {
    super(`API ${status}: ${body.slice(0, 200)}`);
  }
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init.headers ?? {}),
    },
  });
  const text = await r.text();
  if (!r.ok) throw new ApiError(r.status, text);
  return text ? (JSON.parse(text) as T) : (undefined as T);
}
```

`src/api/schemas.ts` (placeholder — `npm run gen:types` overwrites later):
```ts
export type PredictRequest = {
  pickup_datetime: string;
  pickup_longitude: number;
  pickup_latitude: number;
  dropoff_longitude: number;
  dropoff_latitude: number;
  passenger_count: number;
};

export type PredictResponse = {
  prediction: number;
  p_lo: number | null;
  p_hi: number | null;
  request_id: string;
};

export type ExplainSummary = {
  groups: { name: string; importance: number }[];
  run_id: string;
};

export type RunSummary = {
  run_id: string;
  backend: string;
  status: "running" | "success" | "failed";
  val_mae: number | null;
  created_at: string;
};
```

- [ ] **Step 9: Write `src/components/{Card,Field,StatusBadge}.tsx`**

`Card.tsx`:
```tsx
import { PropsWithChildren } from "react";
export default function Card({ children, title }: PropsWithChildren<{ title?: string }>) {
  return (
    <section className="bg-white rounded-lg shadow-sm border p-6">
      {title && <h2 className="text-lg font-semibold mb-4">{title}</h2>}
      {children}
    </section>
  );
}
```

`Field.tsx`:
```tsx
import { ChangeEvent } from "react";
type Props = {
  label: string;
  type?: "number" | "datetime-local" | "text";
  value: string | number;
  step?: string;
  onChange: (v: string) => void;
};
export default function Field({ label, type = "number", value, step, onChange }: Props) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-sm font-medium text-slate-700">{label}</span>
      <input
        className="border rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-slate-700"
        type={type}
        value={value}
        step={step}
        onChange={(e: ChangeEvent<HTMLInputElement>) => onChange(e.target.value)}
      />
    </label>
  );
}
```

`StatusBadge.tsx`:
```tsx
const colors: Record<string, string> = {
  running: "bg-amber-100 text-amber-800",
  success: "bg-emerald-100 text-emerald-800",
  failed: "bg-red-100 text-red-800",
};
export default function StatusBadge({ status }: { status: string }) {
  return (
    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${colors[status] ?? "bg-slate-100 text-slate-700"}`}>
      {status}
    </span>
  );
}
```

- [ ] **Step 10: Write `src/lib/format.ts`**

```ts
export const usd = (n: number) =>
  n.toLocaleString("en-US", { style: "currency", currency: "USD" });

export const ts = (s: string) => new Date(s).toLocaleString();
```

---

## Task D2: Pages

**Files:**
- Create: `003-deepCab-website/src/pages/{Predict,Explain,Runs}.tsx`

- [ ] **Step 1: `Predict.tsx`**

```tsx
import { useState } from "react";
import Card from "@/components/Card";
import Field from "@/components/Field";
import { api, ApiError } from "@/api/client";
import type { PredictRequest, PredictResponse } from "@/api/schemas";
import { usd } from "@/lib/format";

const defaults: PredictRequest = {
  pickup_datetime: "2014-07-06T17:18:00",
  pickup_longitude: -73.985,
  pickup_latitude: 40.748,
  dropoff_longitude: -73.961,
  dropoff_latitude: 40.762,
  passenger_count: 1,
};

export default function Predict() {
  const [form, setForm] = useState<PredictRequest>(defaults);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  const update = <K extends keyof PredictRequest>(k: K, v: string) =>
    setForm((f) => ({ ...f, [k]: k === "pickup_datetime" ? v : Number(v) }));

  const submit = async () => {
    setPending(true);
    setError(null);
    try {
      const r = await api<PredictResponse>("/predict", {
        method: "POST",
        body: JSON.stringify({ ...form, key: `${form.pickup_datetime}` }),
      });
      setResult(r);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setPending(false);
    }
  };

  return (
    <div className="space-y-6">
      <Card title="Predict fare">
        <div className="grid grid-cols-2 gap-4">
          <Field label="Pickup datetime" type="datetime-local" value={form.pickup_datetime} onChange={(v) => update("pickup_datetime", v)} />
          <Field label="Passengers" value={form.passenger_count} step="1" onChange={(v) => update("passenger_count", v)} />
          <Field label="Pickup lon" value={form.pickup_longitude} step="0.0001" onChange={(v) => update("pickup_longitude", v)} />
          <Field label="Pickup lat" value={form.pickup_latitude} step="0.0001" onChange={(v) => update("pickup_latitude", v)} />
          <Field label="Dropoff lon" value={form.dropoff_longitude} step="0.0001" onChange={(v) => update("dropoff_longitude", v)} />
          <Field label="Dropoff lat" value={form.dropoff_latitude} step="0.0001" onChange={(v) => update("dropoff_latitude", v)} />
        </div>
        <button
          onClick={submit}
          disabled={pending}
          className="mt-6 bg-slate-900 text-white px-4 py-2 rounded-md disabled:opacity-50"
        >
          {pending ? "Predicting…" : "Predict"}
        </button>
      </Card>

      {error && (
        <Card title="Error"><pre className="text-sm text-red-700 whitespace-pre-wrap">{error}</pre></Card>
      )}

      {result && (
        <Card title="Result">
          <div className="text-3xl font-semibold">{usd(result.prediction)}</div>
          {result.p_lo !== null && result.p_hi !== null && (
            <div className="text-sm text-slate-500">
              95% interval: {usd(result.p_lo)} — {usd(result.p_hi)}
            </div>
          )}
          <div className="mt-2 text-xs text-slate-400">request_id: {result.request_id}</div>
        </Card>
      )}
    </div>
  );
}
```

- [ ] **Step 2: `Explain.tsx`**

```tsx
import { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import Card from "@/components/Card";
import { api } from "@/api/client";
import type { ExplainSummary } from "@/api/schemas";

export default function Explain() {
  const [data, setData] = useState<ExplainSummary | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api<ExplainSummary>("/explain/summary?run_id=LATEST")
      .then(setData)
      .catch((e) => setErr(String(e)));
  }, []);

  if (err) return <Card title="Explain"><pre className="text-red-700 text-sm">{err}</pre></Card>;
  if (!data) return <Card title="Explain">Loading…</Card>;

  const sorted = [...data.groups].sort((a, b) => b.importance - a.importance);

  return (
    <Card title={`SHAP groups — run ${data.run_id.slice(0, 8)}`}>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={sorted} layout="vertical" margin={{ left: 80 }}>
          <XAxis type="number" />
          <YAxis dataKey="name" type="category" width={120} />
          <Tooltip />
          <Bar dataKey="importance" fill="#0f172a" />
        </BarChart>
      </ResponsiveContainer>
    </Card>
  );
}
```

- [ ] **Step 3: `Runs.tsx`**

```tsx
import { useEffect, useState } from "react";
import Card from "@/components/Card";
import StatusBadge from "@/components/StatusBadge";
import { api } from "@/api/client";
import type { RunSummary } from "@/api/schemas";
import { ts } from "@/lib/format";

export default function Runs() {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api<RunSummary[]>("/train")
      .then(setRuns)
      .catch((e) => setErr(String(e)));
  }, []);

  if (err) return <Card title="Runs"><pre className="text-red-700 text-sm">{err}</pre></Card>;
  if (!runs) return <Card title="Runs">Loading…</Card>;

  return (
    <Card title="Recent training runs">
      <table className="w-full text-sm">
        <thead className="text-left text-slate-500 border-b">
          <tr>
            <th className="py-2">run_id</th>
            <th>backend</th>
            <th>status</th>
            <th>val_mae</th>
            <th>created</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => (
            <tr key={r.run_id} className="border-b last:border-0">
              <td className="py-2 font-mono text-xs">{r.run_id.slice(0, 8)}</td>
              <td>{r.backend}</td>
              <td><StatusBadge status={r.status} /></td>
              <td>{r.val_mae?.toFixed(3) ?? "—"}</td>
              <td className="text-slate-500">{ts(r.created_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}
```

---

## Task D3: Dockerfile + nginx + .env + .gitignore

**Files:**
- Create: `003-deepCab-website/Dockerfile`
- Create: `003-deepCab-website/nginx.conf`
- Create: `003-deepCab-website/.env.example`
- Create: `003-deepCab-website/.gitignore`

- [ ] **Step 1: `Dockerfile` (multi-stage)**

```dockerfile
# Build stage
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci || npm install
COPY . .
ARG VITE_API_BASE_URL
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}
RUN npm run build

# Runtime stage
FROM nginx:alpine AS runtime
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

- [ ] **Step 2: `nginx.conf` (SPA fallback)**

```nginx
server {
  listen 80;
  server_name _;
  root /usr/share/nginx/html;
  index index.html;
  location / {
    try_files $uri $uri/ /index.html;
  }
}
```

- [ ] **Step 3: `.env.example`**

```
VITE_API_BASE_URL=https://api.deepcab.localhost
```

- [ ] **Step 4: `.gitignore`**

```
node_modules
dist
.env
.env.local
```

---

## Task D4: Rewrite Makefile + README

**Files:**
- Modify: `003-deepCab-website/Makefile`
- Modify: `003-deepCab-website/README.md`

- [ ] **Step 1: `Makefile`**

```make
.PHONY: install dev build preview gen_types docker_build clean

install:
	npm install

dev:
	npm run dev

build:
	npm run build

preview:
	npm run preview

gen_types:
	npm run gen:types

docker_build:
	docker build -t deepcab-website:dev --build-arg VITE_API_BASE_URL=$$VITE_API_BASE_URL .

clean:
	rm -rf dist node_modules
```

- [ ] **Step 2: `README.md`**

```markdown
# deepCab — website (React)

Vite + React + TypeScript + Tailwind SPA. Three pages:

- **Predict** — form → `POST /predict` against deepCab API; shows prediction + 95% prediction interval.
- **Explain** — `GET /explain/summary` → bar chart of 5 SHAP groups (passenger, pickup_datetime, distance, pickup_location, dropoff_location).
- **Runs** — `GET /train` → list of recent training runs with backend / status / val_mae.

## Develop

```bash
cp .env.example .env
make install
make dev   # http://localhost:5173 or https://app.deepcab.localhost (via Traefik)
```

The API must be reachable at `VITE_API_BASE_URL`. The quickest path is `make docker_up` in `../001-deepCab-api/` to bring up the full stack with Traefik routing `api.deepcab.localhost` → the FastAPI container.

## Regenerate API types

```bash
make gen_types   # curl /openapi.json → src/api/schemas.ts via openapi-typescript
```

## Build for production

```bash
make build       # → dist/
```

`dist/` is published to the `gh-pages` branch by `.github/workflows/deploy-frontend-gh-pages.yml` on every push to `main` that touches this directory. The runtime API URL is injected at build time via the `GCP_API_URL` GitHub variable.

## Build a container

```bash
make docker_build VITE_API_BASE_URL=https://api.deepcab.com
```

Multi-stage build: `node:20-alpine` build → `nginx:alpine` runtime with SPA fallback. Useful if you'd rather front the React app with Traefik in the same compose stack instead of using GitHub Pages.
```

---

## Task D5: Add react-dev service to dev compose

**Files:**
- Modify: `001-deepCab-api/infra/compose/docker-compose.dev.yml`

- [ ] **Step 1: Append the service**

```yaml
  react-dev:
    image: node:20-alpine
    working_dir: /app
    command: sh -c "npm install && npm run dev"
    environment:
      VITE_API_BASE_URL: https://api.deepcab.localhost
    volumes:
      - ../../../003-deepCab-website:/app
      - react_node_modules:/app/node_modules
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.react.rule=Host(`app.deepcab.localhost`)"
      - "traefik.http.routers.react.tls=true"
      - "traefik.http.services.react.loadbalancer.server.port=5173"
```

Add `react_node_modules:` to the top-level volumes block.

- [ ] **Step 2: Verify**

```bash
docker compose -f infra/compose/docker-compose.yml -f infra/compose/docker-compose.obs.yml -f infra/compose/docker-compose.dev.yml config --quiet && echo OK
```

---

## Task D6: GitHub Pages deploy workflow

**Files:**
- Create: `.github/workflows/deploy-frontend-gh-pages.yml`

- [ ] **Step 1: Write the workflow**

```yaml
name: deploy-frontend-gh-pages

on:
  push:
    branches: [main]
    paths:
      - '003-deepCab-website/**'
      - '.github/workflows/deploy-frontend-gh-pages.yml'
  workflow_dispatch:

permissions:
  contents: write   # to push to gh-pages branch

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    defaults:
      run: { working-directory: 003-deepCab-website }
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
          cache-dependency-path: 003-deepCab-website/package-lock.json

      - run: npm install

      - name: Build
        env:
          VITE_API_BASE_URL: ${{ vars.GCP_API_URL || 'http://localhost:8000' }}
        run: npm run build

      - name: Deploy to gh-pages
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: 003-deepCab-website/dist
          publish_branch: gh-pages
```

`GCP_API_URL` is a GitHub variable set after Cloud Run deploys (e.g. `https://deepcab-api-xxxx.a.run.app`). Until then the fallback is `http://localhost:8000`.

---

## Task D7: Verify React app boots

**Files:** none — manual verification.

- [ ] **Step 1: Local install + dev**

```bash
cd 003-deepCab-website
cp .env.example .env
make install
```

Expected: `node_modules/` populated, no errors.

- [ ] **Step 2: Type-check**

```bash
make build
```

Expected: `dist/` populated, no TS errors.

- [ ] **Step 3: Run dev server**

```bash
make dev &
sleep 5
curl -s http://localhost:5173/ | grep -q 'id="root"' && echo "OK"
kill %1
```

Expected: `OK`.

- [ ] **Step 4: Run via Traefik dev compose**

In `001-deepCab-api/`:

```bash
make docker_dev_up
sleep 30
curl -k https://app.deepcab.localhost/ | grep -q 'id="root"' && echo "OK"
```

Expected: `OK`.

If the React app needs the API live (Explain, Runs queries fail until you train a model), that's expected — Predict will still work against a freshly trained 1k model.

- [ ] **Step 5: Tear down**

```bash
make docker_dev_down
```

---

## Task D8: Colab notebook

**Files:**
- Create: `001-deepCab-api/notebooks/colab-train-and-push.ipynb`
- Modify: `001-deepCab-api/notebooks/README.md`
- Modify: `001-deepCab-api/Makefile`
- Modify: `001-deepCab-api/pyproject.toml`

- [ ] **Step 1: Add `pyngrok` to optional deps**

In `pyproject.toml`, find `[project.optional-dependencies]`. Add (or create) a `notebook` extra:

```toml
notebook = [
  "pyngrok>=7.1.6",
  "jupyter-server>=2.14",
]
```

If a `dev` extra already exists, leave it alone — this is separate so Colab can install it standalone.

- [ ] **Step 2: Write the notebook**

Build `colab-train-and-push.ipynb` as a JSON file with six cells. (Write the JSON directly with the Write tool; do not use `nbformat` from the shell since this is a one-time file. The skeleton:)

```json
{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# deepCab — Colab training + VS Code attach\n",
        "\n",
        "Open this notebook on Colab (Runtime → Change runtime type → GPU). Run cells 1–3, then attach VS Code to the printed URL via *Cmd+Shift+P → Jupyter: Specify Jupyter Server for Connections*. From cell 4 onward, run cells from VS Code — code executes on Colab's GPU, edit happens locally.\n",
        "\n",
        "**Colab secrets needed** (Tools → Secrets):\n",
        "- `NGROK_AUTHTOKEN` — from https://dashboard.ngrok.com\n",
        "- `GH_TOKEN` — fine-grained PAT with `Actions: write` for the deepCab repo\n",
        "- `GCP_PROJECT` — your GCP project id\n",
        "- `GH_REPO` — `<owner>/<name>` (e.g. `juan-garassino/deepCab`)"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "# Cell 1 — setup\n",
        "!pip install -q pyngrok jupyter_server\n",
        "!git clone -q https://github.com/$GH_REPO_NAME.git /content/deepCab || (cd /content/deepCab && git pull -q)\n",
        "%cd /content/deepCab/001-deepCab-api\n",
        "!pip install -q -e ."
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "# Cell 2 — auth\n",
        "from google.colab import auth, userdata\n",
        "auth.authenticate_user()\n",
        "NGROK_AUTHTOKEN = userdata.get('NGROK_AUTHTOKEN')\n",
        "GH_TOKEN = userdata.get('GH_TOKEN')\n",
        "GCP_PROJECT = userdata.get('GCP_PROJECT')\n",
        "GH_REPO = userdata.get('GH_REPO')\n",
        "assert all([NGROK_AUTHTOKEN, GH_TOKEN, GCP_PROJECT, GH_REPO]), 'set all 4 Colab secrets first'"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "# Cell 3 — start jupyter server + ngrok tunnel\n",
        "import secrets as _s, subprocess, time\n",
        "from pyngrok import ngrok\n",
        "TOKEN = _s.token_urlsafe(24)\n",
        "ngrok.set_auth_token(NGROK_AUTHTOKEN)\n",
        "tunnel = ngrok.connect(8888, 'http')\n",
        "subprocess.Popen([\n",
        "    'jupyter', 'server',\n",
        "    '--ip=0.0.0.0', '--port=8888', '--no-browser',\n",
        "    f'--ServerApp.token={TOKEN}',\n",
        "    '--ServerApp.allow_origin=*',\n",
        "    '--ServerApp.disable_check_xsrf=True',\n",
        "])\n",
        "time.sleep(4)\n",
        "url = f'{tunnel.public_url}?token={TOKEN}'\n",
        "print('VS Code → Jupyter: Specify Server →', url)\n",
        "print('Then open this notebook locally and pick that server as the kernel.')"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "# Cell 4 — train on Colab GPU (run from VS Code attached)\n",
        "from deepCab.training.train import run\n",
        "from deepCab.schemas.config import TrainConfig, TorchMLPConfig, DataRef\n",
        "result = run(TrainConfig(\n",
        "    backend=TorchMLPConfig(epochs=50, lr=1e-3, batch_size=512),\n",
        "    data=DataRef(size='full'),\n",
        "))\n",
        "print(result.metrics)\n",
        "RUN_ID = result.run_id"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "# Cell 5 — push artifact to GCS\n",
        "import subprocess\n",
        "subprocess.run(['gsutil', '-m', 'cp', '-r', f'runs/{RUN_ID}', f'gs://deepcab-models/runs/{RUN_ID}/'], check=True)\n",
        "MODEL_URI = f'gs://deepcab-models/runs/{RUN_ID}/'\n",
        "print('uploaded:', MODEL_URI)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "# Cell 6 — trigger Cloud Run deploy\n",
        "import requests\n",
        "r = requests.post(\n",
        "    f'https://api.github.com/repos/{GH_REPO}/actions/workflows/deploy-cloud-run.yml/dispatches',\n",
        "    headers={'Authorization': f'token {GH_TOKEN}', 'Accept': 'application/vnd.github+json'},\n",
        "    json={'ref': 'main', 'inputs': {'tag': RUN_ID, 'model_uri': MODEL_URI}},\n",
        ")\n",
        "r.raise_for_status()\n",
        "print('deploy triggered for', MODEL_URI)"
      ],
      "execution_count": null,
      "outputs": []
    }
  ],
  "metadata": {
    "kernelspec": { "display_name": "Python 3", "language": "python", "name": "python3" },
    "language_info": { "name": "python", "version": "3.11" },
    "colab": { "provenance": [] }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
```

(Write the full JSON above with the Write tool; do not eval-print or shell-here it.)

- [ ] **Step 3: Verify nbformat**

```bash
uv run python -c "import json; nb = json.load(open('notebooks/colab-train-and-push.ipynb')); assert nb['nbformat']==4 and len(nb['cells'])==7"
```

Expected: no error (7 cells = 1 markdown + 6 code).

- [ ] **Step 4: Update `notebooks/README.md`**

Append:

```markdown
## `colab-train-and-push.ipynb`

Train on Colab's GPU with the editor still on your laptop:

1. Open the notebook on Colab (https://colab.research.google.com/github/<owner>/<repo>/blob/main/001-deepCab-api/notebooks/colab-train-and-push.ipynb). Switch to GPU runtime.
2. Set 4 Colab secrets (Tools → Secrets): `NGROK_AUTHTOKEN`, `GH_TOKEN`, `GCP_PROJECT`, `GH_REPO`.
3. Run cells 1–3. Cell 3 prints a URL.
4. In VS Code: `Cmd+Shift+P` → *Jupyter: Specify Jupyter Server for Connections* → paste the URL from cell 3.
5. Open this notebook locally. Use the kernel picker to choose the remote ngrok server.
6. Run cells 4–6 from VS Code. Cell 4 trains on Colab's GPU; cell 5 pushes to GCS; cell 6 triggers `deploy-cloud-run.yml`.

If the Colab runtime disconnects, re-run cell 3 to get a new ngrok URL (free-tier ngrok URLs are ephemeral).
```

- [ ] **Step 5: Add `make colab_kernel` target**

In `Makefile`:

```make
colab_kernel:
	@echo "1. Open https://colab.research.google.com/github/<owner>/<repo>/blob/main/001-deepCab-api/notebooks/colab-train-and-push.ipynb"
	@echo "2. Switch runtime to GPU; set Colab secrets NGROK_AUTHTOKEN, GH_TOKEN, GCP_PROJECT, GH_REPO"
	@echo "3. Run cells 1-3 on Colab; copy the printed URL"
	@echo "4. VS Code: Cmd+Shift+P → 'Jupyter: Specify Jupyter Server for Connections' → paste URL"
	@echo "5. Open notebooks/colab-train-and-push.ipynb locally; pick the remote server as kernel"
	@echo "See notebooks/README.md for the full walkthrough."
```

---

## Task D9: Update top-level docs

**Files:**
- Modify: `001-deepCab-api/CLAUDE.md`
- Modify: `005-products/CLAUDE.md`
- Modify: `005-products/DOCS.md` (if exists)

- [ ] **Step 1: `001-deepCab-api/CLAUDE.md` — Project Directory Layout**

In the table that lists the three packages, change the `003-deepCab-website/` row:

```markdown
| `003-deepCab-website/` | Vite + React + TypeScript SPA. Three pages: Predict, Explain, Runs. Deployed to GitHub Pages via CI; available locally at `app.deepcab.localhost` through Traefik. |
```

Add a "Colab notebook bridge" subsection under "Common commands":

```markdown
## Colab notebook bridge

`notebooks/colab-train-and-push.ipynb` exposes a Colab GPU kernel via ngrok so VS Code can attach as a remote kernel. Run `make colab_kernel` for the step-by-step. The notebook ends by triggering `deploy-cloud-run.yml` with the trained model URI.
```

- [ ] **Step 2: `005-products/CLAUDE.md` — Project Directory Layout**

Find the deepCab line and update the `003-` package descriptor: "Streamlit demo" → "Vite + React SPA (Predict / Explain / Runs)".

- [ ] **Step 3: `005-products/DOCS.md`**

```bash
test -f /Users/juan-garassino/Code/005-products/DOCS.md && echo "edit it" || echo "skip — file does not exist"
```

If it exists, apply the same one-line update.

---

## Task D10: Commit Sub-project D

- [ ] **Step 1: Commit**

```bash
cd /Users/juan-garassino/Code/005-products/006-deep-projects/001-deepCab/001-deepCab-api
git add -A
git status --short
git commit -m "$(cat <<'EOF'
feat(frontend+colab): React SPA replaces Streamlit; Colab notebook bridge

003-deepCab-website/ is now a Vite + React + TypeScript + Tailwind SPA:
- Three pages: Predict (form → POST /predict), Explain (5 SHAP groups bar
  chart), Runs (recent training run list).
- Dev: react-dev container in docker-compose.dev.yml; Traefik routes
  app.deepcab.localhost → vite :5173 with HMR over TLS.
- Prod: dist/ published to gh-pages by deploy-frontend-gh-pages.yml.
- Dockerfile is multi-stage (node:20-alpine build → nginx:alpine runtime).

notebooks/colab-train-and-push.ipynb — 6-cell notebook:
1) install + clone repo
2) Colab auth + secrets
3) start jupyter + ngrok tunnel; print URL for VS Code attach
4) train on Colab GPU (TorchMLP, 50 epochs)
5) gsutil push runs/<id>/ to gs://deepcab-models/
6) workflow_dispatch deploy-cloud-run.yml with the model URI

pyproject.toml: notebook extra (pyngrok, jupyter-server).
Makefile: make colab_kernel prints the VS Code attach steps.

Sub-project D of the GCP infra design.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Done criteria

- [x] Streamlit `app.py` deleted; React + Vite scaffold in place
- [x] Three pages render: Predict (form), Explain (chart), Runs (table)
- [x] `npm run build` produces `dist/` with no TS errors
- [x] `npm run dev` serves at :5173 and via Traefik at `app.deepcab.localhost`
- [x] `gen:types` regenerates `src/api/schemas.ts` from `/openapi.json`
- [x] Multi-stage `Dockerfile` builds a static-served image
- [x] `deploy-frontend-gh-pages.yml` publishes `dist/` to gh-pages
- [x] `colab-train-and-push.ipynb` has 7 cells (1 md + 6 code), valid JSON
- [x] `make colab_kernel` prints VS Code attach steps
- [x] CLAUDE.md (both levels) + DOCS.md updated
- [x] One commit
