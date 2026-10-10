# Portfolio "Proof It Works" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a screener verify StreetLab works without building it: real CI with live badges, a downloadable unsigned `.app` release, a zero-install hosted demo, and accurate repo metadata.

**Architecture:** Two GitHub Actions workflows (`ci.yml` on Linux for every push/PR; `release.yml` on macOS for tags, producing a *draft* release). A `VITE_DEMO=1` build flag makes the existing in-process mock the default transport and shows a "scripted data" banner; that build is deployed to Vercel as a static site. Repo description/topics are set via `gh`. The README is updated last, once the real URLs and badges exist.

**Tech Stack:** GitHub Actions, `astral-sh/setup-uv`, Playwright, Tauri 2 / PyInstaller (existing `scripts/build_app.sh`), Vite, Vercel CLI, `gh`.

**Spec:** [`docs/superpowers/specs/2026-10-04-portfolio-credibility-design.md`](../specs/2026-10-04-portfolio-credibility-design.md), "Feature 2: Proof it works".
**Depends on:** [`2026-10-04-portfolio-front-door.md`](2026-10-04-portfolio-front-door.md) (the README's `## Quickstart` section and badge block must exist).

## Global Constraints

- Repo: `https://github.com/jasonpereira518/streetlab`. Default branch `main`. Python is `>=3.11,<3.12`, managed by `uv`.
- Measured 2026-10-04 at `8bdf8b9`: backend `uv run pytest -q tests ../contract` = **1198 passed, 1 skipped in ~9 minutes**; vitest **276 passed**; Playwright **21 tests in 5 files**.
- Three of five e2e specs (`faultInjection`, `location`, `restartSession`) spawn the real Python backend via `uv run` from `../../streetlab-backend`, so the e2e CI job needs `uv` and the backend dependencies.
- The `location.spec.ts` happy path needs live Nominatim/Overpass; it skips when `STREETLAB_OFFLINE` is set. CI sets `STREETLAB_OFFLINE=1`.
- The `.app` is **unsigned and not notarized**. No Apple Developer account is in scope.
- The hosted demo runs the in-process mock on scripted data, **not** the Python simulator, and must be labelled so on the page and in the README.
- Outward-facing actions need Jason's explicit yes in chat before they run: pushing the branch, pushing a tag, deploying to Vercel, and editing the GitHub repo description/topics/homepage. A release is created as a **draft** and is published by Jason, never by this plan.
- `vercel whoami` returned `tradersatcarolina-2608` on 2026-10-04, which may not be the account Jason wants for a public portfolio URL. Ask which account/team before deploying.
- No hard-coded test counts in new prose beyond a dated Testing line.
- Commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## File Structure

| File | Responsibility |
|---|---|
| `docs/measurements/2026-10-04-e2e-flake-rate.md` | Create. Recorded flake measurement for `app.spec.ts` after PR #20. |
| `streetlab/src/net/wsClient.ts` | Modify. Default to the mock when built with `VITE_DEMO=1`. |
| `streetlab/src/ui/DemoBanner.tsx` | Create. "Scripted data, not the real simulator" banner. |
| `streetlab/src/App.tsx` | Modify. Render the banner in demo builds. |
| `streetlab/src/styles.css` | Modify. Banner styling. |
| `streetlab/tests/wsClient.test.ts` | Modify. Demo-default test. |
| `streetlab/tests/demoBanner.test.tsx` | Create. Banner test. |
| `.github/workflows/ci.yml` | Create. Linux CI. |
| `.github/workflows/release.yml` | Create. macOS `.app` build + draft release. |
| `.github/release-notes.md` | Create. Gatekeeper and checksum instructions. |
| `streetlab/vercel.json` | Create. Static demo build config. |
| `streetlab/scripts/capture_social_preview.mjs` | Create. 1280×640 social image capture. |
| `docs/screenshots/social-preview.png` | Create. Repo social preview. |
| `README.md`, `docs/ARCHITECTURE.md` | Modify. Badge, quickstart links, truthful CI wording. |

---

### Task 1: Re-measure the e2e flake rate

**Files:**
- Create: `docs/measurements/2026-10-04-e2e-flake-rate.md`

**Interfaces:**
- Produces: a recorded pass/fail count that Task 3 uses to decide whether e2e can be a required check.

- [ ] **Step 1: Run the previously flaky spec repeatedly**

`app.spec.ts` uses `?mock=1` and needs no backend. From the worktree root:

```bash
cd /Users/jasonpereira/Projects/claude-worktrees/streetlab/portfolio-credibility-plan-cbca8d/streetlab
npm ci
npx playwright test e2e/app.spec.ts --repeat-each=10 --reporter=line 2>&1 | tee /tmp/e2e-flake.txt | tail -15
```

Expected: a final summary line such as `N passed` / `M failed`. Takes several minutes.

- [ ] **Step 2: Record the result with real values**

```bash
cd ..
COMMIT=$(git rev-parse --short HEAD)
SUMMARY=$(command grep -E '^\s+[0-9]+ (passed|failed|flaky)' /tmp/e2e-flake.txt | tr '\n' ';')
cat > docs/measurements/2026-10-04-e2e-flake-rate.md <<EOF
# e2e flake rate for app.spec.ts after PR #20

Measured 2026-10-04 on commit \`$COMMIT\`, local macOS, Chromium via Playwright.

Command: \`npx playwright test e2e/app.spec.ts --repeat-each=10\`

Result: $SUMMARY

Context: \`app.spec.ts:42\` (the pause-clock test) previously failed on main most
runs; PR #20 made the test wait for the throttled readout to settle. This run
re-measures after that fix. CI treats e2e as required only if every repeat passes.
EOF
cat docs/measurements/2026-10-04-e2e-flake-rate.md
```

Expected: the file shows real counts, not blanks.

- [ ] **Step 3: Decide and note the rule**

If the summary shows any `failed`, do not make e2e a required check in Task 3: leave the job in `ci.yml` but note in Task 3 Step 6 that it is advisory. If all 10 repeats passed, e2e may be required.

- [ ] **Step 4: Commit**

```bash
git add docs/measurements/2026-10-04-e2e-flake-rate.md
git commit -m "docs: record the app.spec.ts flake rate after PR #20

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Demo build mode and banner

**Files:**
- Modify: `streetlab/src/net/wsClient.ts` (inside `createTransportFromLocation`, after the `?backend=` block and before `if (isTauri())`)
- Create: `streetlab/src/ui/DemoBanner.tsx`
- Modify: `streetlab/src/App.tsx`, `streetlab/src/styles.css`
- Test: `streetlab/tests/wsClient.test.ts`, `streetlab/tests/demoBanner.test.tsx`

**Interfaces:**
- Produces: `DemoBanner` and `RELEASES_URL` (`export const RELEASES_URL: string`), and the `VITE_DEMO=1` build flag Task 5 sets in `vercel.json`.
- Precedence after this task: `?mock=1` → `?backend=` → **`VITE_DEMO=1` ⇒ mock** → Tauri sidecar → default `ws://127.0.0.1:8765`.

- [ ] **Step 1: Write the failing transport test**

In `streetlab/tests/wsClient.test.ts`, add `vi` to the vitest import:

```typescript
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
```

Then, in the same `describe` block that contains the test `'defaults to the browser-dev backend when no param and no Tauri IPC'`, add directly after it:

```typescript
  it('defaults to the mock in a VITE_DEMO build, but an explicit ?backend= still wins', async () => {
    vi.stubEnv('VITE_DEMO', '1');
    try {
      expect((await createTransportFromLocation('')).kind).toBe('mock');
      expect((await createTransportFromLocation('?backend=ws://localhost:8765')).kind).toBe('ws');
    } finally {
      vi.unstubAllEnvs();
    }
  });
```

- [ ] **Step 2: Run it and watch it fail**

```bash
cd /Users/jasonpereira/Projects/claude-worktrees/streetlab/portfolio-credibility-plan-cbca8d/streetlab
npx vitest run tests/wsClient.test.ts -t "VITE_DEMO"
```

Expected: FAIL (`expected 'ws' to be 'mock'`).

- [ ] **Step 3: Implement the default**

In `streetlab/src/net/wsClient.ts`, insert immediately before `if (isTauri()) {`:

```typescript
  // The hosted demo is a static site with no backend to reach: built with
  // VITE_DEMO=1, it runs the in-process mock by default. An explicit
  // ?backend= (handled above) still wins, so a visitor can point it at a
  // simulator of their own.
  if (import.meta.env.VITE_DEMO === '1') return createMockTransport();

```

- [ ] **Step 4: Run it and watch it pass**

```bash
npx vitest run tests/wsClient.test.ts
```

Expected: all tests in the file PASS, including the new one and `'defaults to the browser-dev backend ...'`.

- [ ] **Step 5: Write the failing banner test**

Create `streetlab/tests/demoBanner.test.tsx`:

```tsx
// @vitest-environment jsdom
import { describe, expect, it } from 'vitest';
import { render } from '@testing-library/react';
import { DemoBanner, RELEASES_URL } from '../src/ui/DemoBanner';

describe('DemoBanner', () => {
  it('says the demo is scripted and links to the releases page', () => {
    const { getByRole, getByText } = render(<DemoBanner />);
    expect(getByRole('note').textContent).toMatch(/scripted data/i);
    expect(getByText('Get the full app').getAttribute('href')).toBe(RELEASES_URL);
  });
});
```

Run: `npx vitest run tests/demoBanner.test.tsx`. Expected: FAIL (module not found).

- [ ] **Step 6: Implement the banner**

Create `streetlab/src/ui/DemoBanner.tsx`:

```tsx
export const RELEASES_URL = 'https://github.com/jasonpereira518/streetlab/releases';

/** Shown only in VITE_DEMO builds: the hosted demo runs the in-process mock,
 * not the Python simulator, and must say so. */
export function DemoBanner() {
  return (
    <div className="demo-banner" role="note">
      Live demo: scripted data in your browser, not the real Python simulator.{' '}
      <a href={RELEASES_URL} target="_blank" rel="noreferrer">
        Get the full app
      </a>
    </div>
  );
}
```

In `streetlab/src/App.tsx`, add the import beside the other `./ui/` imports:

```tsx
import { DemoBanner } from './ui/DemoBanner';
```

and render it as the first child of the root `<div className={shell}>`:

```tsx
      {import.meta.env.VITE_DEMO === '1' && <DemoBanner />}
```

Append to `streetlab/src/styles.css`:

```css
.demo-banner {
  position: fixed;
  top: 8px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 1000;
  padding: 6px 14px;
  border-radius: 999px;
  font-size: 12px;
  background: rgba(20, 24, 32, 0.92);
  color: #e8ecf2;
  border: 1px solid rgba(255, 255, 255, 0.18);
}
.demo-banner a {
  color: #8fc2ff;
}
```

- [ ] **Step 7: Run tests, typecheck, and look at the banner**

```bash
npx vitest run && npx tsc --noEmit
```

Expected: 278 tests pass (276 + 2), no type errors. If `tsc` rejects `import.meta.env.VITE_DEMO`, add `/// <reference types="vite/client" />` to a new `streetlab/src/vite-env.d.ts` and re-run.

Build the demo and screenshot it with a one-off script (do not commit it):

```bash
VITE_DEMO=1 npm run build
npx vite preview --port 4173 > /tmp/demo-preview.log 2>&1 &
PREVIEW=$!
sleep 3
node --input-type=module -e "
import { chromium } from '@playwright/test';
const b = await chromium.launch({ args: ['--enable-unsafe-webgpu','--enable-features=Vulkan,WebGPU','--use-angle=default'] });
const p = await b.newPage({ viewport: { width: 1440, height: 900 } });
const errs = []; p.on('pageerror', e => errs.push(e.message));
await p.goto('http://localhost:4173/');
await p.locator('.viewport-stats').waitFor({ timeout: 15000 });
console.log('banner:', await p.locator('.demo-banner').innerText());
await p.screenshot({ path: '/tmp/demo-check.png' });
console.log('page errors:', errs.length, errs.join(' | '));
await b.close();
"
kill $PREVIEW
```

Expected: `banner:` prints the banner text, `page errors: 0`. Open `/tmp/demo-check.png` with the Read tool and confirm the banner does not cover the toolbar controls and the scene is animating. If it overlaps the toolbar, move it to `bottom: 8px` instead of `top: 8px` and re-check. Also type an address into the address box and press Enter in this build once, via the same script or by hand, and note in the commit message whether the mock handles it without a page error.

- [ ] **Step 8: Commit**

```bash
cd ..
git add streetlab/src streetlab/tests
git commit -m "feat(demo): default to the mock in VITE_DEMO builds and label it

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: CI workflow

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: Task 1's flake result (decides whether e2e is required).
- Produces: a workflow at `.github/workflows/ci.yml` whose badge URL Task 7 embeds:
  `https://github.com/jasonpereira518/streetlab/actions/workflows/ci.yml/badge.svg`.

- [ ] **Step 1: Write the workflow**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
  workflow_dispatch:

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

permissions:
  contents: read

jobs:
  frontend:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    defaults:
      run:
        working-directory: streetlab
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm
          cache-dependency-path: streetlab/package-lock.json
      - run: npm ci
      - run: npx tsc --noEmit
      - run: npx vitest run

  backend:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    defaults:
      run:
        working-directory: streetlab-backend
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true
          cache-dependency-glob: streetlab-backend/uv.lock
      - run: uv sync --frozen
      - run: uv run pytest -q tests ../contract

  e2e:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    env:
      # location.spec.ts's happy path needs live Nominatim/Overpass; it skips
      # itself when this is set, keeping CI deterministic.
      STREETLAB_OFFLINE: '1'
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm
          cache-dependency-path: streetlab/package-lock.json
      - uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true
          cache-dependency-glob: streetlab-backend/uv.lock
      - name: Install backend (three e2e specs spawn it)
        run: uv sync --frozen
        working-directory: streetlab-backend
      - name: Install frontend
        run: npm ci
        working-directory: streetlab
      - name: Install Chromium
        run: npx playwright install --with-deps chromium
        working-directory: streetlab
      - name: Run Playwright
        run: npm run test:e2e
        working-directory: streetlab
      - name: Keep traces on failure
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: playwright-test-results
          path: streetlab/test-results
```

- [ ] **Step 2: Validate the YAML locally**

```bash
cd /Users/jasonpereira/Projects/claude-worktrees/streetlab/portfolio-credibility-plan-cbca8d
python3 -c "import yaml,sys; d=yaml.safe_load(open('.github/workflows/ci.yml')); print(sorted(d['jobs']))"
```

Expected: `['backend', 'e2e', 'frontend']`.

- [ ] **Step 3: Commit locally**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: run typecheck, vitest, pytest and Playwright on Linux

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Ask before pushing**

**STOP.** Ask Jason in chat: "Push `claude/portfolio-credibility-plan-cbca8d` to origin and trigger the CI workflow on it? This creates a remote branch and uses GitHub Actions minutes." Wait for a clear yes. Do not push without it.

- [ ] **Step 5: Trial run and triage**

After a yes:

```bash
git push -u origin claude/portfolio-credibility-plan-cbca8d
gh workflow run ci.yml --ref claude/portfolio-credibility-plan-cbca8d
sleep 15
RUN=$(gh run list --workflow ci.yml --branch claude/portfolio-credibility-plan-cbca8d --limit 1 --json databaseId -q '.[0].databaseId')
gh run watch "$RUN" --exit-status
```

Expected: either all three jobs pass, or `gh run view "$RUN" --log-failed` shows what failed. Triage by cause:
- A test that depends on macOS (CoreML, `/Users/...` paths, `darwin`-only behaviour): skip it on Linux with a reason and note it in the commit, e.g. `@pytest.mark.skipif(sys.platform != "darwin", reason="needs macOS ...")`. Do not delete or weaken the assertion.
- A WebGL/WebGPU failure on the Linux runner in e2e: the Playwright config already falls back to WebGL2; if Chromium still cannot create a context, add `'--use-gl=angle', '--use-angle=swiftshader'` to the Linux launch args only when `process.env.CI` is set in `streetlab/playwright.config.ts`, and re-run.
- A genuine bug exposed by Linux: fix it in a separate commit with its own test.

Re-trigger and repeat until green. Then run the e2e job twice more to confirm stability: `gh workflow run ci.yml --ref claude/portfolio-credibility-plan-cbca8d` (twice, each watched to completion).

- [ ] **Step 6: Apply the Task 1 rule**

If Task 1 recorded any failure, or if the three CI e2e runs were not all green, note in the commit body that the e2e job is advisory and do **not** recommend it as a required check; if all green, tell Jason it is safe to require all three jobs in branch protection (that setting is his to change).

- [ ] **Step 7: Commit any triage fixes**

```bash
git add -A -- ':!.claude'
git commit -m "ci: make the Linux run green (platform-specific skips, runner flags)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
git push
```

Skip this step if Step 5 needed no changes.

---

### Task 4: Release workflow and draft release

**Files:**
- Create: `.github/workflows/release.yml`
- Create: `.github/release-notes.md`

**Interfaces:**
- Produces: on a `v*` tag, a **draft** GitHub Release with `StreetLab-<tag>-macos-arm64.zip` and its `.sha256`. Task 7's Quickstart links to the repo's Releases page.
- `workflow_dispatch` builds the zip and uploads it as a workflow artifact only; it never creates a release.

- [ ] **Step 1: Write the release notes**

Create `.github/release-notes.md`:

```markdown
Unsigned macOS build for Apple Silicon (M1 or newer). StreetLab is a portfolio/learning project, not a production self-driving system.

**First launch.** macOS will say it cannot verify the app because it is not notarized. Right-click `StreetLab.app`, choose **Open**, then **Open** again. Or remove the quarantine flag: `xattr -dr com.apple.quarantine StreetLab.app`.

**Verify the download.** With the `.sha256` file next to the zip: `shasum -a 256 -c StreetLab-*.zip.sha256`.

It opens on Nob Hill from a bundled OpenStreetMap extract with no network needed; type any address to load it live.
```

- [ ] **Step 2: Write the workflow**

```yaml
name: Release

on:
  push:
    tags: ['v*']
  workflow_dispatch:

permissions:
  contents: write

jobs:
  macos-app:
    runs-on: macos-14
    timeout-minutes: 60
    env:
      TAG: ${{ github.ref_name }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm
          cache-dependency-path: streetlab/package-lock.json
      - uses: dtolnay/rust-toolchain@stable
        with:
          targets: aarch64-apple-darwin
      - uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true
          cache-dependency-glob: streetlab-backend/uv.lock
      - name: Install dependencies
        run: |
          (cd streetlab-backend && uv sync --frozen)
          (cd streetlab && npm ci)
      - name: Build the packaged app
        run: bash scripts/build_app.sh
      - name: Package and check the bundle
        run: |
          APP="streetlab/src-tauri/target/release/bundle/macos/StreetLab.app"
          test -d "$APP"
          find "$APP" -name 'streetlab-server*' | grep -q . || { echo "sidecar missing from bundle"; exit 1; }
          NAME="${TAG//\//-}"
          ZIP="StreetLab-${NAME}-macos-arm64.zip"
          ditto -c -k --keepParent "$APP" "$ZIP"
          shasum -a 256 "$ZIP" > "$ZIP.sha256"
          echo "ZIP=$ZIP" >> "$GITHUB_ENV"
          ls -lh "$ZIP"
      - uses: actions/upload-artifact@v4
        with:
          name: StreetLab-macos-arm64
          path: |
            StreetLab-*.zip
            StreetLab-*.zip.sha256
      - name: Create a DRAFT release
        if: startsWith(github.ref, 'refs/tags/v')
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          gh release create "$TAG" "$ZIP" "$ZIP.sha256" \
            --draft --title "StreetLab $TAG" --notes-file .github/release-notes.md
```

- [ ] **Step 3: Validate the YAML locally**

```bash
cd /Users/jasonpereira/Projects/claude-worktrees/streetlab/portfolio-credibility-plan-cbca8d
python3 -c "import yaml; d=yaml.safe_load(open('.github/workflows/release.yml')); print(list(d['jobs']))"
```

Expected: `['macos-app']`.

- [ ] **Step 4: Commit and ask before the trial**

```bash
git add .github/workflows/release.yml .github/release-notes.md
git commit -m "ci: build the .app on macOS and stage a draft release on tags

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

**STOP.** Ask Jason: "Push this and run the Release workflow via workflow_dispatch? It builds on a macOS runner (slow, and macOS minutes cost 10× Linux) and only uploads a workflow artifact, creating no release." Wait for a yes, then:

```bash
git push
gh workflow run release.yml --ref claude/portfolio-credibility-plan-cbca8d
sleep 15
RUN=$(gh run list --workflow release.yml --limit 1 --json databaseId -q '.[0].databaseId')
gh run watch "$RUN" --exit-status
```

Expected: green. If the build fails, read `gh run view "$RUN" --log-failed`; the likely causes are a missing tool on the runner (add a setup step) or an arch assumption in `scripts/build_app.sh`.

- [ ] **Step 5: Verify the artifact actually launches**

```bash
mkdir -p /tmp/release-check && cd /tmp/release-check && rm -rf ./*
gh run download "$RUN" -n StreetLab-macos-arm64
shasum -a 256 -c StreetLab-*.zip.sha256
unzip -q StreetLab-*.zip
xattr -dr com.apple.quarantine StreetLab.app
open StreetLab.app
```

Expected: checksum `OK`; the app opens and shows the Nob Hill scene with the car moving. Quit it, then confirm no orphaned sidecar: `pgrep -f streetlab-server || echo none`. If it fails to start, do not tag a release until fixed.

- [ ] **Step 6: Tell Jason how to cut the release**

Do not push a tag. Report: to publish, run `git tag v0.1.0 && git push origin v0.1.0`; the workflow creates a **draft** release; review it on GitHub and click Publish.

---

### Task 5: Hosted demo on Vercel

**Files:**
- Create: `streetlab/vercel.json`
- Modify: `.gitignore` (ensure `.vercel` is ignored)

**Interfaces:**
- Consumes: Task 2's `VITE_DEMO=1` mode.
- Produces: a production URL, which Task 6 (repo homepage) and Task 7 (README Quickstart) embed. Record it as the value printed by `vercel --prod`.

- [ ] **Step 1: Write the config**

Create `streetlab/vercel.json`:

```json
{
  "installCommand": "npm ci",
  "buildCommand": "VITE_DEMO=1 npm run build",
  "outputDirectory": "dist"
}
```

- [ ] **Step 2: Ignore the Vercel link directory**

```bash
cd /Users/jasonpereira/Projects/claude-worktrees/streetlab/portfolio-credibility-plan-cbca8d
command grep -qx '.vercel' .gitignore 2>/dev/null || printf '\n.vercel\n' >> .gitignore
command grep -n 'vercel' .gitignore
```

Expected: a `.vercel` line.

- [ ] **Step 3: Prove the exact build command works locally**

```bash
cd streetlab
rm -rf dist
VITE_DEMO=1 npm run build
command grep -l "scripted data" dist/assets/*.js | head -1
```

Expected: build succeeds and the grep prints one bundle filename (the banner text is in the demo build).

- [ ] **Step 4: Commit**

```bash
cd ..
git add .gitignore streetlab/vercel.json
git commit -m "chore(demo): Vercel config for the static mock-backed demo

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 5: Ask before deploying**

**STOP.** Run `vercel whoami` and `vercel teams ls`, then ask Jason: "Deploy the demo to Vercel under `<account/team>` as project `streetlab-demo`? This publishes a public URL." Wait for the account choice and a clear yes. Do not deploy under an account Jason has not confirmed.

- [ ] **Step 6: Deploy**

After a yes, with the chosen scope:

```bash
cd streetlab
vercel link --yes --project streetlab-demo --scope <confirmed-scope>
vercel --prod --scope <confirmed-scope>
```

Expected: prints a `Production:` URL. Record it.

- [ ] **Step 7: Verify the live URL unauthenticated**

```bash
URL=<the production URL printed above>
curl -s -o /dev/null -w "%{http_code}\n" "$URL"
```

Expected: `200`. A `401`/`403` means Vercel Deployment Protection is on for this project; ask Jason to turn it off for Production in the project's settings, then re-check. Then run the one-off Playwright check from Task 2 Step 7 with `http://localhost:4173/` replaced by `$URL`; expected: banner text printed and `page errors: 0`.

---

### Task 6: Repo metadata and social preview

**Files:**
- Create: `streetlab/scripts/capture_social_preview.mjs`
- Create: `docs/screenshots/social-preview.png`

**Interfaces:**
- Consumes: Task 5's production URL (for the repo homepage).

- [ ] **Step 1: Write the capture script**

Create `streetlab/scripts/capture_social_preview.mjs`:

```javascript
// Regenerates docs/screenshots/social-preview.png (1280x640, GitHub's
// recommended social-preview size). Run against a live dev server exactly as
// capture_screenshots.mjs does:
//
//   cd streetlab-backend && tail -f /dev/null | uv run streetlab serve --source osm &
//   cd streetlab && npm run dev &
//   node scripts/capture_social_preview.mjs
import { chromium } from '@playwright/test';
import { mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT_DIR = resolve(HERE, '../../docs/screenshots');
mkdirSync(OUT_DIR, { recursive: true });

const browser = await chromium.launch({
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-angle=default'],
});
const page = await browser.newPage({ viewport: { width: 1280, height: 640 } });
await page.goto('http://localhost:1420/');
await page.getByRole('heading', { name: /./ }).first().waitFor({ timeout: 15000 });
await page.waitForTimeout(6000);
await page.screenshot({ path: resolve(OUT_DIR, 'social-preview.png') });
console.log('social-preview.png written');
await browser.close();
```

- [ ] **Step 2: Run it and check the image**

Use the same single-command backend + Vite + capture pattern as plan 1's Task 2 Step 3, replacing the capture line with `(cd streetlab && node scripts/capture_social_preview.mjs)`.

```bash
file docs/screenshots/social-preview.png
```

Expected: `PNG image data, 1280 x 640`. Open it with the Read tool and confirm the scene is rendered (not black).

- [ ] **Step 3: Commit**

```bash
git add streetlab/scripts/capture_social_preview.mjs docs/screenshots/social-preview.png
git commit -m "docs: add a 1280x640 social preview image and its capture script

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Ask before editing the public repo**

**STOP.** Show Jason exactly what will be set and wait for a yes:
- Description: `Self-driving simulator on real OpenStreetMap streets: Tauri + Three.js frontend, deterministic Python sim, ONNX detector in the loop. A portfolio project.`
- Topics: `simulation`, `tauri`, `threejs`, `webgpu`, `openstreetmap`, `onnx`
- Homepage: the Task 5 production URL

(The current description says "Tesla-style Full Self-Driving", which this replaces.)

- [ ] **Step 5: Apply and verify**

```bash
gh repo edit jasonpereira518/streetlab \
  --description "Self-driving simulator on real OpenStreetMap streets: Tauri + Three.js frontend, deterministic Python sim, ONNX detector in the loop. A portfolio project." \
  --homepage "<Task 5 production URL>" \
  --add-topic simulation --add-topic tauri --add-topic threejs \
  --add-topic webgpu --add-topic openstreetmap --add-topic onnx
gh repo view jasonpereira518/streetlab --json description,homepageUrl,repositoryTopics
```

Expected: the new description, homepage and six topics. Then tell Jason the social preview image cannot be set from the CLI: repo **Settings → General → Social preview → Upload** `docs/screenshots/social-preview.png`.

---

### Task 7: README and ARCHITECTURE updates

**Files:**
- Modify: `README.md` (badge block, `## Quickstart`, `## Testing`)
- Modify: `docs/ARCHITECTURE.md` (the Components paragraph)

**Interfaces:**
- Consumes: Task 3's badge URL, Task 4's Releases page, Task 5's production URL, and Task 3 Step 6's e2e decision.

- [ ] **Step 1: Write the failing check**

```bash
cd /Users/jasonpereira/Projects/claude-worktrees/streetlab/portfolio-credibility-plan-cbca8d
command grep -c 'actions/workflows/ci.yml' README.md; command grep -c 'releases' README.md
```

Expected: `0` and `0`.

- [ ] **Step 2: Add the live CI badge**

Insert as the first line of the badge block (directly above the License badge):

```markdown
[![CI](https://github.com/jasonpereira518/streetlab/actions/workflows/ci.yml/badge.svg)](https://github.com/jasonpereira518/streetlab/actions/workflows/ci.yml)
```

- [ ] **Step 3: Extend the Quickstart**

In `## Quickstart`, replace the opening sentence (`macOS on Apple Silicon. You need Rust, ...`) with:

```markdown
**Zero install:** [try the live demo](<Task 5 production URL>) in your browser. It runs scripted data in the browser, not the real Python simulator. **Full app:** download the prebuilt, unsigned Apple Silicon build from [Releases](https://github.com/jasonpereira518/streetlab/releases) (first launch: right-click → Open). **From source** (macOS on Apple Silicon; you need Rust, Node ≥ 20 and [`uv`](https://docs.astral.sh/uv/), which fetches Python 3.11):
```

Keep the existing three-command code block and the paragraph after it unchanged. Substitute the real production URL for the angle-bracket text; none may remain.

- [ ] **Step 4: Make the Testing section truthful about CI**

Under the Testing code block, replace the line `Counts measured 2026-10-04.` with the matching variant:

- If all three CI jobs were stable (Task 3 Step 6): `CI runs all three suites on every push ([Actions](https://github.com/jasonpereira518/streetlab/actions)); the live-network address-search e2e test is skipped there and run locally. Counts measured 2026-10-04.`
- Otherwise: `CI runs typecheck, vitest and pytest on every push ([Actions](https://github.com/jasonpereira518/streetlab/actions)); Playwright runs there as an advisory job and locally. Counts measured 2026-10-04.`

- [ ] **Step 5: Make the Components paragraph truthful again**

In `docs/ARCHITECTURE.md`, replace `fails the contract tests\non both sides, not just one.` with `fails CI and the local\ncontract tests on both sides, not just one.`

- [ ] **Step 6: Verify**

```bash
test "$(wc -l < README.md)" -le 130 && echo "PASS length: $(wc -l < README.md)" || echo "FAIL length"
command grep -n '<Task\|<the \|TBD\|TODO' README.md docs/ARCHITECTURE.md
python3 - <<'EOF'
import re, pathlib
for doc in ["README.md", "docs/ARCHITECTURE.md"]:
    p = pathlib.Path(doc); t = p.read_text()
    for l in re.findall(r'\]\(([^)#\s]+)(?:#[^)]*)?\)', t) + re.findall(r'src="([^"]+)"', t):
        if l.startswith(("http://", "https://")): continue
        print(("ok   " if (p.parent / l).exists() else "MISSING "), doc, l)
EOF
```

Expected: `PASS length`, the placeholder grep prints nothing, every relative link `ok`. Then confirm the badge renders: `curl -s https://github.com/jasonpereira518/streetlab/actions/workflows/ci.yml/badge.svg | command grep -o 'passing\|failing\|no status' | head -1` (expected `passing` once the branch is on origin and merged or the workflow ran on main; on a feature branch it may say `no status`, which is fine until merge).

- [ ] **Step 7: Commit**

```bash
git add README.md docs/ARCHITECTURE.md
git commit -m "docs: add the live CI badge, demo and release links, and truthful CI wording

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

## Plan self-review notes

- Spec coverage: refinement 1 (CI Linux) → Task 3; 2 (macOS `.app` build on tags) → Task 4; 3 (live badges) → Task 7; 4 (tagged release with zip + SHA-256 + Gatekeeper note) → Task 4; 5 (hosted mock demo, labelled) → Tasks 2 and 5; 6 (repo metadata, social preview) → Task 6; 7 (re-measure flaky e2e) → Task 1.
- Type consistency: `RELEASES_URL` and `DemoBanner` are defined in Task 2 and not referenced elsewhere by a different name; `VITE_DEMO` is read in two places (`wsClient.ts`, `App.tsx`) and set in one (`vercel.json`).
- Known unknowns that the trial runs resolve rather than guess: whether the Linux runner needs swiftshader flags for e2e (Task 3 Step 5), whether any backend test is macOS-only (Task 3 Step 5), and whether the mock tolerates the address box (Task 2 Step 7).
- The three outward-facing actions (push, deploy, repo edit) and the draft-only release each have an explicit STOP step.
