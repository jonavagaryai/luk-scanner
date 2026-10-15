# Luk Setup Scanner

Scans every US stock each weekday morning for Martin Luk-style setups (pullback longs, breakouts,
episodic pivots, pullback shorts) and publishes the results to a web page you can open on any device.

- **Runs on GitHub, for free.** Your computer can be off.
- **Data:** Massive (formerly Polygon.io). The free Basic plan is enough.
- **Schedule:** Tuesday to Saturday at 6:15 AM New York time, covering the previous trading day.
  You can also run it any time with the **Run workflow** button.

Educational tool, not financial advice.

---

## One-time setup (about 10 minutes)

### 1. Create the repository
1. Sign in at github.com (a free account is fine).
2. Click **+** (top right) → **New repository**.
3. Name it `luk-scanner`. Choose **Public**, because GitHub Pages needs a public repository on the free plan.
   Your Massive key is never visible; only the scan results are.
4. Click **Create repository**.

### 2. Upload the files
1. On the new, empty repository page, click **uploading an existing file**.
2. Unzip `luk-scanner.zip`, open the `luk-scanner` folder, select **everything inside it**
   (including the `.github` folder), and drag it onto the page.
3. Click **Commit changes**.

Check that you now see `.github/workflows/scan.yml` in the repository. If the `.github` folder is missing,
click **Add file → Create new file**, type `.github/workflows/scan.yml` as the name, paste in the contents
of that file from the zip, and commit.

### 3. Add your Massive key (kept secret)
1. In the repository: **Settings → Secrets and variables → Actions**.
2. **Secrets** tab → **New repository secret**.
3. Name: `MASSIVE_API_KEY`. Secret: your key. Click **Add secret**.

Optional, on the **Variables** tab of the same page:

| Name | Example | What it does |
|---|---|---|
| `ACCOUNT_SIZE` | `25000` | Account size used for the default share counts (you can also change it on the page). |
| `CHART_TICKERS` | `MU,SNDK,NVDA` | Stocks always included on the page, even when they have no setup. |
| `MASSIVE_PLAN` | `paid` | Set this if you upgrade Massive, so downloads don't wait between calls. |

### 4. Turn on the web page
**Settings → Pages → Build and deployment → Source: GitHub Actions.**

### 5. Run it the first time
1. Open the **Actions** tab. If asked, click **I understand my workflows, go ahead and enable them**.
2. Click **Luk scan** → **Run workflow** → **Run workflow**.
3. The first run takes about **70 minutes**, because it downloads ~300 days of whole-market data at the free
   plan's 5 calls per minute. Every run after that takes a few minutes.

When it finishes, your page is at:

**`https://YOUR-GITHUB-USERNAME.github.io/luk-scanner/`**

Bookmark it on your phone. Each run also writes a short summary on its page in the **Actions** tab.

---

## Everyday use
- **Check the page** before the open. The status line shows the date of the last bar and when it was updated.
- **Run it now:** GitHub app or website → repository → **Actions → Luk scan → Run workflow**.
- **Change your account size or risk** under **Account & rules** on the page. Share counts update instantly
  and are remembered in that browser.
- **Scan your own files** under **Price data** on the page, as before. **Show latest scan** switches back.

## Good to know
- **Data timing:** Massive's free plan publishes each day's data some time after the close. If a run starts
  before it's ready, the page shows the previous day. Use **Run workflow** later to refresh.
- **GitHub pauses schedules** in repositories with no activity for 60 days and emails you first.
  Click **Enable workflow** in the Actions tab to resume.
- **Price history** is kept in GitHub's cache between runs. If it's ever cleared, the next run simply
  downloads it again (about 70 minutes on the free plan).

## Files
- `luk_scanner.py`: the scanner (also runs on your own computer, see the notes at the top of the file).
- `site/index.html`: the web page. Each run adds `results.json` next to it.
- `.github/workflows/scan.yml`: the schedule and steps GitHub runs.
- `luk_setups.pine`: the TradingView indicator.
