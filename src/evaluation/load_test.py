"""Step 49: performance efficiency. Opens the student page in many browser sessions at the same
time and measures (1) how long the page takes to show the rooms and (2) how long a recommendation
takes. Streamlit talks to the browser over a WebSocket, so plain HTTP tools like Locust cannot
trigger the app's code; real (headless) browsers are used instead.

Usage:  python src/evaluation/load_test.py http://<server-ip>:8501 1 10 25 50
"""
import asyncio, math, os, sys, time
import pandas as pd
from playwright.async_api import async_playwright

URL = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8501"
LEVELS = [int(x) for x in sys.argv[2:]] or [1, 10, 25, 50]
TIMEOUT_MS = 120_000
USERS_PER_BROWSER = 5          # spread sessions over several browser processes so the laptop copes
STAGGER_S = 0.2                # users arrive 0.2 s apart instead of in the same millisecond
RUNS_FILE = "results/load_test_runs.csv"


async def one_user(browser, level, user, rows):
    await asyncio.sleep(user * STAGGER_S)
    row = {"users": level, "user": user, "page_s": None, "recommend_s": None, "error": ""}
    ctx = None
    try:
        ctx = await browser.new_context()
        page = await ctx.new_page()
        t0 = time.perf_counter()
        await page.goto(f"{URL}/Student", timeout=TIMEOUT_MS)
        await page.get_by_text("Rooms right now").wait_for(timeout=TIMEOUT_MS)
        row["page_s"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        await page.get_by_role("button", name="Recommend rooms").click(timeout=TIMEOUT_MS)
        await page.get_by_text("Best rooms for").wait_for(timeout=TIMEOUT_MS)
        row["recommend_s"] = time.perf_counter() - t1
    except Exception as e:                                   # timeout or page error counts as a failure
        row["error"] = type(e).__name__
    finally:
        rows.append(row)
        if ctx is not None:
            try:
                await ctx.close()
            except Exception:
                pass


async def run_level(p, level):
    rows = []
    n_browsers = math.ceil(level / USERS_PER_BROWSER)
    browsers = [await p.chromium.launch(args=["--disable-gpu", "--disable-dev-shm-usage"])
                for _ in range(n_browsers)]
    tasks = [one_user(browsers[u % n_browsers], level, u, rows) for u in range(level)]
    await asyncio.gather(*tasks, return_exceptions=True)
    for b in browsers:
        try:
            await b.close()
        except Exception:
            pass
    return rows


def summarize(d):
    def p95(x):
        return x.quantile(0.95)
    s = d.groupby("users").agg(
        sessions=("user", "size"),
        failures=("error", lambda e: int((e.fillna("") != "").sum())),
        page_median_s=("page_s", "median"), page_p95_s=("page_s", p95), page_max_s=("page_s", "max"),
        rec_median_s=("recommend_s", "median"), rec_p95_s=("recommend_s", p95), rec_max_s=("recommend_s", "max"))
    s["success_%"] = (1 - s["failures"] / s["sessions"]) * 100
    return s.round(2)


async def main():
    os.makedirs("results", exist_ok=True)
    async with async_playwright() as p:
        for level in LEVELS:
            print(f"{level} simultaneous users...", flush=True)
            rows = await run_level(p, level)
            d = pd.DataFrame(rows)
            d.to_csv(RUNS_FILE, mode="a", header=not os.path.exists(RUNS_FILE), index=False)   # saved after every level
            print(summarize(d).to_string(), flush=True)
            await asyncio.sleep(10)                          # let the server settle between levels
    s = summarize(pd.read_csv(RUNS_FILE))
    s.to_csv("results/load_test_summary.csv")
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    print(f"\nLoad test against {URL} (all runs in {RUNS_FILE})")
    print(s)


asyncio.run(main())