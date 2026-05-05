import json
import re
import time
from datetime import datetime

OUTPUT_FILE = "scraped_content.json"


def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def scrape_url_playwright(url: str) -> dict:
    from playwright.sync_api import sync_playwright

    result = {
        "url": url,
        "scraped_at": datetime.utcnow().isoformat() + "Z",
        "success": False,
        "raw_text": "",
        "error": None,
    }

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            )

            resp = page.goto(url, wait_until="load", timeout=30000)

            if resp and resp.status >= 400:
                result["error"] = f"HTTP {resp.status} for {url}"
                browser.close()
                print(f"  [FAIL] {url} — HTTP {resp.status}")
                return result

            # Wait for JS-rendered content then allow extra time for full render
            for sel in ["[class*='accordion']", "main", "article", "body"]:
                try:
                    page.wait_for_selector(sel, timeout=12000)
                    break
                except Exception:
                    continue

            # Extra settle time — JS frameworks batch-render DOM nodes
            time.sleep(2)

            # Extract accordion Q&A blocks (primary content on Deriv help pages)
            acc_elements = page.query_selector_all("[class*='accordion']")
            if acc_elements:
                texts = []
                for el in acc_elements:
                    t = el.inner_text().strip()
                    if t:
                        texts.append(t)
                raw = "\n\n".join(texts)
            else:
                # Fallback: try main/article, then full body
                raw = ""
                for sel in ["main", "article", "body"]:
                    el = page.query_selector(sel)
                    if el:
                        raw = el.inner_text()
                        if len(raw.strip()) >= 200:
                            break

            browser.close()

        cleaned = clean_text(raw)

        if len(cleaned) < 200:
            result["error"] = f"Insufficient content after cleaning ({len(cleaned)} chars)"
            result["raw_text"] = cleaned
            print(f"  [WARN] {url} — only {len(cleaned)} chars extracted")
        else:
            result["success"] = True
            result["raw_text"] = cleaned
            print(f"  [OK]   {url} — {len(cleaned)} chars")

    except Exception as exc:
        result["error"] = str(exc)
        print(f"  [FAIL] {url} — {exc}")

    return result


def run(sources_path: str = "sources.json") -> list:
    with open(sources_path) as f:
        sources = json.load(f)["sources"]

    results = []
    for url in sources:
        results.append(scrape_url_playwright(url))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    n_ok = sum(1 for r in results if r["success"])
    print(f"STAGE: CONTENT_SCRAPED — {n_ok} pages scraped")
    return results


if __name__ == "__main__":
    run()
