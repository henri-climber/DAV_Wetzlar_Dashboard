from selenium import webdriver
from selenium.common import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import json, time, os

URL = "https://dav.results.info"

PAUSE = 0.35

driver = webdriver.Chrome()
wait = WebDriverWait(driver, 15)


def select_all_events():
    """Open the middle status dropdown and pick 'All events' via keyboard."""
    filters = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".event-filters")))
    status_dropdown = filters.find_elements(By.CSS_SELECTOR, ".el-select")[1]
    status_dropdown.click()
    status_input = wait.until(
        EC.element_to_be_clickable(
            (By.CSS_SELECTOR, ".event-filters .el-select:nth-of-type(2) input.el-input__inner")
        )
    )
    status_input.click()
    time.sleep(PAUSE)
    # Five downs lands on "All events" given current ordering
    for _ in range(5):
        status_input.send_keys(Keys.ARROW_DOWN)
        time.sleep(PAUSE)
    status_input.send_keys(Keys.ENTER)
    time.sleep(PAUSE)


def scroll_all_cards():
    """Scrolls page to bottom until number of event cards stops increasing."""
    stable_rounds = 0
    last_count = -1
    while stable_rounds < 3:  # need 3 consecutive checks with no growth
        # count current cards
        cards = driver.find_elements(By.CSS_SELECTOR, "a.event-card[href*='/event/']")
        count = len(cards)
        if count == last_count:
            stable_rounds += 1
        else:
            stable_rounds = 0
            last_count = count
        # scroll to bottom
        sh = driver.execute_script("return document.scrollingElement.scrollHeight;")
        driver.execute_script("window.scrollTo(0, arguments[0]);", sh)
        time.sleep(PAUSE)
    return last_count


def get_discipline_for(card):
    """Extract 'Boulder', 'Lead', or 'Speed' text from anywhere inside the card."""
    try:
        # get all visible text inside the card
        text = card.text.strip().lower()
        if "boulder" in text:
            return "Boulder"
        elif "lead" in text:
            return "Lead"
        elif "speed" in text:
            return "Speed"
        else:
            return "Combined"
    except Exception as e:
        print("Discipline parse error:", e)
        return "Unknown"


def harvest_event_map():
    """Build { absolute_url : discipline }."""
    # ensure everything is loaded
    total = scroll_all_cards()
    print(f"Found ~{total} cards rendered.")

    urls = []
    cards = driver.find_elements(By.CSS_SELECTOR, "a.event-card[href*='/event/']")
    for a in cards:
        href = a.get_attribute("href") or a.get_attribute("data-href") or a.get_attribute("route")
        # href += "general"
        if not href:
            continue
        discipline = get_discipline_for(a) or "Combined"
        urls.append(href)  # f"{href}/{discipline}"

    return urls


try:
    driver.get(URL)
    driver.maximize_window()

    # pick "All events" (you can comment this if you already set it earlier)
    select_all_events()

    # harvest
    urls = harvest_event_map()
    print(f"Collected {len(urls)} event URLs.")

    # save to disk
    out_path = os.path.abspath("events_2025.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(urls, f, ensure_ascii=False, indent=2)
    print(f"Saved mapping to {out_path}")


except Exception as e:
    print("❌ Error:", e)

import pandas as pd
import re
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
def _parse_event_dates(date_text: str):
    """
    Convert strings like:
      '01-Feb-2025 – 02-Feb-2025'
      '01 Feb 2025 – 02 Feb 2025'
      '01-Feb-2025'
    into (start, end) as pandas.Timestamp.
    """
    if not date_text:
        return pd.NaT, pd.NaT

    txt = date_text.strip()
    # normalize common dash characters
    txt = txt.replace("—", "–").replace("--", "–")

    # split only on the *range* en dash (with optional spaces)
    parts = re.split(r"\s*[–]\s*", txt)
    start_s = parts[0].strip()
    end_s = parts[1].strip() if len(parts) > 1 else start_s

    def to_ts(s):
        for fmt in ("%d-%b-%Y", "%d %b %Y", "%Y-%m-%d"):
            try:
                return pd.to_datetime(s, format=fmt)
            except Exception:
                continue
        return pd.to_datetime(s, dayfirst=True, errors="coerce")

    return to_ts(start_s), to_ts(end_s)


def get_event_meta(driver, wait):
    """
    Extract event_name, discipline, location, date_start, date_end
    from the header using ONLY its text (keeps original casing).
    """
    header = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.event-header")))
    hdr_text = header.text.strip()

    # split into non-empty lines (keep original case!)
    lines = [ln.strip() for ln in hdr_text.splitlines() if ln.strip()]
    lower  = [ln.lower() for ln in lines]

    # event name = first non-empty line
    event_name = lines[0] if lines else ""

    # discipline from any line (case-insensitive)
    discipline = ""
    for ln in lower:
        if re.search(r"\bboulder\b", ln): discipline = "Boulder"; break
        if re.search(r"\blead\b", ln):    discipline = "Lead";    break
        if re.search(r"\bspeed\b", ln):   discipline = "Speed";   break

    # location: line after the one that equals 'location:'
    location = ""
    if "location:" in lower:
        i = lower.index("location:")
        if i + 1 < len(lines):
            location = lines[i + 1].strip()

    # date: line after the one that equals 'date:'
    date_start = date_end = pd.NaT
    if "date:" in lower:
        i = lower.index("date:")
        if i + 1 < len(lines):
            date_line = lines[i + 1].strip()  # keep original case for months!
            date_start, date_end = _parse_event_dates(date_line)

    return event_name, discipline, location, date_start, date_end

def iter_category_tabs(driver, wait):
    """
    Yield (idx, el, label) for each category tab in the row.
    Always re-find elements each iteration because DOM refreshes on click.
    """
    # container with the nav items
    nav_xpath = "//div[contains(@class,'cat-nav-container')]//div[contains(@class,'cat-nav')]"
    wait.until(EC.presence_of_element_located((By.XPATH, nav_xpath)))
    # collect labels first (so we know how many to click), then click by index
    labels = []
    items = driver.find_elements(By.XPATH, nav_xpath + "//div[contains(@class,'nav-item')]")
    for it in items:
        lab = it.text.strip()
        if lab:
            labels.append(lab)
    for i in range(len(labels)):
        # re-find i-th item every loop
        item = wait.until(EC.element_to_be_clickable(
            (By.XPATH, f"({nav_xpath}//div[contains(@class,'nav-item')])[{i + 1}]")
        ))
        yield i, item, labels[i]


def parse_athlete_rows(driver):
    """
    Return list of dicts for all athlete lines currently visible in the table.
    Robust for desktop/mobile variants.
    """
    rows = []
    # Both desktop and mobile use 'athlete' class; desktop also has 'athlete-line'
    line_sel = "//div[contains(@class,'athlete') and contains(@class,'athlete-line') or contains(@class,'athlete-container')]"
    # Wait for at least one athlete line; some categories may be empty → handle gracefully
    try:
        WebDriverWait(driver, 6).until(EC.presence_of_element_located((By.XPATH, line_sel)))
    except Exception:
        return rows  # empty category
    lines = driver.find_elements(By.XPATH, line_sel)
    for ln in lines:
        try:
            # rank (may be empty for DNS/DNF)
            try:
                rank = ln.find_element(By.XPATH, ".//div[contains(@class,'rank')]").text.strip()
            except Exception:
                rank = ""
            # name anchor
            try:
                name = ln.find_element(By.XPATH, ".//a[contains(@class,'r-name')]").text.strip()
            except Exception:
                name = ""
            # club is usually in r-name-sub: "8 • Darmstadt–Starkenburg"
            team = ""
            try:
                sub = ln.find_element(By.XPATH, ".//div[contains(@class,'r-name-sub')]").text.strip()
                # split on bullet if present
                if "•" in sub:
                    team = sub.split("•", 1)[-1].strip()
                else:
                    team = sub.strip()
            except Exception:
                team = ""
            rows.append({"rank": rank, "name": name, "team": team})
        except Exception:
            continue
    return rows


# load urls from json
event_urls = json.load(open("events_2025.json"))


def click_first_category():
    """
    On an event page (/event/<id>/), click the FIRST category row.
    Works regardless of the category name ('U13 m', 'Youth B w', etc.).
    Returns True if clicked & navigated; False if no categories present.
    """
    try:
        # Ensure the event header/card is there
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".event-card")))

        # Some events show tabs (Boulder/Lead/Speed/Info). Make sure we’re not on 'Info'.
        # If there is a results tab bar, prefer the first non-Info tab.
        try:
            tabs = driver.find_elements(By.XPATH,
                                        "//div[contains(@class,'rs-tabs')]//button|//div[contains(@class,'rs-tabs')]//a")
            for t in tabs:
                label = t.text.strip().lower()
                if label and label != "info":
                    # Click first results tab (e.g., Boulder/Lead/Speed)
                    try:
                        wait.until(EC.element_to_be_clickable(t))
                        t.click()
                        time.sleep(PAUSE)
                        break
                    except Exception:
                        pass
        except Exception:
            pass

        # The category rows container (multiple rows). We want the FIRST anchor inside.
        # Typical structure: <div class="dcat-row ..."><div class="cat-name finished"><a href="/event/1374/general/boulder">U13 m</a></div>...</div>
        link = wait.until(
            EC.element_to_be_clickable(
                (By.XPATH, "(//div[contains(@class,'dcat-row')]//a[contains(@href,'/event/')])[1]")
            )
        )

        # Scroll and click (with JS fallback)
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", link)
        time.sleep(PAUSE)
        try:
            link.click()
        except Exception:
            driver.execute_script("arguments[0].click();", link)

        # Wait until we’re on the category page (URL typically contains '/general/')
        try:
            wait.until(EC.url_contains("/general/"))
        except TimeoutException:
            # Some events might use other segment names; alternatively wait for the Q/F chips
            wait.until(
                EC.any_of(
                    EC.presence_of_element_located(
                        (By.XPATH, "//div[contains(@class,'rs-') or contains(@class,'date-box')]")),
                    EC.presence_of_element_located((By.XPATH, "//span[.='Q']|//div[normalize-space()='Q']"))
                )
            )
        return True
    except TimeoutException:
        return False


final_urls = []
import pandas as pd

# pandas settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)

# Initialize an empty DataFrame outside the loop (do this once, before the try/for)
# columns (add location + dates)
results_df = pd.DataFrame(columns=[
    "rank", "name", "team", "category", "discipline", "event_name",
    "location", "date_start", "date_end", "url"
])

try:
    for i, url in enumerate(event_urls, 1):
        try:
            driver.get(url)
            driver.maximize_window()
            time.sleep(PAUSE)

            event_name, discipline, location, date_start, date_end = get_event_meta(driver, wait)

            ok = click_first_category()
            if not ok:
                print(f"[{i}/{len(event_urls)}] No categories found → SKIP  {url}")
                continue

            # Capture the landing URL (this is the stable /event/<id>/general/<discipline> with state)
            dest = driver.current_url
            final_urls.append(dest)
            print(f"[{i}/{len(event_urls)}] OK  →  {dest}")

            # small politeness pause
            time.sleep(PAUSE)

            # we are now on /event/<id>/general/... after click_first_category()

            base_url = driver.current_url

            # get all categories in the horizontal bar (U13 m, U15 w, etc.)
            for idx, el, label in iter_category_tabs(driver, wait):
                driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
                time.sleep(PAUSE / 2)
                try:
                    el.click()
                except Exception:
                    driver.execute_script("arguments[0].click();", el)
                # give UI time to refresh
                time.sleep(1)

                # parse all visible athletes for this category
                rows = parse_athlete_rows(driver)
                if not rows:
                    continue

                # convert to DataFrame
                df_cat = pd.DataFrame(rows)
                if df_cat.empty:
                    continue

                # filter for Wetzlar
                df_cat = df_cat[df_cat["team"].str.contains("wetzlar", case=False, na=False)]

                # enrich with context
                if not df_cat.empty:
                    df_cat["category"] = label
                    df_cat["category"] = label
                    df_cat["discipline"] = discipline
                    df_cat["event_name"] = event_name
                    df_cat["location"] = location
                    df_cat["date_start"] = date_start  # pandas.Timestamp
                    df_cat["date_end"] = date_end  # pandas.Timestamp
                    df_cat["url"] = driver.current_url

                    # append to master DataFrame
                    results_df = pd.concat([results_df, df_cat], ignore_index=True)
                    time.sleep(1)

            results_df.to_excel("dav_wetzlar_results_2025_final.xlsx", index=False, )
            print(f"✅ Saved all {len(results_df)} Wetzlar results to dav_wetzlar_results_2025_final.xlsx")

        except Exception as e:
            print(f"[{i}/{len(event_urls)}] Error on {url} → {e}")
            # continue to next event
            continue

except Exception as e:
    print("Fatal error:", e)

finally:
    input("Press Enter to quit...")
    driver.quit()
