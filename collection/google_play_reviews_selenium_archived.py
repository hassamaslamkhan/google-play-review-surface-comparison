import re
import time
from copy import copy
from datetime import datetime, timezone

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


# ============================================================
# HARDCODED SETTINGS
# ============================================================
# APP_NAME = "Facebook"



# APP_URL = (
#     "https://play.google.com/store/apps/details?"
#     "id=com.facebook.katana&hl=en_US"
# )
# OUTPUT_FILE = "Newest_facebook_google_play_reviews.xlsx"

APP_NAME = "Instagram"
APP_URL = (
    "https://play.google.com/store/apps/details?"
    "id=com.instagram.android&hl=en_US"
)
OUTPUT_FILE = "Newest_instagram_google_play_reviews.xlsx"


WAIT_SECONDS = 1

SCROLL_PAUSE_SECONDS = 0.5
MAX_SCROLLS = 400
STOP_AFTER_NO_NEW_ROUNDS = 5
CHECKPOINT_EVERY_REVIEWS = 100

# Keep False because you said the browser should open visibly.
HEADLESS = False


# ============================================================
# HELPERS
# ============================================================
def safe_element(parent, by, selector):
    try:
        return parent.find_element(by, selector)
    except (NoSuchElementException, StaleElementReferenceException):
        return None


def safe_text(parent, css_selector):
    el = safe_element(parent, By.CSS_SELECTOR, css_selector)
    if el is None:
        return ""
    try:
        return (el.get_attribute("textContent") or "").strip()
    except StaleElementReferenceException:
        return ""


def safe_attr(parent, css_selector, attribute):
    el = safe_element(parent, By.CSS_SELECTOR, css_selector)
    if el is None:
        return ""
    try:
        return (el.get_attribute(attribute) or "").strip()
    except StaleElementReferenceException:
        return ""


def parse_rating(label):
    """Example label: 'Rated 3 stars out of five stars' -> 3"""
    match = re.search(r"Rated\s+(\d+)", label or "", flags=re.I)
    return int(match.group(1)) if match else None


def normalize_review_date(raw_date):
    """Google Play is forced to English using hl=en_US."""
    if not raw_date:
        return ""
    try:
        return datetime.strptime(raw_date, "%B %d, %Y").date().isoformat()
    except ValueError:
        return ""


def extract_review(card, collected_at_utc):
    """Extract one Google Play review card using the supplied panel HTML."""
    header = safe_element(card, By.CSS_SELECTOR, "header.c1bOId")
    if header is None:
        return None

    try:
        review_id = (header.get_attribute("data-review-id") or "").strip()
    except StaleElementReferenceException:
        return None

    if not review_id:
        return None

    rating_label = safe_attr(card, ".iXRFPc", "aria-label")
    raw_date = safe_text(card, ".bp9Aid")

    helpful_el = safe_element(card, By.CSS_SELECTOR, "[data-original-thumbs-up-count]")
    helpful_count = 0
    if helpful_el is not None:
        try:
            raw_helpful = helpful_el.get_attribute("data-original-thumbs-up-count") or "0"
            helpful_count = int(raw_helpful.replace(",", "").strip())
        except (ValueError, StaleElementReferenceException):
            helpful_count = 0

    return {
        "app_name": APP_NAME,
        "app_url": APP_URL,
        "review_id": review_id,
        "reviewer_name": safe_text(card, ".X5PpBb"),
        "reviewer_avatar_url": safe_attr(card, "img.T75of.abYEib", "src"),
        "rating": parse_rating(rating_label),
        "rating_label": rating_label,
        "review_date": raw_date,
        "review_date_iso": normalize_review_date(raw_date),
        "review_text": safe_text(card, ".h3YV2d"),
        "helpful_count": helpful_count,
        "collected_at_utc": collected_at_utc,
    }


def save_excel(reviews_by_id):
    rows = list(reviews_by_id.values())
    if not rows:
        return

    df = pd.DataFrame(rows)

    # Preserve the order in which reviews were first encountered.
    if "scrape_order" in df.columns:
        df = df.sort_values("scrape_order")

    column_order = [
        "scrape_order",
        "app_name",
        "app_url",
        "review_id",
        "reviewer_name",
        "reviewer_avatar_url",
        "rating",
        "rating_label",
        "review_date",
        "review_date_iso",
        "review_text",
        "helpful_count",
        "collected_at_utc",
    ]
    df = df[column_order]

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="reviews")
        ws = writer.book["reviews"]

        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

        widths = {
            "A": 16,  # scrape_order
            "B": 18,  # app_name
            "C": 55,  # app_url
            "D": 40,  # review_id
            "E": 28,  # reviewer_name
            "F": 60,  # avatar url
            "G": 10,  # rating
            "H": 34,  # rating label
            "I": 18,  # review date
            "J": 18,  # ISO date
            "K": 100, # review text
            "L": 16,  # helpful count
            "M": 28,  # collection timestamp
        }
        for column, width in widths.items():
            ws.column_dimensions[column].width = width

        # Wrap long text and URLs.
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                alignment = copy(cell.alignment)
                alignment.vertical = "top"
                alignment.wrap_text = True
                cell.alignment = alignment


def click_see_all_reviews(driver, wait):
    # This targets the visible button text from the element you supplied.
    locator = (
        By.XPATH,
        "//button[.//span[normalize-space()='See all reviews']]",
    )

    button = wait.until(EC.presence_of_element_located(locator))
    driver.execute_script(
        "arguments[0].scrollIntoView({block: 'center', inline: 'nearest'});",
        button,
    )
    time.sleep(0.5)

    try:
        wait.until(EC.element_to_be_clickable(locator)).click()
    except (ElementClickInterceptedException, StaleElementReferenceException):
        button = wait.until(EC.presence_of_element_located(locator))
        driver.execute_script("arguments[0].click();", button)
    time.sleep(5)
    # Wait until at least one review card from the popup exists.
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.RHo1pe")))


def find_review_scroll_container(driver, wait):
    """
    Find the scrollable popup ancestor dynamically instead of depending on
    Google's outer dialog class names, which are more likely to change.
    """
    first_card = wait.until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div.RHo1pe"))
    )

    container = driver.execute_script(
        """
        const card = arguments[0];

        // First choice: nearest ancestor explicitly configured to scroll.
        let p = card.parentElement;
        while (p && p !== document.body) {
            const style = window.getComputedStyle(p);
            const overflowY = style.overflowY;
            if ((overflowY === 'auto' || overflowY === 'scroll') &&
                p.scrollHeight > p.clientHeight + 10) {
                return p;
            }
            p = p.parentElement;
        }

        // Fallback: ancestor with the greatest scrollable vertical overflow.
        p = card.parentElement;
        let best = null;
        let bestDelta = 0;
        while (p && p !== document.body) {
            const delta = p.scrollHeight - p.clientHeight;
            if (delta > bestDelta) {
                best = p;
                bestDelta = delta;
            }
            p = p.parentElement;
        }
        return best;
        """,
        first_card,
    )

    if container is None:
        raise RuntimeError("Could not identify the reviews popup scroll container.")

    return container


def collect_currently_loaded_reviews(driver, reviews_by_id, collected_at_utc):
    cards = driver.find_elements(By.CSS_SELECTOR, "div.RHo1pe")
    added = 0

    for card in cards:
        try:
            row = extract_review(card, collected_at_utc)
        except StaleElementReferenceException:
            continue

        if not row:
            continue

        review_id = row["review_id"]
        if review_id not in reviews_by_id:
            row["scrape_order"] = len(reviews_by_id) + 1
            reviews_by_id[review_id] = row
            added += 1
        else:
            # Refresh the record in case lazy-loaded attributes became available.
            row["scrape_order"] = reviews_by_id[review_id]["scrape_order"]
            reviews_by_id[review_id].update(row)

    return added, len(cards)


def scroll_reviews_popup(driver, wait, container):
    try:
        before = driver.execute_script(
            "return [arguments[0].scrollTop, arguments[0].scrollHeight, arguments[0].clientHeight];",
            container,
        )
        driver.execute_script(
            "arguments[0].scrollTop = arguments[0].scrollHeight;",
            container,
        )
        return container, before
    except StaleElementReferenceException:
        container = find_review_scroll_container(driver, wait)
        before = driver.execute_script(
            "return [arguments[0].scrollTop, arguments[0].scrollHeight, arguments[0].clientHeight];",
            container,
        )
        driver.execute_script(
            "arguments[0].scrollTop = arguments[0].scrollHeight;",
            container,
        )
        return container, before


# ============================================================
# MAIN
# ============================================================
def main():
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--lang=en-US")

    if HEADLESS:
        chrome_options.add_argument("--headless=new")

    # With current Selenium, Selenium Manager can manage ChromeDriver.
    driver = webdriver.Chrome(options=chrome_options)
    wait = WebDriverWait(driver, WAIT_SECONDS)

    reviews_by_id = {}
    collected_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    try:
        print(f"Opening: {APP_URL}")
        driver.get(APP_URL)

        print("Waiting for 'See all reviews' and opening the review popup...")
        click_see_all_reviews(driver, wait)

        container = find_review_scroll_container(driver, wait)
        print("Review popup opened. Starting collection...")

        no_new_rounds = 0
        last_checkpoint_count = 0

        for scroll_number in range(1, MAX_SCROLLS + 1):
            added, visible_cards = collect_currently_loaded_reviews(
                driver,
                reviews_by_id,
                collected_at_utc,
            )

            print(
                f"Scroll {scroll_number:04d} | "
                f"cards currently in DOM: {visible_cards:4d} | "
                f"new unique reviews: {added:3d} | "
                f"total unique: {len(reviews_by_id):6d}"
            )

            if added == 0:
                no_new_rounds += 1
            else:
                no_new_rounds = 0

            total_reviews = len(reviews_by_id)
            if total_reviews >= last_checkpoint_count + CHECKPOINT_EVERY_REVIEWS:
                save_excel(reviews_by_id)
                last_checkpoint_count = (total_reviews // CHECKPOINT_EVERY_REVIEWS) * CHECKPOINT_EVERY_REVIEWS
                print(
                    f"Checkpoint saved at {total_reviews} unique reviews "
                    f"-> {OUTPUT_FILE}"
                )

            if no_new_rounds >= STOP_AFTER_NO_NEW_ROUNDS:
                print(
                    f"No new review IDs for {STOP_AFTER_NO_NEW_ROUNDS} "
                    "consecutive rounds. Stopping."
                )
                break

            container, _ = scroll_reviews_popup(driver, wait, container)
            time.sleep(SCROLL_PAUSE_SECONDS)

        # One final extraction after the last scroll.
        collect_currently_loaded_reviews(driver, reviews_by_id, collected_at_utc)
        save_excel(reviews_by_id)

        print("\nFinished.")
        print(f"Unique reviews collected: {len(reviews_by_id)}")
        print(f"Excel file: {OUTPUT_FILE}")

    except TimeoutException:
        # Save anything collected before failing.
        save_excel(reviews_by_id)
        print("\nTimed out while waiting for a Google Play element.")
        print("Any reviews collected before the timeout were saved.")
        print("Check whether Google showed a consent page, CAPTCHA, or changed the page layout.")

    except KeyboardInterrupt:
        save_excel(reviews_by_id)
        print("\nStopped by user. Collected reviews were saved.")

    finally:
        print("Closing browser.")
        driver.quit()


if __name__ == "__main__":
    main()