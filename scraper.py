#1_scraper.py
import random
import time
import json
import re
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import os
from datetime import datetime
import csv
from pathlib import Path
import threading



class WebScrapeOtodom:
    def __init__(self, start_page=1, end_page=3):
        self.start_page = start_page
        self.end_page = end_page
        self.all_links = set()
        self.driver = None

    def _setup_driver(self):
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service)

    def scrape_links(self):
        self._setup_driver()
        try:
            for page in range(self.start_page, self.end_page + 1):
                url = (
                    "https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/mazowieckie/"
                    f"warszawa/warszawa/warszawa?viewType=listing&limit=72&page={page}"
                )
                print(f"Przetwarzam stronę: {url}")
                self.driver.get(url)
                time.sleep(random.uniform(2, 4))

                elements = self.driver.find_elements(By.XPATH, '//a[contains(@href, "/pl/oferta/")]')
                for elem in elements:
                    href = elem.get_attribute("href")
                    if href and "/pl/oferta/" in href:
                        self.all_links.add(href)
        finally:
            self.driver.quit()

        return list(self.all_links)

    def _get_description(self, _url: str) -> str:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(_url, headers=headers)
        if response.status_code != 200:
            return f"Nie udało się pobrać opisu (status: {response.status_code})."


        soup_desc = BeautifulSoup(response.text, 'html.parser')
        description_element = soup_desc.find('div', {'data-cy': 'adPageAdDescription'})
        if description_element:
            raw_html = str(description_element)
            desc_soup = BeautifulSoup(raw_html, 'html.parser')
            description_text = desc_soup.get_text(separator="\n", strip=True)
            if len(description_text) > 10:
                return description_text

        script_tag_desc = soup_desc.find('script', id='__NEXT_DATA__')
        if not script_tag_desc:
            return "Brak opisu w HTML i brak skryptu __NEXT_DATA__."

        try:
            data_desc = json.loads(script_tag_desc.string)
            description_json = (
                data_desc
                .get('props', {})
                .get('pageProps', {})
                .get('ad', {})
                .get('description', '')
            )
            if description_json:
                desc_soup = BeautifulSoup(description_json, 'html.parser')
                clean_text = desc_soup.get_text(separator="\n", strip=True)
                return clean_text if len(clean_text) > 0 else "Nie znaleziono istotnej treści w __NEXT_DATA__."
            else:
                return "Nie znaleziono opisu w __NEXT_DATA__."
        except json.JSONDecodeError:
            return "Błąd dekodowania JSON z __NEXT_DATA__."


    def _get_label_value(self, soup: BeautifulSoup, html: str, label: str, json_key: str = None) -> str:
        if not json_key:
            return 'brak danych'

        script = soup.find('script', id='__NEXT_DATA__')
        if not script or not script.string:
            return 'brak danych'

        try:
            data = json.loads(script.string)
            characteristics = (
                data.get('props', {})
                    .get('pageProps', {})
                    .get('ad', {})
                    .get('characteristics', [])
            )
            for ch in characteristics:
                if ch.get('key') == json_key:
                    return ch.get('localizedValue') or 'brak danych'
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass

        return 'brak danych'


    def scrape_offer(self, url: str) -> list:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return [
                "Cena: brak danych",
                "Tytuł: brak danych",
                f"Opis: Nie udało się pobrać strony (status: {response.status_code}).",
                "Powierzchnia: brak danych",
                "Liczba pokoi: brak danych",
                "Piętro: brak danych",
                "Czynsz: brak danych",
                "Rynek: brak danych",
                "Rok budowy: brak danych",
                "Informacje dodatkowe: brak danych",
                "Stan wykończenia: brak danych",
                "Winda: brak danych",
                "Adres: brak danych"
            ]

        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')

        # Cena i tytuł
        price_element = soup.find('strong', {'aria-label': 'Cena'})
        price = price_element.get_text(strip=True) if price_element else 'brak danych'
        title_element = soup.find('h1')
        title = title_element.get_text(strip=True) if title_element else 'brak danych'

        # Opis 
        description = self._get_description(url)
        description = re.sub(r'\s+', ' ', description).strip()

        # --- drobna inicjalizacja, by fallback mógł działać poza blokiem try ---
        features = []

        area = rooms = floor_info = rent = market_type = build_year = additional_info = 'brak danych'
        script_tag = soup.find('script', id='__NEXT_DATA__')
        if script_tag:
            try:
                data = json.loads(script_tag.string)
                ad = data.get('props', {}).get('pageProps', {}).get('ad', {})
                target = ad.get('target', {})

                # Powierzchnia 
                area_raw = target.get('Area')
                if area_raw is not None:
                    s = str(area_raw).replace(',', '.')
                    m = re.search(r'\d+(?:\.\d+)?', s)
                    area = m.group(0) if m else 'brak danych'

                # Liczba pokoi 
                rooms_raw = target.get('Rooms_num')
                if isinstance(rooms_raw, list):
                    rooms_raw = rooms_raw[0] if rooms_raw else None
                rooms = str(rooms_raw) if rooms_raw not in (None, '', []) else 'brak danych'

                # Piętro 
                floor_list = target.get('Floor_no', [])
                building_floors = target.get('Building_floors_num')  
                if floor_list:
                    raw_floor = floor_list[0]
                    match_floor = re.search(r'floor_(\d+)', raw_floor)
                    if match_floor:
                        floor_no = match_floor.group(1)
                    else:
                        floor_no = '0' if raw_floor in ['floor_0', 'floor_parter', 'parter', 'ground_floor'] else 'brak danych'
                else:
                    floor_no = 'brak danych'
                floor_info = floor_no

                # Czynsz
                rent_val = target.get('Rent')
                rent = f"{rent_val} zł" if rent_val else 'brak danych'

                # Rynek
                market = ad.get('market')
                if market:
                    market_type = 'wtórny' if market == 'SECONDARY' else 'pierwotny' if market == 'PRIMARY' else str(market).lower()
                else:
                    market_type = 'brak danych'

                # Rok budowy
                build_year_val = target.get('Build_year')
                build_year = str(build_year_val) if build_year_val else 'brak danych'

                # Informacje dodatkowe
                features = ad.get('features', [])  # <-- ZAPAMIĘTAJ, użyjemy jako fallbacku
                additional_info = ', '.join(features) if features else 'brak danych'

            except json.JSONDecodeError:
                pass  

        # --- PODMIANA: solidne pobieranie Stanu wykończenia i Windy + fallback z features ---
        
        # --- STAN WYKOŃCZENIA z JSON (characteristics) z fallbackiem do HTML ---
        state = self._get_label_value(
            soup, html_content, "Stan wykończenia", json_key="construction_status"
        )

        # --- WINDA: spróbuj też przez JSON (jeśli kiedyś będzie w characteristics), potem HTML ---
        elevator = self._get_label_value(
            soup, html_content, "Winda", json_key="lift"
        )


        # Fallback z features, jeśli dalej 'brak danych'
        features_lower = [f.lower() for f in features] if isinstance(features, list) else []
        if elevator == 'brak danych' and features_lower:
            if any('winda' in f for f in features_lower):
                elevator = 'tak'
            elif any('bez windy' in f or 'no lift' in f for f in features_lower):
                elevator = 'nie'

        addr_elem = soup.find('a', href='#map')
        address = addr_elem.get_text(strip=True) if addr_elem else 'brak danych'

        return [
            f"Cena: {price}",
            f"Tytuł: {title}",
            f"Opis: {description}",
            f"Powierzchnia: {area}",
            f"Liczba pokoi: {rooms}",
            f"Piętro: {floor_info}",
            f"Czynsz: {rent}",
            f"Rynek: {market_type}",
            f"Rok budowy: {build_year}",
            f"Informacje dodatkowe: {additional_info}",
            f"Stan wykończenia: {state}",
            f"Winda: {elevator}",
            f"Adres: {address}"
        ]





if __name__ == "__main__":
    START_PAGE = 1
    END_PAGE = 267
    DATA_DIR = Path("data")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    OFFER_LINKS_PATH = DATA_DIR / "offer_links.txt"
    FINAL_CSV_PATH = DATA_DIR / "otodom_offers.csv"
    CHECKPOINT_EVERY = 1000
    CHECKPOINT_PREFIX = "raw_checkpoint_"
    RESUME_FROM_EXISTING = True  # <- ustaw False, jeśli chcesz zawsze start od zera

    def write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            return

        # 1) Zbierz pełny schemat jako unia kluczy
        all_keys = set()
        for r in rows:
            all_keys.update(r.keys())

        # (opcjonalnie) ustaw preferowaną kolejność — np. offer_url na początku
        preferred = ["offer_url"]
        other_keys = sorted(k for k in all_keys if k not in preferred)
        fieldnames = [k for k in preferred if k in all_keys] + other_keys

        # 2) Zapis z uzupełnianiem braków
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                safe_row = {k: r.get(k, "") for k in fieldnames}
                w.writerow(safe_row)


    def read_csv(path: Path) -> list[dict]:
        if not path.exists():
            return []
        with path.open("r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            return list(r)

    def find_last_checkpoint(dir_path: Path, prefix: str = CHECKPOINT_PREFIX) -> tuple[int, Path | None]:
        """
        Zwraca (max_n, path) dla pliku w formacie raw_checkpoint_{n}.csv.
        Jeśli brak, zwraca (0, None).
        """
        max_n = 0
        max_path = None
        for p in dir_path.glob(f"{prefix}*.csv"):
            try:
                n = int(p.stem.replace(prefix, ""))
                if n > max_n:
                    max_n, max_path = n, p
            except ValueError:
                continue
        return max_n, max_path

    # 1) Zbierz/odczytaj linki do ofert
    if OFFER_LINKS_PATH.exists():
        print(f"[INFO] Czytam linki z pliku: {OFFER_LINKS_PATH}")
        with OFFER_LINKS_PATH.open("r", encoding="utf-8") as f:
            links = [ln.strip() for ln in f if ln.strip()]
        # deduplikacja z zachowaniem kolejności
        seen = set()
        links = [x for x in links if not (x in seen or seen.add(x))]
    else:
        print("[INFO] Plik z linkami nie istnieje — skrobię linki z listingu...")
        scraper_tmp = WebScrapeOtodom(start_page=START_PAGE, end_page=END_PAGE)
        links = scraper_tmp.scrape_links()
        # deduplikacja z zachowaniem kolejności
        seen = set()
        links = [x for x in links if not (x in seen or seen.add(x))]
        with OFFER_LINKS_PATH.open("w", encoding="utf-8") as f:
            for ln in links:
                f.write(ln + "\n")
        print(f"[INFO] Zapisano {len(links)} linków do {OFFER_LINKS_PATH}")

    total_links = len(links)
    print(f"[INFO] Do przetworzenia linków: {total_links}")

    # 2) Wznawianie: najpierw spróbuj z FINAL, potem z checkpointu
    all_data: list[dict] = []
    processed = 0
    resume_from_idx = 0  # zero-based; odpowiada liczbie już przetworzonych

    if RESUME_FROM_EXISTING:
        if FINAL_CSV_PATH.exists():
            print(f"[RESUME] Wykryto plik końcowy: {FINAL_CSV_PATH}. Wznawiam z niego.")
            all_data = read_csv(FINAL_CSV_PATH)
            processed = len(all_data)
            resume_from_idx = processed
            print(f"[RESUME] Załadowano {processed} rekordów. Start od linku #{resume_from_idx + 1}.")
        else:
            max_n, cp_path = find_last_checkpoint(DATA_DIR, CHECKPOINT_PREFIX)
            if cp_path is not None and max_n > 0:
                print(f"[RESUME] Wykryto checkpoint: {cp_path.name} (n={max_n}). Wznawiam z niego.")
                all_data = read_csv(cp_path)
                processed = len(all_data)
                if processed != max_n:
                    print(f"[WARN] Rozbieżność: w pliku {cp_path.name} jest {processed} wierszy, nazwa sugeruje {max_n}. "
                          f"Używam {processed} jako punktu wznowienia.")
                resume_from_idx = processed
                print(f"[RESUME] Załadowano {processed} rekordów. Start od linku #{resume_from_idx + 1}.")
            else:
                print("[RESUME] Brak pliku końcowego i checkpointów — start od zera.")

    # 3) Główna pętla scrapingu
    scraper = WebScrapeOtodom(start_page=START_PAGE, end_page=END_PAGE)

    # iterujemy od miejsca wznowienia
    for idx, url in enumerate(links[resume_from_idx:], resume_from_idx + 1):
        print(f"\n[INFO] Oferta {idx}/{total_links}: {url}")
        try:
            details = scraper.scrape_offer(url)
            entry = {}
            for line in details:
                if ": " in line:
                    key, val = line.split(": ", 1)
                    entry[key] = val
            # (opcjonalnie) zapisz URL — jeśli nie chcesz zmieniać schematu, zakomentuj
            entry.setdefault("offer_url", url)

            all_data.append(entry)
            processed += 1
        except Exception as e:
            print(f"[WARN] Błąd podczas przetwarzania oferty nr {idx}: {e}")

        if processed > 0 and processed % CHECKPOINT_EVERY == 0:
            checkpoint_path = DATA_DIR / f"{CHECKPOINT_PREFIX}{processed}.csv"
            write_csv(checkpoint_path, all_data)
            print(f"[CHECKPOINT] Zapisano {checkpoint_path} ({processed} rekordów).")

        time.sleep(random.uniform(0.5, 1.5))

    # 4) Zapis finałowy
    if all_data:
        write_csv(FINAL_CSV_PATH, all_data)
        print(f"[DONE] Zapisano dane do {FINAL_CSV_PATH} (łącznie {len(all_data)} rekordów).")
    else:
        print("[DONE] Brak danych do zapisania.")
