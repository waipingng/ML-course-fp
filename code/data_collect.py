from bs4 import BeautifulSoup
import csv
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm 
import time

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36'
}

def get_race_results(race_id, retries=3, delay=2):
    url = f"https://en.netkeiba.com/db/race/{race_id}/"

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            rows = soup.select('table.ResultsByRaceDetail tbody tr')

           
            race_name_tag = soup.select_one('h2.RaceName span.RaceName_main')
            race_time_tag = soup.select_one('div.RaceData span')
            track_info_tag = soup.select_one('div.RaceData span.Turf')
            weather_icon_tag = soup.select_one('span.Icon_Weather')
            grade_tag = soup.select_one('span.Icon_GradeType')

            race_name = race_name_tag.text.strip() if race_name_tag else None
            race_time = race_time_tag.text.strip() if race_time_tag else None
            track_info = track_info_tag.text.strip() if track_info_tag else None
            weather_icon = None
            if weather_icon_tag:
                for cls in weather_icon_tag.get("class", []):
                    if cls.startswith("Weather"):
                        weather_icon = cls
                        break
            grade = grade_tag.text.strip() if grade_tag else None

            race_results = []
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 17:
                    horse_link = cells[3].find('a')
                    horse_name = cells[3].text.strip()
                    horse_id = None
                    if horse_link and 'href' in horse_link.attrs:
                        href = horse_link['href']
                        parts = href.rstrip('/').split('/')
                        if len(parts) > 0:
                            horse_id = parts[-1]

                    race_results.append({
                        'Race ID': race_id,
                        'Race Name': race_name,
                        'Race Time': race_time,
                        'Track Info': track_info,
                        'Weather Icon': weather_icon,
                        'Grade': grade,
                        'Finish Position': cells[0].text.strip(),
                        'Bracket Number': cells[1].text.strip(),
                        'Horse Number': cells[2].text.strip(),
                        'Horse Name': horse_name,
                        'Horse ID': horse_id,
                        'Age/Sex': cells[4].text.strip(),
                        'Weight (kg)': cells[5].text.strip(),
                        'Jockey': cells[6].text.strip(),
                        'Final Time': cells[7].text.strip(),
                        'Margin': cells[8].text.strip(),
                        'Position at Bends': cells[9].text.strip(),
                        'Last 3F': cells[10].text.strip(),
                        'Odds': cells[11].text.strip(),
                        'Favorite': cells[12].text.strip(),
                        'Horse Weight (kg)': cells[13].text.strip(),
                        'Trainer': cells[14].text.strip(),
                        'Owner': cells[15].text.strip(),
                        'Prize (¥ mil)': cells[16].text.strip()
                    })
            return race_results
        except Exception as e:
            pass
    return None

def load_race_ids(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def save_results_to_csv(results, filename):
    if not results:
        print("No results to write.")
        return
    fieldnames = results[0].keys()
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

def main():
    race_ids = load_race_ids('race_ids.txt')
    all_results = []
    successful_ids = []
    failed_ids = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(get_race_results, rid): rid for rid in race_ids}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scraping Races"):
            race_id = futures[future]
            try:
                results = future.result()
                if results:
                    all_results.extend(results)
                    successful_ids.append(race_id)
                else:
                    failed_ids.append(race_id)
            except Exception as e:
                print(f"[{race_id}] Unexpected error: {e}")
                failed_ids.append(race_id)

    save_results_to_csv(all_results, 'race_results_with_ids.csv')

    print(f"✔️ 成功 Race ID 数量: {len(successful_ids)} / {len(race_ids)}")
    if failed_ids:
        print(f"❌ 失败的 Race ID（共 {len(failed_ids)} 个）:")
        for fid in failed_ids:
            print(f"  - {fid}")

if __name__ == '__main__':
    main()