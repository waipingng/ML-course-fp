
import requests
from bs4 import BeautifulSoup
import re
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


tunnel = "z404.kdltps.com:15818"
username = "t14539262724614"
password = "b44oxcj8"
proxies = {
    "http": f"http://{username}:{password}@{tunnel}/",
    "https": f"http://{username}:{password}@{tunnel}/"
}

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36'
}

def fetch_race_ids(page):
    url = f"https://en.netkeiba.com/db/race/race_list.html?pid=race_list&start_year=1980&end_year=2025&grade[0]=1&grade[1]=2&grade[2]=3&page={page}"
    race_ids = set()
    try:
        response = requests.get(url, headers=headers, proxies=proxies, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        race_links = soup.find_all("a", href=True)
        for link in race_links:
            match = re.search(r"/race/(\d{12})", link["href"])
            if match:
                race_ids.add(match.group(1))
    except Exception as e:
        pass
    return race_ids

all_race_ids = set()
pages = range(1, 680)

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(fetch_race_ids, page): page for page in pages}
    for future in tqdm(as_completed(futures), total=len(pages), desc="抓取进度"):
        result = future.result()
        all_race_ids.update(result)


with open("race_ids.txt", "w") as f:
    for race_id in sorted(all_race_ids):
        f.write(race_id + "\n")

print(f"\n总共抓取 {len(all_race_ids)} 个比赛ID，已保存到 race_ids.txt")
