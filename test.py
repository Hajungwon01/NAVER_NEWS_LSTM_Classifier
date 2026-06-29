from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time

options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")
driver = webdriver.Chrome(options=options)

driver.get("https://m.sports.naver.com/wfootball")
time.sleep(2)

for _ in range(5):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(1.5)

soup = BeautifulSoup(driver.page_source, "html.parser")
all_a = soup.find_all("a", href=True)
print("전체 a 태그 수:", len(all_a))

# 텍스트 있는 링크 전부 출력
for a in all_a:
    text = a.get_text(strip=True)
    href = a.get("href", "")
    if len(text) >= 10:
        print(repr(href[:80]), "|", text[:40])

driver.quit()