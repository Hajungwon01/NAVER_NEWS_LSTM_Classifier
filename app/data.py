"""네이버 뉴스 카테고리별 제목 수집 모듈."""

from __future__ import annotations

from typing import List, Tuple

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
import time
import requests

import matplotlib
matplotlib.use('Agg')

def get_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    return webdriver.Chrome(options=options)


def _fetch_titles(driver, category_id: str, url_override: str = None) -> List[str]:
    """공통 제목 수집 함수 (최대 300개, 여러 페이지 순회)"""
    titles = []

    for page in range(1, 30):  # 1~5페이지 순회
        url = url_override or f"https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1={category_id}&page={page}"
        driver.get(url)
        time.sleep(2)

        for _ in range(5):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.0)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        before = len(titles)
        for a in soup.find_all("a", href=True):
            text = a.get_text(strip=True)
            if "/article/" in a["href"] and len(text) >= 10 and text not in titles:
                titles.append(text)
        print(f"  page {page} → 누적 {len(titles)}개")

        if len(titles) >= 500:
            break

    return titles

def get_news_economy(driver) -> List[Tuple[str, str]]:
    """경제 뉴스 제목 수집"""
    titles = _fetch_titles(driver, "101")
    print(f"[경제] {len(titles)}개 수집 완료")
    return [(title, "경제") for title in titles]


def get_news_society(driver) -> List[Tuple[str, str]]:
    """사회 뉴스 제목 수집"""
    titles = _fetch_titles(driver, "102")
    print(f"[사회] {len(titles)}개 수집 완료")
    return [(title, "사회") for title in titles]


def get_news_life(driver) -> List[Tuple[str, str]]:
    """생활/문화 뉴스 제목 수집"""
    titles = _fetch_titles(driver, "103", url_override="https://news.naver.com/section/103")
    print(f"[생활/문화] {len(titles)}개 수집 완료")
    return [(title, "생활/문화") for title in titles]


def get_news_it(driver) -> List[Tuple[str, str]]:
    """IT/과학 뉴스 제목 수집"""
    titles = _fetch_titles(driver, "105")
    print(f"[IT/과학] {len(titles)}개 수집 완료")
    return [(title, "IT/과학") for title in titles]

def load_sample_data() -> Tuple[List[str], List[str]]:
    """네이버 뉴스 카테고리별 제목을 수집하여 기사 문장 목록과 라벨 목록으로 반환한다."""

    driver = get_driver()
    DATA = []

    try:
        DATA += get_news_economy(driver)
        DATA += get_news_society(driver)
        DATA += get_news_life(driver)
        DATA += get_news_it(driver)

    finally:
        driver.quit()

    df = pd.DataFrame(DATA, columns=["title", "category"])
    print(f"\n총 {len(df)}개 제목 수집 완료")
    print(df.groupby("category")["title"].count())

    texts  = [title for title, _  in DATA]
    labels = [label for _,     label in DATA]
    return texts, labels