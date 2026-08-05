import os
import json
import time
from datetime import datetime
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# ==========================================
# [설정 및 상수]
# ==========================================
SOURCE_ORGANIZATION = "서울아산병원"
START_URL = "https://www.amc.seoul.kr/asan/healthinfo/disease/diseaseSubmain.do"
BASE_URL = "https://www.amc.seoul.kr"

# 데이터 저장 경로 설정 및 자동 생성
# 상위 폴더가 없더라도 목적 경로(medical_data)를 포함한 디렉터리를 자동 생성함
OUTPUT_DIR = os.path.abspath("medical_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class AMCPerfectCrawler:
    def __init__(self):
        self.results = []
        self.crawl_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        options = uc.ChromeOptions()
        options.add_argument("--disable-blink-features=AutomationControlled")

        try:
            # 크롬 버전 147 환경 우선 적용
            self.driver = uc.Chrome(options=options, version_main=147)
        except:
            self.driver = uc.Chrome(options=options)

        self.wait = WebDriverWait(self.driver, 20)

    def clean_noise(self, text):
        """본문 내 불필요한 UI 텍스트 및 노이즈 제거"""
        if not text:
            return ""

        noise_filters = ['관련질환', '관련진료과', '관련의료진', '목록', '인쇄하기', '궁금해요', '유튜브', 'TAG']
        for kw in noise_filters:
            text = text.split(kw)[0]

        return text.strip()

    def parse_detail(self, url):
        """개별 질환 상세 페이지 파싱 및 표준 JSON 구조 매핑"""
        self.driver.get(url)
        time.sleep(4)
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')

        # 표준 스키마 구조 정의
        item = {
            "metadata": {
                "source_url": url,
                "source_organization": SOURCE_ORGANIZATION,
                "category": "질환정보",
                "language": "ko",
                "last_updated": self.crawl_time,
                "classification": {"main_category": "질환정보", "content_type": "disease_encyclopedia"}
            },
            "medical_info": {
                "name": {"korean": "", "english": ""},
                "entity_type": "disease",
                "definition": "",
                "symptoms": [],
                "diagnosis": {"tests": []},
                "treatment": {"description": ""},
                "prevention_and_lifestyle": "",
                "related_department": [],
                "related_tags": []
            },
            "source_specific": {"original_sections": {}}
        }

        # 1. 질환명(Name) 추출 및 국/영문 분리 (title 태그 활용)
        title_text = soup.title.get_text(strip=True) if soup.title else ""
        if title_text:
            raw_name = title_text.split('|')[0].strip()
            if '(' in raw_name and ')' in raw_name:
                item["medical_info"]["name"]["korean"] = raw_name.split('(')[0].strip()
                item["medical_info"]["name"]["english"] = raw_name.split('(')[1].split(')')[0].strip()
            else:
                item["medical_info"]["name"]["korean"] = raw_name

        # 2. 본문(요약 박스 + 하단 상세 본문) dt-dd 구조 추출 및 병합
        temp_sections = {}
        for dt in soup.find_all('dt'):
            title = dt.get_text(strip=True)
            dd = dt.find_next_sibling('dd')
            if dd:
                content = dd.get_text(separator='\n', strip=True)
                if title in temp_sections:
                    temp_sections[title] += "\n" + content
                else:
                    temp_sections[title] = content

        # 3. 추출된 데이터 정제 및 스키마 필드 매핑
        for title, content in temp_sections.items():
            cleaned = self.clean_noise(content)
            if not cleaned:
                continue

            item["source_specific"]["original_sections"][title] = cleaned

            # 키워드 매칭을 통한 표준 스키마 분류
            lower_title = title.lower()
            if any(k in lower_title for k in ["정의", "이란", "무엇", "개요"]):
                item["medical_info"]["definition"] = cleaned
            elif "증상" in lower_title:
                item["medical_info"]["symptoms"].append(cleaned)
            elif any(k in lower_title for k in ["진단", "검사"]):
                item["medical_info"]["diagnosis"]["tests"].append(cleaned)
            elif any(k in lower_title for k in ["치료", "수술"]):
                item["medical_info"]["treatment"]["description"] = cleaned
            elif any(k in lower_title for k in ["예방", "주의", "생활"]):
                item["medical_info"]["prevention_and_lifestyle"] = cleaned

        return item

    def run(self):
        try:
            print(f"[{self.crawl_time}] 서울아산병원 질환백과 수집 시작...")
            self.driver.get(START_URL)

            # 질환 상세 페이지 링크(contentId 포함 a 태그) 렌더링 대기
            self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='contentId=']")))
            time.sleep(3)

            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            detail_links = []

            # 상세 페이지 URL 리스트 추출
            for a in soup.find_all('a', href=True):
                if 'diseaseDetail.do?contentId=' in a['href']:
                    href = a['href']
                    full_url = f"{BASE_URL}{href}" if href.startswith('/') else href
                    if full_url not in detail_links:
                        detail_links.append(full_url)

            print(f"총 {len(detail_links)}개의 대상 페이지를 발견했습니다.")

            # 파일 저장 경로 동적 매핑
            interim_file_path = os.path.join(OUTPUT_DIR, 'amc_perfect_partial.json')
            final_file_path = os.path.join(OUTPUT_DIR, 'amc_perfect_results.json')

            for i, link in enumerate(detail_links):
                print(f"[{i + 1}/{len(detail_links)}] 수집 중: {link}")
                try:
                    data = self.parse_detail(link)
                    self.results.append(data)
                    print(f"   >> 매핑 완료: {data['medical_info']['name']['korean']}")
                except Exception as e:
                    print(f"   !! 오류 발생: {link} - {e}")

                # 데이터 유실 방지를 위한 5주기 백업 저장
                if (i + 1) % 5 == 0:
                    with open(interim_file_path, 'w', encoding='utf-8') as f:
                        json.dump(self.results, f, ensure_ascii=False, indent=2)

            # 수집 완료 후 최종 데이터 저장
            with open(final_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            print(f"전수 수집 완료. 저장 경로: {final_file_path}")

        finally:
            self.driver.quit()


if __name__ == "__main__":
    AMCPerfectCrawler().run()