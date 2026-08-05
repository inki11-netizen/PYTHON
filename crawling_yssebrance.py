import os
import json
import time
from datetime import datetime
import undetected_chromedriver as uc
from bs4 import BeautifulSoup

# ==========================================
# [설정 및 상수]
# ==========================================
SOURCE_ORGANIZATION = "연세대학교 세브란스병원"
START_URL = "https://health.severance.healthcare/health/encyclopedia/disease/disease.do"
BASE_URL = "https://health.severance.healthcare"

# 데이터 저장 경로 설정 및 자동 생성 (상위 폴더가 없으면 함께 생성)
OUTPUT_DIR = os.path.abspath("medical_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class SeveranceDeepCleaner:
    def __init__(self):
        self.results = []
        self.crawl_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        options = uc.ChromeOptions()
        self.driver = uc.Chrome(options=options, version_main=147)

    def clean_text(self, text):
        """텍스트 내 중복 문구 및 꼬리말/노이즈(TAG, 진료과 등) 제거"""
        if not text:
            return ""

        # 1. 하단 UI 및 불필요한 키워드 필터링
        noise_filters = ['TAG', '진료과', '유튜브', '목록', '관리자', '조회수', '이전글', '다음글']
        for filter_word in noise_filters:
            text = text.split(filter_word)[0]

        # 2. 문장 끝에 남아있는 섹션 타이틀 흔적 제거
        text = text.strip()
        text = text.rstrip("의 진단").rstrip("의 치료").rstrip("의 증상").rstrip("의 원인")

        return text.strip()

    def parse_detail(self, url):
        """질환별 상세 페이지를 파싱하여 표준 JSON 구조로 매핑"""
        self.driver.get(url)
        time.sleep(5)
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
                "definition": "",
                "symptoms": [],
                "diagnosis": {"tests": []},
                "treatment": {"description": ""},
                "prevention_and_lifestyle": ""
            },
            "source_specific": {"original_sections": {}}
        }

        # 1. 질환명(Name) 추출 및 국문/영문 분리
        title_area = soup.select_one('.title-area h3, .cont-tit, h2.tit')
        raw_title = ""

        if title_area:
            raw_title = title_area.get_text(strip=True)
        elif soup.title:  # Title 영역 부재 시 브라우저 탭 텍스트 백업 활용
            raw_title = soup.title.get_text(strip=True).split('|')[0]

        if raw_title:
            raw_title = raw_title.replace('질환찾기', '').strip()

            # '질환명 [영문명]' 패턴 분리 처리
            if '[' in raw_title and ']' in raw_title:
                ko_name = raw_title.split('[')[0].strip()
                en_name = raw_title.split('[')[1].split(']')[0].strip()
                item["medical_info"]["name"]["korean"] = ko_name
                item["medical_info"]["name"]["english"] = en_name
            else:
                item["medical_info"]["name"]["korean"] = raw_title

        # 2. HTML 태그 순회 기반의 섹션별 데이터 블록 그룹화
        content_box = soup.select_one('.content-detail, .post-detail-body, #content')
        if content_box:
            current_h = "개요"
            temp_sections = {}

            for elem in content_box.find_all(['h4', 'strong', 'p', 'div', 'li']):
                txt = elem.get_text(strip=True)
                if not txt or len(txt) < 3:
                    continue

                # h4 혹은 소제목 형태의 강한 강조 태그를 섹션 기준점으로 설정
                if elem.name in ['h4', 'strong'] and len(txt) < 30:
                    current_h = txt
                    temp_sections[current_h] = ""
                else:
                    if current_h not in temp_sections:
                        temp_sections[current_h] = ""
                    if txt not in temp_sections[current_h]:  # 중복 문장 삽입 방지
                        temp_sections[current_h] += txt + " "

            # 3. 추출 데이터 정제 및 표준 필드 매핑
            for title, content in temp_sections.items():
                cleaned_content = self.clean_text(content)
                if not cleaned_content:
                    continue

                item["source_specific"]["original_sections"][title] = cleaned_content

                # 키워드 매칭을 통한 표준 스키마 분류
                lower_title = title.lower()
                if any(k in lower_title for k in ["이란", "정의", "개요", "무엇"]):
                    item["medical_info"]["definition"] = cleaned_content
                elif "증상" in lower_title:
                    item["medical_info"]["symptoms"].append(cleaned_content)
                elif any(k in lower_title for k in ["진단", "검사"]):
                    item["medical_info"]["diagnosis"]["tests"].append(cleaned_content)
                elif any(k in lower_title for k in ["치료", "수술"]):
                    item["medical_info"]["treatment"]["description"] = cleaned_content
                elif any(k in lower_title for k in ["예방", "관리", "생활"]):
                    item["medical_info"]["prevention_and_lifestyle"] = cleaned_content

        return item

    def run(self):
        try:
            print(f"[{self.crawl_time}] 세브란스병원 질환 백과 전수 수집 시작...")
            self.driver.get(START_URL)
            time.sleep(7)

            # 상세 페이지 이동 링크 수집
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            links = []
            for a in soup.find_all('a', href=True):
                if 'articleNo=' in a['href']:
                    href = a['href']
                    full_url = f"{START_URL}{href}" if href.startswith('?') else (
                        f"{BASE_URL}{href}" if href.startswith('/') else href)
                    if full_url not in links:
                        links.append(full_url)

            print(f"총 {len(links)}개의 질환 대상 페이지를 발견했습니다.")

            # 파일 저장 경로 동적 매핑
            interim_file_path = os.path.join(OUTPUT_DIR, 'severance_refined.json')
            final_file_path = os.path.join(OUTPUT_DIR, 'severance_final_success.json')

            for i, link in enumerate(links):
                print(f"[{i + 1}/{len(links)}] 수집 중: {link}")
                data = self.parse_detail(link)
                self.results.append(data)

                # 중간 유실 방지를 위한 5주기 백업 저장
                if (i + 1) % 5 == 0:
                    with open(interim_file_path, 'w', encoding='utf-8') as f:
                        json.dump(self.results, f, ensure_ascii=False, indent=2)

            # 수집 완료 후 최종 데이터 저장
            with open(final_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            print(f"전수 수집이 성공적으로 완료되었습니다. 저장 경로: {final_file_path}")

        finally:
            self.driver.quit()


if __name__ == "__main__":
    SeveranceDeepCleaner().run()