import os
import json
import time
import re
import traceback
from datetime import datetime
import undetected_chromedriver as uc
from bs4 import BeautifulSoup

# ==========================================
# [설정 및 상수]
# ==========================================
REQUEST_DELAY = 3.0
SOURCE_ORGANIZATION = "건강보험심사평가원"

TARGET_URLS = [
    "https://www.hira.or.kr/ra/stcIlnsInfm/stcIlnsInfmView.do?pgmid=HIRAA030502000000&sortSno=189",
    "https://www.hira.or.kr/ra/stcIlnsInfm/stcIlnsInfmView.do?pgmid=HIRAA030502000000&sortSno=190",
    "https://www.hira.or.kr/ra/stcIlnsInfm/stcIlnsInfmView.do?pgmid=HIRAA030502000000&sortSno=191",
    "https://www.hira.or.kr/ra/stcIlnsInfm/stcIlnsInfmView.do?pgmid=HIRAA030502000000&sortSno=192",
    "https://www.hira.or.kr/ra/stcIlnsInfm/stcIlnsInfmView.do?pgmid=HIRAA030502000000&sortSno=195",
    "https://www.hira.or.kr/ra/stcIlnsInfm/stcIlnsInfmView.do?pgmid=HIRAA030502000000&sortSno=193",
    "https://www.hira.or.kr/ra/stcIlnsInfm/stcIlnsInfmView.do?pgmid=HIRAA030502000000&sortSno=194",
    "https://www.hira.or.kr/ra/stcIlnsInfm/stcIlnsInfmView.do?pgmid=HIRAA030502000000&sortSno=196",
    "https://www.hira.or.kr/ra/stcIlnsInfm/stcIlnsInfmView.do?pgmid=HIRAA030502000000&sortSno=197",
    "https://www.hira.or.kr/ra/stcIlnsInfm/stcIlnsInfmView.do?pgmid=HIRAA030502000000&sortSno=198"
]

# 데이터 및 디버그 파일 저장 경로 설정 (상위 폴더 자동 생성)
OUTPUT_DIR = os.path.abspath("medical_data")
DEBUG_DIR = os.path.join(OUTPUT_DIR, "debug_html")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)


class HIRACrawler:
    def __init__(self):
        self.results = []
        self.failed_urls = []
        self.crawl_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        options = uc.ChromeOptions()
        options.add_argument("--disable-blink-features=AutomationControlled")
        try:
            self.driver = uc.Chrome(options=options, version_main=147)
        except:
            self.driver = uc.Chrome(options=options)

        self.driver.implicitly_wait(10)

    def init_schema(self, url):
        """표준 JSON 스키마 초기화"""
        return {
            "metadata": {
                "source_url": url,
                "source_organization": SOURCE_ORGANIZATION,
                "category": "질환정보",
                "language": "ko",
                "last_updated": self.crawl_time,
                "classification": {
                    "main_category": "질환정보", "sub_category": "통계로 보는 질병정보",
                    "site_category_path": ["건강정보", "의료정보", "통계로 보는 질병정보"],
                    "content_type": "disease_encyclopedia", "entity_type": "disease"
                }
            },
            "medical_info": {
                "name": {"korean": "", "english": ""},
                "entity_type": "disease",
                "definition": "",
                "symptoms": [],
                "diagnosis": {"description": "", "tests": [], "screening": ""},
                "treatment": {"methods": [], "description": ""},
                "prevention_and_lifestyle": "",
                "related_department": [],
                "related_tags": []
            },
            "source_specific": {"original_sections": {}, "extra_fields": {"section_parse_status": "initialized"}}
        }

    def parse_detail(self, url):
        """질환 상세 데이터 파싱 및 매핑 로직"""
        self.driver.get(url)
        time.sleep(4)
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        item = self.init_schema(url)

        try:
            # 1. 차트 제목 및 텍스트 패턴 기반 질환명 추출
            raw_text_for_name = soup.get_text(separator=' ')
            raw_name = ""

            # 패턴 1: 그림 번호 + [질병명] + 최근 5년
            match1 = re.search(r'그림\s*\d+\.?\s*(.*?)\s*최근\s*5년', raw_text_for_name)
            if match1:
                raw_name = match1.group(1).strip()

            # 패턴 2: 그림 번호 + [질병명] + 특정 연도
            if not raw_name:
                match2 = re.search(r'그림\s*\d+\.?\s*(.*?)\s*20\d{2}년', raw_text_for_name)
                if match2:
                    raw_name = match2.group(1).strip()

            # 패턴 3: 타이틀 태그 Fallback
            if not raw_name:
                title_el = soup.select_one('.board-title, .title, .tit, .con_title, h2, h3, h4.tit')
                if title_el:
                    raw_name = title_el.get_text(strip=True).replace('통계로 보는 질병정보', '').replace('의료정보', '').strip()

            # 질환명 국/영문 분리 처리
            if raw_name:
                raw_name = re.sub(r'\s+', ' ', raw_name).strip()
                match = re.search(r'^(.*?)[\[\(](.*?)[\]\)]', raw_name)
                if match:
                    item["medical_info"]["name"]["korean"] = match.group(1).strip()
                    item["medical_info"]["name"]["english"] = match.group(2).strip()
                else:
                    item["medical_info"]["name"]["korean"] = raw_name
            else:
                item["medical_info"]["name"]["korean"] = "이름추출실패"

            # 2. 본문 텍스트 추출 및 헤더 분리
            content_area = soup.body
            if not content_area:
                raise Exception("웹페이지 데이터를 식별할 수 없습니다.")

            # 블록 및 제목 태그 기준 개행 삽입
            for tag in content_area.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'dt', 'strong', 'b', 'p', 'div', 'li']):
                tag.insert_before('\n')
                tag.insert_after('\n')
            for br in content_area.find_all('br'):
                br.replace_with('\n')

            raw_text = content_area.get_text(separator='\n', strip=True)

            # 헤더와 본문 텍스트 결합 방지를 위한 정규식 처리
            headers_to_unglue = ['질병 개요', '위험요인 및 증상', '치료 및 예방', '질병 정보 치료 및 예방', '최근 5년간 환자수 현황', '1인당 진료현황']
            for h in headers_to_unglue:
                raw_text = re.sub(rf'(?m)^\s*({re.escape(h)})([가-힣a-zA-Z0-9])', rf'\1\n\2', raw_text)

            lines = [line.strip() for line in raw_text.split('\n') if line.strip()]

            # 표준 필드 매핑 딕셔너리
            known_headers_map = {
                '질병개요': 'definition', '개요': 'definition', '정의': 'definition', '원인': 'definition', '질병의정의': 'definition',
                '증상': 'symptoms', '주요증상': 'symptoms', '위험요인및증상': 'symptoms',
                '진단': 'diagnosis', '검사': 'diagnosis', '진단및검사': 'diagnosis', '진단/검사': 'diagnosis',
                '치료': 'treatment', '치료법': 'treatment',
                '예방': 'prevention', '예방법': 'prevention', '경과': 'prevention', '합병증': 'prevention', '주의사항': 'prevention',
                '생활가이드': 'prevention',
                '치료및예방': 'treatment_and_prevention', '질병정보치료및예방': 'treatment_and_prevention'
            }

            stop_headers = ['목록', '인쇄', '페이스북', '트위터', '블로그', '담당부서', '최종수정일', '만족도', '이전글', '다음글', '첨부파일', '다운로드']

            temp_sections = {}
            current_h = None

            # 줄 단위 순회를 통한 섹션 할당
            for line in lines:
                clean_line = line.replace(" ", "").lower()
                clean_line = re.sub(r'^\d+\.?\s*', '', clean_line)
                for char in ['■', '-', ':', '○', 'ㅇ', '▶', '>', '[', ']']:
                    clean_line = clean_line.replace(char, '')

                if any(clean_line.startswith(sh) for sh in stop_headers):
                    current_h = None
                    continue

                matched_header = None
                for kh in known_headers_map.keys():
                    if clean_line == kh:
                        matched_header = kh
                        break

                if matched_header:
                    current_h = matched_header
                    if current_h not in temp_sections:
                        temp_sections[current_h] = ""
                    continue

                if current_h:
                    temp_sections[current_h] += line + " "

            # 3. 데이터 정제 및 스키마 매핑
            diag_keywords = ['진단', '검사', '내시경', '초음파', '촬영', 'x선', 'mri', 'ct', '혈액']

            for title, content in temp_sections.items():
                content = content.replace('한 줄 설명', '').strip()
                if not content:
                    continue

                clean_content = re.sub(r'\s+', ' ', content).strip()
                item["source_specific"]["original_sections"][title] = clean_content

                # 진단 및 검사 관련 문장 정밀 추출 (정규식 기반)
                sentences = re.split(r'(?<=[.!?])\s+', clean_content)
                for s in sentences:
                    if any(kw in s.lower() for kw in diag_keywords) and len(s) > 8:
                        if s.strip() not in item["medical_info"]["diagnosis"]["tests"]:
                            item["medical_info"]["diagnosis"]["tests"].append(s.strip())

                target_field = known_headers_map[title]

                if target_field == 'definition':
                    item["medical_info"]["definition"] += clean_content + " "
                elif target_field == 'symptoms':
                    item["medical_info"]["symptoms"].append(clean_content)
                elif target_field == 'diagnosis':
                    if clean_content not in item["medical_info"]["diagnosis"]["tests"]:
                        item["medical_info"]["diagnosis"]["tests"].append(clean_content)
                elif target_field == 'treatment':
                    item["medical_info"]["treatment"]["description"] += clean_content + " "
                elif target_field == 'prevention':
                    item["medical_info"]["prevention_and_lifestyle"] += clean_content + " "
                elif target_field == 'treatment_and_prevention':
                    item["medical_info"]["treatment"]["description"] += clean_content + " "
                    item["medical_info"]["prevention_and_lifestyle"] += clean_content + " "

            # 공백 정제
            item["medical_info"]["definition"] = item["medical_info"]["definition"].strip()
            item["medical_info"]["treatment"]["description"] = item["medical_info"]["treatment"]["description"].strip()
            item["medical_info"]["prevention_and_lifestyle"] = item["medical_info"]["prevention_and_lifestyle"].strip()

            # 파싱 성공 여부 평가
            is_success = bool(item["medical_info"]["definition"] or item["medical_info"]["symptoms"] or
                              item["medical_info"]["treatment"]["description"])
            item["source_specific"]["extra_fields"][
                "section_parse_status"] = "standard_sections_extracted" if is_success else "raw_body_extracted"

            return item

        except Exception as e:
            # 파싱 에러 시 원본 HTML 덤프 저장
            file_id = url.split('=')[-1] if '=' in url else str(int(time.time()))
            debug_path = os.path.join(DEBUG_DIR, f"hira_{file_id}.html")
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write(soup.prettify())

            self.failed_urls.append({"url": url, "error": str(e), "debug_file": debug_path})
            item["source_specific"]["extra_fields"]["section_parse_status"] = "failed"
            item["source_specific"]["extra_fields"]["error_reason"] = str(e)
            return item

    def run(self):
        try:
            print(f"[{self.crawl_time}] 건강보험심사평가원 대상 질환({len(TARGET_URLS)}건) 수집 시작...")

            for i, link in enumerate(TARGET_URLS):
                print(f"\n[{i + 1}/{len(TARGET_URLS)}] 파싱 중: {link}")
                data = self.parse_detail(link)

                if data["source_specific"]["extra_fields"]["section_parse_status"] != "failed":
                    self.results.append(data)
                    name = data['medical_info']['name']['korean']
                    print(f"   >> 매핑 완료: {name}")
                else:
                    error_reason = data["source_specific"]["extra_fields"].get("error_reason", "알 수 없는 에러")
                    print(f"   >> 파싱 실패: {error_reason}")

                time.sleep(REQUEST_DELAY)

            self.save_results()

        except Exception as e:
            print(f"치명적 오류 발생:\n{traceback.format_exc()}")
        finally:
            self.driver.quit()

    def save_results(self):
        """최종 결과 및 실패 내역 저장"""
        success_file_path = os.path.join(OUTPUT_DIR, 'hira_results.json')
        failed_file_path = os.path.join(OUTPUT_DIR, 'hira_failed_urls.json')

        with open(success_file_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        if self.failed_urls:
            with open(failed_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.failed_urls, f, ensure_ascii=False, indent=2)

        print(f"\n--- 건강보험심사평가원 데이터 수집 종료 ---")
        print(f"성공: {len(self.results)}건 | 실패: {len(self.failed_urls)}건")
        print(f"결과 저장 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    HIRACrawler().run()