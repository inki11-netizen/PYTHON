import os
import json
import time
import re
import traceback
import urllib.parse
from datetime import datetime
import undetected_chromedriver as uc
from bs4 import BeautifulSoup

# ==========================================
# [설정 및 상수]
# ==========================================
MAX_PER_CATEGORY = 10  # 수집 제한 건수
REQUEST_DELAY = 3.5
SOURCE_ORGANIZATION = "서울대학교병원"
BASE_URL = "https://www.snuh.org"
START_URL = "https://www.snuh.org/health/nMedInfo/nList.do"

# 데이터 및 디버그 파일 저장 경로 설정 (자동 생성)
OUTPUT_DIR = os.path.abspath("medical_data")
DEBUG_DIR = os.path.join(OUTPUT_DIR, "debug_html")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)


class SNUHCrawler:
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

    def clean_noise(self, text):
        """본문 내 불필요한 메타정보 및 꼬리말 제거"""
        if not text:
            return ""

        noise_filters = [
            '관련질환', '관련 질병', '동의어', '진료과', '관련의료진', '관련 의료진',
            '인쇄하기', '이전글', '다음글', '목록', 'SNS공유', '관련 건강정보',
            '뒤로가기', '자세히보기', '+ 더보기'
        ]

        for kw in noise_filters:
            if kw in text:
                text = text.split(kw)[0]

        junk_suffixes = ["의 증상", "의 진단", "의 치료", "의 원인"]
        text = text.strip()
        for suffix in junk_suffixes:
            if text.endswith(suffix):
                text = text[:-(len(suffix))].strip()

        return text

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
                    "main_category": "질환정보", "sub_category": "",
                    "site_category_path": ["건강정보", "의학정보"],
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
        """개별 질환 상세 페이지 파싱"""
        self.driver.get(url)
        time.sleep(4)
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        item = self.init_schema(url)

        try:
            # 1. 질환명(Name) 추출 및 정제
            title_text = ""
            title_el = soup.select_one('.sub_tit, .board_view_tit, h2.title, .title_area h1')
            if title_el:
                title_text = title_el.get_text(strip=True)
            elif soup.title:
                title_text = soup.title.get_text(strip=True).split('|')[0]

            if title_text:
                raw_name = re.sub(r'\s+', ' ', title_text).replace('N의학정보', '').strip()
                if '[' in raw_name and ']' in raw_name:
                    item["medical_info"]["name"]["korean"] = raw_name.split('[')[0].strip()
                    item["medical_info"]["name"]["english"] = raw_name.split('[')[1].split(']')[0].strip()
                else:
                    item["medical_info"]["name"]["korean"] = raw_name

            temp_sections = {}

            # 2. 본문 파싱 분기 처리 (신규 구조 vs 구형 구조)
            # 분기 A: 신규 구조 (detailWrap 기반)
            detail_wrap = soup.select_one('.detailWrap')
            if detail_wrap:
                sections = detail_wrap.select('div[id^="section-"]')
                for sec in sections:
                    h5_tag = sec.select_one('h5')
                    title = h5_tag.get_text(strip=True) if h5_tag else sec.get('id').replace('section-', '')
                    if h5_tag:
                        h5_tag.extract()
                    content = sec.get_text(separator='\n', strip=True)
                    content = re.sub(r'\s+', ' ', content).strip()
                    if content:
                        temp_sections[title] = content

            # 분기 B: 구형 구조 (태그 기반 순회 탐색)
            if not temp_sections:
                content_area = soup.select_one(
                    '#printArea, .board_view, .sub_cont, .guide_view, #content, main, article')
                if not content_area:
                    content_area = soup.body  # Fallback

                if not content_area:
                    raise Exception("웹페이지에서 텍스트 영역을 식별할 수 없습니다.")

                # 불필요한 UI 요소 사전 제거
                for util in content_area.select(
                        '.btn_wrap, .sns_wrap, .print, .share, script, style, .view_util, .blind, header, footer, nav, .gnb, .lnb'):
                    util.extract()

                # 블록/제목 태그 기준 개행 처리
                for tag in content_area.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'dt', 'strong', 'b', 'p', 'div', 'li']):
                    tag.insert_before('\n')
                    tag.insert_after('\n')
                for br in content_area.find_all('br'):
                    br.replace_with('\n')

                raw_text = content_area.get_text(separator='\n', strip=True)

                # 헤더와 본문이 공백 없이 결합된 현상 방지를 위한 정규식 처리
                headers_to_unglue = ['개요', '원인', '증상', '진단/검사', '진단', '검사', '치료', '경과/합병증', '합병증', '예방방법', '식이요법/생활가이드']
                for h in headers_to_unglue:
                    raw_text = re.sub(rf'^\s*({re.escape(h)})([가-힣a-zA-Z0-9])', rf'\1\n\2', raw_text,
                                      flags=re.MULTILINE)

                lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
                current_h = "개요"

                known_headers = ['한 줄 설명', '한줄설명', '개요', '원인', '증상', '진단/검사', '진단', '검사', '치료', '경과/합병증', '합병증', '예방방법',
                                 '예방', '식이요법/생활가이드', '식이요법']
                stop_headers = ['관련의료진', '관련건강정보', '관련질환', '관련질병', '진료과', 'faq', '인쇄하기', '뒤로가기', '의학정보']

                for line in lines:
                    clean_line = line.replace(" ", "").lower()
                    if any(clean_line.startswith(sh) for sh in stop_headers):
                        break

                    header_check = re.sub(r'^\d+\.?\s*', '', line)
                    header_check = header_check.replace(" ", "").replace("■", "").replace("-", "").replace(":",
                                                                                                           "").replace(
                        "[", "").replace("]", "").strip()

                    is_header = False
                    for kh in known_headers:
                        if header_check == kh:
                            current_h = kh
                            is_header = True
                            if current_h not in temp_sections:
                                temp_sections[current_h] = ""
                            break

                    if is_header:
                        continue

                    if current_h not in temp_sections:
                        temp_sections[current_h] = ""
                    temp_sections[current_h] += line + "\n"

            # 3. 추출된 섹션을 표준 필드에 매핑
            for title, content in temp_sections.items():
                content = content.replace('한 줄 설명', '').strip()
                cleaned = self.clean_noise(content)
                if not cleaned:
                    continue

                clean_content = re.sub(r'\s+', ' ', cleaned).strip()
                item["source_specific"]["original_sections"][title] = clean_content

                lower_title = title.lower().replace(" ", "")
                if any(k in lower_title for k in ["개요", "한줄설명", "원인"]):
                    item["medical_info"]["definition"] += clean_content + " "
                elif "증상" in lower_title:
                    item["medical_info"]["symptoms"].append(clean_content)
                elif any(k in lower_title for k in ["진단", "검사"]):
                    item["medical_info"]["diagnosis"]["tests"].append(clean_content)
                elif any(k in lower_title for k in ["치료", "수술"]):
                    item["medical_info"]["treatment"]["description"] += clean_content + " "
                elif any(k in lower_title for k in ["경과", "합병증", "예방", "식이", "생활"]):
                    item["medical_info"]["prevention_and_lifestyle"] += clean_content + " "

            # 데이터 정제 (공백 제거 등)
            item["medical_info"]["definition"] = item["medical_info"]["definition"].strip()
            item["medical_info"]["treatment"]["description"] = item["medical_info"]["treatment"]["description"].strip()
            item["medical_info"]["prevention_and_lifestyle"] = item["medical_info"]["prevention_and_lifestyle"].strip()

            # 매핑 성공 여부 검증
            is_success = bool(item["medical_info"]["symptoms"] or item["medical_info"]["treatment"]["description"])
            item["source_specific"]["extra_fields"][
                "section_parse_status"] = "standard_sections_extracted" if is_success else "raw_body_extracted"

            # 4. 진료과 추출
            depts = soup.select('.dept_list a, .rel_dept a, dd a[href*="dept"]')
            if depts:
                item["medical_info"]["related_department"] = list(set([d.get_text(strip=True) for d in depts]))

            return item

        except Exception as e:
            # 파싱 에러 시 원본 HTML 덤프 저장
            file_id = url.split('=')[-1] if '=' in url else "unknown"
            debug_path = os.path.join(DEBUG_DIR, f"snuh_{file_id}.html")
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write(soup.prettify())

            self.failed_urls.append({"url": url, "error": str(e), "debug_file": debug_path})
            item["source_specific"]["extra_fields"]["section_parse_status"] = "failed"
            item["source_specific"]["extra_fields"]["error_reason"] = str(e)
            return item

    def run(self):
        try:
            print(f"[{self.crawl_time}] 서울대학교병원 데이터 수집 시작...")
            self.driver.get(START_URL)
            time.sleep(6)

            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            detail_links = []

            # 주요 타겟(이번 주 가장 많이 찾은 질병) 링크 우선 추출
            target_title = soup.find(string=re.compile("이번 주 가장 많이 찾은 질병"))
            if target_title:
                parent_box = target_title.find_parent('div')
                if parent_box:
                    for a in parent_box.find_all('a', href=True):
                        if 'medid=' in a['href']:
                            full_url = urllib.parse.urljoin(START_URL, a['href'])
                            if full_url not in detail_links:
                                detail_links.append(full_url)

            # 타겟 영역 미발견 시 전체 질환 링크 수집 Fallback
            if not detail_links:
                for a in soup.find_all('a', href=True):
                    if 'medid=' in a['href']:
                        full_url = urllib.parse.urljoin(START_URL, a['href'])
                        if full_url not in detail_links:
                            detail_links.append(full_url)

            print(f"총 {len(detail_links)}개의 대상이 발견되었습니다. (설정된 최대 수집 건수: {MAX_PER_CATEGORY}개)")

            for i, link in enumerate(detail_links):
                if MAX_PER_CATEGORY and i >= MAX_PER_CATEGORY:
                    break

                print(f"[{i + 1}/{MAX_PER_CATEGORY}] 수집 중: {link}")
                data = self.parse_detail(link)

                if data["source_specific"]["extra_fields"]["section_parse_status"] != "failed":
                    self.results.append(data)
                    print(f"   >> 매핑 완료: {data['medical_info']['name']['korean']}")
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
        """최종 결과 및 실패 URL 목록 저장"""
        success_file_path = os.path.join(OUTPUT_DIR, 'snuh_results.json')
        failed_file_path = os.path.join(OUTPUT_DIR, 'snuh_failed_urls.json')

        with open(success_file_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        if self.failed_urls:
            with open(failed_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.failed_urls, f, ensure_ascii=False, indent=2)

        print(f"\n--- 데이터 수집 종료 ---")
        print(f"성공: {len(self.results)}건 | 실패: {len(self.failed_urls)}건")
        print(f"결과 저장 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    SNUHCrawler().run()
