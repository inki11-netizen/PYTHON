import os
import json
import time
import re
from datetime import datetime
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

# ==========================================
# [설정 및 상수]
# ==========================================
REQUEST_DELAY = 3.0
SOURCE_ORGANIZATION = "질병관리청 국가건강정보포털"

START_URL = "https://health.kdca.go.kr/healthinfo/biz/health/gnrlzHealthInfo/gnrlzHealthInfo/gnrlzHealthInfoMain.do?lclasSn=1"

# 수집 대상 질환 목록
TARGET_DISEASES = [
    "건강노화", "근 손실", "남성갱년기", "노인기능평가", "변실금",
    "불규칙한 월경주기와 월경전증후군", "인지기능저하예방법", "정상 월경의 이해",
    "정상노화의 이해", "1형 양극성장애", "2형 양극성장애", "A형간염"
]

# 데이터 및 디버그 파일 저장 경로 설정 (상위 폴더 자동 생성)
OUTPUT_DIR = os.path.abspath("medical_data")
DEBUG_DIR = os.path.join(OUTPUT_DIR, "debug_html")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)


class KDCACrawler:
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
        """다중 공백 및 줄바꿈 정제"""
        if not text:
            return ""
        return re.sub(r'\s+', ' ', text).strip()

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
                    "main_category": "질환정보",
                    "sub_category": "건강/질병 정보",
                    "site_category_path": ["건강정보", "건강/질병 정보"],
                    "content_type": "disease_encyclopedia",
                    "entity_type": "disease"
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

    def parse_detail(self, url, disease_name, html_source):
        """개별 질환 페이지 파싱 로직"""
        soup = BeautifulSoup(html_source, 'html.parser')
        item = self.init_schema(url)

        try:
            item["medical_info"]["name"]["korean"] = disease_name

            # 본문 DOM 분기 처리 (KDCA 내부 페이지 vs 연계 외부 페이지)
            content_divs = soup.find_all('div', id=re.compile(r'^contentsDiv\d+'))

            temp_sections = {}

            # 분기 A: KDCA 표준 템플릿
            if content_divs and content_divs[0].name == 'div':
                for div in content_divs:
                    h3_tag = div.find(['h3', 'h4', 'h2'])
                    if h3_tag:
                        title = h3_tag.get_text(strip=True)
                        h3_tag.extract()  # 제목 텍스트와 본문 분리
                    else:
                        title = "기타"

                    content = div.get_text(separator=' ', strip=True)
                    if content:
                        temp_sections[title] = content

            # 분기 B: 외부 연계 템플릿 (양극성장애 등)
            else:
                main_area = soup.select_one('.board_view, .view_area, .con_wrap, #contents, article')
                if not main_area:
                    main_area = soup.body

                current_h = "개요"
                temp_sections[current_h] = ""

                # 불필요한 UI 태그 제거
                for util in main_area.select('script, style, nav, header, footer, .gnb, .lnb, .dropdown, .sch_wrap'):
                    util.extract()

                # 태그 순회 기반 섹션 분리
                for tag in main_area.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'p', 'div', 'li']):
                    text = tag.get_text(separator=' ', strip=True)
                    if not text:
                        continue

                    if tag.name in ['h1', 'h2', 'h3', 'h4', 'h5']:
                        current_h = text
                        if current_h not in temp_sections:
                            temp_sections[current_h] = ""
                    else:
                        if current_h not in temp_sections:
                            temp_sections[current_h] = ""
                        temp_sections[current_h] += text + " "

            # 텍스트 데이터 표준 스키마 매핑
            known_headers_map = {
                'definition': ['요약문', '개요', '정의', '원인', '질병개요', '건강문제', '위험요인', '일반적특징'],
                'symptoms': ['증상', '주요증상', '원인및증상', '연관증상', '임상증상'],
                'diagnosis': ['진단', '검사', '진단및검사', '진단/검사', '검사방법', '진단검사', '진단기준'],
                'treatment': ['치료', '치료법', '치료방법', '수술'],
                'prevention': ['예방', '예방법', '합병증', '경과', '생활가이드', '예방및생활습관', '생활습관관리', '관련질환']
            }

            diag_keywords = ['진단', '검사', '내시경', '초음파', '촬영', 'x선', 'mri', 'ct', '혈액', '조직검사', '문진', '인지기능평가']

            for title, content in temp_sections.items():
                clean_title = title.replace(" ", "").lower()
                clean_content = self.clean_noise(content)
                if not clean_content:
                    continue

                item["source_specific"]["original_sections"][title] = clean_content

                # 진단/검사 관련 문장 정밀 추출 (정규식 기반 문장 분리)
                sentences = re.split(r'(?<=[.!?])\s+', clean_content)
                for s in sentences:
                    if any(kw in s.lower() for kw in diag_keywords) and len(s) > 10:
                        if s.strip() not in item["medical_info"]["diagnosis"]["tests"]:
                            item["medical_info"]["diagnosis"]["tests"].append(s.strip())

                # 분야별 매핑
                target_field = None
                for field, kr_list in known_headers_map.items():
                    if any(k in clean_title for k in kr_list):
                        target_field = field
                        break

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

            # 공백 마무리 정리
            item["medical_info"]["definition"] = item["medical_info"]["definition"].strip()
            item["medical_info"]["treatment"]["description"] = item["medical_info"]["treatment"]["description"].strip()
            item["medical_info"]["prevention_and_lifestyle"] = item["medical_info"]["prevention_and_lifestyle"].strip()

            # 매핑 성공 여부 검증
            is_success = bool(item["medical_info"]["definition"] or item["medical_info"]["symptoms"] or
                              item["medical_info"]["treatment"]["description"])
            item["source_specific"]["extra_fields"][
                "section_parse_status"] = "standard_sections_extracted" if is_success else "raw_body_extracted"

            return item

        except Exception as e:
            # 파싱 에러 시 원본 HTML 덤프 저장
            file_id = url.split('=')[-1] if '=' in url else str(int(time.time()))
            debug_path = os.path.join(DEBUG_DIR, f"kdca_{file_id}.html")
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write(soup.prettify())

            self.failed_urls.append({"url": url, "error": str(e), "debug_file": debug_path})
            item["source_specific"]["extra_fields"]["section_parse_status"] = "failed"
            item["source_specific"]["extra_fields"]["error_reason"] = str(e)
            return item

    def run(self):
        print(f"[{self.crawl_time}] 국가건강정보포털 대상 질환({len(TARGET_DISEASES)}건) 수집 시작...")

        for i, disease in enumerate(TARGET_DISEASES):
            print(f"\n[{i + 1}/{len(TARGET_DISEASES)}] 수집 준비: '{disease}'")

            try:
                # 브라우저 세션 유효성 검사 및 복구
                try:
                    self.driver.current_url
                except:
                    print("   >> 브라우저 세션 오류 감지. 재시작을 시도합니다.")
                    options = uc.ChromeOptions()
                    options.add_argument("--disable-blink-features=AutomationControlled")
                    self.driver = uc.Chrome(options=options, version_main=147)
                    self.driver.implicitly_wait(10)

                self.driver.get(START_URL)
                time.sleep(4)

                # '전체' 탭 클릭 시도
                try:
                    all_tab = self.driver.find_element(By.XPATH, "//a[contains(text(), '전체') or contains(., '전체')]")
                    self.driver.execute_script("arguments[0].click();", all_tab)
                    time.sleep(2)
                except:
                    pass

                # 타겟 질환 요소 탐색
                xpath = f"//*[text()[contains(., '{disease}')]]/ancestor-or-self::a | //*[text()[contains(., '{disease}')]]/ancestor-or-self::*[@onclick]"
                elements = self.driver.find_elements(By.XPATH, xpath)

                target_link = None
                for el in elements:
                    if el.is_displayed():
                        target_link = el
                        break

                if not target_link:
                    raise Exception(f"요소 탐색 실패: '{disease}' 버튼을 찾을 수 없습니다.")

                self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", target_link)
                time.sleep(1)

                original_windows = self.driver.window_handles

                # 상세 페이지 진입
                try:
                    target_link.click()
                except:
                    self.driver.execute_script("arguments[0].click();", target_link)

                time.sleep(4)
                new_windows = self.driver.window_handles

                # 새 창(외부 링크) 또는 현재 창 여부에 따른 분기 처리
                if len(new_windows) > len(original_windows):
                    print(f"   -> 외부 템플릿 페이지 진입 감지. 파싱을 시작합니다.")
                    self.driver.switch_to.window(new_windows[-1])
                    data = self.parse_detail(self.driver.current_url, disease, self.driver.page_source)
                    self.driver.close()
                    self.driver.switch_to.window(original_windows[0])
                else:
                    print(f"   -> 표준 템플릿 페이지 진입 완료. 파싱을 시작합니다.")
                    data = self.parse_detail(self.driver.current_url, disease, self.driver.page_source)

                # 파싱 결과 평가
                if data["source_specific"]["extra_fields"]["section_parse_status"] != "failed":
                    self.results.append(data)
                    print(f"   >> 매핑 완료: {disease}")
                else:
                    error_reason = data["source_specific"]["extra_fields"].get("error_reason", "알 수 없는 에러")
                    print(f"   >> 파싱 실패: {error_reason}")

            except Exception as e:
                print(f"   >> '{disease}' 수집 중 예외 발생 (스킵 처리): {e}")

            time.sleep(REQUEST_DELAY)

        self.save_results()

    def save_results(self):
        """최종 결과 및 실패 내역 저장"""
        success_file_path = os.path.join(OUTPUT_DIR, 'kdca_results.json')
        failed_file_path = os.path.join(OUTPUT_DIR, 'kdca_failed_urls.json')

        with open(success_file_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        if self.failed_urls:
            with open(failed_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.failed_urls, f, ensure_ascii=False, indent=2)

        print(f"\n--- 국가건강정보포털 데이터 수집 종료 ---")
        print(f"성공: {len(self.results)}건 | 실패: {len(self.failed_urls)}건")
        print(f"결과 저장 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    KDCACrawler().run()