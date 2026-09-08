# 📁 Portfolio_ML
 
데이터 수집부터 전처리, 분석, 시각화, 모델링까지 진행한 프로젝트를 정리한 저장소입니다.
 
---
 
## 📑 목차
 
| # | 프로젝트명 | 한 줄 설명 | 사용 기술 | 링크 |
|---|---|---|---|---|
| 1 | **한국 의료 취약지구 분석** | 229개 시군구 대상 의료 접근성 격차를 시각화·군집화·회귀분석하여 정책 제언 도출 | `Python`, `Pandas`, `Clustering`, `Regression`, `GeoPandas` | [🔗 바로가기](#1-한국-의료-취약지구-분석) |
| 2 | *(추가 예정)* | | | |
| 3 | *(추가 예정)* | | | |
 
---
 
## 1. 한국 의료 취약지구 분석
 
### Background
전국 229개 시군구 간 의료 접근성 격차가 존재한다는 문제의식에서 출발하여, 어느 지역이 실질적인 의료 취약지구인지 데이터 기반으로 진단하고자 함.
 
### Summary
 
**(1) Data Collection**
- 건강보험공단(HIRA) 및 통계청 자료 기반, 229개 시군구 × 11개 변수 데이터셋 구축
- 미충족 의료 필요율(`sra_01z3`) 등 핵심 변수 추출
**(2) Data Preprocessing**
- 원자료(`chs25_all.sas7bdat`) 전처리 및 시군구 단위 집계
- 지역 코드(`PBHLTH_CODE`) 매핑, 시군구 경계 shapefile 결합
**(3) Analysis Pipeline**
1. 현황 시각화 — 지역별 의료 접근성 지표 지도 시각화
2. 클러스터링 — 유사 특성을 가진 취약지구 군집화
3. 회귀분석 — 접근성에 영향을 미치는 주요 변수 도출
4. 정책 제언 — 분석 결과 기반 우선 지원 지역 및 개선 방향 제안
**(4) Status**
- 진행 중 (전처리 및 데이터 결합 단계)
*보러가기: [저장소 링크](여기에_실제_프로젝트_repo_링크)*
 
---
 
## 🛠 Skills & Tools
 
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-4479A1?style=flat-square&logo=mysql&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square)
 
---
 
## 📬 Contact
 
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:eett308@gmail.com)
 
대표 프로필 페이지: [Jun1048/Jun1048 →](https://github.com/Jun1048/Jun1048)
