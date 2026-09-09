# 📁 Portfolio_ML
 
데이터 수집부터 전처리, 분석, 시각화, 모델링까지 진행한 프로젝트를 정리한 저장소입니다.
 
---
 
## 📑 목차
 
| # | 프로젝트명 | 한 줄 설명 | 사용 기술 | 링크 |
|---|---|---|---|---|
| 1 | **Boston Housing(보스턴 주택 시장 예측)** | 1970년대 보스턴 506개 지역 데이터로 주택가격 결정요인을 규명하고 예측모델(SHAP 해석 포함) 구축 | `Python`, `Pandas`, `Scikit-learn`, `CatBoost`, `SHAP` | [🔗 바로가기](#2-boston-housing-주택가격-예측) |
| 2 | **Diamonds Prices(다이아몬드 가격 예측)** | 4C(캐럿·컷·컬러·투명도) 등급 데이터로 가격 결정요인을 규명하고 예측모델(SHAP 해석 포함) 구축 | `Python`, `Pandas`, `Scikit-learn`, `CatBoost`, `SHAP` | [🔗 바로가기](#2-diamond_price-다이아몬드-가격-예측) |
| 3 | *(추가 예정)* | | | |
 
---
 
## 1. Boston Housing(보스턴 주택 시장 예측)

- Background

주제 : 1970년대 보스턴 지역 주택가격 결정요인 규명 및 예측모델 구축

> 주제 선정 배경 : 강사님 제공 자료는 데이터 품질점검까지만 되어있고, 가설·해석·모델링은 제공되지 않음 <br/>
> 기존 : itwill LAB03(California 주택가격 예측)이 코드+인사이트+해석까지 갖춘 완성형 예시로 별도 존재 <br/>
> ---> California 방법론(6단계: 개요-EDA-전처리-모델링-결과-결론)을 그대로 적용해 Boston Housing을 완성형 포트폴리오로 재구성

- Summary

(1). Data Collection
- 수집대상 : 1970년 보스턴 SMSA 내 506개 census tract, 변수 14개(CRIM~MEDV)
- 수집 출처 : Harrison & Rubinfeld(1978) 원논문 → UCI ML Repository → Kaggle(altavish/boston-housing-dataset) → jussam 라이브러리(강사 제작)

(2). Data Preprocessing
- 결측치·중복행 0건 확인
- 왜도 기준 로그변환 4종(CRIM·ZN·DIS·LSTAT) + 역방향 로그변환 1종(B)
- MEDV($50,000)·AGE(100%) 상한 절단 신호 실측 확인 → 종속변수는 누수 방지 위해 플래그 미생성, 독립변수(AGE)만 플래그화
- TAX-RAD 다중공선성(VIF 8.88·7.40) 확인 → 모델 계열별 변수 표현 분기 처리

(3). Model & Algorithms
- 회귀 모델링 및 변수중요도 기반 해석

> 모델링 과정<br/>
> 프로세스 : 로그변환·플래그 생성 → 학습/검증 분할(8:2) → 계열별 전처리(스케일링/VIF) → 11종 베이스라인 비교 → 하이퍼파라미터 튜닝 → 최종모형 선정 <br/>
> 주요 투입 변수 : 저소득층비율(LSTAT), 평균방수(RM), 고용센터거리(DIS), 대기오염도(NOX), 범죄율(CRIM), 재산세율(TAX), 학생교사비율(PTRATIO) 등 13개 <br/>
> 모델링 : 1. 선형계열 - LinearRegression·Ridge·Lasso·ElasticNet (다중공선성 처리 필수) <br/>
> 2. 비선형/거리기반 - KNN·SVR (스케일링 필수) <br/>
> 3. 트리/앙상블 계열 - DecisionTree·RandomForest·XGBoost·LightGBM·CatBoost <br/>
> 4. 최종 선정 - RMSE 기준 근소격차 그룹핑 후 보조지표(MAE·R²)로 결함 점검하여 SVR 채택

- 해석 모형 (SHAP 분석용)

> 예측 성능 1위(SVR)와 해석 가능성을 함께 고려해, SHAP 분석은 트리 계열인 CatBoost로 별도 수행<br/>
> 저소득층비율(LSTAT)·방수(RM) 단 2개 변수가 가격 변동의 약 50% 설명<br/>
> 고용센터거리(DIS)는 역U자형 비선형 관계 확인 — 너무 가깝거나 너무 멀면 불리, 중간 거리가 최적

(4). Review
- Train-CV 격차(84%)가 표본 크기(506행)의 구조적 한계로 남음 — 변수·모델 복잡도 조정으로도 해소되지 않음
- 종속변수가 $50,000에서 상한 절단되어 있어, 고가 주택 구간(검증셋 절단 관측치 3건)은 모델이 일관되게 과소예측함 → 고가 구간 예측에는 부적합
- 원 데이터의 수집 목적(대기오염 정책평가)과 현재 분석 목적(가격예측)이 달라, 실무 적용 시 이 괴리를 반드시 고지해야 함

보러가기: [Boston Housing 포트폴리오](https://github.com/Jun1048/Portfolio_ML/tree/main/Boston%20Housing)
 
---

## 2. Diamonds Prices(다이아몬드 가격 예측)

- Background

주제 : 다이아몬드 4C(캐럿·컷·컬러·투명도) 등급 기반 가격 결정요인 규명 및 예측모델 구축
> 주제 선정 배경 : 앞서 완성한 Boston Housing 포트폴리오와 동일한 6단계 방법론(개요-EDA-전처리-모델링-결과-결론)을 새로운 도메인(다이아몬드 소매시장)에 적용해 두 번째 완성형 포트폴리오로 구성

- Summary

(1). Data Collection

- 수집대상 : 원형 브릴리언트 컷 다이아몬드 53,940건, 변수 10개(price~z)
- 수집 출처 : 온라인 다이아몬드 판매 플랫폼 원자료 → 공개 데이터셋 등재 → Kaggle 재배포본

(2). Data Preprocessing

- 결측치 0건, 물리적으로 불가능한 0값(x·y·z) 20건 삭제, 중복행 146건은 삭제 대신 그룹보존 방식으로 처리
- 왜도 기준 로그변환 2종(price·carat) — 왜도 각각 1.618→0.116, 1.116→0.581로 개선
- carat·x·y·z 다중공선성(VIF 20~63) 확인 → x·y·z 제거, carat 단독 채택(VIF 1점대로 해소)
- 캐럿 "매직사이즈"(0.5·0.7·1.0·1.5·2.0) 이산점 실측 확인 → 근접 플래그 파생변수 생성

(3). Model & Algorithms

- 회귀 모델링 및 SHAP 기반 해석

> 모델링 과정
> 프로세스 : 로그변환·순서형 인코딩·플래그 생성 → 그룹보존 학습/검증 분할(8:2) → 계열별 전처리(스케일링) → 11종 베이스라인 비교 → 하이퍼파라미터 튜닝 → 최종모형 선정
> 주요 투입 변수 : 캐럿(carat), 컷·컬러·투명도 등급, 깊이비율(depth)·테이블비율(table), 매직사이즈 근접 플래그 등 7개
> 모델링 : 1. 선형계열 - LinearRegression·Ridge·Lasso·ElasticNet
> 2. 비선형/거리기반 - KNN·SVR
> 3. 트리/앙상블 계열 - DecisionTree·RandomForest·XGBoost·LightGBM·CatBoost
> 4. 최종 선정 - RMSE 기준 근소격차 그룹핑 후 보조지표(MAE·R²)로 결함 점검하여 CatBoost 채택(Test RMSE 0.0948, R² 0.9913)

- SHAP 분석

> 캐럿(carat) 단 하나가 가격 변동의 약 77% 설명
> 4C 등급(컷·컬러·투명도)은 원시 데이터에서는 등급이 낮을수록 평균가가 오히려 높아 보이는 역설이 있었으나, 캐럿을 통제한 SHAP 순수효과에서는 등급이 높을수록 가격 기여도가 정상적으로 증가함을 확인(교란관계 해소)

(4). Review

- 변수 축소(7→3개) 시도는 성능을 15.92% 악화시켜 사전 규칙에 따라 기각 — 개별 중요도가 낮다고 항상 제거해도 되는 건 아님을 확인
- 중복행 146건이 단순 데이터 입력 오류인지 서로 다른 실물 다이아몬드인지는 최종적으로 확인 불가
- 데이터 수집 시점·화폐 기준연도가 명시돼 있지 않아 확인 불가로 남음

보러가기: [diamond_price 포트폴리오](https://github.com/Jun1048/Portfolio_ML/tree/main/Diamonds%20Prices)


---


## 🛠 Skills & Tools
 
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-4479A1?style=flat-square&logo=mysql&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square)
![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?style=flat-square)
![SHAP](https://img.shields.io/badge/SHAP-1E90FF?style=flat-square)

---
 
## 📬 Contact
 
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:eett308@gmail.com)
 
대표 프로필 페이지: [Jun1048/Jun1048 →](https://github.com/Jun1048/Jun1048)
