# ==============================================================
# Boston Housing 주택가격 예측 — 전체 분석 코드
# ==============================================================
#
# 원본 데이터: Boston Housing Dataset (Harrison & Rubinfeld, 1978 논문 기반,
# UCI Machine Learning Repository 및 Kaggle에 공개된 506행 x 14열 데이터)
#
# 아래 코드는 원본 CSV(boston_housing_raw.csv)를 입력으로 사용함

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ----------------------------------------------------------------
# 1. 프로젝트 개요 — 데이터 구조/품질 점검
# ----------------------------------------------------------------
df = pd.read_csv("boston_housing_raw.csv")
df_cat = df.copy()
df_cat["CHAS"] = df_cat["CHAS"].astype("category")

print("데이터 크기:", df.shape)
print("중복행:", df.duplicated().sum())
print("결측치:", df.isnull().sum().sum())

# 기술통계량 (평균·중앙값·rel_diff·왜도·첨도·이상치)
num_cols = [c for c in df.columns if c != "CHAS"]
desc = df[num_cols].describe().T
desc["rel_diff"] = abs(desc["mean"] - desc["50%"]) / desc["50%"]
desc["iqr"] = desc["75%"] - desc["25%"]
desc["upper_bound"] = desc["75%"] + 1.5 * desc["iqr"]
desc["lower_bound"] = desc["25%"] - 1.5 * desc["iqr"]
desc["upper_outliers"] = [(df[c] > desc.loc[c, "upper_bound"]).sum() for c in num_cols]
desc["lower_outliers"] = [(df[c] < desc.loc[c, "lower_bound"]).sum() for c in num_cols]
desc["skew"] = df[num_cols].skew()
desc["kurt"] = df[num_cols].kurt()

# 절단(censoring) 신호 확인 — MEDV=50, AGE=100
print("\nMEDV=50 절단:", (df["MEDV"] == 50.0).sum(), "건")
print("AGE=100 절단 의심:", (df["AGE"] == 100.0).sum(), "건",
      "(99.0~99.9 구간은", ((df["AGE"] >= 99.0) & (df["AGE"] < 100.0)).sum(), "건뿐)")
print("RAD=24 이산점:", (df["RAD"] == 24).sum(), "건")

# ----------------------------------------------------------------
# 2. 탐색적 데이터 분석 — 이변량/다변량
# ----------------------------------------------------------------
# CHAS-MEDV: 정규성 위배 확인 후 Mann-Whitney U 검정 적용
g0 = df[df["CHAS"] == 0]["MEDV"]
g1 = df[df["CHAS"] == 1]["MEDV"]
u_stat, u_p = stats.mannwhitneyu(g0, g1, alternative="two-sided")
r_rb = 1 - (2 * u_stat) / (len(g0) * len(g1))
print(f"\nCHAS-MEDV Mann-Whitney: p={u_p:.4f}, effect r={r_rb:.3f}")

# 연속형 변수-MEDV Spearman 상관
spearman_results = []
for c in ["CRIM","ZN","INDUS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]:
    rho, p = stats.spearmanr(df[c], df["MEDV"])
    spearman_results.append({"변수": c, "rho": round(rho, 3), "p": round(p, 4)})
print("\n", pd.DataFrame(spearman_results).sort_values("rho", key=abs, ascending=False))

# 다중공선성(VIF) 진단
vif_features = ["CRIM","ZN","INDUS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]
X_vif = df[vif_features].assign(const=1)
vif_result = [{"변수": c, "VIF": round(variance_inflation_factor(X_vif.values, i), 2)}
              for i, c in enumerate(vif_features)]
print("\n", pd.DataFrame(vif_result).sort_values("VIF", ascending=False))

# ----------------------------------------------------------------
# 3. 모델링용 데이터 전처리
# ----------------------------------------------------------------
df["CRIM_log"] = np.log(df["CRIM"])
df["ZN_log"] = np.log1p(df["ZN"])
df["DIS_log"] = np.log(df["DIS"])
df["LSTAT_log"] = np.log(df["LSTAT"])
df["B_revlog"] = np.log(df["B"].max() + 1 - df["B"])
df["MEDV_log"] = np.log(df["MEDV"])
df["is_age_capped"] = (df["AGE"] == 100).astype(int)
df["is_rad_max"] = (df["RAD"] == 24).astype(int)

# ----------------------------------------------------------------
# 4. 모델링 — 11개 알고리즘 베이스라인 + 튜닝
# ----------------------------------------------------------------
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

LINEAR_FEATURES = ["CRIM_log","ZN_log","DIS_log","LSTAT_log","B_revlog",
                    "NOX","RM","AGE","TAX","PTRATIO","CHAS","is_age_capped","is_rad_max"]
TREE_FEATURES = ["CRIM_log","ZN_log","DIS_log","LSTAT_log","B_revlog",
                  "NOX","RM","AGE","RAD","TAX","PTRATIO","CHAS","is_age_capped"]

y = df["MEDV_log"]
idx_train, idx_test = train_test_split(df.index, test_size=0.2, random_state=42)

def get_split(features):
    return (df.loc[idx_train, features], df.loc[idx_test, features],
            y.loc[idx_train], y.loc[idx_test])

def evaluate(model, Xtr, Xte, ytr, yte, scale=False):
    if scale:
        scaler = StandardScaler()
        Xtr, Xte = scaler.fit_transform(Xtr), scaler.transform(Xte)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return {
        "RMSE": np.sqrt(mean_squared_error(yte, pred)),
        "MAE": mean_absolute_error(yte, pred),
        "R2": r2_score(yte, pred),
    }

Xtr_l, Xte_l, ytr, yte = get_split(LINEAR_FEATURES)
Xtr_t, Xte_t, ytr_t, yte_t = get_split(TREE_FEATURES)

results = {
    "LinearRegression": evaluate(LinearRegression(), Xtr_l, Xte_l, ytr, yte),
    "Ridge": evaluate(Ridge(random_state=42), Xtr_l, Xte_l, ytr, yte, scale=True),
    "Lasso": evaluate(Lasso(alpha=0.001, random_state=42), Xtr_l, Xte_l, ytr, yte, scale=True),
    "ElasticNet": evaluate(ElasticNet(alpha=0.001, l1_ratio=0.1, random_state=42), Xtr_l, Xte_l, ytr, yte, scale=True),
    "KNN": evaluate(KNeighborsRegressor(n_neighbors=3), Xtr_l, Xte_l, ytr, yte, scale=True),
    "SVR": evaluate(SVR(C=10, epsilon=0.1), Xtr_l, Xte_l, ytr, yte, scale=True),  # 최종 선정 모형
    "DecisionTree": evaluate(DecisionTreeRegressor(max_depth=4, random_state=42), Xtr_t, Xte_t, ytr_t, yte_t),
    "RandomForest": evaluate(RandomForestRegressor(n_estimators=400, random_state=42), Xtr_t, Xte_t, ytr_t, yte_t),
    "XGBoost": evaluate(XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.1, random_state=42), Xtr_t, Xte_t, ytr_t, yte_t),
    "LightGBM": evaluate(LGBMRegressor(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42, verbose=-1), Xtr_t, Xte_t, ytr_t, yte_t),
    "CatBoost": evaluate(CatBoostRegressor(random_state=42, verbose=0), Xtr_t, Xte_t, ytr_t, yte_t),  # 해석용 모형
}
print("\n=== 모델 비교 결과 ===")
print(pd.DataFrame(results).T.sort_values("RMSE"))

# ----------------------------------------------------------------
# 5. 분석결과 — Feature Importance & SHAP (해석용 모형: CatBoost)
# ----------------------------------------------------------------
import shap

cb_final = CatBoostRegressor(random_state=42, verbose=0)
cb_final.fit(Xtr_t, ytr_t)

# 참고: CatBoost는 random_state를 고정해도 스레드 환경에 따라 변수중요도의
# 세부 수치가 미세하게 달라질 수 있음(알려진 특성). 순위 자체는 재현됨.
fi = cb_final.get_feature_importance()
fi_df = pd.DataFrame({"변수": TREE_FEATURES, "중요도": fi}).sort_values("중요도", ascending=False)
print("\n=== CatBoost 변수 중요도 ===")
print(fi_df)

explainer = shap.TreeExplainer(cb_final)
shap_values = explainer.shap_values(Xte_t)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_df = pd.DataFrame({"변수": TREE_FEATURES, "mean|SHAP|": mean_abs_shap}).sort_values("mean|SHAP|", ascending=False)
print("\n=== SHAP 중요도 ===")
print(shap_df)

# ==============================================================
# 상세 해석·의문점 규명 과정(절단 근거 문헌 대조, TAX-RAD 다중공선성
# 처리 시도와 실패 기록, 과적합 진단 등)은 통합 리포트(md) 참고
# ==============================================================
