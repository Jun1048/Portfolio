# ==============================================================
# Diamond Price Prediction 다이아몬드 가격 예측 — 전체 분석 코드
# ==============================================================
#
# 원본 데이터: diamonds 데이터셋(Kaggle 재배포본, 53,940행 x 10열)
#
# 아래 코드는 원본 CSV(diamonds_price_raw.csv)를 입력으로 사용함

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

RANDOM_STATE = 42

# ----------------------------------------------------------------
# 1. 프로젝트 개요 — 데이터 구조/품질 점검
# ----------------------------------------------------------------
df = pd.read_csv("diamonds_price_raw.csv")

print("데이터 크기:", df.shape)
print("중복행:", df.duplicated().sum())
print("결측치:", df.isnull().sum().sum())

# 기술통계량 (연속형: 평균·중앙값·왜도·첨도·이상치 / 범주형: count·unique·top·freq)
num_cols = ["price", "carat", "x", "y", "z", "depth", "table"]
cat_cols = ["cut", "color", "clarity"]

desc = df[num_cols].describe().T
desc["iqr"] = desc["75%"] - desc["25%"]
desc["upper_bound"] = desc["75%"] + 1.5 * desc["iqr"]
desc["lower_bound"] = desc["25%"] - 1.5 * desc["iqr"]
desc["upper_outliers"] = [(df[c] > desc.loc[c, "upper_bound"]).sum() for c in num_cols]
desc["lower_outliers"] = [(df[c] < desc.loc[c, "lower_bound"]).sum() for c in num_cols]
desc["skew"] = df[num_cols].skew()
desc["kurt"] = df[num_cols].kurt()

for c in cat_cols:
    vc = df[c].value_counts()
    print(f"\n{c}: count={df[c].count()}, unique={df[c].nunique()}, "
          f"top={vc.idxmax()}, freq={vc.max()}")

# 물리적 이상치(0값) 확인 — 결측치에 준하는 문제
print("\nx=0:", (df["x"] == 0).sum(), "건")
print("y=0:", (df["y"] == 0).sum(), "건")
print("z=0:", (df["z"] == 0).sum(), "건")
print("y 최댓값:", df["y"].max(), " / z 최댓값:", df["z"].max())

# 캐럿 매직사이즈 이산점 확인
for t in [0.5, 0.7, 1.0, 1.5, 2.0]:
    shy = ((df["carat"] >= t - 0.05) & (df["carat"] < t)).sum()
    at = (df["carat"] == t).sum()
    print(f"임계값 {t}: 직전(shy) {shy}건 / 정확히 {at}건")

# ----------------------------------------------------------------
# 2. 탐색적 데이터 분석 — 이변량/다변량
# ----------------------------------------------------------------
df_clean = df.loc[~((df["x"] == 0) | (df["y"] == 0) | (df["z"] == 0))].copy()

# 연속형 변수-price Spearman 상관
cont_vars = ["carat", "x", "y", "z", "depth", "table"]
spearman_results = []
for c in cont_vars:
    rho, p = stats.spearmanr(df_clean[c], df_clean["price"])
    spearman_results.append({"변수": c, "rho": round(rho, 3), "p": round(p, 4)})
print("\n", pd.DataFrame(spearman_results).sort_values("rho", key=abs, ascending=False))

# 범주형(cut/color/clarity) - price Welch ANOVA
for c in cat_cols:
    groups = [g["price"].values for _, g in df_clean.groupby(c)]
    lev_stat, lev_p = stats.levene(*groups)
    f_stat, f_p = stats.f_oneway(*groups)
    print(f"\n{c} - price ANOVA: F={f_stat:.2f}, p={f_p:.4g} "
          f"(Levene p={lev_p:.4g})")

# 4C 등급-가격 역설(교란) 확인: 등급별 평균가 vs 평균캐럿
print("\ncut별 평균가/평균캐럿:")
print(df_clean.groupby("cut")[["price", "carat"]].mean())

# 다중공선성(VIF) 진단 — x,y,z 제거 전
X_vif = df_clean[["carat", "x", "y", "z", "depth", "table"]]
vif_result = [{"변수": c, "VIF": round(variance_inflation_factor(X_vif.values, i), 2)}
              for i, c in enumerate(X_vif.columns)]
print("\n", pd.DataFrame(vif_result).sort_values("VIF", ascending=False))

# ----------------------------------------------------------------
# 3. 모델링용 데이터 전처리
# ----------------------------------------------------------------
cut_order = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
color_order = {"J": 0, "I": 1, "H": 2, "G": 3, "F": 4, "E": 5, "D": 6}
clarity_order = {"I1": 0, "SI2": 1, "SI1": 2, "VS2": 3, "VS1": 4,
                  "VVS2": 5, "VVS1": 6, "IF": 7}

df_clean["cut_encoded"] = df_clean["cut"].map(cut_order)
df_clean["color_encoded"] = df_clean["color"].map(color_order)
df_clean["clarity_encoded"] = df_clean["clarity"].map(clarity_order)

df_clean["log_price"] = np.log1p(df_clean["price"])
df_clean["log_carat"] = np.log1p(df_clean["carat"])

thresholds = [0.5, 0.7, 1.0, 1.5, 2.0]
df_clean["is_near_magic_size"] = df_clean["carat"].apply(
    lambda c: int(any((t - 0.05 <= c < t) for t in thresholds))
)

dup_cols = ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z", "price"]
df_clean["dup_group_id"] = df_clean.groupby(dup_cols).ngroup()

# x, y, z 제거 후 VIF 재확인
X_vif2 = df_clean[["carat", "depth", "table"]]
vif_result2 = [{"변수": c, "VIF": round(variance_inflation_factor(X_vif2.values, i), 2)}
               for i, c in enumerate(X_vif2.columns)]
print("\nx,y,z 제거 후 VIF:\n", pd.DataFrame(vif_result2))

# ----------------------------------------------------------------
# 4. 모델링 — 11개 알고리즘 베이스라인 + 튜닝
# ----------------------------------------------------------------
from sklearn.model_selection import GroupShuffleSplit, KFold
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

FEATURE_COLS = ["log_carat", "cut_encoded", "color_encoded",
                 "clarity_encoded", "depth", "table", "is_near_magic_size"]

X = df_clean[FEATURE_COLS]
y = df_clean["log_price"]
groups = df_clean["dup_group_id"]

gss = GroupShuffleSplit(test_size=0.2, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X, y, groups=groups))
Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
ytr, yte = y.iloc[train_idx], y.iloc[test_idx]


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


results = {
    "LinearRegression": evaluate(LinearRegression(), Xtr, Xte, ytr, yte),
    "Ridge": evaluate(Ridge(random_state=RANDOM_STATE), Xtr, Xte, ytr, yte, scale=True),
    "Lasso": evaluate(Lasso(alpha=0.0001, random_state=RANDOM_STATE), Xtr, Xte, ytr, yte, scale=True),
    "ElasticNet": evaluate(ElasticNet(alpha=0.0001, l1_ratio=0.2, random_state=RANDOM_STATE), Xtr, Xte, ytr, yte, scale=True),
    "KNN": evaluate(KNeighborsRegressor(n_neighbors=10), Xtr, Xte, ytr, yte, scale=True),
    "SVR": evaluate(SVR(C=5, epsilon=0.05), Xtr, Xte, ytr, yte, scale=True),
    "DecisionTree": evaluate(DecisionTreeRegressor(max_depth=12, random_state=RANDOM_STATE), Xtr, Xte, ytr, yte),
    "RandomForest": evaluate(RandomForestRegressor(n_estimators=400, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1), Xtr, Xte, ytr, yte),
    "XGBoost": evaluate(XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05, random_state=RANDOM_STATE, verbosity=0), Xtr, Xte, ytr, yte),
    "LightGBM": evaluate(LGBMRegressor(n_estimators=600, max_depth=-1, learning_rate=0.05, random_state=RANDOM_STATE, verbosity=-1), Xtr, Xte, ytr, yte),
    "CatBoost": evaluate(CatBoostRegressor(iterations=600, depth=8, learning_rate=0.1, random_state=RANDOM_STATE, verbose=0), Xtr, Xte, ytr, yte),  # 최종 선정 모형
}

print("\n=== 모델 비교 결과(튜닝 후) ===")
print(pd.DataFrame(results).T.sort_values("RMSE"))

# 5-Fold 교차검증(학습셋 내부) — 과적합 진단용
cb_params = dict(iterations=600, depth=8, learning_rate=0.1, random_state=RANDOM_STATE, verbose=0)
cb_final = CatBoostRegressor(**cb_params)
cb_final.fit(Xtr, ytr)
pred_tr = cb_final.predict(Xtr)
train_rmse = np.sqrt(mean_squared_error(ytr, pred_tr))

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = []
for tr_i, val_i in kf.split(Xtr):
    m = CatBoostRegressor(**cb_params)
    m.fit(Xtr.iloc[tr_i], ytr.iloc[tr_i])
    p = m.predict(Xtr.iloc[val_i])
    cv_scores.append(np.sqrt(mean_squared_error(ytr.iloc[val_i], p)))
cv_scores = np.array(cv_scores)
gap_pct = (cv_scores.mean() - train_rmse) / max(abs(train_rmse), abs(cv_scores.mean())) * 100
print(f"\nTrain RMSE={train_rmse:.4f}  CV RMSE={cv_scores.mean():.4f}(+/-{cv_scores.std():.4f})  Gap%={gap_pct:.2f}%")

# 변수 축소(7개 -> 3개) 검증 — 사전 규칙: 5% 이상 악화되면 되돌림
reduced_cols = ["log_carat", "clarity_encoded", "color_encoded"]
red_result = evaluate(CatBoostRegressor(**cb_params), Xtr[reduced_cols], Xte[reduced_cols], ytr, yte)
change_pct = (red_result["RMSE"] - results["CatBoost"]["RMSE"]) / results["CatBoost"]["RMSE"] * 100
print(f"\n변수 축소(3개) 후 RMSE={red_result['RMSE']:.4f}  변화율={change_pct:+.2f}% "
      f"-> {'기각(원 모델 유지)' if abs(change_pct) >= 5 else '채택 가능'}")

# ----------------------------------------------------------------
# 5. 분석결과 — Feature Importance & SHAP (최종 모형: CatBoost)
# ----------------------------------------------------------------
import shap

fi = cb_final.get_feature_importance()
fi_df = pd.DataFrame({"변수": FEATURE_COLS, "중요도": fi}).sort_values("중요도", ascending=False)
print("\n=== CatBoost 변수 중요도 ===")
print(fi_df)

explainer = shap.TreeExplainer(cb_final)
rng = np.random.RandomState(RANDOM_STATE)
sample_idx = rng.choice(len(Xte), size=min(800, len(Xte)), replace=False)
X_sample = Xte.iloc[sample_idx]
shap_values = explainer.shap_values(X_sample)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_df = pd.DataFrame({"변수": FEATURE_COLS, "mean|SHAP|": mean_abs_shap}).sort_values("mean|SHAP|", ascending=False)
print("\n=== SHAP 중요도 ===")
print(shap_df)

print("\n=== 최종 성능 ===")
print(f"Test RMSE={results['CatBoost']['RMSE']:.4f}  "
      f"R2={results['CatBoost']['R2']:.4f}  MAE={results['CatBoost']['MAE']:.4f}")
