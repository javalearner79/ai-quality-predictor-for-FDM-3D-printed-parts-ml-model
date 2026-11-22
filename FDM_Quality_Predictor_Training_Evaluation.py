# AI Quality Predictor for FDM 3D-Printed Parts - training and evaluation
# Dataset: Aktepe & Ergun (2026), Mendeley Data, DOI 10.17632/zd6td6svd6.2
# Put the dataset (.xlsx or .csv) in the same folder as this script and run it.

# ================= Cell 1: Libraries and settings =================
import warnings; warnings.filterwarnings("ignore")
import os, glob, itertools
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import joblib
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit,
                                     GridSearchCV, cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
try:
    from IPython.display import display
except Exception:
    display = print
SEED = 42
np.random.seed(SEED)
plt.rcParams["figure.figsize"] = (9, 5); plt.rcParams["axes.grid"] = True
print("Libraries ready")

# ================= Cell 2: Loading the dataset =================
DATA_FILE = None   # e.g. "FDM_dataset.xlsx"; None = auto-find or upload

def find_data_file():
    if DATA_FILE:
        return DATA_FILE
    hits = glob.glob("*.xlsx") + glob.glob("*.xls") + glob.glob("*.csv")
    if hits:
        return hits[0]
    try:
        from google.colab import files
        print("Upload the dataset file from Mendeley Data (.xlsx or .csv)")
        up = files.upload()
        return list(up.keys())[0]
    except ImportError:
        raise FileNotFoundError("Put the dataset in this folder or set DATA_FILE")

def read_table(path):
    if path.lower().endswith(".csv"):
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin-1")
    sheets = pd.read_excel(path, sheet_name=None)
    for name, d in sheets.items():       # pick the main data sheet
        cols = " ".join(map(str, d.columns)).lower()
        if "material" in cols and "rough" in cols:
            print("Using sheet:", name); return d
    name = list(sheets)[0]; print("Using first sheet:", name)
    return sheets[name]

path = find_data_file()
print("Reading file:", path)
raw = read_table(path)
raw.columns = [str(c).strip() for c in raw.columns]
print(raw.shape)
raw.head()

# ================= Cell 3: Column mapping =================
# Map the file's column names to short internal names (by keywords)
def find(*must, avoid=(), required=True):
    for c in raw.columns:
        low = c.lower()
        if all(m in low for m in must) and not any(a in low for a in avoid):
            return c
    if required:
        raise KeyError(f"No column containing {must}. Columns: {list(raw.columns)}")
    return None

COL = {
    "printer":  find("printer", required=False),
    "material": find("material"),
    "layer":    find("layer"),
    "infill":   find("infill", "dens"),
    "pattern":  find("pattern"),
    "speed":    find("speed"),
    "width":    find("width", required=False),
    "spec_t":   find("thick", avoid=("layer",), required=False),  # specimen thickness
    "stress":   find("stress"),
    "rough":    find("rough"),
    "load":     find("load", required=False),
}
print("COLUMN MAPPING (check it!)")
for k, v in COL.items():
    print(f"  {k:9s} <- {v}")
# If a line is wrong, fix it by hand, e.g. COL["layer"] = "Exact Column Name"

# ================= Cell 4: Data cleaning and preparation =================
df = pd.DataFrame()

def clean_text(s):
    s = s.astype(str).str.replace("\u200b", "", regex=False).str.strip().str.lower()
    return s.str.replace(r"\s+", " ", regex=True)

for key in ["printer", "material", "pattern"]:
    if COL[key]:
        df[key] = clean_text(raw[COL[key]])
for key in ["layer", "infill", "speed", "width", "spec_t", "stress", "rough", "load"]:
    if COL.get(key):
        df[key] = pd.to_numeric(raw[COL[key]], errors="coerce")

# Tensile strength in MPa (file gives Peak Stress; 1 MPa = 1000 kPa)
sname = COL["stress"].lower()
if "kpa" in sname:
    df["tensile"] = df["stress"] / 1000.0
elif "mpa" in sname:
    df["tensile"] = df["stress"]
else:
    print("Unit not found in the column name, assuming kPa")
    df["tensile"] = df["stress"] / 1000.0
df["roughness"] = df["rough"]

USE_SPECIMEN_DIMS = False   # width/thickness are measured AFTER printing
CAT = [c for c in ["printer", "material", "pattern"] if c in df]
NUM = ["layer", "infill", "speed"] + (
    [c for c in ["width", "spec_t"] if c in df] if USE_SPECIMEN_DIMS else [])
FEATURES = CAT + NUM
GROUP_COLS = [c for c in ["printer", "material", "layer", "infill",
                          "pattern", "speed"] if c in df]

before = len(df)
df = df.dropna(subset=FEATURES + ["tensile", "roughness"]).reset_index(drop=True)
print(f"Rows kept: {len(df)} of {before} (dropped: {before - len(df)})")

# ---- data quality checks ----
DROP_ZERO_ROUGHNESS = True   # Ra = 0 um is impossible (missing value typed as 0)
bad = df.roughness <= 0
print(f"Rows with roughness <= 0: {int(bad.sum())}")
if bad.any():
    print(df.loc[bad].groupby(["material", "layer"]).size().rename("rows").to_string())
    if DROP_ZERO_ROUGHNESS:
        df = df.loc[~bad].reset_index(drop=True)
        print("These rows were removed. Rows left:", len(df))

for c in CAT:
    print(f"\n{c} values ({df[c].nunique()}):", sorted(df[c].unique()))
print("Check for spelling variants of the same thing (e.g. 'zig zag' / 'zigzag').")

df["group"] = df[GROUP_COLS].astype(str).agg("|".join, axis=1)
groups = df["group"].factorize()[0]
print(f"Unique print-setting combinations: {df.group.nunique()} for {len(df)} rows")

TARGETS = {"Tensile strength (MPa)": "tensile", "Surface roughness (µm)": "roughness"}
LABEL = {"printer": "Printer type", "material": "Material", "pattern": "Infill pattern",
         "layer": "Layer thickness", "infill": "Infill density", "speed": "Print speed",
         "width": "Specimen width", "spec_t": "Specimen thickness"}
print("Inputs :", FEATURES)
print("Targets:", list(TARGETS))

# ================= Cell 5: Data exploration (Figure 1) =================
print(df[["layer", "infill", "speed", "tensile", "roughness"]].describe().round(2))
print("\nRows per material:"); print(df.material.value_counts().to_string())

fig, ax = plt.subplots(2, 2, figsize=(14, 9))
ax[0, 0].hist(df.tensile, bins=30); ax[0, 0].set_xlabel("Tensile strength (MPa)")
ax[0, 0].set_title("Tensile strength distribution")
ax[0, 1].hist(df.roughness, bins=30, color="#f58518"); ax[0, 1].set_xlabel("Roughness Ra (µm)")
ax[0, 1].set_title("Surface roughness distribution")
ax[1, 0].scatter(df.layer, df.roughness, s=12, alpha=.5, color="#f58518")
ax[1, 0].set_xlabel("Layer thickness (mm)"); ax[1, 0].set_ylabel("Roughness (µm)")
ax[1, 0].set_title("Roughness vs layer thickness")
order = df.groupby("material").tensile.median().sort_values().index
ax[1, 1].boxplot([df.loc[df.material == m, "tensile"] for m in order])
ax[1, 1].set_xticks(range(1, len(order) + 1))
ax[1, 1].set_xticklabels(order, rotation=45, ha="right")
ax[1, 1].set_ylabel("Tensile strength (MPa)"); ax[1, 1].set_title("Tensile strength by material")
plt.tight_layout(); plt.show()

print("Spearman, layer thickness vs roughness :", round(spearmanr(df.layer, df.roughness)[0], 3))
print("Spearman, infill density vs tensile    :", round(spearmanr(df.infill, df.tensile)[0], 3))

# ================= Cell 6: Physics check (sigma = F/A) =================
if "load" in df and "width" in df and "spec_t" in df:
    implied = df["load"] / (df["width"] * df["spec_t"])      # N / mm^2 = MPa
    ratio = (df["tensile"] / implied).replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Stress / (Peak load / (width x thickness)): median = {ratio.median():.3f}, "
          f"middle 50% between {ratio.quantile(.25):.3f} and {ratio.quantile(.75):.3f}")
    print("A median near 1 means the file agrees with sigma = F / A.")
    print("Peak load is NOT used as a model input (it would give away the answer).")
else:
    print("Peak load or specimen size not available, physics check skipped.")

# ================= Cell 7: Train/test split, training, tuning and evaluation (Table 1) =================
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr_idx, te_idx = next(gss.split(df, groups=groups))
Xtr, Xte = df.loc[tr_idx, FEATURES], df.loc[te_idx, FEATURES]
gtr = groups[tr_idx]
cv = GroupKFold(n_splits=5)
print(f"Train rows: {len(Xtr)} ({len(np.unique(gtr))} combinations) | Test rows: {len(Xte)}")

def prep(scale):
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
                              ("num", num, NUM)], sparse_threshold=0)

def build(kind):
    if kind == "lin":
        return Pipeline([("prep", prep(True)), ("model", LinearRegression())])
    return Pipeline([("prep", prep(False)),
                     ("model", RandomForestRegressor(n_estimators=300, random_state=SEED))])

PARAM_GRID = {"model__n_estimators": [200, 400],
              "model__max_depth": [None, 8, 14],
              "model__min_samples_leaf": [1, 2, 4],
              "model__max_features": ["sqrt", 1.0]}

rows, fitted, best_params, best_name = [], {}, {}, {}
for tname, tcol in TARGETS.items():
    ytr, yte = df.loc[tr_idx, tcol].values, df.loc[te_idx, tcol].values
    models = {"A. Mean (reference)": DummyRegressor(strategy="mean"),
              "B. Linear Regression": build("lin"),
              "C. Random Forest (default)": build("rf")}
    cvs = {}
    for k, m in models.items():
        s = cross_val_score(m, Xtr, ytr, cv=cv, groups=gtr, scoring="r2")
        cvs[k] = (s.mean(), s.std())
        m.fit(Xtr, ytr)
    gs = GridSearchCV(build("rf"), PARAM_GRID, cv=cv, scoring="r2",
                      n_jobs=-1).fit(Xtr, ytr, groups=gtr)
    models["D. Random Forest (tuned)"] = gs.best_estimator_
    cvs["D. Random Forest (tuned)"] = (gs.best_score_,
                                       gs.cv_results_["std_test_score"][gs.best_index_])
    best_params[tname] = {k.replace("model__", ""): v for k, v in gs.best_params_.items()}
    for k, m in models.items():
        p = m.predict(Xte)
        rows.append({"Target": tname, "Model": k,
                     "CV R2 (mean)": cvs[k][0], "CV R2 (std)": cvs[k][1],
                     "Test MAE": mean_absolute_error(yte, p),
                     "Test RMSE": np.sqrt(mean_squared_error(yte, p)),
                     "Test R2": r2_score(yte, p)})
    best_name[tname] = max([k for k in models if k[0] in "BCD"],
                           key=lambda k: cvs[k][0])      # chosen on CV, not on test
    fitted[tname] = models
    print(f"{tname}: tuned parameters = {best_params[tname]}")

res = pd.DataFrame(rows).round(3)
for tname in TARGETS:
    print("\n" + tname)
    display(res[res.Target == tname].drop(columns="Target").reset_index(drop=True))
print("Note: CV R2 of model D is the best of many tried settings, so it is slightly "
      "optimistic. The Test columns are the honest ones.")

# ================= Cell 8: Model comparison plots =================
fig, ax = plt.subplots(2, 2, figsize=(14, 9))
cols = ["#999", "#f58518", "#4c78a8", "#e45756"]
for i, tname in enumerate(TARGETS):
    r = res[res.Target == tname]
    ax[i, 0].bar(r["Model"].str[:2], r["Test R2"], color=cols)
    ax[i, 0].set_title(f"{tname}\nTest R2 (higher is better)")
    ax[i, 1].bar(r["Model"].str[:2], r["Test MAE"], color=cols)
    ax[i, 1].set_title(f"{tname}\nTest MAE (lower is better)")
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
for a, (tname, tcol) in zip(ax, TARGETS.items()):
    yte = df.loc[te_idx, tcol].values
    p = fitted[tname][best_name[tname]].predict(Xte)
    a.scatter(yte, p, s=18, alpha=.7)
    lo, hi = min(yte.min(), p.min()), max(yte.max(), p.max())
    a.plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect prediction")
    a.set_xlabel("Measured"); a.set_ylabel("Predicted"); a.legend()
    a.set_title(f"{tname}\nbest model: {best_name[tname]} (test set)")
plt.tight_layout(); plt.show()

# ================= Cell 9: Effect of specimen dimensions =================
def quick_eval(num_cols):
    feats = CAT + num_cols
    out = {}
    for tname, tcol in TARGETS.items():
        y_tr, y_te = df.loc[tr_idx, tcol].values, df.loc[te_idx, tcol].values
        for kind, nm in [("lin", "Linear Regression"), ("rf", "Random Forest (tuned settings)")]:
            pre = ColumnTransformer(
                [("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
                 ("num", StandardScaler() if kind == "lin" else "passthrough", num_cols)],
                sparse_threshold=0)
            model = (LinearRegression() if kind == "lin"
                     else RandomForestRegressor(random_state=SEED, **best_params[tname]))
            m = Pipeline([("prep", pre), ("model", model)]).fit(df.loc[tr_idx, feats], y_tr)
            p = m.predict(df.loc[te_idx, feats])
            out[(tname, nm)] = (r2_score(y_te, p), mean_absolute_error(y_te, p))
    return out

base_nums = ["layer", "infill", "speed"]
dims = [c for c in ["width", "spec_t"] if c in df]
if dims:
    a, b = quick_eval(base_nums), quick_eval(base_nums + dims)
    abl = pd.DataFrame([{"Target": k[0], "Model": k[1],
                         "Test R2 without dims": a[k][0], "Test R2 with dims": b[k][0],
                         "Test MAE without dims": a[k][1], "Test MAE with dims": b[k][1]}
                        for k in a]).round(3)
    display(abl)
    print("If 'with dims' is only slightly better (or worse), leave the dimensions out.")
    print("If much better, they may act as a proxy for other factors and are not known "
          "before printing.")
else:
    print("No specimen dimension columns found, nothing to compare.")

# ================= Cell 10: Permutation importance =================
imp = {}
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for a, (tname, tcol) in zip(ax, TARGETS.items()):
    m = fitted[tname][best_name[tname]]
    pi = permutation_importance(m, Xte, df.loc[te_idx, tcol].values,
                                n_repeats=20, random_state=SEED, scoring="r2")
    mean = pd.Series(pi.importances_mean, index=FEATURES)
    std = pd.Series(pi.importances_std, index=FEATURES)
    order_ = mean.sort_values().index
    imp[tname] = mean.sort_values(ascending=False)
    a.barh([LABEL[c] for c in order_], mean[order_].values, xerr=std[order_].values)
    a.set_xlabel("Drop in test R2 when the input is shuffled")
    a.set_title(f"{tname}\n({best_name[tname]})")
plt.tight_layout(); plt.show()
for tname in TARGETS:
    print(tname, "->", ", ".join(f"{LABEL[c]} ({v:.2f})" for c, v in imp[tname].head(3).items()))

# ================= Cell 11: Optimisation under a roughness limit =================
T_TENSILE, T_ROUGH = "Tensile strength (MPa)", "Surface roughness (µm)"
final = {tname: clone(fitted[tname][best_name[tname]]).fit(df[FEATURES], df[tcol])
         for tname, tcol in TARGETS.items()}

# ---- choose what to optimise (change these two lines freely) ----
OPT_MATERIAL = df.material.value_counts().index[0]   # default: most frequent material
ROUGHNESS_LIMIT = float(df.roughness.median())       # default: dataset median (µm)

obs = df[df.material == OPT_MATERIAL]
levels = {c: sorted(obs[c].unique())
          for c in ["layer", "infill", "speed", "pattern"] + (["printer"] if "printer" in df else [])}
cand = pd.DataFrame(list(itertools.product(*levels.values())), columns=list(levels))
cand["material"] = OPT_MATERIAL
for c in ["width", "spec_t"]:
    if c in NUM:
        cand[c] = obs[c].median()
if len(cand) > 100000:
    cand = cand.sample(100000, random_state=SEED)
cand = cand[FEATURES].reset_index(drop=True)
cand["pred_tensile"] = final[T_TENSILE].predict(cand[FEATURES])
cand["pred_rough"] = final[T_ROUGH].predict(cand[FEATURES])
ok = cand[cand.pred_rough <= ROUGHNESS_LIMIT].sort_values("pred_tensile", ascending=False)
print(f"Material: {OPT_MATERIAL} | candidates: {len(cand)} | "
      f"allowed (roughness <= {ROUGHNESS_LIMIT:.2f} µm): {len(ok)}")
show = [c for c in ["printer", "layer", "infill", "pattern", "speed",
                    "pred_tensile", "pred_rough"] if c in ok]
print("\nTop 10 suggested settings (model predictions):")
display(ok[show].head(10).round(2).reset_index(drop=True))

print("Best MEASURED specimens, same material and roughness limit (real data):")
meas = obs[obs.roughness <= ROUGHNESS_LIMIT].sort_values("tensile", ascending=False)
mshow = [c for c in ["printer", "layer", "infill", "pattern", "speed",
                     "tensile", "roughness"] if c in meas]
display(meas[mshow].head(5).round(2).reset_index(drop=True))

plt.scatter(cand.pred_rough, cand.pred_tensile, s=8, alpha=.3, color="grey", label="all candidates")
plt.scatter(ok.pred_rough, ok.pred_tensile, s=10, alpha=.6, color="tab:green", label="allowed")
plt.scatter(ok.pred_rough.head(10), ok.pred_tensile.head(10), s=60, color="red", label="top 10")
plt.axvline(ROUGHNESS_LIMIT, color="k", ls="--", lw=1, label="roughness limit")
plt.xlabel("Predicted roughness (µm)"); plt.ylabel("Predicted tensile strength (MPa)")
plt.title(f"Strength vs smoothness trade-off ({OPT_MATERIAL})"); plt.legend(); plt.show()

# ================= Cell 12: Prediction function predict_quality() =================
def predict_quality(material, layer, infill, pattern, speed,
                    printer=None, width=None, spec_t=None):
    norm = lambda v: " ".join(str(v).strip().lower().split())
    row = {"material": norm(material), "layer": layer, "infill": infill,
           "pattern": norm(pattern), "speed": speed}
    if "printer" in CAT:
        row["printer"] = norm(printer) if printer is not None else df.printer.mode()[0]
    if "width" in NUM:
        row["width"] = width if width is not None else df.width.median()
    if "spec_t" in NUM:
        row["spec_t"] = spec_t if spec_t is not None else df.spec_t.median()
    X = pd.DataFrame([row])[FEATURES]
    out = {tname: round(float(final[tname].predict(X)[0]), 2) for tname in TARGETS}
    warn = []
    for c in ["layer", "infill", "speed"]:
        if not (df[c].min() <= row[c] <= df[c].max()):
            warn.append(f"{LABEL[c]} {row[c]} is outside the training range "
                        f"{df[c].min()} to {df[c].max()}")
    for c in CAT:
        if row[c] not in set(df[c]):
            warn.append(f"{LABEL[c]} '{row[c]}' was never seen in the training data")
    out["warnings"] = warn
    return out

ex = df.iloc[0]
for inf in sorted({df.infill.min(), df.infill.max()}):
    print(f"Infill {inf}% ->",
          predict_quality(ex.material, ex.layer, inf, ex.pattern, ex.speed,
                          printer=ex.get("printer")))

# ================= Cell 13: Saving the model =================
joblib.dump({"models": final, "features": FEATURES, "cat": CAT, "num": NUM,
             "targets": TARGETS}, "fdm_quality_model.joblib")
try:
    from google.colab import files
    files.download("fdm_quality_model.joblib")
except Exception:
    print("Saved locally as fdm_quality_model.joblib")

# ================= Cell 14: Results summary =================
for tname in TARGETS:
    r = res[(res.Target == tname) & (res.Model == best_name[tname])].iloc[0]
    base = res[(res.Target == tname) & (res.Model == "A. Mean (reference)")].iloc[0]
    lin = res[(res.Target == tname) & (res.Model == "B. Linear Regression")].iloc[0]
    top3 = ", ".join(LABEL[c] for c in imp[tname].index[:3])
    print(f"{tname}\n"
          f"  chosen model (by cross-validation): {best_name[tname]}\n"
          f"  test R2 = {r['Test R2']:.3f} | test MAE = {r['Test MAE']:.3f} "
          f"| test RMSE = {r['Test RMSE']:.3f}\n"
          f"  reference: mean-only test R2 = {base['Test R2']:.3f}, "
          f"Linear Regression test R2 = {lin['Test R2']:.3f}\n"
          f"  most influential inputs: {top3}\n")
