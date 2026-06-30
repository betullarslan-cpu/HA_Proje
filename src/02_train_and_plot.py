"""
Kaydedilmis top1-top2 fark dosyasindan ozellik cikarir,
ML modellerini egitir ve grafik dosyalarini olusturur.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = "data/cosmos_t1_kayitlar_768.jsonl"
RESULTS_DIR = "results"
FIGURES_DIR = "figures"


def kayitlari_oku(path):
    kayitlar = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                kayitlar.append(json.loads(line))
    return kayitlar


def ozellik_cikar(diffs):
    x = np.array(diffs, dtype=float)
    if len(x) == 0:
        x = np.array([0.0])

    ilk10 = x[:10]
    son10 = x[-10:]

    if len(x) > 1:
        t = np.arange(len(x))
        trend = np.polyfit(t, x, 1)[0]
    else:
        trend = 0.0

    return {
        "n_tokens": len(x),
        "mean": np.mean(x),
        "std": np.std(x),
        "min": np.min(x),
        "max": np.max(x),
        "median": np.median(x),
        "q25": np.quantile(x, 0.25),
        "q75": np.quantile(x, 0.75),
        "iqr": np.quantile(x, 0.75) - np.quantile(x, 0.25),
        "mean_first10": np.mean(ilk10),
        "mean_last10": np.mean(son10),
        "prop_low_01": np.mean(x < 0.10),
        "prop_low_02": np.mean(x < 0.20),
        "trend": trend,
    }


def ozellik_tablosu(kayitlar):
    satirlar = []
    for r in kayitlar:
        o = ozellik_cikar(r["diffs"])
        o["dogru"] = int(r["dogru"])
        o["id"] = r["id"]
        satirlar.append(o)
    return pd.DataFrame(satirlar)


def train_test_ayir(df, n_test_class=50, random_state=42):
    dogru_sayi = int(df["dogru"].sum())
    yanlis_sayi = int(len(df) - dogru_sayi)
    n_class = min(dogru_sayi, yanlis_sayi)
    n_train_class = n_class - n_test_class

    if n_train_class <= 0:
        raise ValueError("Dengeli train/test ayirmak icin yeterli veri yok.")

    dogru_df = df[df["dogru"] == 1].sample(n=n_class, random_state=random_state)
    yanlis_df = df[df["dogru"] == 0].sample(n=n_class, random_state=random_state)

    dogru_train = dogru_df.sample(n=n_train_class, random_state=random_state)
    yanlis_train = yanlis_df.sample(n=n_train_class, random_state=random_state)

    dogru_test = dogru_df.drop(dogru_train.index).sample(n=n_test_class, random_state=random_state)
    yanlis_test = yanlis_df.drop(yanlis_train.index).sample(n=n_test_class, random_state=random_state)

    df_train = pd.concat([dogru_train, yanlis_train]).sample(frac=1, random_state=random_state)
    df_test = pd.concat([dogru_test, yanlis_test]).sample(frac=1, random_state=random_state)

    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)


def modelleri_egit(df_train, df_test):
    ozellikler = [c for c in df_train.columns if c not in ["dogru", "id"]]
    x_train = df_train[ozellikler]
    y_train = df_train["dogru"]
    x_test = df_test[ozellikler]
    y_test = df_test["dogru"]

    modeller = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    sonuclar = []
    tahminler = {}

    for ad, model in modeller.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        prob = model.predict_proba(x_test)[:, 1]

        sonuclar.append(
            {
                "Model": ad,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred, zero_division=0),
                "Recall": recall_score(y_test, pred, zero_division=0),
                "F1": f1_score(y_test, pred, zero_division=0),
                "ROC_AUC": roc_auc_score(y_test, prob),
            }
        )
        tahminler[ad] = {"pred": pred, "prob": prob, "cm": confusion_matrix(y_test, pred)}

    return modeller, tahminler, pd.DataFrame(sonuclar), ozellikler, x_test, y_test


def grafikleri_kaydet(df_test, modeller, tahminler, ozellikler, y_test):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.hist(df_test[df_test["dogru"] == 1]["mean"], alpha=0.65, label="Dogru cevap", bins=12)
    plt.hist(df_test[df_test["dogru"] == 0]["mean"], alpha=0.65, label="Yanlis cevap", bins=12)
    plt.xlabel("Ortalama Top1-Top2 Farki")
    plt.ylabel("Soru Sayisi")
    plt.title("Dogru ve Yanlis Cevaplarda Ortalama Top1-Top2 Farki")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig_mean_diff_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 5))
    for ad in modeller:
        fpr, tpr, _ = roc_curve(y_test, tahminler[ad]["prob"])
        auc = roc_auc_score(y_test, tahminler[ad]["prob"])
        plt.plot(fpr, tpr, label=f"{ad} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Rastgele")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Egrileri")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig_roc.png", dpi=200)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, ad in zip(axes, modeller):
        sns.heatmap(
            tahminler[ad]["cm"],
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Yanlis", "Dogru"],
            yticklabels=["Yanlis", "Dogru"],
            ax=ax,
        )
        ax.set_title(ad)
        ax.set_xlabel("Tahmin")
        ax.set_ylabel("Gercek")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig_confusion_matrix.png", dpi=200)
    plt.close()

    rf_imp = pd.Series(modeller["Random Forest"].feature_importances_, index=ozellikler).sort_values()
    gb_imp = pd.Series(modeller["Gradient Boosting"].feature_importances_, index=ozellikler).sort_values()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    rf_imp.plot(kind="barh", ax=axes[0], color="seagreen")
    axes[0].set_title("Random Forest - Feature Importance")
    gb_imp.plot(kind="barh", ax=axes[1], color="darkorange")
    axes[1].set_title("Gradient Boosting - Feature Importance")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig_feature_importance.png", dpi=200)
    plt.close()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    kayitlar = kayitlari_oku(DATA_PATH)
    df = ozellik_tablosu(kayitlar)
    df_train, df_test = train_test_ayir(df, n_test_class=50)

    modeller, tahminler, sonuc_df, ozellikler, _, y_test = modelleri_egit(df_train, df_test)

    df.to_csv(f"{RESULTS_DIR}/ozellikler_tum.csv", index=False)
    df_train.to_csv(f"{RESULTS_DIR}/train_240.csv", index=False)
    df_test.to_csv(f"{RESULTS_DIR}/test_100.csv", index=False)
    sonuc_df.to_csv(f"{RESULTS_DIR}/model_sonuclari_test100.csv", index=False)

    grafikleri_kaydet(df_test, modeller, tahminler, ozellikler, y_test)

    print(sonuc_df)


if __name__ == "__main__":
    main()
