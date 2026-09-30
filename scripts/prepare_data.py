"""Загрузка, проверка и подготовка признаков. Запуск из корня проекта:
python scripts/prepare_data.py"""
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize

CSV = "../gamerecfiles/games2.csv"
EMB = "../gamerecfiles/emb.npy"

NEED = ["Name", "Genres", "Tags", "Categories", "Price", "Positive", "Negative",
        "Estimated owners", "Average playtime forever", "Windows", "Mac", "Linux",
        "Header image", "About the game"]


def clean(s):
    return s.fillna("").astype(str)


def make_text(d):
    return ("Genres: " + clean(d["Genres"]) + ". Tags: " + clean(d["Tags"])
            + ". " + clean(d["About the game"]).str.slice(0, 1000))


def to_bool(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().map({"true": True, "false": False}).fillna(False)


def load():
    d = pd.read_csv(CSV, low_memory=False)
    emb = np.load(EMB)
    return d, emb


def fix_types(d, log=print):
    """Приводит типы и проверяет/считает расчётные колонки."""
    for c in ["Price", "Positive", "Negative", "Average playtime forever"]:
        if not pd.api.types.is_numeric_dtype(d[c]):
            log(f"  тип {c}: {d[c].dtype} -> число")
            d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ["Windows", "Mac", "Linux"]:
        if d[c].dtype != bool:
            log(f"  тип {c}: {d[c].dtype} -> bool")
            d[c] = to_bool(d[c])

    rt = d["Positive"] + d["Negative"]
    if "reviews_total" in d:
        log(f"  reviews_total есть, == Positive+Negative: {(d['reviews_total'] == rt).all()}")
    else:
        log("  reviews_total нет, считаю"); d["reviews_total"] = rt
    pr = (d["Positive"] + 7) / (rt + 10)
    if "pos_ratio" in d:
        log(f"  pos_ratio есть, совпадает с формулой: {np.allclose(d['pos_ratio'], pr)}")
    else:
        log("  pos_ratio нет, считаю"); d["pos_ratio"] = pr
    if "year" in d:
        log(f"  year есть: min={d['year'].min()}, max={d['year'].max()}, NaN={d['year'].isna().sum()}")
    else:
        log("  year нет, считаю")
        d["year"] = pd.to_datetime(d["Release date"], errors="coerce").dt.year
    if "owners_mid" in d:
        log(f"  owners_mid есть: min={d['owners_mid'].min()}, max={d['owners_mid'].max()}")
    else:
        log("  owners_mid нет, считаю")
        x = d["Estimated owners"].astype(str).str.extract(r"(\d+)\s*-\s*(\d+)").astype(float)
        d["owners_mid"] = x.mean(axis=1)
    log(f"  min(reviews_total)={d['reviews_total'].min()}")
    return d


def features(d, emb):
    """Возвращает (emb_c, X_num_n): центрированный текст и числовой блок, оба L2-нормированы."""
    num = pd.DataFrame({
        "price": np.log1p(d["Price"].fillna(0)),
        "year": d["year"].fillna(d["year"].median()),
        "pos_ratio": d["pos_ratio"],
        "reviews": np.log1p(d["reviews_total"]),
        "owners": np.log1p(d["owners_mid"].fillna(0)),
        "playtime": np.log1p(d["Average playtime forever"].fillna(0)),
        "win": d["Windows"].astype(int), "mac": d["Mac"].astype(int), "lin": d["Linux"].astype(int),
    })
    X_num_n = normalize(StandardScaler().fit_transform(num.values))
    emb_c = normalize(emb - emb.mean(0))
    return emb_c.astype(np.float32), X_num_n.astype(np.float32)


def hybrid(emb_c, X_num_n, w_num):
    return normalize(np.hstack([emb_c, w_num * X_num_n])).astype(np.float32)


def prepare():
    """Полная подготовка без вывода: (d, emb_c, X_num_n)."""
    d, emb = load()
    d = fix_types(d, log=lambda *a: None)
    mask = d["Genres"].notna().values
    d = d[mask].reset_index(drop=True)
    emb = emb[mask]
    emb_c, X_num_n = features(d, emb)
    return d, emb_c, X_num_n


def main():
    d, emb = load()
    print("games2:", d.shape, "\nemb:", emb.shape, emb.dtype)
    print("колонки:", list(d.columns))
    print("доля пропусков:\n", d.isna().mean().round(4).to_string())
    print(d[["Name", "Genres", "Tags", "Price"]].head(3).to_string())

    if len(d) != emb.shape[0]:
        sys.exit(f"СТОП: len(d)={len(d)} != emb.shape[0]={emb.shape[0]}")
    miss = [c for c in NEED if c not in d.columns]
    if miss:
        sys.exit(f"СТОП: нет колонок {miss}")
    print("dtypes нужных колонок:\n", d[NEED].dtypes.to_string())

    print("проверка расчётных колонок:")
    d = fix_types(d)

    # проверка выравнивания
    from sentence_transformers import SentenceTransformer
    text = make_text(d)
    if "text" in d:
        print("колонка text совпадает с формулой:", (clean(d["text"]) == text).mean())
    idx = np.random.RandomState(0).choice(len(d), 30, replace=False)
    model = SentenceTransformer("intfloat/multilingual-e5-base")
    e = model.encode(["passage: " + t for t in text.iloc[idx]], normalize_embeddings=True)
    cos = (e * emb[idx]).sum(1)
    print("косинус (30 строк): min=%.5f mean=%.5f" % (cos.min(), cos.mean()))
    if not (cos > 0.99).all():
        sys.exit("СТОП: выравнивание не сошлось: " + str(np.round(cos, 3)))
    print("выравнивание OK")

    mask = d["Genres"].notna().values
    print("убрано строк с пустым Genres:", (~mask).sum(), "-> осталось", mask.sum())
    d = d[mask].reset_index(drop=True)
    emb_c, X_num_n = features(d, emb[mask])
    print("emb_c", emb_c.shape, "X_num_n", X_num_n.shape)


if __name__ == "__main__":
    main()
