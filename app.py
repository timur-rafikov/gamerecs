from pathlib import Path

import pandas as pd
from annoy import AnnoyIndex
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse

BASE = Path(__file__).parent

# Данные грузим один раз при старте
meta = pd.read_parquet(BASE / "meta.parquet",
                       columns=["Name", "Genres", "Price", "year", "reviews_total", "Header image"])
ANN = BASE / "games.ann"
if ANN.stat().st_size < 1000:  # вместо индекса пришёл LFS-указатель
    raise RuntimeError("games.ann - это LFS-указатель, а не индекс: на деплое нужен git lfs pull")
idx = AnnoyIndex(777, "angular")
idx.load(str(ANN))
# для поиска убираем знаки ™®©, чтобы «dark souls iii» находило «DARK SOULS™ III»
names_lower = meta["Name"].fillna("").str.replace(r"[™®©]", "", regex=True).str.lower()

app = FastAPI(title="GameRec")


def normalize_query(q):
    return (q or "").replace("™", "").replace("®", "").replace("©", "").strip().lower()


def suggest(query, n=8):
    query = normalize_query(query)
    if len(query) < 2:
        return []
    found = meta[names_lower.str.contains(query, regex=False)].copy()
    found["starts"] = names_lower[found.index].str.startswith(query)
    # сначала названия, начинающиеся с запроса, внутри группы по числу отзывов
    found = found.sort_values(["starts", "reviews_total"], ascending=False)
    return found["Name"].drop_duplicates().head(n).tolist()


def recommend(query, max_price=1e9, min_year=0, k=10):
    query = normalize_query(query)
    if not query:
        return {"message": "Введите название игры.", "results": []}
    found = meta[names_lower.str.contains(query, regex=False)]
    if found.empty:
        return {"message": f"Игра «{query}» не найдена в каталоге.", "results": []}
    i = int(found["reviews_total"].idxmax())  # при нескольких совпадениях берём самую популярную
    src = meta.loc[i]
    msg = f"Выбрана игра: {src['Name']} ({src['year']}). Совпадений: {len(found)}."

    ids, dist = idx.get_nns_by_item(i, 300, include_distances=True)
    res = meta.loc[ids].copy()
    res["sim"] = [round(1 - d ** 2 / 2, 3) for d in dist]  # косинус
    res = res[(res.index != i) & (res["Name"] != src["Name"])]
    res = res[(res["Price"] <= max_price) & (res["year"].fillna(0) >= min_year)]
    res = res.drop_duplicates("Name").head(k)
    if res.empty:
        return {"message": msg + " По заданным фильтрам ничего не найдено, ослабьте их.", "results": []}

    out = [{"name": r["Name"], "genres": r["Genres"], "price": float(r["Price"]),
            "year": None if pd.isna(r["year"]) else int(r["year"]), "sim": float(r["sim"]),
            "image": r["Header image"] if isinstance(r["Header image"], str) and r["Header image"] else None}
           for _, r in res.iterrows()]
    return {"message": msg, "results": out}


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/api/recommend")
def api_recommend(q: str = "", max_price: float = 1e9, min_year: int = 0,
                  k: int = Query(10, ge=5, le=20)):
    return JSONResponse(recommend(q, max_price, min_year, k))


@app.get("/api/suggest")
def api_suggest(q: str = ""):
    return suggest(q)


@app.get("/healthz")
def healthz():
    return {"ok": True, "games": len(meta)}
