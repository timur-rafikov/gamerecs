"""Сборка артефактов: meta.parquet и games.ann. Запуск из корня проекта."""
import os
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from prepare_data import prepare, hybrid

W_NUM = 0.3
TREES = 30


def main():
    d, emb_c, X = prepare()
    item = hybrid(emb_c, X, W_NUM)
    print("item:", item.shape, item.dtype)

    meta = d[["AppID", "Name", "Genres", "Tags", "Price", "year", "reviews_total",
              "pos_ratio", "Header image"]].copy()
    meta["year"] = meta["year"].astype("Int64")
    meta.to_parquet("meta.parquet", index=False)

    idx = AnnoyIndex(item.shape[1], "angular")
    for i, v in enumerate(item):
        idx.add_item(i, v)
    idx.build(TREES)
    idx.save("games.ann")

    # recall@10 относительно точного поиска
    rs = np.random.RandomState(0)
    sample = rs.choice(len(item), 500, replace=False)
    hit = hit300 = 0
    for i in sample:
        s = item @ item[i]
        s[i] = -np.inf
        exact = set(np.argpartition(-s, 10)[:10])
        approx = [j for j in idx.get_nns_by_item(int(i), 11) if j != i][:10]
        hit += len(exact & set(approx))
        # как в приложении: запрос 300 соседей, берём первые 10
        a300 = [j for j in idx.get_nns_by_item(int(i), 300) if j != i][:10]
        hit300 += len(exact & set(a300))
    print("recall@10 Annoy (запрос 11): %.4f" % (hit / (500 * 10)))
    print("recall@10 Annoy (запрос 300, как в app): %.4f" % (hit300 / (500 * 10)))

    for f in ["meta.parquet", "games.ann"]:
        print(f, round(os.path.getsize(f) / 1e6, 1), "МБ")


if __name__ == "__main__":
    main()
