"""Оценка: Jaccard по Tags и Categories для топ-10 соседей. Запуск из корня проекта."""
import json
import numpy as np
from prepare_data import prepare, hybrid

K = 10


def to_sets(s):
    return [set(x.strip() for x in str(v).split(",") if x.strip()) if isinstance(v, str) else set()
            for v in s]


def jaccard(a, b):
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def evaluate(M, idx, tags, cats):
    jt, jc = [], []
    for i in idx:
        s = M @ M[i]
        s[i] = -np.inf  # исключаем саму игру
        nn = np.argpartition(-s, K)[:K]
        jt.append(np.mean([jaccard(tags[i], tags[j]) for j in nn]))
        jc.append(np.mean([jaccard(cats[i], cats[j]) for j in nn]))
    return float(np.mean(jt)), float(np.mean(jc))


def main():
    d, emb_c, X = prepare()
    tags, cats = to_sets(d["Tags"]), to_sets(d["Categories"])
    cand = np.array([i for i in range(len(d)) if len(tags[i]) >= 5])
    idx = np.random.RandomState(0).choice(cand, 1000, replace=False)
    variants = {"только текст": emb_c, "только числа": X}
    for w in [0.05, 0.1, 0.2, 0.3, 0.5]:
        variants[f"гибрид w_num={w}"] = hybrid(emb_c, X, w)
    res = {}
    print(f"{'вариант':22s} Tags   Categories")
    for name, M in variants.items():
        t, c = evaluate(M, idx, tags, cats)
        res[name] = {"jaccard_tags": round(t, 4), "jaccard_categories": round(c, 4)}
        print(f"{name:22s} {t:.4f} {c:.4f}")
    json.dump(res, open("metrics.json", "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
