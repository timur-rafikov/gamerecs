# GameRec

Content-based item2item рекомендательная система видеоигр Steam: по названию игры находит похожие.
Веб-сервис на FastAPI: поиск по названию с подсказками, фильтры по цене и году, обложки и сходство в выдаче.

**Демо:** https://gamerecs-o8un.onrender.com (бесплатный план Render, после простоя первый запрос может идти долго).

## Данные
Kaggle: [fronkongames/steam-games-dataset](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset).
Фильтр каталога: не менее 10 отзывов, есть текст или теги, есть жанры. Итого **56565** игр.

## Модель
- Текст: `Genres: ... Tags: ... <первые 1000 символов About the game>`, эмбеддинги `intfloat/multilingual-e5-base`, затем центрирование и L2-нормировка.
- Числовой блок (StandardScaler + L2): log1p(Price), year, pos_ratio, log1p(reviews_total), log1p(owners_mid), log1p(Average playtime forever), Windows, Mac, Linux.
- Гибрид: `normalize(hstack([emb_c, w_num * X_num_n]))`, **w_num = 0.3**, размерность 777.
- Поиск: Annoy (angular, 30 деревьев) по индексу `games.ann` (197.3 МБ, хранится в Git LFS). Запрашивается 300 соседей, затем фильтры и `drop_duplicates("Name")`. Сходство: cos = 1 - dist^2 / 2.
- Recall@10 Annoy относительно точного поиска (500 игр): 0.8828 при запросе 300 соседей (как в сервисе).
- Модель эмбеддингов в сервис не входит.

## Метрики
Среднее Jaccard-сходство множеств Tags и Categories у топ-10 соседей, 1000 случайных игр с >=5 тегами.
Теги входят в текст эмбеддингов, поэтому метрика по Tags частично круговая; Categories надёжнее.

| Вариант | Jaccard Tags | Jaccard Categories |
|---|---|---|
| только текст | 0.2688 | 0.4944 |
| только числа | 0.1291 | 0.4657 |
| гибрид w_num=0.05 | 0.2691 | 0.4959 |
| гибрид w_num=0.1 | 0.2712 | 0.4995 |
| гибрид w_num=0.2 | 0.2764 | 0.51 |
| гибрид w_num=0.3 | 0.2804 | 0.5218 |
| гибрид w_num=0.5 | 0.2784 | 0.5305 |

## Стек
- **Данные и подготовка:** Python 3.12, pandas, numpy, scikit-learn (StandardScaler, нормировка), pyarrow. Эмбеддинги текстов: sentence-transformers, модель `intfloat/multilingual-e5-base` (используется только при подготовке данных).
- **Рекомендации:** гибрид текстового эмбеддинга и числовых признаков, поиск соседей через Annoy (angular, 30 деревьев).
- **Хранение:** `meta.parquet` (метаданные игр), `games.ann` (индекс, 197 МБ) в Git LFS.
- **Сервис:** FastAPI и uvicorn (`/api/recommend`, `/api/suggest`, `/healthz`), фронтенд на одной HTML-странице с чистым JS (fetch, `<datalist>` для подсказок), без фреймворков и сборки.
- **Деплой:** GitHub, Render Web Service (бесплатный план), автодеплой при коммите. Два способа сборки: `Dockerfile` (переносимый на другие платформы) или нативный Python через `build.sh`. В обоих `annoy` собирается без `-march=native`.

## Запуск локально
```bash
pip install -r requirements.txt
uvicorn app:app --reload
```
API:
- `GET /api/recommend?q=Hades&max_price=30&min_year=2015&k=10`: рекомендации (`max_price`, `min_year` необязательны, `k` от 5 до 20);
- `GET /api/suggest?q=had`: до 8 подсказок названий (от 2 символов);
- `GET /healthz`: проверка работоспособности.

Локально нужен Python 3.10 или новее. `annoy` собирается из исходников, нужен компилятор C++. При запуске на Linux x86_64 прочитайте примечание про `-march=native` ниже.

## Деплой на Render
Сервис запущен как Web Service из этого репозитория. Есть два способа.

**1. Docker runtime** (рекомендуется, переносится на другие платформы): в Settings сервиса Runtime = Docker, Build и Start Command не нужны, всё описано в `Dockerfile`. Health Check Path: `/healthz`.

**2. Нативный Python:**
- Build Command: `bash build.sh` (установка пакетов, сборка `annoy` без `-march=native`)
- Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Health Check Path: `/healthz`, переменная `PYTHON_VERSION=3.12.7`.

`render.yaml` описывает нативный вариант. Файл `games.ann` хранится в Git LFS, Render подтягивает его при клонировании.

**Важно про `annoy`.** По умолчанию он собирается с `-march=native`, то есть под процессор машины сборки. Если сервис работает на другой машине, процесс падает с кодом 132 (SIGILL) на первом поиске. Поэтому и `build.sh`, и `Dockerfile` заменяют флаг на `-march=x86-64`.

## Docker
Образ не зависит от платформы: `annoy` собирается на отдельном этапе без `-march=native`, порт берётся из `$PORT` (по умолчанию 8000). Перед сборкой убедитесь, что `games.ann` скачан по-настоящему (`git lfs pull`), а не остался LFS-указателем.
```bash
docker build -t gamerec .
docker run --rm -p 8000:8000 gamerec
```
Дальше открыть http://localhost:8000. Такой же образ подходит для Render (Docker runtime), Fly.io, Cloud Run или своего сервера. Образ проверен локально на arm64 и amd64 (через эмуляцию).

## Пересборка артефактов
Нужны `../gamerecfiles/games2.csv`, `../gamerecfiles/emb.npy`, `sentence-transformers`, `scikit-learn`. Из корня проекта:
```bash
python scripts/prepare_data.py
python scripts/evaluate.py
python scripts/build_index.py
```
Генерирует `meta.parquet` и `games.ann`.
