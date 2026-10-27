# Лабораторна робота №2: Контейнеризація та розгортання ML-моделі з Docker

Проєкт виконує **обов'язкове завдання базового рівня (розділ 2 методички)** без опціональних завдань розділу 3. Сервіс класифікації голосових команд (`yes`, `no`, `up`, `down`) на базі датасету **Google Speech Commands v0.02** та згорткової мережі **AudioCNN** упаковано у відтворюваний багатоступеневий Docker-образ із FastAPI, `HEALTHCHECK`, непривілейованим користувачем `appuser` та CI-пайплайном GitHub Actions.

---

## 1. Структура проєкту (п. 2.1)

```text
lab2.2/
├── .github/
│   └── workflows/
│       └── ci.yml             # Базовий CI-пайплайн (lint -> build -> run -> smoke test & quality gate)
├── src/
│   ├── __init__.py
│   ├── data.py                # Завантаження датасету, Mel-спектрограми (32 канали, 8 кГц)
│   ├── model.py               # Архітектура CNN (AudioCNN)
│   ├── train.py               # Навчання + оцінка -> artifacts/ (model.pt, labels.json, metrics.json)
│   ├── evaluate.py            # Обчислення accuracy на тестовому наборі
│   └── api.py                 # FastAPI-сервіс (/health, /model-info, /predict)
├── artifacts/                 # Згенеровані артефакти: model.pt, labels.json, metrics.json
├── samples/                   # Тестові .wav файли (yes.wav, no.wav, up.wav, down.wav)
├── scripts/
│   └── benchmark.py           # Вимірювання server та client end-to-end latency (mean / p50 / p95)
├── results/                   # Збережені результати вимірювань JSON
├── requirements-train.txt     # Фіксовані залежності для навчання (CPU PyTorch + tqdm, scikit-learn)
├── requirements-api.txt       # Фіксовані залежності для інференс-сервісу
├── Dockerfile                 # Багатоступенева збірка (Stage 1: trainer, Stage 2: runtime)
├── Dockerfile.single          # Одноступенева збірка (для порівняння розмірів у п. 2.8)
├── .dockerignore              # Виключення .git, .venv, data/, artifacts/ з build context
├── README.md                  # Інструкція із запуску та перевірки
└── REPORT.md                  # Повний звіт до лабораторної роботи (розділи 4 та 5 методички)
```

---

## 2. Локальний запуск у віртуальному середовищі (`.venv`)

```powershell
# Створення та активація середовища
uv venv --python 3.10 .venv
uv pip install --index-strategy unsafe-best-match --python .venv\Scripts\python.exe -r requirements-train.txt

# Навчання моделі (10 епох, фіксований seed 42)
.\.venv\Scripts\python.exe -m src.train --classes yes no up down --epochs 10 --seed 42 --out artifacts

# Перевірка accuracy на тестовому наборі (1655 файлів)
.\.venv\Scripts\python.exe -m src.evaluate --model artifacts/model.pt

# Запуск FastAPI-сервісу
.\.venv\Scripts\python.exe -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

---

## 3. Збирання та запуск у Docker (п. 2.6)

### Багатоступеневий образ (Multi-stage)
```powershell
docker build -t kws-api:1.0.0 .
docker run -d --name kws -p 8000:8000 kws-api:1.0.0
docker ps
docker logs kws
```

### Одноступеневий образ для порівняння (Single-stage, п. 2.8)
```powershell
docker build -f Dockerfile.single -t kws-api:single .
docker images kws-api
docker history kws-api:1.0.0
docker history kws-api:single
```

---

## 4. Перевірка роботи сервісу (п. 2.7)

```powershell
curl.exe http://localhost:8000/health
curl.exe http://localhost:8000/model-info
curl.exe -X POST -F "file=@samples/yes.wav" http://localhost:8000/predict
curl.exe -X POST -F "file=@samples/no.wav" http://localhost:8000/predict
curl.exe -X POST -F "file=@samples/up.wav" http://localhost:8000/predict
curl.exe -X POST -F "file=@samples/down.wav" http://localhost:8000/predict
```

Swagger UI доступний у браузері за адресою: `http://localhost:8000/docs`.

### Вимірювання Latency (п. 2.8)
```powershell
.\.venv\Scripts\python.exe scripts/benchmark.py --url http://localhost:8000 --file samples/yes.wav --requests 100 --warmup 10
```

---

## 5. Зведена таблиця фактичних результатів (п. 2.8)

| Середовище | Модель | Розмір моделі, КБ | Accuracy, % | Latency (сервер) mean / p50 / p95, мс | Latency (клієнт) mean / p50 / p95, мс | Розмір образу / середовища, МБ | RAM, МБ | Холодний старт, с |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Локально (venv)** | AudioCNN | 280.1 | 91.90 | 14.23 / 13.45 / 15.62 | 19.51 / 18.76 / 21.54 | 1926.71 (`.venv` + модель) | 195.22 | 7.56 |
| **Docker (single-stage)** | AudioCNN | 280.1 | 91.90 | 2.42 / 1.90 / 5.44 | 7.85 / 6.70 / 15.36 | 2550.00 (710 МБ compressed) | 143.20 | 2.15 |
| **Docker (multi-stage)** | AudioCNN | 280.1 | 91.90 | 2.13 / 1.97 / 3.28 | 7.12 / 6.31 / 10.92 | 1440.00 (308 МБ compressed) | 151.20 | 2.40 |

Детальний звіт, пошаровий аналіз `docker history` та відповіді на контрольні питання наведено у файлі [REPORT.md](REPORT.md).
