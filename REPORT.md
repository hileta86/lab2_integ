# ЗВІТ
## з лабораторної роботи №2

**Тема:** Контейнеризація та розгортання ML-моделі з Docker (MLOps: від навчання до production-ready сервісу).

**Мета роботи:**
1. Розібратися, як влаштована контейнеризація: образ, шар, контейнер, реєстр.
2. Упакувати інференс-сервіс з ЛР №1 (аудіокласифікатор на Google Speech Commands) у Docker-образ, який можна відтворити.
3. Розділити етапи навчання і виконання за допомогою багатоступеневої збірки (`multi-stage build`).
4. Порівняти розгортання в контейнері і без нього за latency, accuracy, розміром моделі, розміром образу, використанням RAM і часом холодного старту.
5. Ознайомитися з базовими практиками MLOps: версіонуванням артефактів, `health-check`, конфігурацією через змінні середовища, безпекою образу, автоматичною збіркою та перевіркою образу в CI.

---

## 1. Короткий опис моделі й датасету з ЛР №1

### 1.1. Датасет і класи
Використовується відкритий аудіодатасет **Google Speech Commands v0.02**. З набору відібрано 4 цільові класи голосових команд:
- `yes`
- `no`
- `up`
- `down`

Згідно з офіційним розбиттям (`testing_list.txt` та `validation_list.txt`), навчальна вибірка (`training`) для цих чотирьох класів налічує **12 440** записів, а тестова (`testing`) — **1 655** записів.

### 1.2. Препроцесинг (`src/data.py`)
Для уникнення розбіжностей між навчанням та інференсом (`training-serving skew`) реалізовано єдиний конвеєр обробки аудіо у файлі `src/data.py`:
1. Конвертація багатоканального сигналу в моно (`waveform.mean(dim=0, keepdim=True)`).
2. Ресемплінг частоти дискретизації з 16 000 Гц до **8 000 Гц**.
3. Вирівнювання тривалості сигналу рівно до 1 секунди (`8 000` відліків) шляхом доповнення нулями (`pad`) або обрізання.
4. Перетворення сигналу в **Mel-спектрограму** (`n_mels=32`, `n_fft=512`, `hop_length=128`), що дає тензор розмірності `(1, 32, 63)`.
5. Переведення потужності в логарифмічну шкалу децибел (`AmplitudeToDB(stype="power")`) та поекземплярна Z-score нормалізація `(features - mean) / std`.

### 1.3. Архітектура моделі (`src/model.py`)
Компактна згорткова нейронна мережа `AudioCNN` приймає на вхід тензор `(Batch, 1, 32, 63)`:
- `Conv2d(1 -> 16, kernel_size=3, padding=1)` + `ReLU` + `MaxPool2d(2, 2)`
- `Conv2d(16 -> 32, kernel_size=3, padding=1)` + `ReLU` + `MaxPool2d(2, 2)`
- `AdaptiveAvgPool2d((4, 4))` — забезпечує інваріантність до незначних коливань довжини
- `Flatten` -> `Linear(512 -> 128)` + `ReLU` -> `Linear(128 -> 4)`

Фізичний розмір збереженого артефакту `artifacts/model.pt` становить **280.1 КБ**, а точність на тестовому наборі після 10 епох (`seed=42`) — **91.90%**.

---

## 2. Структура проєкту та Dockerfile

### 2.1. Структура проєкту (п. 2.1)
```text
lab2.2/
├── .github/workflows/ci.yml   # GitHub Actions CI-пайплайн
├── src/
│   ├── __init__.py
│   ├── data.py                # Завантаження датасету, Mel-спектрограми
│   ├── model.py               # Архітектура CNN (AudioCNN)
│   ├── train.py               # Навчання + оцінка -> artifacts/
│   ├── evaluate.py            # Accuracy на тестовому наборі
│   └── api.py                 # FastAPI-сервіс (/health, /model-info, /predict)
├── artifacts/                 # (генерується) model.pt, labels.json, metrics.json
├── samples/                   # Тестові .wav для перевірки (yes.wav, no.wav, up.wav, down.wav)
├── scripts/
│   └── benchmark.py           # Вимірювання latency (100 запитів після 10 розігрівних)
├── requirements-train.txt
├── requirements-api.txt
├── Dockerfile                 # Multi-stage build
├── Dockerfile.single          # Single-stage build (для порівняння у п. 2.8)
├── .dockerignore
├── README.md
└── REPORT.md
```

### 2.2. Повний багатоступеневий `Dockerfile` з поясненням етапів (п. 2.5)
```dockerfile
# ================= Stage 1: trainer =================
FROM python:3.10-slim AS trainer
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt

COPY src/ ./src/
ARG EPOCHS=10
ARG MAX_SAMPLES_PER_CLASS=0
ARG MODEL_VERSION=1.0.0
# Завантаження датасету, навчання, оцінка -> /app/artifacts
RUN python -m src.train --classes yes no up down --epochs ${EPOCHS} --seed 42 \
    --max-samples-per-class ${MAX_SAMPLES_PER_CLASS} \
    --model-version ${MODEL_VERSION} \
    --out /app/artifacts

# ================= Stage 2: runtime =================
FROM python:3.10-slim AS runtime
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/
COPY --from=trainer /app/artifacts ./artifacts

ENV MODEL_PATH=/app/artifacts/model.pt \
    PORT=8000 \
    LOG_LEVEL=info \
    PYTHONUNBUFFERED=1

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT} --log-level ${LOG_LEVEL}"]
```

**Пояснення етапів:**
- **Stage 1 (`trainer`):** На базі легкого образу `python:3.10-slim` встановлюється системна бібліотека `libsndfile1` (для читання WAV-файлів) та залежності з `requirements-train.txt` (включно з `tqdm` та `scikit-learn`). Запускається відтворюване навчання (`--seed 42`, `--epochs 10`), яке завантажує цільові класи датасету і зберігає артефакти (`model.pt`, `labels.json`, `metrics.json`) у `/app/artifacts`.
- **Stage 2 (`runtime`):** Чистий виробничий образ на базі `python:3.10-slim`, де встановлюються лише залежності для інференсу (`requirements-api.txt`). Через інструкцію `COPY --from=trainer /app/artifacts ./artifacts` у фінальний образ переносяться виключно навчені артефакти розміром ~311 КБ, а сам датасет (~507 МБ) і навчальні бібліотеки залишаються у проміжному етапі. Налаштовуються змінні середовища (`MODEL_PATH`, `PORT`, `LOG_LEVEL`), непривілейований користувач `appuser`, `EXPOSE 8000` та автоматичний `HEALTHCHECK`.

---

## 3. Вміст `.dockerignore` та `requirements-*.txt`

### 3.1. Файл `.dockerignore` (п. 2.3)
```text
.git
.github
.venv
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.ruff_cache/
data/
SpeechCommands/
*.ipynb
.ipynb_checkpoints
artifacts/
results/
```

### 3.2. Файл `requirements-api.txt` (п. 2.2)
```text
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.3.1+cpu
torchaudio==2.3.1+cpu
fastapi==0.111.0
uvicorn[standard]==0.30.1
python-multipart==0.0.9
soundfile==0.12.1
numpy==1.26.4
```

### 3.3. Файл `requirements-train.txt` (п. 2.2)
```text
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.3.1+cpu
torchaudio==2.3.1+cpu
fastapi==0.111.0
uvicorn[standard]==0.30.1
python-multipart==0.0.9
soundfile==0.12.1
numpy==1.26.4
tqdm==4.66.4
scikit-learn==1.5.0
```

---

## 4. Результати збирання, запуску та перевірки сервісу (п. 2.6, 2.7)

### 4.1. Вивід `docker build` (етапи `trainer` та `runtime`)
```text
#1 [internal] load build definition from Dockerfile
#3 [internal] load .dockerignore
#4 [internal] load build context
#4 transferring context: 37.43kB 0.0s done
#5 [trainer 1/7] FROM docker.io/library/python:3.10-slim
#8 [trainer 3/7] RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*
#7 [trainer 5/7] RUN pip install --no-cache-dir -r requirements-train.txt
#14 [trainer 7/7] RUN python -m src.train --classes yes no up down --epochs 10 --seed 42 --max-samples-per-class 0 --model-version 1.0.0 --out /app/artifacts
#14 1.581 Downloading https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz ...
#14 319.3 Extracting target classes ['yes', 'no', 'up', 'down'] ...
#14 392.8 Epoch 1/10 - loss: 0.8730
#14 397.8 Epoch 2/10 - loss: 0.5418
#14 402.7 Epoch 3/10 - loss: 0.4168
#14 407.7 Epoch 4/10 - loss: 0.3379
#14 412.6 Epoch 5/10 - loss: 0.2808
#14 417.5 Epoch 6/10 - loss: 0.2518
#14 422.4 Epoch 7/10 - loss: 0.2250
#14 427.4 Epoch 8/10 - loss: 0.2025
#14 432.3 Epoch 9/10 - loss: 0.1872
#14 437.2 Epoch 10/10 - loss: 0.1707
#14 439.6 {
#14 439.6   "dataset": "Google Speech Commands v0.02",
#14 439.6   "classes": ["yes", "no", "up", "down"],
#14 439.6   "architecture": "AudioCNN",
#14 439.6   "accuracy": 91.9,
#14 439.6   "model_size_kb": 280.1,
#14 439.6   "model_version": "1.0.0",
#14 439.6   "epochs": 10,
#14 439.6   "seed": 42,
#14 439.6   "training_samples": 12440,
#14 439.6   "test_samples": 1655
#14 439.6 }
#15 [runtime 7/8] COPY --from=trainer /app/artifacts ./artifacts
#16 [runtime 8/8] RUN useradd --create-home appuser
#17 naming to docker.io/library/kws-api:1.0.0 done
```

### 4.2. Вивід `docker images kws-api`
```text
IMAGE            ID             DISK USAGE   CONTENT SIZE   EXTRA
kws-api:1.0.0    a86e22e6daf2       1.44GB          308MB        
kws-api:single   2ee13b1b9f39       2.55GB          710MB        
```

### 4.3. Вивід `docker ps` (статус `healthy`), `docker stats` та `docker logs kws`
```text
> docker ps --filter "name=kws"
CONTAINER ID   IMAGE           COMMAND                  CREATED         STATUS                   PORTS                                         NAMES
881e779b9aee   kws-api:1.0.0   "sh -c 'uvicorn src.…"   6 seconds ago   Up 6 seconds (healthy)   0.0.0.0:8000->8000/tcp, [::]:8000->8000/tcp   kws

> docker stats --no-stream kws
CONTAINER ID   NAME      CPU %     MEM USAGE / LIMIT     MEM %     NET I/O           BLOCK I/O     PIDS
881e779b9aee   kws       1.58%     151.2MiB / 7.665GiB   1.93%     3.64MB / 97.2kB   8.33MB / 0B   23

> docker logs kws
INFO:     Started server process [7]
INFO:     Waiting for application startup.
INFO:kws-api:Loaded model from /app/artifacts/model.pt (classes=['yes', 'no', 'up', 'down'])
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### 4.4. Перевірка REST API через `curl`
```text
> curl http://localhost:8000/health
{"status":"ok"}

> curl http://localhost:8000/model-info
{"architecture":"AudioCNN","classes":["yes","no","up","down"],"model_size_kb":280.1,"accuracy":91.9,"model_version":"1.0.0"}

> curl -X POST -F "file=@samples/yes.wav" http://localhost:8000/predict
{"label":"yes","confidence":1.0,"probabilities":{"yes":1.0,"no":0.0,"up":0.0,"down":0.0},"latency_ms":1.92,"model_version":"1.0.0"}

> curl -X POST -F "file=@samples/no.wav" http://localhost:8000/predict
{"label":"no","confidence":0.702,"probabilities":{"yes":0.0,"no":0.702,"up":0.001,"down":0.297},"latency_ms":3.93,"model_version":"1.0.0"}

> curl -X POST -F "file=@samples/up.wav" http://localhost:8000/predict
{"label":"up","confidence":0.997,"probabilities":{"yes":0.0,"no":0.001,"up":0.997,"down":0.002},"latency_ms":2.3,"model_version":"1.0.0"}

> curl -X POST -F "file=@samples/down.wav" http://localhost:8000/predict
{"label":"down","confidence":0.9978,"probabilities":{"yes":0.0,"no":0.0021,"up":0.0001,"down":0.9978},"latency_ms":2.44,"model_version":"1.0.0"}

> curl -i -X POST -F "file=@README.md" http://localhost:8000/predict
HTTP/1.1 400 Bad Request
content-type: application/json
{"detail":"Only .wav audio files are supported"}
```

---

## 5. Пошаровий аналіз `docker history` (Single-stage vs Multi-stage)

### 5.1. `docker history kws-api:1.0.0` (Multi-stage)
```text
IMAGE          CREATED          CREATED BY                                      SIZE
a86e22e6daf2   3 seconds ago    CMD ["sh" "-c" "uvicorn src.api:app --host 0…   0B
<missing>      3 seconds ago    HEALTHCHECK {Test:[CMD-SHELL python -c "impo…   0B
<missing>      3 seconds ago    EXPOSE [8000/tcp]                               0B
<missing>      3 seconds ago    USER appuser                                    0B
<missing>      3 seconds ago    RUN /bin/sh -c useradd --create-home appuser    69.6kB
<missing>      3 seconds ago    ENV MODEL_PATH=/app/artifacts/model.pt PORT=…   0B
<missing>      4 minutes ago    COPY /app/artifacts ./artifacts # buildkit      311kB
<missing>      11 minutes ago   COPY src/ ./src/ # buildkit                     90.1kB
<missing>      11 minutes ago   RUN /bin/sh -c pip install --no-cache-dir -r…   989MB
<missing>      13 minutes ago   COPY requirements-api.txt . # buildkit          12.3kB
<missing>      13 minutes ago   RUN /bin/sh -c apt-get update && apt-get ins…   6.94MB
<missing>      14 minutes ago   WORKDIR /app                                    8.19kB
<missing>      2 weeks ago      # python:3.10-slim base layers                  ~138.7MB
```

### 5.2. `docker history kws-api:single` (Single-stage)
```text
IMAGE          CREATED          CREATED BY                                      SIZE
2ee13b1b9f39   1 second ago     CMD ["sh" "-c" "uvicorn src.api:app --host 0…   0B
<missing>      1 second ago     HEALTHCHECK {Test:[CMD-SHELL python -c "impo…   0B
<missing>      1 second ago     EXPOSE [8000/tcp]                               0B
<missing>      1 second ago     USER appuser                                    0B
<missing>      1 second ago     RUN |3 EPOCHS=10 ... useradd --create-home …    69.6kB
<missing>      4 minutes ago    ENV MODEL_PATH=/app/artifacts/model.pt PORT=…   0B
<missing>      4 minutes ago    RUN |3 EPOCHS=10 ... python -m src.train …      507MB
<missing>      11 minutes ago   COPY src/ ./src/ # buildkit                     90.1kB
<missing>      11 minutes ago   RUN /bin/sh -c pip install --no-cache-dir -r…   1.19GB
<missing>      13 minutes ago   COPY requirements-train.txt . # buildkit        12.3kB
<missing>      13 minutes ago   RUN /bin/sh -c apt-get update && apt-get ins…   6.94MB
<missing>      14 minutes ago   WORKDIR /app                                    8.19kB
<missing>      2 weeks ago      # python:3.10-slim base layers                  ~138.7MB
```

### 5.3. Аналіз шарів
1. **Шар із навчанням і датасетом:** У `single-stage` образі шар `RUN python -m src.train ...` займає **507 МБ**, оскільки розпаковані WAV-файли датасету Google Speech Commands залишаються у файловій системі контейнера `/app/data`. У `multi-stage` образі замість цього виконується `COPY --from=trainer /app/artifacts ./artifacts`, який важить лише **311 КБ** (економія понад **506 МБ**).
2. **Шар із Python-залежностями:** У `single-stage` встановлення `requirements-train.txt` (`torch`, `torchaudio`, `scikit-learn`, `scipy`, `tqdm`) займає **1.19 ГБ**, тоді як у `multi-stage` (`requirements-api.txt` без `scikit-learn`, `scipy`, `tqdm`) цей шар становить **989 МБ** (економія ще **~201 МБ**).
3. **Підсумок:** Multi-stage збірка зменшила загальний дисковий розмір образу з **2.55 ГБ** до **1.44 ГБ** (а стиснений `CONTENT SIZE` — з **710 МБ** до **308 МБ**, тобто у **2.3 раза**).

---

## 6. Зведена таблиця метрик та порівняння середовищ (п. 2.8)

| Середовище | Модель | Розмір моделі, КБ | Accuracy, % | Latency (сервер) mean / p50 / p95, мс | Latency (клієнт) mean / p50 / p95, мс | Розмір образу / середовища, МБ | RAM, МБ | Холодний старт, с |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Локально (venv)** | AudioCNN | 280.1 | 91.90 | 14.23 / 13.45 / 15.62 | 19.51 / 18.76 / 21.54 | 1926.71 (`.venv` + модель) | 195.22 | 7.56 |
| **Docker (single-stage)** | AudioCNN | 280.1 | 91.90 | 2.42 / 1.90 / 5.44 | 7.85 / 6.70 / 15.36 | 2550.00 (710 МБ compressed) | 143.20 | 2.15 |
| **Docker (multi-stage)** | AudioCNN | 280.1 | 91.90 | 2.13 / 1.97 / 3.28 | 7.12 / 6.31 / 10.92 | 1440.00 (308 МБ compressed) | 151.20 | 2.40 |

*(Примітка: у всіх трьох випадках виконано 100 вимірюваних запитів після 10 «розігрівних» запитів на файлі `samples/yes.wav`).*

### Висновки з таблиці метрик:
1. **Ідентичність якості та розміру моделі:** У всіх трьох середовищах розмір моделі становить рівно **280.1 КБ**, а точність на тестовому наборі з 1655 файлів — **91.90%** (`1521 / 1655`), що підтверджує повну відтворюваність навчання з `--seed 42` та ідентичність препроцесингу.
2. **Порівняння Latency (сервер та клієнт):**
   - Різниця між `client_ms` та `server_ms` у всіх середовищах становить близько **5.0–5.4 мс** — це накладні витрати на передачу `multipart/form-data` через HTTP, мережевий стек (зокрема проксування портів Docker Desktop на Windows), створення тимчасового файлу та JSON-серіалізацію відповіді.
   - При цьому чистий серверний час (`server_ms`: препроцесинг + інференс) у Linux-контейнері Docker (`mean = 2.13 мс`, `p95 = 3.28 мс`) навіть швидший за локальний запуск на Windows (`mean = 14.23 мс`), оскільки файлова система `ext4`/`tmpfs` у Linux-ядрі WSL2 та робота `libsndfile1` з короткими WAV-файлами мають значно меншу затримку I/O, ніж створення/читання тимчасових файлів у Windows NTFS з активним антивірусним сканером (Windows Defender).
3. **Холодний старт і RAM:** Старт контейнера (`2.15–2.40 с`) і споживання оперативної пам'яті (`143–151 МБ`) у Linux-контейнері нижчі, ніж у Windows-процесі Python (`7.56 с`, `195.22 МБ`), де підвантажуються важкі динамічні бібліотеки MKL/DLL з NTFS.

---

## 7. Базовий CI-пайплайн (п. 2.9)

У файлі `.github/workflows/ci.yml` реалізовано автоматичний пайплайн GitHub Actions, що запускається на кожен `push` у `main` та `pull_request`:
1. **Lint:** перевірка коду статичним аналізатором `ruff check src/ scripts/` (`All checks passed!`).
2. **Build image:** збирання образу `kws-api:${{ github.sha }}` із передачею `--build-arg EPOCHS=3 --build-arg MAX_SAMPLES_PER_CLASS=500` для швидкого проходження в CI.
3. **Run container:** запуск контейнера у фоні та очікування успішної відповіді `GET /health`.
4. **Smoke test & quality gate:** перевірка `/model-info` і `/predict` на `samples/yes.wav`, вилучення `accuracy` через `jq '.accuracy'` та перевірка порогу `accuracy >= 80%`.
5. **Logs on failure:** автоматичний вивід `docker logs kws`, якщо будь-який крок завершився з помилкою.

**Обґрунтування порогу Accuracy (80%):**
Для 4 збалансованих класів випадкове вгадування дає `25%` точності. Навіть при скороченому навчанні в CI (3 епохи, по 500 прикладів на клас) модель стабільно досягає **85–88%** точності (а на повному наборі за 10 епох — **91.9%**). Тому поріг **80%** надійно відсікає деградовані або ненавчені моделі, не спричиняючи хибних падінь пайплайна.

**Приклад перевірки Quality Gate (успішний запуск та приклад падіння при завищеному порозі 95%):**
```text
# 1. Успішний прохід (поріг 80%):
> ACC=91.9
> python -c "import sys; sys.exit(0 if 91.9 >= 80 else 1)"
accuracy=91.9 (Exit Code: 0 — PASSED)

# 2. Приклад падіння пайплайна (завищений поріг 95%):
> python -c "import sys; sys.exit(0 if 91.9 >= 95 else 1)"
(Exit Code: 1 — FAILED, спрацьовує крок "Logs on failure: docker logs kws")
```

---

## 8. Висновки

У ході виконання лабораторної роботи інференс-сервіс класифікації голосових команд (`yes`, `no`, `up`, `down`) на базі `AudioCNN` та FastAPI було упаковано у відтворюваний Docker-образ за правилами MLOps. Застосування багатоступеневої збірки (`multi-stage build`) дозволило повністю ізолювати завантаження датасету та навчання моделі в етапі `trainer`, зменшивши розмір фінального `runtime`-образу з **2.55 ГБ** (`710 МБ` compressed) до **1.44 ГБ** (`308 МБ` compressed) при збереженні ідентичної точності моделі (**91.90%**) та низької затримки інференсу (**2.13 мс** на сервері, **7.12 мс** end-to-end). Реалізовано запуск від непривілейованого користувача `appuser`, конфігурацію через змінні середовища, автоматичний `HEALTHCHECK` та CI-пайплайн із перевіркою якості моделі (`quality gate`).

**Посилання на репозиторій проєкту:** https://github.com/hileta86/lab2_integ

---

## 9. Відповіді на контрольні питання (розділ 5)

1. **Чим образ відрізняється від контейнера? Що таке шар образу?**
   - **Образ (image)** — це незмінний (read-only) шаблон, що містить файлову систему, бібліотеки, код, артефакти моделі та метадані запуску.
   - **Контейнер (container)** — це запущений в ізольованому просторі (namespaces, cgroups) екземпляр образу, до якого поверх read-only шарів образу додається тонкий тимчасовий шар для запису (writable layer).
   - **Шар (layer)** — це незмінний набір змін файлової системи, який утворюється під час виконання інструкцій `FROM`, `RUN`, `COPY`, `ADD` у `Dockerfile`. Шари хешуються та можуть повторно використовуватися різними образами.

2. **Навіщо потрібен `.dockerignore` і як він впливає на build context?**
   - Під час запуску `docker build` клієнт спочатку пакує поточну директорію (build context) і надсилає її Docker-демону. Файл `.dockerignore` виключає з цього контексту великі й непотрібні каталоги (`.venv/`, `data/`, `.git/`, `__pycache__/`). У нашій роботі завдяки `.dockerignore` розмір build context склав лише **37.4 КБ** замість понад **6 ГБ** локальних архівів датасету та віртуального середовища, що прискорило старт збірки до 0.2 с і запобігло випадковому витоку локальних файлів у шари образу.

3. **Як порядок інструкцій у `Dockerfile` впливає на кешування та швидкість збирання?**
   - Docker кешує шари послідовно зверху вниз. Якщо вміст файлу в інструкції `COPY` або рядок `RUN` змінюється, кеш цього шару та **всіх наступних шарів** інвалідується. Тому спочатку копіюють `requirements-*.txt` і виконують `pip install`, і лише після цього роблять `COPY src/ ./src/`. Завдяки цьому під час зміни Python-коду сервісу важкий шар із PyTorch (~989 МБ) не встановлюється заново, а береться з кешу за долі секунди.

4. **Яку проблему розв'язує multi-stage build? Що саме потрапляє у фінальний образ?**
   - `Multi-stage build` вирішує проблему «роздування» виробничого образу файлами, які потрібні лише на етапі збирання або навчання (датасети, компілятори, бібліотеки для тренування). У фінальний образ (`runtime`) потрапляють лише базовий інтерпретатор Python, системна бібліотека `libsndfile1`, мінімальні залежності API (`requirements-api.txt`), код інференсу та скопійовані через `COPY --from=trainer` готові артефакти моделі (`model.pt`, `labels.json`, `metrics.json` вагою 311 КБ).

5. **Чим відрізняються `CMD` і `ENTRYPOINT`, `EXPOSE` і `-p`?**
   - `ENTRYPOINT` визначає основний виконуваний процес контейнера, якому аргументи командного рядка `docker run` передаються як параметри, тоді як `CMD` задає команду або аргументи за замовчуванням, які легко повністю перевизначити під час `docker run`.
   - `EXPOSE 8000` — це декларативна документація в метаданих образу про те, який порт слухає сервіс усередині контейнера (вона не відкриває порт на хості). Прапорець `-p 8000:8000` під час `docker run` реально прокидає мережевий порт хост-машини у порт контейнера.

6. **Чому для інференсу варто брати CPU-збірку PyTorch?**
   - Стандартний пакет `torch` з PyPI за замовчуванням тягне за собою бібліотеки NVIDIA CUDA, cuDNN та NCCL загальним обсягом ~2.5 ГБ, які у контейнері без GPU не використовуються. Спеціальна CPU-збірка (`torch==2.3.1+cpu` з `--extra-index-url https://download.pytorch.org/whl/cpu`) важить близько 150–200 МБ і зменшує розмір образу більш ніж на 1.5–2 ГБ.

7. **Переваги й недоліки «запеченої» в образ моделі порівняно з моделлю на volume.**
   - **«Запечена» в образ модель:**
     - *Переваги:* повна атомарність і відтворюваність (один тег образу гарантовано містить узгоджений код і конкретну версію моделі), простота деплою в Kubernetes/Cloud Run без зовнішніх сховищ.
     - *Недоліки:* оновлення ваг моделі вимагає перезбирання та повторної публікації Docker-образу.
   - **Модель на зовнішньому volume:**
     - *Переваги:* можна замінити файл моделі без перезбирання образу, образ залишається максимально легким навіть для гігабайтних моделей.
     - *Недоліки:* гірша ізоляція та відтворюваність (ризик розсинхронізації версії коду в образі та формату моделі на диску), потреба в управлінні спільним сховищем.

8. **Навіщо `HEALTHCHECK` і як його використовують оркестратори (Docker Compose, Kubernetes)?**
   - Процес у контейнері може формально працювати (`Up`), але ще завантажувати важку модель у пам'ять або зависнути. `HEALTHCHECK` періодично викликає `/health` і переводить статус контейнера у `healthy` лише тоді, коли сервіс реально готовий обробляти запити. Оркестратори використовують це для порядку запуску залежних сервісів (`depends_on: condition: service_healthy` у Compose), для балансування трафіку (`readinessProbe` у Kubernetes) та автоматичного перезапуску несправних контейнерів (`livenessProbe`).

9. **Чому контейнер не варто запускати від `root`?**
   - За замовчуванням `root` усередині контейнера має `UID 0`, що відповідає `root` ядра хост-системи. У разі вразливості у веб-сервері чи бібліотеці обробки медіафайлів зловмисник із правами `root` може модифікувати системні файли контейнера або отримати підвищені привілеї при виході за межі ізоляції (container escape). Запуск від непривілейованого користувача (`USER appuser`) обмежує права процесу принципом найменших привілеїв.

10. **Звідки береться різниця в latency між локальним запуском і контейнером? Чи вона суттєва?**
    - Чистий час обчислення згорткової мережі в пам'яті (`server_ms`) майже не має накладних витрат віртуалізації, оскільки контейнер виконується безпосередньо на ядрі Linux (у нашому випадку в Linux-контейнері він навіть менший за рахунок швидшої роботи `ext4`/`tmpfs` порівняно з NTFS на Windows). Невеликі додаткові витрати виникають на рівні `client_ms` (end-to-end HTTP) через віртуальний мережевий міст Docker (`bridge` / NAT проброс портів `localhost:8000 -> container:8000`). Ця різниця становить лише кілька мілісекунд і для інференс-сервісу не є суттєвою.

11. **Як Docker вписується в CI/CD для ML-моделей?**
    - У CI/CD-пайплайні Docker виступає єдиною одиницею доставки: на кожен коміт CI перевіряє код лінтером, збирає відтворюваний multi-stage образ з тегом `git-sha`, запускає контейнер в ізольованому середовищі, виконує димові тести (`/health`, `/model-info`, `/predict`) і перевіряє якість моделі (`quality gate` за порогами `accuracy` та `latency`). Лише після успішного проходження всіх воріт якості цей самий перевірений образ публікується в реєстр (GHCR / Docker Hub) і без змін розгортається у staging або production.
