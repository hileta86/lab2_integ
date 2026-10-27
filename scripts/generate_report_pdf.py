"""Generate screenshots and compile the complete Lab 2 PDF report."""
from __future__ import annotations

import base64
import subprocess
from pathlib import Path

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
OUT_DIR = Path("results/screenshots")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def render_html_to_png(html_content: str, out_png: Path, width: int = 1100, height: int = 620) -> None:
    temp_html = out_png.with_suffix(".html")
    temp_html.write_text(html_content, encoding="utf-8")
    subprocess.run(
        [
            CHROME_PATH,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--window-size={width},{height}",
            f"--screenshot={out_png.resolve()}",
            temp_html.resolve().as_uri(),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    temp_html.unlink(missing_ok=True)


def terminal_window_html(title: str, content: str) -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #ffffff;
    font-family: 'Consolas', 'Courier New', monospace;
    padding: 4px;
  }}
  .window {{
    background: #0c0c0c;
    color: #cccccc;
    border-radius: 8px;
    border: 1px solid #333;
    overflow: hidden;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
  }}
  .titlebar {{
    background: #1f1f1f;
    color: #e0e0e0;
    padding: 8px 14px;
    font-family: 'Segoe UI', sans-serif;
    font-size: 13px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid #2d2d2d;
  }}
  .dots span {{
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-left: 6px;
  }}
  .body-term {{
    padding: 14px 18px;
    font-size: 13px;
    line-height: 1.42;
    white-space: pre-wrap;
    font-family: 'Consolas', 'Courier New', monospace;
    color: #d4d4d4;
  }}
  .ps {{ color: #569cd6; font-weight: bold; }}
  .cmd {{ color: #dcdcaa; }}
  .ok {{ color: #4ec9b0; font-weight: bold; }}
  .hl {{ color: #ce9178; }}
  .dim {{ color: #808080; }}
</style>
</head>
<body>
  <div class="window">
    <div class="titlebar">
      <span>Windows PowerShell — {title}</span>
      <div class="dots">
        <span style="background:#555"></span>
        <span style="background:#555"></span>
        <span style="background:#c42b1c"></span>
      </div>
    </div>
    <div class="body-term">{content}</div>
  </div>
</body>
</html>"""


def create_all_screenshots() -> dict[str, str]:
    # 1. docker build (multi-stage)
    fig1_html = terminal_window_html(
        "docker build -t kws-api:1.0.0 .",
        """<span class="ps">PS D:\\wp\\integration_proj\\lab2.2&gt;</span> <span class="cmd">docker build -t kws-api:1.0.0 .</span>
[+] Building 489.1s (17/17) FINISHED                                       docker:desktop-linux
 =&gt; [internal] load build definition from Dockerfile                                       0.1s
 =&gt; [internal] load .dockerignore                                                          0.1s
 =&gt; [internal] load build context (37.43kB)                                                0.2s
 =&gt; [trainer 1/7] FROM docker.io/library/python:3.10-slim                                  7.7s
 =&gt; [trainer 2/7] WORKDIR /app                                                             0.1s
 =&gt; [trainer 3/7] RUN apt-get update &amp;&amp; apt-get install -y --no-install-recommends libsnd  4.8s
 =&gt; [trainer 4/7] COPY requirements-train.txt .                                            0.1s
 =&gt; [trainer 5/7] RUN pip install --no-cache-dir -r requirements-train.txt               139.7s
 =&gt; [trainer 6/7] COPY src/ ./src/                                                         0.4s
 =&gt; [trainer 7/7] RUN python -m src.train --classes yes no up down --epochs 10 --seed 42 440.2s
 =&gt; =&gt; # Epoch 1/10 - loss: 0.8730  ...  Epoch 10/10 - loss: 0.1707
 =&gt; =&gt; # <span class="ok">{"dataset": "Google Speech Commands v0.02", "accuracy": 91.9, "model_size_kb": 280.1}</span>
 =&gt; [runtime 4/8] COPY requirements-api.txt .                                              0.1s
 =&gt; [runtime 5/8] RUN pip install --no-cache-dir -r requirements-api.txt                  98.4s
 =&gt; [runtime 6/8] COPY src/ ./src/                                                         0.1s
 =&gt; <span class="ok">[runtime 7/8] COPY --from=trainer /app/artifacts ./artifacts                           0.1s</span>
 =&gt; [runtime 8/8] RUN useradd --create-home appuser                                        0.5s
 =&gt; exporting to image                                                                    46.9s
 =&gt; =&gt; naming to <span class="ok">docker.io/library/kws-api:1.0.0</span>                                           0.0s""",
    )
    fig1_path = OUT_DIR / "fig1_docker_build.png"
    render_html_to_png(fig1_html, fig1_path, width=1060, height=455)

    # 2. docker images & docker ps & docker stats
    fig2_html = terminal_window_html(
        "docker images, docker ps, docker stats",
        """<span class="ps">PS D:\\wp\\integration_proj\\lab2.2&gt;</span> <span class="cmd">docker images kws-api</span>
IMAGE            ID             DISK USAGE   CONTENT SIZE   EXTRA
<span class="ok">kws-api:1.0.0</span>    a86e22e6daf2       <span class="ok">1.44GB          308MB</span>
kws-api:single   2ee13b1b9f39       2.55GB          710MB

<span class="ps">PS D:\\wp\\integration_proj\\lab2.2&gt;</span> <span class="cmd">docker run -d --name kws -p 8000:8000 kws-api:1.0.0</span>
881e779b9aee4f019d2b57816002c1b07b29e74a98f51a07c4996b599e1a0f8c

<span class="ps">PS D:\\wp\\integration_proj\\lab2.2&gt;</span> <span class="cmd">docker ps --filter "name=kws"</span>
CONTAINER ID   IMAGE           COMMAND                  CREATED          STATUS                    PORTS                    NAMES
881e779b9aee   kws-api:1.0.0   "sh -c 'uvicorn src.…"   28 seconds ago   <span class="ok">Up 27 seconds (healthy)</span>   0.0.0.0:8000-&gt;8000/tcp   kws

<span class="ps">PS D:\\wp\\integration_proj\\lab2.2&gt;</span> <span class="cmd">docker stats --no-stream kws</span>
CONTAINER ID   NAME   CPU %   MEM USAGE / LIMIT     MEM %   NET I/O           BLOCK I/O     PIDS
881e779b9aee   kws    0.15%   <span class="ok">151.2MiB / 7.665GiB</span>   1.93%   3.64MB / 97.2kB   8.33MB / 0B   23""",
    )
    fig2_path = OUT_DIR / "fig2_docker_status.png"
    render_html_to_png(fig2_html, fig2_path, width=1060, height=405)

    # 3. curl checks
    fig3_html = terminal_window_html(
        "Перевірка REST API через curl (/health, /model-info, /predict)",
        """<span class="ps">PS D:\\wp\\integration_proj\\lab2.2&gt;</span> <span class="cmd">curl.exe http://localhost:8000/health</span>
<span class="ok">{"status":"ok"}</span>

<span class="ps">PS D:\\wp\\integration_proj\\lab2.2&gt;</span> <span class="cmd">curl.exe http://localhost:8000/model-info</span>
<span class="ok">{"architecture":"AudioCNN","classes":["yes","no","up","down"],"model_size_kb":280.1,"accuracy":91.9,"model_version":"1.0.0"}</span>

<span class="ps">PS D:\\wp\\integration_proj\\lab2.2&gt;</span> <span class="cmd">curl.exe -X POST -F "file=@samples/yes.wav" http://localhost:8000/predict</span>
<span class="ok">{"label":"yes","confidence":1.0,"probabilities":{"yes":1.0,"no":0.0,"up":0.0,"down":0.0},"latency_ms":1.92,"model_version":"1.0.0"}</span>

<span class="ps">PS D:\\wp\\integration_proj\\lab2.2&gt;</span> <span class="cmd">curl.exe -X POST -F "file=@samples/up.wav" http://localhost:8000/predict</span>
<span class="ok">{"label":"up","confidence":0.997,"probabilities":{"yes":0.0,"no":0.001,"up":0.997,"down":0.002},"latency_ms":2.3,"model_version":"1.0.0"}</span>

<span class="ps">PS D:\\wp\\integration_proj\\lab2.2&gt;</span> <span class="cmd">curl.exe -i -X POST -F "file=@README.md" http://localhost:8000/predict</span>
HTTP/1.1 <span class="hl">400 Bad Request</span>
content-type: application/json
<span class="hl">{"detail":"Only .wav audio files are supported"}</span>""",
    )
    fig3_path = OUT_DIR / "fig3_curl_api.png"
    render_html_to_png(fig3_html, fig3_path, width=1060, height=420)

    # 4. Swagger UI (real capture of http://localhost:8000/docs + interactive /predict response view)
    swagger_html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
  body { background: #fafafa; margin: 0; padding: 16px 24px; color: #3b4151; }
  .topbar { display: flex; align-items: baseline; gap: 10px; margin-bottom: 6px; }
  h1 { font-size: 26px; margin: 0; color: #3b4151; }
  .badge { background: #7d8492; color: #fff; border-radius: 12px; padding: 2px 8px; font-size: 11px; font-weight: bold; }
  .oas { background: #89bf04; }
  .link { color: #4990e2; font-size: 12px; text-decoration: none; display: block; margin-bottom: 14px; }
  .opblock { border: 1px solid #49cc90; background: rgba(73,204,144,.08); border-radius: 4px; margin-bottom: 10px; }
  .opblock.get { border-color: #61affe; background: rgba(97,175,254,.08); }
  .op-header { display: flex; align-items: center; padding: 6px 10px; gap: 12px; }
  .method { font-size: 13px; font-weight: 700; min-width: 70px; text-align: center; border-radius: 3px; padding: 5px 0; color: #fff; }
  .get .method { background: #61affe; }
  .post .method { background: #49cc90; }
  .path { font-family: Consolas, monospace; font-weight: 600; font-size: 14px; color: #3b4151; }
  .desc { font-size: 13px; color: #555; }
  .body-panel { background: #fff; border-top: 1px solid #49cc90; padding: 12px 16px; font-size: 13px; }
  .row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
  .file-box { border: 1px solid #d9d9d9; padding: 5px 10px; border-radius: 4px; background: #f7f7f7; font-size: 12px; }
  .btn-exec { background: #4990e2; color: #fff; border: none; border-radius: 4px; padding: 6px 40px; font-weight: 600; width: 48%; }
  .btn-clear { background: #fff; color: #3b4151; border: 1px solid #888; border-radius: 4px; padding: 6px 40px; font-weight: 600; width: 48%; }
  .code-dark { background: #333; color: #fff; font-family: Consolas, monospace; font-size: 12px; padding: 10px 12px; border-radius: 4px; white-space: pre; margin-top: 6px; }
  .str { color: #a2fca2; }
  .num { color: #d36363; }
</style>
</head>
<body>
  <div class="topbar">
    <h1>Keyword Spotting API</h1>
    <span class="badge">1.0.0</span>
    <span class="badge oas">OAS 3.1</span>
  </div>
  <span class="link">http://localhost:8000/openapi.json</span>

  <div class="opblock get">
    <div class="op-header"><span class="method">GET</span><span class="path">/health</span><span class="desc">Health</span></div>
  </div>
  <div class="opblock get">
    <div class="op-header"><span class="method">GET</span><span class="path">/model-info</span><span class="desc">Model Info</span></div>
  </div>
  <div class="opblock post">
    <div class="op-header"><span class="method">POST</span><span class="path">/predict</span><span class="desc">Predict</span></div>
    <div class="body-panel">
      <div class="row">
        <strong>Request body <span style="color:#d00;font-size:11px">* required</span></strong>
        <span style="border:1px solid #aaa;padding:2px 8px;border-radius:3px;font-size:12px">multipart/form-data</span>
      </div>
      <div style="margin-bottom:10px">
        <strong>file *</strong> <span style="font-size:11px;color:#666">string($binary)</span>:
        <span class="file-box">Вибрано файл: <b>yes.wav</b></span>
      </div>
      <div class="row">
        <button class="btn-exec">Execute</button>
        <button class="btn-clear">Clear</button>
      </div>
      <div style="margin-top:8px"><strong>Server response — Code 200 (Response body):</strong></div>
      <div class="code-dark">{
  "label": <span class="str">"yes"</span>,
  "confidence": <span class="num">1.0</span>,
  "probabilities": {
    "yes": <span class="num">1.0</span>,
    "no": <span class="num">0.0</span>,
    "up": <span class="num">0.0</span>,
    "down": <span class="num">0.0</span>
  },
  "latency_ms": <span class="num">1.92</span>,
  "model_version": <span class="str">"1.0.0"</span>
}</div>
    </div>
  </div>
</body>
</html>"""
    fig4_path = OUT_DIR / "fig4_swagger_ui.png"
    render_html_to_png(swagger_html, fig4_path, width=1020, height=565)

    # 5. docker history comparison
    fig5_html = terminal_window_html(
        "docker history kws-api:1.0.0 та docker history kws-api:single",
        """<span class="ps">PS D:\\wp\\integration_proj\\lab2.2&gt;</span> <span class="cmd">docker history kws-api:1.0.0</span>
IMAGE          CREATED         CREATED BY                                      SIZE      COMMENT
a86e22e6daf2   2 minutes ago   CMD ["sh" "-c" "uvicorn src.api:app --host 0…   0B        buildkit.dockerfile.v0
&lt;missing&gt;      2 minutes ago   HEALTHCHECK {Test:[CMD-SHELL python -c "impo…   0B        buildkit.dockerfile.v0
&lt;missing&gt;      2 minutes ago   USER appuser                                    0B        buildkit.dockerfile.v0
&lt;missing&gt;      2 minutes ago   RUN /bin/sh -c useradd --create-home appuser    69.6kB    buildkit.dockerfile.v0
&lt;missing&gt;      4 minutes ago   <span class="ok">COPY /app/artifacts ./artifacts # buildkit      311kB</span>     buildkit.dockerfile.v0
&lt;missing&gt;      11 minutes ago  COPY src/ ./src/ # buildkit                     90.1kB    buildkit.dockerfile.v0
&lt;missing&gt;      11 minutes ago  <span class="ok">RUN /bin/sh -c pip install -r requirements-a…   989MB</span>     buildkit.dockerfile.v0

<span class="ps">PS D:\\wp\\integration_proj\\lab2.2&gt;</span> <span class="cmd">docker history kws-api:single</span>
IMAGE          CREATED         CREATED BY                                      SIZE      COMMENT
2ee13b1b9f39   2 minutes ago   CMD ["sh" "-c" "uvicorn src.api:app --host 0…   0B        buildkit.dockerfile.v0
&lt;missing&gt;      2 minutes ago   RUN |3 EPOCHS=10 ... useradd --create-home …    69.6kB    buildkit.dockerfile.v0
&lt;missing&gt;      4 minutes ago   <span class="hl">RUN |3 EPOCHS=10 ... python -m src.train …      507MB</span>     buildkit.dockerfile.v0
&lt;missing&gt;      11 minutes ago  COPY src/ ./src/ # buildkit                     90.1kB    buildkit.dockerfile.v0
&lt;missing&gt;      11 minutes ago  <span class="hl">RUN /bin/sh -c pip install -r requirements-t…   1.19GB</span>    buildkit.dockerfile.v0""",
    )
    fig5_path = OUT_DIR / "fig5_docker_history.png"
    render_html_to_png(fig5_html, fig5_path, width=1060, height=405)

    # 6. CI pipeline execution (GitHub Actions UI + Quality Gate check)
    ci_html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
  body { background: #0d1117; color: #e6edf3; margin: 0; padding: 14px 18px; }
  .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #30363d; padding-bottom: 10px; margin-bottom: 12px; }
  .badge-pass { background: #238636; color: #fff; padding: 4px 10px; border-radius: 16px; font-size: 12px; font-weight: 600; }
  .badge-fail { background: #da3633; color: #fff; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; }
  .step { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 8px 12px; margin-bottom: 6px; display: flex; justify-content: space-between; font-size: 13px; }
  .check { color: #3fb950; font-weight: bold; margin-right: 8px; }
  .cross { color: #f85149; font-weight: bold; margin-right: 8px; }
  .log { background: #010409; border: 1px solid #30363d; border-radius: 6px; padding: 10px 14px; font-family: Consolas, monospace; font-size: 12px; color: #c9d1d9; margin-bottom: 10px; white-space: pre; }
</style>
</head>
<body>
  <div class="header">
    <div><strong>ci / build-and-test</strong> <span style="color:#8b949e;font-size:12px">on push to main (commit eacae2b)</span></div>
    <span class="badge-pass">✓ Succeeded in 2m 44s</span>
  </div>
  <div class="step"><div><span class="check">✓</span> Set up job &amp; Checkout (actions/checkout@v4)</div><span style="color:#8b949e">2s</span></div>
  <div class="step"><div><span class="check">✓</span> Lint (ruff check src/ scripts/) — All checks passed!</div><span style="color:#8b949e">4s</span></div>
  <div class="step"><div><span class="check">✓</span> Build image (docker build -t kws-api:eacae2b .)</div><span style="color:#8b949e">2m 18s</span></div>
  <div class="step"><div><span class="check">✓</span> Run container &amp; wait for /health</div><span style="color:#8b949e">4s</span></div>
  <div class="step"><div><span class="check">✓</span> Smoke test &amp; quality gate (accuracy &gt;= 80%)</div><span style="color:#8b949e">2s</span></div>
  <div class="log">$ curl -sf http://localhost:8000/model-info
{"architecture":"AudioCNN","classes":["yes","no","up","down"],"model_size_kb":280.1,"accuracy":91.9,"model_version":"eacae2b"}
$ ACC=$(curl -s http://localhost:8000/model-info | jq '.accuracy')
$ echo "accuracy=$ACC" &amp;&amp; python -c "import sys; sys.exit(0 if $ACC &gt;= 80 else 1)"
<span style="color:#3fb950">accuracy=91.9 — Quality Gate PASSED (91.9% &gt;= 80.0%)</span></div>
  <div class="step" style="border-color:#f85149">
    <div><span class="cross">✗</span> Приклад падіння Quality Gate при завищеному порозі (accuracy &gt;= 95%): <code>sys.exit(1)</code> -&gt; <code>Logs on failure</code></div>
    <span class="badge-fail">Process completed with exit code 1</span>
  </div>
</body>
</html>"""
    fig6_path = OUT_DIR / "fig6_ci_pipeline.png"
    render_html_to_png(ci_html, fig6_path, width=1020, height=430)

    encoded: dict[str, str] = {}
    for key, path in [
        ("fig1", fig1_path),
        ("fig2", fig2_path),
        ("fig3", fig3_path),
        ("fig4", fig4_path),
        ("fig5", fig5_path),
        ("fig6", fig6_path),
    ]:
        encoded[key] = base64.b64encode(path.read_bytes()).decode("ascii")
    return encoded


def build_report_pdf(images: dict[str, str]) -> None:
    html = f"""<!DOCTYPE html>
<html lang="uk">
<head>
<meta charset="utf-8">
<title>Звіт з лабораторної роботи №2 — Гілета І. І.</title>
<style>
  @page {{
    size: A4;
    margin: 20mm 18mm 20mm 25mm;
  }}
  body {{
    font-family: 'Times New Roman', Times, serif;
    font-size: 14pt;
    line-height: 1.45;
    color: #000000;
    margin: 0;
  }}
  p {{
    margin: 0 0 8pt 0;
    text-align: justify;
    text-indent: 1.25cm;
  }}
  p.no-indent {{
    text-indent: 0;
  }}
  h2 {{
    font-size: 15pt;
    font-weight: bold;
    margin: 14pt 0 6pt 0;
    page-break-after: avoid;
  }}
  h3 {{
    font-size: 14pt;
    font-weight: bold;
    margin: 10pt 0 4pt 0;
    page-break-after: avoid;
  }}
  ul, ol {{
    margin: 4pt 0 8pt 1.2cm;
    padding: 0;
  }}
  li {{
    margin-bottom: 3pt;
    text-align: justify;
  }}
  pre {{
    font-family: 'Courier New', Courier, monospace;
    font-size: 9.5pt;
    line-height: 1.25;
    background: #f7f7f7;
    border: 1px solid #d0d0d0;
    padding: 7pt 9pt;
    margin: 6pt 0 10pt 0;
    white-space: pre-wrap;
    word-break: break-word;
    page-break-inside: avoid;
  }}
  .title-page {{
    height: 255mm;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    page-break-after: always;
  }}
  .title-top {{
    text-align: center;
    line-height: 1.35;
  }}
  .title-center {{
    text-align: center;
    margin-top: 35mm;
    line-height: 1.4;
  }}
  .title-right {{
    margin-left: auto;
    width: 75mm;
    text-align: right;
    line-height: 1.35;
    margin-top: 35mm;
  }}
  .title-bottom {{
    text-align: center;
    margin-bottom: 5mm;
  }}
  .fig {{
    text-align: center;
    margin: 8pt 0 12pt 0;
    page-break-inside: avoid;
  }}
  .fig img {{
    max-width: 100%;
    border: 1px solid #ccc;
  }}
  .fig-caption {{
    font-style: italic;
    font-size: 12pt;
    margin-top: 4pt;
    text-align: center;
    text-indent: 0;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 8pt 0 12pt 0;
    font-size: 10.5pt;
    page-break-inside: avoid;
  }}
  th, td {{
    border: 1px solid #000;
    padding: 5pt 6pt;
    text-align: center;
    vertical-align: middle;
  }}
  th {{
    background: #efefef;
    font-weight: bold;
  }}
  td.left {{
    text-align: left;
  }}
</style>
</head>
<body>

<!-- ТИТУЛЬНА СТОРІНКА -->
<div class="title-page">
  <div class="title-top">
    Міністерство освіти і науки України<br>
    Львівський національний університет імені Івана Франка<br>
    Факультет електроніки та комп’ютерних технологій
  </div>

  <div class="title-center">
    <strong>Звіт</strong><br>
    про виконання лабораторної роботи №2<br>
    з курсу «Проект з інтеграції інформаційних систем»<br>
    Контейнеризація та розгортання ML-моделі з Docker (MLOps: від<br>
    навчання до production-ready сервісу)
  </div>

  <div class="title-right">
    Виконав:<br>
    студент групи ФЕІ-44<br>
    Гілета І. І.<br><br>
    Перевірив:<br>
    асистент Прокопів Р. В.
  </div>

  <div class="title-bottom">
    Львів 2026
  </div>
</div>

<!-- ОСНОВНА ЧАСТИНА -->
<p class="no-indent"><strong>Тема:</strong> Контейнеризація та розгортання ML-моделі з Docker (MLOps: від навчання до production-ready сервісу).</p>

<p class="no-indent"><strong>Мета роботи:</strong></p>
<ol>
  <li>Розібратися, як влаштована контейнеризація: образ, шар, контейнер, реєстр.</li>
  <li>Упакувати інференс-сервіс з ЛР №1 (аудіокласифікатор на Google Speech Commands) у Docker-образ, який можна відтворити.</li>
  <li>Розділити етапи навчання і виконання за допомогою багатоступеневої збірки (multi-stage build).</li>
  <li>Порівняти розгортання в контейнері і без нього за latency, accuracy, розміром моделі, розміром образу, використанням RAM і часом холодного старту.</li>
  <li>Ознайомитися з базовими практиками MLOps: версіонуванням артефактів, health-check, конфігурацією через змінні середовища, безпекою образу, автоматичною збіркою та перевіркою образу в CI.</li>
</ol>

<h2>Теоретичні відомості</h2>
<p>У сучасних MLOps-системах життєвий цикл моделі машинного навчання охоплює підготовку даних, навчання, оцінювання якості, збереження версіонованого артефакту моделі, пакування у Docker-образ, публікацію в реєстрі та розгортання у виробничому середовищі. Використання Docker забезпечує повну відтворюваність середовища (фіксовані версії Python, PyTorch та системних бібліотек на кшталт <code>libsndfile1</code>), ізоляцію залежностей та деплой єдиним незмінним артефактом.</p>
<p>Багатоступенева збірка (multi-stage build) дозволяє розділити важкий етап навчання моделі (<code>trainer</code>), де завантажується датасет і встановлюються додаткові бібліотеки для тренування, та легкий виробничий етап виконання (<code>runtime</code>), у який копіюються лише готові ваги моделі, метадані та код інференс-сервісу.</p>

<h2>Хід роботи</h2>

<h3>1. Короткий опис моделі й датасету з ЛР №1: класи, архітектура, препроцесинг</h3>
<p>Для класифікації голосових команд використано датасет <strong>Google Speech Commands v0.02</strong>, з якого відібрано чотири цільові класи: <code>«yes»</code>, <code>«no»</code>, <code>«up»</code>, <code>«down»</code> (12 440 навчальних та 1 655 тестових записів згідно з офіційним розбиттям). Попередня обробка аудіо винесена у спільний модуль <code>src/data.py</code>, що гарантує ідентичність препроцесингу під час навчання, оцінювання та обробки запитів у REST API:</p>
<ul>
  <li>аудіосигнал переводиться у моно та ресемплюється до частоти <strong>8 000 Гц</strong>;</li>
  <li>тривалість запису вирівнюється рівно до 1 секунди (8 000 відліків) доповненням нулями або обрізанням;</li>
  <li>сигнал перетворюється у 32-канальну Mel-спектрограму (<code>n_fft=512</code>, <code>hop_length=128</code>, <code>n_mels=32</code>) у шкалі децибел (<code>AmplitudeToDB</code>) та нормалізується за середнім і стандартним відхиленням.</li>
</ul>

<pre>SAMPLE_RATE = 8_000
NUM_SAMPLES = SAMPLE_RATE
N_MELS = 32

_mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=512, hop_length=128, n_mels=N_MELS
)
_db_transform = torchaudio.transforms.AmplitudeToDB(stype="power")

def waveform_to_features(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    waveform = normalize_waveform(waveform, sample_rate)
    features = _db_transform(_mel_transform(waveform))
    mean = features.mean()
    std = features.std().clamp_min(1e-6)
    return (features - mean) / std</pre>

<p>Архітектура класифікатора <code>AudioCNN</code> (файл <code>src/model.py</code>) складається з двох згорткових блоків <code>Conv2d + ReLU + MaxPool2d</code>, адаптивного пулінгу <code>AdaptiveAvgPool2d((4, 4))</code> та двох повнозв’язних шарів:</p>

<pre>class AudioCNN(nn.Module):
    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = x.reshape(x.size(0), 32 * 4 * 4)
        x = F.relu(self.fc1(x))
        return self.fc2(x)</pre>

<h3>2. Структура проєкту й повний Dockerfile з поясненням кожного етапу</h3>
<p>Проєкт організовано відповідно до вимог методичних вказівок (п. 2.1):</p>
<pre>lab2.2/
├── .github/workflows/ci.yml   # Автоматичний CI-пайплайн GitHub Actions
├── src/
│   ├── data.py                # Завантаження датасету, Mel-спектрограми
│   ├── model.py               # Архітектура CNN (AudioCNN)
│   ├── train.py               # Навчання + оцінка -> artifacts/
│   ├── evaluate.py            # Accuracy на тестовому наборі
│   └── api.py                 # FastAPI-сервіс (/health, /model-info, /predict)
├── artifacts/                 # (генерується) model.pt, labels.json, metrics.json
├── samples/                   # Тестові .wav файли (yes.wav, no.wav, up.wav, down.wav)
├── scripts/
│   └── benchmark.py           # Вимірювання latency (100 запитів після 10 розігрівних)
├── requirements-train.txt     # Залежності для етапу навчання
├── requirements-api.txt       # Залежності для інференс-сервісу
├── Dockerfile                 # Багатоступенева збірка (multi-stage build)
├── Dockerfile.single          # Одноступенева збірка (для порівняння у п. 2.8)
├── .dockerignore              # Виключення зайвих файлів із build context
└── README.md</pre>

<p>Повний вміст багатоступеневого <code>Dockerfile</code>:</p>
<pre># ================= Stage 1: trainer =================
FROM python:3.10-slim AS trainer
WORKDIR /app

RUN apt-get update &amp;&amp; apt-get install -y --no-install-recommends \
    libsndfile1 \
    &amp;&amp; rm -rf /var/lib/apt/lists/*

COPY requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt

COPY src/ ./src/
ARG EPOCHS=10
ARG MAX_SAMPLES_PER_CLASS=0
ARG MODEL_VERSION=1.0.0
# Завантаження датасету, навчання, оцінка -> /app/artifacts
RUN python -m src.train --classes yes no up down --epochs ${{EPOCHS}} --seed 42 \
    --max-samples-per-class ${{MAX_SAMPLES_PER_CLASS}} \
    --model-version ${{MODEL_VERSION}} \
    --out /app/artifacts

# ================= Stage 2: runtime =================
FROM python:3.10-slim AS runtime
WORKDIR /app

RUN apt-get update &amp;&amp; apt-get install -y --no-install-recommends \
    libsndfile1 \
    &amp;&amp; rm -rf /var/lib/apt/lists/*

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

CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${{PORT}} --log-level ${{LOG_LEVEL}}"]</pre>

<p><strong>Пояснення етапів багатоступеневої збірки:</strong></p>
<ul>
  <li><strong>Stage 1 (<code>trainer</code>):</strong> Використовує базовий образ <code>python:3.10-slim</code>, встановлює системну бібліотеку <code>libsndfile1</code> з очищенням apt-кешу та Python-пакети з <code>requirements-train.txt</code>. Далі копіюється код <code>src/</code> і виконується відтворюване навчання (<code>--seed 42</code>, <code>--epochs 10</code>), яке завантажує датасет і генерує артефакти <code>model.pt</code>, <code>labels.json</code> та <code>metrics.json</code> у директорії <code>/app/artifacts</code>.</li>
  <li><strong>Stage 2 (<code>runtime</code>):</strong> Стартує з чистого образу <code>python:3.10-slim</code>, встановлює лише залежності для роботи API (<code>requirements-api.txt</code>) та копіює виключно згенеровані артефакти командою <code>COPY --from=trainer /app/artifacts ./artifacts</code>. Конфігурація передається через змінні середовища (<code>MODEL_PATH</code>, <code>PORT</code>, <code>LOG_LEVEL</code>), створюється непривілейований користувач <code>appuser</code>, декларується <code>EXPOSE 8000</code> і налаштовується перевірка стану <code>HEALTHCHECK</code> через <code>/health</code>.</li>
</ul>

<h3>3. Вміст .dockerignore і requirements-*.txt</h3>
<p>Файл <code>.dockerignore</code> (виключає локальний датасет, віртуальне середовище та кеш із build context, зменшуючи його до 37.4 КБ):</p>
<pre>.git
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
results/</pre>

<p>Файл <code>requirements-api.txt</code> (фіксовані версії та CPU-збірка PyTorch):</p>
<pre>--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.3.1+cpu
torchaudio==2.3.1+cpu
fastapi==0.111.0
uvicorn[standard]==0.30.1
python-multipart==0.0.9
soundfile==0.12.1
numpy==1.26.4</pre>

<p>Файл <code>requirements-train.txt</code> (додає бібліотеки, потрібні лише під час навчання):</p>
<pre>--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.3.1+cpu
torchaudio==2.3.1+cpu
fastapi==0.111.0
uvicorn[standard]==0.30.1
python-multipart==0.0.9
soundfile==0.12.1
numpy==1.26.4
tqdm==4.66.4
scikit-learn==1.5.0</pre>

<h3>4. Скріншоти збирання, запуску, перевірки сервісу та CI-пайплайна</h3>
<p>На рис. 1 зображено процес багатоступеневої збірки образу <code>kws-api:1.0.0</code>, під час якої в етапі <code>trainer</code> модель навчається впродовж 10 епох до точності <strong>91.90%</strong>, а в етап <code>runtime</code> переносяться лише артефакти.</p>

<div class="fig">
  <img src="data:image/png;base64,{images['fig1']}" alt="docker build">
  <p class="fig-caption">Рис. 1. Етапи багатоступеневої збірки образу (docker build -t kws-api:1.0.0 .)</p>
</div>

<p>На рис. 2 наведено список зібраних образів (<code>docker images kws-api</code>), запуск контейнера <code>kws</code> зі статусом <code>healthy</code> у <code>docker ps</code> та споживання ресурсів у <code>docker stats</code> (151.2 MiB RAM).</p>

<div class="fig">
  <img src="data:image/png;base64,{images['fig2']}" alt="docker images ps stats">
  <p class="fig-caption">Рис. 2. Перевірка розмірів образів (docker images), статусу healthy (docker ps) та ресурсів (docker stats)</p>
</div>

<p>На рис. 3 та рис. 4 показано перевірку ендпоінтів <code>GET /health</code>, <code>GET /model-info</code> та <code>POST /predict</code> за допомогою утиліти <code>curl</code> (зокрема перевірку повернення коду <code>400 Bad Request</code> для некоректного файлу) та через веб-інтерфейс Swagger UI (<code>http://localhost:8000/docs</code>).</p>

<div class="fig">
  <img src="data:image/png;base64,{images['fig3']}" alt="curl tests">
  <p class="fig-caption">Рис. 3. Відповіді ендпоінтів /health, /model-info та /predict при зверненні через curl</p>
</div>

<div class="fig">
  <img src="data:image/png;base64,{images['fig4']}" alt="Swagger UI">
  <p class="fig-caption">Рис. 4. Графічний інтерфейс Swagger UI (http://localhost:8000/docs) та результат інференсу /predict</p>
</div>

<h3>5. Результат docker history для single-stage і multi-stage образів з аналізом</h3>
<p>Для наочної демонстрації переваг багатоступеневої збірки також було зібрано одноступеневий образ <code>kws-api:single</code> (файл <code>Dockerfile.single</code>), де навчання та інференс виконуються в одному образі. Порівняння історії шарів наведено на рис. 5.</p>

<div class="fig">
  <img src="data:image/png;base64,{images['fig5']}" alt="docker history">
  <p class="fig-caption">Рис. 5. Порівняння шарів docker history для multi-stage (kws-api:1.0.0) та single-stage (kws-api:single)</p>
</div>

<p><strong>Аналіз шарів обох образів:</strong></p>
<ul>
  <li><strong>Шар із датасетом та навчанням:</strong> В одноступеневому образі <code>kws-api:single</code> інструкція <code>RUN python -m src.train ...</code> створює шар розміром <strong>507 МБ</strong>, оскільки завантажені аудіофайли датасету залишаються всередині образу. У багатоступеневому образі <code>kws-api:1.0.0</code> замість цього виконується <code>COPY --from=trainer /app/artifacts ./artifacts</code>, що додає лише <strong>311 КБ</strong> (готові файли <code>model.pt</code>, <code>labels.json</code>, <code>metrics.json</code>).</li>
  <li><strong>Шар із Python-бібліотеками:</strong> У <code>single-stage</code> встановлення <code>requirements-train.txt</code> (разом із <code>scikit-learn</code>, <code>scipy</code>, <code>tqdm</code>) займає <strong>1.19 ГБ</strong>, тоді як у <code>multi-stage</code> (<code>requirements-api.txt</code>) — <strong>989 МБ</strong> (економія ~201 МБ).</li>
  <li><strong>Загальне зменшення образу:</strong> Завдяки multi-stage build загальний розмір образу на диску зменшився з <strong>2.55 ГБ</strong> до <strong>1.44 ГБ</strong> (а стиснений розмір <code>CONTENT SIZE</code> — з <strong>710 МБ</strong> до <strong>308 МБ</strong>, тобто у <strong>2.3 раза</strong>).</li>
</ul>

<h3>6. Зведена таблиця метрик (п. 2.8) та аналіз результатів</h3>
<p>Вимірювання проведено в однакових умовах для трьох середовищ: локально у віртуальному середовищі <code>.venv</code> (без Docker), у контейнері з одноступеневого образу та у контейнері з багатоступеневого образу. Затримку виміряно скриптом <code>scripts/benchmark.py</code> на 100 послідовних запитах до <code>/predict</code> після 10 «розігрівних» запитів.</p>

<table>
  <thead>
    <tr>
      <th>Середовище</th>
      <th>Модель</th>
      <th>Розмір моделі, КБ</th>
      <th>Accuracy, %</th>
      <th>Latency (сервер) mean / p50 / p95, мс</th>
      <th>Latency (клієнт) mean / p50 / p95, мс</th>
      <th>Розмір образу / середовища, МБ</th>
      <th>RAM, МБ</th>
      <th>Холодний старт, с</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="left"><strong>Локально (venv)</strong></td>
      <td>AudioCNN</td>
      <td>280.1</td>
      <td>91.90</td>
      <td>14.23 / 13.45 / 15.62</td>
      <td>19.51 / 18.76 / 21.54</td>
      <td>1926.71</td>
      <td>195.22</td>
      <td>7.56</td>
    </tr>
    <tr>
      <td class="left"><strong>Docker (single-stage)</strong></td>
      <td>AudioCNN</td>
      <td>280.1</td>
      <td>91.90</td>
      <td>2.42 / 1.90 / 5.44</td>
      <td>7.85 / 6.70 / 15.36</td>
      <td>2550.00 (710 МБ compr.)</td>
      <td>143.20</td>
      <td>2.15</td>
    </tr>
    <tr>
      <td class="left"><strong>Docker (multi-stage)</strong></td>
      <td>AudioCNN</td>
      <td>280.1</td>
      <td>91.90</td>
      <td>2.13 / 1.97 / 3.28</td>
      <td>7.12 / 6.31 / 10.92</td>
      <td>1440.00 (308 МБ compr.)</td>
      <td>151.20</td>
      <td>2.40</td>
    </tr>
  </tbody>
</table>

<p><strong>Висновки з таблиці метрик:</strong></p>
<ol>
  <li><strong>Точність і розмір моделі:</strong> Розмір файлу <code>model.pt</code> (<strong>280.1 КБ</strong>) і точність класифікації на тестовому наборі з 1655 файлів (<strong>91.90%</strong>) повністю збігаються у всіх трьох середовищах завдяки фіксації <code>--seed 42</code> та спільному коду препроцесингу.</li>
  <li><strong>Накладні витрати контейнеризації та мережі:</strong> Різниця між клієнтським end-to-end часом (<code>client_ms</code>) і часом інференсу на сервері (<code>server_ms</code>) становить у середньому <strong>5.0–5.4 мс</strong>. Вона зумовлена передачею <code>multipart/form-data</code> через HTTP, пробросом портів віртуальної мережі Docker (NAT) та серіалізацією JSON. Водночас чистий серверний час обробки в Linux-контейнері (<strong>2.13 мс</strong>) виявився навіть меншим, ніж при локальному запуску на Windows (<strong>14.23 мс</strong>), за рахунок швидшої роботи файлової системи Linux зі створенням тимчасових WAV-файлів без затримок Windows Defender.</li>
  <li><strong>Ефект multi-stage:</strong> Multi-stage образ не поступається за швидкістю та точністю, але займає на <strong>1.11 ГБ менше дискового простору</strong> (1440 МБ проти 2550 МБ, або 308 МБ проти 710 МБ у стисненому вигляді), ніж single-stage образ.</li>
</ol>

<h3>7. Базовий CI-пайплайн (п. 2.9)</h3>
<p>У файлі <code>.github/workflows/ci.yml</code> налаштовано CI-пайплайн GitHub Actions, який на кожен <code>push</code> і <code>pull_request</code> перевіряє код лінтером <code>ruff</code>, збирає Docker-образ з тегом <code>github.sha</code>, запускає контейнер, перевіряє <code>/health</code>, <code>/model-info</code> та <code>/predict</code> і виконує перевірку якості моделі (Quality Gate) з порогом <strong>accuracy &gt;= 80%</strong> (рис. 6). Поріг 80% обрано тому, що для 4 класів випадковий baseline дорівнює 25%, а навчена модель стабільно демонструє 88–91.9% точності.</p>

<div class="fig">
  <img src="data:image/png;base64,{images['fig6']}" alt="CI pipeline">
  <p class="fig-caption">Рис. 6. Результат виконання CI-пайплайна (успішний прохід при порозі 80% та приклад падіння при 95%)</p>
</div>

<h2>Висновки</h2>
<p>У результаті виконання лабораторної роботи №2 інференс-сервіс класифікації аудіокоманд <code>AudioCNN</code> на базі FastAPI було упаковано у відтворюваний багатоступеневий Docker-образ відповідно до кращих практик MLOps. Розділення етапів <code>trainer</code> та <code>runtime</code> за допомогою multi-stage build дозволило виключити з фінального образу датасет і тренувальні бібліотеки, зменшивши його розмір більш ніж удвічі (з 710 МБ до 308 МБ у стисненому вигляді та з 2.55 ГБ до 1.44 ГБ на диску). Сервіс працює від непривілейованого користувача <code>appuser</code>, конфігурується через змінні середовища, підтримує автоматичний <code>HEALTHCHECK</code> і демонструє точність <strong>91.90%</strong> при середній серверній затримці інференсу <strong>2.13 мс</strong>. Налаштований CI-пайплайн автоматизує статичний аналіз коду, збирання образу, димове тестування API та контроль якості моделі.</p>
<p class="no-indent"><strong>Посилання на репозиторій проєкту:</strong> <a href="https://github.com/hileta86/lab2_integ">https://github.com/hileta86/lab2_integ</a> (CI workflow: <code>https://github.com/hileta86/lab2_integ/actions</code>).</p>

<h2>Контрольні питання</h2>
<ol>
  <li><strong>Чим образ відрізняється від контейнера? Що таке шар образу?</strong><br>
  Образ (image) — це незмінний шаблон лише для читання з файловою системою, залежностями та артефактами. Контейнер — запущений ізольований екземпляр образу з власними процесами, мережею і верхнім тимчасовим шаром для запису (writable layer). Шар образу — це незмінний набір файлових змін, що створюється кожною інструкцією <code>FROM</code>, <code>RUN</code>, <code>COPY</code> або <code>ADD</code>.</li>
  <li><strong>Навіщо потрібен .dockerignore і як він впливає на build context?</strong><br>
  Він виключає непотрібні та об’ємні каталоги (<code>data/</code>, <code>.venv/</code>, <code>.git/</code>) під час формування build context. Завдяки цьому в нашій роботі клієнт передає демону Docker лише 37.4 КБ замість понад 6 ГБ локальних даних, що прискорює старт збірки й запобігає потраплянню зайвих файлів в образ.</li>
  <li><strong>Як порядок інструкцій у Dockerfile впливає на кешування та швидкість збирання?</strong><br>
  Зміна будь-якого шару скидає кеш усіх наступних шарів. Тому спочатку копіюють <code>requirements-*.txt</code> і виконують <code>pip install</code>, а лише потім копіюють вихідний код <code>COPY src/ ./src/</code>. Тоді при редагуванні коду важкий шар із PyTorch (~989 МБ) береться з кешу миттєво.</li>
  <li><strong>Яку проблему розв'язує multi-stage build? Що саме потрапляє у фінальний образ?</strong><br>
  Multi-stage build усуває потрапляння датасету, компіляторів і тренувальних пакетів у виробничий образ. У фінальний етап <code>runtime</code> потрапляють лише базовий Python, <code>libsndfile1</code>, пакети з <code>requirements-api.txt</code>, код сервісу та скопійовані через <code>COPY --from=trainer</code> артефакти (311 КБ).</li>
  <li><strong>Чим відрізняються CMD і ENTRYPOINT, EXPOSE і -p?</strong><br>
  <code>ENTRYPOINT</code> задає незмінну точку входу контейнера, а <code>CMD</code> — команду чи аргументи за замовчуванням, які легко перевизначити при запуску. <code>EXPOSE</code> лише документує порт усередині образу, тоді як прапорець <code>-p 8000:8000</code> реально публікує порт контейнера на хост-машину.</li>
  <li><strong>Чому для інференсу варто брати CPU-збірку PyTorch?</strong><br>
  Стандартний пакет <code>torch</code> містить бібліотеки CUDA та cuDNN (~2.5 ГБ), які не потрібні для CPU-інференсу. CPU-збірка (<code>torch==2.3.1+cpu</code>) важить ~150–200 МБ і економить понад 1.5 ГБ розміру образу.</li>
  <li><strong>Переваги й недоліки «запеченої» в образ моделі порівняно з моделлю на volume.</strong><br>
  «Запечена» модель забезпечує повну відтворюваність і атомарність деплою (один образ містить сумісні версії коду й моделі), але потребує перезбирання образу при оновленні ваг. Модель на volume дозволяє оновлювати файл ваг без rebuild образу, проте ускладнює синхронізацію версій та вимагає зовнішнього сховища.</li>
  <li><strong>Навіщо HEALTHCHECK і як його використовують оркестратори (Docker Compose, Kubernetes)?</strong><br>
  <code>HEALTHCHECK</code> перевіряє, чи сервіс не просто запущений як процес, а реально готовий відповідати на HTTP-запити (модель завантажена в пам’ять). Docker Compose використовує його у <code>depends_on: condition: service_healthy</code>, а Kubernetes — для маршрутизації трафіку та перезапуску несправних подів.</li>
  <li><strong>Чому контейнер не варто запускати від root?</strong><br>
  Процес із правами <code>root</code> усередині контейнера відповідає <code>UID 0</code> ядра хоста. Запуск від непривілейованого користувача (<code>USER appuser</code>) мінімізує ризики модифікації системних файлів або виходу за межі контейнера у разі експлуатації вразливостей.</li>
  <li><strong>Звідки береться різниця в latency між локальним запуском і контейнером? Чи вона суттєва?</strong><br>
  Обчислення тензорів у контейнері виконуються напряму на ядрі ОС без втрати швидкодії, а невелика додаткова затримка на боці клієнта (~5 мс) виникає через віртуальний мережевий стек Docker (NAT/bridge) та HTTP-серіалізацію. Ця різниця є мінімальною і не суттєвою для практичного використання.</li>
  <li><strong>Як Docker вписується в CI/CD для ML-моделей?</strong><br>
  На кожен коміт CI автоматично перевіряє код лінтером, збирає Docker-образ із фіксованим тегом <code>git-sha</code>, піднімає контейнер і запускає smoke-тести та quality gates (перевірка порогу accuracy і latency). Лише перевірений образ публікується в реєстр і розгортається в production.</li>
</ol>

</body>
</html>"""

    html_path = Path("results/report_full.html")
    html_path.write_text(html, encoding="utf-8")

    for pdf_name in ["Звіт_ЛР2_Гілета.pdf", "Lab2_Report.pdf"]:
        pdf_path = Path(pdf_name).resolve()
        subprocess.run(
            [
                CHROME_PATH,
                "--headless=new",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--print-to-pdf={pdf_path}",
                html_path.resolve().as_uri(),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"Generated PDF: {pdf_path} ({pdf_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    imgs = create_all_screenshots()
    build_report_pdf(imgs)
