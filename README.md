# Lens — a Gemma 4 multimodal writing companion

A small Streamlit app that turns an image into structured technical writing
using **Gemma 4** via the **Google AI Studio API**. Built for the dev.to
Gemma 4 Challenge.

Two visible Gemma 4 capabilities are wired in:

- 🧠 **Thinking mode** — surfaces the model's reasoning trace alongside the answer.
- 🎯 **Element detection** — Gemma 4 returns bounding boxes natively; Lens overlays them on the uploaded image.

## Requirements

- MacOS (or any OS with Python 3.10+)
- A Google AI Studio API key — get one at <https://aistudio.google.com/apikey>

## Quickstart

```bash
cd gemma4-lens

# 1. Virtual env + install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Add your API key
cp .env.example .env
# edit .env and paste your key for GEMINI_API_KEY

# 3. Run the app
streamlit run app.py
```

Streamlit opens in your browser. Upload an image, pick an intent
(`documentation`, `blog-outline`, `alt-text`), toggle thinking or detection
if you want, and hit **Generate**.

## Run the automated test battery

```bash
# 1. Drop 3-5 images into test-assets/  (see test-assets/README.md)
# 2. Run the matrix
python3 tests/test_battery.py
```

This runs every image through every intent (with thinking on/off) plus a
detection pass, and writes:

- `reports/raw-<timestamp>.json` — full structured results
- `reports/report-<timestamp>.md` — human-readable summary + **screenshot checklist** telling you exactly what to capture in the browser for the blog post

Tune `--sleep` (seconds between calls) if you hit rate limits, or `--limit N`
to cap the number of images.

## Models

| Model ID | Type | Default? |
|---|---|---|
| `gemma-4-26b-a4b-it` | MoE — fast, lighter on tokens | ✅ default |
| `gemma-4-31b-it` | Dense flagship — slower, stronger reasoning | toggle in sidebar |

The edge variants (E2B / E4B) are not exposed via the Gemini API and so are
out of scope for this build.

## File layout

```
gemma4-lens/
├── app.py              # Streamlit UI
├── gemma_client.py     # google-genai wrapper, thinking + error handling
├── prompts.py          # Intent prompt templates + detection prompt
├── render.py           # Image resize, JSON parse, box drawing, thought split
├── tests/
│   └── test_battery.py # Automated matrix runner + report writer
├── test-assets/        # Drop your 3-5 test images here (gitignored)
├── reports/            # Generated reports (gitignored)
├── requirements.txt
├── .env.example
└── README.md
```

## Notes for the blog post

The build deliberately leans on two unique Gemma 4 capabilities that aren't
present in Gemma 3:

- Native structured JSON for object detection (no grammar-constrained generation)
- A `thinking` mode that exposes reasoning traces (with a graceful prompt-based fallback if the SDK doesn't expose `thinking_config` directly — `gemma_client.py` probes on first call and adapts)

Both are visible in the UI and captured in the test report — they're the
strongest evidence you used Gemma 4 _intentionally_ rather than dropping a
prompt into any LLM.
