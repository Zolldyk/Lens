"""Automated test battery for Lens.

Runs a fixed matrix of (images × intents × thinking on/off) plus a detection
pass per image. Captures raw outputs, latencies, token usage, and failures.
Emits a markdown report with a screenshot checklist that tells Zoll exactly
what to capture by hand for the blog post.

Usage:
  cd gemma4-lens
  python3 tests/test_battery.py
  python3 tests/test_battery.py --sleep 2 --model gemma-4-31b-it
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Allow running as `python tests/test_battery.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from gemma_client import DEFAULT_MODEL, SUPPORTED_MODELS, generate  # noqa: E402
from prompts import DETECTION_PROMPT, INTENT_PROMPTS, build_prompt  # noqa: E402
from render import parse_detections  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT / "test-assets"
REPORTS_DIR = ROOT / "reports"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
INTENTS = list(INTENT_PROMPTS.keys())


def discover_images() -> list[Path]:
    if not ASSETS_DIR.exists():
        return []
    return sorted(
        p for p in ASSETS_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_SUFFIXES
        and not p.name.startswith((".", "_"))  # skip macOS ._* and .DS_Store
    )


def mime_for(path: Path) -> str:
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }[path.suffix.lower()]


def run_matrix(images: list[Path], model: str, sleep_sec: float) -> dict:
    cases: list[dict] = []
    thinking_states = (False, True)
    total = len(images) * (len(INTENTS) * len(thinking_states) + 1)
    counter = 0
    interrupted = False
    started = time.time()

    try:
        for image_path in images:
            try:
                image_bytes = image_path.read_bytes()
            except OSError as err:
                print(f"  skip {image_path.name}: {err}", file=sys.stderr)
                continue
            mime = mime_for(image_path)

            for intent in INTENTS:
                for thinking in thinking_states:
                    counter += 1
                    print(
                        f"[{counter}/{total}] {image_path.name} · {intent} "
                        f"· thinking={thinking}",
                        flush=True,
                    )
                    prompt = build_prompt(intent)
                    result = generate(image_bytes, mime, prompt, model=model, thinking=thinking)
                    cases.append({
                        "case_id": counter,
                        "kind": "intent",
                        "image": image_path.name,
                        "intent": intent,
                        "thinking_requested": thinking,
                        **result.to_dict(),
                    })
                    if counter < total:
                        time.sleep(sleep_sec)

            counter += 1
            print(f"[{counter}/{total}] {image_path.name} · detection", flush=True)
            det_result = generate(image_bytes, mime, DETECTION_PROMPT, model=model, thinking=False)
            det_parsed = parse_detections(det_result.text) if det_result.ok else None
            cases.append({
                "case_id": counter,
                "kind": "detection",
                "image": image_path.name,
                "intent": "detection",
                "thinking_requested": False,
                "detections_parsed_count": len(det_parsed) if det_parsed else 0,
                "detections_parsed_ok": det_parsed is not None,
                **det_result.to_dict(),
            })
            if counter < total:
                time.sleep(sleep_sec)
    except KeyboardInterrupt:
        interrupted = True
        print("\n  interrupted — saving partial results", file=sys.stderr)

    return {
        "model": model,
        "started_at": datetime.fromtimestamp(started).isoformat(timespec="seconds"),
        "duration_sec": round(time.time() - started, 1),
        "total_cases": len(cases),
        "interrupted": interrupted,
        "cases": cases,
    }


def summarize(run: dict) -> dict:
    cases = run["cases"]
    ok = [c for c in cases if c.get("ok")]
    failed = [c for c in cases if not c.get("ok")]
    by_error: dict[str, int] = {}
    for c in failed:
        kind = c.get("error_kind") or "other"
        by_error[kind] = by_error.get(kind, 0) + 1
    latencies = [c["latency_ms"] for c in ok if c.get("latency_ms")]
    avg_latency = round(sum(latencies) / len(latencies)) if latencies else 0
    longest = max(ok, key=lambda c: len(c.get("text") or ""), default=None)
    detection_cases = [c for c in cases if c.get("kind") == "detection"]
    best_detection = max(
        (c for c in detection_cases if c.get("detections_parsed_ok")),
        key=lambda c: c["detections_parsed_count"],
        default=None,
    )
    thinking_cases = [c for c in ok if c.get("thinking_requested") and c.get("thoughts")]
    return {
        "ok_count": len(ok),
        "failed_count": len(failed),
        "by_error": by_error,
        "avg_latency_ms": avg_latency,
        "longest_output_case": longest["case_id"] if longest else None,
        "best_detection_case": best_detection["case_id"] if best_detection else None,
        "thinking_returned_count": len(thinking_cases),
        "first_thinking_case": thinking_cases[0]["case_id"] if thinking_cases else None,
    }


def write_reports(run: dict) -> tuple[Path, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    raw_path = REPORTS_DIR / f"raw-{stamp}.json"
    md_path = REPORTS_DIR / f"report-{stamp}.md"

    raw_path.write_text(json.dumps(run, indent=2), encoding="utf-8")

    summary = summarize(run)
    md = _render_markdown(run, summary)
    md_path.write_text(md, encoding="utf-8")
    return raw_path, md_path


def _render_markdown(run: dict, summary: dict) -> str:
    lines: list[str] = []
    lines.append(f"# Lens Test Battery Report — {run['started_at']}")
    lines.append("")
    lines.append(f"- **Model:** `{run['model']}`")
    lines.append(f"- **Duration:** {run['duration_sec']} s")
    lines.append(f"- **Cases run:** {run['total_cases']}")
    lines.append(f"- **OK / Failed:** {summary['ok_count']} / {summary['failed_count']}")
    lines.append(f"- **Avg latency:** {summary['avg_latency_ms']} ms")
    lines.append(f"- **Thoughts returned:** {summary['thinking_returned_count']}")
    if summary["by_error"]:
        breakdown = ", ".join(f"{k}: {v}" for k, v in summary["by_error"].items())
        lines.append(f"- **Error breakdown:** {breakdown}")
    lines.append("")

    lines.append("## Case results")
    lines.append("")
    lines.append("| # | Image | Intent | Thinking | OK | Latency | Tokens (in/out) | Notes |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for c in run["cases"]:
        ok_mark = "✅" if c.get("ok") else "❌"
        latency = f"{c['latency_ms']} ms" if c.get("latency_ms") else "-"
        in_tok = c.get("input_tokens") if c.get("input_tokens") is not None else "-"
        out_tok = c.get("output_tokens") if c.get("output_tokens") is not None else "-"
        notes_bits: list[str] = []
        if c.get("kind") == "detection":
            if c.get("detections_parsed_ok"):
                notes_bits.append(f"{c['detections_parsed_count']} boxes parsed")
            else:
                notes_bits.append("JSON parse failed")
        if c.get("thinking_requested"):
            mode = c.get("thinking_mode")
            if c.get("thoughts"):
                notes_bits.append(f"thoughts via {mode}")
            elif c.get("ok"):
                notes_bits.append("thinking requested, none returned")
        if not c.get("ok"):
            notes_bits.append(f"error: {c.get('error_kind')}")
        lines.append(
            f"| {c['case_id']} | {c['image']} | {c['intent']} | "
            f"{c.get('thinking_requested')} | {ok_mark} | {latency} | "
            f"{in_tok}/{out_tok} | {'; '.join(notes_bits) or '-'} |"
        )
    lines.append("")

    # Most interesting outputs
    lines.append("## Highlighted outputs")
    lines.append("")
    highlights: list[tuple[str, dict | None]] = [
        ("Longest output", _case_by_id(run, summary["longest_output_case"])),
        ("Best detection", _case_by_id(run, summary["best_detection_case"])),
        ("First thoughts trace", _case_by_id(run, summary["first_thinking_case"])),
    ]
    for label, case in highlights:
        if not case:
            continue
        lines.append(f"### {label} — case #{case['case_id']}")
        lines.append(
            f"_{case['image']} · {case.get('intent')} · "
            f"thinking_requested={case.get('thinking_requested')}_"
        )
        lines.append("")
        if case.get("thoughts"):
            lines.append("**Thoughts**")
            lines.append("")
            lines.append("```")
            lines.append((case["thoughts"] or "")[:1200])
            lines.append("```")
            lines.append("")
        lines.append("**Answer**")
        lines.append("")
        lines.append("```")
        lines.append((case.get("text") or "")[:1500])
        lines.append("```")
        lines.append("")

    # Screenshot checklist
    lines.append("## 📸 Manual screenshot checklist")
    lines.append("")
    lines.append("Open the Lens app (`streamlit run app.py`) and capture these:")
    lines.append("")
    lines.append("1. **Empty state** — fresh app load, sidebar visible. _For the post intro._")
    lines.append("2. **Upload + preview** — upload a screenshot, before clicking Generate.")
    intent_for_screenshot = INTENTS[0]
    lines.append(
        f"3. **Result + Run info** — pick `{intent_for_screenshot}`, generate, capture both "
        "the result text and the metrics panel on the right."
    )
    if summary["first_thinking_case"]:
        case = _case_by_id(run, summary["first_thinking_case"])
        if case:
            lines.append(
                f"4. **Thoughts panel expanded** — repeat case #{case['case_id']} "
                f"(`{case['image']}`, `{case['intent']}`, thinking ON) and expand "
                "the Thoughts panel."
            )
    if summary["best_detection_case"]:
        case = _case_by_id(run, summary["best_detection_case"])
        if case:
            lines.append(
                f"5. **Detection overlay** — upload `{case['image']}`, toggle Detect "
                "elements ON, generate, capture the boxes overlay and the label list."
            )
    lines.append("6. **Rate-limit error state (optional)** — only if you hit one naturally.")
    lines.append("")
    lines.append("Save screenshots to `_bmad-output/implementation-artifacts/screenshots/`.")

    return "\n".join(lines) + "\n"


def _case_by_id(run: dict, case_id: int | None) -> dict | None:
    if not case_id:
        return None
    for c in run["cases"]:
        if c["case_id"] == case_id:
            return c
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Lens automated test battery")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=SUPPORTED_MODELS)
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="Seconds between API calls (raise if you hit 429s)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Optional: cap number of images processed")
    args = parser.parse_args()

    images = discover_images()
    if not images:
        print(
            f"No images found in {ASSETS_DIR}. "
            "See test-assets/README.md for what to add.",
            file=sys.stderr,
        )
        return 1
    if args.limit:
        images = images[: args.limit]

    print(f"Running matrix over {len(images)} image(s) with model {args.model}...")
    run = run_matrix(images, args.model, args.sleep)
    raw_path, md_path = write_reports(run)
    print()
    print(f"Raw JSON: {raw_path.relative_to(ROOT)}")
    print(f"Report:   {md_path.relative_to(ROOT)}")
    print()
    print("Open the markdown report — the screenshot checklist tells you exactly")
    print("what to capture in the browser.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
