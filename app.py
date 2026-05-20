"""Lens — a Gemma 4 multimodal writing companion.

Run: streamlit run app.py
"""
from __future__ import annotations

import io
import os

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

from gemma_client import DEFAULT_MODEL, ALT_MODEL, generate  # noqa: E402
from prompts import DETECTION_PROMPT, INTENT_PROMPTS, build_prompt  # noqa: E402
from render import draw_boxes, parse_detections  # noqa: E402

st.set_page_config(page_title="Lens — Gemma 4", page_icon="🔍", layout="wide")

st.title("🔍 Lens")
st.caption(
    "A multimodal writing companion powered by Gemma 4 via Google AI Studio. "
    "Upload an image, pick an intent, and let Gemma 4 draft from it."
)

# ----- API key check -----
key_present = bool(os.getenv("GEMINI_API_KEY", "").strip()) and \
    os.getenv("GEMINI_API_KEY", "").strip() != "your-key-here"
if not key_present:
    st.error(
        "**GEMINI_API_KEY is not set.** Copy `.env.example` to `.env` and add your "
        "key from https://aistudio.google.com/apikey, then restart this app."
    )
    st.stop()

# ----- Sidebar controls -----
with st.sidebar:
    st.header("Settings")
    model = st.selectbox(
        "Model",
        options=[DEFAULT_MODEL, ALT_MODEL],
        index=0,
        help="MoE (26b-a4b) is faster and easier on rate limits. "
             "31b is the flagship dense model.",
    )
    intent = st.selectbox(
        "Intent",
        options=list(INTENT_PROMPTS.keys()),
        index=0,
        format_func=lambda s: s.replace("-", " ").title(),
    )
    thinking = st.toggle(
        "Show thinking",
        value=False,
        help="Surface the model's reasoning trace alongside the answer.",
    )
    detect = st.toggle(
        "Detect elements",
        value=False,
        help="Ask Gemma 4 to return bounding boxes for prominent elements, "
             "then overlay them on the image.",
    )
    st.divider()
    st.caption(
        "Built for the dev.to Gemma 4 Challenge. "
        "Default model: `gemma-4-26b-a4b-it`."
    )

# ----- Image upload -----
upload = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=False,
)

if not upload:
    st.info("👆 Upload a screenshot, diagram, or photo to begin.")
    st.stop()

image_bytes = upload.getvalue()
mime = upload.type or "image/png"

if not image_bytes:
    st.error("Uploaded file is empty. Pick another image.")
    st.stop()

preview_col, _ = st.columns([2, 1])
with preview_col:
    st.image(image_bytes, caption=f"Preview · {upload.name}", use_container_width=True)

if not st.button("Generate", type="primary", use_container_width=True):
    st.stop()

# ----- Generation -----
results_box = st.container()

with st.spinner(f"Calling {model}..."):
    main_prompt = build_prompt(intent)
    main_result = generate(image_bytes, mime, main_prompt, model=model, thinking=thinking)

detection_result = None
detections = None
overlay_image: Image.Image | None = None
if detect:
    with st.spinner("Detecting elements..."):
        detection_result = generate(image_bytes, mime, DETECTION_PROMPT, model=model, thinking=False)
    if detection_result.ok:
        detections = parse_detections(detection_result.text)
        if detections:
            try:
                overlay_image = draw_boxes(image_bytes, detections)
            except Exception as exc:  # pragma: no cover
                st.warning(f"Could not render boxes: {exc}")

# ----- Render results -----
with results_box:
    st.divider()
    if not main_result.ok:
        if main_result.error_kind == "rate_limit":
            st.error(
                "**Rate limit reached.** Wait a minute and click Generate again. "
                "Check your active limits at https://aistudio.google.com/rate-limit."
            )
        elif main_result.error_kind == "auth":
            st.error("**Auth error.** Double-check your `GEMINI_API_KEY` in `.env`.")
        elif main_result.error_kind == "network":
            st.error("**Network error.** Check your connection and retry.")
        else:
            st.error(f"**Something went wrong.** {main_result.error}")
        st.stop()

    left, right = st.columns([3, 2])

    with left:
        st.subheader(f"Result · {intent.replace('-', ' ').title()}")
        st.markdown(main_result.text or "_(empty response)_")

        if main_result.thoughts:
            with st.expander("🧠 Thoughts", expanded=False):
                st.markdown(
                    f"_via {main_result.thinking_mode}-mode_\n\n{main_result.thoughts}"
                )
        elif thinking:
            st.caption("_Thinking was requested but no thoughts were returned._")

    with right:
        st.subheader("Run info")
        st.metric("Latency", f"{main_result.latency_ms} ms")
        if main_result.input_tokens is not None:
            st.metric("Input tokens", main_result.input_tokens)
        if main_result.output_tokens is not None:
            st.metric("Output tokens", main_result.output_tokens)
        st.caption(f"Model: `{main_result.model}`")
        st.caption(f"Thinking: `{main_result.thinking_mode}`")

    if detect:
        st.divider()
        st.subheader("🎯 Detected elements")
        if not detection_result or not detection_result.ok:
            err_msg = detection_result.error if detection_result else "unknown"
            st.warning(f"Detection call failed: {err_msg}")
        elif detections is None:
            st.warning(
                "Model response was not parseable as bounding-box JSON. "
                "Raw output below."
            )
            st.code(detection_result.text or "(empty)", language="text")
        else:
            overlay_col, list_col = st.columns([2, 1])
            with overlay_col:
                if overlay_image is not None:
                    buf = io.BytesIO()
                    overlay_image.save(buf, format="PNG")
                    st.image(buf.getvalue(), caption="Overlay", use_container_width=True)
            with list_col:
                st.markdown("**Labels**")
                for d in detections:
                    st.markdown(f"- {d['label']}")
