"""Prompt templates for the Lens app.

One template per supported intent, plus a dedicated detection prompt that asks
Gemma 4 to emit bounding boxes as raw JSON.
"""

INTENT_PROMPTS = {
    "documentation": (
        "You are a senior technical writer. Look at the image and produce a "
        "concise documentation section about what it shows.\n\n"
        "Structure your response as:\n"
        "1. **Overview** — one short paragraph describing the subject of the image.\n"
        "2. **Key elements** — a bulleted list of the most important things visible.\n"
        "3. **Notes for readers** — any caveats, prerequisites, or context a reader "
        "would need.\n\n"
        "Use clear, neutral, developer-friendly language."
    ),
    "blog-outline": (
        "You are an experienced dev.to blogger. Look at the image and draft a "
        "blog post outline inspired by it.\n\n"
        "Structure your response as:\n"
        "- **Working title**: one punchy title\n"
        "- **Hook**: 1-2 sentences that would open the post\n"
        "- **Sections**: 4-6 bullet points, each a section heading + one-line summary\n"
        "- **Call to action**: one closing line\n\n"
        "Tone: curious, practical, first-person."
    ),
    "alt-text": (
        "You are an accessibility specialist. Look at the image and produce "
        "high-quality alt text for it.\n\n"
        "Return TWO variants:\n"
        "- **Short** (≤ 125 characters): for inline use\n"
        "- **Long** (2-3 sentences): for figure captions or rich contexts\n\n"
        "Describe what is visible, not what you infer. Avoid 'image of' or 'picture of'."
    ),
}

DETECTION_PROMPT = (
    "You are a UI element detector. Look at the image and return ONLY a JSON "
    "array.\n"
    "Each item must be: "
    '{"box_2d": [y_min, x_min, y_max, x_max], "label": "<short noun phrase>"}.\n'
    "Coordinates are normalised 0-1000 against image dimensions "
    "(0,0 is top-left).\n"
    "Detect up to 10 of the most prominent elements.\n"
    "No prose. No markdown code fences. Just the JSON array."
)

THINKING_FALLBACK_PREFIX = (
    "Before answering, think step-by-step inside a <thinking>...</thinking> "
    "block. Then on a new line, give your final answer. The <thinking> block "
    "must come first and must be self-contained.\n\n"
)


def build_prompt(intent: str, thinking_fallback: bool = False) -> str:
    """Compose the final prompt for an intent, optionally with thinking fallback."""
    base = INTENT_PROMPTS[intent]
    if thinking_fallback:
        return THINKING_FALLBACK_PREFIX + base
    return base
