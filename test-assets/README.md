# test-assets/

The automated test battery (`tests/test_battery.py`) runs Gemma 4 against
every image it finds in this folder. To get a useful report and a strong set
of blog screenshots, drop **3–5 images that showcase variety**.

## Recommended set (one of each)

1. **UI screenshot** — a screenshot of any app or website with multiple
   clickable elements. _Used to demo bounding-box detection on real UI._
   Suggested: a Stripe dashboard, a GitHub PR page, your own dev.to drafts.

2. **Dense diagram or chart** — an architecture diagram, a flowchart, or a
   data visualisation with labels. _Tests how well Gemma 4 handles structured
   technical visuals._

3. **Handwritten or whiteboard photo** — a phone snap of a notebook page,
   sticky notes, or a whiteboard. _Tests handwriting + low-fidelity input._

4. **Portrait-aspect phone screenshot** — anything tall (Twitter thread,
   mobile app, Notion page on phone). _Tests Gemma 4's variable aspect ratio
   handling._

5. _(Optional)_ **PDF page screenshot or dense documentation** — a page from
   a research paper, API docs, or a Stripe-style spec. _Tests long-form
   document understanding._

## Naming conventions

- Use descriptive filenames (`stripe-dashboard.png`, `whiteboard-sketch.jpg`).
  They'll appear in the report tables and screenshot checklist.
- Supported formats: `.png`, `.jpg`, `.jpeg`, `.webp`.
- Keep originals < ~5 MB. The client auto-resizes anything > 1.5 MB before
  sending to the API.

## Privacy reminder

These files are sent to the Google AI Studio API. Don't drop in anything
sensitive (real user data, secrets, internal-only screenshots) you wouldn't
publish on dev.to.

This README is the only file in this folder that's tracked by git — the
images themselves are gitignored.
