# Palette Journal - UX & Accessibility Learnings

## 2025-05-18 - Tkinter Canvas Mousewheel & Video Empty State UX
**Learning:** Tkinter `Canvas` with scrollable frames requires explicit `<MouseWheel>`, `<Button-4>`, and `<Button-5>` bindings for intuitive trackpad/wheel scrolling across OS environments. Also, raw black video labels when disconnected lack orientation context for users; providing clear centered text in empty states guides users directly to action buttons.
**Action:** Always bind mousewheel events to Tkinter scrollable canvases and set clear actionable empty state text on inactive display areas.
