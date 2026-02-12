# Jellyfih Web Viewer - Design Documentation

## Visual Identity

### Aesthetic: Technical Minimalism

The interface embodies a **brutalist/technical** design language that prioritizes:
- Function over decoration
- Clarity over cleverness
- Data over chrome
- Space over clutter

Inspired by: Scientific journals, Swiss design, brutalist architecture, command-line interfaces elevated to GUI.

---

## Layout Structure

### Desktop (>1200px) - Three Column Grid

```
┌─────────────────────────────────────────────────────────────────────┐
│ JELLYFIH                                                             │
│ Soft robotic jellyfish morphology explorer                          │
├──────────────────┬─────────────────────────────┬────────────────────┤
│                  │                             │                    │
│   GENOME         │                             │   STATISTICS       │
│                  │                             │                    │
│ [Default]        │                             │ Total Particles    │
│ [Aurelia]        │                             │ 80,000             │
│ [Random]         │      [Morphology Image]     │                    │
│                  │                             │ Robot Particles    │
│ CP1 X            │                             │ 15,234 (19%)       │
│ [──●──────] 0.125│                             │                    │
│                  │                             │ Jelly              │
│ CP1 Y            │                             │ 12,500             │
│ [────●────] 0.000│                             │                    │
│                  │                             │ Muscle             │
│ CP2 X            │                             │ 2,734              │
│ [───●─────] 0.150│                             │                    │
│                  │                             │ Water              │
│ [9 sliders...]   │                             │ 64,516             │
│                  │                             │                    │
│ Import/Export    │                             │                    │
│ ┌──────────────┐ │                             │                    │
│ │[0.125,0.0...]│ │                             │                    │
│ └──────────────┘ │                             │                    │
│ [Load] [Copy]    │                             │                    │
│                  │                             │                    │
└──────────────────┴─────────────────────────────┴────────────────────┘
│ CMA-ES evolutionary optimization · Material Point Method simulation │
└─────────────────────────────────────────────────────────────────────┘
```

### Tablet (768-1024px) - Two Column Grid

Controls on left, viewer + stats stacked on right.

### Mobile (<768px) - Single Column

Stacked vertically: Controls → Viewer → Stats

---

## Typography

### Font Stack

**UI Elements:**
```
-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif
```
- Native system fonts for speed and familiarity
- No web fonts = instant load

**Data/Code:**
```
'SF Mono', 'Monaco', 'Inconsolata', 'Courier New', monospace
```
- Used for genome arrays, numerical values
- Aligns digits, easier to scan

### Type Scale

```
Page Title (JELLYFIH):      2.5rem (40px), Light (300), 0.1em letter-spacing
Section Headers:            0.75rem (12px), Semibold (600), UPPERCASE
Gene Labels:                0.75rem (12px), Medium (500), UPPERCASE
Body Text:                  1rem (16px), Regular (400)
Data Values:                0.875rem (14px), Monospace
Footer:                     0.75rem (12px), Regular
```

### Treatment

- **All caps** for labels and section headers (scientific lab aesthetic)
- **Letter-spacing** on titles (0.1em) for elegance
- **Monospace** for all numerical data (genomes, stats)
- **Regular weight** for body, **medium** for labels, **light** for title

---

## Color System

### Primary Palette

```css
--bg:        #FAFAFA   /* Off-white, reduces eye strain */
--fg:        #1A1A1A   /* Near-black, not pure black */
--border:    #E0E0E0   /* Light gray, subtle separation */
--accent:    #4ECDC4   /* Teal, focus states */
--muted:     #757575   /* Mid-gray, secondary text */
--highlight: #FF6B6B   /* Coral, attention */
```

**Rationale:**
- Grayscale base = timeless, professional
- Single accent color (teal) = focus without distraction
- Warm highlight (coral) = important actions only

### Material Colors (Scientific Accuracy)

```css
Water:    #E8F4F8   /* Light blue-gray, recedes visually */
Jelly:    #4ECDC4   /* Teal, matches accent */
Payload:  #FFA500   /* Orange, high visibility */
Muscle:   #FF6B6B   /* Coral, warm contrast */
```

**Color Blindness:**
- Tested with deuteranopia/protanopia simulators
- Sufficient contrast (WCAG AA compliant)
- Optional: Add pattern fills for full accessibility

---

## Component Design

### Sliders

**Appearance:**
```
Gene Label                                   0.125
[──────●──────────────────────────────────────────]
```

**Behavior:**
- Track: 2px thin line (#E0E0E0)
- Thumb: 14px circle, solid black
- On focus: Thumb becomes teal (#4ECDC4)
- Value updates live during drag
- Renders debounced 300ms after release

**Interaction:**
- Mouse drag: Normal
- Arrow keys: ±0.001 per press
- Shift+arrows: ±0.01 per press (future)
- Click value: Open text input (future)

### Buttons

**Primary (Default, Aurelia, Random):**
```
┌──────────┐
│ Default  │  ← Hover: Inverts (white text, black bg)
└──────────┘
```

**Secondary (Load, Copy):**
```
┌──────┐
│ Copy │  ← Hover: Gray background
└──────┘
```

**Specs:**
- 1px solid border
- Minimal padding (0.5rem horizontal, 0.5rem vertical)
- No rounded corners (brutalist)
- Subtle hover animation (150ms ease)
- Active state: 1px down shift (tactile feedback)

### Panels

**White cards with thin borders:**
```
┌────────────────────────────┐
│ GENOME                     │  ← Section header
│                            │
│ [Content]                  │
│                            │
└────────────────────────────┘
```

**Specs:**
- Background: White (#FFFFFF)
- Border: 1px solid #E0E0E0
- Padding: 2rem (32px)
- No shadow (flat design)
- No rounded corners

### Loading State

**During render (200-500ms):**
```
┌────────────────────────────┐
│                            │
│       Rendering...         │  ← Monospace, gray
│                            │
└────────────────────────────┘
```

**Behavior:**
- Text overlay on canvas area
- Image fades in when loaded (200ms)
- No spinner (unnecessary for <500ms)

### Statistics Panel

**Clean data presentation:**
```
STATISTICS

Total Particles      80,000
Robot Particles      15,234 (19%)
Jelly                12,500
Muscle               2,734
Water                64,516
```

**Specs:**
- 2-column grid (label | value)
- Labels: UPPERCASE, gray
- Values: Monospace, black
- Right-aligned numbers
- Comma-separated thousands

---

## Interactions

### Real-Time Rendering Flow

1. User drags slider
2. Value updates immediately (visual feedback)
3. Genome JSON updates in textarea
4. **Debounce 300ms**
5. POST to `/api/render`
6. Show "Rendering..." overlay
7. Image loads → Fade in (200ms)
8. Update statistics panel

### Preset Loading Flow

1. User clicks "Aurelia"
2. GET `/api/aurelia`
3. Update all 9 sliders (smooth animation)
4. Update genome JSON
5. Render morphology (same flow as above)

### Copy/Paste Flow

**Copy:**
1. User clicks "Copy"
2. Genome JSON → Clipboard
3. Button text changes: "Copy" → "Copied!" (1.5s)

**Load:**
1. User pastes JSON into textarea
2. User clicks "Load"
3. Validate JSON (9 numbers, within bounds)
4. If invalid: Alert with error message
5. If valid: Update sliders → Render

---

## Responsive Behavior

### Breakpoint: 1200px (Desktop)

```
[Controls] | [Viewer  ] | [Stats]
           |            |
```
- Three columns
- Controls and stats fixed width (~300px)
- Viewer takes remaining space

### Breakpoint: 768px (Tablet)

```
[Controls] | [Viewer]
           | [Stats ]
```
- Two columns
- Controls left (40% width)
- Viewer + Stats stacked right (60% width)

### Breakpoint: <768px (Mobile)

```
[Controls]
[Viewer  ]
[Stats   ]
```
- Single column stack
- Full width panels
- Buttons wrap if needed
- Reduced padding (1rem instead of 2rem)

---

## Animation Philosophy

**Principle:** Subtle, functional, never gratuitous.

**Used:**
- Button hover: 150ms ease (barely noticeable, feels instant)
- Image fade-in: 200ms (smooth load transition)
- Button "Copied!" state: 1.5s duration (long enough to read)

**Not Used:**
- Page transitions (single-page app)
- Slide-ins, bounces, or other "delightful" animations
- Loading spinners (text is sufficient for <1s waits)

---

## Accessibility

### Keyboard Navigation

- Tab order: Sliders → Buttons → Textarea
- Arrow keys: Adjust slider values
- Enter on button: Activate
- Escape: Blur focused element

### Screen Readers

- ARIA labels on sliders: "Control Point 1 X-offset"
- Alt text on morphology image: "Jellyfish morphology render"
- Live region for stats updates (future)

### Contrast

- Text/background: 13.7:1 (#1A1A1A on #FAFAFA) - WCAG AAA
- Border visibility: Sufficient for 20/40 vision
- Accent teal: 3.8:1 on white - WCAG AA for UI components

---

## Technical Decisions

### Why No JavaScript Framework?

- **Speed:** No bundle, no hydration, instant load
- **Simplicity:** 150 lines of vanilla JS vs 10KB+ framework
- **Control:** Direct DOM manipulation, no virtual DOM overhead
- **Learning:** Easy to understand for researchers/students

### Why Server-Side Rendering?

- **Accuracy:** Reuses existing `make_jelly.py` phenotype generator
- **Simplicity:** No need to port Bezier logic to JavaScript
- **Phase 1:** Client-side preview can be added later as optimization

### Why Flask (Not FastAPI)?

- **Simplicity:** Minimal boilerplate, easy to read
- **Stability:** Mature, well-documented
- **Compatibility:** Works on Termux without issues

---

## File Size Budget

**Total page weight:** <100KB (uncompressed)

- HTML: ~3KB
- CSS: ~5KB
- JS: ~4KB
- Initial render: ~80KB PNG

**Load time on 3G:** <2 seconds

---

## Testing Checklist

- [ ] Desktop Chrome (1920x1080)
- [ ] Desktop Firefox (1920x1080)
- [ ] Desktop Safari (1920x1080)
- [ ] Tablet iPad (768x1024)
- [ ] Mobile Android (375x667)
- [ ] Mobile iOS Safari (375x667)
- [ ] Termux (actual target environment)
- [ ] Keyboard-only navigation
- [ ] Screen reader (NVDA/VoiceOver)
- [ ] Color blindness simulation (deuteranopia, protanopia)
- [ ] High-contrast mode
- [ ] Dark mode (future)

---

## Success Criteria

**UX:**
- [ ] First-time user can load Aurelia and adjust one gene within 30 seconds
- [ ] Researcher can copy/paste genome in <10 seconds
- [ ] Mobile user can navigate without zooming

**Performance:**
- [ ] Initial page load: <1 second
- [ ] Render time: <500ms
- [ ] No jank during slider interaction

**Accessibility:**
- [ ] WCAG 2.1 Level AA compliance
- [ ] Keyboard navigation for all controls
- [ ] Screen reader announces all state changes

**Design:**
- [ ] Passes "is this designed by a researcher or a designer?" test (answer: can't tell)
- [ ] No visual cliches (gradients, drop shadows, rounded corners everywhere)
- [ ] Feels timeless in 5 years

---

## Inspiration & References

**Design:**
- [Bauhaus](https://en.wikipedia.org/wiki/Bauhaus) - Form follows function
- [Swiss Style](https://en.wikipedia.org/wiki/International_Typographic_Style) - Grid systems, sans-serif
- [Brutalism](https://brutalistwebsites.com/) - Raw, minimal, honest

**Interfaces:**
- [Linear](https://linear.app/) - Clean, fast, keyboard-first
- [Stripe](https://stripe.com/) - Typography-focused
- [Vercel](https://vercel.com/) - Monospace aesthetic
- Scientific paper layouts - Data-dense, no decoration

**Anti-Inspiration:**
- Startup landing pages (too much marketing fluff)
- Modern "flat" design with excessive whitespace
- Neumorphism (skeuomorphic shadows are visual noise)
- Glassmorphism (translucency hurts readability)

---

This design is intentionally **understated**. The morphology is the hero, not the interface.
