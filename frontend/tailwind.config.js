/** @type {import('tailwindcss').Config} */

// Colors are CSS variables (see src/index.css) so the same classes work in light and dark mode.
const v = (name) => `rgb(var(--${name}) / <alpha-value>)`

export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Surfaces
        canvas: v('canvas'),   // page background
        surface: v('surface'), // cards
        raised: v('raised'),   // inputs, nested panels, hover
        line: {
          DEFAULT: v('line'),  // hairline borders
          strong: v('line-strong'),
        },
        // Text
        fg: {
          DEFAULT: v('fg'),
          2: v('fg-2'),
          3: v('fg-3'),
        },
        // Brand accent for interactive UI (buttons, links, focus)
        accent: {
          DEFAULT: v('accent'),
          hover: v('accent-hover'),
          ink: v('accent-ink'),
        },
        // Verdict colors — validated CVD-safe pairs for each theme's surface
        ai: { DEFAULT: v('ai'), text: v('ai-text') },
        human: { DEFAULT: v('human'), text: v('human-text') },
        danger: { DEFAULT: v('danger'), ink: v('danger-ink') },
      },
      opacity: {
        3: '0.03',
        8: '0.08',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'Segoe UI', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      },
    },
  },
  plugins: [],
}
