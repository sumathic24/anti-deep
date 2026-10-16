/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Surfaces (dark theme)
        canvas: '#070b14',   // page background
        surface: '#0d1320',  // cards
        raised: '#121b2d',   // inputs, nested panels, hover
        line: {
          DEFAULT: '#1c2536', // hairline borders
          strong: '#2a3549',
        },
        // Text
        fg: {
          DEFAULT: '#e6eaf2',
          2: '#a3acbf',
          3: '#7c869a',
        },
        // Brand accent for interactive UI (buttons, links, focus)
        accent: {
          DEFAULT: '#22d3ee',
          hover: '#67e8f9',
          ink: '#04111d',
        },
        // Verdict colors — validated CVD-safe pair on the surface color
        ai: { DEFAULT: '#e66767', text: '#fca5a5' },
        human: { DEFAULT: '#3987e5', text: '#93c5fd' },
      },
      opacity: {
        3: '0.03',
        8: '0.08',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'Segoe UI', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      },
      maxWidth: {
        '8xl': '88rem',
      },
    },
  },
  plugins: [],
}
