import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        border: "var(--border)", // <-- Changed!
        input: "var(--input)", // <-- Changed!
        ring: "var(--ring)", // <-- Changed!
        background: "var(--background)", // <-- Changed!
        foreground: "var(--foreground)", // <-- Changed!
        primary: {
          DEFAULT: "var(--primary)", // <-- Changed!
          foreground: "var(--primary-foreground)", // <-- Changed!
        },
        secondary: {
          DEFAULT: "var(--secondary)", // <-- Changed!
          foreground: "var(--secondary-foreground)", // <-- Changed!
        },
        destructive: {
          DEFAULT: "var(--destructive)", // <-- Changed!
          foreground: "var(--destructive-foreground)", // <-- Changed!
        },
        muted: {
          DEFAULT: "var(--muted)", // <-- Changed!
          foreground: "var(--muted-foreground)", // <-- Changed!
        },
        accent: {
          DEFAULT: "var(--accent)", // <-- Changed!
          foreground: "var(--accent-foreground)", // <-- Changed!
        },
        popover: {
          DEFAULT: "var(--popover)", // <-- Changed!
          foreground: "var(--popover-foreground)", // <-- Changed!
        },
        card: {
          DEFAULT: "var(--card)", // <-- Changed!
          foreground: "var(--card-foreground)", // <-- Changed!
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
    },
  },
  plugins: [],
};

export default config;
