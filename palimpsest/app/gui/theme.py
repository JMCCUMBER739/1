"""Shared Streamlit CSS / branding."""

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Source+Sans+3:wght@400;500;600;700&display=swap');

:root {
  --ink: #1A1612;
  --paper: #F7F1E8;
  --accent: #C45C26;
  --slate: #2C3E50;
  --sage: #4A6B5C;
  --gold: #B08946;
}

html, body, [class*="css"] {
  font-family: 'Source Sans 3', sans-serif;
  color: var(--ink);
}

.stApp {
  background:
    radial-gradient(ellipse at 12% 8%, rgba(196,92,38,0.14), transparent 42%),
    radial-gradient(ellipse at 88% 0%, rgba(74,107,92,0.16), transparent 40%),
    linear-gradient(165deg, #E8DFD0 0%, #F7F1E8 38%, #EAF0EB 100%);
}

h1, h2, h3, .brand-title {
  font-family: 'Cormorant Garamond', Georgia, serif !important;
  letter-spacing: 0.01em;
}

.hero {
  padding: 1.4rem 1.6rem 1.2rem;
  margin-bottom: 1.2rem;
  background: linear-gradient(120deg, #1A1612 0%, #2C3E50 62%, #4A6B5C 130%);
  color: #F7F1E8;
  border-radius: 0;
  position: relative;
  overflow: hidden;
  animation: fadeRise 0.8s ease-out;
}

.hero::after {
  content: '';
  position: absolute;
  inset: auto -10% -40% 40%;
  height: 180%;
  background: radial-gradient(circle, rgba(196,92,38,0.35), transparent 55%);
  pointer-events: none;
}

.hero .eyebrow {
  text-transform: uppercase;
  letter-spacing: 0.32em;
  font-size: 0.72rem;
  color: #E8C4A8;
  margin-bottom: 0.35rem;
}

.hero h1 {
  font-size: 2.8rem;
  margin: 0;
  line-height: 1.05;
  color: #F7F1E8 !important;
}

.hero p {
  margin: 0.55rem 0 0;
  max-width: 42rem;
  opacity: 0.9;
}

.metric-card {
  background: rgba(255,255,255,0.55);
  border-left: 3px solid var(--accent);
  padding: 0.9rem 1rem;
  animation: fadeRise 0.7s ease-out;
}

.metric-card .label {
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--slate);
}

.metric-card .value {
  font-family: 'Cormorant Garamond', serif;
  font-size: 1.9rem;
  font-weight: 600;
  line-height: 1.1;
}

div[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #1F1A16, #2A333C);
}

div[data-testid="stSidebar"] * {
  color: #F2EBE1 !important;
}

.stButton > button {
  background: var(--accent);
  color: white;
  border: none;
  border-radius: 2px;
  font-weight: 600;
  letter-spacing: 0.04em;
  transition: transform 0.18s ease, box-shadow 0.18s ease;
}

.stButton > button:hover {
  transform: translateY(-1px);
  box-shadow: 0 8px 20px rgba(196,92,38,0.28);
}

@keyframes fadeRise {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

.ref-excerpt {
  background: rgba(255,255,255,0.5);
  border-left: 3px solid var(--sage);
  padding: 0.75rem 1rem;
  margin-bottom: 0.6rem;
  animation: fadeRise 0.55s ease-out;
}

.section-rule {
  height: 1px;
  background: linear-gradient(90deg, var(--accent), transparent);
  margin: 0.4rem 0 1rem;
}
"""
