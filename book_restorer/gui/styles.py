"""Custom CSS injected into the Streamlit app for a polished, "studio"-style
dark UI: gradient header, glassy cards, and refined typography.
"""

CUSTOM_CSS = """
<style>
:root {
  --accent-1: #7c3aed;
  --accent-2: #db2777;
  --accent-3: #22d3ee;
  --panel-bg: rgba(255,255,255,0.03);
  --panel-border: rgba(255,255,255,0.08);
}

.stApp {
  background: radial-gradient(1200px 600px at 10% -10%, #1e1b4b 0%, #0b0b16 55%, #05050a 100%);
}

section[data-testid="stSidebar"] {
  background: #0d0d18;
  border-right: 1px solid var(--panel-border);
}

.hero-banner {
  background: linear-gradient(120deg, #4c1d95 0%, #7c3aed 45%, #db2777 100%);
  padding: 2.4rem 2.4rem;
  border-radius: 20px;
  margin-bottom: 1.6rem;
  box-shadow: 0 20px 60px rgba(124, 58, 237, 0.35);
}
.hero-banner h1 {
  color: white;
  font-size: 2.3rem;
  margin: 0 0 0.4rem 0;
  font-weight: 800;
  letter-spacing: -0.02em;
}
.hero-banner p {
  color: rgba(255,255,255,0.88);
  font-size: 1.05rem;
  margin: 0;
  max-width: 800px;
}

.metric-card {
  background: var(--panel-bg);
  border: 1px solid var(--panel-border);
  border-radius: 16px;
  padding: 1.1rem 1.2rem;
  text-align: center;
  backdrop-filter: blur(6px);
}
.metric-card .value {
  font-size: 1.7rem;
  font-weight: 800;
  background: linear-gradient(90deg, var(--accent-3), var(--accent-1));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.metric-card .label {
  font-size: 0.82rem;
  color: rgba(255,255,255,0.65);
  margin-top: 0.2rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.section-card {
  background: var(--panel-bg);
  border: 1px solid var(--panel-border);
  border-radius: 18px;
  padding: 1.3rem 1.5rem;
  margin-bottom: 1.1rem;
}

.excerpt-box {
  background: rgba(124,58,237,0.08);
  border-left: 3px solid var(--accent-1);
  padding: 0.7rem 1rem;
  border-radius: 8px;
  font-style: italic;
  margin: 0.4rem 0;
}

.term-pill {
  display:inline-block;
  background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
  color: white;
  padding: 2px 10px;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 600;
  margin-right: 6px;
}

.stTabs [data-baseweb="tab-list"] {
  gap: 6px;
}
.stTabs [data-baseweb="tab"] {
  background: rgba(255,255,255,0.04);
  border-radius: 10px 10px 0 0;
  padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(90deg, var(--accent-1), var(--accent-2)) !important;
  color: white !important;
}

.stButton>button {
  background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
  color: white;
  border: none;
  border-radius: 10px;
  font-weight: 700;
  padding: 0.6rem 1.2rem;
}
.stButton>button:hover {
  filter: brightness(1.1);
  color: white;
}
</style>
"""


def metric_card_html(value: str, label: str) -> str:
    return f'<div class="metric-card"><div class="value">{value}</div><div class="label">{label}</div></div>'
