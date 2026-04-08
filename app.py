"""
app.py — Bracket Visualization
================================
Serves the predicted bracket on a local web page.

Usage:
    python app.py
    open http://localhost:5000
"""

import json
from pathlib import Path

from flask import Flask, render_template_string

app = Flask(__name__)
BRACKET_FILE = Path("bracket.json")

TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>2026 NCAA Bracket — Quantum Prediction</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body { background:#0d1117; color:#e6edf3; font-family:'Segoe UI',sans-serif; }
    h1   { color:#58a6ff; }

    .champion-card {
      background: linear-gradient(135deg, #b8860b, #ffd700);
      color:#000; border-radius:16px; padding:2rem;
      text-align:center; margin-bottom:2rem;
    }
    .champion-name { font-size:2.8rem; font-weight:800; }

    .panel {
      background:#161b22; border:1px solid #30363d;
      border-radius:8px; padding:1rem; height:100%;
    }
    .round-label {
      font-size:0.7rem; font-weight:700; text-transform:uppercase;
      letter-spacing:1px; color:#8b949e; margin:0.75rem 0 0.3rem;
    }
    .region-title { font-size:1rem; font-weight:700; color:#58a6ff;
      border-bottom:1px solid #30363d; padding-bottom:0.4rem; margin-bottom:0.6rem; }

    .matchup { display:flex; align-items:center; font-size:0.82rem; margin-bottom:3px; }
    .vs { color:#484f58; margin:0 5px; }
    .winner { font-weight:700; color:#3fb950; }
    .loser  { color:#484f58; text-decoration:line-through; }

    .badge-team {
      display:inline-block; border-radius:4px;
      padding:2px 8px; margin:2px; font-size:0.8rem;
      background:#21262d; border:1px solid #30363d;
    }
    .badge-r2   { background:#0d2137; border-color:#2d6db5; }
    .badge-s16  { background:#1d1035; border-color:#6e40c9; }
    .badge-e8   { background:#0d2110; border-color:#238636; }
    .badge-f4   { background:#1a2a0d; border-color:#56d364; color:#56d364; font-weight:600; }
    .badge-champ{ background:#b8860b; border-color:#ffd700; color:#000; font-weight:700; }
  </style>
</head>
<body>
<div class="container py-4">

  <h1 class="text-center mb-1">2026 NCAA Tournament</h1>
  <p class="text-center mb-4" style="color:#8b949e;">
    Quantum-Classical Hybrid Prediction &middot; {{ data.generated_at[:10] }}
  </p>

  <!-- Champion -->
  <div class="champion-card">
    <div style="font-size:0.85rem;font-weight:600;text-transform:uppercase;letter-spacing:2px;">
      🏆 Predicted Champion
    </div>
    <div class="champion-name">{{ data.champion }}</div>
  </div>

  <!-- Championship -->
  <div class="panel mb-3">
    <div class="region-title">Championship</div>
    {% set cm = data.championship %}
    <div class="matchup" style="font-size:1rem;">
      <span class="{{ 'winner' if cm.teams[0] == cm.winner else 'loser' }}">{{ cm.teams[0] }}</span>
      <span class="vs">vs</span>
      <span class="{{ 'winner' if cm.teams[1] == cm.winner else 'loser' }}">{{ cm.teams[1] }}</span>
    </div>
  </div>

  <!-- Final Four -->
  <div class="panel mb-3">
    <div class="region-title">Final Four</div>
    {% for m in data.final_four %}
      <div class="matchup">
        <span class="{{ 'winner' if m.teams[0] == m.winner else 'loser' }}">{{ m.teams[0] }}</span>
        <span class="vs">vs</span>
        <span class="{{ 'winner' if m.teams[1] == m.winner else 'loser' }}">{{ m.teams[1] }}</span>
      </div>
    {% endfor %}
  </div>

  <!-- Regions -->
  <div class="row g-3">
    {% for region, rd in data.regions.items() %}
    <div class="col-md-6">
      <div class="panel">
        <div class="region-title">{{ region }} Region</div>

        <div class="round-label">Round 1</div>
        {% for m in rd.r1 %}
          <div class="matchup">
            <span class="{{ 'winner' if m.teams[0] == m.winner else 'loser' }}">{{ m.teams[0] }}</span>
            <span class="vs">vs</span>
            <span class="{{ 'winner' if m.teams[1] == m.winner else 'loser' }}">{{ m.teams[1] }}</span>
          </div>
        {% endfor %}

        <div class="round-label">Round 2</div>
        {% for m in rd.r2 %}
          <div class="matchup">
            <span class="{{ 'winner' if m.teams[0] == m.winner else 'loser' }}">{{ m.teams[0] }}</span>
            <span class="vs">vs</span>
            <span class="{{ 'winner' if m.teams[1] == m.winner else 'loser' }}">{{ m.teams[1] }}</span>
          </div>
        {% endfor %}

        <div class="round-label">Sweet 16</div>
        {% for m in rd.s16 %}
          <div class="matchup">
            <span class="{{ 'winner' if m.teams[0] == m.winner else 'loser' }}">{{ m.teams[0] }}</span>
            <span class="vs">vs</span>
            <span class="{{ 'winner' if m.teams[1] == m.winner else 'loser' }}">{{ m.teams[1] }}</span>
          </div>
        {% endfor %}

        <div class="round-label">Elite 8</div>
        <div class="matchup">
          <span class="{{ 'winner' if rd.e8.teams[0] == rd.e8.winner else 'loser' }}">{{ rd.e8.teams[0] }}</span>
          <span class="vs">vs</span>
          <span class="{{ 'winner' if rd.e8.teams[1] == rd.e8.winner else 'loser' }}">{{ rd.e8.teams[1] }}</span>
        </div>
      </div>
    </div>
    {% endfor %}
  </div>

</div>
</body>
</html>
"""


@app.route("/")
def index():
    if not BRACKET_FILE.exists():
        return (
            "<h2 style='font-family:sans-serif;padding:2rem'>"
            "Run <code>python main.py</code> first to generate bracket predictions."
            "</h2>",
            404,
        )
    data = json.loads(BRACKET_FILE.read_text())
    return render_template_string(TEMPLATE, data=data)


if __name__ == "__main__":
    print("Bracket viewer → http://localhost:8080")
    app.run(debug=False, port=8080)
