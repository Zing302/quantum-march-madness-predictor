"""
fetch_stats.py — ESPN Team Stats Fetcher
=========================================
Pulls current-season and historical team stats and tournament results
from ESPN's API. Pure I/O — no modelling here.

Endpoints used (all return 200 as of 2026):
  - /teams                        → team IDs + metadata
  - /teams/{id}                   → per-team record (PPG, PAPG, W%, SOS)
  - /statistics/byteam            → shooting efficiency, FG%, 3P%, FT%
  - /scoreboard?seasontype=3      → tournament game results

Public API
----------
fetch_current_stats()          -> pd.DataFrame   current season, all D-I teams
fetch_historical_stats(season) -> pd.DataFrame   one past season
fetch_historical_games()       -> pd.DataFrame   tournament results 2010-2024
"""

from __future__ import annotations

import sys
import time

import pandas as pd
import requests

# ── Constants ─────────────────────────────────────────────────────────────────

HISTORICAL_SEASONS = [s for s in range(2010, 2025) if s != 2020]
CURRENT_SEASON     = 2026

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
STATS_URL      = "https://site.web.api.espn.com/apis/common/v3/sports/basketball/mens-college-basketball/statistics/byteam"
SCOREBOARD_URL = f"{BASE}/scoreboard"
TEAMS_URL      = f"{BASE}/teams"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    )
}

# ── Internal helpers ──────────────────────────────────────────────────────────

def _get(url: str, params: dict = {}, retries: int = 3, backoff: float = 1.5) -> dict:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=20)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            wait = backoff ** attempt
            print(f"    Retry {attempt + 1}/{retries} after {wait:.1f}s — {exc}")
            time.sleep(wait)
    return {}


def _safe_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _stat(stats: list[dict], name: str) -> float | None:
    for s in stats:
        if s.get("name") == name:
            return _safe_float(s.get("value"))
    return None


# ── Team ID fetching ──────────────────────────────────────────────────────────

def _fetch_all_team_ids(season: int | None = None) -> list[dict]:
    """Return list of {team_id, team_name, abbreviation} for all D-I teams."""
    teams = []
    page = 1
    while True:
        params = {"limit": 100, "page": page, "groups": 50}
        if season:
            params["season"] = season
        try:
            data = _get(TEAMS_URL, params)
        except Exception:
            break
        page_teams = []
        for sport in data.get("sports", []):
            for league in sport.get("leagues", []):
                for t in league.get("teams", []):
                    team = t.get("team", {})
                    page_teams.append({
                        "team_id":      str(team.get("id", "")),
                        "team_name":    team.get("displayName", team.get("name", "")),
                        "abbreviation": team.get("abbreviation", ""),
                    })
        if not page_teams:
            break
        teams.extend(page_teams)
        page += 1
        time.sleep(0.2)
    return teams


# ── Per-team record fetching ──────────────────────────────────────────────────

def _fetch_team_record(team_id: str) -> dict:
    """Fetch W%, PPG, PAPG, point differential for a single team."""
    try:
        data = _get(f"{BASE}/teams/{team_id}")
        record_items = data.get("team", {}).get("record", {}).get("items", [])
        # Use overall record (first item, type='total')
        for item in record_items:
            if item.get("type") == "total":
                stats = item.get("stats", [])
                return {
                    "win_pct":     _stat(stats, "winPercent"),
                    "ppg":         _stat(stats, "avgPointsFor"),
                    "papg":        _stat(stats, "avgPointsAgainst"),
                    "point_diff":  _stat(stats, "differential"),
                    "games":       _stat(stats, "gamesPlayed"),
                    "wins":        _stat(stats, "wins"),
                    "losses":      _stat(stats, "losses"),
                }
    except Exception:
        pass
    return {}


# ── Shooting stats fetching ───────────────────────────────────────────────────

def _fetch_shooting_stats(season: int | None = None) -> dict[str, dict]:
    """
    Return {team_id: {fg_pct, three_pct, ft_pct, sc_eff, sh_eff}} for all teams.
    Uses the statistics/byteam endpoint.
    """
    # Offensive category column order (from glossary):
    # 0:avgPoints, 1:fieldGoalPct, 2:threePointFieldGoalPct, 3:freeThrowPct,
    # 4:avgTurnovers, 24:scoringEfficiency, 25:shootingEfficiency
    OFF_IDX = {"fg_pct": 1, "three_pct": 2, "ft_pct": 3, "sc_eff": 24, "sh_eff": 25}

    result: dict[str, dict] = {}
    page = 1
    while True:
        params = {"limit": 100, "page": page, "groups": 50}
        if season:
            params["season"] = season
        try:
            data = _get(STATS_URL, params)
        except Exception:
            break

        for team_entry in data.get("teams", []):
            tid = str(team_entry.get("team", {}).get("id", ""))
            off_vals = []
            for cat in team_entry.get("categories", []):
                if cat.get("name") == "offensive":
                    off_vals = cat.get("values", [])
                    break

            def _vi(idx):
                try:
                    v = off_vals[idx]
                    return float(v) if v is not None else None
                except (IndexError, TypeError):
                    return None

            result[tid] = {k: _vi(i) for k, i in OFF_IDX.items()}

        pagination = data.get("pagination", {})
        if page >= pagination.get("pages", 1):
            break
        page += 1
        time.sleep(0.2)

    return result


# ── Main stat builder ─────────────────────────────────────────────────────────

def _fetch_stats(season: int | None = None) -> pd.DataFrame:
    label = str(season) if season else "current"
    print(f"  [{label}] fetching team list…", end="", flush=True)

    teams = _fetch_all_team_ids(season)
    print(f" {len(teams)} teams. Fetching shooting stats…", end="", flush=True)

    shooting = _fetch_shooting_stats(season)
    print(f" done. Fetching per-team records (this takes ~{len(teams)//5}s)…")

    rows = []
    for i, t in enumerate(teams):
        tid = t["team_id"]
        record = _fetch_team_record(tid)
        row = {**t, **record, **shooting.get(tid, {})}
        rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(teams)} teams fetched…")
        time.sleep(0.05)  # ~5 req/s — polite rate

    df = pd.DataFrame(rows)
    # Require at minimum a win% and PPG to be useful
    df = df.dropna(subset=["win_pct", "ppg"])
    print(f"  [{label}] {len(df)} usable teams.")
    return df


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_current_stats() -> pd.DataFrame:
    """Return stats for all D-I teams in the current season."""
    return _fetch_stats(season=None)


# Keep old name as alias so win_probability.py import doesn't break
fetch_current_bpi = fetch_current_stats


def fetch_historical_stats(season: int) -> pd.DataFrame:
    """Return stats for all D-I teams in a specific past season."""
    return _fetch_stats(season=season)


fetch_historical_bpi = fetch_historical_stats


def fetch_historical_games() -> pd.DataFrame:
    """
    Return completed NCAA tournament game results for all historical seasons
    (2010–2024, excluding 2020).

    Columns: season, winner_id, loser_id, winner_score, loser_score
    """
    print("Fetching historical NCAA tournament results…")
    all_games: list[dict] = []
    for season in HISTORICAL_SEASONS:
        params = {
            "groups":     100,  # NCAA Tournament bracket group
            "seasontype": 3,    # postseason
            "limit":      200,
            "dates":      season,
        }
        try:
            data  = _get(SCOREBOARD_URL, params)
            games = _parse_tournament_games(data, season)
        except Exception as exc:
            print(f"  {season}: FAILED ({exc})")
            games = []
        print(f"  {season}: {len(games):3d} games")
        all_games.extend(games)
        time.sleep(0.3)

    df = pd.DataFrame(all_games)
    print(f"  Total: {len(df)} historical tournament games.")
    return df


def fetch_team_schedule_features(team_id: str, season: int | None = None) -> dict:
    """
    Fetch a team's regular season game log and compute derived features:
      recent_form    — win% over last 10 games
      road_win_pct   — win% in away games
      margin_std     — std dev of point margins (lower = more consistent)
      close_win_pct  — win% in games decided by ≤5 points
    """
    params: dict = {"seasontype": 2}  # regular season only
    if season:
        params["season"] = season

    try:
        data = _get(f"{BASE}/teams/{team_id}/schedule", params)
    except Exception:
        return {}

    games = []
    for event in data.get("events", []):
        for comp in event.get("competitions", []):
            if not comp.get("status", {}).get("type", {}).get("completed", False):
                continue
            comps = comp.get("competitors", [])
            if len(comps) != 2:
                continue
            our = next((c for c in comps if str(c.get("team", {}).get("id", "")) == str(team_id)), None)
            opp = next((c for c in comps if str(c.get("team", {}).get("id", "")) != str(team_id)), None)
            if not our or not opp:
                continue
            try:
                def _score(c):
                    s = c.get("score", 0)
                    if isinstance(s, dict):
                        return int(s.get("value", 0) or 0)
                    return int(s or 0)
                our_score = _score(our)
                opp_score = _score(opp)
            except (TypeError, ValueError):
                continue
            games.append({
                "win":    bool(our.get("winner", False)),
                "margin": our_score - opp_score,
                "home":   our.get("homeAway") == "home",
            })

    if not games:
        return {}

    n      = len(games)
    wins   = sum(g["win"] for g in games)
    margins = [g["margin"] for g in games]

    recent      = games[-10:]
    recent_form = sum(g["win"] for g in recent) / len(recent)

    away         = [g for g in games if not g["home"]]
    road_win_pct = sum(g["win"] for g in away) / len(away) if away else wins / n

    import statistics
    margin_std = statistics.stdev(margins) if len(margins) > 1 else 0.0

    close         = [g for g in games if abs(g["margin"]) <= 5]
    close_win_pct = sum(g["win"] for g in close) / len(close) if close else wins / n

    return {
        "recent_form":   recent_form,
        "road_win_pct":  road_win_pct,
        "margin_std":    margin_std,
        "close_win_pct": close_win_pct,
    }


def _parse_tournament_games(data: dict, season: int) -> list[dict]:
    games = []
    for event in data.get("events", []):
        for comp in event.get("competitions", []):
            if not comp.get("status", {}).get("type", {}).get("completed", False):
                continue
            competitors = comp.get("competitors", [])
            if len(competitors) != 2:
                continue
            c0, c1 = competitors[0], competitors[1]
            if c0.get("winner"):
                winner, loser = c0, c1
            elif c1.get("winner"):
                winner, loser = c1, c0
            else:
                continue
            games.append({
                "season":       season,
                "winner_id":    str(winner["team"]["id"]),
                "loser_id":     str(loser["team"]["id"]),
                "winner_score": int(winner.get("score") or 0),
                "loser_score":  int(loser.get("score") or 0),
            })
    return games


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("fetch_stats — ESPN Team Stats Fetcher")
    print("=" * 50)

    print("\n[1/2] Current-season stats…")
    current = fetch_current_stats()
    if current.empty:
        print("ERROR: No stats returned.")
        sys.exit(1)
    current.to_csv("current_stats.csv", index=False)
    print(f"Saved → current_stats.csv  ({len(current)} teams)")
    print(current[["team_name", "win_pct", "ppg", "papg", "point_diff"]].head(10).to_string())

    print("\n[2/2] Historical tournament games…")
    games = fetch_historical_games()
    if games.empty:
        print("ERROR: No historical games returned.")
        sys.exit(1)
    games.to_csv("historical_games.csv", index=False)
    print(f"Saved → historical_games.csv  ({len(games)} games)")
