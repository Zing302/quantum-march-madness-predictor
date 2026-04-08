"""
bracket_2026.py — Official 2026 NCAA Tournament Bracket
=========================================================
Source: Official ESPN bracket (Selection Sunday, March 15 2026)

First Four (March 17-18):
  (16) Howard  vs (16) UMBC          → winner plays (1) Michigan  [Midwest]
  (16) Lehigh  vs (16) Prairie View  → winner plays (1) Florida   [South]
  (11) NC State vs (11) Texas        → winner plays (6) BYU       [West]
  (11) SMU     vs (11) Miami (OH)    → winner plays (6) Tennessee [Midwest]

Round 1: March 19-20
"""

# ── First Four ────────────────────────────────────────────────────────────────

FIRST_FOUR = [
    ("Howard",       "UMBC"),           # → Midwest 16-seed
    ("Lehigh",       "Prairie View"),   # → South  16-seed
    ("NC State",     "Texas"),          # → West   11-seed (plays BYU)
    ("SMU",          "Miami (OH)"),     # → Midwest 11-seed (plays Tennessee)
]

# ── Round 1 matchups by region ────────────────────────────────────────────────
# Format: (higher seed, lower seed) — First Four slots listed as tuple

EAST = [
    ("Duke",         "Siena"),
    ("Ohio St.",     "TCU"),
    ("St. John's",   "Northern Iowa"),
    ("Kansas",       "Cal Baptist"),
    ("Louisville",   "South Florida"),
    ("Michigan St.", "North Dakota St."),
    ("UCLA",         "UCF"),
    ("UConn",        "Furman"),
]

SOUTH = [
    ("Florida",      "Lehigh"),         # Lehigh/Prairie View First Four winner; using Lehigh as placeholder
    ("Clemson",      "Iowa"),
    ("Vanderbilt",   "McNeese"),
    ("Nebraska",     "Troy"),
    ("North Carolina", "VCU"),
    ("Illinois",     "Penn"),
    ("Saint Mary's", "Texas A&M"),
    ("Houston",      "Idaho"),
]

WEST = [
    ("Arizona",      "Long Island"),
    ("Villanova",    "Utah St."),
    ("Wisconsin",    "High Point"),
    ("Arkansas",     "Hawaii"),
    ("BYU",          "NC State"),        # NC State/Texas First Four winner; using NC State as placeholder
    ("Gonzaga",      "Kennesaw St."),
    ("Miami (FL)",   "Missouri"),
    ("Purdue",       "Queens (NC)"),
]

MIDWEST = [
    ("Michigan",     "Howard"),         # Howard/UMBC First Four winner; using Howard as placeholder
    ("Georgia",      "Saint Louis"),
    ("Texas Tech",   "Akron"),
    ("Alabama",      "Hofstra"),
    ("Tennessee",    "SMU"),            # SMU/Miami OH First Four winner; using SMU as placeholder
    ("Virginia",     "Wright St."),
    ("Kentucky",     "Santa Clara"),
    ("Iowa St.",     "Tennessee St."),
]

BRACKET = {
    "East":    EAST,
    "South":   SOUTH,
    "West":    WEST,
    "Midwest": MIDWEST,
}

# ── Name aliases (ESPN API name → bracket name) ───────────────────────────────
# Add entries here if main.py reports "not found in win-prob matrix" warnings

ESPN_NAME_MAP = {
    # ESPN full name → bracket short name
    "Duke Blue Devils":                 "Duke",
    "Ohio State Buckeyes":              "Ohio St.",
    "St. John's Red Storm":             "St. John's",
    "Kansas Jayhawks":                  "Kansas",
    "Louisville Cardinals":             "Louisville",
    "Michigan State Spartans":          "Michigan St.",
    "UCLA Bruins":                      "UCLA",
    "UConn Huskies":                    "UConn",
    "Florida Gators":                   "Florida",
    "Clemson Tigers":                   "Clemson",
    "Iowa Hawkeyes":                    "Iowa",
    "Vanderbilt Commodores":            "Vanderbilt",
    "Nebraska Cornhuskers":             "Nebraska",
    "North Carolina Tar Heels":         "North Carolina",
    "Illinois Fighting Illini":         "Illinois",
    "Saint Mary's Gaels":               "Saint Mary's",
    "Houston Cougars":                  "Houston",
    "Arizona Wildcats":                 "Arizona",
    "Villanova Wildcats":               "Villanova",
    "Utah State Aggies":                "Utah St.",
    "Wisconsin Badgers":                "Wisconsin",
    "Arkansas Razorbacks":              "Arkansas",
    "BYU Cougars":                      "BYU",
    "NC State Wolfpack":                "NC State",
    "Gonzaga Bulldogs":                 "Gonzaga",
    "Kennesaw State Owls":              "Kennesaw St.",
    "Miami Hurricanes":                 "Miami (FL)",
    "Missouri Tigers":                  "Missouri",
    "Purdue Boilermakers":              "Purdue",
    "Michigan Wolverines":              "Michigan",
    "Georgia Bulldogs":                 "Georgia",
    "Saint Louis Billikens":            "Saint Louis",
    "Texas Tech Red Raiders":           "Texas Tech",
    "Akron Zips":                       "Akron",
    "Alabama Crimson Tide":             "Alabama",
    "Hofstra Pride":                    "Hofstra",
    "Tennessee Volunteers":             "Tennessee",
    "SMU Mustangs":                     "SMU",
    "Miami (OH) RedHawks":              "Miami (OH)",
    "Virginia Cavaliers":               "Virginia",
    "Wright State Raiders":             "Wright St.",
    "Kentucky Wildcats":                "Kentucky",
    "Santa Clara Broncos":              "Santa Clara",
    "Iowa State Cyclones":              "Iowa St.",
    "Tennessee State Tigers":           "Tennessee St.",
    "Siena Saints":                     "Siena",
    "TCU Horned Frogs":                 "TCU",
    "Northern Iowa Panthers":           "Northern Iowa",
    "California Baptist Lancers":       "Cal Baptist",
    "South Florida Bulls":              "South Florida",
    "VCU Rams":                         "VCU",
    "Pennsylvania Quakers":             "Penn",
    "Texas A&M Aggies":                 "Texas A&M",
    "Idaho Vandals":                    "Idaho",
    "Long Island University Sharks":    "Long Island",
    "High Point Panthers":              "High Point",
    "McNeese Cowboys":                  "McNeese",
    "North Dakota State Bison":         "North Dakota St.",
    "UCF Knights":                      "UCF",
    "Troy Trojans":                     "Troy",
    "Furman Paladins":                  "Furman",
    "Howard Bison":                     "Howard",
    "UMBC Retrievers":                  "UMBC",
    "Lehigh Mountain Hawks":            "Lehigh",
    "Prairie View A&M Panthers":        "Prairie View",
    "Texas Longhorns":                  "Texas",
    # Hawaii and Queens (NC) not in ESPN stats endpoint — will be skipped
}


def _normalize(name: str) -> str:
    return ESPN_NAME_MAP.get(name, name)


# ── Public API ────────────────────────────────────────────────────────────────

def get_round1_matchups() -> list[tuple[str, str]]:
    """Return all 32 Round 1 matchups as (team_a, team_b) name pairs."""
    matchups = []
    for region_games in BRACKET.values():
        matchups.extend(region_games)
    return matchups


def get_round1_matchups_indexed(names: list[str]) -> list[tuple[int, int]]:
    """
    Return Round 1 matchups as (index_a, index_b) pairs aligned to
    the `names` list from build_win_prob_matrix().
    Teams not found in names are skipped with a warning.
    """
    name_to_idx = {n: i for i, n in enumerate(names)}
    # Also index normalized names
    normalized_idx = {_normalize(n): i for n, i in name_to_idx.items()}
    name_to_idx.update(normalized_idx)

    matchups = []
    for a, b in get_round1_matchups():
        ia = name_to_idx.get(a) or name_to_idx.get(_normalize(a))
        ib = name_to_idx.get(b) or name_to_idx.get(_normalize(b))
        if ia is None or ib is None:
            missing = a if ia is None else b
            print(f"  WARNING: '{missing}' not found in win-prob matrix — skipping")
            continue
        matchups.append((ia, ib))
    return matchups


if __name__ == "__main__":
    print("2026 NCAA Tournament — Official Round 1 Matchups\n")
    for region, games in BRACKET.items():
        print(f"  {region}")
        seeds = [1,8,5,4,6,3,7,2]
        opponents = [16,9,12,13,11,14,10,15]
        for (a, b), s1, s2 in zip(games, seeds, opponents):
            print(f"    ({s1:2d}) {a:<25s} vs  ({s2:2d}) {b}")
        print()
    print("  First Four:")
    for a, b in FIRST_FOUR:
        print(f"    {a} vs {b}")
