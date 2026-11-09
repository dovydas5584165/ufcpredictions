import pandas as pd

from fightdata import build_features
from fightdata.schema import bouts_frame, normalize_method, weight_limit_kg, detect_sport
from fightdata.sources.ufc_api import parse_event, parse_fight_stats
from fightdata.sources.wikipedia import parse_results_page

PAGE = """
<div class="mw-heading mw-heading2"><h2 id="Glory_1">Glory 1</h2></div>
<table class="infobox"><tr><th>Glory 1</th></tr><tr><th>Date</th><td>February 22, 2025</td></tr></table>
<table class="wikitable">
<tr><th colspan="8">Glory 1 (DAZN)</th></tr>
<tr><th>Weight Class</th><th></th><th></th><th></th><th>Method</th><th>Round</th><th>Time</th><th>Notes</th></tr>
<tr><td>Welterweight 77 kg</td><td><span class="flagicon"><a href="/wiki/Netherlands">x</a></span> <a href="/wiki/Ann_Alpha">Ann Alpha</a> (c)<sup>[1]</sup></td>
<td>def.</td><td>Bea Beta</td><td>TKO (punches)</td><td>2</td><td>1:23</td><td>For the Glory Welterweight Championship</td></tr>
<tr><td>Flyweight Muay Thai</td><td>Cid Gamma</td><td>vs.</td><td>Dan Delta</td><td>Draw (majority)</td><td>3</td><td>3:00</td><td></td></tr>
<tr><td>Flyweight</td><td>Eve Eps</td><td>vs.</td><td>Fay Zeta</td><td></td><td></td><td></td><td></td></tr>
</table>
<div class="mw-heading mw-heading2"><h2 id="Awards">Awards</h2></div>
<table class="wikitable"><tr><th>#</th><th></th><th></th><th></th><th>Method</th></tr>
<tr><td>1</td><td>Ann Alpha</td><td>def.</td><td>Bea Beta</td><td>KO</td></tr></table>
"""


def test_wikipedia_parser():
    rows = parse_results_page(PAGE, "glory", "Glory", "kickboxing", {"flyweight": 57.0})
    assert len(rows) == 3  # awards table ignored
    first, draw, upcoming = rows
    assert first["event_date"] == "2025-02-22" and first["is_title_bout"]
    winner_name = first["r_name"] if first["winner"] == "Red" else first["b_name"]
    assert winner_name == "Ann Alpha"  # footnote and (c) stripped
    assert {first["r_id"], first["b_id"]} == {"wiki:Ann Alpha", "name:glory:bea beta"}
    assert first["method"] == "KO/TKO" and first["finish_time_sec"] == 83 and first["weight_class_kg"] == 77
    assert draw["winner"] == "Draw" and draw["sport"] == "muay_thai"
    assert upcoming["status"] == "upcoming" and upcoming["winner"] is None and upcoming["weight_class_kg"] == 57


def test_ufc_parsers():
    fighter = lambda fid, corner, outcome: {"FighterId": fid, "Corner": corner, "Name": {"FirstName": "A", "LastName": str(fid)},
                                            "DOB": "1990-01-01", "Height": 70, "Reach": 72, "WeighIn": 155.5,
                                            "Outcome": {"Outcome": outcome}}
    event = {"EventId": 1, "Name": "UFC X", "StartTime": "2025-01-01T00:00Z", "Status": "Final",
             "Organization": {"Name": "Ultimate Fighting Championship"},
             "FightCard": [{"FightId": 9, "FightOrder": 1, "Status": "Final",
                            "Fighters": [fighter(1, "Red", "Loss"), fighter(2, "Blue", "Win")],
                            "Result": {"Method": "Decision - Split", "EndingRound": 3, "EndingTime": "5:00"},
                            "WeightClass": {"Description": "Lightweight", "Weight": "146-155"},
                            "RuleSet": {"PossibleRounds": 3, "Description": "3 Rnd (5-5-5)"},
                            "Accolades": [{"Type": "Belt", "Name": "UFC Lightweight Title"}]}]}
    (row,) = parse_event(event)
    assert row["winner"] == "Blue" and row["method"] == "DEC" and row["is_title_bout"]
    assert row["fight_minutes"] == 15 and row["weight_class_kg"] == 70.3 and row["r_reach_cm"] == 182.9
    stats = parse_fight_stats({"LiveFightDetail": {"FightStats": [
        {"FighterId": 1, "TotalStrikesAttempted": 10, "SigStrikesLanded": 5, "SigStrikesAttempted": 10, "ControlTime": "1:00"}]}})
    assert stats[1]["sig_str_landed"] == 5 and stats[1]["ctrl_sec"] == 60


def test_features_use_only_earlier_bouts():
    def bout(i, date, red, blue, winner, method="KO/TKO", status="final"):
        return {"bout_id": f"t:{i}", "promotion": "t", "sport": "mma", "status": status, "event_date": date,
                "bout_order": i, "r_id": red, "b_id": blue, "winner": winner, "method": method, "finish_round": 1,
                "r_sig_str_landed": 10, "r_sig_str_att": 20, "b_sig_str_landed": 5, "b_sig_str_att": 20, "fight_minutes": 5}
    df = build_features(bouts_frame([
        bout(1, "2020-01-01", "A", "B", "Red"),
        bout(2, "2020-06-01", "A", "C", "Red"),
        bout(3, "2020-06-01", "B", "C", "Blue"),   # same night as bout 2: must not see it
        bout(4, "2021-01-01", "A", "B", None, status="upcoming"),
    ])).set_index("bout_id")
    assert df.loc["t:1", "r_pre_fights"] == 0
    assert df.loc["t:2", "r_pre_fights"] == 1 and df.loc["t:2", "b_pre_fights"] == 0
    assert df.loc["t:3", "b_pre_fights"] == 0  # C's bout 2 on the same date is excluded
    assert df.loc["t:4", "r_pre_streak"] == 2 and df.loc["t:4", "b_pre_chin"] == 0.0
    assert df.loc["t:4", "r_pre_sig_str_acc"] == 0.5 and pd.isna(df.loc["t:4", "target"])


def test_helpers():
    assert normalize_method("TKO - Doctor's Stoppage") == "KO/TKO"
    assert normalize_method("Decision (unanimous)") == "DEC" and normalize_method("NC (eye poke)") == "NC"
    assert weight_limit_kg("Catchweight (130 lbs) Muay Thai") == 59.0
    assert detect_sport("Bantamweight", "Kickboxing", default="mma") == "kickboxing"
