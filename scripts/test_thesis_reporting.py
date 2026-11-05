"""Reporting regressions: complete settlement P&L and data-driven table winners."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


path = Path(__file__).resolve().parents[1] / "analysis/graph_2026_10/b61_make_tables.py"
spec = importlib.util.spec_from_file_location("thesis_tables", path)
tables = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tables)


class ReportingTests(unittest.TestCase):
    def test_lowest_drawdown_is_highlighted_with_ties_and_missing_values(self):
        rows = [["MLB", "Model", "4.18"], ["", "Baseline", "2.00"],
                ["", "Other", "--"], ["NBA", "Model", "0.00"],
                ["", "Baseline", "0.00"]]
        result = tables.highlight_best(rows, (2,), minimize=(2,))
        self.assertEqual(result[0][2], "4.18")
        self.assertEqual(result[1][2], r"\textbf{2.00}")
        self.assertEqual(result[2][2], "--")
        self.assertEqual(result[3][2], r"\textbf{0.00}")
        self.assertEqual(result[4][2], r"\textbf{0.00}")

    def test_drawdown_includes_loss_from_initial_zero_balance(self):
        self.assertEqual(tables.maximum_drawdown([-10, 2, 3]), 10)
        self.assertEqual(tables.maximum_drawdown([5, -8, 2]), 8)
        self.assertEqual(tables.maximum_drawdown([1, 2]), 0)

    def test_partial_exits_and_settlement_both_enter_fill_pnl(self):
        lots = [dict(ticker="long", kind="spread", side=1, size=2, remaining=1,
                     entry=40, entry_fee_c=1, exits=[dict(size=1, price=50, fee_c=1)]),
                dict(ticker="short", kind="spread", side=-1, size=1, remaining=1,
                     entry=80, entry_fee_c=1, exits=[])]
        with tempfile.TemporaryDirectory() as temp:
            game = Path(temp) / "game"
            (game / "host").mkdir(parents=True)
            (game / "result.json").write_text(json.dumps(dict(root="fixture")))
            path = game / "host/lots.json"
            path.write_text(json.dumps(lots))
            with patch.object(tables, "settlement_prices", return_value={"long": 100, "short": 100}):
                values = tables.lots_pnl(path, ("spread",))
        self.assertEqual(values, [(68, True), (-21, False)])
        self.assertEqual(sum(v for v, _ in values), 47)

    def test_table_highlights_baseline_when_it_actually_wins(self):
        rows = [["MLB", "Model", "", "", "-1", "40\\%", "0.8", "-0.1"],
                ["", "Baseline", "", "", "1", "50\\%", "1.2", "0.2"]]
        result = tables.highlight_best(rows, (4, 5, 6, 7))
        self.assertTrue(all("textbf" not in v for v in result[0]))
        self.assertTrue(all("textbf" in v for v in result[1][4:]))


if __name__ == "__main__":
    unittest.main()
