import csv
import math
import tempfile
import unittest
from pathlib import Path

from analyze_jst_cagr import build_output, cagr_from_index, cagr_from_returns, read_dataset


class AnalyzeJstCagrTests(unittest.TestCase):
    def test_cagr_from_index_uses_year_span(self):
        cagr, length = cagr_from_index([(2000, 100.0), (2002, 121.0)])

        self.assertEqual(length, 2)
        self.assertTrue(math.isclose(cagr, 0.1, rel_tol=1e-9))

    def test_cagr_from_returns_uses_geometric_mean(self):
        cagr, length = cagr_from_returns([0.10, 0.21])

        self.assertEqual(length, 2)
        self.assertTrue(math.isclose(cagr, 0.1545341492, rel_tol=1e-9))

    def test_build_output_reports_requested_fields(self):
        rows_by_country = {
            "A": [
                {"year": "2000", "cpi": "100", "eq_tr": "0.10", "housing_tr": "", "bond_tr": "0.03", "bill_rate": "0.02"},
                {"year": "2001", "cpi": "110", "eq_tr": "0.00", "housing_tr": "0.05", "bond_tr": "0.04", "bill_rate": "0.01"},
                {"year": "2002", "cpi": "121", "eq_tr": "0.10", "housing_tr": "0.15", "bond_tr": "", "bill_rate": "0.03"},
            ]
        }

        output = build_output(rows_by_country, "year")

        self.assertEqual(output[0]["country"], "A")
        self.assertEqual(output[0]["cpi_length"], "3")
        self.assertEqual(output[0]["eq_tr_length"], "3")
        self.assertEqual(output[0]["housing_tr_length"], "2")
        self.assertEqual(output[0]["bond_tr_length"], "2")
        self.assertEqual(output[0]["bill_rate_length"], "3")
        self.assertEqual(output[0]["cpi_cagr"], "0.100000")

    def test_read_dataset_groups_by_country(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["country", "year", "cpi", "eq_tr", "housing_tr", "bond_tr", "bill_rate"])
                writer.writerow(["A", "2000", "100", "0.01", "0.02", "0.03", "0.04"])
                writer.writerow(["B", "2001", "105", "0.05", "0.06", "0.07", "0.08"])

            rows_by_country = read_dataset(path, "country", "year")

        self.assertEqual(sorted(rows_by_country), ["A", "B"])


if __name__ == "__main__":
    unittest.main()
