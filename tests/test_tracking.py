from src.parkinson_cv.tracking import flatten, metric_name, parse_report

REPORT = """# Rung 3: small CNN from scratch

Blocked CV.

| subset | n | prevalence | accuracy | balanced acc | AUC-ROC |
|---|---|---|---|---|---|
| spiral | 1839 | 0.542 | 0.757 [0.739-0.773] | 0.747 [0.728-0.763] | 0.869 [0.851-0.883] |
| wave | 1382 | 0.575 | 0.731 [0.709-0.752] | 0.711 [0.688-0.733] | 0.804 [0.783-0.827] |

## second table

| subset | n | AUC-ROC |
|---|---|---|
| pooled | 3221 | 0.658 [0.639-0.677] |
"""


def test_parse_report_reads_values_ci_and_sections():
    rows = parse_report(REPORT)
    assert [r["subset"] for r in rows] == ["spiral", "wave", "pooled"]
    spiral = rows[0]
    assert spiral["section"].startswith("Rung 3")
    assert spiral["metrics"]["AUC-ROC"] == (0.869, 0.851, 0.883)
    assert spiral["metrics"]["n"] == (1839.0, None, None)
    assert rows[2]["section"] == "second table"


def test_flatten_makes_safe_metric_names_with_ci_bounds():
    flat = flatten(parse_report(REPORT)[0]["metrics"])
    assert flat["auc_roc"] == 0.869 and flat["auc_roc_ci_lo"] == 0.851 and flat["auc_roc_ci_hi"] == 0.883
    assert "n_ci_lo" not in flat
    assert metric_name("balanced acc") == "balanced_acc" and metric_name("AUC-PR") == "auc_pr"


def test_text_without_tables_gives_no_rows():
    assert parse_report("# title\n\njust text\n") == []
