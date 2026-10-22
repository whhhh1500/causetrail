from causetrail.cli import main


def test_cli_ask_writes_report(data_dir, tmp_path, capsys):
    rc = main(["ask", "why did refunds spike in week 40 in APAC?", "--data", str(data_dir), "--llm", "none",
               "--out", str(tmp_path), "--quiet"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Most likely root cause" in out
    assert (tmp_path / "report.md").exists() and (tmp_path / "evidence.json").exists()


def test_cli_sql(data_dir, capsys):
    assert main(["sql", "SELECT COUNT(*) AS n FROM {refunds}", "--data", str(data_dir)]) == 0
    assert "n" in capsys.readouterr().out
