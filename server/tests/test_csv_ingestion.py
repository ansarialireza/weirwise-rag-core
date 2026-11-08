import pytest
from server.app.ingestion import extract_text_from_file

def test_extract_csv_formatting(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("item,qty,price\nWidget A,10,$5.00\nWidget B,20,$12.50\n", encoding="utf-8")
    
    extracted = extract_text_from_file(str(csv_file), "text/csv")
    assert "item: Widget A | qty: 10 | price: $5.00" in extracted
    assert "item: Widget B | qty: 20 | price: $12.50" in extracted

def test_extract_csv_empty(tmp_path):
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text("", encoding="utf-8")
    assert extract_text_from_file(str(csv_file), "text/csv") == ""
