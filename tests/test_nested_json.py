"""Nested API-style JSON files load as flat tables (data_pipeline.io_utils)."""
import json

import pandas as pd
import pytest

from data_pipeline.io_utils import read_structured_file


def _write(tmp_path, obj, name="data.json", raw=None):
    p = tmp_path / name
    p.write_text(raw if raw is not None else json.dumps(obj), encoding="utf-8")
    return p


def test_frankfurter_time_series_shape(tmp_path):
    obj = {"amount": 1.0, "base": "EUR", "start_date": "2024-01-02", "end_date": "2024-01-04",
           "rates": {"2024-01-02": {"USD": 1.09, "GBP": 0.86},
                     "2024-01-03": {"USD": 1.092, "GBP": 0.865},
                     "2024-01-04": {"USD": 1.095}}}
    df = read_structured_file(_write(tmp_path, obj))
    assert list(df.columns) == ["date", "USD", "GBP"]
    assert df["date"].tolist() == ["2024-01-02", "2024-01-03", "2024-01-04"]
    assert df["USD"].tolist() == [1.09, 1.092, 1.095]
    assert pd.isna(df["GBP"].iloc[2])                       # missing key → NaN, not an error


def test_open_meteo_hourly_shape_picks_largest_table(tmp_path):
    obj = {"latitude": 33.75, "longitude": -84.39, "timezone": "GMT",
           "hourly_units": {"time": "iso8601", "temperature_2m": "°C"},
           "daily": {"time": ["2024-01-01"], "tmax": [12.0]},
           "hourly": {"time": ["2024-01-01T00:00", "2024-01-01T01:00", "2024-01-01T02:00"],
                      "temperature_2m": [5.1, 4.8, None]}}
    df = read_structured_file(_write(tmp_path, obj))
    assert list(df.columns) == ["time", "temperature_2m"]
    assert len(df) == 3 and pd.isna(df["temperature_2m"].iloc[2])


def test_wrapped_records_and_nested_records_are_flattened(tmp_path):
    obj = {"count": 2, "next": None,
           "results": [{"id": 1, "user": {"name": "a", "geo": {"lat": 1}}},
                       {"id": 2, "user": {"name": "b", "geo": {"lat": 2}}}]}
    df = read_structured_file(_write(tmp_path, obj))
    assert {"id", "user.name", "user.geo.lat"} <= set(df.columns)
    df2 = read_structured_file(_write(tmp_path, obj["results"], name="list.json"))
    assert df2["user.name"].tolist() == ["a", "b"]


def test_non_date_keys_use_member_name(tmp_path):
    obj = {"source": "x", "products": {"p1": {"price": 3}, "p2": {"price": 4}}}
    df = read_structured_file(_write(tmp_path, obj))
    assert df.columns[0] == "products" and df["products"].tolist() == ["p1", "p2"]


def test_key_column_does_not_overwrite_a_field_named_date(tmp_path):
    obj = {"meta": 1, "rates": {"2024-01-01": {"date": "x", "v": 1}}}
    df = read_structured_file(_write(tmp_path, obj))
    assert {"date_key", "date", "v"} <= set(df.columns)


@pytest.mark.parametrize("obj", [
    [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}],                       # flat records
    {"a": {"0": 1, "1": 2}, "b": {"0": "x", "1": "y"}},             # orient="columns"
    {"a": [1, 2], "b": ["x", "y"]},                                 # dict of lists
])
def test_shapes_pandas_already_reads_are_unchanged(tmp_path, obj):
    p = _write(tmp_path, obj)
    df = read_structured_file(p)
    assert df["a"].tolist() == [1, 2] and df["b"].tolist() == ["x", "y"]


def test_json_lines_and_nrows(tmp_path):
    raw = "\n".join(json.dumps({"a": i, "b": str(i)}) for i in range(5)) + "\n"
    p = _write(tmp_path, None, raw=raw)
    assert len(read_structured_file(p)) == 5
    assert len(read_structured_file(p, nrows=2)) == 2
    obj = {"base": "EUR", "rates": {f"2024-01-0{i}": {"USD": i} for i in range(1, 6)}}
    assert len(read_structured_file(_write(tmp_path, obj, name="f.json"), nrows=3)) == 3


def test_bom_and_unhelpful_shapes(tmp_path):
    obj = {"base": "EUR", "rates": {"2024-01-01": {"USD": 1.1}}}
    p = tmp_path / "bom.json"
    p.write_text("﻿" + json.dumps(obj), encoding="utf-8")
    assert read_structured_file(p)["USD"].tolist() == [1.1]
    # Scalars only, or ragged lists: no table to find, so pandas decides (and may raise).
    for bad in ({"a": 1, "b": "x"}, {"a": 1, "h": {"t": [1, 2], "v": [1]}}):
        try:
            read_structured_file(_write(tmp_path, bad, name="bad.json"))
        except ValueError:
            pass
    # Empty containers never crash the nested reader.
    for empty in ([], {}, {"a": 1, "rates": {}}):
        try:
            read_structured_file(_write(tmp_path, empty, name="e.json"))
        except ValueError:
            pass
