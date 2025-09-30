import pandas as pd
import pytest

from models.utils import ColumnSpec, resolve_from_csvs


def _write_csv(path, df):
    df.to_csv(path, index=False)


def test_resolve_from_csvs_regex_selection(tmp_path):
    train_df = pd.DataFrame(
        {
            "input_ax": [1.0, 2.0],
            "input_ay": [3.0, 4.0],
            "target_b": [5.0, 6.0],
        }
    )
    test_df = pd.DataFrame(
        {
            "input_ax": [0.5],
            "input_ay": [1.5],
            "target_b": [2.5],
        }
    )
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    _write_csv(train_path, train_df)
    _write_csv(test_path, test_df)

    resolved = resolve_from_csvs(
        [train_path],
        [test_path],
        input_spec=ColumnSpec(patterns=["^input_"]),
        target_spec=ColumnSpec(patterns=["^target_"]),
    )

    assert resolved["input"] == ["input_ax", "input_ay"]
    assert resolved["target"] == ["target_b"]


def test_resolve_from_csvs_empty_spec_raises(tmp_path):
    df = pd.DataFrame({"a": [1], "b": [2]})
    path = tmp_path / "data.csv"
    _write_csv(path, df)

    with pytest.raises(ValueError, match="No columns specified"):
        resolve_from_csvs([path], [path], ColumnSpec(), ColumnSpec(patterns=["b"]))


def test_resolve_from_csvs_invalid_regex(tmp_path):
    df = pd.DataFrame({"a": [1], "b": [2]})
    path = tmp_path / "data.csv"
    _write_csv(path, df)

    with pytest.raises(ValueError, match="Invalid regex pattern"):
        resolve_from_csvs(
            [path],
            [path],
            ColumnSpec(patterns=["[unclosed"]),
            ColumnSpec(patterns=["b"]),
        )
