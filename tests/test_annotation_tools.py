import json
from pathlib import Path

import pytest
from PIL import Image

from ui import annotation_tools as at


# ---------------------------------------------------------------------------
# Text relabelling
# ---------------------------------------------------------------------------

CSV_TEXT = (
    'id,text,price,label\n'
    '007,"hello, world",1.50,pos\n'
    '\n'                                   # pandas skips blank lines
    '010,plain,2.00,neg\n'
)


def test_csv_relabel_changes_only_that_cell_and_keeps_original(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(CSV_TEXT, encoding="utf-8")
    choices, _ = at.list_text_annotation_rows(path, "text", "label")
    second_row = choices[1][1]  # row index as the UI gives it

    assert at.save_text_relabel(path, second_row, "label", "neutral") == "neutral"

    assert path.read_text(encoding="utf-8") == CSV_TEXT.replace("2.00,neg", "2.00,neutral")
    assert at.backup_path_for(path).read_text(encoding="utf-8") == CSV_TEXT  # untouched original


def test_backup_is_taken_only_once(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(CSV_TEXT, encoding="utf-8")
    at.save_text_relabel(path, 0, "label", "first")
    at.save_text_relabel(path, 0, "label", "second")
    assert at.backup_path_for(path).read_text(encoding="utf-8") == CSV_TEXT


def test_tsv_relabel(tmp_path):
    path = tmp_path / "data.tsv"
    path.write_bytes(b"text\tlabel\r\nok\ta\r\nfine\tb\r\n")  # bytes: text mode would add \r on Windows
    at.save_text_relabel(path, 0, "label", "z")
    assert path.read_bytes() == b"text\tlabel\r\nok\tz\r\nfine\tb\r\n"  # CRLF kept


def test_json_lines_stay_json_lines(tmp_path):
    path = tmp_path / "data.json"
    path.write_text('{"text": "a", "label": "x"}\n{"text": "b", "label": "y"}\n', encoding="utf-8")
    at.save_text_relabel(path, 1, "label", "new")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(ln)["label"] for ln in lines] == ["x", "new"]


def test_json_array_stays_an_array(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(json.dumps([{"text": "a", "label": "x"}, {"text": "b", "label": "y"}], indent=2), encoding="utf-8")
    at.save_text_relabel(path, 0, "label", "new")
    assert [r["label"] for r in json.loads(path.read_text(encoding="utf-8"))] == ["new", "y"]


def test_bad_row_leaves_file_unchanged(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(CSV_TEXT, encoding="utf-8")
    with pytest.raises(ValueError, match="Row 9"):
        at.save_text_relabel(path, 9, "label", "x")
    assert path.read_text(encoding="utf-8") == CSV_TEXT


# ---------------------------------------------------------------------------
# Folder relabelling
# ---------------------------------------------------------------------------

@pytest.fixture
def folder_dataset(tmp_path):
    root = tmp_path / "ds"
    (root / "cat").mkdir(parents=True)
    (root / "cat" / "a.png").write_bytes(b"x")
    return root


@pytest.mark.parametrize("label", ["../escaped", "/abs", "a/b", ".hidden", "..", "c:d"])
def test_folder_relabel_rejects_labels_that_are_not_plain_names(folder_dataset, label):
    with pytest.raises(ValueError, match="can't be used as a class label"):
        at.relabel_folder_sample(folder_dataset, "cat/a.png", label)
    assert (folder_dataset / "cat" / "a.png").exists()


def test_folder_relabel_rejects_sample_paths_outside_dataset(folder_dataset, tmp_path):
    (tmp_path / "outside.png").write_bytes(b"x")
    with pytest.raises(ValueError, match="outside the dataset"):
        at.relabel_folder_sample(folder_dataset, "../outside.png", "dog")
    assert (tmp_path / "outside.png").exists()


def test_folder_relabel_moves_file(folder_dataset):
    rel, label = at.relabel_folder_sample(folder_dataset, "cat/a.png", "dog")
    assert (rel, label) == (str(Path("dog") / "a.png"), "dog")
    assert (folder_dataset / "dog" / "a.png").exists()


# ---------------------------------------------------------------------------
# Object-box review export
# ---------------------------------------------------------------------------

def _image(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 50)).save(path)
    return str(path)


def _box(label, x1=10, y1=10, x2=50, y2=40):
    return {"label": label, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "keep": True}


def test_class_ids_are_shared_across_images(tmp_path):
    out = tmp_path / "reviews"
    first = at.save_object_box_review(_image(tmp_path / "a.png"), [_box("dog"), _box("cat")], output_dir=out)
    second = at.save_object_box_review(_image(tmp_path / "b.png"), [_box("cat"), _box("bird")], output_dir=out)
    assert (out / "classes.txt").read_text().split() == ["dog", "cat", "bird"]
    assert [ln.split()[0] for ln in Path(first).read_text().splitlines()] == ["0", "1"]
    assert [ln.split()[0] for ln in Path(second).read_text().splitlines()] == ["1", "2"]  # cat stays 1


def test_same_named_images_do_not_overwrite_each_other(tmp_path):
    out = tmp_path / "reviews"
    one = at.save_object_box_review(_image(tmp_path / "x" / "img.png"), [_box("dog")], output_dir=out)
    two = at.save_object_box_review(_image(tmp_path / "y" / "img.png"), [_box("cat")], output_dir=out)
    assert one != two and Path(one).exists() and Path(two).exists()


def test_default_output_is_under_the_project_not_the_cwd(tmp_path, monkeypatch):
    monkeypatch.setattr(at, "_PROJECT_ROOT", tmp_path / "project")
    monkeypatch.chdir(tmp_path)
    saved = at.save_object_box_review(_image(tmp_path / "a.png"), [_box("dog")])
    assert Path(saved).is_relative_to(tmp_path / "project" / "outputs" / "object_box_reviews")


def test_boxes_are_clamped_to_the_image(tmp_path):
    saved = at.save_object_box_review(_image(tmp_path / "a.png"), [_box("dog", -20, -5, 150, 80)],
                                      output_dir=tmp_path / "r")
    values = [float(v) for v in Path(saved).read_text().split()[1:]]
    assert all(0.0 <= v <= 1.0 for v in values)
