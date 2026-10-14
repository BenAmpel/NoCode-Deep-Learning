from models.registry import get_compatible_tasks, get_models


def test_whisper_is_not_offered_for_audio():
    for mode in ("fine-tune", "from_scratch"):
        assert not any("Whisper" in m for m in get_models("audio", mode))


def test_multi_label_is_not_offered_for_any_modality():
    for modality in ("image", "text", "tabular", "audio", "timeseries", "video", "graph"):
        assert "multi-label" not in get_compatible_tasks(modality)


def test_core_tasks_still_offered():
    assert "classification" in get_compatible_tasks("image")
    assert "regression" in get_compatible_tasks("tabular")
