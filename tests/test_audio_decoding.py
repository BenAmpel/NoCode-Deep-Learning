import numpy as np
import pytest
import soundfile

from export import bundle_runtime as rt

SR = 16000
TONE = np.sin(np.linspace(0, 440 * 2 * np.pi, SR)).astype(np.float32) * 0.5
FORMATS = {
    ".wav": ("WAV", "PCM_16"),
    ".flac": ("FLAC", "PCM_16"),
    ".mp3": ("MP3", "MPEG_LAYER_III"),
    ".ogg": ("OGG", "VORBIS"),
}


@pytest.fixture
def no_ffmpeg(monkeypatch):
    """Behave like a machine without ffmpeg (the Windows CI runner, .pkg installs)."""
    real_which = rt.shutil.which
    monkeypatch.setattr(rt.shutil, "which", lambda name, *a, **k: None if name == "ffmpeg" else real_which(name, *a, **k))


def _write(path, ext):
    fmt, subtype = FORMATS[ext]
    soundfile.write(path, TONE, SR, format=fmt, subtype=subtype)
    return str(path)


@pytest.mark.parametrize("ext", sorted(FORMATS))
def test_common_formats_decode_without_ffmpeg(tmp_path, no_ffmpeg, ext):
    path = _write(tmp_path / f"tone{ext}", ext)
    assert not rt.needs_ffmpeg(path)
    waveform, sr = rt.load_audio_waveform(path, target_sr=8000)
    assert sr == 8000 and waveform.shape[0] == 1
    assert waveform.shape[1] == pytest.approx(8000, abs=1200)  # mp3 adds encoder padding
    assert float(waveform.abs().max()) == pytest.approx(0.5, abs=0.05)


def test_m4a_without_ffmpeg_explains_the_fix(tmp_path, no_ffmpeg):
    path = tmp_path / "clip.m4a"
    path.write_bytes(b"\x00" * 64)
    assert rt.needs_ffmpeg(str(path))
    with pytest.raises(RuntimeError, match="needs ffmpeg, which isn't installed"):
        rt.load_audio_waveform(str(path))


def _dataset(root, extra_ext=".m4a"):
    for label in ("a", "b"):
        (root / label).mkdir(parents=True)
        for i in range(3):
            _write(root / label / f"{i}.wav", ".wav")
    (root / "a" / f"phone{extra_ext}").write_bytes(b"\x00" * 64)
    return root


def test_training_refuses_up_front_when_files_need_missing_ffmpeg(tmp_path, no_ffmpeg):
    from modalities.audio import load_audio_data

    with pytest.raises(ValueError, match=r"1 audio file\(s\) \(\.m4a\) need ffmpeg.*Skip unreadable audio files"):
        load_audio_data(str(_dataset(tmp_path / "ds")), sample_rate=SR, image_size=32)


def test_skip_unreadable_trains_without_them(tmp_path, no_ffmpeg):
    from modalities.audio import load_audio_data

    train, val, classes, _ = load_audio_data(str(_dataset(tmp_path / "ds")), sample_rate=SR,
                                             image_size=32, verify_files=True, augmentation="none")
    paths = [p for p, _ in train.dataset.samples + val.dataset.samples]
    assert classes == ["a", "b"] and len(paths) == 6 and not any(p.endswith(".m4a") for p in paths)


def test_dataset_preview_warns_about_files_needing_ffmpeg(tmp_path, no_ffmpeg):
    from data_pipeline.validation import validate_dataset

    issues = validate_dataset(str(_dataset(tmp_path / "ds")), "audio")
    assert any(i.startswith("⚠️") and ".m4a" in i and "ffmpeg" in i for i in issues)


def test_no_warning_when_every_file_decodes(tmp_path, no_ffmpeg):
    from data_pipeline.validation import validate_dataset

    issues = validate_dataset(str(_dataset(tmp_path / "ds", extra_ext=".wav")), "audio")
    assert not any("ffmpeg" in i for i in issues)
