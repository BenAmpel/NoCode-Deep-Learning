from unittest import mock

import config


def test_unusable_mps_falls_back_to_cpu(monkeypatch):
    # Virtualised Macs report MPS as available but fail every allocation.
    monkeypatch.delenv("NOCODE_DL_DEVICE", raising=False)
    with mock.patch("torch.backends.mps.is_available", return_value=True), \
         mock.patch("torch.zeros", side_effect=RuntimeError("MPS backend out of memory")), \
         mock.patch("torch.cuda.is_available", return_value=False):
        assert config.get_device() == "cpu"


def test_device_can_be_forced(monkeypatch):
    monkeypatch.setenv("NOCODE_DL_DEVICE", "CPU")
    assert config.get_device() == "cpu"


def test_unknown_forced_device_is_ignored(monkeypatch):
    monkeypatch.setenv("NOCODE_DL_DEVICE", "tpu")
    with mock.patch("torch.backends.mps.is_available", return_value=False), \
         mock.patch("torch.cuda.is_available", return_value=False):
        assert config.get_device() == "cpu"
