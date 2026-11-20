"""Install/launch decisions, tested without running pip for real."""
import json
import platform
import socket
import subprocess
import sys
from pathlib import Path

import pytest

import install
from runtime import runtime_setup as rs


def _make_venv(venv: Path, **state_overrides) -> Path:
    python = rs.venv_python_path(venv)
    python.parent.mkdir(parents=True)
    python.write_text("")
    state = {
        "requirements_hash": rs.requirements_hash(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "venv_python": str(python),
        "requirements": rs.requirement_names(),
        **state_overrides,
    }
    rs.install_state_path(venv).write_text(json.dumps(state))
    return python


def test_current_install_is_current(tmp_path):
    _make_venv(tmp_path / "venv")
    assert rs.install_is_current(tmp_path / "venv") == (True, "Install marker is current.")


def test_dangling_venv_python_is_not_current(tmp_path):
    venv = tmp_path / "venv"
    python = _make_venv(venv)
    python.unlink()
    python.symlink_to(tmp_path / "removed-base-python")  # base interpreter gone
    ok, reason = rs.install_is_current(venv)
    assert not ok and "Python is missing" in reason


@pytest.mark.parametrize("override, expected", [
    ({"python_version": "3.9.1"}, "Python version changed"),
    ({"machine": "some-other-cpu"}, "CPU architecture changed"),
    ({"requirements_hash": "stale"}, "Pinned dependencies changed"),
])
def test_changed_environment_is_not_current(tmp_path, override, expected):
    _make_venv(tmp_path / "venv", **override)
    ok, reason = rs.install_is_current(tmp_path / "venv")
    assert not ok and expected in reason


def test_no_venv_fallback_install_stays_current(tmp_path):
    # Windows fallback installs into the system Python; there is no venv python.
    venv = tmp_path / "venv"
    venv.mkdir()
    rs.save_install_state(venv, python_path=Path(sys.executable))
    assert rs.install_is_current(venv)[0]


def test_pip_commands_use_only_pypi_on_mac_and_windows(monkeypatch):
    for plat in ("darwin", "win32"):
        monkeypatch.setattr(rs.sys, "platform", plat)
        cmds = rs.pip_install_commands("python")
        assert len(cmds) == 1
        assert "--extra-index-url" not in cmds[0] and "--index-url" not in cmds[0]
        assert "--retries" in cmds[0] and "--timeout" in cmds[0]


def test_pip_commands_install_torch_from_cpu_index_first_on_linux(monkeypatch):
    monkeypatch.setattr(rs.sys, "platform", "linux")
    torch_cmd, rest_cmd = rs.pip_install_commands("python")
    assert torch_cmd[torch_cmd.index("--index-url") + 1] == rs.TORCH_CPU_INDEX
    assert {a.split("==")[0] for a in torch_cmd if "==" in a} == {"torch", "torchvision", "torchaudio"}
    assert "--index-url" not in rest_cmd and "--extra-index-url" not in rest_cmd


def test_find_free_port_skips_a_busy_port():
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        port = busy.getsockname()[1]
        assert rs.find_free_port(port, attempts=5) != port


def test_install_reuses_a_matching_environment(tmp_path):
    venv = tmp_path / "venv"
    _make_venv(venv)
    assert install._can_reuse(venv)


def test_install_starts_fresh_when_a_requirement_was_removed(tmp_path):
    venv = tmp_path / "venv"
    _make_venv(venv, requirements=rs.requirement_names() + ["opencv-python-headless"])
    assert not install._can_reuse(venv)


def test_install_starts_fresh_for_markers_without_package_list(tmp_path):
    venv = tmp_path / "venv"
    _make_venv(venv)
    state = json.loads(rs.install_state_path(venv).read_text())
    del state["requirements"]
    rs.install_state_path(venv).write_text(json.dumps(state))
    assert not install._can_reuse(venv)


def test_install_reports_venv_unavailable(tmp_path, monkeypatch):
    def broken_create(self, env_dir):
        raise RuntimeError("ensurepip is not available")

    monkeypatch.setattr(install.venv.EnvBuilder, "create", broken_create)
    monkeypatch.setattr(sys, "argv", ["install.py", "--venv-dir", str(tmp_path / "venv")])
    assert install.main() == rs.VENV_UNAVAILABLE_EXIT


def test_failed_pip_install_keeps_existing_environment(tmp_path, monkeypatch):
    venv = tmp_path / "venv"
    python = _make_venv(venv, requirements_hash="old")  # needs an update, but reusable

    def failing_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(install.subprocess, "run", failing_run)
    monkeypatch.setattr(sys, "argv", ["install.py", "--venv-dir", str(venv)])
    assert install.main() == 1
    assert python.exists()  # not deleted
    assert json.loads(rs.install_state_path(venv).read_text())["requirements_hash"] == "old"  # not marked current


def test_uv_commands_match_the_pip_index_rules(monkeypatch):
    monkeypatch.setattr(rs.sys, "platform", "darwin")
    (cmd,) = rs.uv_install_commands("uv", "python")
    assert cmd[:4] == ["uv", "pip", "install", "--python"]
    assert "--compile-bytecode" in cmd and "sync" not in cmd
    assert "--index-url" not in cmd and "--extra-index-url" not in cmd
    monkeypatch.setattr(rs.sys, "platform", "linux")
    torch_cmd, rest_cmd = rs.uv_install_commands("uv", "python")
    assert torch_cmd[torch_cmd.index("--index-url") + 1] == rs.TORCH_CPU_INDEX
    assert {a.split("==")[0] for a in torch_cmd if "==" in a} == {"torch", "torchvision", "torchaudio"}
    assert "--index-url" not in rest_cmd and "--extra-index-url" not in rest_cmd


def test_bundled_uv_needs_an_executable_file(tmp_path, monkeypatch):
    monkeypatch.delenv(rs.UV_ENV_VAR, raising=False)
    assert rs.bundled_uv() is None
    uv = tmp_path / "uv"
    monkeypatch.setenv(rs.UV_ENV_VAR, str(uv))
    assert rs.bundled_uv() is None  # missing
    uv.write_text("")
    uv.chmod(0o755)
    assert rs.bundled_uv() == uv


def test_uv_cache_lives_beside_the_venv(tmp_path, monkeypatch):
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    env = rs.uv_env(tmp_path / ".venv")
    assert env["UV_CACHE_DIR"] == str(tmp_path / "uv-cache")
    assert env["UV_PYTHON_DOWNLOADS"] == "never"


def test_install_uses_uv_when_bundled(tmp_path, monkeypatch):
    uv = tmp_path / "uv"
    uv.write_text("")
    uv.chmod(0o755)
    monkeypatch.setenv(rs.UV_ENV_VAR, str(uv))
    venv = tmp_path / "venv"
    _make_venv(venv, requirements_hash="old")  # reusable, needs an update
    calls = []
    monkeypatch.setattr(install.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)))
    monkeypatch.setattr(sys, "argv", ["install.py", "--venv-dir", str(venv)])
    assert install.main() == 0
    assert calls and all(cmd[0] == str(uv) for cmd, _ in calls)
    assert all(kw["env"]["UV_CACHE_DIR"] for _, kw in calls)
    assert not any("pip" == cmd[1] and "--upgrade" in cmd for cmd, _ in calls)


def test_pip_path_restores_pip_in_a_venv_uv_made(tmp_path, monkeypatch):
    # An older launcher (no bundled uv) reusing a venv uv created: no pip there yet.
    monkeypatch.delenv(rs.UV_ENV_VAR, raising=False)
    venv = tmp_path / "venv"
    _make_venv(venv, requirements_hash="old")
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 1 if cmd[1:] == ["-m", "pip", "--version"] else 0)

    monkeypatch.setattr(install.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["install.py", "--venv-dir", str(venv)])
    assert install.main() == 0
    assert calls[1][1:] == ["-m", "ensurepip", "--upgrade"]
    assert calls[2][1:4] == ["-m", "pip", "install"]


@pytest.mark.parametrize("uv_failure", [
    OSError("blocked by antivirus"),
    subprocess.CalledProcessError(2, ["uv"]),
])
def test_uv_failure_falls_back_to_pip(tmp_path, monkeypatch, uv_failure):
    uv = tmp_path / "uv"
    uv.write_text("")
    uv.chmod(0o755)
    monkeypatch.setenv(rs.UV_ENV_VAR, str(uv))
    venv = tmp_path / "venv"
    _make_venv(venv, requirements_hash="old")
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if cmd[0] == str(uv):
            raise uv_failure
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(install.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["install.py", "--venv-dir", str(venv)])
    assert install.main() == 0
    assert calls[0][0] == str(uv)
    assert any(cmd[1:4] == ["-m", "pip", "install"] for cmd in calls[1:])
    assert json.loads(rs.install_state_path(venv).read_text())["requirements_hash"] != "old"


def test_uv_trusts_the_system_certificate_store(tmp_path, monkeypatch):
    monkeypatch.delenv("UV_SYSTEM_CERTS", raising=False)
    assert rs.uv_env(tmp_path / "venv")["UV_SYSTEM_CERTS"] == "1"
