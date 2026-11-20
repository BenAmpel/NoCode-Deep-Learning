//! Native launcher for NoCode-DL.
//!
//! It replaces the shell launchers' first job: find a Python to run the app
//! with. Instead of asking for an administrator password to install python.org
//! Python, it uses the bundled uv to put a private CPython 3.12 under the app's
//! user folder (no admin rights, nothing added to PATH or the registry). If uv
//! can't fetch it (offline, blocked), it falls back to a Python 3.11-3.13 that
//! is already installed. Then it hands over to the same Python bootstrap the
//! shell launchers start, which shows the progress window and installs the
//! packages.
//!
//! macOS: the launcher `exec`s the bootstrap, so the process macOS started
//! becomes the app, exactly as the bash launcher did.
//! Windows: the bootstrap runs as a child inside a Job Object that kills the
//! whole process tree if the launcher goes away.

#![cfg_attr(windows, windows_subsystem = "windows")]

use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{self, Command, Stdio};

const APP: &str = "NoCode-DL";
/// uv picks the newest 3.12 it knows; the bundled uv's version is pinned, so
/// this is the same build for every user of one release.
const PYTHON_REQUEST: &str = "3.12";
/// numpy 2.4 / scikit-learn 1.8 need >= 3.11; 3.14 is not verified.
const FALLBACK_VERSION_CHECK: &str =
    "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)";

struct Layout {
    uv: PathBuf,
    bootstrap: PathBuf,
    workdir: PathBuf,
    user_root: PathBuf,
    log: PathBuf,
}

impl Layout {
    fn python_dir(&self) -> PathBuf {
        self.user_root.join("python")
    }
    fn uv_cache(&self) -> PathBuf {
        self.user_root.join("uv-cache")
    }
}

fn smoke_mode() -> bool {
    env::var("NOCODE_DL_SMOKE_TEST").map(|v| !v.is_empty() && v != "0").unwrap_or(false)
}

fn open_log(path: &Path) -> Option<File> {
    if let Some(dir) = path.parent() {
        let _ = fs::create_dir_all(dir);
    }
    OpenOptions::new().create(true).append(true).open(path).ok()
}

fn log(layout: &Layout, msg: &str) {
    if let Some(mut f) = open_log(&layout.log) {
        let _ = writeln!(f, "[launcher] {msg}");
    }
}

fn fail(layout: &Layout, msg: &str) -> ! {
    log(layout, &format!("error: {msg}"));
    if !smoke_mode() {
        platform::error_dialog(msg);
    }
    process::exit(1);
}

/// A uv command with the app's settings, output appended to the log.
fn uv(layout: &Layout, args: &[&str]) -> Command {
    let mut cmd = Command::new(&layout.uv);
    cmd.args(args)
        .env("UV_PYTHON_INSTALL_DIR", layout.python_dir())
        .env("UV_CACHE_DIR", layout.uv_cache())
        .env("UV_SYSTEM_CERTS", "1")
        .env("UV_NO_PROGRESS", "1")
        .env("UV_HTTP_TIMEOUT", "60")
        .env("UV_HTTP_RETRIES", "5")
        .stdin(Stdio::null());
    platform::hide_console(&mut cmd);
    cmd
}

fn managed_python(layout: &Layout) -> Option<PathBuf> {
    let out = uv(layout, &["python", "find", "--managed-python", "--no-python-downloads", PYTHON_REQUEST])
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let path = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
    // `find` may answer with a Python outside the private folder; only ours counts.
    (path.is_file() && path.starts_with(layout.python_dir())).then_some(path)
}

fn install_managed_python(layout: &Layout) -> bool {
    let log_out = open_log(&layout.log);
    let mut cmd = uv(layout, &["python", "install", "--no-bin", PYTHON_REQUEST]);
    if cfg!(windows) {
        cmd.arg("--no-registry");
    }
    if let Some(f) = log_out {
        if let Ok(f2) = f.try_clone() {
            cmd.stdout(f).stderr(f2);
        }
    }
    matches!(cmd.status(), Ok(s) if s.success())
}

fn python_is_supported(python: &Path) -> bool {
    let mut cmd = Command::new(python);
    cmd.args(["-c", FALLBACK_VERSION_CHECK]).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
    platform::hide_console(&mut cmd);
    matches!(cmd.status(), Ok(s) if s.success())
}

fn find_on_path(name: &str) -> Option<PathBuf> {
    let exe = if cfg!(windows) { format!("{name}.exe") } else { name.to_string() };
    env::split_paths(&env::var_os("PATH")?).map(|d| d.join(&exe)).find(|p| p.is_file())
}

fn fallback_python() -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = platform::known_pythons();
    for name in platform::path_python_names() {
        if let Some(p) = find_on_path(name) {
            candidates.push(p);
        }
    }
    candidates.into_iter().find(|p| p.is_file() && python_is_supported(p))
}

fn choose_python(layout: &Layout) -> PathBuf {
    if let Some(p) = managed_python(layout) {
        return p;
    }
    log(layout, "Setting up a private Python with uv (one time)");
    if !smoke_mode() {
        platform::notify("Setting up Python for the first launch. This takes about a minute.");
    }
    if install_managed_python(layout) {
        if let Some(p) = managed_python(layout) {
            return p;
        }
    }
    log(layout, "uv could not set up Python; looking for an installed Python 3.11-3.13");
    fallback_python().unwrap_or_else(|| {
        fail(
            layout,
            "NoCode-DL could not download Python. Check your internet connection and open NoCode-DL again, \
             or install Python 3.12 from https://www.python.org.",
        )
    })
}

fn main() {
    let layout = platform::layout();
    // An unset HOME / LOCALAPPDATA would turn every path relative to the
    // current folder; stop before writing anything there.
    if !layout.user_root.is_absolute() {
        if !smoke_mode() {
            platform::error_dialog("Cannot find your user folder (HOME / LOCALAPPDATA is not set).");
        }
        eprintln!("[launcher] error: user folder not set");
        process::exit(1);
    }
    platform::preflight(&layout);
    if !layout.uv.is_file() {
        fail(&layout, &format!("The app is incomplete ({} is missing). Reinstall NoCode-DL.", layout.uv.display()));
    }
    let python = choose_python(&layout);
    log(&layout, &format!("Using Python: {}", python.display()));
    platform::run_bootstrap(&layout, &python);
}

#[cfg(target_os = "macos")]
mod platform {
    use super::*;
    use std::os::unix::process::CommandExt;

    pub fn layout() -> Layout {
        // Contents/MacOS/NoCode-DL -> Contents
        let exe = env::current_exe().and_then(fs::canonicalize).unwrap_or_default();
        let contents = exe.parent().and_then(Path::parent).map(Path::to_path_buf).unwrap_or_default();
        let home = PathBuf::from(env::var_os("HOME").unwrap_or_default());
        let resource_app = contents.join("Resources").join("app");
        Layout {
            uv: contents.join("Helpers").join("uv"),
            bootstrap: resource_app.join("runtime").join("bootstrap_macos.py"),
            workdir: resource_app,
            user_root: home.join("Library").join("Application Support").join(APP),
            log: home.join("Library").join("Logs").join(APP).join("launcher.log"),
        }
    }

    fn osascript(script: &str) {
        let _ = Command::new("/usr/bin/osascript")
            .args(["-e", script])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }

    fn quoted(text: &str) -> String {
        format!("\"{}\"", text.replace('\\', "\\\\").replace('"', "\\\""))
    }

    pub fn error_dialog(msg: &str) {
        osascript(&format!(
            "display dialog {} with title \"{APP}\" buttons {{\"OK\"}} default button \"OK\" with icon caution",
            quoted(msg)
        ));
    }

    pub fn notify(msg: &str) {
        osascript(&format!("display notification {} with title \"{APP}\"", quoted(msg)));
    }

    pub fn hide_console(_cmd: &mut Command) {}

    pub fn known_pythons() -> Vec<PathBuf> {
        vec![PathBuf::from("/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12")]
    }

    pub fn path_python_names() -> &'static [&'static str] {
        &["python3.12", "python3.13", "python3.11", "python3"]
    }

    /// PyTorch dropped Intel Macs. `hw.optional.arm64` is 1 on Apple Silicon
    /// even when this process runs under Rosetta.
    pub fn preflight(layout: &Layout) {
        let arm = Command::new("/usr/sbin/sysctl")
            .args(["-n", "hw.optional.arm64"])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim() == "1")
            .unwrap_or(false);
        if !arm {
            fail(layout, "NoCode-DL requires a Mac with Apple Silicon (M1 or later). PyTorch no longer supports Intel Macs.");
        }
    }

    /// Replace this process with the bootstrap, like the bash launcher's `exec`.
    /// `arch -arm64` keeps packages arm64 if the app was opened under Rosetta.
    pub fn run_bootstrap(layout: &Layout, python: &Path) -> ! {
        let mut cmd = Command::new("/usr/bin/arch");
        cmd.arg("-arm64").arg(python).arg(&layout.bootstrap).current_dir(&layout.workdir);
        cmd.env("NCDL_UV", &layout.uv).stdin(Stdio::null());
        if let Some(f) = open_log(&layout.log) {
            if let Ok(f2) = f.try_clone() {
                cmd.stdout(f).stderr(f2);
            }
        }
        let err = cmd.exec();
        fail(layout, &format!("Could not start NoCode-DL: {err}"));
    }
}

#[cfg(windows)]
mod platform {
    use super::*;
    use std::ffi::OsStr;
    use std::os::windows::ffi::OsStrExt;
    use std::os::windows::io::AsRawHandle;
    use std::os::windows::process::CommandExt;
    use windows_sys::Win32::Foundation::HANDLE;
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation, SetInformationJobObject,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };
    use windows_sys::Win32::System::Threading::CREATE_NO_WINDOW;
    use windows_sys::Win32::UI::WindowsAndMessaging::{MessageBoxW, MB_ICONERROR, MB_OK};

    pub fn layout() -> Layout {
        // {app}\NoCode-DL.exe, next to runtime\ and bin\uv.exe
        let exe = env::current_exe().unwrap_or_default();
        let app = exe.parent().map(Path::to_path_buf).unwrap_or_default();
        let user_root = PathBuf::from(env::var_os("LOCALAPPDATA").unwrap_or_default()).join(APP);
        Layout {
            uv: app.join("bin").join("uv.exe"),
            bootstrap: app.join("runtime").join("bootstrap_windows.py"),
            workdir: app,
            // bootstrap_windows.py rotates launcher.log, so this one is separate.
            log: user_root.join("logs").join("native-launcher.log"),
            user_root,
        }
    }

    fn wide(s: &str) -> Vec<u16> {
        OsStr::new(s).encode_wide().chain(std::iter::once(0)).collect()
    }

    pub fn error_dialog(msg: &str) {
        let (text, title) = (wide(msg), wide(APP));
        unsafe {
            MessageBoxW(std::ptr::null_mut(), text.as_ptr(), title.as_ptr(), MB_OK | MB_ICONERROR);
        }
    }

    /// The bootstrap's own window appears within seconds; Windows has no
    /// lightweight notification without extra dependencies.
    pub fn notify(_msg: &str) {}

    /// This is a GUI-subsystem program; without this, every console child
    /// (uv.exe, python.exe) would flash a console window.
    pub fn hide_console(cmd: &mut Command) {
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    pub fn known_pythons() -> Vec<PathBuf> {
        let local = PathBuf::from(env::var_os("LOCALAPPDATA").unwrap_or_default());
        let mut v: Vec<PathBuf> = ["Python312", "Python313", "Python311"]
            .iter()
            .map(|d| local.join("Programs").join("Python").join(d).join("python.exe"))
            .collect();
        v.push(PathBuf::from(r"C:\Python312\python.exe"));
        if let Some(pf) = env::var_os("ProgramFiles") {
            v.push(PathBuf::from(pf).join("Python312").join("python.exe"));
        }
        v
    }

    pub fn path_python_names() -> &'static [&'static str] {
        &["python"]
    }

    pub fn preflight(_layout: &Layout) {}

    /// A job that kills every process in it when its last handle closes, so
    /// Python and the server it starts never outlive the launcher.
    fn kill_on_close_job() -> Option<HANDLE> {
        unsafe {
            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if job.is_null() {
                return None;
            }
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            let ok = SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                &info as *const _ as *const _,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            );
            (ok != 0).then_some(job)
        }
    }

    pub fn run_bootstrap(layout: &Layout, python: &Path) -> ! {
        let mut cmd = Command::new(python);
        cmd.arg(&layout.bootstrap).current_dir(&layout.workdir);
        cmd.env("NCDL_UV", &layout.uv).stdin(Stdio::null());
        hide_console(&mut cmd);
        if let Some(f) = open_log(&layout.log) {
            if let Ok(f2) = f.try_clone() {
                cmd.stdout(f).stderr(f2);
            }
        }
        let job = kill_on_close_job();
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(err) => fail(layout, &format!("Could not start NoCode-DL: {err}")),
        };
        if let Some(job) = job {
            unsafe {
                AssignProcessToJobObject(job, child.as_raw_handle() as HANDLE);
            }
        }
        // bootstrap_windows.py reports its own errors; its exit code is the app's.
        let code = child.wait().ok().and_then(|s| s.code()).unwrap_or(1);
        process::exit(code);
    }
}
