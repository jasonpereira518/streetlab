use std::sync::Mutex;
use tauri::menu::{Menu, PredefinedMenuItem, Submenu};
use tauri::{AppHandle, Manager, RunEvent, State};
use tauri_plugin_shell::process::CommandEvent;
use tauri_plugin_shell::ShellExt;
use tokio::sync::watch;

/// Metadata the frontend can query over IPC. The backend simulator lives in a
/// separate process and speaks WebSocket — this shell deliberately knows
/// nothing about the simulation schema.
#[derive(serde::Serialize)]
struct ShellInfo {
    name: &'static str,
    version: &'static str,
    platform: &'static str,
}

#[tauri::command]
fn shell_info() -> ShellInfo {
    ShellInfo {
        name: "StreetLab",
        version: env!("CARGO_PKG_VERSION"),
        platform: std::env::consts::OS,
    }
}

/// The sidecar's `STREETLAB_READY {...}` handshake, mirrored verbatim to the
/// frontend — field names match `src/net/wsClient.ts`'s `BackendHandshake`.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct BackendHandshake {
    ws: String,
    http: String,
    pid: u32,
    protocol: u32,
}

type HandshakeResult = Result<BackendHandshake, String>;

const READY_PREFIX: &str = "STREETLAB_READY ";
const READY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15);
const TERM_GRACE_PERIOD: std::time::Duration = std::time::Duration::from_millis(500);

/// The sidecar's own PID, as it reported via `os.getpid()` in its handshake —
/// **not** `CommandChild::pid()`, which is the PyInstaller one-file
/// bootloader's PID. Killing the bootloader alone can leave the real server
/// running; the bootloader waits on this exact PID and exits once it does,
/// so signalling it directly brings both down. Verified manually against the
/// packaged binary (two distinct PIDs; the real one's exit takes the
/// bootloader with it) before writing this.
struct RealPid(Mutex<Option<u32>>);

/// Waits for the sidecar's handshake and resolves once, ever — repeat calls
/// (there shouldn't be any; the frontend calls this once at boot) get the
/// same cached result. `?mock=1` and `?backend=` both skip this command
/// entirely, so a slow or absent sidecar never blocks those paths.
#[tauri::command]
async fn backend_url(
    state: State<'_, watch::Receiver<Option<HandshakeResult>>>,
) -> HandshakeResult {
    let mut rx = state.inner().clone();
    let wait = async {
        loop {
            if let Some(result) = rx.borrow().clone() {
                return result;
            }
            if rx.changed().await.is_err() {
                return Err("sidecar task ended without ever reporting a result".into());
            }
        }
    };
    match tokio::time::timeout(READY_TIMEOUT, wait).await {
        Ok(result) => result,
        Err(_) => Err("timed out waiting for the simulator to start".into()),
    }
}

/// Spawn the Python sidecar and keep draining its output for the app's
/// lifetime, so neither pipe ever backs up. Parses the `STREETLAB_READY`
/// line into `BackendHandshake` for `backend_url()`, and records the real
/// PID for teardown at `RunEvent::Exit`.
fn spawn_sidecar(app: &AppHandle) {
    let (tx, rx) = watch::channel::<Option<HandshakeResult>>(None);
    app.manage(rx);

    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        let command = match app.shell().sidecar("streetlab-server") {
            Ok(cmd) => cmd,
            Err(err) => {
                let _ = tx.send(Some(Err(format!(
                    "could not resolve the sidecar binary: {err}"
                ))));
                return;
            }
        };

        // `--source osm` is load-bearing, not a preference: the CLI defaults to
        // `synthetic`, so omitting it shipped an app that could only ever drive
        // the 6-road placeholder grid — the address search box, real OSM
        // streets and the bundled offline extract were all unreachable in the
        // packaged build even though every one of them was implemented and
        // tested. The bundled extract is what keeps this safe with no network:
        // the default Nob Hill scene resolves from `_MEIPASS/bundled` without
        // touching Nominatim or Overpass.
        let (mut events, child) = match command
            .args(["serve", "--port", "0", "--source", "osm"])
            .spawn()
        {
            Ok(pair) => pair,
            Err(err) => {
                let _ = tx.send(Some(Err(format!("failed to spawn the sidecar: {err}"))));
                return;
            }
        };

        // Kept alive for the app's lifetime: dropping `child` drops its piped
        // stdin, which the sidecar's own watchdog treats as EOF — closing it
        // this early would kill the sidecar before it even finished starting.
        app.manage(Mutex::new(Some(child)));

        let mut ready_sent = false;
        while let Some(event) = events.recv().await {
            match event {
                CommandEvent::Stdout(bytes) => {
                    let line = String::from_utf8_lossy(&bytes);
                    let Some(payload) = line.strip_prefix(READY_PREFIX) else {
                        continue;
                    };
                    let result = serde_json::from_str::<BackendHandshake>(payload.trim())
                        .map_err(|err| format!("malformed handshake from sidecar: {err}"));
                    if let Ok(handshake) = &result {
                        if let Some(state) = app.try_state::<RealPid>() {
                            *state.0.lock().unwrap() = Some(handshake.pid);
                        }
                    }
                    ready_sent = true;
                    let _ = tx.send(Some(result));
                }
                CommandEvent::Stderr(bytes) => {
                    // The sidecar's human-readable startup lines and uvicorn's
                    // own warning-level logging both land here — useful for
                    // debugging, never parsed for control flow.
                    let line = String::from_utf8_lossy(&bytes);
                    eprintln!("[streetlab-server] {}", line.trim_end());
                }
                CommandEvent::Terminated(payload) => {
                    if !ready_sent {
                        let _ = tx.send(Some(Err(format!(
                            "simulator exited before starting (code {:?})",
                            payload.code
                        ))));
                    }
                    break;
                }
                CommandEvent::Error(err) => {
                    if !ready_sent {
                        let _ = tx.send(Some(Err(format!("simulator process error: {err}"))));
                    }
                }
                _ => {}
            }
        }
    });
}

/// SIGTERM the sidecar's real PID, give it a grace period to exit cleanly
/// (its own `SimLoop.stop()` / FastAPI lifespan shutdown), then SIGKILL. A
/// signal to an already-exited PID is a harmless no-op (`ESRCH`).
fn teardown_sidecar(pid: u32) {
    unsafe {
        libc::kill(pid as libc::pid_t, libc::SIGTERM);
    }
    std::thread::sleep(TERM_GRACE_PERIOD);
    unsafe {
        libc::kill(pid as libc::pid_t, libc::SIGKILL);
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(RealPid(Mutex::new(None)))
        // Replace the default File/Edit/View/Window/Help menu bar with a
        // minimal app submenu. macOS always renders the app-name menu, so this
        // is as close to "no menu" as the platform allows while keeping the
        // standard Cmd+H / Cmd+Q shortcuts alive.
        .menu(|handle| {
            let app = Submenu::with_items(
                handle,
                "StreetLab",
                true,
                &[
                    &PredefinedMenuItem::hide(handle, None)?,
                    &PredefinedMenuItem::separator(handle)?,
                    &PredefinedMenuItem::quit(handle, None)?,
                ],
            )?;
            Menu::with_items(handle, &[&app])
        })
        .setup(|app| {
            spawn_sidecar(app.handle());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![shell_info, backend_url])
        .build(tauri::generate_context!())
        .expect("error while running StreetLab")
        .run(|app_handle, event| {
            if let RunEvent::Exit = event {
                if let Some(state) = app_handle.try_state::<RealPid>() {
                    if let Some(pid) = state.0.lock().unwrap().take() {
                        teardown_sidecar(pid);
                    }
                }
            }
        });
}
