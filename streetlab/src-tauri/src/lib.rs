use tauri::menu::{Menu, PredefinedMenuItem, Submenu};

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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
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
        .invoke_handler(tauri::generate_handler![shell_info])
        .run(tauri::generate_context!())
        .expect("error while running StreetLab");
}
