Unsigned macOS build for Apple Silicon (M1 or newer). StreetLab is a portfolio/learning project, not a production self-driving system.

**First launch.** macOS will say it cannot verify the app because it is not notarized. Right-click `StreetLab.app`, choose **Open**, then **Open** again. Or remove the quarantine flag: `xattr -dr com.apple.quarantine StreetLab.app`.

**Verify the download.** With the `.sha256` file next to the zip: `shasum -a 256 -c StreetLab-*.zip.sha256`.

It opens on Nob Hill from a bundled OpenStreetMap extract with no network needed; type any address to load it live.
