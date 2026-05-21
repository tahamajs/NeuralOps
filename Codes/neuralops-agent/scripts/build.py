#!/usr/bin/env python3
"""
Unified build script for NeuralOps GPU Provider Agent.

Usage:
    python scripts/build.py                    # auto-detect OS
    python scripts/build.py --platform macos   # force macOS
    python scripts/build.py --platform windows # force Windows
    python scripts/build.py --platform linux   # force Linux
    python scripts/build.py --all              # build all platform packages (CI only)

Produces:
    macOS:   dist/NeuralOps-Agent-{version}.dmg
    Windows: dist/NeuralOps-Agent-{version}-Setup.exe
    Linux:   dist/NeuralOps-Agent-{version}.AppImage
"""
import argparse, os, shutil, subprocess, sys, tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = PROJECT_ROOT / 'dist'
VERSION = '1.0.0'   # sync with setup.py

# ─────────────────────────────────────────────────────────────────
# macOS .DMG builder
# ─────────────────────────────────────────────────────────────────
def build_macos():
    print("=" * 60)
    print("Building macOS .dmg installer")
    print("=" * 60)

    # Step 1: PyInstaller → .app bundle
    subprocess.run([
        sys.executable, '-m', 'PyInstaller',
        '--clean', '--noconfirm',
        str(PROJECT_ROOT / 'pyinstaller.spec'),
    ], check=True, cwd=str(PROJECT_ROOT))

    app_bundle = DIST_DIR / 'neuralops-agent.app'

    # Step 2: Create .dmg using create-dmg (or hdiutil fallback)
    dmg_path = DIST_DIR / f'NeuralOps-Agent-{VERSION}.dmg'

    # Try create-dmg first (gives a nicer DMG with background)
    if shutil.which('create-dmg'):
        subprocess.run([
            'create-dmg',
            '--volname', f'NeuralOps Agent {VERSION}',
            '--volicon', str(PROJECT_ROOT / 'resources' / 'icon.icns'),
            '--window-pos', '200', '120',
            '--window-size', '600', '400',
            '--icon-size', '100',
            '--icon', 'neuralops-agent.app', '140', '170',
            '--hide-extension', 'neuralops-agent.app',
            '--app-drop-link', '400', '170',
            str(dmg_path),
            str(app_bundle),
        ], check=True)
    else:
        # Fallback: plain hdiutil
        subprocess.run([
            'hdiutil', 'create',
            '-volname', f'NeuralOps Agent {VERSION}',
            '-srcfolder', str(app_bundle),
            '-ov', '-format', 'UDZO',
            str(dmg_path),
        ], check=True)

    print(f"✅ macOS DMG created: {dmg_path}")
    return dmg_path


# ─────────────────────────────────────────────────────────────────
# Windows .exe installer builder (NSIS)
# ─────────────────────────────────────────────────────────────────
def build_windows():
    print("=" * 60)
    print("Building Windows .exe installer")
    print("=" * 60)

    # Step 1: PyInstaller → onefile .exe
    subprocess.run([
        sys.executable, '-m', 'PyInstaller',
        '--clean', '--noconfirm',
        '--onefile',
        '--name', 'neuralops-agent',
        '--icon', str(PROJECT_ROOT / 'resources' / 'icon.ico'),
        '--add-data', f"agent-config.yaml{os.pathsep}.",
        str(PROJECT_ROOT / 'agent' / 'main.py'),
    ], check=True, cwd=str(PROJECT_ROOT))

    exe_path = DIST_DIR / 'neuralops-agent.exe'

    # Step 2: Build NSIS installer
    nsi_script = PROJECT_ROOT / 'scripts' / 'installer.nsi'
    if shutil.which('makensis'):
        # Write a temporary NSIS script with correct paths
        nsi_content = f"""!include "MUI2.nsh"

Name "NeuralOps GPU Agent {VERSION}"
OutFile "{DIST_DIR}\\NeuralOps-Agent-{VERSION}-Setup.exe"
InstallDir "$PROGRAMFILES\\NeuralOps Agent"
RequestExecutionLevel admin

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "{PROJECT_ROOT}\\LICENSE"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_LANGUAGE "English"

Section "Install"
    SetOutPath "$INSTDIR"
    File "{exe_path}"
    File "{PROJECT_ROOT}\\agent-config.yaml"
    CreateDirectory "$INSTDIR\\logs"
    CreateDirectory "$INSTDIR\\jobs"
    WriteUninstaller "$INSTDIR\\uninstall.exe"
    CreateShortCut "$DESKTOP\\NeuralOps Agent.lnk" "$INSTDIR\\neuralops-agent.exe"
SectionEnd

Section "Uninstall"
    Delete "$INSTDIR\\neuralops-agent.exe"
    Delete "$INSTDIR\\agent-config.yaml"
    Delete "$INSTDIR\\uninstall.exe"
    Delete "$DESKTOP\\NeuralOps Agent.lnk"
    RMDir /r "$INSTDIR\\logs"
    RMDir /r "$INSTDIR\\jobs"
    RMDir "$INSTDIR"
SectionEnd
"""
        tmp_nsi = PROJECT_ROOT / 'scripts' / '_installer.nsi'
        tmp_nsi.write_text(nsi_content)
        subprocess.run(['makensis', str(tmp_nsi)], check=True)
        tmp_nsi.unlink()
    else:
        print("WARNING: NSIS not found. Only standalone .exe produced. Install NSIS:")
        print("  winget install NSIS.NSIS")
        # Still produce a zip for distribution
        zip_path = DIST_DIR / f'NeuralOps-Agent-{VERSION}-portable.zip'
        shutil.make_archive(str(zip_path.with_suffix('')), 'zip', DIST_DIR, 'neuralops-agent.exe')

    print(f"✅ Windows installer created in {DIST_DIR}")
    return DIST_DIR / f'NeuralOps-Agent-{VERSION}-Setup.exe'


# ─────────────────────────────────────────────────────────────────
# Linux .AppImage builder
# ─────────────────────────────────────────────────────────────────
def build_linux():
    print("=" * 60)
    print("Building Linux .AppImage")
    print("=" * 60)

    # Step 1: PyInstaller → directory of files
    subprocess.run([
        sys.executable, '-m', 'PyInstaller',
        '--clean', '--noconfirm',
        str(PROJECT_ROOT / 'pyinstaller.spec'),
    ], check=True, cwd=str(PROJECT_ROOT))

    # Step 2: Build AppDir structure
    appdir = DIST_DIR / 'NeuralOps-Agent.AppDir'
    appdir.mkdir(parents=True, exist_ok=True)

    # Copy PyInstaller output into AppDir
    pyinst_output = DIST_DIR / 'neuralops-agent'
    shutil.copytree(pyinst_output, appdir / 'usr' / 'bin', dirs_exist_ok=True)

    # Create AppRun
    apprun = appdir / 'AppRun'
    apprun.write_text("""#!/bin/bash
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/bin/neuralops-agent" "$@"
""")
    apprun.chmod(0o755)

    # Create .desktop file
    desktop = appdir / 'neuralops-agent.desktop'
    desktop.write_text(f"""[Desktop Entry]
Name=NeuralOps GPU Agent
Comment=Share your GPU and earn money
Exec=neuralops-agent
Icon=neuralops-agent
Type=Application
Categories=Utility;
""")

    # Copy icon
    icon_src = PROJECT_ROOT / 'resources' / 'neuralops-agent.png'
    if icon_src.exists():
        shutil.copy2(icon_src, appdir / 'neuralops-agent.png')
        shutil.copy2(icon_src, appdir / '.DirIcon')

    # Step 3: Build AppImage using appimagetool
    appimage_path = DIST_DIR / f'NeuralOps-Agent-{VERSION}.AppImage'

    if shutil.which('appimagetool'):
        subprocess.run([
            'appimagetool', str(appdir), str(appimage_path),
        ], check=True)
    else:
        # Download appimagetool on the fly
        appimagetool_url = (
            'https://github.com/AppImage/AppImageKit/releases/download/continuous/'
            'appimagetool-x86_64.AppImage'
        )
        tool_path = PROJECT_ROOT / 'scripts' / 'appimagetool.AppImage'
        if not tool_path.exists():
            import urllib.request
            print("Downloading appimagetool...")
            urllib.request.urlretrieve(appimagetool_url, tool_path)
            tool_path.chmod(0o755)
        subprocess.run([str(tool_path), str(appdir), str(appimage_path)], check=True)

    print(f"✅ Linux AppImage created: {appimage_path}")

    # Optional: also build .deb
    _maybe_build_deb(appdir, appimage_path)

    return appimage_path


def _maybe_build_deb(appdir, appimage_path):
    """Optionally produce a .deb from the AppDir (if dpkg-deb is available)."""
    if not shutil.which('dpkg-deb'):
        return
    deb_root = DIST_DIR / 'deb_root'
    deb_root.mkdir(exist_ok=True)
    deb_usr = deb_root / 'usr'
    shutil.copytree(appdir / 'usr' / 'bin', deb_usr / 'bin', dirs_exist_ok=True)
    shutil.copytree(appdir / 'usr' / 'share', deb_usr / 'share', dirs_exist_ok=True)

    # Control file
    control_dir = deb_root / 'DEBIAN'
    control_dir.mkdir(exist_ok=True)
    (control_dir / 'control').write_text(f"""Package: neuralops-agent
Version: {VERSION}
Section: utils
Priority: optional
Architecture: amd64
Maintainer: NeuralOps <support@neuralops.com>
Description: NeuralOps GPU Provider Agent
 Share your GPU and earn money by renting it out on the NeuralOps platform.
""")
    deb_path = DIST_DIR / f'neuralops-agent_{VERSION}_amd64.deb'
    subprocess.run(['dpkg-deb', '--build', str(deb_root), str(deb_path)], check=True)
    print(f"   Also created: {deb_path}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Build NeuralOps Agent packages')
    parser.add_argument('--platform', choices=['macos', 'windows', 'linux'],
                        help='Target platform (default: auto-detect)')
    parser.add_argument('--all', action='store_true',
                        help='Build all platform packages')
    args = parser.parse_args()

    DIST_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        print("Building all platform packages via platform-specific GitHub Actions runners.")
        print("This is meant to be called from CI. Locally, use --platform.")
        # When called from CI on a specific runner, detect OS and build
        platform = sys.platform
        if platform == 'darwin':
            build_macos()
        elif platform == 'win32':
            build_windows()
        else:
            build_linux()
    elif args.platform:
        {'macos': build_macos, 'windows': build_windows, 'linux': build_linux}[args.platform]()
    else:
        # Auto-detect
        platform = sys.platform
        if platform == 'darwin':
            build_macos()
        elif platform == 'win32':
            build_windows()
        else:
            build_linux()


if __name__ == '__main__':
    main()