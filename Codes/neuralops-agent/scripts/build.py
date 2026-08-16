#!/usr/bin/env python3
"""
Unified build and packaging script for NeuralOps GPU Provider Agent and Desktop GUI.

Usage:
    python scripts/build.py                    # auto-detect OS and build agent + GUI
    python scripts/build.py --cli-only         # build only Python CLI binary
    python scripts/build.py --gui-only         # build only Electron GUI package
    python scripts/build.py --platform macos   # force macOS bundle
    python scripts/build.py --platform windows # force Windows bundle
    python scripts/build.py --platform linux   # force Linux bundle
    python scripts/build.py --all              # build all targets (for CI)

Produces:
    macOS:   dist/NeuralOps-Agent-{version}.dmg, dist/neuralops-agent.app
    Windows: dist/NeuralOps-Agent-{version}-Setup.exe, dist/neuralops-agent.exe
    Linux:   dist/NeuralOps-Agent-{version}.AppImage, dist/neuralops-agent
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("neuralops.build")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GUI_ROOT = PROJECT_ROOT / "neuralops-gui"
DIST_DIR = PROJECT_ROOT / "dist"
VERSION = "1.0.0"


def check_prerequisites():
    """Ensure essential tools and directories exist."""
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    resources_dir = PROJECT_ROOT / "resources"
    resources_dir.mkdir(parents=True, exist_ok=True)


def build_pyinstaller_cli():
    """Build standalone executable using PyInstaller."""
    logger.info("=" * 60)
    logger.info("Building Standalone Python CLI Agent with PyInstaller")
    logger.info("=" * 60)

    spec_file = PROJECT_ROOT / "pyinstaller.spec"
    if spec_file.exists():
        cmd = [
            sys.executable,
            "-m",
            "PyInstaller",
            "--clean",
            "--noconfirm",
            str(spec_file),
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "PyInstaller",
            "--clean",
            "--noconfirm",
            "--onedir",
            "--name",
            "neuralops-agent",
            "--add-data",
            f"agent-config.yaml{os.pathsep}.",
            str(PROJECT_ROOT / "agent" / "main.py"),
        ]

    try:
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
        logger.info("✅ PyInstaller agent build completed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error("❌ PyInstaller build failed: %s", e)
        raise


def build_gui():
    """Build React frontend bundle and package Electron app."""
    if not GUI_ROOT.exists():
        logger.warning("GUI directory %s not found, skipping GUI build.", GUI_ROOT)
        return

    logger.info("=" * 60)
    logger.info("Building Desktop GUI (Webpack & Electron)")
    logger.info("=" * 60)

    # 1. Build Webpack bundle
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    try:
        logger.info("Running npm run build in %s...", GUI_ROOT)
        subprocess.run([npm_cmd, "run", "build"], check=True, cwd=str(GUI_ROOT))
    except Exception as e:
        logger.error("❌ Webpack build failed: %s", e)
        raise

    # 2. Package with Electron Builder if installed
    try:
        logger.info("Packaging Desktop GUI with electron-builder...")
        subprocess.run([npm_cmd, "run", "dist"], check=True, cwd=str(GUI_ROOT))
        logger.info("✅ Electron GUI packaging completed.")
    except Exception as e:
        logger.warning("⚠️ Electron-builder packaging step skipped or failed: %s", e)


def build_macos():
    """Build macOS .app bundle and .dmg installer."""
    logger.info("=" * 60)
    logger.info("Building macOS .dmg installer")
    logger.info("=" * 60)

    build_pyinstaller_cli()
    app_bundle = DIST_DIR / "neuralops-agent.app"
    dmg_path = DIST_DIR / f"NeuralOps-Agent-{VERSION}.dmg"

    if app_bundle.exists():
        if shutil.which("create-dmg"):
            icon_path = PROJECT_ROOT / "resources" / "icon.icns"
            cmd = [
                "create-dmg",
                "--volname",
                f"NeuralOps Agent {VERSION}",
                "--window-pos",
                "200",
                "120",
                "--window-size",
                "600",
                "400",
                "--icon-size",
                "100",
                "--icon",
                "neuralops-agent.app",
                "140",
                "170",
                "--hide-extension",
                "neuralops-agent.app",
                "--app-drop-link",
                "400",
                "170",
                str(dmg_path),
                str(app_bundle),
            ]
            if icon_path.exists():
                cmd.extend(["--volicon", str(icon_path)])
            try:
                subprocess.run(cmd, check=True)
                logger.info("✅ macOS DMG created via create-dmg: %s", dmg_path)
                return dmg_path
            except subprocess.CalledProcessError:
                logger.warning("create-dmg failed, falling back to hdiutil...")

        # Fallback to hdiutil
        try:
            subprocess.run(
                [
                    "hdiutil",
                    "create",
                    "-volname",
                    f"NeuralOps Agent {VERSION}",
                    "-srcfolder",
                    str(app_bundle),
                    "-ov",
                    "-format",
                    "UDZO",
                    str(dmg_path),
                ],
                check=True,
            )
            logger.info("✅ macOS DMG created via hdiutil: %s", dmg_path)
            return dmg_path
        except Exception as e:
            logger.warning("hdiutil creation failed: %s", e)

    return None


def build_windows():
    """Build Windows .exe standalone binary and NSIS setup installer."""
    logger.info("=" * 60)
    logger.info("Building Windows .exe installer")
    logger.info("=" * 60)

    build_pyinstaller_cli()
    exe_path = DIST_DIR / "neuralops-agent.exe"

    if shutil.which("makensis"):
        nsi_content = f"""!include "MUI2.nsh"

Name "NeuralOps GPU Agent {VERSION}"
OutFile "{DIST_DIR}\\NeuralOps-Agent-{VERSION}-Setup.exe"
InstallDir "$PROGRAMFILES\\NeuralOps Agent"
RequestExecutionLevel admin

!insertmacro MUI_PAGE_WELCOME
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
        tmp_nsi = PROJECT_ROOT / "scripts" / "_installer.nsi"
        tmp_nsi.write_text(nsi_content, encoding="utf-8")
        try:
            subprocess.run(["makensis", str(tmp_nsi)], check=True)
            logger.info("✅ Windows NSIS setup created.")
        finally:
            if tmp_nsi.exists():
                tmp_nsi.unlink()
    else:
        logger.info("ℹ️ NSIS not found; creating portable zip package...")
        zip_path = DIST_DIR / f"NeuralOps-Agent-{VERSION}-windows-x64.zip"
        if exe_path.exists():
            shutil.make_archive(
                str(zip_path.with_suffix("")),
                "zip",
                DIST_DIR,
                "neuralops-agent.exe",
            )
            logger.info("✅ Portable zip package created: %s", zip_path)


def build_linux():
    """Build Linux standalone binary, AppDir, and AppImage."""
    logger.info("=" * 60)
    logger.info("Building Linux AppImage & binaries")
    logger.info("=" * 60)

    build_pyinstaller_cli()
    appdir = DIST_DIR / "NeuralOps-Agent.AppDir"
    appdir.mkdir(parents=True, exist_ok=True)

    pyinst_output = DIST_DIR / "neuralops-agent"
    if pyinst_output.exists():
        shutil.copytree(pyinst_output, appdir / "usr" / "bin", dirs_exist_ok=True)

    # AppRun
    apprun = appdir / "AppRun"
    apprun.write_text("""#!/bin/bash
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/bin/neuralops-agent" "$@"
""", encoding="utf-8")
    apprun.chmod(0o755)

    # .desktop
    desktop = appdir / "neuralops-agent.desktop"
    desktop.write_text(f"""[Desktop Entry]
Name=NeuralOps GPU Agent
Comment=Decentralized GPU Compute Provider Daemon
Exec=neuralops-agent
Icon=neuralops-agent
Type=Application
Categories=Utility;System;
""", encoding="utf-8")

    appimage_path = DIST_DIR / f"NeuralOps-Agent-{VERSION}.AppImage"
    if shutil.which("appimagetool"):
        try:
            subprocess.run(["appimagetool", str(appdir), str(appimage_path)], check=True)
            logger.info("✅ Linux AppImage created: %s", appimage_path)
            return appimage_path
        except Exception as e:
            logger.warning("appimagetool failed: %s", e)

    return None


def main():
    parser = argparse.ArgumentParser(description="NeuralOps Unified Build Script")
    parser.add_argument(
        "--platform",
        choices=["macos", "windows", "linux"],
        help="Target platform (default: auto-detect)",
    )
    parser.add_argument("--cli-only", action="store_true", help="Build only Python CLI executable")
    parser.add_argument("--gui-only", action="store_true", help="Build only Electron GUI package")
    parser.add_argument("--all", action="store_true", help="Build all targets")
    args = parser.parse_args()

    check_prerequisites()

    if args.gui_only:
        build_gui()
        return

    if args.cli_only:
        build_pyinstaller_cli()
        return

    # Full build: CLI + GUI
    platform_map = {
        "darwin": "macos",
        "win32": "windows",
        "linux": "linux",
    }
    target_platform = args.platform or platform_map.get(sys.platform, "linux")

    logger.info("Building for target platform: %s (Version: %s)", target_platform, VERSION)

    if target_platform == "macos":
        build_macos()
    elif target_platform == "windows":
        build_windows()
    else:
        build_linux()

    if not args.cli_only:
        try:
            build_gui()
        except Exception as e:
            logger.warning("GUI build skipped or encountered non-critical error: %s", e)

    logger.info("=" * 60)
    logger.info("Build pipeline completed. Artifacts located in: %s", DIST_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
