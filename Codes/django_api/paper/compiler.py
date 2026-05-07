"""
LaTeX compilation engine – analogous to Overleaf's CLSI.

Handles:
- Running pdflatex/xelatex/lualatex
- Log parsing for errors, warnings, and bad boxes
- Synctex generation
- Output file collection
- Timeout enforcement
"""
import subprocess
import tempfile
import os
import re
from pathlib import Path
from typing import Tuple, Optional
from django.conf import settings


class LatexCompiler:
    """Compiles a LaTeX project and returns structured results."""

    def __init__(self, project_files: dict, root_file: str, compiler: str = 'pdflatex',
                 draft_mode: bool = False, stop_on_first_error: bool = False,
                 timeout: int = 60):
        """
        Args:
            project_files: dict of filename -> content
            root_file: e.g. 'main.tex'
            compiler: 'pdflatex', 'xelatex', or 'lualatex'
            draft_mode: pass 'draft' option
            stop_on_first_error: halt on first error
            timeout: max seconds for compilation
        """
        self.project_files = project_files
        self.root_file = root_file
        self.compiler = compiler
        self.draft_mode = draft_mode
        self.stop_on_first_error = stop_on_first_error
        self.timeout = timeout
        self._tmpdir = None

    def compile(self) -> dict:
        """Run compilation and return structured result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._tmpdir = tmpdir
            self._write_project_files(tmpdir)

            tex_path = os.path.join(tmpdir, self.root_file)
            pdf_path = os.path.join(tmpdir, 'output.pdf')
            log_path = os.path.join(tmpdir, 'output.log')

            # Build command
            cmd = [self.compiler]
            cmd.extend(['-interaction=nonstopmode'])
            cmd.extend(['-output-directory', tmpdir])
            cmd.extend(['-jobname', 'output'])
            if self.draft_mode:
                cmd.append('-draftmode')
            if self.stop_on_first_error:
                cmd.append('-halt-on-error')
            cmd.append(tex_path)

            # If .bib files exist, we should run bibtex too. For simplicity,
            # we assume the user handles bibliography in the .tex file.
            # A production version would detect .bib files and run the
            # pdflatex → bibtex → pdflatex → pdflatex cycle.

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=tmpdir,
                )
            except subprocess.TimeoutExpired:
                return {
                    'success': False,
                    'error': f'Compilation timed out after {self.timeout}s',
                    'log': '',
                    'pdf_path': None,
                    'errors': 0,
                    'warnings': 0,
                    'status': 'timed_out',
                }
            except FileNotFoundError:
                return {
                    'success': False,
                    'error': f"Compiler '{self.compiler}' not found. Install TeX Live.",
                    'log': '',
                    'pdf_path': None,
                    'errors': 0,
                    'warnings': 0,
                    'status': 'failed',
                }

            # Parse the log
            log_text = self._read_log(log_path, result)
            errors, warnings = self._parse_log(log_text)
            pdf_exists = os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0

            if not pdf_exists:
                return {
                    'success': False,
                    'error': 'PDF generation failed – check log for details',
                    'log': log_text,
                    'pdf_path': None,
                    'errors': errors,
                    'warnings': warnings,
                    'status': 'failed',
                }

            return {
                'success': True,
                'error': None,
                'log': log_text,
                'pdf_path': pdf_path,
                'errors': errors,
                'warnings': warnings,
                'status': 'completed',
            }

    def _write_project_files(self, tmpdir: str):
        """Write all project files to the temporary directory."""
        for filename, content in self.project_files.items():
            filepath = os.path.join(tmpdir, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            # If content is bytes, write binary; otherwise text
            if isinstance(content, bytes):
                with open(filepath, 'wb') as f:
                    f.write(content)
            else:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)

    def _read_log(self, log_path: str, result: subprocess.CompletedProcess) -> str:
        """Combine compiler stdout/stderr and the .log file."""
        log_parts = []
        if result.stdout:
            log_parts.append(result.stdout)
        if result.stderr:
            log_parts.append(result.stderr)
        if os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                log_parts.append(f.read())
        return '\n'.join(log_parts)

    def _parse_log(self, log_text: str) -> Tuple[int, int]:
        """
        Parse the LaTeX log for errors and warnings.

        Returns (error_count, warning_count).

        Detects:
        - Lines starting with '!' (errors)
        - Lines containing 'Error:' 
        - Lines containing 'Warning:'
        - 'LaTeX Warning:' pattern
        - 'Overfull' / 'Underfull' boxes (warnings)
        - 'No file' messages (warnings)
        """
        error_count = 0
        warning_count = 0

        error_patterns = [
            r'^! ',                          # Standard TeX error
            r'^.*Error:',                    # Generic error
            r'Fatal error',                  # Fatal error
        ]
        warning_patterns = [
            r'Warning:',                    # Generic warning
            r'LaTeX Warning:',              # LaTeX specific
            r'Package \w+ Warning:',        # Package warnings
            r'Overfull \\hbox',             # Overfull hbox
            r'Underfull \\hbox',            # Underfull hbox
            r'Overfull \\vbox',             # Overfull vbox
            r'Underfull \\vbox',            # Underfull vbox
            r'No file ',                    # Missing file references
            r'Citation .* undefined',       # Missing citations
            r'Reference .* undefined',      # Missing references
        ]

        for line in log_text.split('\n'):
            for pattern in error_patterns:
                if re.search(pattern, line):
                    error_count += 1
                    break
            else:
                for pattern in warning_patterns:
                    if re.search(pattern, line):
                        warning_count += 1
                        break

        return error_count, warning_count