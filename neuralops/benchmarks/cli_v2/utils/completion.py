"""Shell completion script generator for NeuralOps CLI v2.

Generates autocomplete scripts for bash, zsh, and fish shells.
"""

from __future__ import annotations

from typing import Sequence

SUPPORTED_SHELLS: tuple[str, ...] = ("bash", "zsh", "fish")

CANONICAL_COMMANDS: tuple[str, ...] = (
    "benchmark",
    "plan",
    "results",
    "doctor",
    "ui",
    "completion",
)


def generate_completion_script(
    shell: str,
    commands: Sequence[str] = CANONICAL_COMMANDS,
) -> str:
    """Generate shell completion script for bash, zsh, or fish.

    Args:
        shell: Target shell name ("bash", "zsh", or "fish").
        commands: Sequence of subcommand names to offer for completion.

    Returns:
        Shell script content as a string.

    Raises:
        ValueError: If shell is not one of ('bash', 'zsh', 'fish').
    """
    shell_norm = shell.lower().strip()
    cmds_str = " ".join(commands)

    if shell_norm == "bash":
        return f"""# NeuralOps Bash completion
_neuralops_complete() {{
    local cur prev words cword
    _init_completion || return
    if [[ $cword -eq 1 ]]; then
        COMPREPLY=( $(compgen -W "{cmds_str}" -- "$cur") )
        return
    fi
}}
complete -F _neuralops_complete neuralops neuralops-cli nops
"""
    elif shell_norm == "zsh":
        return f"""#compdef neuralops neuralops-cli nops
_neuralops() {{
    local -a commands
    commands=({cmds_str})
    _arguments '1: :compadd -a commands' '*:::args:_files'
}}
_neuralops "$@"
"""
    elif shell_norm == "fish":
        cmds_fish = " ".join([f"-a {c}" for c in commands])
        return f"""# NeuralOps Fish completion
complete -c neuralops -f {cmds_fish}
complete -c neuralops-cli -f {cmds_fish}
complete -c nops -f {cmds_fish}
"""
    else:
        supported = ", ".join(SUPPORTED_SHELLS)
        raise ValueError(f"Unsupported shell '{shell}'. Supported shells are: {supported}.")
