#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
CLI entry point for recipe surgery commands.
"""

from neuralops.cli.surgery_cmd import add_surgery_subparser, handle_surgery_command, main

__all__ = ["add_surgery_subparser", "handle_surgery_command", "main"]

if __name__ == "__main__":
    import sys
    sys.exit(main())
