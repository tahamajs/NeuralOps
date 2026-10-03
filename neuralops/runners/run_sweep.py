#!/usr/bin/env python3
"""
Master runner for module sweeps.
Replaces the 200+ duplicated run_all_4_*.py scripts.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from neuralops.runners.unified_orchestrator import run_orchestrator

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
REGISTRY_PATH = HERE / "sweep_registry.json"

def get_registry():
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Sweep registry not found at {REGISTRY_PATH}")
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Master runner for module sweeps.")
    parser.add_argument(
        "--module", 
        type=str, 
        required=True,
        help="The key of the module to sweep (e.g. 'cur_linear')."
    )
    # Allows passing remaining arguments to the orchestrator
    args, unknown_args = parser.parse_known_args()
    
    registry = get_registry()
    if args.module not in registry:
        logger.error(f"Module '{args.module}' not found in registry. Available modules: {list(registry.keys())[:5]}...")
        sys.exit(1)
        
    default_args = registry[args.module]
    
    # Prepend unknown_args so sys.argv is manipulated correctly for run_orchestrator
    sys.argv = [sys.argv[0]] + unknown_args
    
    logger.info(f"Running sweep for {args.module} with default args: {default_args}")
    run_orchestrator(args.module, default_args)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
