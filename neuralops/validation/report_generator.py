#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Validation Report Generator

Generates structured markdown and JSON summary reports from validation suites.
"""

import json
import os
from datetime import datetime

class ReportGenerator:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.entries = []

    def add(self, category, name, status, value=None, notes=''):
        self.entries.append({
            'category': category, 'name': name, 'status': status,
            'value': value, 'notes': notes,
            'timestamp': datetime.now().isoformat()
        })

    def save(self, filename):
        os.makedirs(self.output_dir, exist_ok=True)
        path = f'{self.output_dir}/{filename}'
        with open(path, 'w') as f:
            f.write(f'# Validation Report\n\nGenerated: {datetime.now()}\n\n')
            f.write('| Category | Name | Status | Value | Notes |\n')
            f.write('|----------|------|--------|-------|-------|\n')
            for e in self.entries:
                f.write(f"| {e['category']} | {e['name']} | {e['status']} | {e['value'] or ''} | {e['notes']} |\n")
        json_path = path.replace('.md', '.json')
        with open(json_path, 'w') as f:
            json.dump(self.entries, f, indent=2)
        print(f'Report saved to {path}')
        return path
