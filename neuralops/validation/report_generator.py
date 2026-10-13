"""Generate markdown and JSON validation reports."""
import json, os
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
