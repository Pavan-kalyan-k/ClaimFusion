import json
import os

notebooks = ['car_box.ipynb', 'car_severity_corrected (1).ipynb', 'car_ml.ipynb']
with open('code_summary.txt', 'w', encoding='utf-8') as out:
    for nb in notebooks:
        out.write(f'\n\n================ {nb} ================\n\n')
        try:
            with open(nb, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for cell in data.get('cells', []):
                    if cell.get('cell_type') == 'code':
                        source = ''.join(cell.get('source', []))
                        if source.strip():
                            out.write(source + '\n')
                            out.write('-'*40 + '\n')
        except Exception as e:
            out.write(f'Error reading {nb}: {e}\n')
