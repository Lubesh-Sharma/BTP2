#!/usr/bin/env python3
"""Check the report's references, figure assets and editable diagram structure."""
from pathlib import Path
import collections
import re
import sys
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parents[1]
files = [root / 'main.tex']
files += sorted((root / 'Chapters').glob('Chapter[1-5].tex'))
files += [root / 'Appendices/AppendixA.tex']
labels, refs, citations, graphics = [], [], [], []
errors = []
for path in files:
    source = re.sub(r'(?m)(?<!\\)%.*$', '', path.read_text())
    braces = 0
    for match in re.finditer(r'(?<!\\)[{}]', source):
        braces += 1 if match.group() == '{' else -1
        if braces < 0:
            errors.append(f'{path.name}: unmatched closing brace')
            break
    if braces:
        errors.append(f'{path.name}: brace balance {braces}')
    labels += re.findall(r'\\label\{([^}]+)\}', source)
    refs += re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', source)
    for group in re.findall(r'\\cite\w*\{([^}]+)\}', source):
        citations += [key.strip() for key in group.split(',')]
    graphics += re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', source)
    if '**' in source:
        errors.append(f'{path.name}: Markdown bold syntax in LaTeX')
keys = re.findall(r'@\w+\s*\{\s*([^,]+)', (root / 'Bibliography.bib').read_text())
for key in sorted(set(citations) - set(keys)):
    errors.append(f'Missing bibliography key: {key}')
for label in sorted(set(refs) - set(labels)):
    errors.append(f'Missing reference label: {label}')
for label, count in collections.Counter(labels).items():
    if count > 1:
        errors.append(f'Duplicate label: {label}')
for name in graphics:
    if not any((base / name).exists() for base in [root, root / 'Pictures']):
        errors.append(f'Missing figure: {name}')
for path in sorted((root / 'Pictures/current').glob('*.drawio')):
    tree = ET.parse(path)
    cells = tree.findall('.//mxCell')
    ids = [c.get('id') for c in cells]
    if len(ids) != len(set(ids)):
        errors.append(f'{path.name}: duplicate diagram ids')
    for cell in cells:
        if cell.get('edge') == '1' and cell.find('mxGeometry') is None:
            errors.append(f'{path.name}: edge without geometry')
        for attr in ('source', 'target', 'parent'):
            if cell.get(attr) and cell.get(attr) not in ids:
                errors.append(f'{path.name}: missing {attr} {cell.get(attr)}')
if errors:
    print('\n'.join(errors))
    sys.exit(1)
print(f'OK: {len(files)} TeX sources, {len(set(citations))} cited works, '
      f'{len(graphics)} figures; references and diagram XML valid.')
