import sys
sys.path.insert(0, '.')
from pathlib import Path

# Test with actual files
test_files = [
    'PSVAP/PSVAP files/semaglutide/combined.gro',
    'PSVAP/PSVAP files/semaglutide/combined.itp',
]

print('Testing parsers with actual files:\n')

for file_path in test_files:
    p = Path(file_path)
    if not p.exists():
        print(f'? File not found: {file_path}')
        continue
    
    try:
        from PSVAP.io.base_parser import detect_parser
        parser = detect_parser(p)
        atoms, frames, metadata = parser.parse(p)
        
        n_atoms = len(atoms)
        n_frames = len(frames)
        print(f'? {p.name:20} | {n_atoms:5} atoms | {n_frames:2} frame(s)')
        
        if atoms:
            sample_atom = atoms[0]
            print(f"   First atom: {sample_atom.name or 'N/A':8} (element: {sample_atom.element or 'N/A'})")
    except Exception as e:
        print(f'? {p.name}: {type(e).__name__}: {e}')

print('\nDone!')
