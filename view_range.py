import pathlib
lines=open('app.py', encoding='utf-8').read().splitlines()
print('\n'.join(f'{i+1}: {lines[i]}' for i in range(110,220)))
