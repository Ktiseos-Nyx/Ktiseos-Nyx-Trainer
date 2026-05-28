for fname in ['BETA_PLANNING.md', 'documentation/installation/INSTALLATION.md']:
    with open(fname, encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if line.strip() == '```':
                print(f'{fname}:{i}')
