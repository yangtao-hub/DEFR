from pathlib import Path
import numpy as np

def validate_names(names, image_dir=None, label_dir=None):
    names = list(names)
    if not names or len(names) != len(set(names)):
        raise ValueError('empty or duplicate case list')
    if any((Path(n).name != n or '/' in n or chr(92) in n or (n in {'.', '..'}) for n in names)):
        raise ValueError('basenames required')
    for directory in (image_dir, label_dir):
        if directory is not None and any((not (Path(directory) / n).is_file() for n in names)):
            raise ValueError('missing listed file')
    return names

def read_names(path, image_dir=None, label_dir=None):
    return validate_names([s.strip() for s in Path(path).read_text().splitlines() if s.strip()], image_dir, label_dir)

def require_disjoint(first, second):
    if set(first) & set(second):
        raise ValueError('case list overlap')

def historical_selection(names, mode, train_rate=1.0, num_data=64):
    names = np.array(sorted(validate_names(names)))
    train_num = int(len(names) * train_rate)
    if train_num % 2 != len(names) % 2:
        train_num -= 1
    rng = np.random.RandomState(12345)
    if mode in ('train', 'valid'):
        index = rng.choice(len(names), train_num, replace=False)
        return list(names[index] if mode == 'train' else np.delete(names, index))
    if num_data == -1:
        return list(names)
    index = rng.choice(len(names), num_data, replace=False)
    return list(np.delete(names, index) if mode == 'test_final' else names[index])
