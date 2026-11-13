from .common import execute

def estimate_rigid(tool, atlas, corrected_reference, transform):
    execute([tool, '-d', '3', '-a', '-m', 'NMI', '-i', atlas, corrected_reference,
             '-o', transform, '-ia-image-centers', '-n', '100x50x10', '-dof', '6'])

def apply_rigid(tool, atlas, source, destination, transform, label=False):
    execute([tool, '-d', '3', '-rf', atlas, '-ri', 'NN' if label else 'LINEAR',
             '-rm', source, destination, '-r', transform])
