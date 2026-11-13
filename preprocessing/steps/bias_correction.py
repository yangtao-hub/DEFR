from .common import execute

def correct_reference(tool, source, destination):
    execute([tool, "-i", source, "-o", destination, "-n4"])
