from .common import execute

def reorient(tool, source, destination):
    execute([tool, "-i", source, "-o", destination, "-or", "RAI"])
