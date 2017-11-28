try:
    import ipdb as pdb
except ImportError:
    import pdb

def trace():
    pdb.set_trace()

print("DEBUGGER has been imported. It should be removed from production code.")


