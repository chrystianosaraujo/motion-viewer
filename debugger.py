try:
    import ipdb as pdb
except ImportError:
    import pdb

import cProfile
import pstats
import io

def trace():
    pdb.set_trace()

def start_profiler():
    pr = cProfile.Profile()
    pr.enable()
    return pr

def finish_profiler(pr, sortby='time'):
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

print("DEBUGGER has been imported. It should removed from production code.")
