import cProfile
import pstats


def profile(filename: str = ''):
    '''
    A decorator that uses cProfile to profile a function.
    If :param: filename is used, a pstats.Stats object will
    be stored in a file under the same name. Otherwise the
    profiling will be printed to standard output.
    NOTE: A function which uses this profiling MUST NOT have
          a keyword argument called 'filename'.

    :param filename: string path to file where pstats.Stats will be saved
    '''

    def wrap(func):
        def wrapped_f(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            retval = func(*args, **kwargs)
            pr.disable()
            sortby = 'cumulative'
            ps = pstats.Stats(pr).sort_stats(sortby)
            if filename != '': ps.dump_stats(filename)
            else: print(ps.print_stats())
            return retval
        return wrapped_f
    return wrap
