import sys
import random


def eprint(*args, **kwargs):
    """
        Debug print function, prints to standard error stream
    """
    print(*args, file=sys.stderr, **kwargs, sep="")
