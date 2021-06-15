#-------------------------------------------------------------------------------
# @author: Tony Ribeiro
# @created: 2019/03/20
# @updated: 2021/06/15
#
# @desc: pylfit general utility functions
#-------------------------------------------------------------------------------

import sys

def eprint(*args, **kwargs):
    """
        Debug print function, prints to standard error stream
    """
    print(*args, file=sys.stderr, **kwargs, sep="")
