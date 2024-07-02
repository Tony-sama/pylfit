#-------------------------------------------------------------------------------
# @author: Tony Ribeiro
# @created: 2019/03/20
# @updated: 2023/12/27
#
# @desc: pylfit general utility functions
#-------------------------------------------------------------------------------

import sys

def eprint(*args, **kwargs):
    """
        Debug print function, prints to standard error stream
    """
    print(*args, file=sys.stderr, **kwargs, sep="")
