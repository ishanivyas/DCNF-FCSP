import sys
import traceback
import logging

def fail(assertionError):
    _, _, trace = sys.exc_info()
    tb = traceback.extract_tb(trace)
    filename, line, func, text = tb[-1]
    #-print(__file__ + ": FAIL: " + str(err), file=sys.stderr)
    logging.error("Test: %s:%s in %s(): FAIL: %s" % (filename, line, func, text))
