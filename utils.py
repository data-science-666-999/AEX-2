import json
import numpy

class NumpyJSONEncoder(json.JSONEncoder):
    """
    A JSONEncoder that can handle numpy float32, float64, int32, and int64 types.
    """
    def default(self, o):
        if isinstance(o, numpy.float32):
            return float(o)
        elif isinstance(o, numpy.float64):
            return float(o)
        elif isinstance(o, numpy.int32):
            return int(o)
        elif isinstance(o, numpy.int64):
            return int(o)
        return super(NumpyJSONEncoder, self).default(o)
