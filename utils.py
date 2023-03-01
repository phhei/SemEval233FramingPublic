import json
import numpy
from pathlib import Path


class MLJsonSerializer(json.JSONEncoder):
    """Custom encoder for json serialization of numpy.float32 and pathlib.Path objects."""
    def default(self, obj):
        if isinstance(obj, numpy.float32):
            return float(obj)
        elif isinstance(obj, numpy.int_):
            return int(obj)
        elif isinstance(obj, Path):
            return str(obj.absolute())
        else:
            return super(MLJsonSerializer, self).default(obj)
            