import sys
sys.path.append(__file__[:-12])
import Mymodels
import MapPalette
import Mymap
import Myfit
import Mysigmap
import Myspectrum

import importlib

importlib.reload(MapPalette)
importlib.reload(Mymap)
importlib.reload(Mymodels)
importlib.reload(Myfit)
importlib.reload(Mysigmap)
importlib.reload(Myspectrum)

from Mymodels import *
from MapPalette import *
from Mymap import *
from Myfit import *
from Mysigmap import *
from Myspectrum import *



print("Yourlib init successful!!!")