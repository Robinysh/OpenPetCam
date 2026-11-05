"""One module per sensor. A sensor declares its band, can collect() raw data from the device into eval/world/raw/, and
turns what has been collected into layers(). Sensors never move the camera or play sound: collection reads what the
devices already produce (frames, the microphone tap, link statistics)."""
from .acoustic import ActiveSonar, AmbientSound
from .geometry import DepthAnything3Depth
from .infrared import NightInfrared
from .nir_active import ActivePassiveNIR
from .presence import Presence
from .visible import VisibleCamera
from .wifi import WifiEnvironment, WifiLink

ALL = [VisibleCamera, NightInfrared, ActiveSonar, AmbientSound, WifiLink, WifiEnvironment, DepthAnything3Depth, ActivePassiveNIR, Presence]
