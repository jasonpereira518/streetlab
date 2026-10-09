"""The one User-Agent every outbound request identifies itself with.

Nominatim, Photon and Overpass all ask for a descriptive agent that points at the
project; default library agents are rejected by some of them.
"""

USER_AGENT = "StreetLab/0.2 (driving simulator; https://github.com/jasonpereira518/streetlab)"
