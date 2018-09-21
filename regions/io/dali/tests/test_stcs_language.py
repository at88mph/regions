# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import distutils.version as vers

from numpy.testing import assert_allclose
import pytest

from astropy.utils.data import get_pkg_data_filename, get_pkg_data_filenames
import astropy.version as astrov
from astropy.coordinates import Angle, SkyCoord
from astropy.tests.helper import catch_warnings, assert_quantity_allclose
from astropy.utils.exceptions import AstropyUserWarning
from astropy import units as u

from ....shapes.circle import CircleSkyRegion
from ..read import read_stcss, STCSParser
from ..core import STCSRegionParserError
# from ..write import write_ds9, ds9_objects_to_string


def test_stcs_line():
    regstr = 'Circle ICRS 188.5557102 12.0314056 0.05 # color=red'
    parser = STCSParser(regstr)
    regions = parser.shapes
    reg = regions[0].to_region()
    assert reg.center.ra.value == 188.5557102
    assert reg.center.dec.value == 12.0314056
    assert reg.radius.value == 0.05

def test_empty_stcs():
    """
    Checks whether a the line is valid STCS format.
    """
    line_str = ''

    with pytest.raises(STCSRegionParserError) as err:
        STCSParser(line_str)

    assert "No usable values found for '{}'.".format(line_str) in str(err)

def test_valid_crtf_line():
    """
    Checks whether a the line is valid STCS format.
    """
    line_str = 'coord=B1950_VLA, frame=BARY, corr=[I, Q], color=blue'

    with pytest.raises(STCSRegionParserError) as err:
        STCSParser(line_str)

    assert 'Not a valid STCS line:' in str(err)


def test_valid_region_type():
    """
    Checks whether the region type is valid in STCS format
    """
    reg_str = 'Polybogus ICRS 0.9 4.5 6.78 10.9 34.5 54.6'

    with pytest.raises(STCSRegionParserError) as err:
        STCSParser(reg_str)

    assert "Region type 'Polybogus' was identified, but it is not one of the known region types." in str(err)

