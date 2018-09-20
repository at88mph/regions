# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

from astropy.utils.exceptions import AstropyUserWarning

__all__ = [
    'STCSRegionParserWarning',
    'STCSRegionParserError',
]


class STCSRegionParserWarning(AstropyUserWarning):
    """
    A generic warning class for STCS region parsing inherited from astropy's
    warnings
    """


class STCSRegionParserError(ValueError):
    """
    A generic error class for STCS region parsing
    """
