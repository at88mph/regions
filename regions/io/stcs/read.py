# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import string
import itertools
import re
import copy
from collections import OrderedDict
from warnings import warn

from astropy import units as u
from astropy import coordinates
from astropy import log

from ..core import reg_mapping
from ..core import Shape, ShapeList
from .core import STCSRegionParserError, STCSRegionParserWarning

__all__ = [
    'read_stcs',
    'STCSParser',
    'STCSRegionParser',
    'CoordinateParser'
]

# Regular expression to extract region type or coodinate system
regex_global = re.compile("^#? *([a-zA-Z0-9]+)")

# Regular expression to extract meta attributes
regex_meta = re.compile("([a-zA-Z]+)(=)({.*?}|\'.*?\'|\".*?\"|[0-9\s]+\s?|[^=\s]+\s?[0-9]*)\s?")

# Regular expression to strip parenthesis
regex_paren = re.compile("[()]")

# Regular expression to split coordinate strings
regex_splitter = re.compile("[, ]")


def read_stcs(filename, errors='strict'):
    """
    Read a STCS region file in as a `list` of `~regions.Region` objects.

    Parameters
    ----------
    filename : `str`
        The file path
    errors : ``warn``, ``ignore``, ``strict``, optional
      The error handling scheme to use for handling parsing errors.
      The default is 'strict', which will raise a `~regions.DS9RegionParserError`.
      ``warn`` will raise a `~regions.DS9RegionParserWarning`, and
      ``ignore`` will do nothing (i.e., be silent).

    Returns
    -------
    regions : `list`
        Python list of `~regions.Region` objects.

    Examples
    --------
    >>> from regions import read_ds9
    >>> from astropy.utils.data import get_pkg_data_filename
    >>> file = get_pkg_data_filename('data/physical_reference.reg', package='regions.io.ds9.tests')
    >>> regs = read_ds9(file, errors='warn')
    >>> print(regs[0])
    Region: CirclePixelRegion
    center: PixCoord(x=330.0, y=1090.0)
    radius: 40.0
    >>> print(regs[0].meta)
    {'label': 'Circle', 'select': '1', 'highlite': '1', 'fixed': '0', 'edit': '1', 'move': '1', 'delete': '1', 'source': '1', 'tag': ['{foo}', '{foo bar}'], 'include': True}
    >>> print(regs[0].visual)
    {'dashlist': '8 3', 'dash': '0', 'color': 'pink', 'linewidth': '3', 'font': 'times', 'fontsize': '10', 'fontstyle': 'normal', 'fontweight': 'roman'}

    """
    with open(filename) as fh:
        region_string = fh.read()

    parser = STCSParser(region_string, errors=errors)
    return parser.shapes.to_regions()


class CoordinateParser(object):
    """
    Helper class to structure coordinate parser
    """
    @staticmethod
    def parse_coordinate(string_rep, unit):
        """
        Parse a single coordinate
        """
        # explicit radian ('r') value
        if string_rep[-1] == 'r':
            return coordinates.Angle(string_rep[:-1], unit=u.rad)
        # explicit image ('i') and physical ('p') pixels
        elif string_rep[-1] in ['i', 'p']:
            return u.Quantity(string_rep[:-1]) - 1
        # Any ds9 coordinate representation (sexagesimal or degrees)
        elif 'd' in string_rep or 'h' in string_rep:
            return coordinates.Angle(string_rep)
        elif unit is 'hour_or_deg':
            if ':' in string_rep:
                spl = tuple([float(x) for x in string_rep.split(":")])
                return coordinates.Angle(spl, u.hourangle)
            else:
                ang = float(string_rep)
                return coordinates.Angle(ang, u.deg)
        elif unit.is_equivalent(u.deg):
            # return coordinates.Angle(string_rep, unit=unit)
            if ':' in string_rep:
                ang = tuple([float(x) for x in string_rep.split(":")])
            else:
                ang = float(string_rep)
            return coordinates.Angle(ang, u.deg)
        elif unit.is_equivalent(u.dimensionless_unscaled):
            return u.Quantity(float(string_rep), unit) - 1
        else:
            return u.Quantity(float(string_rep), unit)

    @staticmethod
    def parse_angular_length_quantity(string_rep, unit=u.deg):
        """
        Given a string that is either a number or a number and a unit, return a
        Quantity of that string.  e.g.:

            23.9 -> 23.9*u.deg
            50" -> 50*u.arcsec
        """
        unit_mapping = {
            '"': u.arcsec,
            "'": u.arcmin,
            'd': u.deg,
            'r': u.rad,
            'i': u.dimensionless_unscaled,
            'p': u.dimensionless_unscaled
        }
        has_unit = string_rep[-1] not in string.digits
        if has_unit:
            unit = unit_mapping[string_rep[-1]]
            return u.Quantity(float(string_rep[:-1]), unit=unit)
        else:
            return u.Quantity(float(string_rep), unit=unit)


class STCSParser(object):
    """
    Parse an STCS string

    This class transforms a STCS string to a `~regions.io.core.ShapeList`. The
    result is stored as ``shapes`` attribute.

    Each line is tested for either containing a region type or a coordinate
    system. If a coordinate system is found the global coordsys state of the
    parser is modified. If a region type is found the
    `~regions.STCSRegionParser` is invokes to transform the line into a
    `~regions.Shape` object.

    Parameters
    ----------
    region_string : `str`
        STCS region string
    errors : ``warn``, ``ignore``, ``strict``, optional
      The error handling scheme to use for handling parsing errors.
      The default is 'strict', which will raise a `~regions.STCSRegionParserError`.
      ``warn`` will raise a `~regions.STCSRegionParserWarning`, and
      ``ignore`` will do nothing (i.e., be silent).

    Examples
    --------
    >>> from regions import STCSParser
    >>> reg_str = 'Circle ICRS 331.00 1091.00 0.5 # dashlist=8 3 select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 source=1 text={Circle} tag={foo} tag={foo bar} This is a Comment color=pink width=3 font="times 10 normal roman"'
    >>> regs = STCSParser(reg_str, errors='warn').shapes.to_regions()
    >>> print(regs[0])
    Region: CirclePixelRegion
    center: PixCoord(x=330.0, y=1090.0)
    radius: 0.5
    >>> print(regs[0].meta)
    {'label': 'Circle', 'select': '1', 'highlite': '1', 'fixed': '0', 'edit': '1', 'move': '1', 'delete': '1', 'source': '1', 'tag': ['{foo}', '{foo bar}'], 'include': True}
    >>> print(regs[0].visual)
    {'dashlist': '8 3', 'dash': '0', 'color': 'pink', 'linewidth': '3', 'font': 'times', 'fontsize': '10', 'fontstyle': 'normal', 'fontweight': 'roman'}
    """

    # List of valid coordinate system
    coordinate_systems = ['fk5', 'fk4', 'icrs', 'galactic', 'wcs', 'physical', 'image', 'ecliptic', 'J2000']
    coordinate_systems += ['wcs{0}'.format(letter) for letter in string.ascii_lowercase]

    # Map to convert coordinate system names
    coordsys_mapping = dict(zip(coordinates.frame_transform_graph.get_names(),
                                coordinates.frame_transform_graph.get_names()))
    coordsys_mapping['ecliptic'] = 'geocentrictrueecliptic'
    coordsys_mapping['J2000'] = 'fk5'

    language_specs = ['point', 'box', 'circle', 'polygon', 'position', 'union', 'not', 'intersection']
    reference_positions = ['babycenter', 'geocenter', 'heliocenter', 'lsr', 'topocenter', 'relocatable', 'unknownrefpos']
    flavours = ['cartesian2', 'cartesian3', 'spherical2']

    def __init__(self, region_string, errors='strict'):
        if errors not in ('strict', 'ignore', 'warn'):
            msg = "``errors`` must be one of strict, ignore, or warn; is {}"
            raise ValueError(msg.format(errors))
        self.region_string = region_string
        self.errors = errors

        # Global states
        self.coordsys = None
        self.global_meta = {}

        # Results
        self.shapes = ShapeList()

        self.run()

    def __str__(self):
        ss = self.__class__.__name__
        ss += '\nErrors: {}'.format(self.errors)
        ss += '\nCoordsys: {}'.format(self.coordsys)
        ss += '\nGlobal meta: {}'.format(self.global_meta)
        ss += '\nShapes: {}'.format(self.shapes)
        ss += '\n'
        return ss

    def set_coordsys(self, coordsys):
        """
        Transform coordinate system

        # TODO: needs expert attention
        """
        if coordsys in self.coordsys_mapping:
            self.coordsys = self.coordsys_mapping[coordsys]
        else:
            self.coordsys = coordsys

    def run(self):
        """
        Run all steps
        """
        for line in self.region_string.split('\n'):
            self.parse_line(line)

    def parse_line(self, line):
        """
        Parse one line
        """
        log.debug('Parsing {}'.format(line))

        # Skip blanks
        if line == '':
            return

        # Skip comments
        if line[0] == '#':
            return

        # Try to parse the line
        values = line.split(' ')

        if values:
            curr_index = 0
            next_val = values[curr_index]
            region_type = next_val

            if not self.language_specs[region_type]:
                self._raise_error("Region type '{0}' was identified, but it is not one of \
                                  the known region types.".format(region_type))
                return

            ref_position = None
            flavour = None
            next_val = values[curr_index]

            if self.coordinate_systems[next_val.lower()]:
                frame = next_val
                self.set_coordsys(frame)
                curr_index += 1
                next_val = values[curr_index]
            else:
                frame = None

            if self.reference_positions[next_val.lower()]:
                ref_position = next_val
                curr_index += 1
                next_val = values[curr_index]
            else:
                ref_position = None

            if self.flavours[next_val.lower()]:
                flavour = next_val
                curr_index += 1
            else:
                flavour = None
        else:
            self._raise_error("No usable values found for '{0}'.".format(line))
            return

        # Found region specification,
        region_end = ' '.join(values[curr_index:])
        helper = STCSRegionParser(self.global_meta, include, type_, region_type,
                                    *crtf_line.group('region', 'parameters'))
        self.shapes.append(helper.shape)

    def _raise_error(self, msg):
        if self.errors == 'warn':
            warn(msg, STCSRegionParserWarning)
        elif self.errors == 'strict':
            raise STCSRegionParserError(msg)

    @staticmethod
    def parse_meta(meta_str):
        """
        Parse the metadata for a single ds9 region string.

        Parameters
        ----------
        meta_str : `str`
            Meta string, the metadata is everything after the close-paren of the
            region coordinate specification. All metadata is specified as
            key=value pairs separated by whitespace, but sometimes the values
            can also be whitespace separated.

        Returns
        -------
        meta : `~collections.OrderedDict`
            Dictionary containing the meta data
        """
        keys_vals = [(x, y) for x, _, y in regex_meta.findall(meta_str.strip())]
        extra_text = regex_meta.split(meta_str.strip())[-1]
        result = OrderedDict()
        for key, val in keys_vals:
            # regex can include trailing whitespace or inverted commas
            # remove it
            val = val.strip().strip("'").strip('"')
            if key == 'text':
                val = val.lstrip("{").rstrip("}")
            if key in result:
                if key == 'tag':
                    result[key].append(val)
                else:
                    raise ValueError("Duplicate key {0} found".format(key))
            else:
                if key == 'tag':
                    result[key] = [val]
                else:
                    result[key] = val
        if extra_text:
            result['comment'] = extra_text

        return result

    def parse_region(self, include, region_type, region_end, line):
        """
        Extract a Shape from a region string
        """
        if self.coordsys is None:
            raise STCSRegionParserError("No coordinate system specified and a"
                                       " region has been found.")
        else:
            helper = STCSRegionParser(coordsys=self.coordsys,
                                     include=include,
                                     region_type=region_type,
                                     region_end=region_end,
                                     global_meta=self.global_meta,
                                     line=line)
            helper.parse()
            self.shapes.append(helper.shape)


# Global definitions to improve readability
radius = CoordinateParser.parse_angular_length_quantity
width = CoordinateParser.parse_angular_length_quantity
height = CoordinateParser.parse_angular_length_quantity
angle = CoordinateParser.parse_angular_length_quantity
coordinate = CoordinateParser.parse_coordinate


class STCSRegionParser(object):
    """
    Parse a STCS region string

    This will turn a line containing a STCS region into a Shape

    Parameters
    ----------
    coordsys : `str`
        Coordinate system
    include : `str` {'', '-'}
        Flag at the beginning of the line
    region_type : `str`
        Region type
    region_end : `int`
        Coordinate of the end of the regions name, this is passed in order to
        handle whitespaces correctly
    global_meta : `dict`
        Global meta data
    line : `str`
        Line to parse
    """

    # Coordinate unit transformations
    coordinate_units = {'fk5': ('hour_or_deg', u.deg),
                        'fk4': ('hour_or_deg', u.deg),
                        'icrs': ('hour_or_deg', u.deg),
                        'geocentrictrueecliptic': (u.deg, u.deg),
                        'galactic': (u.deg, u.deg),
                        'physical': (u.dimensionless_unscaled, u.dimensionless_unscaled),
                        'image': (u.dimensionless_unscaled, u.dimensionless_unscaled),
                        'wcs': (u.dimensionless_unscaled, u.dimensionless_unscaled),
                        }
    for letter in string.ascii_lowercase:
        coordinate_units['wcs{0}'.format(letter)] = (u.dimensionless_unscaled, u.dimensionless_unscaled)

    # STCS language specification. This defines how a certain region is read.
    language_spec = {'point': (coordinate, coordinate),
                     'text': (coordinate, coordinate),
                     'circle': (coordinate, coordinate, radius),
                     # This is a special case to deal with n elliptical annuli
                     'ellipse': itertools.chain((coordinate, coordinate),
                                                itertools.cycle((radius,))),
                     'box': (coordinate, coordinate, width, height, angle),
                     'polygon': itertools.cycle((coordinate,)),
                     'line': (coordinate, coordinate, coordinate, coordinate),
                     'annulus': itertools.chain((coordinate, coordinate),
                                                itertools.cycle((radius,))),
                     }

    def __init__(self, frame, reference_position, flavour, region_end, line):

        self.frame = frame
        self.reference_position = reference_position
        self.flavour = flavour
        self.region_end = region_end
        self.line = line

        self.meta_str = None
        self.coord_str = None
        self.composite = None
        self.coord = None
        self.meta = None
        self.shape = None

    def __str__(self):
        ss = self.__class__.__name__
        ss += '\nLine : {}'.format(self.line)
        ss += '\nRegion end : {}'.format(self.region_end)
        ss += '\nMeta string : {}'.format(self.meta_str)
        ss += '\nCoord string: {}'.format(self.coord_str)
        ss += '\nShape: {}'.format(self.shape)
        ss += '\n'
        return ss

    def parse(self):
        """
        Convert line to shape object
        """
        log.debug(self)

        self.parse_composite()
        self.split_line()
        self.convert_coordinates()
        self.convert_meta()
        self.make_shape()
        log.debug(self)

    def parse_composite(self):
        """
        Determine whether the region is composite
        """
        self.composite = "||" in self.line

    def split_line(self):
        """
        Split line into coordinates and meta string
        """
        # coordinate of the # symbol or end of the line (-1) if not found
        hash_or_end = self.line.find("#")
        temp = self.line[self.region_end:hash_or_end].strip(" |")
        self.coord_str = regex_paren.sub("", temp)

        # don't want any meta_str if there is no metadata found
        if hash_or_end >= 0:
            self.meta_str = self.line[hash_or_end:]
        else:
            self.meta_str = ""

    def convert_coordinates(self):
        """
        Convert coordinate string to objects
        """
        coord_list = []
        # strip out "null" elements, i.e. ''.  It might be possible to eliminate
        # these some other way, i.e. with regex directly, but I don't know how.
        # We need to copy in order not to burn up the iterators
        elements = [x for x in regex_splitter.split(self.coord_str) if x]
        element_parsers = self.language_spec[self.region_type]
        for ii, (element, element_parser) in enumerate(zip(elements,
                                                           element_parsers)):
            if element_parser is coordinate:
                unit = self.coordinate_units[self.coordsys][ii % 2]
                coord_list.append(element_parser(element, unit))
            elif self.coordinate_units[self.coordsys][0] is u.dimensionless_unscaled:
                coord_list.append(element_parser(element, unit=u.dimensionless_unscaled))
            else:
                coord_list.append(element_parser(element))

        if self.region_type in ['ellipse', 'box'] and len(coord_list) % 2 == 1:
            coord_list[-1] = CoordinateParser.parse_angular_length_quantity(elements[len(coord_list)-1])

        # Reset iterator for ellipse and annulus
        # Note that this cannot be done with copy.deepcopy on python2
        if self.region_type in ['ellipse', 'annulus']:
            self.language_spec[self.region_type] = itertools.chain(
                (coordinate, coordinate), itertools.cycle((radius,)))

        self.coord = coord_list

    def convert_meta(self):
        """
        Convert meta string to dict
        """
        meta_ = STCSParser.parse_meta(self.meta_str)
        self.meta = copy.deepcopy(self.global_meta)
        self.meta.update(meta_)
        # the 'include' is not part of the metadata string;
        # it is pre-parsed as part of the shape type and should always
        # override the global one
        self.include = self.meta.get('include', True) if self.include == '' else self.include != '-'
        self.meta['include'] = self.include

    def make_shape(self):
        """
        Make shape object
        """
        # In DS9, ellipse can also represents an elliptical annulus
        # For elliptical annulus angle is optional.
        if self.region_type == 'ellipse':
            self.coord[2:] = [x * 2 for x in self.coord[2:]]
            if len(self.coord) % 2 == 1:  # This checks if angle is present
                self.coord[-1] /= 2

        if 'point' in self.meta:
            point = self.meta['point'].split(" ")
            if len(point) > 1:
                self.meta['symsize'] = point[1]
            self.meta['point'] = valid_symbols_stcs[point[0]]

        if 'font' in self.meta:
            fonts = self.meta['font'].split(" ")
            keys = ['font', 'fontsize', 'fontstyle', 'fontweight']
            for i, val in enumerate(fonts):
                self.meta[keys[i]] = val

        self.meta.pop('coord', None)

        self.shape = Shape(coordsys=self.coordsys,
                           region_type=reg_mapping['STCS'][self.region_type],
                           coord=self.coord,
                           meta=self.meta,
                           composite=self.composite,
                           include=self.include,
                          )
