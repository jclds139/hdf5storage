# Copyright (c) 2013-2015, Freja Nordsiek
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys
import copy
import os
import os.path
import math
import random
import collections

import numpy as np
import numpy.random

import hdf5storage

from nose.tools import raises

from asserts import *
from make_randoms import *


random.seed()


class TestPythonMatlabFormat(object):
    # Test for the ability to write python types to an HDF5 file that
    # type information and matlab information are stored in, and then
    # read it back and have it be the same.
    def __init__(self):
        self.filename = 'data.mat'
        self.options = hdf5storage.Options()

        # Need a list of the supported numeric dtypes to test, excluding
        # those not supported by MATLAB. 'S' and 'U' dtype chars have to
        # be used for the bare byte and unicode string dtypes since the
        # dtype strings (but not chars) are not the same in Python 2 and
        # 3.
        self.dtypes = ['bool', 'uint8', 'uint16', 'uint32', 'uint64',
                       'int8', 'int16', 'int32', 'int64',
                       'float32', 'float64', 'complex64', 'complex128',
                       'S', 'U']

    def write_readback(self, data, name, options):
        # Write the data to the proper file with the given name, read it
        # back, and return the result. The file needs to be deleted
        # before and after to keep junk from building up.
        if os.path.exists(self.filename):
            os.remove(self.filename)
        try:
            hdf5storage.write(data, path=name, filename=self.filename,
                              options=options)
            out = hdf5storage.read(path=name, filename=self.filename,
                                   options=options)
        except:
            raise
        finally:
            if os.path.exists(self.filename):
                os.remove(self.filename)
        return out

    def assert_equal(self, a, b):
        assert_equal(a, b)

    def check_numpy_scalar(self, dtype):
        # Makes a random numpy scalar of the given type, writes it and
        # reads it back, and then compares it.
        data = random_numpy_scalar(dtype)
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def check_numpy_array(self, dtype, dimensions):
        # Makes a random numpy array of the given type, writes it and
        # reads it back, and then compares it.
        shape = random_numpy_shape(dimensions,
                                   max_array_axis_length)
        data = random_numpy(shape, dtype)
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def check_numpy_empty(self, dtype):
        # Makes an empty numpy array of the given type, writes it and
        # reads it back, and then compares it.
        data = np.array([], dtype)
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)
    
    def check_numpy_structured_array(self, dimensions):
        # Makes a random structured ndarray of the given type, writes it
        # and reads it back, and then compares it.
        shape = random_numpy_shape(dimensions, \
            max_structured_ndarray_axis_length)
        data = random_structured_numpy_array(shape)
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)
    
    def check_numpy_structured_array_empty(self, dimensions):
        # Makes a random structured ndarray of the given type, writes it
        # and reads it back, and then compares it.
        shape = random_numpy_shape(dimensions, \
            max_structured_ndarray_axis_length)
        data = random_structured_numpy_array(shape, (1, 0))
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)
    
    def check_python_collection(self, tp):
        # Makes a random collection of the specified type, writes it and
        # reads it back, and then compares it.
        if tp in (set, frozenset):
            data = tp(random_list(max_list_length,
                      python_or_numpy='python'))
        else:
            data = tp(random_list(max_list_length,
                      python_or_numpy='numpy'))
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_None(self):
        data = None
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_bool_True(self):
        data = True
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_bool_False(self):
        data = False
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_int(self):
        data = random_int()
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    # Only relevant in Python 2.x.
    def test_long(self):
        if sys.hexversion < 0x03000000:
            data = long(random_int())
            out = self.write_readback(data, random_name(),
                                      self.options)
            self.assert_equal(out, data)

    def test_int_or_long_too_big(self):
        if sys.hexversion >= 0x03000000:
            data = 2**64 * random_int()
        else:
            data = long(2)**64 * long(random_int())
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_float(self):
        data = random_float()
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_float_inf(self):
        data = float(np.inf)
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_float_ninf(self):
        data = float(-np.inf)
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_float_nan(self):
        data = float(np.nan)
        out = self.write_readback(data, random_name(),
                                  self.options)
        assert math.isnan(out)

    def test_complex(self):
        data = random_float() + 1j*random_float()
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_complex_real_nan(self):
        data = complex(np.nan, random_float())
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_complex_imaginary_nan(self):
        data = complex(random_float(), np.nan)
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_str_ascii(self):
        data = random_str_ascii(random.randint(1,
                                max_string_length))
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_str_ascii_encoded_utf8(self):
        data = random_str_some_unicode(random.randint(1, \
            max_string_length)).encode('UTF-8')
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_str_with_null(self):
        strs = [random_str_ascii(
                random.randint(1, max_string_length))
                for i in range(2)]
        if sys.hexversion < 0x03000000:
            data = u'\x00'.join(strs)
        else:
            data = '\x00'.join(strs)
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_str_unicode(self):
        data = random_str_some_unicode(random.randint(1,
                                       max_string_length))
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_str_empty(self):
        data = ''
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_bytes(self):
        data = random_bytes(random.randint(1,
                            max_string_length))
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_bytes_empty(self):
        data = b''
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_bytes_with_null(self):
        strs = [random_bytes(
                random.randint(1, max_string_length))
                for i in range(2)]
        if sys.hexversion < 0x03000000:
            data = '\x00'.join(strs)
        else:
            data = b'\x00'.join(strs)
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_bytearray(self):
        data = bytearray(random_bytes(random.randint(1,
                         max_string_length)))
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_bytearray_empty(self):
        data = bytearray(b'')
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_numpy_scalar(self):
        for dt in self.dtypes:
            yield self.check_numpy_scalar, dt

    def test_numpy_array_1d(self):
        dts = copy.deepcopy(self.dtypes)
        dts.append('object')
        for dt in dts:
            yield self.check_numpy_array, dt, 1

    def test_numpy_array_2d(self):
        dts = copy.deepcopy(self.dtypes)
        dts.append('object')
        for dt in dts:
            yield self.check_numpy_array, dt, 2

    def test_numpy_array_3d(self):
        dts = copy.deepcopy(self.dtypes)
        dts.append('object')
        for dt in dts:
            yield self.check_numpy_array, dt, 3

    def test_numpy_empty(self):
        for dt in self.dtypes:
            yield self.check_numpy_empty, dt
    
    def test_numpy_structured_array(self):
        for i in range(1, 4):
            yield self.check_numpy_structured_array, i
    
    def test_numpy_structured_array_empty(self):
        for i in range(1, 4):
            yield self.check_numpy_structured_array_empty, i

    def test_numpy_structured_array_unicode_fields(self):
        # Makes a random 1d structured ndarray with non-ascii characters
        # in its fields, writes it and reads it back, and then compares
        # it.
        shape = random_numpy_shape(1, \
            max_structured_ndarray_axis_length)
        data = random_structured_numpy_array(shape,
                                             nonascii_fields=True)
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_python_collection(self):
        for tp in (list, tuple, set, frozenset, collections.deque):
            yield self.check_python_collection, tp

    def test_dict(self):
        data = random_dict()
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)


class TestPythonFormat(TestPythonMatlabFormat):
    def __init__(self):
        # The parent does most of the setup. All that has to be changed
        # is turning MATLAB compatibility off and changing the file
        # name.
        TestPythonMatlabFormat.__init__(self)
        self.options = hdf5storage.Options(matlab_compatible=False)
        self.filename = 'data.h5'


class TestNoneFormat(TestPythonMatlabFormat):
    def __init__(self):
        # The parent does most of the setup. All that has to be changed
        # is turning off the storage of type information as well as
        # MATLAB compatibility.
        TestPythonMatlabFormat.__init__(self)
        self.options = hdf5storage.Options(store_python_metadata=False,
                                           matlab_compatible=False)

        # Add in float16 to the set of types tested.
        self.dtypes.append('float16')

    def assert_equal(self, a, b):
        assert_equal_none_format(a, b)


class TestMatlabFormat(TestPythonMatlabFormat):
    def __init__(self):
        # The parent does most of the setup. All that has to be changed
        # is turning on the matlab compatibility, and changing the
        # filename.
        TestPythonMatlabFormat.__init__(self)
        self.options = hdf5storage.Options(store_python_metadata=False,
                                           matlab_compatible=True)
        self.filename = 'data.mat'

    def assert_equal(self, a, b):
        assert_equal_matlab_format(a, b)
