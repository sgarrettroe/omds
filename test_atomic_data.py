import pytest
from atomic_data_draft import Axis, Polarization, UNITS
import numpy as np

ATOL = 1e-10


class TestAxis:
    def test_t_to_w_1(self):
        """no zero pad, no fftshift, even number"""
        t = np.arange(32, dtype=float)
        opts = {'zeropadded_length': len(t)}
        dim = Axis(t, UNITS.FS, options=opts)
        dim.t_to_w(frequency_units=UNITS.INV_CM)
        ans = np.array([0.   , 1042.38779749, 2084.77559499, 3127.16339248,
                               4169.55118998, 5211.93898747, 6254.32678497, 7296.71458246,
                               8339.10237995, 9381.49017745, 10423.87797494, 11466.26577244,
                               12508.65356993, 13551.04136742, 14593.42916492, 15635.81696241,
                               16678.20475991, 17720.5925574 , 18762.9803549 , 19805.36815239,
                               20847.75594988, 21890.14374738, 22932.53154487, 23974.91934237,
                               25017.30713986, 26059.69493736, 27102.08273485, 28144.47053234,
                               29186.85832984, 30229.24612733, 31271.63392483, 32314.02172232])
        np.testing.assert_allclose(dim.x, ans)

    def test_t_to_w_2(self):
        """zero pad * 2, no fftshift, even number"""
        t = np.arange(32, dtype=float)
        opts= {}
        dim = Axis(t, UNITS.FS, options=opts)
        dim.t_to_w(frequency_units=UNITS.INV_CM)
        ans = np.array([0., 521.19389875,  1042.38779749,  1563.58169624,
        2084.77559499,  2605.96949374,  3127.16339248,  3648.35729123,
        4169.55118998,  4690.74508872,  5211.93898747,  5733.13288622,
        6254.32678497,  6775.52068371,  7296.71458246,  7817.90848121,
        8339.10237995,  8860.2962787 ,  9381.49017745,  9902.6840762 ,
       10423.87797494, 10945.07187369, 11466.26577244, 11987.45967118,
       12508.65356993, 13029.84746868, 13551.04136742, 14072.23526617,
       14593.42916492, 15114.62306367, 15635.81696241, 16157.01086116,
       16678.20475991, 17199.39865865, 17720.5925574 , 18241.78645615,
       18762.9803549 , 19284.17425364, 19805.36815239, 20326.56205114,
       20847.75594988, 21368.94984863, 21890.14374738, 22411.33764613,
       22932.53154487, 23453.72544362, 23974.91934237, 24496.11324111,
       25017.30713986, 25538.50103861, 26059.69493736, 26580.8888361 ,
       27102.08273485, 27623.2766336 , 28144.47053234, 28665.66443109,
       29186.85832984, 29708.05222859, 30229.24612733, 30750.44002608,
       31271.63392483, 31792.82782357, 32314.02172232, 32835.21562107])
        np.testing.assert_allclose(dim.x, ans)

    def test_t_to_w_3(self):
        """zero pad * 2, fftshift = True, even number"""
        t = np.arange(32, dtype=float)
        opts = {'fftshift': True}
        dim = Axis(t, UNITS.FS, options=opts)
        dim.t_to_w(frequency_units=UNITS.INV_CM)
        ans = np.array([-16678.20475991, -16157.01086116, -15635.81696241, -15114.62306367,
       -14593.42916492, -14072.23526617, -13551.04136742, -13029.84746868,
       -12508.65356993, -11987.45967118, -11466.26577244, -10945.07187369,
       -10423.87797494,  -9902.6840762 ,  -9381.49017745,  -8860.2962787 ,
        -8339.10237995,  -7817.90848121,  -7296.71458246,  -6775.52068371,
        -6254.32678497,  -5733.13288622,  -5211.93898747,  -4690.74508872,
        -4169.55118998,  -3648.35729123,  -3127.16339248,  -2605.96949374,
        -2084.77559499,  -1563.58169624,  -1042.38779749,   -521.19389875,
            0.        ,    521.19389875,   1042.38779749,   1563.58169624,
         2084.77559499,   2605.96949374,   3127.16339248,   3648.35729123,
         4169.55118998,   4690.74508872,   5211.93898747,   5733.13288622,
         6254.32678497,   6775.52068371,   7296.71458246,   7817.90848121,
         8339.10237995,   8860.2962787 ,   9381.49017745,   9902.6840762 ,
        10423.87797494,  10945.07187369,  11466.26577244,  11987.45967118,
        12508.65356993,  13029.84746868,  13551.04136742,  14072.23526617,
        14593.42916492,  15114.62306367,  15635.81696241,  16157.01086116])
        np.testing.assert_allclose(dim.x, ans)

    def test_t_to_w_4(self):
        """zero pad * 2, fftshift = True, even number"""
        t = np.arange(33, dtype=float)
        opts = {'fftshift': True, 'zeropadded_length': 33}
        dim = Axis(t, UNITS.FS, options=opts)
        dim.t_to_w(frequency_units=UNITS.INV_CM)
        ans = np.array([-16172.80461567, -15162.00432719, -14151.20403871, -13140.40375023,
       -12129.60346175, -11118.80317327, -10108.00288479,  -9097.20259631,
        -8086.40230783,  -7075.60201935,  -6064.80173088,  -5054.0014424 ,
        -4043.20115392,  -3032.40086544,  -2021.60057696,  -1010.80028848,
            0.        ,   1010.80028848,   2021.60057696,   3032.40086544,
         4043.20115392,   5054.0014424 ,   6064.80173088,   7075.60201935,
         8086.40230783,   9097.20259631,  10108.00288479,  11118.80317327,
        12129.60346175,  13140.40375023,  14151.20403871,  15162.00432719,
        16172.80461567])
        np.testing.assert_allclose(dim.x, ans)

    def test_t_to_w_5(self):
        """zero pad * 2, fftshift = True, even number using defaults"""
        t = np.arange(33, dtype=float)
        opts = {'fftshift': True, 'zeropadded_length': 33}
        defs = {'time_units': UNITS.FS,
                    'frequency_units': UNITS.INV_CM}
        dim = Axis(t, UNITS.FS, options=opts, defaults=defs)
        dim.t_to_w()
        ans = np.array([-16172.80461567, -15162.00432719, -14151.20403871, -13140.40375023,
       -12129.60346175, -11118.80317327, -10108.00288479,  -9097.20259631,
        -8086.40230783,  -7075.60201935,  -6064.80173088,  -5054.0014424 ,
        -4043.20115392,  -3032.40086544,  -2021.60057696,  -1010.80028848,
            0.        ,   1010.80028848,   2021.60057696,   3032.40086544,
         4043.20115392,   5054.0014424 ,   6064.80173088,   7075.60201935,
         8086.40230783,   9097.20259631,  10108.00288479,  11118.80317327,
        12129.60346175,  13140.40375023,  14151.20403871,  15162.00432719,
        16172.80461567])
        np.testing.assert_allclose(dim.x, ans)

class TestPolarization:
    def test_jones_to_stokes_Ex(self):
        j = [1, 0]
        ans = Polarization.jones_to_stokes(j)
        actual = np.array([1, 1, 0, 0])
        np.testing.assert_allclose(actual, ans)

    def test_jones_to_stokes_Ey(self):
        j = [0, 1]
        ans = Polarization.jones_to_stokes(j)
        actual = np.array([1, -1, 0, 0])
        np.testing.assert_allclose(actual, ans)

    def test_jones_to_stokes_R(self):
        j = np.array([1, -1j])/np.sqrt(2)
        ans = Polarization.jones_to_stokes(j)
        actual = np.array([1, 0, 0, 1])
        np.testing.assert_allclose(actual, ans)

    def test_jones_to_stokes_L(self):
        j = np.array([1, 1j])/np.sqrt(2)
        ans = Polarization.jones_to_stokes(j)
        actual = np.array([1, 0, 0, -1])
        np.testing.assert_allclose(actual, ans)

    def test_jones_to_stokes_45(self):
        j = np.array([1, 1])/np.sqrt(2)
        ans = Polarization.jones_to_stokes(j)
        actual = np.array([1, 0, 1, 0])
        np.testing.assert_allclose(actual, ans)

    def test_angles_to_stokes_Ex(self):
        a = [0, 0]
        ans = Polarization.angles_to_stokes(a)
        actual = np.array([1, 1, 0, 0])
        np.testing.assert_allclose(actual, ans, atol=ATOL)

    def test_angles_to_stokes_Ey(self):
        a = [np.pi/2, 0]
        ans = Polarization.angles_to_stokes(a)
        actual = np.array([1, -1, 0, 0])
        np.testing.assert_allclose(actual, ans, atol=ATOL)

    def test_angles_to_stokes_R(self):
        a = np.array([0, -np.pi/4])
        ans = Polarization.angles_to_stokes(a)
        actual = np.array([1, 0, 0, 1])
        np.testing.assert_allclose(actual, ans, atol=ATOL)

    def test_angles_to_stokes_L(self):
        a = np.array([0, np.pi/4])
        ans = Polarization.angles_to_stokes(a)
        actual = np.array([1, 0, 0, -1])
        np.testing.assert_allclose(actual, ans, atol=ATOL)

    def test_angles_to_stokes_45(self):
        a = np.array([np.pi/4, 0])
        ans = Polarization.angles_to_stokes(a)
        actual = np.array([1, 0, 1, 0])
        np.testing.assert_allclose(actual, ans, atol=ATOL)
