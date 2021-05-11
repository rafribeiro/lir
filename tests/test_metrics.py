import numpy as np
import unittest

from context import lir

from lir.metrics import devpav_estimated as devpav, devPAVtot_f, calcsurface_f, withoutinf0_f

class TestDevPAV_estimated(unittest.TestCase):
    def test_devpav_error(self):
        lrs = np.ones(10)
        y = np.concatenate([np.ones(10)])
        with self.assertRaises(ValueError):
            devpav(lrs, y, 10)

    def test_devpav(self):
        # naive system
        lrs = np.ones(10)
        y = np.concatenate([np.ones(5), np.zeros(5)])
        self.assertEqual(devpav(lrs, y, 10), 0)

        # badly calibrated naive system
        lrs = 2*np.ones(10)
        y = np.concatenate([np.ones(5), np.zeros(5)])
        self.assertAlmostEqual(devpav(lrs, y, 10), 0)  # TODO: what should be the outcome?

        # infinitely bad calibration
        lrs = np.array([5, 5, 5, .2, .2, .2, np.inf])
        y = np.concatenate([np.ones(3), np.zeros(4)])
        self.assertAlmostEqual(devpav(lrs, y, 10), np.inf)

        # binary system
        lrs = np.array([5, 5, 5, .2, 5, .2, .2, .2])
        y = np.concatenate([np.ones(4), np.zeros(4)])
        self.assertAlmostEqual(devpav(lrs, y, 1000), 0.1390735326086103)  # TODO: check this value externally

        # somewhat normal
        lrs = np.array([6, 5, 5, .2, 5, .2, .2, .1])
        y = np.concatenate([np.ones(4), np.zeros(4)])
        self.assertAlmostEqual(devpav(lrs, y, 1000), 0.262396610197457)  # TODO: check this value externally


class TestDevPAV(unittest.TestCase):
    def test_devPAVtot(self):
        # #test on dummy data: only one finite PAV coordinate
        LRssame = (0.01, 1, 10**3)
        LRsdif = (0.001, 1, 10**2)
        PAVresult = (0, 10, float('inf'), 0, 10, float('inf'))
        self.assertAlmostEqual(devPAVtot_f(LRssame, LRsdif, PAVresult), 1)

        ### tests on data with LLRs -Inf or Inf
        #test on dummy data: de PAV transform loopt tot {-inf, -inf}, evenwijdig aan lijn Y=X
        LRssame = (0, 1, 10**3)
        LRsdif = (0.001, 2, 10**2)
        PAVresult = (0, 10, float('inf'), 0.1, 20, float('inf'))
        self.assertAlmostEqual(devPAVtot_f(LRssame, LRsdif, PAVresult), 2)


        #test on dummy data: de PAV transform loopt tot {-inf, -inf}, maar is daarvoor al -inf
        LRssame = (0, 1, 10**3)
        LRsdif = (0.001, 2, 10**2)
        PAVresult = (0, 10, float('inf'), 0, 20, float('inf'))
        self.assertAlmostEqual(devPAVtot_f(LRssame, LRsdif, PAVresult), float('inf'))

        # #test on dummy data: de PAV transform loopt tot {-inf, finite}
        LRssame = (0, 1, 10**3)
        LRsdif = (0.001, 2, 10**2)
        PAVresult = (0.0001, 10, float('inf'), 0.001, 20, float('inf'))
        self.assertAlmostEqual(devPAVtot_f(LRssame, LRsdif, PAVresult), float('inf'))


        #test on dummy data: de PAV transform loopt tot {inf, inf}, evenwijdig aan lijn Y=X
        LRssame = (0.01, 1, float('inf'))
        LRsdif = (0.001, 2, 10**2)
        PAVresult = (0, 10, float('inf'), 0.1, 20, 100)
        self.assertAlmostEqual(devPAVtot_f(LRssame, LRsdif, PAVresult), 0)


        # #test on dummy data: de PAV transform loopt tot {inf, inf}, maar is daarvoor ook al inf
        LRssame = (0.01, 1, float('inf'))
        LRsdif = (0.001, 2, 10**2)
        PAVresult = (0, 10, float('inf'), 0, 20, float('inf'))
        self.assertAlmostEqual(devPAVtot_f(LRssame, LRsdif, PAVresult), float('inf'))



        # #test on dummy data: de PAV transform loopt tot {inf, finite}
        LRssame = (0.01, 1, float('inf'))
        LRsdif = (0.001, 2, 10**2)
        PAVresult = (0.0001, 10, 10**4, 0.001, 20, 10**3)
        self.assertAlmostEqual(devPAVtot_f(LRssame, LRsdif, PAVresult), float('inf'))



        # #test on dummy data: de PAV transform loopt tot {-inf, -inf} en tot {inf , inf} en is daartussen finite
        LRssame = (0, 1, float('inf'))
        LRsdif = (0.001, 2, 10**2)
        PAVresult = (0, 10, float('inf'), .1, 20, 1000)
        self.assertAlmostEqual(devPAVtot_f(LRssame, LRsdif, PAVresult), 1.5)



        # #test on dummy data: de PAV transform loopt tot {-inf, -inf} en tot {inf , inf} en is daarvoor al -Inf
        LRssame = (0, 1, float('inf'))
        LRsdif = (0.001, 2, 10**2)
        PAVresult = (0, 10, float('inf'), 0, 20, 1000)
        self.assertAlmostEqual(devPAVtot_f(LRssame, LRsdif, PAVresult), float('inf'))


        # #test on dummy data: de PAV transform loopt tot {-inf, -inf} en tot {inf , inf} en is daarvoor al Inf
        LRssame = (0, 1, float('inf'))
        LRsdif = (0.001, 2, 10**2)
        PAVresult = (0, 10, float('inf'), .1, 20, float('inf'))
        self.assertAlmostEqual(devPAVtot_f(LRssame, LRsdif, PAVresult), float('inf'))

        ### tests on ordinary data

        #test on dummy data. This PAV-transform is parallel to the identity line
        LRssame = (1, 10**3)
        LRsdif = (0.1, 10)
        PAVresult = (1, float('inf'), 0, 1)
        self.assertAlmostEqual(devPAVtot_f(LRssame, LRsdif, PAVresult), 0.5)



        #test on dummy data 2, this PAV-transform crosses the identity line
        LRssame = (0.1, 100, 10**3)
        LRsdif = (10**-3, 10**-2, 10)
        PAVresult = (10**-2, 10**2, float('inf'), 0, 10**-3, 10**2)
        self.assertAlmostEqual(devPAVtot_f(LRssame, LRsdif, PAVresult), (1 + 2 * (0.5 * 2 * 1 - 0.5 * 1 * 1) + 0.5)/4)

        # test on dummy data 3, this PAV-transform is finite
        LRssame = (0.1, 100)
        LRsdif = (10**-2, 10)
        PAVresult = (10**-2, 10**2, 10**-3, 10**2)
        self.assertAlmostEqual(devPAVtot_f(LRssame, LRsdif, PAVresult), (1 + 2 * (0.5 * 2 * 1 - 0.5 * 1 * 1) + 0.5)/4)

    def test_calcsurface(self):
        # tests for the calcsurface_f function

        # the line segment is parallel to the identity line
        c1 = (-1,2)
        c2 = (0,3)
        self.assertAlmostEqual(calcsurface_f(c1, c2), 1)

        # 2nd possibility (situation 1 of 2 in code below, the intersection with the identity line is within the line segment, y1 < x1)
        c1 = (-1,-2)
        c2 = (0,3)
        self.assertAlmostEqual(calcsurface_f(c1, c2), 1.25)

        # 3rd possibility (situation 2 of 2 in code below, the intersection with the identity line is within the line segment, y1 >= x1)
        c1 = (0,3)
        c2 = (10,4)
        self.assertAlmostEqual(calcsurface_f(c1, c2), 25)

        # 5th possibility (situation 1 of 4 in code below, both coordinates are below the identity line, intersection with identity line on left hand side)
        c1 = (-1,-2)
        c2 = (0, -1.5)
        self.assertAlmostEqual(calcsurface_f(c1, c2), 1.25)

        # 6th possibility (situation 2 van 4 in code below, both coordinates are above the identity line, intersection with identity line on right hand side)
        c1 = (1,2)
        c2 = (1.5,2)
        self.assertAlmostEqual(calcsurface_f(c1, c2), 0.375)

        # 7th possibility (situation 2 of 4 in code below, both coordinates are above the identity line, intersection with identity line on left hand side)
        c1 = c(1,2)
        c2 = c(2,4)
        self.assertAlmostEqual(calcsurface_f(c1, c2), 1.5)

        # 8th possibility (situation 3 of 4 in code below, both coordinates are below the identity line, intersection with identity line on right hand side)
        c1 = c(-1,-2)
        c2 = c(0,-0.5)
        self.assertAlmostEqual(calcsurface_f(c1, c2), 0.75)

        # situation 4 of 4 in code below should never occur and results in an error message

    # def test_withoutInf0(self):
    #     #test
    #     t = (float('-inf'), 1.0, 2, 3, float('inf'))
    #     self.assertAlmostEqual(withoutinf0_f(t), list[1,2,3])

