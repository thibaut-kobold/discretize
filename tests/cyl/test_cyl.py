import unittest
import numpy as np

import discretize
from discretize import tests, utils

np.random.seed(13)


class TestCylSymmetricMesh(unittest.TestCase):
    def setUp(self):
        hx = np.r_[1, 1, 0.5]
        hz = np.r_[2, 1]
        self.mesh = discretize.CylMesh([hx, 1, hz], np.r_[0.0, 0.0, 0.0])

    def test_dim(self):
        self.assertEqual(self.mesh.dim, 3)

    def test_nC(self):
        self.assertEqual(self.mesh.nC, 6)
        self.assertEqual(self.mesh.nCx, 3)
        self.assertEqual(self.mesh.nCy, 1)
        self.assertEqual(self.mesh.nCz, 2)
        self.assertEqual(self.mesh.vnC, (3, 1, 2))

    def test_nN(self):
        self.assertEqual(self.mesh.nN, 0)
        self.assertEqual(self.mesh.nNx, 3)
        self.assertEqual(self.mesh.nNy, 0)
        self.assertEqual(self.mesh.nNz, 3)
        self.assertEqual(self.mesh.vnN, (3, 0, 3))

    def test_nF(self):
        self.assertEqual(self.mesh.nFx, 6)
        self.assertEqual(self.mesh.vnFx, (3, 1, 2))
        self.assertEqual(self.mesh.nFy, 0)
        self.assertEqual(self.mesh.vnFy, (3, 0, 2))
        self.assertEqual(self.mesh.nFz, 9)
        self.assertEqual(self.mesh.vnFz, (3, 1, 3))
        self.assertEqual(self.mesh.nF, 15)
        self.assertEqual(self.mesh.vnF, (6, 0, 9))

    def test_nE(self):
        self.assertEqual(self.mesh.nEx, 0)
        self.assertEqual(self.mesh.vnEx, (3, 0, 3))
        self.assertEqual(self.mesh.nEy, 9)
        self.assertEqual(self.mesh.vnEy, (3, 1, 3))
        self.assertEqual(self.mesh.nEz, 0)
        self.assertEqual(self.mesh.vnEz, (3, 0, 2))
        self.assertEqual(self.mesh.nE, 9)
        self.assertEqual(self.mesh.vnE, (0, 9, 0))

    def test_vectorsCC(self):
        v = np.r_[0.5, 1.5, 2.25]
        self.assertEqual(np.linalg.norm((v - self.mesh.vectorCCx)), 0)
        v = np.r_[0]
        self.assertEqual(np.linalg.norm((v - self.mesh.vectorCCy)), 0)
        v = np.r_[1, 2.5]
        self.assertEqual(np.linalg.norm((v - self.mesh.vectorCCz)), 0)

    def test_vectorsN(self):
        v = np.r_[1, 2, 2.5]
        self.assertEqual(np.linalg.norm((v - self.mesh.vectorNx)), 0)
        v = np.r_[0]
        self.assertEqual(np.linalg.norm((v - self.mesh.vectorNy)), 0)
        v = np.r_[0, 2, 3.0]
        self.assertEqual(np.linalg.norm((v - self.mesh.vectorNz)), 0)

    def test_edge(self):
        edge = np.r_[1, 2, 2.5, 1, 2, 2.5, 1, 2, 2.5] * 2 * np.pi
        self.assertEqual(np.linalg.norm((edge - self.mesh.edge)), 0)

    def test_area(self):
        r = np.r_[0, 1, 2, 2.5]
        a = r[1:] * 2 * np.pi
        areaX = np.r_[2 * a, a]
        a = (r[1:] ** 2 - r[:-1] ** 2) * np.pi
        areaZ = np.r_[a, a, a]
        area = np.r_[areaX, areaZ]
        self.assertEqual(np.linalg.norm((area - self.mesh.area)), 0)

    def test_vol(self):
        r = np.r_[0, 1, 2, 2.5]
        a = (r[1:] ** 2 - r[:-1] ** 2) * np.pi
        vol = np.r_[2 * a, a]
        self.assertEqual(np.linalg.norm((vol - self.mesh.vol)), 0)

    def test_vol_simple(self):
        mesh = discretize.CylMesh([1.0, 1.0, 1.0])
        self.assertEqual(mesh.vol, np.pi)

        mesh = discretize.CylMesh([2.0, 1.0, 1.0])
        self.assertTrue(np.all(mesh.vol == np.pi * np.r_[0.5 ** 2, 1 - 0.5 ** 2]))

    def test_gridSizes(self):
        self.assertEqual(self.mesh.gridCC.shape, (self.mesh.nC, 3))
        self.assertEqual(self.mesh._nodes_full.shape, (9, 3))

        self.assertEqual(self.mesh.gridFx.shape, (self.mesh.nFx, 3))
        self.assertTrue(self.mesh.gridFy is None)
        self.assertEqual(self.mesh.gridFz.shape, (self.mesh.nFz, 3))

        self.assertTrue(self.mesh.gridEx is None)
        self.assertEqual(self.mesh.gridEy.shape, (self.mesh.nEy, 3))
        self.assertTrue(self.mesh.gridEz is None)

    def test_gridCC(self):
        x = np.r_[0.5, 1.5, 2.25, 0.5, 1.5, 2.25]
        y = np.zeros(6)
        z = np.r_[1, 1, 1, 2.5, 2.5, 2.5]
        G = np.c_[x, y, z]
        self.assertEqual(np.linalg.norm((G - self.mesh.gridCC).ravel()), 0)

    def test_gridN(self):
        x = np.r_[1, 2, 2.5, 1, 2, 2.5, 1, 2, 2.5]
        y = np.zeros(9)
        z = np.r_[0, 0, 0, 2, 2, 2, 3, 3, 3]
        G = np.c_[x, y, z]
        self.assertEqual(np.linalg.norm((G - self.mesh.gridN).ravel()), 0)

    def test_gridFx(self):
        x = np.r_[1, 2, 2.5, 1, 2, 2.5]
        y = np.zeros(6)
        z = np.r_[1, 1, 1, 2.5, 2.5, 2.5]
        G = np.c_[x, y, z]
        self.assertEqual(np.linalg.norm((G - self.mesh.gridFx).ravel()), 0)

    def test_gridFz(self):
        x = np.r_[0.5, 1.5, 2.25, 0.5, 1.5, 2.25, 0.5, 1.5, 2.25]
        y = np.zeros(9)
        z = np.r_[0, 0, 0, 2, 2, 2, 3, 3, 3.0]
        G = np.c_[x, y, z]
        self.assertEqual(np.linalg.norm((G - self.mesh.gridFz).ravel()), 0)

    def test_gridEy(self):
        x = np.r_[1, 2, 2.5, 1, 2, 2.5, 1, 2, 2.5]
        y = np.zeros(9)
        z = np.r_[0, 0, 0, 2, 2, 2, 3, 3, 3.0]
        G = np.c_[x, y, z]
        self.assertEqual(np.linalg.norm((G - self.mesh.gridEy).ravel()), 0)

    def test_lightOperators(self):
        self.assertTrue(self.mesh.nodalGrad is None)

    def test_getInterpMatCartMesh_Cells(self):

        Mr = discretize.TensorMesh([100, 100, 2], x0="CC0")
        Mc = discretize.CylMesh(
            [np.ones(10) / 5, 1, 10], x0="0C0", cartesianOrigin=[-0.2, -0.2, 0]
        )

        mc = np.arange(Mc.nC)
        xr = np.linspace(0, 0.4, 50)
        xc = np.linspace(0, 0.4, 50) + 0.2
        Pr = Mr.getInterpolationMat(
            np.c_[xr, np.ones(50) * -0.2, np.ones(50) * 0.5], "CC"
        )
        Pc = Mc.getInterpolationMat(np.c_[xc, np.zeros(50), np.ones(50) * 0.5], "CC")
        Pc2r = Mc.getInterpolationMatCartMesh(Mr, "CC")

        assert np.abs(Pr * (Pc2r * mc) - Pc * mc).max() < 1e-3

    def test_getInterpMatCartMesh_Cells2Nodes(self):

        Mr = discretize.TensorMesh([100, 100, 2], x0="CC0")
        Mc = discretize.CylMesh(
            [np.ones(10) / 5, 1, 10], x0="0C0", cartesianOrigin=[-0.2, -0.2, 0]
        )

        mc = np.arange(Mc.nC)
        xr = np.linspace(0, 0.4, 50)
        xc = np.linspace(0, 0.4, 50) + 0.2
        Pr = Mr.getInterpolationMat(
            np.c_[xr, np.ones(50) * -0.2, np.ones(50) * 0.5], "N"
        )
        Pc = Mc.getInterpolationMat(np.c_[xc, np.zeros(50), np.ones(50) * 0.5], "CC")
        Pc2r = Mc.getInterpolationMatCartMesh(Mr, "CC", locTypeTo="N")

        assert np.abs(Pr * (Pc2r * mc) - Pc * mc).max() < 1e-3

    def test_getInterpMatCartMesh_Faces(self):

        Mr = discretize.TensorMesh([100, 100, 2], x0="CC0")
        Mc = discretize.CylMesh(
            [np.ones(10) / 5, 1, 10], x0="0C0", cartesianOrigin=[-0.2, -0.2, 0]
        )

        Pf = Mc.getInterpolationMatCartMesh(Mr, "F")
        mf = np.ones(Mc.nF)

        frect = Pf * mf

        fxcc = Mr.aveFx2CC * Mr.r(frect, "F", "Fx")
        fycc = Mr.aveFy2CC * Mr.r(frect, "F", "Fy")
        fzcc = Mr.r(frect, "F", "Fz")

        indX = utils.closestPoints(Mr, [0.45, -0.2, 0.5])
        indY = utils.closestPoints(Mr, [-0.2, 0.45, 0.5])

        TOL = 1e-2
        assert np.abs(float(fxcc[indX]) - 1) < TOL
        assert np.abs(float(fxcc[indY]) - 0) < TOL
        assert np.abs(float(fycc[indX]) - 0) < TOL
        assert np.abs(float(fycc[indY]) - 1) < TOL
        assert np.abs((fzcc - 1).sum()) < TOL

        mag = (fxcc ** 2 + fycc ** 2) ** 0.5
        dist = ((Mr.gridCC[:, 0] + 0.2) ** 2 + (Mr.gridCC[:, 1] + 0.2) ** 2) ** 0.5

        assert np.abs(mag[dist > 0.1].max() - 1) < TOL
        assert np.abs(mag[dist > 0.1].min() - 1) < TOL

    def test_getInterpMatCartMesh_Faces2Edges(self):

        Mr = discretize.TensorMesh([100, 100, 2], x0="CC0")
        Mc = discretize.CylMesh(
            [np.ones(10) / 5, 1, 10], x0="0C0", cartesianOrigin=[-0.2, -0.2, 0]
        )

        self.assertTrue((Mc.cartesianOrigin == [-0.2, -0.2, 0]).all())

        Pf2e = Mc.getInterpolationMatCartMesh(Mr, "F", locTypeTo="E")
        mf = np.ones(Mc.nF)

        ecart = Pf2e * mf

        excc = Mr.aveEx2CC * Mr.r(ecart, "E", "Ex")
        eycc = Mr.aveEy2CC * Mr.r(ecart, "E", "Ey")
        ezcc = Mr.r(ecart, "E", "Ez")

        indX = utils.closestPoints(Mr, [0.45, -0.2, 0.5])
        indY = utils.closestPoints(Mr, [-0.2, 0.45, 0.5])

        TOL = 1e-2
        assert np.abs(float(excc[indX]) - 1) < TOL
        assert np.abs(float(excc[indY]) - 0) < TOL
        assert np.abs(float(eycc[indX]) - 0) < TOL
        assert np.abs(float(eycc[indY]) - 1) < TOL
        assert np.abs((ezcc - 1).sum()) < TOL

        mag = (excc ** 2 + eycc ** 2) ** 0.5
        dist = ((Mr.gridCC[:, 0] + 0.2) ** 2 + (Mr.gridCC[:, 1] + 0.2) ** 2) ** 0.5

        assert np.abs(mag[dist > 0.1].max() - 1) < TOL
        assert np.abs(mag[dist > 0.1].min() - 1) < TOL

    def test_getInterpMatCartMesh_Edges(self):

        Mr = discretize.TensorMesh([100, 100, 2], x0="CC0")
        Mc = discretize.CylMesh(
            [np.ones(10) / 5, 1, 10], x0="0C0", cartesianOrigin=[-0.2, -0.2, 0]
        )

        Pe = Mc.getInterpolationMatCartMesh(Mr, "E")
        me = np.ones(Mc.nE)

        ecart = Pe * me

        excc = Mr.aveEx2CC * Mr.r(ecart, "E", "Ex")
        eycc = Mr.aveEy2CC * Mr.r(ecart, "E", "Ey")
        ezcc = Mr.aveEz2CC * Mr.r(ecart, "E", "Ez")

        indX = utils.closestPoints(Mr, [0.45, -0.2, 0.5])
        indY = utils.closestPoints(Mr, [-0.2, 0.45, 0.5])

        TOL = 1e-2
        assert np.abs(float(excc[indX]) - 0) < TOL
        assert np.abs(float(excc[indY]) + 1) < TOL
        assert np.abs(float(eycc[indX]) - 1) < TOL
        assert np.abs(float(eycc[indY]) - 0) < TOL
        assert np.abs(ezcc.sum()) < TOL

        mag = (excc ** 2 + eycc ** 2) ** 0.5
        dist = ((Mr.gridCC[:, 0] + 0.2) ** 2 + (Mr.gridCC[:, 1] + 0.2) ** 2) ** 0.5

        assert np.abs(mag[dist > 0.1].max() - 1) < TOL
        assert np.abs(mag[dist > 0.1].min() - 1) < TOL

    def test_getInterpMatCartMesh_Edges2Faces(self):

        Mr = discretize.TensorMesh([100, 100, 2], x0="CC0")
        Mc = discretize.CylMesh(
            [np.ones(10) / 5, 1, 10], x0="0C0", cartesianOrigin=[-0.2, -0.2, 0]
        )

        Pe2f = Mc.getInterpolationMatCartMesh(Mr, "E", locTypeTo="F")
        me = np.ones(Mc.nE)

        frect = Pe2f * me

        excc = Mr.aveFx2CC * Mr.r(frect, "F", "Fx")
        eycc = Mr.aveFy2CC * Mr.r(frect, "F", "Fy")
        ezcc = Mr.r(frect, "F", "Fz")

        indX = utils.closestPoints(Mr, [0.45, -0.2, 0.5])
        indY = utils.closestPoints(Mr, [-0.2, 0.45, 0.5])

        TOL = 1e-2
        assert np.abs(float(excc[indX]) - 0) < TOL
        assert np.abs(float(excc[indY]) + 1) < TOL
        assert np.abs(float(eycc[indX]) - 1) < TOL
        assert np.abs(float(eycc[indY]) - 0) < TOL
        assert np.abs(ezcc.sum()) < TOL

        mag = (excc ** 2 + eycc ** 2) ** 0.5
        dist = ((Mr.gridCC[:, 0] + 0.2) ** 2 + (Mr.gridCC[:, 1] + 0.2) ** 2) ** 0.5

        assert np.abs(mag[dist > 0.1].max() - 1) < TOL
        assert np.abs(mag[dist > 0.1].min() - 1) < TOL

    def test_serialization(self):
        mesh = discretize.CylMesh.deserialize(self.mesh.serialize())
        self.assertTrue(np.all(self.mesh.x0 == mesh.x0))
        self.assertTrue(np.all(self.mesh.shape_cells == mesh.shape_cells))
        self.assertTrue(np.all(self.mesh.hx == mesh.hx))
        self.assertTrue(np.all(self.mesh.hy == mesh.hy))
        self.assertTrue(np.all(self.mesh.hz == mesh.hz))
        self.assertTrue(np.all(self.mesh.gridCC == mesh.gridCC))


MESHTYPES = ["uniform_symmetric_CylMesh"]
call2 = lambda fun, xyz: fun(xyz[:, 0], xyz[:, 2])
call3 = lambda fun, xyz: fun(xyz[:, 0], xyz[:, 1], xyz[:, 2])
cyl_row2 = lambda g, xfun, yfun: np.c_[call2(xfun, g), call2(yfun, g)]
cyl_row3 = lambda g, xfun, yfun, zfun: np.c_[
    call3(xfun, g), call3(yfun, g), call3(zfun, g)
]
cylF2 = lambda M, fx, fy: np.vstack(
    (cyl_row2(M.gridFx, fx, fy), cyl_row2(M.gridFz, fx, fy))
)


class TestFaceDiv2D(tests.OrderTest):
    name = "FaceDiv"
    meshTypes = MESHTYPES
    meshDimension = 3

    def getError(self):

        funR = lambda r, z: np.sin(2.0 * np.pi * r)
        funZ = lambda r, z: np.sin(2.0 * np.pi * z)

        sol = lambda r, t, z: (
            2 * np.pi * r * np.cos(2 * np.pi * r) + np.sin(2 * np.pi * r)
        ) / r + 2 * np.pi * np.cos(2 * np.pi * z)

        Fc = cylF2(self.M, funR, funZ)
        Fc = np.c_[Fc[:, 0], np.zeros(self.M.nF), Fc[:, 1]]
        F = self.M.projectFaceVector(Fc)

        divF = self.M.faceDiv.dot(F)
        divF_ana = call3(sol, self.M.gridCC)

        err = np.linalg.norm((divF - divF_ana), np.inf)
        return err

    def test_order(self):
        self.orderTest()


class TestEdgeCurl2D(tests.OrderTest):
    name = "EdgeCurl"
    meshTypes = MESHTYPES
    meshDimension = 3

    def getError(self):
        # To Recreate or change the functions:

        # import sympy
        # r, t, z = sympy.symbols('r, t, z')

        # fR = 0
        # fZ = 0
        # fT = sympy.sin(2.*sympy.pi*z)

        # print(1/r*sympy.diff(fZ, t) - sympy.diff(fT, z))
        # print(sympy.diff(fR, z) - sympy.diff(fZ, r))
        # print(1/r*(sympy.diff(r*fT, r) - sympy.diff(fR, t)))

        funT = lambda r, t, z: np.sin(2.0 * np.pi * z)

        solR = lambda r, z: -2.0 * np.pi * np.cos(2.0 * np.pi * z)
        solZ = lambda r, z: np.sin(2.0 * np.pi * z) / r

        E = call3(funT, self.M.gridEy)

        curlE = self.M.edgeCurl.dot(E)

        Fc = cylF2(self.M, solR, solZ)
        Fc = np.c_[Fc[:, 0], np.zeros(self.M.nF), Fc[:, 1]]
        curlE_ana = self.M.projectFaceVector(Fc)

        err = np.linalg.norm((curlE - curlE_ana), np.inf)
        return err

    def test_order(self):
        self.orderTest()


class TestCellGrad2D_Dirichlet(unittest.TestCase):
    # name = "Cell Grad 2 - Dirichlet"
    # meshTypes = MESHTYPES
    # meshDimension = 2
    # meshSizes = [8, 16, 32, 64]

    # def getError(self):
    #     #Test function
    #     fx = lambda x, z: -2*np.pi*np.sin(2*np.pi*x)*np.cos(2*np.pi*z)
    #     fz = lambda x, z: -2*np.pi*np.sin(2*np.pi*z)*np.cos(2*np.pi*x)
    #     sol = lambda x, z: np.cos(2*np.pi*x)*np.cos(2*np.pi*z)

    #     xc = call2(sol, self.M.gridCC)

    #     Fc = cylF2(self.M, fx, fz)
    #     Fc = np.c_[Fc[:, 0], np.zeros(self.M.nF), Fc[:, 1]]
    #     gradX_ana = self.M.projectFaceVector(Fc)

    #     gradX = self.M.cellGrad.dot(xc)

    #     err = np.linalg.norm((gradX-gradX_ana), np.inf)

    #     return err

    # def test_order(self):
    #     self.orderTest()

    def setUp(self):
        hx = np.random.rand(10)
        hz = np.random.rand(10)
        self.mesh = discretize.CylMesh([hx, 1, hz])

    def test_NotImplementedError(self):
        with self.assertRaises(NotImplementedError):
            self.mesh.cellGrad


class TestAveragingSimple(unittest.TestCase):
    def setUp(self):
        hx = np.random.rand(10)
        hz = np.random.rand(10)
        self.mesh = discretize.CylMesh([hx, 1, hz])

    def test_simpleEdges(self):
        edge_vec = self.mesh.gridEy[:, 0]
        assert (
            np.linalg.norm(self.mesh.aveE2CC * edge_vec - self.mesh.gridCC[:, 0])
            < 1e-10
        )
        assert (
            np.linalg.norm(self.mesh.aveE2CCV * edge_vec - self.mesh.gridCC[:, 0])
            < 1e-10
        )

    def test_constantFaces(self):
        face_vec = np.hstack([self.mesh.gridFx[:, 0], self.mesh.gridFz[:, 0]])
        assert np.linalg.norm(
            self.mesh.aveF2CCV * face_vec - np.hstack(2 * [self.mesh.gridCC[:, 2]])
        )


class TestAveE2CC(tests.OrderTest):
    name = "aveE2CC"
    meshTypes = MESHTYPES
    meshDimension = 3
    meshSizes = [8, 16, 32, 64]

    def getError(self):

        fun = lambda r, t, z: np.sin(2.0 * np.pi * z) * np.sin(np.pi * r)

        E = call3(fun, self.M.gridEy)

        aveE = self.M.aveE2CC * E
        aveE_ana = call3(fun, self.M.gridCC)

        err = np.linalg.norm((aveE - aveE_ana), np.inf)
        return err

    def test_order(self):
        self.orderTest()


class TestAveE2CCV(tests.OrderTest):
    name = "aveE2CCV"
    meshTypes = MESHTYPES
    meshDimension = 3
    meshSizes = [8, 16, 32, 64]

    def getError(self):

        fun = lambda r, t, z: np.sin(2.0 * np.pi * z) * np.sin(2 * np.pi * r)

        E = call3(fun, self.M.gridEy)

        aveE = self.M.aveE2CCV * E
        aveE_ana = call3(fun, self.M.gridCC)

        err = np.linalg.norm((aveE - aveE_ana), np.inf)
        return err

    def test_order(self):
        self.orderTest()


class TestAveF2CCV(tests.OrderTest):
    name = "aveF2CCV"
    meshTypes = MESHTYPES
    meshDimension = 3

    def getError(self):

        funR = lambda r, z: np.sin(2.0 * np.pi * z) * np.sin(np.pi * r)
        funZ = lambda r, z: np.sin(3.0 * np.pi * z) * np.sin(2.0 * np.pi * r)

        Fc = cylF2(self.M, funR, funZ)
        Fc = np.c_[Fc[:, 0], np.zeros(self.M.nF), Fc[:, 1]]
        F = self.M.projectFaceVector(Fc)

        aveF = self.M.aveF2CCV * F

        aveF_anaR = funR(self.M.gridCC[:, 0], self.M.gridCC[:, 2])
        aveF_anaZ = funZ(self.M.gridCC[:, 0], self.M.gridCC[:, 2])

        aveF_ana = np.hstack([aveF_anaR, aveF_anaZ])

        err = np.linalg.norm((aveF - aveF_ana), np.inf)
        return err

    def test_order(self):
        self.orderTest()


class TestAveF2CC(tests.OrderTest):
    name = "aveF2CC"
    meshTypes = MESHTYPES
    meshDimension = 3

    def getError(self):

        fun = lambda r, z: np.sin(2.0 * np.pi * z) * np.sin(np.pi * r)

        Fc = cylF2(self.M, fun, fun)
        Fc = np.c_[Fc[:, 0], np.zeros(self.M.nF), Fc[:, 1]]
        F = self.M.projectFaceVector(Fc)

        aveF = self.M.aveF2CC * F
        aveF_ana = fun(self.M.gridCC[:, 0], self.M.gridCC[:, 2])

        err = np.linalg.norm((aveF - aveF_ana), np.inf)
        return err

    def test_order(self):
        self.orderTest()


class TestInnerProducts2D(tests.OrderTest):
    """Integrate an function over a unit cube domain using edgeInnerProducts and faceInnerProducts."""

    meshTypes = MESHTYPES
    meshDimension = 3
    meshSizes = [4, 8, 16, 32, 64, 128]

    def getError(self):

        funR = lambda r, t, z: np.cos(2.0 * np.pi * z)
        funT = lambda r, t, z: 0 * t
        funZ = lambda r, t, z: np.sin(2.0 * np.pi * r)


#         call = lambda fun, xyz: fun(xyz[:, 0], xyz[:, 1], xyz[:, 2])

#         sigma1 = lambda r, t, z: z+1
#         sigma2 = lambda r, t, z: r*z+50
#         sigma3 = lambda r, t, z: 3+t*r
#         sigma4 = lambda r, t, z: 0.1*r*t*z
#         sigma5 = lambda r, t, z: 0.2*z*r*t
#         sigma6 = lambda r, t, z: 0.1*t

#         Gc = self.M.gridCC
#         if self.sigmaTest == 1:
#             sigma = np.c_[call(sigma1, Gc)]
#             analytic = 144877./360  # Found using sympy. z=5
#         elif self.sigmaTest == 2:
#             sigma = np.c_[call(sigma1, Gc), call(sigma2, Gc)]
#             analytic = 189959./120  # Found using sympy. z=5
#         elif self.sigmaTest == 3:
#             sigma = np.r_[call(sigma1, Gc), call(sigma2, Gc), call(sigma3, Gc)]
#             analytic = 781427./360  # Found using sympy. z=5

#         if self.location == 'edges':
#             E = call(funT, self.M.gridEy)
#             A = self.M.getEdgeInnerProduct(sigma)
#             numeric = E.T.dot(A.dot(E))
#         elif self.location == 'faces':
#             Fr = call(funR, self.M.gridFx)
#             Fz = call(funZ, self.M.gridFz)
#             A = self.M.getFaceInnerProduct(sigma)
#             F = np.r_[Fr, Fz]
#             numeric = F.T.dot(A.dot(F))

#         print(numeric)
#         err = np.abs(numeric - analytic)
#         return err

#     def test_order1_faces(self):
#         self.name = "2D Face Inner Product - Isotropic"
#         self.location = 'faces'
#         self.sigmaTest = 1
#         self.orderTest()


class TestCyl3DMesh(unittest.TestCase):
    def setUp(self):
        hx = np.r_[1, 1, 0.5]
        hy = np.r_[np.pi, np.pi]
        hz = np.r_[2, 1]
        self.mesh = discretize.CylMesh([hx, hy, hz])

    def test_dim(self):
        self.assertEqual(self.mesh.dim, 3)

    def test_nC(self):
        self.assertEqual(self.mesh.nCx, 3)
        self.assertEqual(self.mesh.nCy, 2)
        self.assertEqual(self.mesh.nCz, 2)
        self.assertEqual(self.mesh.vnC, (3, 2, 2))

    def test_nN(self):
        self.assertEqual(self.mesh.nN, 21)
        self.assertEqual(self.mesh.nNx, 4)
        self.assertEqual(self.mesh.nNy, 2)
        self.assertEqual(self.mesh.nNz, 3)
        self.assertEqual(self.mesh.vnN, (4, 2, 3))

    def test_nF(self):
        self.assertEqual(self.mesh.nFx, 12)
        self.assertEqual(self.mesh.vnFx, (3, 2, 2))
        self.assertEqual(self.mesh.nFy, 12)
        self.assertEqual(self.mesh.vnFy, (3, 2, 2))
        self.assertEqual(self.mesh.nFz, 18)
        self.assertEqual(self.mesh.vnFz, (3, 2, 3))
        self.assertEqual(self.mesh.nF, 42)
        self.assertEqual(self.mesh.vnF, (12, 12, 18))

    def test_nE(self):
        self.assertEqual(self.mesh.nEx, 18)
        self.assertEqual(self.mesh.vnEx, (3, 2, 3))
        self.assertEqual(self.mesh.nEy, 18)
        self.assertEqual(self.mesh.vnEy, (3, 2, 3))
        self.assertEqual(self.mesh.nEz, 12 + 2)
        self.assertEqual(self.mesh.vnEz, (4, 2, 2))
        self.assertEqual(self.mesh.nE, 50)
        self.assertEqual(self.mesh.vnE, (18, 18, 14))

    def test_vectorsCC(self):
        v = np.r_[0.5, 1.5, 2.25]
        self.assertEqual(np.linalg.norm((v - self.mesh.vectorCCx)), 0)
        v = np.r_[0, np.pi] + np.pi / 2
        self.assertEqual(np.linalg.norm((v - self.mesh.vectorCCy)), 0)
        v = np.r_[1, 2.5]
        self.assertEqual(np.linalg.norm((v - self.mesh.vectorCCz)), 0)

    def test_vectorsN(self):
        v = np.r_[0, 1, 2, 2.5]
        self.assertEqual(np.linalg.norm((v - self.mesh.vectorNx)), 0)
        v = np.r_[0.0, np.pi]
        self.assertEqual(np.linalg.norm((v - self.mesh.vectorNy)), 0)
        v = np.r_[0, 2, 3]
        self.assertEqual(np.linalg.norm((v - self.mesh.vectorNz)), 0)


if __name__ == "__main__":
    unittest.main()
