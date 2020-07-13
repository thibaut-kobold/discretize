import numpy as np
import unittest
import discretize
from discretize.utils import volume_average
from numpy.testing import assert_array_equal, assert_allclose



class TestVolumeAverage(unittest.TestCase):

    def test_tensor_to_tensor(self):
        h1 = np.random.rand(16)
        h1 /= h1.sum()
        h2 = np.random.rand(16)
        h2 /= h2.sum()

        h1s = []
        h2s = []
        for i in range(3):
            print(f"Tensor to Tensor {i+1}D: ", end="")
            h1s.append(h1)
            h2s.append(h2)
            mesh1 = discretize.TensorMesh(h1s)
            mesh2 = discretize.TensorMesh(h2s)

            in_put = np.random.rand(mesh1.nC)
            out_put = np.empty(mesh2.nC)
            # test the three ways of calling...
            out1 = volume_average(mesh1, mesh2, in_put, out_put)
            assert_array_equal(out1, out_put)

            out2 = volume_average(mesh1, mesh2, in_put)
            assert_allclose(out1, out2)

            Av = volume_average(mesh1, mesh2)
            out3 = Av@in_put
            assert_allclose(out1, out3)

            vol1 = np.sum(mesh1.vol*in_put)
            vol2 = np.sum(mesh2.vol*out3)
            print(vol1, vol2)
            self.assertAlmostEqual(vol1, vol2)

    def test_tree_to_tree(self):
        h1 = np.random.rand(16)
        h1 /= h1.sum()
        h2 = np.random.rand(16)
        h2 /= h2.sum()

        h1s = [h1]
        h2s = [h2]
        insert_1 = [0.25]
        insert_2 = [0.75]
        for i in range(1, 3):
            print(f"Tree to Tree {i+1}D: ", end="")
            h1s.append(h1)
            h2s.append(h2)
            insert_1.append(0.25)
            insert_2.append(0.75)
            mesh1 = discretize.TreeMesh(h1s)
            mesh1.insert_cells([insert_1], [4])
            mesh2 = discretize.TreeMesh(h2s)
            mesh2.insert_cells([insert_2], [4])

            in_put = np.random.rand(mesh1.nC)
            out_put = np.empty(mesh2.nC)
            # test the three ways of calling...
            out1 = volume_average(mesh1, mesh2, in_put, out_put)
            assert_array_equal(out1, out_put)

            out2 = volume_average(mesh1, mesh2, in_put)
            assert_allclose(out1, out2)

            Av = volume_average(mesh1, mesh2)
            out3 = Av@in_put
            assert_allclose(out1, out3)

            vol1 = np.sum(mesh1.vol*in_put)
            vol2 = np.sum(mesh2.vol*out3)
            print(vol1, vol2)
            self.assertAlmostEqual(vol1, vol2)

    def test_tree_to_tensor(self):
        h1 = np.random.rand(16)
        h1 /= h1.sum()
        h2 = np.random.rand(16)
        h2 /= h2.sum()

        h1s = [h1]
        h2s = [h2]
        insert_1 = [0.25]
        for i in range(1, 3):
            print(f"Tree to Tensor {i+1}D: ", end="")
            h1s.append(h1)
            h2s.append(h2)
            insert_1.append(0.25)
            mesh1 = discretize.TreeMesh(h1s)
            mesh1.insert_cells([insert_1], [4])
            mesh2 = discretize.TensorMesh(h2s)

            in_put = np.random.rand(mesh1.nC)
            out_put = np.empty(mesh2.nC)
            # test the three ways of calling...
            out1 = volume_average(mesh1, mesh2, in_put, out_put)
            assert_array_equal(out1, out_put)

            out2 = volume_average(mesh1, mesh2, in_put)
            assert_allclose(out1, out2)

            Av = volume_average(mesh1, mesh2)
            out3 = Av@in_put
            assert_allclose(out1, out3)

            vol1 = np.sum(mesh1.vol*in_put)
            vol2 = np.sum(mesh2.vol*out3)
            print(vol1, vol2)
            self.assertAlmostEqual(vol1, vol2)

    def test_tensor_to_tree(self):
        h1 = np.random.rand(16)
        h1 /= h1.sum()
        h2 = np.random.rand(16)
        h2 /= h2.sum()

        h1s = [h1]
        h2s = [h2]
        insert_2 = [0.75]
        for i in range(1, 3):
            print(f"Tensor to Tree {i+1}D: ", end="")
            h1s.append(h1)
            h2s.append(h2)
            insert_2.append(0.75)
            mesh1 = discretize.TensorMesh(h1s)
            mesh2 = discretize.TreeMesh(h2s)
            mesh2.insert_cells([insert_2], [4])

            in_put = np.random.rand(mesh1.nC)
            out_put = np.empty(mesh2.nC)
            # test the three ways of calling...
            out1 = volume_average(mesh1, mesh2, in_put, out_put)
            assert_array_equal(out1, out_put)

            out2 = volume_average(mesh1, mesh2, in_put)
            assert_allclose(out1, out2)

            Av = volume_average(mesh1, mesh2)
            out3 = Av@in_put
            assert_allclose(out1, out3)

            vol1 = np.sum(mesh1.vol*in_put)
            vol2 = np.sum(mesh2.vol*out3)
            print(vol1, vol2)
            self.assertAlmostEqual(vol1, vol2)

    def test_errors(self):
        h1 = np.random.rand(16)
        h1 /= h1.sum()
        h2 = np.random.rand(16)
        h2 /= h2.sum()
        mesh1D = discretize.TensorMesh([h1])
        mesh2D = discretize.TensorMesh([h1, h1])
        mesh3D = discretize.TensorMesh([h1, h1, h1])


        hr = np.r_[1, 1, 0.5]
        hz = np.r_[2, 1]
        meshCyl = discretize.CylMesh([hr, 1, hz], np.r_[0., 0., 0.])
        mesh2 = discretize.TreeMesh([h2, h2])
        mesh2.insert_cells([0.75, 0.75], [4])

        with self.assertRaises(TypeError):
            # Gives a wrong typed object to the function
            volume_average(mesh1D, h1)
        with self.assertRaises(NotImplementedError):
            # Gives a wrong typed mesh
            volume_average(meshCyl, mesh2)
        with self.assertRaises(ValueError):
            # Gives mismatching mesh dimensions
            volume_average(mesh2D, mesh3D)

        model1 = np.random.randn(mesh2D.nC)
        bad_model1 = np.random.randn(3)
        bad_model2 = np.random.rand(1)
        # gives input values with incorrect lengths
        with self.assertRaises(ValueError):
            volume_average(mesh2D, mesh2, bad_model1)
        with self.assertRaises(ValueError):
            volume_average(mesh2D, mesh2, model1, bad_model2)



if __name__ == '__main__':
    unittest.main()
