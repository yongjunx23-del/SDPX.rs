"""Check convention/signs using hand-written operators, without solving."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
try:
    import numpy as np
    from scipy.sparse import csc_matrix
except ImportError:
    np = None
else:
    spec = importlib.util.spec_from_file_location('import_sedumi', Path(__file__).parents[1]/'import_sedumi.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)


@unittest.skipIf(np is None, "optional NumPy/SciPy input importer dependencies")
class ImportTests(unittest.TestCase):
    def test_free_orthant_soc_and_asymmetric_psd_coefficients(self):
        k = SimpleNamespace(_fieldnames=['f', 'l', 'q', 's'], f=1, l=1, q=3, s=2)
        # Coordinates: free, orthant, SOC3, full column-major PSD2.
        D = np.arange(1., 19.).reshape(2, 9)
        q = np.arange(9.)
        result = m.convert_data(dict(A=csc_matrix(D), c=q, b=np.array([4., 7.]), K=k))
        a = result['A']
        A = csc_matrix((a['nzval'], a['rowval'], a['colptr']), shape=(a['m'], a['n']))
        x = np.array([2., 3., 5., 1., 2., 7., np.sqrt(2.)*11., 13.])
        full = np.array([2., 3., 5., 1., 2., 7., 11., 11., 13.])
        np.testing.assert_allclose((A@x)[:2], D@full, rtol=1e-15)
        np.testing.assert_array_equal((A@x)[2:], -x[1:])
        self.assertAlmostEqual(np.dot(result['q'], x), q@full, places=12)
        self.assertEqual(result['cones'], [{'ZeroConeT':2}, {'NonnegativeConeT':1},
                                          {'SecondOrderConeT':3}, {'PSDTriangleConeT':2}])

    def test_unsupported_data_rejected(self):
        for k, A in [(SimpleNamespace(_fieldnames=['r'], r=3), csc_matrix((0,3))),
                     (SimpleNamespace(_fieldnames=['s'], s=2), csc_matrix((0,3))),
                     (SimpleNamespace(_fieldnames=['l'], l=3), csc_matrix([[1j,0,0]]))]:
            with self.assertRaises(ValueError):
                m.convert_data(dict(A=A, c=np.zeros(A.shape[1]), b=np.zeros(A.shape[0]), K=k))


if __name__ == '__main__':
    unittest.main()
