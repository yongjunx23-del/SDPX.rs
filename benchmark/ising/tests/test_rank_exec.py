import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('rank_exec', Path(__file__).parents[1] / 'rank_exec.py')
rank_exec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rank_exec)


class RankBindingTests(unittest.TestCase):
    def env(self, rank=1, local_rank=1, size=4, local_size=4):
        return dict(OMPI_COMM_WORLD_RANK=str(rank), OMPI_COMM_WORLD_LOCAL_RANK=str(local_rank),
                    OMPI_COMM_WORLD_SIZE=str(size), OMPI_COMM_WORLD_LOCAL_SIZE=str(local_size),
                    SDPX_RANK_CPUS='0,2,4,6')

    def test_existing_single_node_binding(self):
        self.assertEqual(rank_exec.selected_cpu(self.env(), 'node-a', [0, 2, 4, 6]), (1, 1, 2))

    def test_multinode_requires_host_map(self):
        with self.assertRaises(ValueError):
            rank_exec.selected_cpu(self.env(size=8), 'node-b', list(range(8)))

    def test_global_rank_and_host_specific_cpu(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'cpus.json'
            p.write_text(json.dumps({'node-a': [0, 2, 4, 6], 'node-b': [8, 10, 12, 14]}))
            env = self.env(rank=5, local_rank=1, size=8)
            env['SDPX_RANK_CPU_MAP'] = str(p)
            self.assertEqual(rank_exec.selected_cpu(env, 'node-b', [8, 10, 12, 14]), (5, 1, 10))
            with self.assertRaises(KeyError):
                rank_exec.selected_cpu(env, 'unknown-node', [0, 2, 4, 6])

    def test_oversubscription_and_outside_affinity(self):
        for cpus, allowed in [('0,0,4,6', [0, 4, 6]), ('0,2,4,6', [0, 4, 6]), ('0', [0])]:
            env = self.env(); env['SDPX_RANK_CPUS'] = cpus
            with self.assertRaises(ValueError):
                rank_exec.selected_cpu(env, 'node-a', allowed)

    def test_invalid_rank(self):
        with self.assertRaises(ValueError):
            rank_exec.selected_cpu(self.env(rank=4), 'node-a', [0, 2, 4, 6])


if __name__ == '__main__':
    unittest.main()
