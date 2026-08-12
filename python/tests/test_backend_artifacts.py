"""Exercise every laddu distribution installed in the current environment."""

from __future__ import annotations

import importlib.util
import unittest

from installed_artifact_contract import check_installed_artifact


class InstalledArtifactContracts(unittest.TestCase):
    def test_primary_distribution(self) -> None:
        if importlib.util.find_spec('laddu') is None:
            self.skipTest('the primary laddu distribution is not installed')
        expected = 'mpi' if importlib.util.find_spec('_laddu_mpi') is not None else 'local'
        check_installed_artifact('laddu', expected, 'laddu')

    def test_local_adapter_when_installed(self) -> None:
        if importlib.util.find_spec('_laddu_local') is None:
            self.skipTest('laddu-local is not installed')
        check_installed_artifact('_laddu_local', 'local', 'laddu-local')

    def test_mpi_adapter_when_installed(self) -> None:
        if importlib.util.find_spec('_laddu_mpi') is None:
            self.skipTest('laddu-mpi is not installed')
        check_installed_artifact('_laddu_mpi', 'mpi', 'laddu-mpi')


if __name__ == '__main__':
    unittest.main()
