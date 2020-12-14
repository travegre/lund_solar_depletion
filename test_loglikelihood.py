import work
import pytest
import numpy as np


def test():
    np.random.seed(100)
    work.mcmc_fit(0, 1, 1, testing=True)
    assert work.lnprob(work.p0[0]) == pytest.approx(1670.895460353493)
