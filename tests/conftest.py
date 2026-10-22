import pytest

from causetrail.datagen import generate
from causetrail.sources import LocalGateway


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("demo")
    generate(d, seed=7)
    return d


@pytest.fixture(scope="session")
def gw(data_dir):
    return LocalGateway(data_dir)
