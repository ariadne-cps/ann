from pyariadne import neural_version


def test_neural_bootstrap():
    assert neural_version() == "0.1.0"
