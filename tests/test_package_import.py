from aseams import AMS, CollectiveVariables, MultiWalkerSampler, SingleWalkerSampler


def test_public_package_exports():
    assert AMS.__name__ == "AMS"
    assert CollectiveVariables.__name__ == "CollectiveVariables"
    assert SingleWalkerSampler.__name__ == "SingleWalkerSampler"
    assert MultiWalkerSampler.__name__ == "MultiWalkerSampler"
