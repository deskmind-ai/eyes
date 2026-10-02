"""The grounding server's token check, without loading a model (deskmind_eyes.ground_server.authorized)."""

from deskmind_eyes.ground_server import authorized


def test_no_token_configured_lets_everything_through():
    assert authorized(None, "")
    assert authorized("Bearer anything", "")


def test_a_configured_token_must_match():
    assert authorized("Bearer s3cret", "s3cret")
    assert not authorized(None, "s3cret")
    assert not authorized("Bearer wrong", "s3cret")
    assert not authorized("s3cret", "s3cret")


if __name__ == "__main__":
    test_no_token_configured_lets_everything_through()
    test_a_configured_token_must_match()
    print("ok")
