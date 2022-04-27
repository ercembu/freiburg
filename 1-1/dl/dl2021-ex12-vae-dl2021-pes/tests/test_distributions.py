from lib.distributions import std_normal, normal


def test_distributions():
    # The approximation method is inaccurate so we need lots of samples and a high error tolerance
    n_samples = 10000
    epsilon = 0.2

    # Test creation of standard normal distribution by sampling a uniform distribution.
    target_mean, target_stddev = 0., 1.
    samples = std_normal(n_samples)
    mean = samples.mean()
    var = samples.var()
    assert (mean - target_mean) ** 2 < epsilon and (var - target_stddev ** 2) ** 2 < epsilon * target_stddev, (
        f"Mean and variance should be {target_mean} and {target_stddev ** 2} but are {mean} and {var}")

    # Test normal distribution
    target_mean, target_stddev = 1., 3.
    samples = normal(target_mean, target_stddev, n_samples)
    mean = samples.mean()
    var = samples.var()
    assert (mean - target_mean) ** 2 < epsilon and (var - target_stddev ** 2) ** 2 < epsilon * target_stddev, (
        f"Mean and variance should be {target_mean} and {target_stddev ** 2} but are {mean} and {var}")


if __name__ == "__main__":
    test_distributions()
    print('Test complete.')
