import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH


def get_configspace() -> CS.ConfigurationSpace:
    """ Define a conditional hyperparameter search-space.

    hyperparameters:
        lr              from 1e-6 to 1e-0 (log, float) and default=1e-2
        num_filters_1   from    2 to    8 (int) and default=5
        num_filters_2   from    2 to    8 (int) and default=5
        num_conv_layers from    1 to    2 (int) and default=1

    conditions:
        include num_filters_2 only if num_conv_layers > 1

    Returns:
        Configurationspace

    Note:
        please name the hyperparameters as given above.

    Hint:
        use example = CS.GreaterThanCondition(..,..,..) and then
        cs.add_condition(example) to add a conditional hyperparameter
        for num_filters_1 and num_filters_2.
    """
    cs = CS.ConfigurationSpace(seed=0)
    # START TODO #################
    lr = CSH.UniformFloatHyperparameter('lr', lower = 1e-6, upper = 1e-0, log = True, default_value = 1e-2)
    cs.add_hyperparameter(lr)

    num_filters_1 = CSH.UniformIntegerHyperparameter('num_filters_1', lower = 2, upper = 8, default_value = 5)
    cs.add_hyperparameter(num_filters_1)

    num_filters_2 = CSH.UniformIntegerHyperparameter('num_filters_2', lower = 2, upper = 8, default_value = 5)
    cs.add_hyperparameter(num_filters_2)

    num_conv_layers = CSH.UniformIntegerHyperparameter('num_conv_layers', lower = 1, upper = 2, default_value = 1)
    cs.add_hyperparameter(num_conv_layers)

    cond = CS.GreaterThanCondition(num_filters_2, num_conv_layers, 1)
    cs.add_condition(cond)
    # END TODO ###################
    return cs
