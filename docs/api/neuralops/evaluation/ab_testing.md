Module neuralops.evaluation.ab_testing
======================================

Classes
-------

`ABTestResult(metric: str, mean_a: float, mean_b: float, difference: float, p_value: float, significant: bool, better_model: str, confidence_interval: tuple[float, float])`
:   ABTestResult(metric: str, mean_a: float, mean_b: float, difference: float, p_value: float, significant: bool, better_model: str, confidence_interval: tuple[float, float])

    ### Instance variables

    `better_model: str`
    :   The type of the None singleton.

    `confidence_interval: tuple[float, float]`
    :   The type of the None singleton.

    `difference: float`
    :   The type of the None singleton.

    `mean_a: float`
    :   The type of the None singleton.

    `mean_b: float`
    :   The type of the None singleton.

    `metric: str`
    :   The type of the None singleton.

    `p_value: float`
    :   The type of the None singleton.

    `significant: bool`
    :   The type of the None singleton.

`ABTester(alpha: float = 0.05)`
:   

    ### Methods

    `compare(self, results_a: dict[str, list[float]], results_b: dict[str, list[float]]) ‑> list[neuralops.evaluation.ab_testing.ABTestResult]`
    :   Compare.