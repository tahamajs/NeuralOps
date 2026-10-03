Module neuralops.data.validation
================================
Simple data validation – missing values, infinite, zero variance, schema checks.

Classes
-------

`DataValidator(schema: dict[str, typing.Any] = None)`
:   

    ### Methods

    `validate(self, dataset: dict[str, numpy.ndarray]) ‑> neuralops.data.validation.ValidationReport`
    :   Validate.

`ValidationReport(checks_passed: int, checks_failed: int, issues: list[dict[str, typing.Any]], is_valid: bool)`
:   ValidationReport(checks_passed: int, checks_failed: int, issues: list[dict[str, typing.Any]], is_valid: bool)

    ### Instance variables

    `checks_failed: int`
    :   The type of the None singleton.

    `checks_passed: int`
    :   The type of the None singleton.

    `is_valid: bool`
    :   The type of the None singleton.

    `issues: list[dict[str, typing.Any]]`
    :   The type of the None singleton.