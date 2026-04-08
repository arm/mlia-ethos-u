# SPDX-FileCopyrightText: Copyright 2022-2023, 2026, Arm Limited and/or its
# affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Ethos-U advice generation."""

from __future__ import annotations

import pytest

from mlia.cli.helpers import CLIActionResolver
from mlia.core.advice_generation import Advice
from mlia.core.common import AdviceCategory, DataItem
from mlia.core.context import ExecutionContext
from mlia.core.helpers import ActionResolver, APIActionResolver
from mlia.core.output_schema import AdviceCategory as SchemaAdviceCategory
from mlia.core.output_schema import AdviceSeverity, OperatorIdentifier, OperatorScope
from mlia.target.ethos_u.optimization_shims import OptimizationSettings
from mlia.target.ethos_u.advice_generation import (
    EthosUAdviceProducer,
    EthosUStaticAdviceProducer,
)
from mlia.target.ethos_u.data_analysis import (
    AllOperatorsSupportedOnNPU,
    HasCPUOnlyOperators,
    HasUnsupportedOnNPUOperators,
    OptimizationDiff,
    OptimizationResults,
    PerfMetricDiff,
)
from mlia.target.ethos_u.pattern_analysis import IneffectiveActivationPattern


def assert_advices_match(actual: list[Advice], expected: list[Advice]) -> None:
    """Assert the advice fields relevant to Ethos-U behavior."""
    assert len(actual) == len(expected)
    for actual_adv, expected_adv in zip(actual, expected):
        assert actual_adv.message == expected_adv.message
        assert actual_adv.category == expected_adv.category
        assert actual_adv.severity == expected_adv.severity
        assert actual_adv.affected_entities == expected_adv.affected_entities


@pytest.mark.parametrize(
    "input_data, advice_category, action_resolver, expected_advice",
    [
        pytest.param(
            AllOperatorsSupportedOnNPU(),
            {AdviceCategory.COMPATIBILITY},
            APIActionResolver(),
            [
                Advice(
                    id="0",
                    category=SchemaAdviceCategory.COMPATIBILITY,
                    severity=AdviceSeverity.INFO,
                    message=(
                        "You don't have any unsupported operators, your "
                        "model will run completely on NPU."
                    ),
                )
            ],
            id="all_ops_supported_compat_api",
        ),
        pytest.param(
            AllOperatorsSupportedOnNPU(),
            {AdviceCategory.COMPATIBILITY},
            CLIActionResolver(
                {
                    "target_profile": "sample_target",
                    "model": "sample_model.tflite",
                }
            ),
            [
                Advice(
                    id="0",
                    category=SchemaAdviceCategory.COMPATIBILITY,
                    severity=AdviceSeverity.INFO,
                    message=(
                        "You don't have any unsupported operators, your "
                        "model will run completely on NPU. Check the "
                        "estimated performance by running the following "
                        "command:  mlia check sample_model.tflite "
                        "--target-profile sample_target --performance"
                    ),
                )
            ],
            id="all_ops_supported_compat_cli_perf_cmd",
        ),
        pytest.param(
            HasCPUOnlyOperators(cpu_only_ops=["OP1", "OP2", "OP3"]),
            {AdviceCategory.COMPATIBILITY},
            APIActionResolver(),
            [
                Advice(
                    id="0",
                    category=SchemaAdviceCategory.COMPATIBILITY,
                    severity=AdviceSeverity.WARNING,
                    message=(
                        "You have at least 3 operators that is CPU only: "
                        "OP1,OP2,OP3. Using operators that are supported by "
                        "the NPU will improve performance."
                    ),
                )
            ],
            id="cpu_only_ops_compat_api",
        ),
        pytest.param(
            HasCPUOnlyOperators(cpu_only_ops=["OP1", "OP2", "OP3"]),
            {AdviceCategory.COMPATIBILITY},
            CLIActionResolver({}),
            [  # Line 78 context
                Advice(
                    id="0",
                    category=SchemaAdviceCategory.COMPATIBILITY,
                    severity=AdviceSeverity.WARNING,
                    message=(
                        "You have at least 3 operators that is CPU only: "
                        "OP1,OP2,OP3. Using operators that are supported "
                        "by the NPU will improve performance."
                    ),
                )
            ],
            id="cpu_only_ops_compat_cli",
        ),
        pytest.param(
            HasUnsupportedOnNPUOperators(npu_unsupported_ratio=0.4),
            {AdviceCategory.COMPATIBILITY},
            APIActionResolver(),
            [
                Advice(
                    id="0",
                    category=SchemaAdviceCategory.COMPATIBILITY,
                    severity=AdviceSeverity.WARNING,
                    message=(
                        "You have 40% of operators that cannot be placed on "
                        "the NPU. For better performance, please review the "
                        "reasons reported in the table, and adjust the model "
                        "accordingly where possible."
                    ),
                )
            ],
            id="unsupported_ops_compat_api",
        ),
        pytest.param(
            HasUnsupportedOnNPUOperators(npu_unsupported_ratio=0.4),
            {AdviceCategory.COMPATIBILITY},
            CLIActionResolver({}),
            [  # Line 104 context
                Advice(
                    id="0",
                    category=SchemaAdviceCategory.COMPATIBILITY,
                    severity=AdviceSeverity.WARNING,
                    message=(
                        "You have 40% of operators that cannot be placed "
                        "on the NPU. For better performance, please review "
                        "the reasons reported in the table, and adjust the "
                        "model accordingly where possible."
                    ),
                )
            ],
            id="unsupported_ops_compat_cli",
        ),
        pytest.param(
            IneffectiveActivationPattern(
                affected_layers=["Layer1", "Layer2"],
                layer_count=2,
                activation_types=["MISH", "SELU"],
                recommendation="Consider replacing them with NPU-friendly alternatives.",
            ),
            {AdviceCategory.PERFORMANCE},
            APIActionResolver(),
            [
                Advice(
                    id="0",
                    category=SchemaAdviceCategory.PERFORMANCE,
                    severity=AdviceSeverity.WARNING,
                    message=(
                        "Detected 2 layers using "
                        "suboptimal activation functions (MISH, SELU). "
                        "Consider replacing them with NPU-friendly alternatives."
                    ),
                    affected_entities=[
                        OperatorIdentifier(
                            scope=OperatorScope.OPERATOR,
                            name="Layer1",
                            location="Layer1",
                        ),
                        OperatorIdentifier(
                            scope=OperatorScope.OPERATOR,
                            name="Layer2",
                            location="Layer2",
                        ),
                    ],
                )
            ],
            id="ineffective_activation_pattern_perf_api",
        ),
        pytest.param(
            OptimizationResults(
                [
                    OptimizationDiff(
                        opt_type=[OptimizationSettings("pruning", 0.5, None)],
                        opt_diffs={
                            "sram": PerfMetricDiff(100, 150),
                            "dram": PerfMetricDiff(100, 50),
                            "on_chip_flash": PerfMetricDiff(100, 100),
                            "off_chip_flash": PerfMetricDiff(100, 100),
                            "npu_total_cycles": PerfMetricDiff(10, 5),
                        },
                    ),
                ]
            ),
            {AdviceCategory.OPTIMIZATION},
            APIActionResolver(),
            [
                Advice(
                    id="0",
                    category=SchemaAdviceCategory.OPTIMIZATION,
                    severity=AdviceSeverity.INFO,
                    message=(
                        "With the selected optimization (pruning: 0.5) - "
                        "You have achieved 50.00% performance improvement "
                        "in DRAM used (KB) - You have achieved 50.00% "
                        "performance improvement in NPU total cycles - "
                        "SRAM used (KB) have degraded by 50.00% You can "
                        "try to push the optimization target higher "
                        "(e.g. pruning: 0.6) to check if those results "
                        "can be further improved."
                    ),
                ),
                Advice(  # Line 170 context
                    id="1",
                    category=SchemaAdviceCategory.OPTIMIZATION,
                    severity=AdviceSeverity.INFO,
                    message=(
                        "The applied tooling techniques have an impact on "
                        "accuracy. Additional hyperparameter tuning may be "
                        "required after any optimization."
                    ),
                ),
            ],
            id="opt_results_pruning_api",
        ),
        pytest.param(
            OptimizationResults(
                [
                    OptimizationDiff(
                        opt_type=[OptimizationSettings("pruning", 0.5, None)],
                        opt_diffs={
                            "sram": PerfMetricDiff(100, 150),
                            "dram": PerfMetricDiff(100, 50),
                            "on_chip_flash": PerfMetricDiff(100, 100),
                            "off_chip_flash": PerfMetricDiff(100, 100),
                            "npu_total_cycles": PerfMetricDiff(10, 5),
                        },
                    ),
                ]
            ),
            {AdviceCategory.OPTIMIZATION},
            CLIActionResolver({"model": "sample_model.h5"}),
            [
                Advice(
                    id="0",
                    category=SchemaAdviceCategory.OPTIMIZATION,
                    severity=AdviceSeverity.INFO,
                    message=(
                        "With the selected optimization (pruning: 0.5) - "
                        "You have achieved 50.00% performance improvement "
                        "in DRAM used (KB) - You have achieved 50.00% "
                        "performance improvement in NPU total cycles - "
                        "SRAM used (KB) have degraded by 50.00% You can "
                        "try to push the optimization target higher "
                        "(e.g. pruning: 0.6) to check if those results "
                        "can be further improved. For more info: mlia "
                        "optimize --help Optimization command: mlia "
                        "optimize sample_model.h5 --pruning "
                        "--pruning-target 0.6"
                    ),
                ),
                Advice(  # Line 170 context
                    id="1",
                    category=SchemaAdviceCategory.OPTIMIZATION,
                    severity=AdviceSeverity.INFO,
                    message=(
                        "The applied tooling techniques have an impact on "
                        "accuracy. Additional hyperparameter tuning may be "
                        "required after any optimization."
                    ),
                ),
            ],
            id="opt_results_pruning_cli",
        ),
        pytest.param(
            OptimizationResults(
                [
                    OptimizationDiff(
                        opt_type=[
                            OptimizationSettings("pruning", 0.5, None),
                            OptimizationSettings("clustering", 32, None),
                        ],
                        opt_diffs={
                            "sram": PerfMetricDiff(100, 150),
                            "dram": PerfMetricDiff(100, 50),
                            "on_chip_flash": PerfMetricDiff(100, 100),
                            "off_chip_flash": PerfMetricDiff(100, 100),
                            "npu_total_cycles": PerfMetricDiff(10, 5),
                        },
                    ),
                ]
            ),
            {AdviceCategory.OPTIMIZATION},
            APIActionResolver(),
            [
                Advice(
                    id="0",
                    category=SchemaAdviceCategory.OPTIMIZATION,
                    severity=AdviceSeverity.INFO,
                    message=(
                        "With the selected optimization (pruning: 0.5, "
                        "clustering: 32) - You have achieved 50.00% "
                        "performance improvement in DRAM used (KB) - You "
                        "have achieved 50.00% performance improvement in "
                        "NPU total cycles - SRAM used (KB) have degraded "
                        "by 50.00% You can try to push the optimization "
                        "target higher (e.g. pruning: 0.6 and/or "
                        "clustering: 16) to check if those results can be "
                        "further improved."
                    ),
                ),
                Advice(  # Line 205 context
                    id="1",
                    category=SchemaAdviceCategory.OPTIMIZATION,
                    severity=AdviceSeverity.INFO,
                    message=(
                        "The applied tooling techniques have an impact on "
                        "accuracy. Additional hyperparameter tuning may be "
                        "required after any optimization."
                    ),
                ),
            ],
            id="opt_results_pruning_clustering_api",
        ),
        pytest.param(
            OptimizationResults(
                [
                    OptimizationDiff(
                        opt_type=[
                            OptimizationSettings("clustering", 2, None),
                        ],
                        opt_diffs={
                            "sram": PerfMetricDiff(100, 150),
                            "dram": PerfMetricDiff(100, 50),
                            "on_chip_flash": PerfMetricDiff(100, 100),
                            "off_chip_flash": PerfMetricDiff(100, 100),
                            "npu_total_cycles": PerfMetricDiff(10, 5),
                        },
                    ),
                ]
            ),
            {AdviceCategory.OPTIMIZATION},
            APIActionResolver(),
            [
                Advice(
                    id="0",
                    category=SchemaAdviceCategory.OPTIMIZATION,
                    severity=AdviceSeverity.INFO,
                    message=(
                        "With the selected optimization (clustering: 2) - "
                        "You have achieved 50.00% performance improvement "
                        "in DRAM used (KB) - You have achieved 50.00% "
                        "performance improvement in NPU total cycles - "
                        "SRAM used (KB) have degraded by 50.00%"
                    ),
                ),
                Advice(  # Line 239 context
                    id="1",
                    category=SchemaAdviceCategory.OPTIMIZATION,
                    severity=AdviceSeverity.INFO,
                    message=(
                        "The applied tooling techniques have an impact on "
                        "accuracy. Additional hyperparameter tuning may be "
                        "required after any optimization."
                    ),
                ),
            ],
            id="opt_results_clustering_api",
        ),
        pytest.param(
            OptimizationResults(
                [
                    OptimizationDiff(
                        opt_type=[OptimizationSettings("pruning", 0.5, None)],
                        opt_diffs={
                            "sram": PerfMetricDiff(100, 150),
                            "dram": PerfMetricDiff(100, 150),
                            "on_chip_flash": PerfMetricDiff(100, 150),
                            "off_chip_flash": PerfMetricDiff(100, 150),
                            "npu_total_cycles": PerfMetricDiff(10, 100),
                        },
                    ),
                ]
            ),
            {AdviceCategory.OPTIMIZATION},
            APIActionResolver(),
            [
                Advice(
                    id="0",
                    category=SchemaAdviceCategory.OPTIMIZATION,
                    severity=AdviceSeverity.WARNING,
                    message=(
                        "With the selected optimization (pruning: 0.5) - "
                        "DRAM used (KB) have degraded by 50.00% - SRAM "
                        "used (KB) have degraded by 50.00% - On chip flash "
                        "used (KB) have degraded by 50.00% - Off chip "
                        "flash used (KB) have degraded by 50.00% - NPU "
                        "total cycles have degraded by 900.00% The "
                        "performance seems to have degraded after applying "
                        "the selected optimizations, try exploring "
                        "different optimization types/targets."
                    ),
                ),
                Advice(
                    id="1",
                    category=SchemaAdviceCategory.OPTIMIZATION,
                    severity=AdviceSeverity.INFO,
                    message=(
                        "The applied tooling techniques have an impact on "
                        "accuracy. Additional hyperparameter tuning may be "
                        "required after any optimization."
                    ),
                ),
            ],
            id="opt_results_degraded_perf_api",
        ),
        pytest.param(
            OptimizationResults(
                [
                    OptimizationDiff(
                        opt_type=[OptimizationSettings("pruning", 0.5, None)],
                        opt_diffs={
                            "sram": PerfMetricDiff(100, 150),
                            "dram": PerfMetricDiff(100, 150),
                            "on_chip_flash": PerfMetricDiff(100, 150),
                            "off_chip_flash": PerfMetricDiff(100, 150),
                            "npu_total_cycles": PerfMetricDiff(10, 100),
                        },
                    ),
                    OptimizationDiff(
                        opt_type=[OptimizationSettings("pruning", 0.6, None)],
                        opt_diffs={
                            "sram": PerfMetricDiff(100, 150),
                            "dram": PerfMetricDiff(100, 150),
                            "on_chip_flash": PerfMetricDiff(100, 150),
                            "off_chip_flash": PerfMetricDiff(100, 150),
                            "npu_total_cycles": PerfMetricDiff(10, 100),
                        },
                    ),
                ]
            ),
            {AdviceCategory.OPTIMIZATION},
            APIActionResolver(),
            [],  # no advice for more than one optimization result
            id="opt_results_multiple_diff_no_advice_api",
        ),
    ],
)
def test_ethosu_advice_producer(
    tmpdir: str,
    input_data: DataItem,
    expected_advice: list[Advice],
    advice_category: set[AdviceCategory] | None,
    action_resolver: ActionResolver,
) -> None:
    """Test Ethos-U Advice producer."""
    producer = EthosUAdviceProducer()

    context = ExecutionContext(
        advice_category=advice_category,
        output_dir=tmpdir,
        action_resolver=action_resolver,
    )

    producer.set_context(context)
    producer.produce_advice(input_data)

    actual = producer.get_advice()
    assert isinstance(actual, list)
    assert_advices_match(actual, expected_advice)


@pytest.mark.parametrize(
    "advice_category, action_resolver, expected_advice",
    [
        pytest.param(
            {AdviceCategory.COMPATIBILITY, AdviceCategory.PERFORMANCE},
            None,
            [],
            id="static_compat_perf_no_resolver",
        ),
        pytest.param(
            {AdviceCategory.COMPATIBILITY},
            None,
            [],
            id="static_compat_no_resolver",
        ),
        pytest.param(
            {AdviceCategory.PERFORMANCE},
            APIActionResolver(),
            [
                Advice(
                    id="0",
                    category=SchemaAdviceCategory.PERFORMANCE,
                    severity=AdviceSeverity.INFO,
                    message=(
                        "You can improve the inference time by using only "
                        "operators that are supported by the NPU."
                    ),
                ),
                Advice(
                    id="1",
                    category=SchemaAdviceCategory.PERFORMANCE,
                    severity=AdviceSeverity.INFO,
                    message=(
                        "Check if you can improve the performance by "
                        "applying tooling techniques to your model."
                    ),
                ),
            ],
            id="static_perf_api",
        ),
        pytest.param(
            {AdviceCategory.PERFORMANCE},
            CLIActionResolver(
                {"model": "test_model.h5", "target_profile": "sample_target"}
            ),
            [
                Advice(
                    id="0",
                    category=SchemaAdviceCategory.PERFORMANCE,
                    severity=AdviceSeverity.INFO,
                    message=(
                        "You can improve the inference time by using only "
                        "operators that are supported by the NPU. Try "
                        "running the following command to verify that: "
                        "mlia check test_model.h5 --target-profile "
                        "sample_target"
                    ),
                ),
                Advice(
                    id="1",
                    category=SchemaAdviceCategory.PERFORMANCE,
                    severity=AdviceSeverity.INFO,
                    message=(
                        "Check if you can improve the performance by "
                        "applying tooling techniques to your model. Note: "
                        "you will need a Keras model for that. For example: "
                        "mlia optimize test_model.h5 --pruning --clustering "
                        "--pruning-target 0.5 --clustering-target 32 For more "
                        "info: mlia optimize --help"
                    ),
                ),
            ],
            id="static_perf_cli_with_cmd",
        ),
        pytest.param(
            {AdviceCategory.OPTIMIZATION},
            APIActionResolver(),
            [
                Advice(
                    id="0",
                    category=SchemaAdviceCategory.OPTIMIZATION,
                    severity=AdviceSeverity.INFO,
                    message=(
                        "For better performance, make sure that all the "
                        "operators of your final TensorFlow Lite model are "
                        "supported by the NPU."
                    ),
                )
            ],
            id="static_optim_api",
        ),
        pytest.param(
            {AdviceCategory.OPTIMIZATION},
            CLIActionResolver({"model": "test_model.h5"}),
            [
                Advice(
                    id="0",
                    category=SchemaAdviceCategory.OPTIMIZATION,
                    severity=AdviceSeverity.INFO,
                    message=(
                        "For better performance, make sure that all the "
                        "operators of your final TensorFlow Lite model are "
                        "supported by the NPU. For more details, run: mlia "
                        "check --help"
                    ),
                )
            ],
            id="static_optim_cli",
        ),
    ],
)
def test_ethosu_static_advice_producer(
    tmpdir: str,
    advice_category: set[AdviceCategory] | None,
    action_resolver: ActionResolver,
    expected_advice: list[Advice],
) -> None:
    """Test static advice generation."""
    producer = EthosUStaticAdviceProducer()

    context = ExecutionContext(
        advice_category=advice_category,
        output_dir=tmpdir,
        action_resolver=action_resolver,
    )
    producer.set_context(context)
    actual = producer.get_advice()
    assert isinstance(actual, list)
    assert_advices_match(actual, expected_advice)
