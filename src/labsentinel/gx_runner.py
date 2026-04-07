from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import great_expectations as gx
import pandas as pd


def _get_or_create_pandas_datasource(context: gx.AbstractDataContext, name: str):
    try:
        return context.data_sources.get(name)
    except Exception:
        return context.data_sources.add_pandas(name=name)


def _get_or_create_dataframe_asset(data_source, name: str):
    try:
        return data_source.get_asset(name)
    except Exception:
        return data_source.add_dataframe_asset(name=name)


def _get_or_create_batch_definition(data_asset, name: str):
    try:
        return data_asset.get_batch_definition(name)
    except Exception:
        return data_asset.add_batch_definition_whole_dataframe(name)


def _get_or_create_expectation_suite(context: gx.AbstractDataContext, suite_name: str) -> gx.ExpectationSuite:
    try:
        suite = context.suites.get(suite_name)
        # wyczyść expectations, żeby uniknąć duplikatów między runami
        suite.expectations = []
        return suite
    except Exception:
        suite = gx.ExpectationSuite(name=suite_name)
        return context.suites.add(suite)


def _get_or_create_validation_definition(
    context: gx.AbstractDataContext,
    name: str,
    batch_definition,
    suite: gx.ExpectationSuite,
):
    try:
        validation_definition = context.validation_definitions.get(name)
        validation_definition.data = batch_definition
        validation_definition.suite = suite
        validation_definition.save()
        return validation_definition
    except Exception:
        validation_definition = gx.ValidationDefinition(
            name=name,
            data=batch_definition,
            suite=suite,
        )
        return context.validation_definitions.add(validation_definition)


def _get_or_create_checkpoint(
    context: gx.AbstractDataContext,
    name: str,
    validation_definition,
):
    try:
        checkpoint = context.checkpoints.get(name)
        checkpoint.validation_definitions = [validation_definition]
        checkpoint.save()
        return checkpoint
    except Exception:
        checkpoint = gx.Checkpoint(
            name=name,
            validation_definitions=[validation_definition],
        )
        return context.checkpoints.add(checkpoint)


def _build_expectations(suite: gx.ExpectationSuite) -> gx.ExpectationSuite:
    required_columns = [
        "sample_id",
        "product",
        "parameter",
        "unit",
        "date",
        "value",
    ]

    for column in required_columns:
        suite.add_expectation(
            gx.expectations.ExpectColumnToExist(column=column)
        )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="parameter",
            value_set=["Water", "Sulfur", "Chloride", "Ash", "Viscosity"],
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="product",
            value_set=["Fuel_A", "Fuel_B", "Fuel_C"],
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="sample_id",
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="product",
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="parameter",
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="unit",
            mostly=0.98,
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="unit",
            value_set=["mg/kg", "% m/m", "cSt", "mg/L", "unknown"],
            mostly=0.98,
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="value_num",
            min_value=0,
            max_value=100,
            mostly=0.95,
        )
    )

    suite.save()
    return suite


def _extract_validation_payload(checkpoint_result: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "success": bool(getattr(checkpoint_result, "success", False)),
        "run_results": [],
    }

    run_results = getattr(checkpoint_result, "run_results", {}) or {}

    for key, validation_result in run_results.items():
        result_entry = {
            "validation_key": str(key),
            "success": bool(getattr(validation_result, "success", False)),
            "statistics": getattr(validation_result, "statistics", None),
            "suite_name": getattr(validation_result, "suite_name", None),
        }

        expectation_results = []
        for item in getattr(validation_result, "results", []) or []:
            expectation_results.append(
                {
                    "success": bool(getattr(item, "success", False)),
                    "expectation_type": getattr(
                        getattr(item, "expectation_config", None),
                        "type",
                        None,
                    ),
                    "kwargs": getattr(
                        getattr(item, "expectation_config", None),
                        "kwargs",
                        None,
                    ),
                    "result": getattr(item, "result", None),
                }
            )

        result_entry["results"] = expectation_results
        payload["run_results"].append(result_entry)

    return payload


def run_gx_validation(
    df: pd.DataFrame,
    run_dir: str | Path,
) -> dict[str, Any]:
    """
    Run Great Expectations validation on an in-memory DataFrame
    and save validation artifacts into the run directory.
    """
    run_dir = Path(run_dir)
    gx_dir = run_dir / "gx"
    gx_dir.mkdir(parents=True, exist_ok=True)

    context = gx.get_context()

    data_source = _get_or_create_pandas_datasource(context, "labsentinel_pandas_ds")
    data_asset = _get_or_create_dataframe_asset(data_source, "labsentinel_dataframe_asset")
    batch_definition = _get_or_create_batch_definition(data_asset, "whole_dataframe")

    suite = _get_or_create_expectation_suite(context, "labsentinel_qc_suite")
    suite = _build_expectations(suite)

    validation_definition = _get_or_create_validation_definition(
        context=context,
        name="labsentinel_validation_definition",
        batch_definition=batch_definition,
        suite=suite,
    )

    checkpoint = _get_or_create_checkpoint(
        context=context,
        name="labsentinel_checkpoint",
        validation_definition=validation_definition,
    )

    checkpoint_result = checkpoint.run(batch_parameters={"dataframe": df})

    suite_payload = suite.to_json_dict()
    validation_payload = _extract_validation_payload(checkpoint_result)

    gx_suite_path = gx_dir / "gx_expectation_suite.json"
    gx_validation_path = gx_dir / "gx_validation.json"

    gx_suite_path.write_text(
        json.dumps(suite_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    gx_validation_path.write_text(
        json.dumps(validation_payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return {
        "success": validation_payload["success"],
        "suite_path": str(gx_suite_path),
        "validation_path": str(gx_validation_path),
    }