"""Contract tests for the documented public Python API."""

from __future__ import annotations

import inspect

import pytest

import sr_harness.agents
import sr_harness.api
import sr_harness.core
import sr_harness.evaluator
import sr_harness.runtime
import sr_harness.tools
import sr_harness_engine


PUBLIC_MODULES = (
    sr_harness.agents,
    sr_harness.api,
    sr_harness.core,
    sr_harness.evaluator,
    sr_harness.runtime,
    sr_harness.tools,
    sr_harness_engine,
)


def _public_callables():
    seen: set[int] = set()
    for module in PUBLIC_MODULES:
        for export in module.__all__:
            obj = getattr(module, export)
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            if inspect.isfunction(obj):
                yield f"{module.__name__}.{export}", obj
                continue
            if not inspect.isclass(obj):
                continue
            for name, descriptor in obj.__dict__.items():
                if name.startswith("_"):
                    continue
                if isinstance(descriptor, (classmethod, staticmethod)):
                    member = descriptor.__func__
                elif isinstance(descriptor, property):
                    member = descriptor.fget
                else:
                    member = descriptor
                if inspect.isfunction(member):
                    yield f"{obj.__module__}.{obj.__qualname__}.{name}", member


@pytest.mark.parametrize(("qualified_name", "callable_object"), list(_public_callables()))
def test_public_callable_has_documented_typed_signature(qualified_name, callable_object):
    """Require docs and complete annotations for every exported callable."""
    assert inspect.getdoc(callable_object), f"{qualified_name} has no docstring"
    signature = inspect.signature(callable_object)
    for parameter in signature.parameters.values():
        if parameter.name in {"self", "cls"}:
            continue
        assert parameter.annotation is not inspect.Parameter.empty, (
            f"{qualified_name} parameter {parameter.name!r} has no type annotation"
        )
    assert signature.return_annotation is not inspect.Signature.empty, (
        f"{qualified_name} has no return type annotation"
    )
