"""Workspace and lifecycle HTTP endpoints for the interactive workbench."""
from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import Body, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from ..core import json_value, update_context_data_descriptions
from .session import InteractiveSession

MAX_UPLOAD = 256 * 1024 * 1024


def mount_platform(app, session: InteractiveSession):
    """Run the ``mount platform`` operation.

    Args:
        app: The app value.
        session: The session value.
    """
    app.state.session = session

    def context_preview_groups():
        with session.lock:
            axis_names = session.context.axis_names()
            groups = []
            for axis in axis_names:
                variables = [
                    name for name, dimensions in session.context.variable_axes.items()
                    if dimensions == (axis,) and session.context.data[name].ndim == 1
                ]
                if not variables:
                    continue
                groups.append({
                    "id": f"axis:{axis}", "kind": "axis", "axis": axis,
                    "axes": [axis], "variables": variables,
                    "shape": [len(session.context.data[axis])],
                })
            for name, dimensions in session.context.variable_axes.items():
                value = session.context.data[name]
                if value.ndim <= 1:
                    continue
                groups.append({
                    "id": f"variable:{name}",
                    "kind": "relation" if name in session.context.relation_names else "variable",
                    "variable": name,
                    "axes": list(dimensions), "shape": list(value.shape),
                    "dtype": str(value.dtype),
                })
            return groups

    @app.get('/api/session')
    def status():
        return session.snapshot()

    @app.get('/api/session/capabilities')
    def capabilities(agent: str = "search"):
        try:
            return session.capabilities(agent)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get('/api/session/provider-credential')
    def provider_credential(provider: str):
        try:
            return session.provider_credential(provider)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.put('/api/session/provider-credential')
    def set_provider_credential(payload: dict = Body(...)):
        try:
            return session.set_provider_credential(
                str(payload.get("provider", "")),
                payload.get("api_key"),
            )
        except (OSError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post('/api/session/start')
    def start(payload: dict = Body(...)):
        try:
            return session.start(payload)
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post('/api/data/agent')
    def prepare_data(payload: dict = Body(...)):
        try:
            return session.prepare_data(str(payload.get("message", "")))
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get('/api/data/agent/events')
    def data_agent_events(after_seq: int = 0):
        return session.data_interaction_manager.get_recent_events(max(0, after_seq))

    @app.post('/api/data/agent/stop')
    def stop_data_preparation():
        try:
            return session.stop_data_preparation()
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.put('/api/data/agent/settings')
    def configure_data_agent(payload: dict = Body(...)):
        try:
            return session.configure_data_agent(payload)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post('/api/data/agent/test')
    def test_data_agent_model(payload: dict = Body(...)):
        try:
            return session.test_data_agent_model(payload)
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post('/api/session/test')
    def test_model(payload: dict = Body(...)):
        try:
            return session.test_model(payload)
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get('/api/evaluator')
    def evaluator_configuration():
        return session.evaluator_configuration()

    @app.put('/api/evaluator')
    def configure_evaluator(payload: dict = Body(...)):
        try:
            return session.configure_evaluator(payload)
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post('/api/evaluator/test')
    def test_evaluator(payload: dict = Body(...)):
        try:
            return session.test_evaluator(payload)
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post('/api/evaluator/agent/start')
    def start_evaluator_assistance(payload: dict = Body(...)):
        try:
            return session.start_evaluator_assistance(payload)
        except (OSError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get('/api/evaluator/agent/events')
    def evaluator_agent_events(after_seq: int = 0):
        return session.evaluator_interaction_manager.get_recent_events(max(0, after_seq))

    @app.post('/api/evaluator/agent/stop')
    def stop_evaluator_assistance():
        try:
            return session.stop_evaluator_assistance()
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.put('/api/evaluator/agent/settings')
    def configure_evaluator_agent(payload: dict = Body(...)):
        try:
            return session.configure_evaluator_agent(payload)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post('/api/evaluator/agent/test')
    def test_evaluator_agent_model(payload: dict = Body(...)):
        try:
            return session.test_evaluator_agent_model(payload)
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get('/api/data/context')
    def data_context(rows: int = 300):
        rows = max(1, min(rows, 1000))
        with session.lock:
            schema = session.context.schema()
            all_axis_names = set(session.context.axis_names())
            groups = {}
            for name, value in session.context.data.items():
                if name in all_axis_names or value.ndim != 1:
                    continue
                key = tuple(session.context.variable_axes.get(name, (f"length:{len(value)}",)))
                groups.setdefault(key, []).append(name)
            preferred = next((
                names for names in groups.values()
                if session.context.target in names
            ), None)
            names = preferred or max(groups.values(), key=lambda item: (len(item), item), default=[])
            total = len(session.context.data[names[0]]) if names else 0
            referenced_axes = list(dict.fromkeys(
                axis
                for name in names
                for axis in session.context.variable_axes.get(name, ())
            ))
            matching_axes = [name for name in all_axis_names if len(session.context.data[name]) == total]
            axis_names = [
                name for name in referenced_axes if name in matching_axes
            ] + [
                name for name in matching_axes if name not in referenced_axes
            ]
            preview_columns = [*axis_names, *names]
            columns = [
                *session.context.axis_names(),
                *(name for name in session.context.data if name not in all_axis_names),
            ]
            arrays = {
                **{name: session.context.data[name] for name in axis_names},
                **{name: session.context.data[name] for name in names},
            }
            count = min(total, rows)
            records = [
                {name: json_value(arrays[name][index]) for name in preview_columns}
                for index in range(count)
            ]
            descriptions = {
                name: session.context.variable_descriptions.get(name, "")
                for name in session.context.data
            }
        return {
            **json_value(schema),
            "preview_groups": context_preview_groups(),
            "columns": columns,
            "preview_columns": preview_columns,
            "column_kinds": {
                **{name: "axis" for name in all_axis_names},
                **{name: "variable" for name in session.context.data if name not in all_axis_names},
            },
            "rows": total,
            "variable_descriptions": descriptions,
            "data": records,
            "truncated": total > count,
        }

    @app.get('/api/data/context/preview')
    def data_context_preview(group: str, limit: int = 100):
        limit = max(1, min(limit, 300))
        with session.lock:
            groups = {item["id"]: item for item in context_preview_groups()}
            selected = groups.get(group)
            if selected is None:
                raise HTTPException(404, "Unknown context data preview group")
            if selected["kind"] == "axis":
                axis = selected["axis"]
                columns = [axis, *selected["variables"]]
                arrays = {name: session.context.data[name] for name in columns}
                size = len(arrays[axis])
                count = min(size, limit)
                return {
                    **selected, "dtype": str(arrays[axis].dtype),
                    "columns": columns,
                    "column_kinds": {axis: "axis"} | {
                        name: "variable" for name in selected["variables"]
                    },
                    "data": [
                        {name: json_value(arrays[name][index]) for name in columns}
                        for index in range(count)
                    ],
                    "truncated": size > count,
                }

            name = selected["variable"]
            value = session.context.data[name]
            axes = selected["axes"]
            if selected["kind"] == "relation":
                num_nodes = session.context.num_nodes or 0
                relation_limit = min(limit, len(value))
                candidate_axes = []
                for order, axis in enumerate(session.context.axis_names()):
                    if len(session.context.data[axis]) != num_nodes:
                        continue
                    uses_as_structure_axis = sum(
                        dimensions and dimensions[-1] == axis
                        for variable, dimensions in session.context.variable_axes.items()
                        if variable not in session.context.relation_names
                        and variable not in session.context.variable_structures
                    )
                    if uses_as_structure_axis:
                        candidate_axes.append((-uses_as_structure_axis, order, axis))
                node_axis = min(candidate_axes)[2] if candidate_axes else None
                labels = (
                    session.context.data[node_axis]
                    if node_axis is not None else np.arange(num_nodes)
                )
                max_nodes = 160
                if num_nodes <= max_nodes:
                    node_ids = list(range(num_nodes))
                else:
                    node_ids = []
                    seen = set()
                    for row in value[:relation_limit]:
                        for endpoint in row:
                            endpoint = int(endpoint)
                            if endpoint not in seen and len(node_ids) < max_nodes:
                                seen.add(endpoint)
                                node_ids.append(endpoint)
                visible_nodes = set(node_ids)
                coordinates = [
                    row for row in value[:relation_limit]
                    if all(int(endpoint) in visible_nodes for endpoint in row)
                ]
                return {
                    **selected,
                    "description": session.context.variable_descriptions.get(name, ""),
                    "values": json_value(value[:relation_limit]),
                    "axis_values": {
                        axis: json_value(session.context.data[axis][:limit])
                        for axis in axes if axis in session.context.data
                    },
                    "node_axis": node_axis,
                    "nodes": [
                        {"id": node, "label": json_value(labels[node])}
                        for node in node_ids
                    ],
                    "coordinates": json_value(np.asarray(coordinates)),
                    "endpoint_count": int(value.shape[1]),
                    "relation_count": int(value.shape[0]),
                    "truncated": (
                        len(coordinates) < len(value) or len(node_ids) < num_nodes
                    ),
                }
            result = {
                **selected,
                "description": session.context.variable_descriptions.get(name, ""),
                "axis_values": {
                    axis: json_value(session.context.data[axis][:limit])
                    for axis in axes
                },
                "axis_truncated": {
                    axis: len(session.context.data[axis]) > limit for axis in axes
                },
            }
            if value.ndim == 2:
                row_count = min(value.shape[0], limit)
                column_count = min(value.shape[1], limit)
                result.update({
                    "values": json_value(value[:row_count, :column_count]),
                    "truncated": [
                        value.shape[0] > row_count, value.shape[1] > column_count,
                    ],
                })
            elif value.ndim >= 3:
                flattened = value.reshape(-1)
                result["sample_values"] = json_value(flattened[:min(8, flattened.size)])
                if value.dtype.kind in "iufcb" and flattened.size:
                    numeric = np.asarray(flattened, dtype=float)
                    finite = numeric[np.isfinite(numeric)]
                    if finite.size:
                        result["value_range"] = [float(finite.min()), float(finite.max())]
            return result

    @app.get('/api/data/context/plot')
    def data_context_plot(
        x: str, y: str, hue: str | None = None, size: str | None = None,
        limit: int = 2000,
    ):
        limit = max(1, min(limit, 5000))
        with session.lock:
            selected = {
                channel: name for channel, name in {
                    "x": x, "y": y, "hue": hue, "size": size,
                }.items() if name
            }
            missing = sorted(set(selected.values()) - set(session.context.data))
            if missing:
                raise HTTPException(404, f"Unknown context variables: {missing}")
            axis_names = set(session.context.axis_names())
            dimensions = {
                name: ((name,) if name in axis_names else session.context.variable_axes[name])
                for name in selected.values()
            }
            repeated = {
                name: axes for name, axes in dimensions.items()
                if len(set(axes)) != len(axes)
            }
            if repeated:
                raise HTTPException(400, f"Repeated axis names are not supported: {repeated}")
            broadcast_axes = list(dict.fromkeys(
                axis for name in selected.values() for axis in dimensions[name]
            ))
            broadcast_shape = tuple(
                len(session.context.data[axis]) for axis in broadcast_axes
            )
            total = int(np.prod(broadcast_shape, dtype=np.int64)) if broadcast_shape else 1
            count = min(total, limit)
            flat_indices = (
                np.arange(total, dtype=np.int64) if total <= limit else
                np.linspace(0, total - 1, count, dtype=np.int64)
            )
            coordinates = np.unravel_index(flat_indices, broadcast_shape) if broadcast_shape else ()
            values = {}
            numeric = {}
            for name in selected.values():
                value = session.context.data[name]
                source_axes = dimensions[name]
                permutation = sorted(
                    range(len(source_axes)), key=lambda index: broadcast_axes.index(source_axes[index])
                )
                ordered_axes = tuple(source_axes[index] for index in permutation)
                aligned = np.transpose(value, permutation) if permutation else value
                aligned_shape = tuple(
                    aligned.shape[ordered_axes.index(axis)] if axis in ordered_axes else 1
                    for axis in broadcast_axes
                )
                broadcast = np.broadcast_to(aligned.reshape(aligned_shape), broadcast_shape)
                sampled = broadcast[coordinates] if broadcast_shape else np.asarray([broadcast.item()])
                values[name] = json_value(sampled)
                numeric[name] = value.dtype.kind in "iufcb"
            return {
                "channels": selected,
                "axes": broadcast_axes,
                "shape": list(broadcast_shape),
                "total": total,
                "count": count,
                "truncated": total > count,
                "values": values,
                "numeric": numeric,
            }

    @app.post('/api/data/context/heatmap')
    def data_context_heatmap(payload: dict = Body(...)):
        with session.lock:
            raw_row = payload.get("row", [])
            raw_column = payload.get("column", [])
            if not isinstance(raw_row, list) or not isinstance(raw_column, list):
                raise HTTPException(400, "row and column must be lists")
            row = [str(name) for name in raw_row]
            column = [str(name) for name in raw_column]
            color = str(payload.get("color", ""))
            z = str(payload.get("z", ""))
            try:
                z_index = int(payload.get("z_index", 0))
            except (TypeError, ValueError) as exc:
                raise HTTPException(400, "z_index must be an integer") from exc
            axis_names = set(session.context.axis_names())
            names = {*row, *column, *([color] if color else []), *([z] if z else [])}
            missing = sorted(names - set(session.context.data))
            if missing:
                raise HTTPException(404, f"Unknown context variables: {missing}")

            if color:
                if color in axis_names:
                    raise HTTPException(400, "Color must be a two- or three-dimensional variable")
                color_axes = session.context.variable_axes[color]
                if len(color_axes) not in {2, 3}:
                    raise HTTPException(400, "Color must be a two- or three-dimensional variable")
                if len(row) != 1 or row[0] not in axis_names:
                    raise HTTPException(400, "Row must contain exactly one axis")
                if len(column) != 1 or column[0] not in axis_names:
                    raise HTTPException(400, "Column must contain exactly one axis")
                row_axis, column_axis = row[0], column[0]
                if row_axis == column_axis or row_axis not in color_axes or column_axis not in color_axes:
                    raise HTTPException(400, "Row and column must be different axes of the color variable")
                remaining = [axis for axis in color_axes if axis not in {row_axis, column_axis}]
                if remaining:
                    if z != remaining[0]:
                        raise HTTPException(400, f"Z must be the remaining color axis {remaining[0]!r}")
                    z_size = len(session.context.data[z])
                    if not 0 <= z_index < z_size:
                        raise HTTPException(400, f"z_index must be in [0, {z_size})")
                    matrix = np.take(
                        session.context.data[color], z_index,
                        axis=color_axes.index(z),
                    )
                    remaining_axes = tuple(axis for axis in color_axes if axis != z)
                    z_value = json_value(session.context.data[z][z_index])
                else:
                    if z:
                        raise HTTPException(400, "Z is only used with a three-dimensional color variable")
                    matrix = session.context.data[color]
                    remaining_axes = color_axes
                    z_size = 0
                    z_value = None
                if remaining_axes != (row_axis, column_axis):
                    matrix = np.transpose(matrix, (
                        remaining_axes.index(row_axis), remaining_axes.index(column_axis),
                    ))
                row_labels = session.context.data[row_axis]
                column_labels = session.context.data[column_axis]
                mode = "tensor"
            else:
                row_is_axis = len(row) == 1 and row[0] in axis_names
                column_is_axis = len(column) == 1 and column[0] in axis_names
                if row_is_axis == column_is_axis:
                    raise HTTPException(
                        400, "Place one axis in row or column and aligned one-dimensional variables on the other side",
                    )
                axis = row[0] if row_is_axis else column[0]
                variables = column if row_is_axis else row
                if not variables:
                    raise HTTPException(400, "Select at least one aligned one-dimensional variable")
                invalid = [
                    name for name in variables
                    if name in axis_names or session.context.variable_axes[name] != (axis,)
                ]
                if invalid:
                    raise HTTPException(400, f"Variables must use only axis {axis!r}: {invalid}")
                arrays = [session.context.data[name] for name in variables]
                if row_is_axis:
                    matrix = np.column_stack(arrays)
                    row_axis, column_axis = axis, "variables"
                    row_labels, column_labels = session.context.data[axis], np.asarray(variables)
                else:
                    matrix = np.vstack(arrays)
                    row_axis, column_axis = "variables", axis
                    row_labels, column_labels = np.asarray(variables), session.context.data[axis]
                z_size = 0
                z_value = None
                mode = "aligned_variables"

            row_count = min(matrix.shape[0], 80)
            column_count = min(matrix.shape[1], 80)
            shown = matrix[:row_count, :column_count]
            return {
                "mode": mode,
                "row_axis": row_axis,
                "column_axis": column_axis,
                "row_labels": json_value(row_labels[:row_count]),
                "column_labels": json_value(column_labels[:column_count]),
                "values": json_value(shown),
                "shape": list(matrix.shape),
                "dtype": str(matrix.dtype),
                "numeric": matrix.dtype.kind in "iufcb",
                "truncated": row_count < matrix.shape[0] or column_count < matrix.shape[1],
                "z_axis": z or None,
                "z_index": z_index if z else None,
                "z_value": z_value,
                "z_size": z_size,
            }

    @app.post('/api/data/context/reload')
    def reload_data_context():
        try:
            return session.reload_context_data()
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.put('/api/data/selection')
    def update_data_selection(payload: dict = Body(...)):
        with session.lock:
            if session.data_state in {"running", "stopping"}:
                raise HTTPException(
                    409,
                    "Wait for the data-preparation agent to finish before changing variable roles",
                )
            if session.state not in {"idle", "running"}:
                raise HTTPException(409, "Variable roles cannot be changed in the current run state")
            if session.state == "running":
                control_status = session.sr_interaction_manager.status()
                if not (
                    control_status["interaction_state"] == "paused"
                ):
                    raise HTTPException(
                        409,
                        "Pause symbolic regression and wait for the safe-boundary acknowledgement "
                        "before changing variable roles",
                    )
            target = str(payload.get("target", "")).strip()
            features = payload.get("features")
            if not isinstance(features, list) or any(
                not isinstance(name, str) for name in features
            ):
                raise HTTPException(400, "features must be a list of variable names")
            try:
                descriptions = payload.get("variable_descriptions", {})
                if not isinstance(descriptions, dict) or any(
                    not isinstance(name, str) or not isinstance(value, str)
                    for name, value in descriptions.items()
                ):
                    raise ValueError(
                        "variable_descriptions must map variable names to text"
                    )
                descriptions = {
                    name: value.strip() for name, value in descriptions.items()
                }
                manifest_path = session.workspace / "context.data" / "manifest.json"
                if manifest_path.is_file():
                    update_context_data_descriptions(
                        manifest_path.parent, descriptions,
                    )
                change = session.context.update_selection(
                    target=target,
                    features=features,
                    variable_descriptions=descriptions,
                )
            except (OSError, ValueError) as exc:
                raise HTTPException(400, str(exc)) from exc
            session.variable_descriptions = dict(
                session.context.variable_descriptions
            )
            return {
                **change,
                "context": session.context.schema(),
            }

    @app.put('/api/data/descriptions')
    def update_data_descriptions(payload: dict = Body(...)):
        with session.lock:
            if session.data_state in {"running", "stopping"}:
                raise HTTPException(
                    409,
                    "Wait for the data-preparation agent to finish before changing variable descriptions",
                )
            if session.state == "running":
                control_status = session.sr_interaction_manager.status()
                if not (
                    control_status["interaction_state"] == "paused"
                ):
                    raise HTTPException(
                        409,
                        "Pause symbolic regression and wait for the safe-boundary acknowledgement "
                        "before changing variable descriptions",
                    )
            descriptions = payload.get("variable_descriptions", {})
            if not isinstance(descriptions, dict) or any(
                not isinstance(name, str) or not isinstance(value, str)
                for name, value in descriptions.items()
            ):
                raise HTTPException(
                    400, "variable_descriptions must map variable names to text",
                )
            descriptions = {
                name: value.strip() for name, value in descriptions.items()
            }
            manifest_path = session.workspace / "context.data" / "manifest.json"
            try:
                if manifest_path.is_file():
                    update_context_data_descriptions(
                        manifest_path.parent, descriptions,
                    )
                elif unknown := set(descriptions) - set(session.context.data):
                    raise ValueError(
                        f"unknown context variables: {sorted(unknown)}"
                    )
            except (OSError, ValueError) as exc:
                raise HTTPException(400, str(exc)) from exc
            changed = False
            for name in session.context.data:
                if (
                    name in descriptions
                    and session.context.variable_descriptions[name] != descriptions[name]
                ):
                    session.context.variable_descriptions[name] = descriptions[name]
                    changed = True
            if changed:
                session.context.args.data_revision += 1
            session.variable_descriptions = dict(
                session.context.variable_descriptions
            )
            return session.context.schema()

    @app.post('/api/session/settings')
    def settings(payload: dict = Body(...)):
        try:
            return session.configure(payload)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    def resolve(path, *, access="read"):
        try:
            return session.resolve(path, access=access)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get('/api/workspace')
    def files(path: str = '', recursive: bool = False):
        with session.lock:
            directory = resolve(path)
            if not directory.is_dir():
                raise HTTPException(404, 'Directory not found')
            readonly_roots = {
                logical.name for logical in session.workspace_manager.mount_map
            }

            def list_entries(logical_directory, resolved_directory, ancestors=frozenset()):
                try:
                    identity = (resolved_directory.stat().st_dev, resolved_directory.stat().st_ino)
                except FileNotFoundError:
                    return []
                if identity in ancestors:
                    return []
                descendants = ancestors | {identity}
                entries = []
                for item in resolved_directory.iterdir():
                    logical = logical_directory / item.name
                    try:
                        resolved = resolve(str(logical))
                        is_directory = resolved.is_dir()
                        mounted = bool(
                            logical.parts and logical.parts[0] in readonly_roots
                        )
                        locked = not mounted and session.workspace_manager.is_locked(logical)
                        entry = {'name': item.name, 'path': str(logical),
                                 'directory': is_directory,
                                 'read_only': mounted or locked,
                                 'mounted': mounted,
                                 'locked': locked}
                        if is_directory:
                            children = list_entries(logical, resolved, descendants)
                            entry['size'] = None
                            if recursive:
                                entry['children'] = children
                        else:
                            entry['size'] = resolved.stat().st_size
                        entries.append(entry)
                    except (FileNotFoundError, HTTPException):
                        continue
                return sorted(entries, key=lambda entry: (
                    not entry['directory'], entry['name'].lower(), entry['name'],
                ))

            entries = list_entries(Path(path), directory)
            return {'path': path, 'entries': entries}

    @app.get('/api/workspace/size')
    def workspace_item_size(path: str):
        with session.lock:
            item = resolve(path)
            if item.is_file():
                return {'path': path, 'size': item.stat().st_size}
            if not item.is_dir():
                raise HTTPException(404, 'File or directory not found')
            total = 0
            for current, _, filenames in os.walk(item):
                relative_directory = Path(current).relative_to(item)
                for filename in filenames:
                    relative = relative_directory / filename
                    try:
                        source = resolve(str(Path(path) / relative))
                    except HTTPException:
                        continue
                    if source.is_file():
                        total += source.stat().st_size
            return {'path': path, 'size': total}

    @app.get('/api/workspace/download')
    def download(path: str):
        with session.lock:
            item = resolve(path)
            if item.is_file():
                return FileResponse(
                    item, filename=item.name, media_type='application/octet-stream',
                )
            if not item.is_dir():
                raise HTTPException(404, 'File or directory not found')
            descriptor, archive_name = tempfile.mkstemp(suffix='.zip')
            os.close(descriptor)
            archive = Path(archive_name)
            try:
                with zipfile.ZipFile(
                    archive, mode='w', compression=zipfile.ZIP_DEFLATED,
                ) as output:
                    for current, directories, filenames in os.walk(item):
                        current_path = Path(current)
                        relative_directory = current_path.relative_to(item)
                        if not directories and not filenames:
                            output.writestr(
                                str(Path(item.name) / relative_directory) + '/', '',
                            )
                        for filename in filenames:
                            relative = relative_directory / filename
                            try:
                                source = resolve(str(Path(path) / relative))
                            except HTTPException:
                                continue
                            if source.is_file():
                                output.write(source, Path(item.name) / relative)
            except Exception:
                archive.unlink(missing_ok=True)
                raise
            return FileResponse(
                archive, filename=f'{item.name}.zip', media_type='application/zip',
                background=BackgroundTask(archive.unlink, missing_ok=True),
            )

    @app.get('/api/workspace/preview')
    def preview(path: str):
        with session.lock:
            file = resolve(path)
            if not file.is_file():
                raise HTTPException(404, 'File not found')
            if file.suffix.lower() == '.npy':
                try:
                    try:
                        array = np.load(file, mmap_mode='r', allow_pickle=False)
                    except ValueError as exc:
                        if 'shape is empty' not in str(exc):
                            raise
                        array = np.load(file, allow_pickle=False)
                    text = np.array2string(
                        array,
                        threshold=1000,
                        edgeitems=4,
                        max_line_width=120,
                    )
                except (OSError, TypeError, ValueError) as exc:
                    raise HTTPException(
                        400,
                        f'Unable to preview NPY file safely: {exc}',
                    ) from exc
                truncated = array.size > 1000 or len(text) > 65536
                return {
                    'kind': 'npy',
                    'shape': list(array.shape),
                    'dtype': str(array.dtype),
                    'size': int(array.size),
                    'text': text[:65536],
                    'truncated': truncated,
                }
            with file.open('rb') as stream:
                data = stream.read(65537)
            return {
                'kind': 'text',
                'text': data[:65536].decode('utf-8', errors='replace'),
                'truncated': len(data) > 65536,
            }

    @app.post('/api/workspace/directory')
    def create_directory(payload: dict = Body(...)):
        path = str(payload.get('path', '')).strip()
        if not path:
            raise HTTPException(400, 'Directory path is required')
        with session.lock:
            try:
                directory = resolve(path, access="create")
                if directory.exists():
                    raise HTTPException(409, 'A file or directory already exists at this path')
                directory.mkdir(parents=False)
            except HTTPException:
                raise
            except (OSError, ValueError) as exc:
                raise HTTPException(400, str(exc)) from exc
        return {'path': path}

    @app.put('/api/workspace/lock')
    def set_workspace_item_lock(payload: dict = Body(...)):
        path = str(payload.get('path', '')).strip()
        locked = payload.get('locked')
        if not path:
            raise HTTPException(400, 'The workspace root cannot be locked')
        if not isinstance(locked, bool):
            raise HTTPException(400, 'locked must be a boolean')
        with session.lock:
            try:
                session.workspace_manager.set_locked(path, locked)
            except (OSError, ValueError) as exc:
                raise HTTPException(400, str(exc)) from exc
        return {'path': path, 'locked': locked}

    @app.patch('/api/workspace')
    def move_workspace_item(payload: dict = Body(...)):
        source_path = str(payload.get('source', '')).strip()
        destination_path = str(payload.get('destination', '')).strip()
        if not source_path or not destination_path:
            raise HTTPException(400, 'Source and destination paths are required')
        with session.lock:
            try:
                source = session.resolve(source_path, access="remove")
                destination = session.resolve(destination_path, access="create")
                if not source.exists():
                    raise HTTPException(404, 'Source file or directory not found')
                if destination.exists():
                    raise HTTPException(409, 'A file or directory already exists at the destination')
                if source.is_dir() and destination.is_relative_to(source):
                    raise HTTPException(400, 'A directory cannot be moved inside itself')
                destination.parent.mkdir(parents=True, exist_ok=True)
                source.rename(destination)
            except HTTPException:
                raise
            except (OSError, ValueError) as exc:
                raise HTTPException(400, str(exc)) from exc
        return {'source': source_path, 'destination': destination_path}

    @app.delete('/api/workspace')
    def delete_workspace_item(path: str):
        path = path.strip()
        if not path:
            raise HTTPException(400, 'The workspace root cannot be deleted')
        with session.lock:
            try:
                item = session.resolve(path, access="remove")
                if not item.exists():
                    raise HTTPException(404, 'File or directory not found')
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            except HTTPException:
                raise
            except (OSError, ValueError) as exc:
                raise HTTPException(400, str(exc)) from exc
        return {'path': path}

    @app.get('/api/data/csv-files')
    def csv_files():
        with session.lock:
            files = [str(relative) for relative, path in session.workspace_manager.iter_files()
                     if path.suffix.lower() in {'.csv', '.xlsx'}]
            return {'files': sorted(files)}

    @app.get('/api/data/preview')
    def data_preview(path: str, rows: int = 300):
        rows = max(5, min(rows, 1000))
        with session.lock:
            file = resolve(path)
            if file.suffix.lower() not in {'.csv', '.xlsx'} or not file.is_file():
                raise HTTPException(400, 'Select an existing CSV or Excel file')
            try:
                frame = (
                    pd.read_excel(file, nrows=rows + 1)
                    if file.suffix.lower() == '.xlsx'
                    else pd.read_csv(file, nrows=rows + 1)
                )
            except Exception as exc:
                raise HTTPException(400, f'Unable to read table: {exc}') from exc
        if frame.empty or len(frame.columns) < 2:
            raise HTTPException(400, 'CSV needs at least two columns and one row')
        columns = [str(column) for column in frame.columns]
        if len(set(columns)) != len(columns):
            raise HTTPException(400, 'CSV column names must be unique')
        frame.columns = columns
        truncated = len(frame) > rows
        frame = frame.iloc[:rows]
        numeric = {
            column: bool(pd.api.types.is_numeric_dtype(frame[column]))
            for column in columns
        }
        serializable = frame.astype(object).where(frame.notna(), None)
        serializable = serializable.replace([float('inf'), float('-inf')], None)
        records = serializable.to_dict(orient='records')
        return {'path': path, 'columns': columns, 'numeric': numeric,
                'rows': records, 'truncated': truncated}

    @app.post('/api/data/demo')
    def create_demo(payload: dict | None = Body(default=None)):
        try:
            path = session.create_demo(str((payload or {}).get("kind", "polynomial")))
        except FileExistsError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        relative = str(path.relative_to(session.workspace))
        size = sum(file.stat().st_size for file in path.iterdir() if file.is_file())
        session.data_interaction_manager.publish_event('workspace_changed', {
            'operation': 'created', 'path': relative, 'size': size,
        })
        return {'path': relative}

    @app.post('/api/data/prompts')
    def initial_prompts(payload: dict = Body(...)):
        try:
            return session.preview_initial_prompts(payload)
        except (ValueError, OSError, pd.errors.ParserError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.put('/api/workspace/upload')
    async def upload(request: Request, path: str):
        # Keep the staging file beside its destination so the final atomic
        # replace also works when the run directory and workspace are on
        # different filesystems.
        temp = None
        size = 0
        try:
            with session.lock:
                destination = resolve(path, access="create")
                if destination.exists():
                    raise HTTPException(409, 'File already exists; rename it before uploading')
                destination.parent.mkdir(parents=True, exist_ok=True)
                fd, temp_name = tempfile.mkstemp(
                    prefix='.sr-harness-upload-', dir=destination.parent,
                )
                temp = Path(temp_name)
            with os.fdopen(fd, 'wb') as output:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_UPLOAD:
                        raise HTTPException(413, 'Maximum file size is 256 MiB')
                    output.write(chunk)
            with session.lock:
                current_destination = resolve(path, access="create")
                if current_destination != destination:
                    raise HTTPException(409, 'Workspace changed while the file was uploading')
                if destination.exists():
                    raise HTTPException(409, 'File already exists; rename it before uploading')
                os.replace(temp, destination)
            session.data_interaction_manager.publish_event('workspace_changed', {
                'operation': 'created', 'path': path, 'size': size,
            })
            return {'path': path, 'size': size}
        except HTTPException:
            raise
        except OSError as exc:
            raise HTTPException(400, f'Unable to upload file: {exc}') from exc
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)
