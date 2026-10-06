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

from ..core import json_value
from .session import InteractiveSession

MAX_UPLOAD = 256 * 1024 * 1024


def mount_platform(app, session: InteractiveSession):
    """Run the ``mount platform`` operation.

    Args:
        app: The app value.
        session: The session value.
    """
    app.state.session = session

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

    @app.get('/api/data/context')
    def data_context(rows: int = 300):
        rows = max(1, min(rows, 1000))
        with session.lock:
            schema = session.context.schema()
            groups = {}
            for name, value in session.context.data.items():
                if value.ndim != 1:
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
            matching_axes = [
                name for name, axis in session.context.axes.items()
                if (
                    axis.values.ndim == 1
                    and len(axis.values) == total
                    and name not in session.context.data
                )
            ]
            axis_names = [
                name for name in referenced_axes if name in matching_axes
            ] + [
                name for name in matching_axes if name not in referenced_axes
            ]
            preview_columns = [*axis_names, *names]
            columns = [
                *session.context.axes,
                *(name for name in session.context.data if name not in session.context.axes),
            ]
            arrays = {
                **{name: session.context.axes[name].values for name in axis_names},
                **{name: session.context.data[name] for name in names},
            }
            count = min(total, rows)
            records = [
                {name: json_value(arrays[name][index]) for name in preview_columns}
                for index in range(count)
            ]
            descriptions = {
                **{
                    name: session.context.variable_descriptions.get(
                        name, axis.description,
                    )
                    for name, axis in session.context.axes.items()
                },
                **{
                    name: session.context.variable_descriptions.get(name, "")
                    for name in session.context.data
                },
            }
        return {
            **json_value(schema),
            "columns": columns,
            "preview_columns": preview_columns,
            "column_kinds": {
                **{name: "axis" for name in session.context.axes},
                **{name: "variable" for name in session.context.data},
            },
            "rows": total,
            "variable_descriptions": descriptions,
            "data": records,
            "truncated": total > count,
        }

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
                control_status = session.controller.status()
                if not (
                    control_status["paused"]
                    and control_status["waiting_at_boundary"]
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
                change = session.context.update_selection(
                    target=target,
                    features=features,
                    variable_descriptions=descriptions,
                )
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            session.variable_descriptions = dict(
                session.context.variable_descriptions
            )
            return {
                **change,
                "context": session.context.schema(),
            }

    @app.post('/api/session/settings')
    def settings(payload: dict = Body(...)):
        try:
            return session.configure(payload)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    def resolve(path, *, write=False):
        try:
            return session.resolve(path, write=write)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get('/api/workspace')
    def files(path: str = '', recursive: bool = False):
        with session.lock:
            directory = resolve(path)
            if not directory.is_dir():
                raise HTTPException(404, 'Directory not found')
            readonly_roots = {
                logical.name for logical in session.workspace_manager.readonly_mounts
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
                        locked = not mounted and session.workspace_manager.is_locked(resolved)
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
                directory = resolve(path, write=True)
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
                source = resolve(source_path, write=True)
                destination = resolve(destination_path, write=True)
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
                item = resolve(path, write=True)
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
    def create_demo():
        try:
            path = session.create_demo()
        except FileExistsError as exc:
            raise HTTPException(409, str(exc)) from exc
        relative = str(path.relative_to(session.workspace))
        size = sum(file.stat().st_size for file in path.iterdir() if file.is_file())
        session.controller.publish('file_uploaded', {'path': relative, 'size': size})
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
                destination = resolve(path, write=True)
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
                current_destination = resolve(path, write=True)
                if current_destination != destination:
                    raise HTTPException(409, 'Workspace changed while the file was uploading')
                if destination.exists():
                    raise HTTPException(409, 'File already exists; rename it before uploading')
                os.replace(temp, destination)
            session.controller.publish('file_uploaded', {'path': path, 'size': size})
            return {'path': path, 'size': size}
        except HTTPException:
            raise
        except OSError as exc:
            raise HTTPException(400, f'Unable to upload file: {exc}') from exc
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)
