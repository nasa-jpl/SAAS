from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


class CsvSimulationLogger:
    """Small CSV writer for syssim value and fault logs."""

    def __init__(self, directory: str | Path, *, values: bool = True, faults: bool = True):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._last_fault_active: dict[str, bool] = {}
        self._value_file = None
        self._fault_file = None
        self.values_writer = None
        self.faults_writer = None

        if values:
            self._value_file = (self.directory / "values.csv").open("w", newline="")
            self.values_writer = csv.DictWriter(
                self._value_file,
                fieldnames=["time", "node", "kind", "name", "sample_time", "type", "value"],
            )
            self.values_writer.writeheader()
        if faults:
            self._fault_file = (self.directory / "faults.csv").open("w", newline="")
            self.faults_writer = csv.DictWriter(
                self._fault_file,
                fieldnames=["time", "fault", "enabled", "triggered", "active", "targets", "event"],
            )
            self.faults_writer.writeheader()

    def log_values(self, time: float, system: "NodeSystem") -> None:
        if self.values_writer is None:
            return
        for node in system.nodes:
            for port in node.iter_ports():
                sample = port.read()
                self.values_writer.writerow(
                    {
                        "time": time,
                        "node": node.name,
                        "kind": "port",
                        "name": port.attr_name,
                        "sample_time": sample.time,
                        "type": type(sample.value).__name__,
                        "value": _serialize(sample.value),
                    }
                )
            for parameter in node.iter_parameters():
                sample = parameter.sample
                self.values_writer.writerow(
                    {
                        "time": time,
                        "node": node.name,
                        "kind": "parameter",
                        "name": parameter.attr_name,
                        "sample_time": sample.time,
                        "type": type(sample.value).__name__,
                        "value": _serialize(sample.value),
                    }
                )

    def log_faults(self, time: float, faults) -> None:
        if self.faults_writer is None:
            return
        for fault in faults:
            fault_name = fault.name or fault.__class__.__name__
            was_active = self._last_fault_active.get(fault_name, False)
            event = "inactive"
            if fault.active and not was_active:
                event = "start"
            elif fault.active and was_active:
                event = "active"
            elif not fault.active and was_active:
                event = "end"
            self._last_fault_active[fault_name] = fault.active
            self.faults_writer.writerow(
                {
                    "time": time,
                    "fault": fault_name,
                    "enabled": fault.enabled,
                    "triggered": fault.triggered,
                    "active": fault.active,
                    "targets": json.dumps([target.full_name for target in fault.targets]),
                    "event": event,
                }
            )

    def close(self) -> None:
        for file_obj in (self._value_file, self._fault_file):
            if file_obj is not None:
                file_obj.close()


def _serialize(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())
    if isinstance(value, np.generic):
        return json.dumps(value.item())
    try:
        return json.dumps(value)
    except TypeError:
        return json.dumps(repr(value))