#!/usr/bin/env python3
"""
simulate_blood_voyager.py

Event-driven Monte Carlo simulator for a Blood-Voyager-style nanobot
circulation model based on:
  * a CSV-defined vascular topology,
  * 3-D Euclidean vessel geometry,
  * continuous Poiseuille laminar flow,
  * area-uniform cross-sectional sampling,
  * cross-sectional-area routing at branches,
  * an effective arteriole-capillary-venule organ model, and
  * exponential heart residence times.

This program intentionally does NOT implement the legacy BloodVoyager
21-stream model, legacy 4-cm organ residence model, 0.1-s stepping,
ID-based hard-coded routing, or hard-coded artery/vein speeds.

All stochastic operations use one NumPy Generator instance so that a
fixed topology, configuration, and seed reproduce the same realization
sequence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


ObjectID = Union[int, str]


# ============================================================================
# Configuration
# ============================================================================

@dataclass(frozen=True)
class Configuration:
    input_file: str = "Pvasculature.csv"
    output_file: str = "PcsvNanoSec.csv"
    detailed_output_file: str = "PcsvNanoSecDetailed.csv"
    event_log_file: str = "nanobot_events.csv"
    number_of_cycles: int = 1000
    random_seed: int = 12345
    topology_tolerance_cm: float = 1e-6
    minimum_velocity_cm_s: float = 0.001

    left_heart_subtype: int = 10
    right_heart_subtype: int = 11
    organ_subtype: int = 0

    left_heart_mean_residence_s: float = 1.20
    right_heart_mean_residence_s: float = 1.50

    organ_arteriole_diameter_cm: float = 0.002
    organ_arteriole_length_cm: float = 0.3
    organ_arteriole_mean_velocity_cm_s: float = 0.3

    organ_capillary_diameter_cm: float = 0.001
    organ_capillary_length_cm: float = 0.1
    organ_capillary_mean_velocity_cm_s: float = 0.04

    organ_venule_diameter_cm: float = 0.004
    organ_venule_length_cm: float = 0.3
    organ_venule_mean_velocity_cm_s: float = 0.03

    # Safety limits. They are deliberately finite so an accidental cycle
    # cannot run forever. They are easy to override from the CLI.
    max_cycle_time_s: float = 1_000_000.0
    max_objects_per_cycle: int = 10_000

    # By default, cycles alternate Left -> Right, then Right -> Left.
    start_with_left_heart: bool = True

    # Event logging can be disabled to avoid generating a large file.
    write_event_log: bool = True


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class VesselRegion:
    object_id: ObjectID
    name: str
    subtype: int
    start: np.ndarray
    stop: np.ndarray
    diameter_cm: Optional[float]
    mean_velocity_cm_s: Optional[float]

    # Computed geometry. The length is useful for all objects diagnostically,
    # but it is NOT used for organ residence time.
    length_cm: float = field(init=False)
    direction: Optional[np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        self.length_cm = float(np.linalg.norm(self.stop - self.start))
        if self.length_cm > 0.0:
            self.direction = (self.stop - self.start) / self.length_cm
        else:
            self.direction = None

    @property
    def is_organ(self) -> bool:
        return self.subtype == 0

    @property
    def is_left_heart(self) -> bool:
        return self.subtype == 10

    @property
    def is_right_heart(self) -> bool:
        return self.subtype == 11

    @property
    def is_heart(self) -> bool:
        return self.subtype in (10, 11)

    @property
    def is_normal_vessel(self) -> bool:
        return 1 <= self.subtype <= 9


@dataclass(frozen=True)
class OrganSection:
    name: str
    diameter_cm: float
    length_cm: float
    mean_velocity_cm_s: float


@dataclass
class SampledVesselTraversal:
    radial_position_cm: float
    theta_rad: float
    local_velocity_cm_s: float
    entry_position: np.ndarray
    exit_position: np.ndarray
    residence_time_s: float


@dataclass
class OrganSectionTraversal:
    section_name: str
    entry_time_s: float
    exit_time_s: float
    residence_time_s: float
    radial_position_cm: float
    theta_rad: float
    local_velocity_cm_s: float


@dataclass
class CycleRecord:
    nanobot_id: int
    cycle_number: int
    organ_id: Optional[ObjectID]
    starting_heart_id: ObjectID
    expected_destination_heart_id: ObjectID
    destination_heart_id: Optional[ObjectID]
    total_circulation_time_s: float
    number_of_passed_objects: int
    residence_times_s: List[float]
    cycle_status: str
    termination_reason: str
    passed_object_ids: List[ObjectID]
    passed_object_units: List[str]


@dataclass
class DetailedTraversal:
    nanobot_id: int
    cycle_number: int
    starting_heart_id: ObjectID
    object_id: ObjectID
    object_name: str
    subtype: int
    entry_time_s: float
    exit_time_s: float
    residence_time_s: float
    start: np.ndarray
    stop: np.ndarray
    length_cm: float
    diameter_cm: Optional[float]
    radius_cm: Optional[float]
    mean_velocity_cm_s: Optional[float]
    radial_position_cm: Optional[float]
    theta_rad: Optional[float]
    local_velocity_cm_s: Optional[float]
    organ_sections: List[OrganSectionTraversal] = field(default_factory=list)


@dataclass
class Event:
    time_s: float
    nanobot_id: int
    event_type: str
    from_region_id: Optional[ObjectID] = None
    from_region_name: Optional[str] = None
    to_region_id: Optional[ObjectID] = None
    to_region_name: Optional[str] = None
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    residence_time_s: Optional[float] = None
    reason: Optional[str] = None


class InputValidationError(ValueError):
    """Raised when the topology CSV is not valid for this simulator."""


# ============================================================================
# Utility functions
# ============================================================================

REQUIRED_COLUMNS = [
    "ID",
    "Name",
    "SubType",
    "StartX",
    "StartY",
    "StartZ",
    "StopX",
    "StopY",
    "StopZ",
    "Diameter(cm)",
    "Mean Velocity (cm/s)",
]


def normalize_column_name(name: str) -> str:
    """Normalize CSV header text for tolerant matching."""
    return " ".join(name.strip().lower().split())


def resolve_column(fieldnames: Sequence[str], canonical: str) -> str:
    """Return the actual CSV fieldname corresponding to a canonical field."""
    normalized = {normalize_column_name(f): f for f in fieldnames}
    target = normalize_column_name(canonical)
    if target in normalized:
        return normalized[target]
    raise InputValidationError(
        f"Missing required CSV column '{canonical}'. Available columns: {list(fieldnames)}"
    )


def parse_id(raw: str, row_number: int) -> ObjectID:
    text = raw.strip()
    if not text:
        raise InputValidationError(f"Row {row_number}: field 'ID' is empty")
    # Preserve non-numeric identifiers, while still returning conventional
    # integer IDs as integers.
    try:
        numeric = float(text)
        if math.isfinite(numeric) and numeric.is_integer():
            return int(numeric)
    except ValueError:
        pass
    return text


def parse_required_float(raw: str, field: str, object_id: ObjectID, row_number: int) -> float:
    text = raw.strip() if raw is not None else ""
    if not text:
        raise InputValidationError(
            f"Object ID {object_id!r}, row {row_number}: field '{field}' is required but blank"
        )
    try:
        value = float(text)
    except ValueError as exc:
        raise InputValidationError(
            f"Object ID {object_id!r}, row {row_number}: field '{field}' is not numeric: {raw!r}"
        ) from exc
    if not math.isfinite(value):
        raise InputValidationError(
            f"Object ID {object_id!r}, row {row_number}: field '{field}' must be finite"
        )
    return value


def parse_optional_positive_float(
    raw: str, field: str, object_id: ObjectID, row_number: int
) -> Optional[float]:
    """Blank -> None; nonblank must be finite and strictly positive."""
    text = raw.strip() if raw is not None else ""
    if not text:
        return None
    try:
        value = float(text)
    except ValueError as exc:
        raise InputValidationError(
            f"Object ID {object_id!r}, row {row_number}: field '{field}' is not numeric: {raw!r}"
        ) from exc
    if not math.isfinite(value):
        raise InputValidationError(
            f"Object ID {object_id!r}, row {row_number}: field '{field}' must be finite"
        )
    if value <= 0.0:
        raise InputValidationError(
            f"Object ID {object_id!r}, row {row_number}: field '{field}' must be positive when provided"
        )
    return value


def parse_subtype(raw: str, object_id: ObjectID, row_number: int) -> int:
    text = raw.strip() if raw is not None else ""
    try:
        value = int(text)
    except ValueError as exc:
        raise InputValidationError(
            f"Object ID {object_id!r}, row {row_number}: field 'SubType' must be an integer"
        ) from exc
    if value < 0 or value > 11:
        raise InputValidationError(
            f"Object ID {object_id!r}, row {row_number}: unsupported SubType={value}; expected 0..11"
        )
    return value


def point_to_tuple(point: np.ndarray) -> Tuple[float, float, float]:
    return float(point[0]), float(point[1]), float(point[2])


def json_dumps_compact(value: Any) -> str:
    """Stable compact JSON suitable for a CSV cell."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


# ============================================================================
# CSV loading and topology
# ============================================================================

class VasculatureNetwork:
    """CSV-backed vascular graph with coordinate-derived downstream edges."""

    def __init__(self, regions: Mapping[ObjectID, VesselRegion], tolerance_cm: float) -> None:
        self.regions: Dict[ObjectID, VesselRegion] = dict(regions)
        self.tolerance_cm = float(tolerance_cm)
        self.downstream: Dict[ObjectID, List[ObjectID]] = {rid: [] for rid in self.regions}

        left = [r.object_id for r in self.regions.values() if r.is_left_heart]
        right = [r.object_id for r in self.regions.values() if r.is_right_heart]
        organs = [r.object_id for r in self.regions.values() if r.is_organ]
        if len(left) != 1:
            raise InputValidationError("Exactly one object with SubType=10 (Left heart) is required")
        if len(right) != 1:
            raise InputValidationError("Exactly one object with SubType=11 (Right heart) is required")
        if len(organs) < 1:
            raise InputValidationError("At least one object with SubType=0 (ORGAN) is required")
        self.left_heart_id: ObjectID = left[0]
        self.right_heart_id: ObjectID = right[0]
        self.organ_ids: List[ObjectID] = organs

        self._build_topology()
        self._validate_graph_basics()

    @classmethod
    def from_csv(cls, path: str, tolerance_cm: float) -> "VasculatureNetwork":
        file_path = Path(path)
        if not file_path.exists():
            raise InputValidationError(f"Input file does not exist: {file_path}")

        with file_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise InputValidationError("CSV file has no header row")

            cols = {canonical: resolve_column(reader.fieldnames, canonical) for canonical in REQUIRED_COLUMNS}
            regions: Dict[ObjectID, VesselRegion] = {}
            seen_rows: Dict[ObjectID, int] = {}

            for row_number, row in enumerate(reader, start=2):
                object_id = parse_id(row[cols["ID"]], row_number)
                if object_id in regions:
                    raise InputValidationError(
                        f"Duplicate object ID {object_id!r}: first seen on row {seen_rows[object_id]}, "
                        f"again on row {row_number}"
                    )
                seen_rows[object_id] = row_number

                subtype = parse_subtype(row[cols["SubType"]], object_id, row_number)
                name = (row[cols["Name"]] or "").strip()

                sx = parse_required_float(row[cols["StartX"]], "StartX", object_id, row_number)
                sy = parse_required_float(row[cols["StartY"]], "StartY", object_id, row_number)
                sz = parse_required_float(row[cols["StartZ"]], "StartZ", object_id, row_number)
                tx = parse_required_float(row[cols["StopX"]], "StopX", object_id, row_number)
                ty = parse_required_float(row[cols["StopY"]], "StopY", object_id, row_number)
                tz = parse_required_float(row[cols["StopZ"]], "StopZ", object_id, row_number)

                diameter = parse_optional_positive_float(
                    row[cols["Diameter(cm)"]], "Diameter(cm)", object_id, row_number
                )
                mean_velocity = parse_optional_positive_float(
                    row[cols["Mean Velocity (cm/s)"]], "Mean Velocity (cm/s)", object_id, row_number
                )

                if 1 <= subtype <= 9:
                    if diameter is None:
                        raise InputValidationError(
                            f"Object ID {object_id!r}, row {row_number}: 'Diameter(cm)' is required for SubType {subtype}"
                        )
                    if mean_velocity is None:
                        raise InputValidationError(
                            f"Object ID {object_id!r}, row {row_number}: 'Mean Velocity (cm/s)' is required for SubType {subtype}"
                        )
                # For organ/heart rows, blank diameter and velocity are expected.

                region = VesselRegion(
                    object_id=object_id,
                    name=name,
                    subtype=subtype,
                    start=np.array([sx, sy, sz], dtype=float),
                    stop=np.array([tx, ty, tz], dtype=float),
                    diameter_cm=diameter,
                    mean_velocity_cm_s=mean_velocity,
                )

                if region.is_normal_vessel and region.length_cm <= 0.0:
                    raise InputValidationError(
                        f"Object ID {object_id!r}, row {row_number}: normal vessel has zero 3-D length; "
                        "Start and Stop coordinates must differ"
                    )

                regions[object_id] = region

        return cls(regions, tolerance_cm)

    def _build_topology(self) -> None:
        """Build downstream edges from Stop-to-Start coordinate matching.

        A spatial hash keeps construction inexpensive for larger topologies,
        while checking neighboring cells guarantees that candidates near a
        cell boundary are still considered.
        """
        if self.tolerance_cm <= 0.0:
            raise InputValidationError("topology_tolerance_cm must be > 0")

        cell = self.tolerance_cm
        start_buckets: Dict[Tuple[int, int, int], List[ObjectID]] = {}

        for region in self.regions.values():
            key = tuple(np.floor(region.start / cell).astype(np.int64))
            start_buckets.setdefault(key, []).append(region.object_id)

        tol2 = cell * cell
        for region in self.regions.values():
            stop_key = tuple(np.floor(region.stop / cell).astype(np.int64))
            candidates: List[ObjectID] = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        bucket_key = (stop_key[0] + dx, stop_key[1] + dy, stop_key[2] + dz)
                        for candidate_id in start_buckets.get(bucket_key, []):
                            candidate = self.regions[candidate_id]
                            diff = candidate.start - region.stop
                            if float(np.dot(diff, diff)) < tol2:
                                candidates.append(candidate_id)

            # Preserve CSV/object insertion order and remove duplicates.
            seen: set[ObjectID] = set()
            ordered_unique = []
            for candidate_id in candidates:
                if candidate_id not in seen:
                    ordered_unique.append(candidate_id)
                    seen.add(candidate_id)
            self.downstream[region.object_id] = ordered_unique

    def _validate_graph_basics(self) -> None:
        if len(self.regions) == 0:
            raise InputValidationError("The topology CSV contains no objects")
        if len([r for r in self.regions.values() if r.is_left_heart]) != 1:
            raise InputValidationError("Exactly one object with SubType=10 (Left heart) is required")
        if len([r for r in self.regions.values() if r.is_right_heart]) != 1:
            raise InputValidationError("Exactly one object with SubType=11 (Right heart) is required")
        if len(self.organ_ids) < 1:
            raise InputValidationError("At least one object with SubType=0 (ORGAN) is required")

    def get(self, object_id: ObjectID) -> VesselRegion:
        return self.regions[object_id]

    def candidates(self, object_id: ObjectID) -> List[ObjectID]:
        return list(self.downstream.get(object_id, []))

    def validate_reachable_direction(self) -> List[str]:
        """Return non-fatal graph diagnostics without assuming a single topology."""
        warnings: List[str] = []
        for rid, region in self.regions.items():
            if region.is_normal_vessel and not self.downstream[rid]:
                warnings.append(f"Object {rid!r} (SubType={region.subtype}) is a dead end")
        return warnings


# ============================================================================
# Flow models
# ============================================================================

class LaminarFlowModel:
    """Continuous Poiseuille flow model for a cylindrical vessel."""

    def __init__(self, rng: np.random.Generator, minimum_velocity_cm_s: float) -> None:
        if minimum_velocity_cm_s <= 0.0:
            raise ValueError("minimum_velocity_cm_s must be > 0")
        self.rng = rng
        self.minimum_velocity_cm_s = float(minimum_velocity_cm_s)

    @staticmethod
    def poiseuille_velocity_cm_s(
        radial_position_cm: float,
        radius_cm: float,
        mean_velocity_cm_s: float,
        minimum_velocity_cm_s: float = 0.001,
    ) -> float:
        if radius_cm <= 0.0:
            raise ValueError("radius_cm must be > 0")
        if mean_velocity_cm_s <= 0.0:
            raise ValueError("mean_velocity_cm_s must be > 0")
        if radial_position_cm < 0.0 or radial_position_cm > radius_cm:
            raise ValueError("radial_position_cm must satisfy 0 <= r <= R")

        ratio = radial_position_cm / radius_cm
        v = 2.0 * mean_velocity_cm_s * (1.0 - ratio * ratio)
        return max(v, minimum_velocity_cm_s)

    def sample_area_uniform_position(self, radius_cm: float) -> Tuple[float, float]:
        """Sample r and theta uniformly in circular area."""
        if radius_cm <= 0.0:
            raise ValueError("radius_cm must be > 0")
        u1, u2 = self.rng.random(2)
        radial_position_cm = radius_cm * math.sqrt(float(u1))
        theta_rad = 2.0 * math.pi * float(u2)
        return radial_position_cm, theta_rad

    @staticmethod
    def orthonormal_perpendicular_basis(direction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Construct two orthonormal vectors perpendicular to direction."""
        d = np.asarray(direction, dtype=float)
        norm = float(np.linalg.norm(d))
        if norm <= 0.0:
            raise ValueError("Direction vector cannot be zero")
        d = d / norm

        # Pick an axis least aligned with d; this avoids a nearly-zero cross product.
        helper_index = int(np.argmin(np.abs(d)))
        helper = np.zeros(3, dtype=float)
        helper[helper_index] = 1.0

        u = np.cross(d, helper)
        u_norm = float(np.linalg.norm(u))
        if u_norm <= 0.0:
            # This is theoretically unreachable with the least-aligned axis,
            # but the explicit fallback keeps the routine numerically robust.
            helper = np.array([1.0, 0.0, 0.0], dtype=float)
            if abs(float(np.dot(d, helper))) > 0.9:
                helper = np.array([0.0, 1.0, 0.0], dtype=float)
            u = np.cross(d, helper)
            u_norm = float(np.linalg.norm(u))
        u /= u_norm

        v = np.cross(d, u)
        v_norm = float(np.linalg.norm(v))
        if v_norm <= 0.0:
            raise ValueError("Failed to construct perpendicular basis")
        v /= v_norm
        return u, v

    def traverse_vessel(self, region: VesselRegion) -> SampledVesselTraversal:
        if not region.is_normal_vessel:
            raise ValueError("traverse_vessel requires a SubType 1-9 region")
        if region.diameter_cm is None or region.mean_velocity_cm_s is None:
            raise ValueError(f"Normal vessel {region.object_id!r} lacks diameter or mean velocity")
        if region.direction is None or region.length_cm <= 0.0:
            raise ValueError(f"Normal vessel {region.object_id!r} has invalid length/direction")

        radius_cm = region.diameter_cm / 2.0
        radial_position_cm, theta_rad = self.sample_area_uniform_position(radius_cm)
        local_velocity_cm_s = self.poiseuille_velocity_cm_s(
            radial_position_cm,
            radius_cm,
            region.mean_velocity_cm_s,
            self.minimum_velocity_cm_s,
        )
        residence_time_s = region.length_cm / local_velocity_cm_s

        u, v = self.orthonormal_perpendicular_basis(region.direction)
        entry_position = (
            region.start
            + radial_position_cm * math.cos(theta_rad) * u
            + radial_position_cm * math.sin(theta_rad) * v
        )
        exit_position = region.stop.copy()

        return SampledVesselTraversal(
            radial_position_cm=radial_position_cm,
            theta_rad=theta_rad,
            local_velocity_cm_s=local_velocity_cm_s,
            entry_position=entry_position,
            exit_position=exit_position,
            residence_time_s=residence_time_s,
        )


class OrganModel:
    """Effective arteriole-capillary-venule organ residence model."""

    def __init__(self, flow_model: LaminarFlowModel, config: Configuration) -> None:
        self.flow_model = flow_model
        self.sections: Tuple[OrganSection, ...] = (
            OrganSection(
                name="arteriole",
                diameter_cm=config.organ_arteriole_diameter_cm,
                length_cm=config.organ_arteriole_length_cm,
                mean_velocity_cm_s=config.organ_arteriole_mean_velocity_cm_s,
            ),
            OrganSection(
                name="capillary",
                diameter_cm=config.organ_capillary_diameter_cm,
                length_cm=config.organ_capillary_length_cm,
                mean_velocity_cm_s=config.organ_capillary_mean_velocity_cm_s,
            ),
            OrganSection(
                name="venule",
                diameter_cm=config.organ_venule_diameter_cm,
                length_cm=config.organ_venule_length_cm,
                mean_velocity_cm_s=config.organ_venule_mean_velocity_cm_s,
            ),
        )
        for section in self.sections:
            if section.diameter_cm <= 0.0 or section.length_cm <= 0.0 or section.mean_velocity_cm_s <= 0.0:
                raise ValueError(f"Invalid organ section parameters for {section.name}")

    def simulate(self, entry_time_s: float) -> Tuple[float, List[OrganSectionTraversal]]:
        current_time = float(entry_time_s)
        traversals: List[OrganSectionTraversal] = []

        for section in self.sections:
            radius_cm = section.diameter_cm / 2.0
            r_cm, theta_rad = self.flow_model.sample_area_uniform_position(radius_cm)
            v_cm_s = self.flow_model.poiseuille_velocity_cm_s(
                r_cm,
                radius_cm,
                section.mean_velocity_cm_s,
                self.flow_model.minimum_velocity_cm_s,
            )
            residence_s = section.length_cm / v_cm_s
            section_entry = current_time
            section_exit = current_time + residence_s
            current_time = section_exit

            traversals.append(
                OrganSectionTraversal(
                    section_name=section.name,
                    entry_time_s=section_entry,
                    exit_time_s=section_exit,
                    residence_time_s=residence_s,
                    radial_position_cm=r_cm,
                    theta_rad=theta_rad,
                    local_velocity_cm_s=v_cm_s,
                )
            )

        return current_time, traversals


class HeartModel:
    """Exponential heart residence-time model parameterized by the mean."""

    def __init__(self, rng: np.random.Generator, left_subtype: int, right_subtype: int, left_mean_s: float, right_mean_s: float) -> None:
        if left_mean_s <= 0.0 or right_mean_s <= 0.0:
            raise ValueError("Heart mean residence times must be > 0")
        self.rng = rng
        self.means = {int(left_subtype): float(left_mean_s), int(right_subtype): float(right_mean_s)}

    def sample_residence(self, subtype: int) -> float:
        if subtype not in self.means:
            raise ValueError(f"HeartModel received non-heart SubType={subtype}")
        # U is sampled strictly below 1 in NumPy's Generator API.
        u = float(self.rng.random())
        mean_s = self.means[subtype]
        return -mean_s * math.log1p(-u)


# ============================================================================
# Routing
# ============================================================================

class BranchRouter:
    """Select downstream objects using area weighting or uniform fallback."""

    def __init__(self, network: VasculatureNetwork, rng: np.random.Generator) -> None:
        self.network = network
        self.rng = rng

    def probabilities(self, candidate_ids: Sequence[ObjectID]) -> np.ndarray:
        if not candidate_ids:
            raise ValueError("candidate_ids cannot be empty")

        diameters = [self.network.get(cid).diameter_cm for cid in candidate_ids]
        if all(d is not None and d > 0.0 for d in diameters):
            # A = pi*(D/2)^2 = (pi/4)D^2, so the common constant cancels.
            weights = np.square(np.asarray([float(d) for d in diameters], dtype=float))
            probabilities = weights / weights.sum()
            return probabilities

        # Exact requested rule: if any candidate has no valid diameter,
        # routing is uniform across ALL candidates. Do not partially weight.
        return np.full(len(candidate_ids), 1.0 / len(candidate_ids), dtype=float)

    def choose(self, candidate_ids: Sequence[ObjectID]) -> Tuple[ObjectID, np.ndarray]:
        probs = self.probabilities(candidate_ids)
        index = int(self.rng.choice(len(candidate_ids), p=probs))
        return candidate_ids[index], probs


# ============================================================================
# Recorder
# ============================================================================

class SimulationRecorder:
    """Accumulates primary, detailed, and event records."""

    def __init__(self) -> None:
        self.cycles: List[CycleRecord] = []
        self.details: List[DetailedTraversal] = []
        self.events: List[Event] = []

    def add_cycle(self, record: CycleRecord) -> None:
        self.cycles.append(record)

    def add_detail(self, detail: DetailedTraversal) -> None:
        self.details.append(detail)

    def add_event(self, event: Event) -> None:
        self.events.append(event)

    def write_csvs(
        self,
        primary_path: str,
        detailed_path: str,
        event_path: str,
        write_event_log: bool = True,
    ) -> None:
        self._write_primary(primary_path, self.cycles)
        self._write_detailed(detailed_path, self.details)
        if write_event_log:
            self._write_events(event_path, self.events)

    @staticmethod
    def _write_primary(path: str, records: Sequence[CycleRecord]) -> None:
        fieldnames = [
            "Nanobot_ID",
            "Cycle_Number",
            "Organ_ID",
            "Starting_Heart_ID",
            "Expected_Destination_Heart_ID",
            "Destination_Heart_ID",
            "Total_Circulation_Time_s",
            "Number_of_Passed_Objects",
            "Residence_Times_s",
            "Cycle_Status",
            "Termination_Reason",
            "Passed_Object_IDs",
            "Expanded_Passed_Object_Units",
        ]
        with Path(path).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        "Nanobot_ID": record.nanobot_id,
                        "Cycle_Number": record.cycle_number,
                        "Organ_ID": "" if record.organ_id is None else record.organ_id,
                        "Starting_Heart_ID": record.starting_heart_id,
                        "Expected_Destination_Heart_ID": record.expected_destination_heart_id,
                        "Destination_Heart_ID": "" if record.destination_heart_id is None else record.destination_heart_id,
                        "Total_Circulation_Time_s": f"{record.total_circulation_time_s:.12g}",
                        "Number_of_Passed_Objects": record.number_of_passed_objects,
                        "Residence_Times_s": json_dumps_compact(record.residence_times_s),
                        "Cycle_Status": record.cycle_status,
                        "Termination_Reason": record.termination_reason,
                        "Passed_Object_IDs": json_dumps_compact(record.passed_object_ids),
                        "Expanded_Passed_Object_Units": json_dumps_compact(record.passed_object_units),
                    }
                )

    @staticmethod
    def _write_detailed(path: str, records: Sequence[DetailedTraversal]) -> None:
        fieldnames = [
            "Nanobot_ID",
            "Cycle_Number",
            "Starting_Heart_ID",
            "Object_ID",
            "Object_Name",
            "SubType",
            "Entry_Time_s",
            "Exit_Time_s",
            "Residence_Time_s",
            "StartX_cm",
            "StartY_cm",
            "StartZ_cm",
            "StopX_cm",
            "StopY_cm",
            "StopZ_cm",
            "Length_cm",
            "Diameter_cm",
            "Radius_cm",
            "Mean_Velocity_cm_s",
            "Radial_Position_cm",
            "Theta_rad",
            "Local_Velocity_cm_s",
            "Organ_Arteriole_Entry_s",
            "Organ_Arteriole_Exit_s",
            "Organ_Arteriole_Residence_s",
            "Organ_Arteriole_Radial_cm",
            "Organ_Arteriole_Theta_rad",
            "Organ_Arteriole_Local_Velocity_cm_s",
            "Organ_Capillary_Entry_s",
            "Organ_Capillary_Exit_s",
            "Organ_Capillary_Residence_s",
            "Organ_Capillary_Radial_cm",
            "Organ_Capillary_Theta_rad",
            "Organ_Capillary_Local_Velocity_cm_s",
            "Organ_Venule_Entry_s",
            "Organ_Venule_Exit_s",
            "Organ_Venule_Residence_s",
            "Organ_Venule_Radial_cm",
            "Organ_Venule_Theta_rad",
            "Organ_Venule_Local_Velocity_cm_s",
        ]

        with Path(path).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                section_map = {s.section_name: s for s in record.organ_sections}
                row: Dict[str, Any] = {
                    "Nanobot_ID": record.nanobot_id,
                    "Cycle_Number": record.cycle_number,
                    "Starting_Heart_ID": record.starting_heart_id,
                    "Object_ID": record.object_id,
                    "Object_Name": record.object_name,
                    "SubType": record.subtype,
                    "Entry_Time_s": f"{record.entry_time_s:.12g}",
                    "Exit_Time_s": f"{record.exit_time_s:.12g}",
                    "Residence_Time_s": f"{record.residence_time_s:.12g}",
                    "StartX_cm": record.start[0],
                    "StartY_cm": record.start[1],
                    "StartZ_cm": record.start[2],
                    "StopX_cm": record.stop[0],
                    "StopY_cm": record.stop[1],
                    "StopZ_cm": record.stop[2],
                    "Length_cm": record.length_cm,
                    "Diameter_cm": "" if record.diameter_cm is None else record.diameter_cm,
                    "Radius_cm": "" if record.radius_cm is None else record.radius_cm,
                    "Mean_Velocity_cm_s": "" if record.mean_velocity_cm_s is None else record.mean_velocity_cm_s,
                    "Radial_Position_cm": "" if record.radial_position_cm is None else record.radial_position_cm,
                    "Theta_rad": "" if record.theta_rad is None else record.theta_rad,
                    "Local_Velocity_cm_s": "" if record.local_velocity_cm_s is None else record.local_velocity_cm_s,
                }

                for section_name, prefix in (
                    ("arteriole", "Organ_Arteriole"),
                    ("capillary", "Organ_Capillary"),
                    ("venule", "Organ_Venule"),
                ):
                    s = section_map.get(section_name)
                    if s is None:
                        for suffix in (
                            "Entry_s",
                            "Exit_s",
                            "Residence_s",
                            "Radial_cm",
                            "Theta_rad",
                            "Local_Velocity_cm_s",
                        ):
                            row[f"{prefix}_{suffix}"] = ""
                    else:
                        row[f"{prefix}_Entry_s"] = f"{s.entry_time_s:.12g}"
                        row[f"{prefix}_Exit_s"] = f"{s.exit_time_s:.12g}"
                        row[f"{prefix}_Residence_s"] = f"{s.residence_time_s:.12g}"
                        row[f"{prefix}_Radial_cm"] = s.radial_position_cm
                        row[f"{prefix}_Theta_rad"] = s.theta_rad
                        row[f"{prefix}_Local_Velocity_cm_s"] = s.local_velocity_cm_s

                writer.writerow(row)

    @staticmethod
    def _write_events(path: str, records: Sequence[Event]) -> None:
        fieldnames = [
            "time_s",
            "nanobot_id",
            "event_type",
            "from_region_id",
            "from_region_name",
            "to_region_id",
            "to_region_name",
            "x",
            "y",
            "z",
            "residence_time_s",
            "reason",
        ]
        with Path(path).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for event in records:
                writer.writerow(
                    {
                        "time_s": f"{event.time_s:.12g}",
                        "nanobot_id": event.nanobot_id,
                        "event_type": event.event_type,
                        "from_region_id": "" if event.from_region_id is None else event.from_region_id,
                        "from_region_name": "" if event.from_region_name is None else event.from_region_name,
                        "to_region_id": "" if event.to_region_id is None else event.to_region_id,
                        "to_region_name": "" if event.to_region_name is None else event.to_region_name,
                        "x": "" if event.x is None else event.x,
                        "y": "" if event.y is None else event.y,
                        "z": "" if event.z is None else event.z,
                        "residence_time_s": "" if event.residence_time_s is None else event.residence_time_s,
                        "reason": "" if event.reason is None else event.reason,
                    }
                )


# ============================================================================
# Cycle simulator
# ============================================================================

class CycleSimulator:
    """Generate independent event-driven circulation realizations."""

    def __init__(
        self,
        network: VasculatureNetwork,
        config: Configuration,
        rng: np.random.Generator,
        recorder: SimulationRecorder,
    ) -> None:
        self.network = network
        self.config = config
        self.rng = rng
        self.recorder = recorder

        self.flow_model = LaminarFlowModel(rng, config.minimum_velocity_cm_s)
        self.organ_model = OrganModel(self.flow_model, config)
        self.heart_model = HeartModel(
            rng,
            config.left_heart_subtype,
            config.right_heart_subtype,
            config.left_heart_mean_residence_s,
            config.right_heart_mean_residence_s,
        )
        self.router = BranchRouter(network, rng)

        self.left_heart_id = network.left_heart_id
        self.right_heart_id = network.right_heart_id

    def start_heart_for_cycle(self, cycle_index: int) -> ObjectID:
        """Cycle 1 can start Left/Right; subsequent cycles alternate."""
        if self.config.start_with_left_heart:
            return self.left_heart_id if cycle_index % 2 == 1 else self.right_heart_id
        return self.right_heart_id if cycle_index % 2 == 1 else self.left_heart_id

    def expected_destination(self, starting_heart_id: ObjectID) -> ObjectID:
        if starting_heart_id == self.left_heart_id:
            return self.right_heart_id
        if starting_heart_id == self.right_heart_id:
            return self.left_heart_id
        raise ValueError(f"Unknown starting heart ID {starting_heart_id!r}")

    def run(self) -> None:
        for cycle_number in range(1, self.config.number_of_cycles + 1):
            record, details = self.simulate_cycle(cycle_number)
            self.recorder.add_cycle(record)
            for detail in details:
                self.recorder.add_detail(detail)

    def simulate_cycle(self, cycle_number: int) -> Tuple[CycleRecord, List[DetailedTraversal]]:
        nanobot_id = cycle_number
        start_heart_id = self.start_heart_for_cycle(cycle_number)
        expected_destination_id = self.expected_destination(start_heart_id)

        current_time = 0.0
        residence_times: List[float] = []
        passed_object_ids: List[ObjectID] = []
        passed_object_units: List[str] = []
        details: List[DetailedTraversal] = []
        organ_ids_seen: List[ObjectID] = []
        cycle_status = "COMPLETED"
        termination_reason = ""
        destination_heart_id: Optional[ObjectID] = None
        current_position = self.network.get(start_heart_id).start.copy()

        self._event(
            time_s=current_time,
            nanobot_id=nanobot_id,
            event_type="CYCLE_START",
            to_region_id=start_heart_id,
            to_region_name=self.network.get(start_heart_id).name,
            x=current_position[0],
            y=current_position[1],
            z=current_position[2],
        )

        current_id = start_heart_id

        while True:
            region = self.network.get(current_id)
            traversal_cost = 3 if region.is_organ else 1
            if len(residence_times) + traversal_cost > self.config.max_objects_per_cycle:
                cycle_status = "ABORTED"
                termination_reason = "max_objects"
                self._event(
                    time_s=current_time,
                    nanobot_id=nanobot_id,
                    event_type="CYCLE_ABORTED",
                    from_region_id=current_id,
                    from_region_name=region.name,
                    x=current_position[0],
                    y=current_position[1],
                    z=current_position[2],
                    reason=termination_reason,
                )
                break

            # -----------------------------------------------------------------
            # Enter/simulate the CURRENT object. Destination heart is never
            # placed here; it is intercepted at the downstream-selection step.
            # -----------------------------------------------------------------
            entry_time = current_time
            self._event(
                time_s=entry_time,
                nanobot_id=nanobot_id,
                event_type="HEART_ENTRY" if region.is_heart else "ORGAN_ENTRY" if region.is_organ else "VESSEL_ENTRY",
                from_region_id=(passed_object_ids[-1] if passed_object_ids else None),
                from_region_name=(self.network.get(passed_object_ids[-1]).name if passed_object_ids else None),
                to_region_id=region.object_id,
                to_region_name=region.name,
                x=current_position[0],
                y=current_position[1],
                z=current_position[2],
            )

            if region.is_heart:
                residence = self.heart_model.sample_residence(region.subtype)
                residence_times.append(residence)
                passed_object_ids.append(region.object_id)
                passed_object_units.append(str(region.object_id))
                exit_time = current_time + residence

                if exit_time > self.config.max_cycle_time_s:
                    current_time = exit_time
                    cycle_status = "ABORTED"
                    termination_reason = "max_cycle_time"
                    self._add_detail(
                        details,
                        nanobot_id,
                        cycle_number,
                        start_heart_id,
                        region,
                        entry_time,
                        exit_time,
                        residence,
                        None,
                    )
                    current_position = region.stop.copy()
                    self._event(
                        time_s=exit_time,
                        nanobot_id=nanobot_id,
                        event_type="HEART_EXIT",
                        from_region_id=region.object_id,
                        from_region_name=region.name,
                        to_region_id=None,
                        to_region_name=None,
                        x=current_position[0],
                        y=current_position[1],
                        z=current_position[2],
                        residence_time_s=residence,
                    )
                    self._event(
                        time_s=exit_time,
                        nanobot_id=nanobot_id,
                        event_type="CYCLE_ABORTED",
                        from_region_id=region.object_id,
                        from_region_name=region.name,
                        x=current_position[0],
                        y=current_position[1],
                        z=current_position[2],
                        reason=termination_reason,
                    )
                    break

                current_time = exit_time
                current_position = region.stop.copy()
                self._add_detail(
                    details,
                    nanobot_id,
                    cycle_number,
                    start_heart_id,
                    region,
                    entry_time,
                    exit_time,
                    residence,
                    None,
                )
                self._event(
                    time_s=exit_time,
                    nanobot_id=nanobot_id,
                    event_type="HEART_EXIT",
                    from_region_id=region.object_id,
                    from_region_name=region.name,
                    x=current_position[0],
                    y=current_position[1],
                    z=current_position[2],
                    residence_time_s=residence,
                )

            elif region.is_organ:
                organ_ids_seen.append(region.object_id)
                before = current_time
                current_time, section_traversals = self.organ_model.simulate(current_time)
                total_organ_residence = current_time - before

                # The primary residence-time list contains the three section
                # times, not one aggregate organ time.
                residence_times.extend([s.residence_time_s for s in section_traversals])
                passed_object_ids.append(region.object_id)
                passed_object_units.extend([f"{region.object_id}:{s.section_name}" for s in section_traversals])
                current_position = region.stop.copy()
                self._add_detail(
                    details,
                    nanobot_id,
                    cycle_number,
                    start_heart_id,
                    region,
                    entry_time,
                    current_time,
                    total_organ_residence,
                    section_traversals,
                )

                for s in section_traversals:
                    self._event(
                        time_s=s.entry_time_s,
                        nanobot_id=nanobot_id,
                        event_type="ORGAN_SECTION_ENTRY",
                        from_region_id=region.object_id,
                        from_region_name=region.name,
                        to_region_id=region.object_id,
                        to_region_name=f"{region.name}:{s.section_name}",
                        x=region.start[0],
                        y=region.start[1],
                        z=region.start[2],
                        reason=json_dumps_compact(
                            {
                                "section": s.section_name,
                                "radial_position_cm": s.radial_position_cm,
                                "theta_rad": s.theta_rad,
                                "local_velocity_cm_s": s.local_velocity_cm_s,
                            }
                        ),
                    )
                    self._event(
                        time_s=s.exit_time_s,
                        nanobot_id=nanobot_id,
                        event_type="ORGAN_SECTION_EXIT",
                        from_region_id=region.object_id,
                        from_region_name=f"{region.name}:{s.section_name}",
                        to_region_id=region.object_id,
                        to_region_name=region.name,
                        x=region.stop[0],
                        y=region.stop[1],
                        z=region.stop[2],
                        residence_time_s=s.residence_time_s,
                    )

                self._event(
                    time_s=current_time,
                    nanobot_id=nanobot_id,
                    event_type="ORGAN_EXIT",
                    from_region_id=region.object_id,
                    from_region_name=region.name,
                    x=current_position[0],
                    y=current_position[1],
                    z=current_position[2],
                    residence_time_s=total_organ_residence,
                )

            else:
                traversal = self.flow_model.traverse_vessel(region)
                residence_times.append(traversal.residence_time_s)
                passed_object_ids.append(region.object_id)
                passed_object_units.append(str(region.object_id))
                exit_time = current_time + traversal.residence_time_s
                current_time = exit_time
                current_position = traversal.exit_position.copy()
                self._add_detail(
                    details,
                    nanobot_id,
                    cycle_number,
                    start_heart_id,
                    region,
                    entry_time,
                    exit_time,
                    traversal.residence_time_s,
                    None,
                    traversal,
                )
                self._event(
                    time_s=exit_time,
                    nanobot_id=nanobot_id,
                    event_type="VESSEL_EXIT",
                    from_region_id=region.object_id,
                    from_region_name=region.name,
                    x=current_position[0],
                    y=current_position[1],
                    z=current_position[2],
                    residence_time_s=traversal.residence_time_s,
                )

            # -------------------------------------------------------------
            # Safety limit after object traversal.
            # -------------------------------------------------------------
            if current_time > self.config.max_cycle_time_s:
                cycle_status = "ABORTED"
                termination_reason = "max_cycle_time"
                self._event(
                    time_s=current_time,
                    nanobot_id=nanobot_id,
                    event_type="CYCLE_ABORTED",
                    from_region_id=current_id,
                    from_region_name=region.name,
                    x=current_position[0],
                    y=current_position[1],
                    z=current_position[2],
                    reason=termination_reason,
                )
                break

            # -------------------------------------------------------------
            # Select downstream object. The selected destination heart is
            # detected BEFORE entering it.
            # -------------------------------------------------------------
            candidates = self.network.candidates(current_id)
            if not candidates:
                cycle_status = "INVALID"
                termination_reason = "dead_end"
                self._event(
                    time_s=current_time,
                    nanobot_id=nanobot_id,
                    event_type="CYCLE_INVALID",
                    from_region_id=current_id,
                    from_region_name=region.name,
                    x=current_position[0],
                    y=current_position[1],
                    z=current_position[2],
                    reason=termination_reason,
                )
                break

            next_id, probabilities = self.router.choose(candidates)
            candidate_payload = []
            for cid, prob in zip(candidates, probabilities):
                candidate_region = self.network.get(cid)
                candidate_payload.append(
                    {
                        "id": cid,
                        "subtype": candidate_region.subtype,
                        "diameter_cm": candidate_region.diameter_cm,
                        "probability": float(prob),
                    }
                )
            self._event(
                time_s=current_time,
                nanobot_id=nanobot_id,
                event_type="BRANCH_SELECTION",
                from_region_id=current_id,
                from_region_name=region.name,
                to_region_id=next_id,
                to_region_name=self.network.get(next_id).name,
                x=current_position[0],
                y=current_position[1],
                z=current_position[2],
                reason=json_dumps_compact(
                    {
                        "candidates": candidate_payload,
                        "selected": next_id,
                        "selection_rule": (
                            "area_weighted" if np.all(probabilities > 0) and len(candidates) > 1 and
                            all(self.network.get(cid).diameter_cm is not None for cid in candidates)
                            else "uniform_due_to_missing_diameter"
                            if len(candidates) > 1
                            else "single_candidate"
                        ),
                    }
                ),
            )

            next_region = self.network.get(next_id)
            if next_region.is_heart:
                if next_id == expected_destination_id:
                    destination_heart_id = next_id
                    cycle_status = "COMPLETED"
                    termination_reason = "destination_heart_selected"
                    self._event(
                        time_s=current_time,
                        nanobot_id=nanobot_id,
                        event_type="CYCLE_COMPLETE",
                        from_region_id=current_id,
                        from_region_name=region.name,
                        to_region_id=next_id,
                        to_region_name=next_region.name,
                        x=current_position[0],
                        y=current_position[1],
                        z=current_position[2],
                        reason=termination_reason,
                    )
                    break

                cycle_status = "INVALID"
                if next_id == start_heart_id:
                    termination_reason = "same_starting_heart_selected"
                else:
                    termination_reason = "wrong_heart_selected_before_destination"
                self._event(
                    time_s=current_time,
                    nanobot_id=nanobot_id,
                    event_type="CYCLE_INVALID",
                    from_region_id=current_id,
                    from_region_name=region.name,
                    to_region_id=next_id,
                    to_region_name=next_region.name,
                    x=current_position[0],
                    y=current_position[1],
                    z=current_position[2],
                    reason=termination_reason,
                )
                break

            current_id = next_id

        # The completed-cycle definition and the invalid-cycle diagnostic both
        # validate the organ count explicitly.
        if cycle_status == "COMPLETED":
            if len(organ_ids_seen) == 0:
                cycle_status = "INVALID"
                termination_reason = "invalid_organ_count_zero"
                self._event(
                    time_s=current_time,
                    nanobot_id=nanobot_id,
                    event_type="CYCLE_INVALID",
                    x=current_position[0],
                    y=current_position[1],
                    z=current_position[2],
                    reason=termination_reason,
                )
            elif len(organ_ids_seen) > 1:
                cycle_status = "INVALID"
                termination_reason = "invalid_organ_count_multiple"
                self._event(
                    time_s=current_time,
                    nanobot_id=nanobot_id,
                    event_type="CYCLE_INVALID",
                    x=current_position[0],
                    y=current_position[1],
                    z=current_position[2],
                    reason=termination_reason,
                )
        # The output specification identifies a single Organ_ID for valid
        # cycles. For invalid cycles with multiple organs, preserve a compact
        # representation rather than silently choosing one.
        if len(organ_ids_seen) == 1:
            organ_id: Optional[ObjectID] = organ_ids_seen[0]
        elif len(organ_ids_seen) == 0:
            organ_id = None
        else:
            organ_id = json_dumps_compact(organ_ids_seen)

        expected_count = len(residence_times)
        expanded_count = len(passed_object_units)
        if expanded_count != expected_count:
            raise RuntimeError(
                f"Internal accounting error in cycle {cycle_number}: "
                f"expanded object count={expanded_count}, residence-time count={expected_count}"
            )

        total_time = float(sum(residence_times))
        # current_time should agree exactly to normal floating-point summation;
        # total_time is defined from the recorded residence-time list.
        record = CycleRecord(
            nanobot_id=nanobot_id,
            cycle_number=cycle_number,
            organ_id=organ_id,
            starting_heart_id=start_heart_id,
            expected_destination_heart_id=expected_destination_id,
            destination_heart_id=destination_heart_id,
            total_circulation_time_s=total_time,
            number_of_passed_objects=expanded_count,
            residence_times_s=residence_times,
            cycle_status=cycle_status,
            termination_reason=termination_reason,
            passed_object_ids=passed_object_ids,
            passed_object_units=passed_object_units,
        )
        return record, details

    def _add_detail(
        self,
        details: List[DetailedTraversal],
        nanobot_id: int,
        cycle_number: int,
        start_heart_id: ObjectID,
        region: VesselRegion,
        entry_time: float,
        exit_time: float,
        residence: float,
        organ_sections: Optional[List[OrganSectionTraversal]],
        traversal: Optional[SampledVesselTraversal] = None,
    ) -> None:
        diameter = region.diameter_cm
        radius = None if diameter is None else diameter / 2.0
        radial = None if traversal is None else traversal.radial_position_cm
        theta = None if traversal is None else traversal.theta_rad
        velocity = None if traversal is None else traversal.local_velocity_cm_s
        details.append(
            DetailedTraversal(
                nanobot_id=nanobot_id,
                cycle_number=cycle_number,
                starting_heart_id=start_heart_id,
                object_id=region.object_id,
                object_name=region.name,
                subtype=region.subtype,
                entry_time_s=entry_time,
                exit_time_s=exit_time,
                residence_time_s=residence,
                start=region.start.copy(),
                stop=region.stop.copy(),
                length_cm=region.length_cm,
                diameter_cm=diameter,
                radius_cm=radius,
                mean_velocity_cm_s=region.mean_velocity_cm_s,
                radial_position_cm=radial,
                theta_rad=theta,
                local_velocity_cm_s=velocity,
                organ_sections=[] if organ_sections is None else list(organ_sections),
            )
        )

    def _event(
        self,
        time_s: float,
        nanobot_id: int,
        event_type: str,
        from_region_id: Optional[ObjectID] = None,
        from_region_name: Optional[str] = None,
        to_region_id: Optional[ObjectID] = None,
        to_region_name: Optional[str] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        residence_time_s: Optional[float] = None,
        reason: Optional[str] = None,
    ) -> None:
        self.recorder.add_event(
            Event(
                time_s=float(time_s),
                nanobot_id=nanobot_id,
                event_type=event_type,
                from_region_id=from_region_id,
                from_region_name=from_region_name,
                to_region_id=to_region_id,
                to_region_name=to_region_name,
                x=None if x is None else float(x),
                y=None if y is None else float(y),
                z=None if z is None else float(z),
                residence_time_s=None if residence_time_s is None else float(residence_time_s),
                reason=reason,
            )
        )


# ============================================================================
# Visualization
# ============================================================================

SUBTYPE_NAMES = {
    0: "ORGAN",
    1: "Aorta",
    2: "Large arteries",
    3: "Small arteries",
    4: "Arterioles",
    5: "Capillaries",
    6: "Venules",
    7: "Small veins",
    8: "Large veins",
    9: "Venae cava",
    10: "Left heart",
    11: "Right heart",
}


def plot_topology(network: VasculatureNetwork) -> None:
    """Display an optional non-interactive-by-default Matplotlib 3-D topology plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "--plot-topology requires matplotlib. Install it with 'pip install matplotlib'."
        ) from exc

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection="3d")

    subtypes = sorted({r.subtype for r in network.regions.values()})
    cmap = plt.get_cmap("tab20")
    color_for = {subtype: cmap(i / max(1, len(subtypes) - 1)) for i, subtype in enumerate(subtypes)}

    for region in network.regions.values():
        xyz = np.vstack([region.start, region.stop])
        ax.plot(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            linewidth=3.0 if region.is_heart or region.is_organ else 1.2,
            color=color_for[region.subtype],
        )

    # Make X, Y, and Z use the same distance scale
    all_points = np.vstack([
        np.vstack([region.start, region.stop])
        for region in network.regions.values()
    ])

    x_min, y_min, z_min = all_points.min(axis=0)
    x_max, y_max, z_max = all_points.max(axis=0)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    # Equal physical scaling for X, Y, and Z
    ax.set_box_aspect((
        x_max - x_min,
        y_max - y_min,
        z_max - z_min
    ))
    
    # One legend handle per subtype, not per region.
    import matplotlib.lines as mlines

    handles = [
        mlines.Line2D([], [], color=color_for[subtype], linewidth=2.0, label=f"{subtype}: {SUBTYPE_NAMES.get(subtype, 'Unknown')}")
        for subtype in subtypes
    ]
    ax.legend(handles=handles, loc="best", fontsize=8)


    # Camera angle
    ax.view_init(elev=25, azim=-55)

    # Axis labels with more separation from the axes
    ax.set_xlabel("X (cm)", labelpad=12)
    ax.set_ylabel("Y (cm)", labelpad=12)
    ax.set_zlabel("Z (cm)", labelpad=15)

    # Increase distance between tick labels and axes
    ax.tick_params(axis="x", pad=4)
    ax.tick_params(axis="y", pad=4)
    ax.tick_params(axis="z", pad=8)

    # Fewer Z ticks to prevent overlap
    z_ticks = np.linspace(z_min, z_max, 3)
    ax.set_zticks(z_ticks)


    ax.set_title("Pvasculature topology")
    plt.tight_layout()
    plt.show()


# ============================================================================
# Simulation reporting / CLI
# ============================================================================

@dataclass
class SimulationSummary:
    requested_cycles: int
    completed_cycles: int
    invalid_cycles: int
    aborted_cycles: int
    left_start_cycles: int
    right_start_cycles: int
    total_output_rows: int
    elapsed_wall_time_s: float


def summarize(records: Sequence[CycleRecord], requested_cycles: int, elapsed_s: float, network: VasculatureNetwork) -> SimulationSummary:
    completed = sum(r.cycle_status == "COMPLETED" for r in records)
    invalid = sum(r.cycle_status == "INVALID" for r in records)
    aborted = sum(r.cycle_status == "ABORTED" for r in records)
    left_starts = sum(r.starting_heart_id == network.left_heart_id for r in records)
    right_starts = sum(r.starting_heart_id == network.right_heart_id for r in records)
    return SimulationSummary(
        requested_cycles=requested_cycles,
        completed_cycles=completed,
        invalid_cycles=invalid,
        aborted_cycles=aborted,
        left_start_cycles=left_starts,
        right_start_cycles=right_starts,
        total_output_rows=len(records),
        elapsed_wall_time_s=elapsed_s,
    )


def print_start_summary(network: VasculatureNetwork, config: Configuration) -> None:
    print("Blood-Voyager-style laminar Monte Carlo simulator")
    print(f"Input file:                {config.input_file}")
    print(f"Topology objects:          {len(network.regions)}")
    print(f"Organs:                    {len(network.organ_ids)}")
    print(f"Left-heart actual ID:      {network.left_heart_id}")
    print(f"Right-heart actual ID:     {network.right_heart_id}")
    print(f"Cycles:                    {config.number_of_cycles}")
    print(f"Random seed:               {config.random_seed}")
    print(f"Topology tolerance (cm):   {config.topology_tolerance_cm:g}")
    print(f"Minimum velocity (cm/s):   {config.minimum_velocity_cm_s:g}")


def print_end_summary(summary: SimulationSummary) -> None:
    print("\nSimulation complete")
    print(f"Requested cycles:           {summary.requested_cycles}")
    print(f"Completed cycles:           {summary.completed_cycles}")
    print(f"Invalid cycles:             {summary.invalid_cycles}")
    print(f"Aborted cycles:             {summary.aborted_cycles}")
    print(f"Cycles starting at left:   {summary.left_start_cycles}")
    print(f"Cycles starting at right:  {summary.right_start_cycles}")
    print(f"Total output rows:          {summary.total_output_rows}")
    print(f"Elapsed wall time (s):      {summary.elapsed_wall_time_s:.3f}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Event-driven Monte Carlo circulation simulator using continuous Poiseuille flow."
    )
    parser.add_argument("--input", default="Pvasculature.csv", help="Input vascular topology CSV")
    parser.add_argument("--output", default="PcsvNanoSec.csv", help="Primary output CSV")
    parser.add_argument("--detailed-output", default="PcsvNanoSecDetailed.csv", help="Detailed traversal CSV")
    parser.add_argument("--event-log", default="nanobot_events.csv", help="Event log CSV")
    parser.add_argument("--num-cycles", type=int, default=1000, help="Number of independent circulation realizations")
    parser.add_argument("--seed", type=int, default=12345, help="NumPy random seed")
    parser.add_argument("--min-velocity", type=float, default=0.001, help="Minimum local velocity in cm/s")
    parser.add_argument("--topology-tolerance", type=float, default=1e-6, help="Stop-to-Start graph tolerance in cm")
    parser.add_argument("--max-cycle-time", type=float, default=1_000_000.0, help="Maximum cycle time in seconds")
    parser.add_argument("--max-objects", type=int, default=10_000, help="Maximum expanded objects per cycle")
    parser.add_argument("--plot-topology", action="store_true", help="Display a 3-D topology plot")
    parser.add_argument("--no-event-log", action="store_true", help="Do not create nanobot_events.csv")
    parser.add_argument("--verbose", action="store_true", help="Print non-fatal topology diagnostics")
    parser.add_argument("--strict-validation", action="store_true", help="Stop after the first invalid/aborted cycle")
    parser.add_argument("--start-with-right", action="store_true", help="Start cycle 1 from the right heart instead of the left")
    return parser.parse_args(argv)


def namespace_to_config(args: argparse.Namespace) -> Configuration:
    if args.num_cycles <= 0:
        raise ValueError("--num-cycles must be > 0")
    if args.min_velocity <= 0.0:
        raise ValueError("--min-velocity must be > 0")
    if args.topology_tolerance <= 0.0:
        raise ValueError("--topology-tolerance must be > 0")
    if args.max_cycle_time <= 0.0:
        raise ValueError("--max-cycle-time must be > 0")
    if args.max_objects <= 0:
        raise ValueError("--max-objects must be > 0")

    return Configuration(
        input_file=args.input,
        output_file=args.output,
        detailed_output_file=args.detailed_output,
        event_log_file=args.event_log,
        number_of_cycles=args.num_cycles,
        random_seed=args.seed,
        topology_tolerance_cm=args.topology_tolerance,
        minimum_velocity_cm_s=args.min_velocity,
        max_cycle_time_s=args.max_cycle_time,
        max_objects_per_cycle=args.max_objects,
        start_with_left_heart=not args.start_with_right,
        write_event_log=not args.no_event_log,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        config = namespace_to_config(args)
        rng = np.random.default_rng(config.random_seed)
        network = VasculatureNetwork.from_csv(config.input_file, config.topology_tolerance_cm)
        print_start_summary(network, config)

        if args.verbose:
            diagnostics = network.validate_reachable_direction()
            if diagnostics:
                print(f"Topology diagnostics: {len(diagnostics)} dead-end object(s) detected")
                for line in diagnostics[:20]:
                    print(f"  {line}")
                if len(diagnostics) > 20:
                    print(f"  ... and {len(diagnostics) - 20} more")
            else:
                print("Topology diagnostics: no normal-vessel dead ends detected")

        recorder = SimulationRecorder()
        simulator = CycleSimulator(network, config, rng, recorder)

        start = time.perf_counter()
        for cycle_number in range(1, config.number_of_cycles + 1):
            record, details = simulator.simulate_cycle(cycle_number)
            recorder.add_cycle(record)
            for detail in details:
                recorder.add_detail(detail)

            if args.strict_validation and record.cycle_status != "COMPLETED":
                print(
                    f"Strict validation stopped simulation after cycle {cycle_number}: "
                    f"status={record.cycle_status}, reason={record.termination_reason}"
                )
                break
        elapsed = time.perf_counter() - start

        recorder.write_csvs(
            config.output_file,
            config.detailed_output_file,
            config.event_log_file,
            write_event_log=config.write_event_log,
        )
        summary = summarize(recorder.cycles, config.number_of_cycles, elapsed, network)
        print_end_summary(summary)

        if args.plot_topology:
            plot_topology(network)

        return 0

    except (InputValidationError, ValueError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
