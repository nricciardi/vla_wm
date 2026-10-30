"""Scene descriptions, JSON loading, and validation for the tabletop lab.

This module uses only the Python standard library. Scene construction and the
simulation loop are in 06_deformable_tabletop.py.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).with_name("tabletop_scene.json")


@dataclass
class TableSpec:
    """Static tabletop dimensions, world position, and contact material."""

    size: tuple[float, float, float] = (1.4, 0.9, 0.10)
    center: tuple[float, float, float] = (0.0, 0.0, 0.75)

    static_friction: float = 0.7
    dynamic_friction: float = 0.5
    color: tuple[float, float, float] = (0.45, 0.30, 0.18)


@dataclass
class ObjectSpec:
    """One rigid body or volume deformable, with SI units and XYZW rotation."""

    name: str
    shape: str = "box"  # box, sphere, cylinder, or usd
    deformable: bool = True

    # Primitive geometry. Only the fields used by the selected shape apply.
    size: tuple[float, float, float] = (0.16, 0.16, 0.16)
    radius: float = 0.08
    height: float = 0.16  # Cylinder height along its local Z axis.

    # External USD geometry. Scale applies only to imported assets.
    usd_path: str | None = None
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Position is relative to the tabletop surface center; velocity is in world coordinates.
    position: tuple[float, float, float] = (0.0, 0.0, 0.30)
    rotation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    mass: float = 0.5  # Total body mass in kg.
    static_friction: float = 0.7
    dynamic_friction: float = 0.5

    # Elasticity and meshing parameters apply only to deformable bodies.
    youngs_modulus: float = 1.0e5  # Pa
    poissons_ratio: float = 0.4
    elasticity_damping: float = 0.005  # 1/s
    edge_refinement: float = 4.0  # Procedural deformable meshes only.

    color: tuple[float, float, float] = (0.15, 0.55, 0.85)


def load_scene(path: Path) -> tuple[TableSpec, list[ObjectSpec]]:
    """Load scene descriptions and resolve local USD paths relative to the JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict) or set(data) - {"table", "objects"}:
        raise ValueError("The scene may contain only 'table' and 'objects'")

    if not isinstance(data.get("table", {}), dict):
        raise ValueError("table must be a JSON object")

    if not isinstance(data.get("objects"), list) or not data["objects"]:
        raise ValueError("objects must contain at least one object")

    try:
        table = TableSpec(**data.get("table", {}))
        objects = [ObjectSpec(**item) for item in data["objects"]]
    except TypeError as exc:
        raise ValueError(f"Invalid configuration fields: {exc}") from exc

    validate_table(table)

    names = set()

    for obj in objects:
        validate_object(obj)

        if obj.name in names:
            raise ValueError(f"Duplicate object name: {obj.name}")

        names.add(obj.name)

        if obj.shape == "usd":
            obj.usd_path = resolve_asset_path(obj.usd_path, path)

    return table, objects


def validate_table(table: TableSpec) -> None:
    validate_vector(table.size, 3, "table.size", positive=True)
    validate_vector(table.center, 3, "table.center")
    validate_vector(table.color, 3, "table.color", color=True)
    validate_friction(table, "table")

    if table.center[2] - table.size[2] / 2 < 0:
        raise ValueError("The tabletop must be above the ground plane at z=0")


def validate_object(obj: ObjectSpec) -> None:
    if not isinstance(obj.name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", obj.name):
        raise ValueError("Each name must be a simple USD identifier without slashes or spaces")

    if obj.shape not in ("box", "sphere", "cylinder", "usd"):
        raise ValueError(f"{obj.name}: shape must be box, sphere, cylinder, or usd")

    if not isinstance(obj.deformable, bool):
        raise ValueError(f"{obj.name}.deformable: expected true or false")

    validate_number(obj.mass, f"{obj.name}.mass", minimum=0, strict=True)
    validate_vector(obj.position, 3, f"{obj.name}.position")
    validate_vector(obj.rotation_xyzw, 4, f"{obj.name}.rotation_xyzw")
    validate_vector(obj.linear_velocity, 3, f"{obj.name}.linear_velocity")
    validate_vector(obj.color, 3, f"{obj.name}.color", color=True)
    validate_friction(obj, obj.name)

    rotation_norm_squared = sum(component**2 for component in obj.rotation_xyzw)

    if not math.isclose(rotation_norm_squared, 1.0, abs_tol=1e-4):
        raise ValueError(f"{obj.name}: rotation_xyzw must be a unit quaternion")

    if obj.shape == "box":
        validate_vector(obj.size, 3, f"{obj.name}.size", positive=True)

    if obj.shape in ("sphere", "cylinder"):
        validate_number(obj.radius, f"{obj.name}.radius", minimum=0, strict=True)

    if obj.shape == "cylinder":
        validate_number(obj.height, f"{obj.name}.height", minimum=0, strict=True)

    if obj.shape == "usd":
        validate_vector(obj.scale, 3, f"{obj.name}.scale", positive=True)

        if not isinstance(obj.usd_path, str) or not obj.usd_path.strip():
            raise ValueError(f"{obj.name}: shape='usd' requires usd_path")

    elif obj.usd_path is not None:
        raise ValueError(f"{obj.name}: usd_path requires shape='usd'")

    if obj.deformable:
        validate_number(obj.youngs_modulus, f"{obj.name}.youngs_modulus", minimum=0, strict=True)
        validate_number(obj.poissons_ratio, f"{obj.name}.poissons_ratio")
        validate_number(obj.elasticity_damping, f"{obj.name}.elasticity_damping", minimum=0)
        validate_number(obj.edge_refinement, f"{obj.name}.edge_refinement", minimum=1)

        if not -1 < obj.poissons_ratio < 0.5:
            raise ValueError(f"{obj.name}: poissons_ratio must be in (-1, 0.5)")


def resolve_asset_path(usd_path: str, config_path: Path) -> str:
    """Leave remote URIs unchanged; check and resolve local files."""
    if "://" in usd_path:
        return usd_path

    asset_path = Path(usd_path).expanduser()

    if not asset_path.is_absolute():
        asset_path = config_path.resolve().parent / asset_path

    if not asset_path.is_file():
        raise ValueError(f"Asset not found: {asset_path}")

    return str(asset_path.resolve())


def validate_number(value, label: str, *, minimum: float | None = None, strict: bool = False) -> None:
    """Reject booleans and nonfinite values before the GPU runtime starts."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label}: expected a finite number")

    if minimum is not None and (value < minimum or (strict and value == minimum)):
        relation = ">" if strict else ">="
        raise ValueError(f"{label}: must be {relation} {minimum}")


def validate_vector(value, size: int, label: str, *, positive: bool = False, color: bool = False) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{label}: expected {size} components")

    for component in value:
        minimum = 0 if positive or color else None
        validate_number(component, label, minimum=minimum, strict=positive)

        if color and component > 1:
            raise ValueError(f"{label}: color channels must be in [0, 1]")


def validate_friction(spec, label: str) -> None:
    validate_number(spec.static_friction, f"{label}.static_friction", minimum=0)
    validate_number(spec.dynamic_friction, f"{label}.dynamic_friction", minimum=0)

    if spec.dynamic_friction > spec.static_friction:
        raise ValueError(f"{label}: use dynamic_friction <= static_friction")


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    """Keep Isaac Lab imports optional for configuration checks and offline help."""
    parser = argparse.ArgumentParser(description="Create a tabletop with rigid and deformable objects")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Scene JSON file")
    parser.add_argument("--validate-only", action="store_true", help="Validate the JSON without Isaac Lab")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--dt", type=float, default=1.0 / 120, help="Physics time step [s]")
    parser.add_argument("--log-every", type=int, default=120)

    if "--validate-only" not in argv:
        try:
            from isaaclab.app import add_launcher_args
        except ModuleNotFoundError:
            if "--help" not in argv and "-h" not in argv:
                parser.error("Isaac Lab is unavailable: run from its checkout with uv run, or use --validate-only")
        else:
            add_launcher_args(parser)

    args = parser.parse_args(argv)

    try:
        validate_number(args.dt, "dt", minimum=0, strict=True)

        if args.steps < 1 or args.log_every < 1:
            raise ValueError("steps and log-every must be positive")

    except ValueError as exc:
        parser.error(str(exc))

    return args
