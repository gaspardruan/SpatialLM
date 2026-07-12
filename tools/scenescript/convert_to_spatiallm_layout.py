import argparse
import sys
from pathlib import Path


SCENESCRIPT_ROOT = Path(__file__).resolve().parents[2] / "baselines" / "SceneScript"
sys.path.insert(0, str(SCENESCRIPT_ROOT))

from src.data.geometries import BboxEntity, DoorEntity, WallEntity, WindowEntity  # noqa: E402
from src.data.language_sequence import LanguageSequence  # noqa: E402


def fmt_float(value):
    return f"{float(value):.6f}"


def convert_language_to_spatiallm(language_sequence, wall_thickness):
    lines = []
    wall_id_map = {}

    for entity in language_sequence.entities:
        if not isinstance(entity, WallEntity):
            continue

        src_id = int(entity.params["id"])
        dst_id = len(wall_id_map)
        wall_id_map[src_id] = dst_id
        thickness = max(float(entity.params.get("thickness", 0.0)), wall_thickness)
        lines.append(
            "wall_{id}=Wall({ax},{ay},{az},{bx},{by},{bz},{height},{thickness})".format(
                id=dst_id,
                ax=fmt_float(entity.params["a_x"]),
                ay=fmt_float(entity.params["a_y"]),
                az=fmt_float(entity.params["a_z"]),
                bx=fmt_float(entity.params["b_x"]),
                by=fmt_float(entity.params["b_y"]),
                bz=fmt_float(entity.params["b_z"]),
                height=fmt_float(entity.params["height"]),
                thickness=fmt_float(thickness),
            )
        )

    door_id = 0
    window_id = 0
    for entity in language_sequence.entities:
        if not isinstance(entity, (DoorEntity, WindowEntity)):
            continue

        src_wall_id = int(entity.params.get("wall0_id", entity.params.get("wall_id", -1)))
        wall_id = wall_id_map.get(src_wall_id)
        if wall_id is None:
            continue

        if isinstance(entity, DoorEntity) and not isinstance(entity, WindowEntity):
            lines.append(
                "door_{id}=Door(wall_{wall_id},{x},{y},{z},{width},{height})".format(
                    id=door_id,
                    wall_id=wall_id,
                    x=fmt_float(entity.params["position_x"]),
                    y=fmt_float(entity.params["position_y"]),
                    z=fmt_float(entity.params["position_z"]),
                    width=fmt_float(entity.params["width"]),
                    height=fmt_float(entity.params["height"]),
                )
            )
            door_id += 1
        else:
            lines.append(
                "window_{id}=Window(wall_{wall_id},{x},{y},{z},{width},{height})".format(
                    id=window_id,
                    wall_id=wall_id,
                    x=fmt_float(entity.params["position_x"]),
                    y=fmt_float(entity.params["position_y"]),
                    z=fmt_float(entity.params["position_z"]),
                    width=fmt_float(entity.params["width"]),
                    height=fmt_float(entity.params["height"]),
                )
            )
            window_id += 1

    bbox_id = 0
    for entity in language_sequence.entities:
        if not isinstance(entity, BboxEntity):
            continue
        lines.append(
            "bbox_{id}=Bbox({class_name},{x},{y},{z},{angle},{sx},{sy},{sz})".format(
                id=bbox_id,
                class_name=str(entity.params["class"]).replace("_", " "),
                x=fmt_float(entity.params["position_x"]),
                y=fmt_float(entity.params["position_y"]),
                z=fmt_float(entity.params["position_z"]),
                angle=fmt_float(entity.params["angle_z"]),
                sx=fmt_float(entity.params["scale_x"]),
                sy=fmt_float(entity.params["scale_y"]),
                sz=fmt_float(entity.params["scale_z"]),
            )
        )
        bbox_id += 1

    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        "Convert SceneScript language output to SpatialLM layout txt."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wall_thickness", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()
    language_sequence = LanguageSequence.load_from_file(args.input)
    layout = convert_language_to_spatiallm(language_sequence, args.wall_thickness)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(layout)
    print(output)


if __name__ == "__main__":
    main()
