"""
Analyze German municipal population data from the Destatis 1000A dataset.
Loads the semicolon-separated CSV, filters population records, and computes
basic statistics (mean, std, min, max) for population density per municipality.
"""

import csv
from pathlib import Path


DATASET_PATH = Path("user_files/datasets/1000A-0001_de_flat.csv")


def load_records(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        return list(reader)


def filter_by_variable(records: list[dict], variable_code: str) -> list[dict]:
    return [r for r in records if r.get("value_variable_code") == variable_code]


def parse_values(records: list[dict]) -> list[float]:
    values = []
    for r in records:
        raw = r.get("value", "").replace(",", ".")
        try:
            values.append(float(raw))
        except ValueError:
            pass
    return values


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    variance = sum((v - m) ** 2 for v in values) / len(values)
    return variance ** 0.5


def main():
    records = load_records(DATASET_PATH)

    # Population density (Bevölkerungsdichte) — variable code PRS017
    density_records = filter_by_variable(records, "PRS017")
    density_values = parse_values(density_records)

    print(f"Municipalities with density data: {len(density_values)}")
    print(f"Mean density  : {mean(density_values):.2f} Ew/qkm")
    print(f"Std deviation : {std(density_values):.2f} Ew/qkm")
    print(f"Min density   : {min(density_values):.2f} Ew/qkm")
    print(f"Max density   : {max(density_values):.2f} Ew/qkm")

    return density_values


if __name__ == "__main__":
    main()
