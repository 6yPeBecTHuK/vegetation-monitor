"""
Стартовый набор заготовленных полей -> data/fields.geojson
Запуск: python make_demo_fields.py
"""
import json

import numpy as np

from src.config import DATA_DIR

REGIONS = [
    ("Дон (Ростовский район)", 47.35, 39.85),
    ("Кубань (Каневской район)", 46.10, 38.90),
    ("Ставрополье (Шпаковский округ)", 45.20, 41.90),
]

CROPS = ["озимая пшеница", "кукуруза", "подсолнечник", "ячмень", "пар"]


def main(seed: int = 42):
    rng = np.random.default_rng(seed)
    features = []

    for region_name, lat0, lon0 in REGIONS:
        for i in range(8):
            lat = lat0 + rng.uniform(-0.12, 0.12)
            lon = lon0 + rng.uniform(-0.18, 0.18)
            w = rng.uniform(0.015, 0.04)
            h = w * rng.uniform(0.6, 1.4)

            coords = [
                [lon, lat],
                [lon + w, lat],
                [lon + w, lat + h],
                [lon, lat + h],
                [lon, lat],
            ]
            coords = [[round(c[0], 5), round(c[1], 5)] for c in coords]

            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [coords]},
                    "properties": {
                        "name": f"{region_name} — поле {i + 1}",
                        "crop": str(rng.choice(CROPS)),
                        "source": "prepared",
                    },
                }
            )

    geojson = {"type": "FeatureCollection", "features": features}
    out = DATA_DIR / "fields.geojson"
    out.write_text(json.dumps(geojson, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Сохранено {len(features)} полей -> {out}")


if __name__ == "__main__":
    main()