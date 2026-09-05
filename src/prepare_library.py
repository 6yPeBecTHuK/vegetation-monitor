"""
Одноразовая подготовка географической библиотеки полей.
Запускается ОДИН РАЗ на машине с интернетом:

    python -m src.prepare_library

Результат: data/fields_library.geojson — готовые OSM-контуры демо-регионов.
После этого сервер отдаёт их мгновенно и БЕЗ обращения в OSM.
"""
import json

from src.config import DATA_DIR
from src.data import osm_client

REGIONS = {
    "Кубань (Каневской район)": (46.0, 38.6, 46.6, 39.4),
    "Дон (Ростовский район)": (47.0, 39.4, 47.6, 40.2),
    "Ставрополье": (44.9, 41.6, 45.5, 42.4),
}


def main():
    features = []

    for name, bbox in REGIONS.items():
        try:
            gj = osm_client.get_farmland_polygons(bbox, limit=100)
            for f in gj["features"]:
                f["properties"]["region"] = name
            features.extend(gj["features"])
            print(f"{name}: {len(gj['features'])} полей")
        except Exception as e:
            print(f"{name}: ошибка — {e}")

    out = DATA_DIR / "fields_library.geojson"
    out.write_text(
        json.dumps({"type": "FeatureCollection", "features": features},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Сохранено: {out} ({len(features)} полей)")


if __name__ == "__main__":
    main()