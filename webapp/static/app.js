// ================================================================
// Vegetation Monitor — стабильный проверочный фронт
// Без фейковых полей датасета на карте.
// ================================================================

const $ = (id) => document.getElementById(id);

function exists(id) {
  return $(id) !== null;
}

async function api(path, opts) {
  const res = await fetch(path, opts);

  if (!res.ok) {
    let msg = "HTTP " + res.status;
    try {
      const j = await res.json();
      msg = j.detail || msg;
    } catch (e) {}
    throw new Error(msg);
  }

  return res.json();
}

function setStatus(text, isErr = false) {
  if (!exists("status")) return;

  const el = $("status");
  el.textContent = text;
  el.classList.toggle("err", !!isErr);
}

const handle = (fn) => async (...args) => {
  try {
    await fn(...args);
  } catch (e) {
    console.error(e);
    setStatus("Ошибка: " + e.message, true);
  }
};

// ----------------------------------------------------------------
// Health
// ----------------------------------------------------------------
handle(async () => {
  if (!exists("health")) return;

  const h = await api("/api/health");

  $("health").textContent =
    "API: " + h.status +
    " | train: " + h.train_loaded +
    " | model: " + h.model_loaded;
})();

// ----------------------------------------------------------------
// Список полигонов датасета
// ----------------------------------------------------------------
handle(async () => {
  if (!exists("polygonSelect")) return;

  const list = await api("/api/polygons");
  const sel = $("polygonSelect");

  sel.innerHTML = "";

  if (!list.length) {
    sel.innerHTML = '<option value="">нет полигонов</option>';
    return;
  }

  list.slice(0, 500).forEach((p) => {
    const option = document.createElement("option");
    option.value = p.id;
    option.textContent = p.id + " (" + p.n_points + " точек)";
    sel.appendChild(option);
  });
})();

if (exists("btnLoadPolygon")) {
  $("btnLoadPolygon").onclick = handle(async () => {
    const id = $("polygonSelect").value;
    if (!id) return;

    setStatus("Загрузка ряда " + id + " из датасета…");

    const data = await api("/api/timeseries/" + encodeURIComponent(id));
    renderAll(data);
  });
}

// ----------------------------------------------------------------
// Режимы
// ----------------------------------------------------------------
function setMode(mode) {
  if (exists("secDataset")) {
    $("secDataset").style.display = mode === "dataset" ? "" : "none";
  }

  if (exists("secDraw")) {
    $("secDraw").style.display = mode === "draw" ? "" : "none";
  }

  if (mode === "draw") {
    setStatus("Режим 2: нарисуйте фигуру на карте — будет спутник + погода");
  }
}

document.querySelectorAll('input[name="mode"]').forEach((radio) => {
  radio.addEventListener("change", (e) => {
    setMode(e.target.value);
  });
});

setMode("dataset");

// ----------------------------------------------------------------
// Карта
// ----------------------------------------------------------------
const map = L.map("map").setView([45.0, 39.0], 6);

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: "© OpenStreetMap",
  maxZoom: 19,
}).addTo(map);

const drawnItems = new L.FeatureGroup().addTo(map);
const fieldsLayer = L.layerGroup().addTo(map);

let osmGeo = null;

map.addControl(new L.Control.Draw({
  position: "topright",
  draw: {
    polygon: true,
    rectangle: true,
    polyline: false,
    circle: false,
    circlemarker: false,
    marker: false,
  },
  edit: {
    featureGroup: drawnItems,
  },
}));

map.on(L.Draw.Event.CREATED, handle(async (e) => {
  drawnItems.clearLayers();
  drawnItems.addLayer(e.layer);

  const radio = document.querySelector('input[name="mode"][value="draw"]');
  if (radio) {
    radio.checked = true;
    setMode("draw");
  }

  await analyzeGeometry(e.layer.toGeoJSON().geometry);
}));

// ----------------------------------------------------------------
// Поиск региона и загрузка OSM-полей
// ----------------------------------------------------------------
if (exists("btnSearchRegion")) {
  $("btnSearchRegion").onclick = handle(async () => {
    const q = $("regionInput").value.trim();

    if (!q) {
      setStatus("Введите название региона", true);
      return;
    }

    setStatus("Поиск региона…");

    const regions = await api("/api/regions/search?q=" + encodeURIComponent(q));

    if (!regions.length) {
      setStatus("Регион не найден", true);
      return;
    }

    const r = regions[0];

    if (exists("regionInfo")) {
      $("regionInfo").textContent = r.name || "";
    }

    if (r.bbox) {
      // Nominatim: [south, north, west, east]
      const [south, north, west, east] = r.bbox;
      map.fitBounds([[south, west], [north, east]]);
    } else {
      map.setView([r.lat, r.lon], 10);
    }

    await loadFields(map.getBounds());
  });
}

async function loadFields(bounds) {
  setStatus("Загрузка сельхозконтуров из OSM…");

  const bbox = [
    bounds.getSouth(),
    bounds.getWest(),
    bounds.getNorth(),
    bounds.getEast(),
  ].join(",");

  const gj = await api("/api/fields?bbox=" + bbox + "&limit=20");

  fieldsLayer.clearLayers();

  osmGeo = L.geoJSON(gj, {
    style: {
      color: "#795548",
      weight: 1,
      fillColor: "#8bc34a",
      fillOpacity: 0.22,
    },
    onEachFeature: (f, layer) => {
      const p = f.properties || {};
      const name = p.name || p.landuse || p.crop || "сельхозконтур";

      layer.bindTooltip(name);

      layer.on("click", handle(async () => {
        const radio = document.querySelector('input[name="mode"][value="draw"]');
        if (radio) {
          radio.checked = true;
          setMode("draw");
        }

        await analyzeGeometry(f.geometry);
      }));
    },
  });

  fieldsLayer.addLayer(osmGeo);

  setStatus(
    "Контуров загружено: " + gj.features.length +
    ". Кликните по полю для анализа."
  );
}

// ----------------------------------------------------------------
// Анализ нарисованного или OSM-полигона
// ----------------------------------------------------------------
async function analyzeGeometry(geometry) {
  setStatus("Анализ: спутниковые данные + погода по фигуре…");

  const start = exists("startDate") ? $("startDate").value : "2024-04-01";
  const end = exists("endDate") ? $("endDate").value : "2024-09-30";

  const data = await api("/api/analyze", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      geojson: geometry,
      start,
      end,
    }),
  });

  renderAll(data);
}

// ----------------------------------------------------------------
// Инференс
// ----------------------------------------------------------------
if (exists("btnInference")) {
  $("btnInference").onclick = handle(async () => {
    setStatus("Формирование submission.csv…");

    const result = await api("/api/inference", {
      method: "POST",
    });

    if (exists("inferenceResult")) {
      $("inferenceResult").textContent = "Готово: " + result.submission;
    }

    if (exists("downloadLink")) {
      $("downloadLink").hidden = false;
    }

    setStatus("submission.csv создан");
  });
}

// ----------------------------------------------------------------
// Рендер всего результата
// ----------------------------------------------------------------
function renderAll(d) {
  let msg =
    "Полигон " + d.polygon_id +
    " | источник: " + d.source +
    " | аномалий: " + (d.periods ? d.periods.length : 0);

  if (d.centroid) {
    msg += " | центр: " + d.centroid[0] + ", " + d.centroid[1];
  }

  if (d.weather) {
    msg += " | погода: " + d.weather.source;
  }

  setStatus(msg);

  if (exists("satInfo")) {
    if (d.satellite) {
      $("satInfo").textContent =
        "🛰 " + d.satellite.sensor +
        " | наблюдений: " + d.satellite.n_obs +
        " | " + d.satellite.source +
        (d.satellite.note ? " | " + d.satellite.note : "");
    } else {
      $("satInfo").textContent =
        "🛰 спутниковые источники недоступны — демо-режим";
    }
  }

  renderChart(d);
  renderZ(d);
  renderWeather(d);
  renderPeriods(d.periods);
}

// ----------------------------------------------------------------
// NDVI-график
// ----------------------------------------------------------------
function renderChart(d) {
  const shapes = (d.periods || []).map((p) => ({
    type: "rect",
    xref: "x",
    yref: "paper",
    x0: p.start,
    x1: p.end,
    y0: 0,
    y1: 1,
    fillcolor:
      p.severity === "critical"
        ? "rgba(211,47,47,0.15)"
        : "rgba(255,152,0,0.15)",
    line: { width: 0 },
  }));

  const traces = [
    {
      name: "Норма ±1.96σ",
      x: [...d.dates, ...[...d.dates].reverse()],
      y: [...d.clim_upper, ...[...d.clim_lower].reverse()],
      fill: "toself",
      fillcolor: "rgba(46,125,50,0.08)",
      line: { color: "transparent" },
      hoverinfo: "skip",
    },
    {
      name: "Климатическая норма",
      x: d.dates,
      y: d.clim_mean,
      line: { color: "#2e7d32", dash: "dot", width: 1 },
    },
    {
      name: "Восстановленный ряд",
      x: d.dates,
      y: d.filled,
      line: { color: "#1565c0", width: 2 },
    },
    {
      name: "Наблюдения",
      x: d.dates.filter((_, i) => d.raw[i] !== null),
      y: d.raw.filter((v) => v !== null),
      mode: "markers",
      marker: { color: "#333", size: 5 },
    },
    {
      name: "Пропуски",
      x: d.dates.filter((_, i) => d.is_gap[i]),
      y: d.filled.filter((_, i) => d.is_gap[i]),
      mode: "markers",
      marker: { color: "#d32f2f", size: 8, symbol: "x" },
    },
  ];

  Plotly.newPlot("chart", traces, {
    title: "NDVI — " + d.polygon_id,
    shapes,
    margin: { t: 40, l: 45, r: 10, b: 30 },
    legend: { orientation: "h" },
  }, {
    responsive: true,
  });
}

// ----------------------------------------------------------------
// Z-score
// ----------------------------------------------------------------
function renderZ(d) {
  if (!d.dates || !d.dates.length) return;

  const last = d.dates[d.dates.length - 1];

  Plotly.newPlot("zchart", [
    {
      x: d.dates,
      y: d.z,
      type: "scatter",
      mode: "lines",
      line: { color: "#455a64", width: 1.5 },
    },
  ], {
    title: "Z-score",
    margin: { t: 40, l: 45, r: 10, b: 30 },
    shapes: [
      {
        type: "line",
        x0: d.dates[0],
        x1: last,
        y0: -1,
        y1: -1,
        line: { color: "#ff9800", dash: "dash" },
      },
      {
        type: "line",
        x0: d.dates[0],
        x1: last,
        y0: -2,
        y1: -2,
        line: { color: "#d32f2f", dash: "dash" },
      },
    ],
  }, {
    responsive: true,
  });
}

// ----------------------------------------------------------------
// Погода
// ----------------------------------------------------------------
function renderWeather(d) {
  if (!exists("wchart")) return;

  const el = $("wchart");

  if (
    !d.weather ||
    ((!d.weather.temperature || !d.weather.temperature.length) &&
      (!d.weather.precipitation || !d.weather.precipitation.length))
  ) {
    Plotly.purge(el);
    el.innerHTML = "";
    return;
  }

  const traces = [];

  if (d.weather.precipitation && d.weather.precipitation.length) {
    traces.push({
      x: d.dates,
      y: d.weather.precipitation,
      type: "bar",
      name: "Осадки, мм",
      marker: { color: "rgba(21,101,192,.45)" },
    });
  }

  if (d.weather.temperature && d.weather.temperature.length) {
    traces.push({
      x: d.dates,
      y: d.weather.temperature,
      type: "scatter",
      mode: "lines",
      name: "T, °C",
      yaxis: "y2",
      line: { color: "#e65100", width: 1.5 },
    });
  }

  Plotly.newPlot(el, traces, {
    title: "Погода — " + (d.weather.source || ""),
    margin: { t: 40, l: 45, r: 45, b: 30 },
    yaxis: { title: "мм" },
    yaxis2: {
      title: "°C",
      overlaying: "y",
      side: "right",
    },
    legend: { orientation: "h" },
  }, {
    responsive: true,
  });
}

// ----------------------------------------------------------------
// Аномальные периоды
// ----------------------------------------------------------------
function renderPeriods(periods) {
  if (!exists("periods")) return;

  const box = $("periods");
  box.innerHTML = "";

  if (!periods || !periods.length) {
    box.innerHTML = "<p class='hint'>Аномалий не найдено ✅</p>";
    return;
  }

  periods.forEach((p) => {
    const div = document.createElement("div");
    div.className = "period " + p.severity;

    div.innerHTML =
      "<b>" + p.label + "</b><br>" +
      p.start + " → " + p.end + " (" + p.days + " дн.)<br>" +
      "Z min: " + p.z_min + ", NDVI min: " + p.ndvi_min + "<br>" +
      "<i>" + p.interpretation + "</i>";

    box.appendChild(div);
  });
}