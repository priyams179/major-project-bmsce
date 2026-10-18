/* Flood - affected zones & severity, multi-city (Leaflet) */
(() => {
  "use strict";
  const $ = (x) => document.getElementById(x);

  // ---- map ----------------------------------------------------------- //
  const map = L.map("map", { zoomControl: true, preferCanvas: true })
    .setView([40.700, -73.930], 11);

  // Keyless OpenStreetMap tiles; a CSS filter (style.css) darkens them so the
  // basemap recedes behind the flood overlay. Overlay panes are not filtered.
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution:
      '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  }).addTo(map);

  const svgRenderer = L.svg({ padding: 0.5 }); // zones/routes as SVG so CSS animation works
  const extentLayer = L.layerGroup().addTo(map); // model scenes: the detected flood raster
  const zonesLayer = L.layerGroup().addTo(map);
  const routesLayer = L.layerGroup().addTo(map);
  const marksLayer = L.layerGroup().addTo(map);
  const noiseLayer = L.layerGroup().addTo(map);

  // ---- colour helper ------------------------------------------------ //
  function depthColour(d) {
    const t = Math.max(0, Math.min(1, d / 300)); // 0..3 m
    const stops = [
      [140, 200, 235], [90, 155, 210], [70, 110, 180],
      [235, 150, 85], [205, 65, 40], [110, 8, 8],
    ];
    const seg = t * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(seg));
    const f = seg - i;
    const c = (a, b) => Math.round(a + (b - a) * f);
    const [r, g, b] = [0, 1, 2].map((j) => c(stops[i][j], stops[i + 1][j]));
    return `rgb(${r},${g},${b})`;
  }

  // ---- controls --------------------------------------------------- //
  const ctl = {
    city: $("city"), stage_m: $("stage_m"), eps_m: $("eps_m"),
    min_samples: $("min_samples"), depth_weight: $("depth_weight"), seed: $("seed"),
  };

  let cityMeta = null;       // description of the current city
  let cityChanged = true;    // recenter the map on the next draw

  function syncLabels() {
    $("stage-v").textContent = (+ctl.stage_m.value).toFixed(1) + " m";
    $("eps_m-v").textContent = ctl.eps_m.value + " m";
    $("min_samples-v").textContent = ctl.min_samples.value;
    $("depth_weight-v").textContent = (+ctl.depth_weight.value).toFixed(2);
    $("seed-v").textContent = ctl.seed.value;
    if (cityMeta && cityMeta.references) {
      const s = +ctl.stage_m.value;
      let best = cityMeta.references[0];
      for (const r of cityMeta.references)
        if (Math.abs(r.stage_m - s) < Math.abs(best.stage_m - s)) best = r;
      $("surge-note").textContent = `≈ ${best.label}`;
    }
  }

  function applyCityMeta(m) {
    cityMeta = m;
    $("title").textContent = `${m.name} flood — affected zones`;
    $("subtitle").textContent = m.subtitle;
    // A model scene is a measured flood: no stage to dial, no scenario to reseed.
    const model = m.source === "model";
    $("stage-card").hidden = model;
    $("seed-control").hidden = model;
    $("model-card").hidden = !model;
    $("t-extent-label").hidden = !model;
    $("no-zones").innerHTML = model
      ? "No zone meets the clustering threshold &mdash; widen the DBSCAN radius or lower min marks."
      : "No zone meets the clustering threshold yet &mdash; raise the stage.";
    if (model) {
      const i = m.model;
      $("model-info").innerHTML =
        `<span><b>${i.flood_km2}</b> km&sup2; flooded</span>` +
        (i.perm_water_km2 ? `<span><b>${i.perm_water_km2}</b> km&sup2; permanent water</span>` : "") +
        `<p class="note">Marks every ${i.stride_m} m; zones use marks within ` +
        `${i.exposure_m} m of a road. Depth: ${i.depth_source}. Safe points: ${i.safe_source}.</p>`;
    } else {
      $("stage-head").textContent = m.driver[0].toUpperCase() + m.driver.slice(1);
      $("stage-label").textContent = m.stage_label;
      const sl = ctl.stage_m;
      sl.min = m.stage_min;
      sl.max = m.stage_max;
      if (+sl.value < m.stage_min || +sl.value > m.stage_max) sl.value = m.stage_default;
    }
    if (cityChanged) {
      extentLayer.clearLayers();
      if (m.overlay) L.imageOverlay(m.overlay.url, m.overlay.bounds).addTo(extentLayer);
      map.setView(m.center, m.zoom);
      cityChanged = false;
    }
  }

  const layerByKey = { zones: zonesLayer, routes: routesLayer,
                       marks: marksLayer, noise: noiseLayer, extent: extentLayer };
  for (const key of Object.keys(layerByKey)) {
    $("t-" + key).addEventListener("change", (e) => {
      if (e.target.checked) layerByKey[key].addTo(map);
      else layerByKey[key].remove();
    });
  }

  // ---- render --------------------------------------------------- //
  const zoneRefs = new Map();
  const routeRefs = new Map();

  function draw(data) {
    applyCityMeta(data.city);
    zonesLayer.clearLayers();
    routesLayer.clearLayers();
    marksLayer.clearLayers();
    noiseLayer.clearLayers();
    zoneRefs.clear();
    routeRefs.clear();

    for (const f of data.marks.features) {
      const [lon, lat] = f.geometry.coordinates;
      const p = f.properties;
      const target = p.noise ? noiseLayer : marksLayer;
      L.circleMarker([lat, lon], {
        radius: p.noise ? 2.5 : 3 + Math.sqrt(p.depth_cm) * 0.22,
        color: p.noise ? "#8b95a1" : "#00000055",
        weight: p.noise ? 0 : 0.5,
        fillColor: p.noise ? "#8b95a1" : depthColour(p.depth_cm),
        fillOpacity: p.noise ? 0.5 : 0.9,
      }).bindPopup(
        `<b>Flood mark</b><br>depth <b>${p.depth_cm} cm</b>` +
        (p.ground_elev_m != null ? `<br>ground elev ${p.ground_elev_m} m` : "") +
        `<br>${p.noise ? "unclustered (noise)" : p.area + " &middot; zone " + p.zone}`
      ).addTo(target);
    }

    const feats = [...data.zones.features].sort(
      (a, b) => (b.properties.area_km2 || 0) - (a.properties.area_km2 || 0)
    );
    for (const f of feats) {
      const p = f.properties;
      const latlngs = f.geometry.coordinates[0].map((c) => [c[1], c[0]]);
      const poly = L.polygon(latlngs, {
        renderer: svgRenderer,
        color: p.fill, weight: 2.5, opacity: 0.95, dashArray: "6 4",
        fillColor: p.fill,
        fillOpacity: p.tier === "Critical" ? 0.42 : 0.28,
        className: "zone-" + p.tier,
      }).bindPopup(
        `<b>${p.name}</b><br>` +
        `<span style="color:${p.fill};font-weight:700">${p.tier}</span> ` +
        `&middot; score ${(p.severity_score ?? 0).toFixed(3)}<br>` +
        `${p.n_marks} marks &middot; ${(p.area_km2 ?? 0).toFixed(3)} km&sup2;<br>` +
        (p.mean_depth != null
          ? `depth mean ${p.mean_depth} cm &middot; max ${p.max_depth} cm<br>` : "") +
        `${(p.localities || []).join(", ")}`
      );
      poly.addTo(zonesLayer);
      zoneRefs.set(p.zone, poly);
    }

    // evacuation routes: zone -> nearest safe high ground
    for (const f of (data.routes ? data.routes.features : [])) {
      const p = f.properties;
      const latlngs = f.geometry.coordinates.map((c) => [c[1], c[0]]);
      const line = L.polyline(latlngs, {
        renderer: svgRenderer,
        color: p.color, weight: 3.5, opacity: 0.95,
        dashArray: "2 9", lineCap: "round",
        className: "evac-route",
      }).bindPopup(
        `<b>Evacuation route — zone ${p.zone}</b><br>` +
        `<span style="color:${p.color};font-weight:700">${p.tier}</span> zone ` +
        `&rarr; <b>${p.safe_name}</b> (${p.bearing})<br>` +
        `${p.distance_km} km by road &middot; deepest water on route ${p.max_depth_cm} cm`
      );
      line.addTo(routesLayer);
      routeRefs.set(p.zone, line);
    }

    // safe points (static per city)
    for (const sp of (data.city.safe_points || [])) {
      L.marker([sp.lat, sp.lon], {
        icon: L.divIcon({ className: "safe-pin", iconSize: [18, 18],
                          iconAnchor: [9, 9], html: "<span></span>" }),
      }).bindTooltip(`Safe: ${sp.name}`, { direction: "top", offset: [0, -6] })
        .addTo(routesLayer);
    }

    renderPanel(data);
  }

  function renderPanel(data) {
    const s = data.stats;
    $("stats").innerHTML =
      `<span><b>${s.n_wet}</b> ${data.city.source === "model"
        ? "flooded cells near roads" : "marks under water"}</span>` +
      `<span><b>${s.n_zones}</b> affected zones</span>` +
      `<span><b>${Math.round(s.max_depth_cm)}</b> cm deepest</span>` +
      (s.n_cutoff
        ? `<span class="warn"><b>${s.n_cutoff}</b> cut off</span>`
        : `<span><b>${s.n_noise}</b> isolated</span>`);
    $("zone-count").textContent = data.summary.length ? `(${data.summary.length})` : "";

    const tbody = $("zone-rows");
    tbody.innerHTML = "";
    $("no-zones").hidden = data.summary.length > 0;
    for (const z of data.summary) {
      const tr = document.createElement("tr");
      const nm = z.name.length > 16 ? z.name.slice(0, 15) + "…" : z.name;
      const rt = z.route;
      const evac = !rt ? "-"
        : !rt.reachable ? `<span class="warn">cut off</span>`
        : `<span title="to ${rt.safe_name}">${rt.bearing}&nbsp;${rt.distance_km}&nbsp;km</span>`;
      tr.innerHTML =
        `<td><span class="chip" style="background:${data.tier_colour[z.tier]}">${z.tier}</span></td>` +
        `<td title="${z.name}">${nm}</td>` +
        `<td>${(z.severity_score ?? 0).toFixed(3)}</td>` +
        `<td>${z.n_marks}</td>` +
        `<td>${z.max_depth ?? "-"}</td>` +
        `<td class="evac">${evac}</td>`;
      const hi = (on) => {
        const poly = zoneRefs.get(z.zone);
        if (poly) poly.setStyle(on
          ? { weight: 3, opacity: 1, fillOpacity: 0.6 }
          : { weight: 2.5, opacity: 0.95,
              fillOpacity: z.tier === "Critical" ? 0.42 : 0.28 });
        const line = routeRefs.get(z.zone);
        if (line) line.setStyle({ weight: on ? 6 : 3.5, opacity: on ? 1 : 0.95 });
      };
      tr.addEventListener("mouseenter", () => hi(true));
      tr.addEventListener("mouseleave", () => hi(false));
      tr.addEventListener("click", () => {
        const line = routeRefs.get(z.zone);
        const poly = zoneRefs.get(z.zone);
        if (line) { map.fitBounds(line.getBounds().pad(0.15)); line.openPopup(); }
        else if (poly) { map.fitBounds(poly.getBounds().pad(0.4)); poly.openPopup(); }
      });
      tbody.appendChild(tr);
    }
  }

  // ---- fetch loop -------------------------------------------- //
  let timer = null;
  let inflight = null;
  async function refresh() {
    syncLabels();
    const q = new URLSearchParams({
      city: ctl.city.value, stage_m: ctl.stage_m.value, eps_m: ctl.eps_m.value,
      min_samples: ctl.min_samples.value, depth_weight: ctl.depth_weight.value,
      seed: ctl.seed.value,
    });
    if (inflight) inflight.abort();
    inflight = new AbortController();
    $("map").classList.add("loading");
    try {
      const data = await (await fetch("/api/flood?" + q, { signal: inflight.signal })).json();
      draw(data);
      syncLabels();
    } catch (e) {
      if (e.name !== "AbortError") throw e;
      return;
    } finally {
      if (!inflight.signal.aborted) $("map").classList.remove("loading");
    }
  }
  const debounced = () => { clearTimeout(timer); timer = setTimeout(refresh, 200); };

  ctl.city.addEventListener("change", () => { cityChanged = true; refresh(); });
  for (const k of ["stage_m", "eps_m", "min_samples", "depth_weight", "seed"]) {
    ctl[k].addEventListener("input", () => { syncLabels(); debounced(); });
  }

  refresh();
})();
