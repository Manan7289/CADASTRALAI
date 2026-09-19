// Records tab: import an existing GIS parcel layer and GNSS / ground-truth
// points, compare them with the AI parcels, and show where they disagree.
(function () {
"use strict";
var WB = window.WB;
var esc = WB.esc, fmt = WB.fmt;
var rec = {status: null, refLayer: null, missingLayer: null, gnssLayer: null, encLayer: null, tolerance: 1.0, busy: false};

function base() { return '/api/surveys/' + encodeURIComponent(WB.state.sid); }

function clearLayers() {
  var ctl = WB.layersControl();
  ['refLayer', 'missingLayer', 'gnssLayer', 'encLayer'].forEach(function (k) {
    if (rec[k]) { WB.map.removeLayer(rec[k]); if (ctl) ctl.removeLayer(rec[k]); rec[k] = null; }
  });
}

function errColour(e) {
  if (e == null) return '#9FB3A8';
  return e <= rec.tolerance ? '#6CBE81' : (e <= 2 * rec.tolerance ? '#E0A94A' : '#E2685F');
}

function drawLayers() {
  clearLayers();
  var ctl = WB.layersControl();
  var sid = encodeURIComponent(WB.state.sid);
  var s = rec.status || {};
  var cmp = s.comparison || {};
  var jobs = [];
  if (s.has_parcels) {
    jobs.push(fetch('/surveys/' + sid + '/reference/parcels.geojson', {cache: 'no-store'}).then(function (r) { return r.json(); }).then(function (fc) {
      rec.refLayer = L.geoJSON(fc, {
        interactive: false, pmIgnore: true,
        style: {color: '#F2D16B', weight: 1.6, dashArray: '6 4', fill: false}
      }).addTo(WB.map);
      if (ctl) ctl.addOverlay(rec.refLayer, 'Existing parcel record');
    }));
  }
  var missing = (cmp.parcels || {}).missing || [];
  if (missing.length) {
    rec.missingLayer = L.geoJSON({type: 'FeatureCollection', features: missing.map(function (m) {
      return {type: 'Feature', properties: {ref_id: m.ref_id}, geometry: m.geometry};
    })}, {interactive: false, pmIgnore: true, style: {color: '#FF5FD2', weight: 2.5, dashArray: '2 4', fillColor: '#FF5FD2', fillOpacity: 0.12}}).addTo(WB.map);
    if (ctl) ctl.addOverlay(rec.missingLayer, 'Record parcels missing from AI');
  }
  var enc = (cmp.encroachment || {}).rows || [];
  if (enc.length) {
    rec.encLayer = L.geoJSON({type: 'FeatureCollection', features: enc.map(function (r) {
      return {type: 'Feature', properties: r, geometry: r.geometry};
    })}, {pmIgnore: true, style: {color: '#FF2D55', weight: 3, fillColor: '#FF2D55', fillOpacity: 0.35},
      onEachFeature: function (f, l) { l.bindTooltip(esc(f.properties.message), {sticky: true}); }}).addTo(WB.map);
    if (ctl) ctl.addOverlay(rec.encLayer, 'Encroachment vs record');
  }
  var rows = (cmp.gnss || {}).rows;
  if (rows && rows.length) {
    rec.gnssLayer = L.layerGroup();
    rows.forEach(function (r) {
      var p = r.point.coordinates, v = r.nearest_vertex.coordinates;
      var col = r.is_corner ? errColour(r.error_to_vertex_m) : '#9FB3A8';
      if (r.is_corner) L.polyline([[p[1], p[0]], [v[1], v[0]]], {color: col, weight: 1.5, interactive: false}).addTo(rec.gnssLayer);
      L.circleMarker([p[1], p[0]], {radius: 5, color: '#0D1310', weight: 1.5, fillColor: col, fillOpacity: 1, pmIgnore: true})
        .bindTooltip(esc(r.point_id) + (r.is_corner ? ' · ' + r.error_to_vertex_m.toFixed(2) + ' m to nearest AI vertex' : ' · ' + esc(r.type) + ' (not scored)'))
        .addTo(rec.gnssLayer);
    });
    rec.gnssLayer.addTo(WB.map);
    if (ctl) ctl.addOverlay(rec.gnssLayer, 'GNSS / ground-truth points');
  } else if (s.has_gnss) {
    jobs.push(fetch('/surveys/' + sid + '/reference/gnss.geojson', {cache: 'no-store'}).then(function (r) { return r.json(); }).then(function (fc) {
      rec.gnssLayer = L.geoJSON(fc, {pointToLayer: function (f, ll) {
        return L.circleMarker(ll, {radius: 5, color: '#0D1310', weight: 1.5, fillColor: '#9FB3A8', fillOpacity: 1, pmIgnore: true})
          .bindTooltip(esc(f.properties.point_id) + ' · not compared yet');
      }}).addTo(WB.map);
      if (ctl) ctl.addOverlay(rec.gnssLayer, 'GNSS / ground-truth points');
    }));
  }
  return Promise.all(jobs);
}

function load() {
  return WB.api('GET', base() + '/reference').then(function (s) {
    rec.status = s;
    if (s.comparison && s.comparison.tolerance_m) rec.tolerance = s.comparison.tolerance_m;
    return drawLayers();
  }).then(function () { if (WB.state.tab === 'records') WB.renderTabs(); })
    .catch(function (e) { WB.toast('Could not load reference data: ' + e.message, true); });
}

document.addEventListener('wb:survey-opened', function () { rec.status = null; clearLayers(); load(); });

function tile(num, label, colour) {
  return '<div class="stat-tile"><div class="num"' + (colour ? ' style="color:' + colour + '"' : '') + '>' + esc(num) + '</div><div class="lbl">' + esc(label) + '</div></div>';
}

function uploadCard(kind) {
  var s = rec.status || {}, m = (s.meta || {})[kind];
  var isParcels = kind === 'parcels';
  var title = isParcels ? 'Existing parcel layer' : 'GNSS / ground-truth points';
  if (m) {
    var detail = isParcels
      ? m.imported_polygons + ' parcels imported' + (m.records_skipped ? ' · ' + m.records_skipped + ' outside the survey or not polygons' : '') + (m.id_field ? ' · ID field: ' + m.id_field : '')
      : m.imported + ' points imported' + (m.outside_survey ? ' · ' + m.outside_survey + ' outside the survey' : '') + (m.epsg ? ' · EPSG:' + m.epsg : '');
    return '<div class="export-card"><b>' + title + '</b><small>' + esc(m.file) + ' · ' + esc(detail) + '</small>' +
      '<a class="btn" href="#" data-rec-clear="' + kind + '">Remove</a></div>';
  }
  return '<form class="export-card rec-upload" data-kind="' + kind + '" style="grid-template-columns:1fr">' +
    '<b>' + title + '</b>' +
    '<small>' + (isParcels
      ? 'Official or earlier cadastral layer — GeoJSON, GeoPackage, KML or zipped Shapefile, any declared CRS.'
      : 'CSV with lat/lon, or easting/northing plus EPSG code; optional ID, type (corner / control) and accuracy columns. GeoJSON points also work.') + '</small>' +
    '<input type="file" name="file" required accept="' + (isParcels ? '.geojson,.json,.gpkg,.zip,.kml' : '.csv,.txt,.geojson,.json') + '" style="font-size:0.72rem;color:var(--text-dim)">' +
    (isParcels
      ? '<input type="text" name="id_field" placeholder="ID field (optional, e.g. SURVEY_NO)" class="rec-input">'
      : '<input type="text" name="epsg" placeholder="EPSG of easting/northing (e.g. 32644)" class="rec-input">') +
    '<button class="btn primary" type="submit">Import</button></form>';
}

function encroachResults(e) {
  var html = '<div><h3 class="panel-title">Buildings vs record <span class="note">possible encroachments</span></h3>' +
    '<div class="stat-grid">' + tile(e.crosses_record, 'Across a recorded boundary', e.crosses_record ? '#FF2D55' : null) +
    tile(e.outside_record, 'Outside every recorded parcel', e.outside_record ? '#FF2D55' : null) + '</div>';
  if (e.rows.length) {
    html += '<div class="list" style="margin-top:10px">' + e.rows.slice(0, 60).map(function (r, i) {
      return '<div class="row" data-enc="' + i + '"><span class="status-dot" style="background:#FF2D55"></span>' +
        '<div><div class="pin">Building ' + esc(r.building_id) + '</div><div class="sub">' + esc(r.message) + '</div></div><div class="right">' + fmt(r.area_m2) + ' m²</div></div>';
    }).join('') + '</div>';
  } else {
    html += '<div class="clean-msg">No building crosses a recorded boundary or stands outside the record.</div>';
  }
  return html + '<div class="note" style="margin-top:6px">Send these for a field check before the record is updated.</div></div>';
}

function parcelResults(p) {
  var c = p.counts, R = WB.RECORD;
  var html = '<div><h3 class="panel-title">AI vs existing record</h3><div class="stat-grid">' +
    tile(fmt(p.agreement_pct, 1) + '%', 'AI parcels matching record', '#6CBE81') +
    tile(p.mean_iou_matched == null ? '—' : p.mean_iou_matched.toFixed(2), 'Mean IoU of matches') +
    tile(p.median_area_diff_pct == null ? '—' : fmt(p.median_area_diff_pct, 1) + '%', 'Median area difference') +
    tile(p.missing_from_ai, 'Record parcels missing from AI', p.missing_from_ai ? '#FF5FD2' : null) +
    '</div><div class="btn-row" style="margin-top:10px;gap:6px">' + Object.keys(R).map(function (k) {
      return '<span class="tag" style="color:' + R[k].color + ';border-color:' + R[k].color + '">' + R[k].label + ' ' + c[k] + '</span>';
    }).join('') + '</div>' +
    '<div class="note" style="margin-top:8px">Match = IoU ≥ ' + p.thresholds.match_iou + '; boundary differs = IoU ≥ ' + p.thresholds.partial_iou + '. Switch "Colour parcels by" to <b>Record</b> to see this on the map.</div></div>';
  var diffs = p.per_parcel.filter(function (r) { return r.status !== 'MATCH'; })
    .sort(function (a, b) { return a.iou - b.iou; });
  if (diffs.length) {
    html += '<div><h3 class="panel-title">Disagreements <span class="note">' + diffs.length + '</span></h3><div class="list">' +
      diffs.slice(0, 150).map(function (r) {
        return '<div class="row" data-pid="' + r.parcel_id + '"><span class="status-dot" style="background:' + R[r.status].color + '"></span>' +
          '<div><div class="pin">#' + r.parcel_id + ' · ' + esc(R[r.status].label) + '</div><div class="sub">' +
          (r.ref_id ? 'record ' + esc(r.ref_id) + ' · area ' + (r.area_diff_pct > 0 ? '+' : '') + fmt(r.area_diff_pct, 1) + '% · offset ' + fmt(r.boundary_offset_m, 1) + ' m' : 'no overlapping record parcel') +
          '</div></div><div class="right">IoU ' + r.iou.toFixed(2) + '</div></div>';
      }).join('') + '</div></div>';
  }
  return html;
}

function gnssResults(g) {
  var html = '<div><h3 class="panel-title">Positional accuracy vs GNSS</h3><div class="stat-grid">' +
    tile(g.rmse_m == null ? '—' : g.rmse_m.toFixed(2) + ' m', 'RMSE (corners)') +
    tile(g.ce90_m == null ? '—' : g.ce90_m.toFixed(2) + ' m', 'CE90') +
    tile(g.max_error_m == null ? '—' : g.max_error_m.toFixed(2) + ' m', 'Max error') +
    tile(g.within_tolerance_pct == null ? '—' : fmt(g.within_tolerance_pct, 0) + '%', 'Within ' + rec.tolerance + ' m', '#6CBE81') +
    '</div><div class="note" style="margin-top:8px">' + g.corner_points + ' surveyed corner(s) scored against the nearest AI parcel vertex' +
    (g.points > g.corner_points ? '; ' + (g.points - g.corner_points) + ' non-corner point(s) shown but not scored' : '') + '.</div></div>';
  var worst = g.rows.filter(function (r) { return r.is_corner; }).sort(function (a, b) { return b.error_to_vertex_m - a.error_to_vertex_m; });
  if (worst.length) {
    html += '<div><h3 class="panel-title">Points by error</h3><div class="list">' + worst.slice(0, 100).map(function (r, i) {
      return '<div class="row" data-gnss="' + i + '"><span class="status-dot" style="background:' + errColour(r.error_to_vertex_m) + '"></span>' +
        '<div><div class="pin">' + esc(r.point_id) + '</div><div class="sub">' + esc(r.type) + (r.accuracy_m != null ? ' · GNSS ±' + r.accuracy_m + ' m' : '') +
        ' · ' + r.error_to_boundary_m.toFixed(2) + ' m to boundary line</div></div><div class="right">' + r.error_to_vertex_m.toFixed(2) + ' m</div></div>';
    }).join('') + '</div></div>';
    rec.worst = worst;
  }
  return html;
}

function render() {
  var s = rec.status;
  if (!s) return '<div class="note">Loading reference data…</div>';
  var cmp = s.comparison;
  var html = '<div><h3 class="panel-title">Reference data</h3><div class="note">Compare the AI parcel map with what already exists: an earlier or official parcel layer, and field-surveyed GNSS / CORS corner points. Everything is measured in ' + esc(WB.state.meta.crs) + ' metres.</div></div>' +
    uploadCard('parcels') + uploadCard('gnss');
  if (s.has_parcels || s.has_gnss) {
    html += '<div class="btn-row" style="align-items:center"><label class="note" style="flex:none">Corner tolerance (m)</label>' +
      '<input type="number" min="0.05" step="0.05" value="' + rec.tolerance + '" id="recTol" class="rec-input" style="max-width:90px">' +
      '<button class="btn primary" data-rec-run="1"' + (rec.busy ? ' disabled' : '') + '>' + (rec.busy ? 'Comparing…' : (cmp ? 'Re-run comparison' : 'Run comparison')) + '</button></div>';
    if (cmp) {
      html += '<div class="note">Last compared ' + esc(cmp.compared_at) + '. Re-run after editing parcels.</div>';
      if (cmp.parcels) html += parcelResults(cmp.parcels);
      if (cmp.encroachment) html += encroachResults(cmp.encroachment);
      if (cmp.gnss) html += gnssResults(cmp.gnss);
    }
  }
  return html;
}

function wire(body) {
  body.querySelectorAll('[data-enc]').forEach(function (el) {
    el.addEventListener('click', function () {
      var r = (((rec.status || {}).comparison || {}).encroachment || {}).rows[Number(el.dataset.enc)];
      if (r) WB.zoomTo({type: 'Feature', geometry: r.geometry});
    });
  });
  body.querySelectorAll('form.rec-upload').forEach(function (form) {
    form.addEventListener('submit', function (e) {
      e.preventDefault();
      var kind = form.dataset.kind, fd = new FormData(form);
      var btn = form.querySelector('button');
      btn.disabled = true; btn.textContent = 'Importing…';
      fetch(base() + '/reference/' + kind, {method: 'POST', body: fd})
        .then(function (r) { return r.json().then(function (j) { if (!r.ok) throw new Error(j.error); return j; }); })
        .then(function (s) { rec.status = s; WB.toast('Imported.'); return drawLayers(); })
        .then(function () { WB.renderTabs(); })
        .catch(function (err) { btn.disabled = false; btn.textContent = 'Import'; WB.toast(err.message, true); });
    });
  });
  body.querySelectorAll('[data-rec-clear]').forEach(function (a) {
    a.addEventListener('click', function (e) {
      e.preventDefault();
      if (!confirm('Remove this reference data from the survey?')) return;
      WB.api('DELETE', base() + '/reference/' + a.dataset.recClear)
        .then(function (s) { rec.status = s; return WB.reloadParcels(); })
        .then(drawLayers).then(function () { WB.renderTabs(); });
    });
  });
  var run = body.querySelector('[data-rec-run]');
  if (run) run.addEventListener('click', function () {
    var tol = parseFloat(body.querySelector('#recTol').value);
    if (!(tol > 0)) { WB.toast('Tolerance must be a positive number of metres.', true); return; }
    rec.tolerance = tol; rec.busy = true; WB.renderTabs();
    WB.api('POST', base() + '/compare', {tolerance_m: tol})
      .then(function (cmp) {
        rec.status.comparison = cmp;
        return WB.reloadParcels().then(drawLayers);
      })
      .then(function () { rec.busy = false; WB.renderTabs(); WB.toast('Comparison updated.'); })
      .catch(function (err) { rec.busy = false; WB.renderTabs(); WB.toast('Comparison failed: ' + err.message, true); });
  });
  body.querySelectorAll('[data-gnss]').forEach(function (el) {
    el.addEventListener('click', function () {
      var r = rec.worst[Number(el.dataset.gnss)];
      var c = r.point.coordinates;
      WB.map.setView([c[1], c[0]], Math.max(WB.map.getZoom(), 21));
    });
  });
}

WB.registerTab('records', render, wire);
})();
