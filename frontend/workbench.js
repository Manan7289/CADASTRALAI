// CadastraAI Survey Workbench: review, edit, validate and export AI-extracted parcels.
(function () {
"use strict";

var STATUS = {
  draft:       {label: 'To review',   color: '#9FB3A8'},
  approved:    {label: 'Approved',    color: '#6CBE81'},
  field_check: {label: 'Field check', color: '#E0A94A'},
  rejected:    {label: 'Rejected',    color: '#E2685F'}
};
// One palette for every layer, legend and chart (backend survey.PALETTE draws the rasters):
// a colour means the same thing wherever it appears.
var PALETTE = {building: '#FF6B4A', road: '#E8E8E8', paved: '#969696', tree: '#2E8B3E', grass: '#8FD16A',
               agriculture: '#C9D65B', bare: '#C8955A', water: '#3A86FF', open: '#D8C9A3'};
// suggested land use per parcel (backend land_use.py)
var LANDUSE = {
  'Residential': '#F2C14E', 'Residential - apartments': '#E8913A', 'Commercial / mixed use': '#D1495B',
  'Institutional / large complex': '#8F7CF6', 'Vacant plot': '#C8955A', 'Open space / green': '#8FD16A', 'Water body': '#3A86FF'
};
var LANDCOVER = {
  'Built-up': PALETTE.building, 'Vegetated open land': PALETTE.grass, 'Open / vacant land': PALETTE.open,
  'Barren land': PALETTE.bare, 'Paved / developed open land': PALETTE.paved, 'Water': PALETTE.water
};
// 8-class land-cover layer (land cover v2): [key in the survey stats, label, colour]
var LC8 = [['building', 'Building', PALETTE.building], ['road', 'Road', PALETTE.road], ['tree', 'Tree', PALETTE.tree],
           ['grass / scrub', 'Grass / scrub', PALETTE.grass], ['agriculture', 'Farmland', PALETTE.agriculture],
           ['bare land', 'Bare / barren land', PALETTE.bare], ['water', 'Water', PALETTE.water],
           ['paved / developed', 'Paved open area', PALETTE.paved]];
var CLASS_META = [
  ['building', 'Building', PALETTE.building], ['road', 'Road / paved', PALETTE.road],
  ['low_veg', 'Low vegetation', PALETTE.grass], ['tree', 'Tree', PALETTE.tree], ['clutter', 'Other', PALETTE.paved]
];
var ROAD_CLASS = {'lane': ['#FFB020', 'Lane (under 3 m)'], 'street': ['#8FE0FF', 'Street (3–8 m)'], 'main road': ['#FF5FD2', 'Main road (over 8 m)']};
var FILL_SOURCE = 'fill';          // building footprints filled in from the land-cover model carry this in `source`
var ENCROACH_COLOUR = '#FF2D55';
var ISSUE_LABEL = {
  INVALID: 'Invalid geometry', MULTIPART: 'Multipart parcel', OVERLAP: 'Overlap', DUPLICATE: 'Duplicate',
  GAP: 'Gap between parcels', SLIVER: 'Sliver', TOO_SMALL: 'Too small', HOLE: 'Encloses another area'
};
var MIN_DRAWN_M2 = 3;
var RECORD = {
  MATCH: {label: 'Matches record', color: '#6CBE81'},
  BOUNDARY_DIFFERS: {label: 'Boundary differs', color: '#E0A94A'},
  SPLIT_OR_MERGED: {label: 'Split / merged vs record', color: '#B694DA'},
  NOT_IN_RECORD: {label: 'Not in record', color: '#E2685F'}
};

var state = {
  surveys: [], sid: null, meta: null,
  parcels: null, issues: null, buildings: null, corridors: null,
  selected: [], tab: 'overview', colourMode: 'landuse', tool: 'select',
  reviewFilter: 'draft', editingLayer: null,
  view: 'parcels', on: {}  // on: which layer keys should be on the map
};

function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
    return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c];
  });
}
function $(id) { return document.getElementById(id); }
function fmt(n, d) { return n == null ? '—' : Number(n).toLocaleString('en-IN', {maximumFractionDigits: d == null ? 1 : d}); }
function toast(msg, isError) {
  document.querySelectorAll('.toast').forEach(function (old) { old.remove(); });
  var t = document.createElement('div');
  t.className = 'toast' + (isError ? ' error' : '');
  t.textContent = msg;
  (document.querySelector('.map-panel') || document.body).appendChild(t);
  setTimeout(function () { t.remove(); }, isError ? 6000 : 2800);
}
function setSave(stateName, text) {
  var chip = $('saveChip');
  chip.dataset.state = stateName;
  chip.textContent = text;
}
function api(method, url, body) {
  return fetch(url, {
    method: method, headers: body ? {'Content-Type': 'application/json'} : {},
    body: body ? JSON.stringify(body) : undefined
  }).then(function (r) {
    return r.json().then(function (j) {
      if (!r.ok) throw new Error(j.error || ('HTTP ' + r.status));
      return j;
    });
  });
}
function confColour(c) {
  if (c == null) return '#5C6B63';
  if (c < 0.5) return '#E2685F';
  if (c < 0.7) return '#E0A94A';
  return '#6CBE81';
}

// ---------------------------------------------------------------- map
var map = L.map('map', {zoomControl: true, minZoom: 12, maxZoom: 23, zoomSnap: 0, zoomDelta: 0.5});
map.pm.setGlobalOptions({snappable: true, snapDistance: 12, allowSelfIntersection: false});
map.pm.setPathOptions({color: '#45C7B8', fillOpacity: 0.15});

// raster overlays sit under every vector layer and never catch clicks
map.createPane('rasters');
map.getPane('rasters').style.zIndex = 350;
map.getPane('rasters').style.pointerEvents = 'none';

var overlays = {};
var extraLayers = [];  // layers other modules add (records.js): [{layer, name}]
var parcelLayer = null, issueLayer = null, buildingLayer = null, corridorLayer = null, labelLayer = null;

function clearMap() {
  [parcelLayer, issueLayer, buildingLayer, corridorLayer, labelLayer].forEach(function (l) { if (l) map.removeLayer(l); });
  Object.keys(overlays).forEach(function (k) { map.removeLayer(overlays[k]); });
  extraLayers.forEach(function (x) { map.removeLayer(x.layer); });
  overlays = {};
  extraLayers = [];
  parcelLayer = issueLayer = buildingLayer = corridorLayer = labelLayer = null;
}

// what records.js expects from a Leaflet layers control
var layersControl = {
  addOverlay: function (layer, name) { extraLayers.push({layer: layer, name: name}); renderLayerPanel(); },
  removeLayer: function (layer) {
    extraLayers = extraLayers.filter(function (x) { return x.layer !== layer; });
    renderLayerPanel();
  }
};

function parcelStyle(f) {
  var p = f.properties;
  var sel = state.selected.indexOf(p.id) >= 0;
  var fill;
  if (state.colourMode === 'confidence') fill = confColour(p.confidence);
  else if (state.colourMode === 'landcover') fill = LANDCOVER[p.landcover] || '#5C6B63';
  else if (state.colourMode === 'landuse') fill = LANDUSE[p.land_use] || LANDCOVER[p.landcover] || '#5C6B63';
  else if (state.colourMode === 'record') fill = RECORD[p.record_status] ? RECORD[p.record_status].color : '#5C6B63';
  else fill = STATUS[p.status] ? STATUS[p.status].color : '#9FB3A8';
  var hasIssue = p.issues && p.issues.length;
  return {
    color: sel ? '#45C7B8' : (hasIssue ? '#E2685F' : '#F2E9DC'),
    weight: sel ? 3 : (hasIssue ? 2 : 1.2),
    opacity: 0.95,
    fillColor: fill,
    fillOpacity: sel ? 0.35 : (state.colourMode === 'status' && p.status === 'draft' ? 0.08 : 0.3)
  };
}

function buildParcelLayer() {
  if (parcelLayer) map.removeLayer(parcelLayer);
  if (labelLayer) map.removeLayer(labelLayer);
  parcelLayer = L.geoJSON(state.parcels, {
    style: parcelStyle,
    pmIgnore: false,
    onEachFeature: function (f, layer) {
      layer.on('click', function (e) {
        // while drawing/editing, let the click reach Geoman on the map instead of selecting
        if (state.tool === 'edit' || map.pm.globalDrawModeEnabled()) return;
        L.DomEvent.stopPropagation(e);
        selectParcel(f.properties.id, e.originalEvent.shiftKey || state.tool === 'merge');
      });
    }
  });
  labelLayer = L.layerGroup();
  state.parcels.features.forEach(function (f) {
    var c = L.geoJSON(f).getBounds().getCenter();
    labelLayer.addLayer(L.tooltip({permanent: true, direction: 'center', className: 'parcel-label', interactive: false})
      .setLatLng(c).setContent(String(f.properties.id)));
  });
  overlays.parcels = parcelLayer;
  applyLayers();
}

function updateLabelVisibility() {
  if (!labelLayer) return;
  if (map.getZoom() >= 19.5 && map.hasLayer(parcelLayer)) map.addLayer(labelLayer);
  else map.removeLayer(labelLayer);
}
map.on('zoomend', updateLabelVisibility);

function buildIssueLayer() {
  if (issueLayer) map.removeLayer(issueLayer);
  issueLayer = L.geoJSON(state.issues, {
    interactive: false,
    style: function (f) {
      var high = f.properties.severity === 'high';
      return {color: high ? '#E2685F' : '#E0A94A', weight: 2, dashArray: '4 3', fillColor: high ? '#E2685F' : '#E0A94A', fillOpacity: 0.45};
    },
    pointToLayer: function (f, ll) { return L.circleMarker(ll, {radius: 6}); }
  });
  overlays.issues = issueLayer;
  applyLayers();
}

function restyle() { if (parcelLayer) parcelLayer.setStyle(parcelStyle); renderLayerPanel(); }

// ---------------------------------------------------------------- views + layer panel
// The five views follow the pipeline, so each one is a self-explanatory screen (and a slide).
var VIEWS = [
  {id: 'image', label: 'Image', layers: ['ori']},
  {id: 'buildings', label: 'Buildings', layers: ['ori', 'buildings']},
  {id: 'landcover', label: 'Land cover', layers: ['ori', 'LC']},
  {id: 'parcels', label: 'Parcels', layers: ['ori', 'corridors', 'parcels'], colour: 'landuse'},
  {id: 'review', label: 'Review', layers: ['ori', 'parcels', 'issues'], colour: 'status'}
];

function viewLayers(v) {
  return v.layers.map(function (k) { return k === 'LC' ? (state.meta && state.meta.has_landcover_layer ? 'landcover' : 'classes') : k; });
}

function setView(id) {
  var v = VIEWS.filter(function (x) { return x.id === id; })[0] || VIEWS[3];
  state.view = v.id;
  state.on = {};
  viewLayers(v).forEach(function (k) { state.on[k] = true; });
  if (v.colour) state.colourMode = v.colour;
  if (parcelLayer) parcelLayer.setStyle(parcelStyle);
  applyLayers();
}

function toggleLayer(key, on) {
  state.on[key] = on;
  var keys = Object.keys(state.on).filter(function (k) { return state.on[k]; }).sort().join();
  var match = VIEWS.filter(function (v) { return viewLayers(v).slice().sort().join() === keys; })[0];
  state.view = match ? match.id : null;
  applyLayers();
}

function layerByKey() {
  return {ori: overlays.ori, classes: overlays.classes, height: overlays.height, landcover: overlays.landcover,
          corridors: corridorLayer, roadnet: overlays.roadnet, parcels: parcelLayer, buildings: buildingLayer, issues: issueLayer};
}

function applyLayers() {
  var byKey = layerByKey();
  Object.keys(byKey).forEach(function (k) {
    var l = byKey[k];
    if (!l) return;
    if (state.on[k] && !map.hasLayer(l)) map.addLayer(l);
    else if (!state.on[k] && map.hasLayer(l)) map.removeLayer(l);
  });
  // drawing order: roads under parcels, building outlines over parcels, problems on top
  [corridorLayer, parcelLayer, overlays.roadnet, buildingLayer, issueLayer].concat(extraLayers.map(function (x) { return x.layer; }))
    .forEach(function (l) { if (l && map.hasLayer(l) && l.bringToFront) l.bringToFront(); });
  updateLabelVisibility();
  var tb = document.querySelector('.map-toolbar');
  if (tb) tb.classList.toggle('hidden', !state.on.parcels);
  renderViewBar();
  renderLayerPanel();
}

function renderViewBar() {
  if (state.sid) history.replaceState(null, '', '?survey=' + encodeURIComponent(state.sid) + (state.view ? '&view=' + state.view : ''));
  document.querySelectorAll('#viewBar button').forEach(function (b) { b.classList.toggle('active', b.dataset.view === state.view); });
  var cap = $('viewCaption');
  var text = state.meta && state.view ? viewCaption(state.view) : '';
  cap.innerHTML = text;
  cap.style.display = text ? '' : 'none';
}

function nFilled() {
  return state.buildings ? state.buildings.features.filter(function (f) { return (f.properties.source || '').indexOf(FILL_SOURCE) >= 0; }).length : 0;
}

function topLandCover(n) {
  var lcf = state.meta.stats && state.meta.stats.land_cover_fraction;
  if (!lcf) return '';
  return LC8.slice().sort(function (a, b) { return (lcf[b[0]] || 0) - (lcf[a[0]] || 0); }).slice(0, n)
    .map(function (c) { return esc(c[1].toLowerCase()) + ' ' + Math.round(100 * (lcf[c[0]] || 0)) + '%'; }).join(', ');
}

function viewCaption(id) {
  var m = state.meta, s = m.stats || {}, nP = state.parcels.features.length, nB = state.buildings.features.length, nF = nFilled();
  var done = state.parcels.features.filter(function (f) { return f.properties.status !== 'draft'; }).length;
  if (id === 'image') return '<b>1 · Input image.</b> The aerial orthoimage (ORI), ' + fmt(m.gsd_m * 100, 0) +
    ' cm per pixel' + (m.used_height ? ', with a surface-height model (DSM)' : '') + '. Everything else on this map is drawn by the AI from it.';
  if (id === 'buildings') return '<b>2 · Buildings.</b> ' + nB + ' roofs outlined by the AI.' +
    (nF ? ' <span class="k-solid"></span> Solid: roof model. <span class="k-dash"></span> Dashed: missed by the roof model, found by the land-cover model — check these.' : '');
  if (id === 'landcover') return '<b>3 · Land cover.</b> Every pixel classified as ' + (m.has_landcover_layer
    ? 'building, road, tree, grass, farmland, bare land, water or paved area' : 'building, road / paved, low vegetation, tree or other') +
    (topLandCover(3) ? ' — here mostly ' + topLandCover(3) : '') + '.';
  if (id === 'parcels') return '<b>4 · Parcels.</b> ' + nP + ' numbered plots. Each plot is the land around one house: up to the road in front, and midway to the neighbouring houses at the sides and back, where the shared wall or the side setbacks are. Roads and lanes separate plots; open ground (parks, grounds) stays one parcel. Colour: main land use.';
  if (id === 'review') return '<b>5 · Review.</b> ' + done + ' of ' + nP + ' parcels checked. The list on the right starts with the least certain ones: approve, send for a field check or reject. ' +
    (state.issues.features.length ? state.issues.features.length + ' topology problem(s) shown in red.' : 'No topology problems.');
  return '';
}

function sw(style) { return '<span class="sw" style="' + style + '"></span>'; }
function keyRows(rows) {
  return '<div class="lp-key">' + rows.map(function (r) {
    return '<div>' + sw(r[1]) + '<span>' + esc(r[0]) + '</span>' + (r[2] != null ? '<span class="n">' + esc(r[2]) + '</span>' : '') + '</div>';
  }).join('') + '</div>';
}

function parcelKey() {
  var modes = [['landuse', 'Land use'], ['landcover', 'Land cover'], ['status', 'Review'], ['confidence', 'Confidence'], ['record', 'Record']];
  var html = '<div class="lp-chips">' + modes.map(function (m) {
    return '<button data-colour="' + m[0] + '" class="' + (state.colourMode === m[0] ? 'active' : '') + '">' + m[1] + '</button>';
  }).join('') + '</div>';
  if (state.colourMode === 'confidence') {
    return html + '<div class="lp-key"><div class="ramp"></div><div class="ramp-lbl"><span>low</span><span>0.5</span><span>0.7</span><span>high</span></div></div>';
  }
  var src = state.colourMode === 'record'
    ? Object.keys(RECORD).map(function (k) { return [RECORD[k].label, RECORD[k].color]; }).concat([['Not compared yet', '#5C6B63']])
    : state.colourMode === 'status'
      ? Object.keys(STATUS).map(function (k) { return [STATUS[k].label, STATUS[k].color]; })
      : state.colourMode === 'landuse'
        ? Object.keys(LANDUSE).map(function (k) { return [k, LANDUSE[k]]; })
        : Object.keys(LANDCOVER).map(function (k) { return [k, LANDCOVER[k]]; });
  return html + keyRows(src.map(function (r) { return [r[0], 'background:' + r[1]]; }));
}

function layerDefs() {
  var m = state.meta, s = m.stats || {}, nB = state.buildings.features.length, nF = nFilled();
  var defs = [
    {group: 'Input', key: 'ori', name: 'Aerial image (ORI)', info: fmt(m.gsd_m * 100, 0) + ' cm/px'},
    {group: 'What the AI found', key: 'buildings', name: 'Buildings', info: nB, legend: keyRows(nF
      ? [['Roof model', 'border:2px solid ' + PALETTE.building, nB - nF], ['Filled in — check', 'border:2px dashed ' + PALETTE.building, nF]]
      : [['Roof outline', 'border:2px solid ' + PALETTE.building]]).concat(s.encroachments ? keyRows([['On the road — possible encroachment', 'background:' + ENCROACH_COLOUR, s.encroachments]]) : '')}
  ];
  if (overlays.landcover) defs.push({key: 'landcover', name: 'Land cover', info: '8 classes',
    legend: '<div class="lp-grid">' + LC8.map(function (c) { return '<div>' + sw('background:' + c[2]) + esc(c[1]) + '</div>'; }).join('') + '</div>'});
  var byCls = s.road_length_by_class_m || {};
  defs.push({key: 'corridors', name: 'Roads & lanes', info: fmt(s.corridor_length_m, 0) + ' m',
    legend: keyRows([['Paved road area', 'background:rgba(232,232,232,0.3);border:1px dashed ' + PALETTE.road]])});
  if (state.roads && state.roads.features.length) defs.push({key: 'roadnet', name: 'Road network', info: 'by width',
    legend: keyRows(Object.keys(ROAD_CLASS).map(function (k) {
      return [ROAD_CLASS[k][1], 'height:4px;border-radius:2px;background:' + ROAD_CLASS[k][0], byCls[k] != null ? fmt(byCls[k], 0) + ' m' : null];
    }))});
  defs.push({key: 'parcels', name: 'Parcels', info: state.parcels.features.length, legend: parcelKey()});
  defs.push({key: 'issues', name: 'Topology problems', info: state.issues.features.length,
    legend: keyRows([['Overlap, gap, sliver…', 'background:rgba(226,104,95,0.45);border:1px dashed #E2685F']])});
  if (overlays.height) defs.push({group: 'More', key: 'height', name: 'Height above ground', info: 'nDSM'});
  defs.push({group: overlays.height ? null : 'More', key: 'classes', name: 'Raw AI classes', info: '5 classes',
    legend: '<div class="lp-grid">' + CLASS_META.map(function (c) { return '<div>' + sw('background:' + c[2]) + esc(c[1]) + '</div>'; }).join('') + '</div>'});
  return defs;
}

function renderLayerPanel() {
  var el = $('layerPanel');
  if (!el || !state.meta || !state.buildings) return;
  var html = '';
  layerDefs().forEach(function (d) {
    if (d.group) html += '<div class="lp-group">' + esc(d.group) + '</div>';
    var on = !!state.on[d.key];
    html += '<div class="lp-row' + (on ? ' on' : '') + '"><label><input type="checkbox" data-layer="' + d.key + '"' + (on ? ' checked' : '') + '>' +
      '<span class="lp-name">' + esc(d.name) + '</span><span class="n">' + esc(d.info) + '</span></label>' +
      (on && d.legend ? d.legend : '') + '</div>';
  });
  extraLayers.forEach(function (x, i) {
    var on = map.hasLayer(x.layer);
    html += '<div class="lp-row' + (on ? ' on' : '') + '"><label><input type="checkbox" data-extra="' + i + '"' + (on ? ' checked' : '') + '>' +
      '<span class="lp-name">' + esc(x.name) + '</span></label></div>';
  });
  el.querySelector('.lp-body').innerHTML = html;
}

$('layerPanel').addEventListener('change', function (e) {
  var t = e.target;
  if (t.dataset.layer) toggleLayer(t.dataset.layer, t.checked);
  else if (t.dataset.extra != null) {
    var x = extraLayers[Number(t.dataset.extra)];
    if (x) { if (t.checked) map.addLayer(x.layer); else map.removeLayer(x.layer); applyLayers(); }
  }
});
$('layerPanel').addEventListener('click', function (e) {
  var b = e.target.closest('[data-colour]');
  if (b) { state.colourMode = b.dataset.colour; restyle(); }
  if (e.target.closest('.lp-head')) $('layerPanel').classList.toggle('collapsed');
});
document.querySelectorAll('#viewBar button').forEach(function (b) {
  b.addEventListener('click', function () { setView(b.dataset.view); });
});

// ---------------------------------------------------------------- data
function loadSurveys() {
  return api('GET', '/api/surveys').then(function (list) {
    state.surveys = list;
    var sel = $('surveySelect');
    sel.innerHTML = list.map(function (s) {
      return '<option value="' + esc(s.id) + '">' + esc(s.name) + '</option>';
    }).join('');
    if (!list.length) {
      $('loadingScreen').innerHTML = 'No surveys yet.<br><a href="upload.html" style="color:var(--accent)">Process a survey</a>';
      return;
    }
    var want = new URLSearchParams(location.search).get('survey');
    var sid = list.some(function (s) { return s.id === want; }) ? want : list[list.length - 1].id;
    sel.value = sid;
    return openSurvey(sid);
  });
}

function openSurvey(sid) {
  $('loadingScreen').style.display = 'flex';
  $('loadingScreen').textContent = 'Loading survey…';
  var base = '/surveys/' + encodeURIComponent(sid) + '/';
  return Promise.all([
    fetch(base + 'meta.json').then(function (r) { return r.json(); }),
    fetch(base + 'parcels.geojson').then(function (r) { return r.json(); }),
    fetch(base + 'issues.geojson').then(function (r) { return r.json(); }),
    fetch(base + 'buildings.geojson').then(function (r) { return r.json(); }),
    fetch(base + 'corridors.geojson').then(function (r) { return r.json(); }),
    fetch(base + 'roads.geojson').then(function (r) { return r.ok ? r.json() : {type: 'FeatureCollection', features: []}; })
  ]).then(function (res) {
    clearMap();
    state.sid = sid;
    state.meta = res[0]; state.parcels = res[1]; state.issues = res[2]; state.buildings = res[3]; state.corridors = res[4]; state.roads = res[5];
    state.lastFixLog = null;   // the auto-fix log belongs to the survey it ran on
    state.selected = [];
    $('surveySubtitle').textContent = 'Survey Workbench · ' + state.meta.name;

    var b = state.meta.bounds;
    var tiles = state.meta.tiles;
    overlays.ori = tiles && tiles.count
      ? L.tileLayer(base + 'tiles/{z}/{x}/{y}.webp', {
          minNativeZoom: tiles.min_zoom, maxNativeZoom: tiles.max_zoom, maxZoom: 23,
          bounds: L.latLngBounds(b), attribution: esc(state.meta.source)})
      : L.imageOverlay(base + 'ori.webp', b, {attribution: esc(state.meta.source), pane: 'rasters'});
    overlays.classes = L.imageOverlay(base + 'classes.png', b, {opacity: 0.6, pane: 'rasters'});
    if (state.meta.has_height_layer) overlays.height = L.imageOverlay(base + 'height.png', b, {opacity: 0.6, pane: 'rasters'});
    if (state.meta.has_landcover_layer) overlays.landcover = L.imageOverlay(base + 'landcover.png', b, {opacity: 0.6, pane: 'rasters'});
    buildingLayer = L.geoJSON(state.buildings, {interactive: false, pmIgnore: true, style: function (f) {
      var filled = (f.properties.source || '').indexOf(FILL_SOURCE) >= 0;
      if (f.properties.encroachment_m2) return {color: ENCROACH_COLOUR, weight: 3, fillColor: ENCROACH_COLOUR, fillOpacity: 0.35};
      return {color: PALETTE.building, weight: filled ? 1.6 : 1.8, dashArray: filled ? '5 4' : null,
              fillColor: PALETTE.building, fillOpacity: 0.1};
    }});
    // roads & lanes: the paved area; the centreline network (width classes) is a layer of its own
    corridorLayer = L.geoJSON(state.corridors, {interactive: false, pmIgnore: true,
      style: {color: PALETTE.road, weight: 1, dashArray: '3 3', fillColor: PALETTE.road, fillOpacity: 0.2}});
    overlays.roadnet = L.geoJSON(state.roads, {pmIgnore: true, style: function (f) {
      return {color: (ROAD_CLASS[f.properties.class] || ROAD_CLASS.street)[0], weight: 3, opacity: 0.95};
    }, onEachFeature: function (f, l) {
      l.bindTooltip((ROAD_CLASS[f.properties.class] || ROAD_CLASS.street)[1].split(' (')[0] + ' · ' + f.properties.width_m + ' m wide · ' + f.properties.length_m + ' m long', {sticky: true});
    }});
    overlays.buildings = buildingLayer;
    overlays.corridors = corridorLayer;
    var wantView = new URLSearchParams(location.search).get('view');
    setView(VIEWS.some(function (v) { return v.id === wantView; }) ? wantView : (state.view || 'parcels'));
    buildParcelLayer();
    buildIssueLayer();

    map.invalidateSize();
    map.fitBounds(b);
    renderTabs();
    $('loadingScreen').style.display = 'none';
    document.dispatchEvent(new CustomEvent('wb:survey-opened', {detail: {sid: sid, base: base}}));
  }).catch(function (e) {
    console.error('openSurvey failed', e);
    $('loadingScreen').textContent = 'Could not load survey: ' + e.message;
  });
}

function applyServerResult(res) {
  state.parcels = res.parcels;
  state.issues = res.issues;
  var ids = state.parcels.features.map(function (f) { return f.properties.id; });
  state.selected = state.selected.filter(function (id) { return ids.indexOf(id) >= 0; });
  buildParcelLayer();
  buildIssueLayer();
  renderTabs();
}

function save(action) {
  setSave('saving', 'Saving…');
  return api('PUT', '/api/surveys/' + encodeURIComponent(state.sid) + '/parcels?action=' + encodeURIComponent(action || 'edited'), state.parcels)
    .then(function (res) {
      applyServerResult(res);
      var n = res.issues.features.length;
      setSave('idle', n ? 'Saved · ' + n + ' issue' + (n === 1 ? '' : 's') : 'Saved · topology clean');
    })
    .catch(function (e) { setSave('error', 'Save failed'); toast('Save failed: ' + e.message, true); });
}

function parcelById(id) {
  for (var i = 0; i < state.parcels.features.length; i++) {
    if (state.parcels.features[i].properties.id === id) return state.parcels.features[i];
  }
  return null;
}

// ---------------------------------------------------------------- selection + review
function selectParcel(id, additive) {
  if (additive) {
    var i = state.selected.indexOf(id);
    if (i >= 0) state.selected.splice(i, 1); else state.selected.push(id);
  } else {
    state.selected = [id];
  }
  restyle();
  if (state.tab !== 'review') state.tab = 'review';
  renderTabs();
}

function zoomTo(geojson) {
  var bounds = L.geoJSON(geojson).getBounds();
  if (bounds.isValid()) map.fitBounds(bounds, {maxZoom: 21, padding: [80, 80]});
}

function setStatus(status) {
  if (!state.selected.length) return;
  state.selected.forEach(function (id) {
    var f = parcelById(id);
    if (f) f.properties.status = status;
  });
  var label = STATUS[status].label;
  save('status -> ' + status + ' for ' + state.selected.join(','));
  toast(state.selected.length + ' parcel' + (state.selected.length > 1 ? 's' : '') + ' marked "' + label + '"');
  if (status !== 'draft' && state.selected.length === 1) nextToReview();
}

function reviewQueue() {
  return state.parcels.features.slice().filter(function (f) {
    if (state.reviewFilter === 'encroach') return !!f.properties.encroachment_m2;
    return state.reviewFilter === 'all' || f.properties.status === state.reviewFilter;
  }).sort(function (a, b) {
    var ia = (a.properties.issues || []).length ? 0 : 1, ib = (b.properties.issues || []).length ? 0 : 1;
    if (ia !== ib) return ia - ib;
    return (a.properties.confidence || 0) - (b.properties.confidence || 0);
  });
}

function nextToReview() {
  var q = state.parcels.features.filter(function (f) {
    return f.properties.status === 'draft' && state.selected.indexOf(f.properties.id) < 0;
  }).sort(function (a, b) { return (a.properties.confidence || 0) - (b.properties.confidence || 0); });
  if (!q.length) { toast('Every parcel has been reviewed.'); return; }
  state.selected = [q[0].properties.id];
  zoomTo(q[0]);
  restyle();
  renderTabs();
}

// ---------------------------------------------------------------- tools
function setTool(tool) {
  if (state.editingLayer) finishEdit(false);
  map.pm.disableDraw();
  if (tool === 'edit') {
    if (state.selected.length !== 1) { toast('Select exactly one parcel to edit its boundary.'); tool = 'select'; }
    else startEdit(state.selected[0]);
  } else if (tool === 'split') {
    if (state.selected.length !== 1) { toast('Select one parcel, then press Split and draw a line across it.'); tool = 'select'; }
    else { map.pm.enableDraw('Line', {snappable: true, snapDistance: 12}); toast('Draw a line across parcel #' + state.selected[0] + ' (click to add points, click the last point again to finish).'); }
  } else if (tool === 'draw') {
    map.pm.enableDraw('Polygon', {snappable: true, snapDistance: 12});
  } else if (tool === 'merge') {
    if (state.selected.length >= 2) { doMerge(); tool = 'select'; }
    else toast('Shift-click (or click in Merge mode) two or more touching parcels, then press Merge again.');
  } else if (tool === 'delete') {
    doDelete();
    tool = 'select';
  }
  state.tool = tool;
  document.querySelectorAll('.map-toolbar button').forEach(function (b) {
    b.classList.toggle('active', b.dataset.tool === tool);
  });
}

function startEdit(id) {
  var target = null;
  parcelLayer.eachLayer(function (l) { if (l.feature.properties.id === id) target = l; });
  if (!target) return;
  state.editingLayer = target;
  target.pm.enable({snappable: true, allowSelfIntersection: false});
  zoomTo(target.feature);
  toast('Drag vertices to adjust the boundary. Press Enter or click "Edit boundary" again to save, Esc to cancel.');
}

function finishEdit(commit) {
  var layer = state.editingLayer;
  state.editingLayer = null;
  if (!layer) return;
  layer.pm.disable();
  if (commit) {
    var f = parcelById(layer.feature.properties.id);
    f.geometry = layer.toGeoJSON().geometry;
    f.properties.source = 'edited';
    f.properties.status = 'draft';
    save('boundary edited for parcel ' + f.properties.id);
  } else {
    buildParcelLayer();
  }
}

// planar area of a lon/lat ring in square metres (fine at parcel scale)
function ringAreaM2(coords) {
  var lat0 = coords[0][1] * Math.PI / 180, kx = 111320 * Math.cos(lat0), ky = 110574, a = 0;
  for (var i = 0, j = coords.length - 1; i < coords.length; j = i++) {
    a += (coords[j][0] * kx) * (coords[i][1] * ky) - (coords[i][0] * kx) * (coords[j][1] * ky);
  }
  return Math.abs(a / 2);
}

map.on('pm:create', function (e) {
  var gj = e.layer.toGeoJSON();
  map.removeLayer(e.layer);
  map.pm.disableDraw();
  if (state.tool === 'split') {
    var pid = state.selected[0];
    setTool('select');
    setSave('saving', 'Splitting…');
    api('POST', '/api/surveys/' + encodeURIComponent(state.sid) + '/split', {id: pid, line: gj.geometry.coordinates})
      .then(function (res) {
        state.selected = res.ids;
        applyServerResult(res);
        setSave('idle', 'Saved');
        toast('Parcel #' + pid + ' split into ' + res.ids.map(function (i) { return '#' + i; }).join(' and ') + '.');
      })
      .catch(function (err) { setSave('error', 'Split failed'); toast(err.message, true); });
    return;
  }
  var area = ringAreaM2(gj.geometry.coordinates[0]);
  if (area < MIN_DRAWN_M2) {
    setTool('select');
    toast('That shape is only ' + area.toFixed(1) + ' m² — too small to be a parcel, so it was not added.', true);
    return;
  }
  var nextId = state.parcels.features.reduce(function (m, f) { return Math.max(m, f.properties.id); }, 0) + 1;
  gj.properties = {id: nextId, prov_pin: state.sid.toUpperCase() + '-' + String(nextId).padStart(4, '0'),
    status: 'draft', source: 'manual', confidence: null, landcover: null, issues: []};
  state.parcels.features.push(gj);
  state.selected = [nextId];
  setTool('select');
  save('drew new parcel ' + nextId);
});

function doMerge() {
  var ids = state.selected.slice();
  setSave('saving', 'Merging…');
  api('POST', '/api/surveys/' + encodeURIComponent(state.sid) + '/merge', {ids: ids})
    .then(function (res) {
      state.selected = [];
      applyServerResult(res);
      setSave('idle', 'Saved');
      toast('Merged ' + ids.length + ' parcels.');
    })
    .catch(function (e) { setSave('error', 'Merge failed'); toast(e.message, true); });
}

function doDelete() {
  if (!state.selected.length) { toast('Select the parcels to delete first.'); return; }
  if (!confirm('Delete ' + state.selected.length + ' parcel(s)? This is recorded in the survey edit log.')) return;
  var ids = state.selected.slice();
  state.parcels.features = state.parcels.features.filter(function (f) { return ids.indexOf(f.properties.id) < 0; });
  state.selected = [];
  save('deleted parcels ' + ids.join(','));
}

function doAutoFix() {
  setSave('saving', 'Auto-fixing…');
  api('POST', '/api/surveys/' + encodeURIComponent(state.sid) + '/autofix')
    .then(function (res) {
      applyServerResult(res);
      state.lastFixLog = res.log;
      setSave('idle', 'Saved');
      toast(res.log.length ? 'Auto-fix applied ' + res.log.length + ' change(s).' : 'Nothing to auto-fix.');
      renderTabs();
    })
    .catch(function (e) { setSave('error', 'Auto-fix failed'); toast(e.message, true); });
}

// ---------------------------------------------------------------- side panel
function renderTabs() {
  document.querySelectorAll('.tabs button').forEach(function (b) { b.classList.toggle('active', b.dataset.tab === state.tab); });
  var draft = state.parcels.features.filter(function (f) { return f.properties.status === 'draft'; }).length;
  var nIssues = state.issues.features.length;
  $('reviewCount').textContent = draft;
  $('issueCount').textContent = nIssues;
  $('issueCount').classList.toggle('warn', nIssues > 0);
  var body = $('tabBody');
  var renderer = TAB_RENDERERS[state.tab] || overviewHtml;
  body.innerHTML = renderer();
  wireTab(body);
  if (TAB_WIRERS[state.tab]) TAB_WIRERS[state.tab](body);
  if (state.tab === 'review' && state.selected.length === 1 && $('parcelExtras')) {
    document.dispatchEvent(new CustomEvent('wb:parcel-detail', {detail: {id: state.selected[0], el: $('parcelExtras')}}));
  }
}

function overviewHtml() {
  var m = state.meta, s = m.stats || {}, feats = state.parcels.features;
  var counts = {draft: 0, approved: 0, field_check: 0, rejected: 0};
  feats.forEach(function (f) { counts[f.properties.status] = (counts[f.properties.status] || 0) + 1; });
  var total = feats.length || 1;
  var lowConf = feats.filter(function (f) { return f.properties.confidence != null && f.properties.confidence < 0.5; }).length;
  var nB = state.buildings.features.length, nF = nFilled();
  var lcf = s.land_cover_fraction, cf = s.class_fraction || {};
  var done = total - counts.draft;

  var html = '<div><h3 class="panel-title">' + esc(m.name) + '</h3>' +
    '<div class="how">The steps along the top of the map follow the pipeline: <b>image → buildings → land cover → parcels → review</b>. Click a card below to jump to its step.</div></div>';

  html += '<div><h3 class="panel-title">What the AI found</h3><div class="found">' +
    foundCard('buildings', 'border:2px solid ' + PALETTE.building, nB, 'buildings',
      nF ? (nB - nF) + ' from the roof model · ' + nF + ' filled in from land cover (check)' : 'outlined by the AI, one per house') +
    (lcf ? foundCard('landcover', 'background:conic-gradient(' + PALETTE.building + ' 0 33%,' + PALETTE.tree + ' 0 66%,' + PALETTE.road + ' 0)', '8', 'land cover classes',
      'mostly ' + topLandCover(3)) : '') +
    foundCard('parcels', 'background:' + PALETTE.open + ';border:1px solid #F2E9DC', feats.length, 'parcels',
      'house + wall + courtyard as one numbered plot') +
    foundCard('parcels', 'background:rgba(232,232,232,0.3);border:1px dashed ' + PALETTE.road, fmt(s.corridor_length_m, 0) + ' m', 'roads & lanes',
      fmt(s.narrow_lane_length_m, 0) + ' m narrower than 3 m') +
    (s.encroachments ? '<button class="found-card" data-act="encroach">' + sw('background:' + ENCROACH_COLOUR) +
      '<span class="fc-main"><b>' + s.encroachments + '</b> possible encroachments<small>buildings extending onto a road or lane: check on site</small></span><span class="fc-go">→</span></button>' : '') +
    foundCard('review', 'background:' + STATUS.approved.color, Math.round(100 * done / total) + '%', 'reviewed',
      lowConf ? lowConf + ' low-confidence parcels to check first' : 'no low-confidence parcels') +
    '</div><div class="btn-row" style="margin-top:10px"><button class="btn primary" data-act="next">Start reviewing, least certain first</button></div></div>';

  // speed and effort, measured on this survey (PS: faster surveys, less manual digitising)
  var pr = m.processing || {};
  var appr = feats.filter(function (f) { return f.properties.status === 'approved'; });
  var asIs = appr.filter(function (f) { return f.properties.source === 'ai'; }).length;
  var recd = feats.filter(function (f) { return f.properties.record_status; });
  var recOk = recd.filter(function (f) { return f.properties.record_status === 'MATCH'; }).length;
  html += '<div><h3 class="panel-title">Speed &amp; accuracy</h3><div class="stat-grid">' +
    (pr.area_km2 ? tile(fmt(pr.area_km2, 2) + ' km²', 'Area processed') : '') +
    (pr.model_seconds ? tile(fmt(pr.model_seconds / 60, 1) + ' min', 'AI models (Kaggle GPU, incl. upload)') : '') +
    (pr.build_seconds ? tile(fmt(pr.build_seconds, 0) + ' s', 'Buildings, roads, parcels, topology') : '') +
    tile(appr.length ? Math.round(100 * asIs / appr.length) + '%' : '—', 'Approved parcels accepted with no edit') +
    (recd.length ? tile(Math.round(100 * recOk / recd.length) + '%', 'Parcels matching the existing record') : '') +
    '</div><div class="note" style="margin-top:6px">' +
    (appr.length ? asIs + ' of ' + appr.length + ' approved parcels were accepted exactly as the AI drew them; the rest needed an edit. ' : 'As parcels are approved, this shows how many the AI got right with no manual digitising. ') +
    '<b>Tested model accuracy</b> (held-out data): ' + esc((m.model && m.model.summary) || '—') + '</div></div>';

  var luc = s.land_use_count;
  if (luc) {
    var nlu = Object.keys(luc).reduce(function (a, k) { return a + luc[k]; }, 0) || 1;
    html += '<div><h3 class="panel-title">Land use <span class="note">suggested, per parcel</span></h3>' +
      Object.keys(LANDUSE).filter(function (k) { return luc[k]; }).map(function (k) {
        return '<div class="class-row"><span>' + sw('background:' + LANDUSE[k]) + esc(k) + '</span><div class="bar"><span style="width:' + (100 * luc[k] / nlu) + '%;background:' + LANDUSE[k] + '"></span></div><span class="pct">' + luc[k] + '</span></div>';
      }).join('') +
      '<div class="note" style="margin-top:6px">From buildings (count, size, share of the plot), storeys from the DSM when there is one, the width of the road in front and land cover. Each parcel shows its reason; the surveyor confirms it.</div></div>';
  }
  html += '<div><h3 class="panel-title">' + (lcf ? 'Land cover' : 'AI classes') + '</h3>' +
    (lcf ? LC8.map(function (c) {
      var v = lcf[c[0]] || 0;
      return '<div class="class-row"><span>' + sw('background:' + c[2]) + esc(c[1]) + '</span><div class="bar"><span style="width:' + (100 * v) + '%;background:' + c[2] + '"></span></div><span class="pct">' + Math.round(100 * v) + '%</span></div>';
    }).join('') : CLASS_META.map(function (c) {
      var v = cf[c[0]] || 0;
      return '<div class="class-row"><span>' + sw('background:' + c[2]) + esc(c[1]) + '</span><div class="bar"><span style="width:' + (100 * v) + '%;background:' + c[2] + '"></span></div><span class="pct">' + Math.round(100 * v) + '%</span></div>';
    }).join('')) + '</div>';

  html += '<div><h3 class="panel-title">Review progress <span class="note">' + done + ' of ' + feats.length + '</span></h3>' +
    '<div class="progress">' + ['approved', 'field_check', 'rejected'].map(function (k) {
      return '<span style="width:' + (100 * counts[k] / total) + '%;background:' + STATUS[k].color + '"></span>';
    }).join('') + '</div>' +
    '<div class="btn-row" style="margin-top:8px;gap:6px">' + Object.keys(STATUS).map(function (k) {
      return '<span class="tag"><span class="status-dot" style="display:inline-block;background:' + STATUS[k].color + ';margin-right:5px;width:7px;height:7px"></span>' + STATUS[k].label + ' ' + counts[k] + '</span>';
    }).join('') + '</div></div>';

  html += '<details class="about"><summary>About this survey and the models</summary>' +
    '<div class="note" style="margin-top:8px">' + esc(m.source) + '</div>' +
    '<div class="btn-row" style="margin-top:8px;gap:6px">' +
    '<span class="tag">' + esc(m.crs) + '</span><span class="tag">' + fmt(m.gsd_m * 100, 0) + ' cm / pixel</span>' +
    (m.used_height ? '<span class="tag ok">Image + height (DSM/DTM)</span>' : '<span class="tag warn">Image only, no height</span>') + '</div>' +
    '<div class="note" style="margin-top:8px"><b>Models:</b> ' + esc((m.model && m.model.name) || '—') +
    (m.model && m.model.summary ? '<br>' + esc(m.model.summary) : '') +
    (m.model && m.model.limits ? '<br><b>Known limits:</b> ' + esc(m.model.limits) : '') + '</div></details>';
  return html;
}

function foundCard(view, swatch, num, what, sub) {
  return '<button class="found-card" data-view="' + view + '">' + sw(swatch) +
    '<span class="fc-main"><b>' + esc(num) + '</b> ' + esc(what) + '<small>' + esc(sub) + '</small></span><span class="fc-go">→</span></button>';
}

function tile(num, label) {
  return '<div class="stat-tile"><div class="num">' + esc(num) + '</div><div class="lbl">' + esc(label) + '</div></div>';
}

function reviewHtml() {
  var html = '';
  if (state.selected.length === 1) {
    var f = parcelById(state.selected[0]);
    if (f) html += parcelDetailHtml(f);
  } else if (state.selected.length > 1) {
    html += '<div><h3 class="panel-title">' + state.selected.length + ' parcels selected</h3>' +
      '<div class="btn-row"><button class="btn primary" data-act="merge">Merge into one</button><button class="btn" data-act="clear">Clear</button></div>' +
      statusButtons() + '</div>';
  }
  var filters = [['draft', 'To review'], ['encroach', 'Encroachment'], ['field_check', 'Field check'], ['approved', 'Approved'], ['rejected', 'Rejected'], ['all', 'All']];
  var q = reviewQueue();
  html += '<div><h3 class="panel-title">Review queue <span class="note">issues first, then lowest confidence</span></h3>' +
    '<div class="filter-row">' + filters.map(function (fl) {
      return '<button data-filter="' + fl[0] + '" class="' + (state.reviewFilter === fl[0] ? 'active' : '') + '">' + fl[1] + '</button>';
    }).join('') + '</div>' +
    '<div class="list" style="margin-top:10px">' + (q.length ? q.slice(0, 200).map(function (f) {
      var p = f.properties, sel = state.selected.indexOf(p.id) >= 0;
      return '<div class="row' + (sel ? ' sel' : '') + '" data-pid="' + p.id + '">' +
        '<span class="status-dot" style="background:' + (STATUS[p.status] || STATUS.draft).color + '"></span>' +
        '<div><div class="pin">#' + p.id + ' · ' + fmt(p.area_m2, 0) + ' m²</div><div class="sub">' + esc(p.landcover || 'manual parcel') +
        ((p.issues || []).length ? ' · <span style="color:var(--critical)">' + p.issues.map(function (t) { return ISSUE_LABEL[t] || t; }).join(', ') + '</span>' : '') + '</div></div>' +
        '<div class="right" style="color:' + confColour(p.confidence) + '">' + (p.confidence == null ? '—' : p.confidence.toFixed(2)) + '</div></div>';
    }).join('') : '<div class="note">Nothing here.</div>') + '</div></div>';
  return html;
}

function statusButtons() {
  return '<div class="btn-row" style="margin-top:10px">' +
    '<button class="btn ok" data-status="approved">Approve<kbd>A</kbd></button>' +
    '<button class="btn warn" data-status="field_check">Field check<kbd>F</kbd></button>' +
    '<button class="btn bad" data-status="rejected">Reject<kbd>R</kbd></button></div>';
}

function parcelDetailHtml(f) {
  var p = f.properties, st = STATUS[p.status] || STATUS.draft;
  var conf = p.confidence;
  var issues = state.issues.features.filter(function (i) { return i.properties.parcel_ids.indexOf(p.id) >= 0; });
  return '<div><h3 class="panel-title"><span>Parcel #' + p.id + '</span><span class="back" data-act="next">Next to review →<kbd style="margin-left:4px">N</kbd></span></h3>' +
    '<div class="btn-row" style="gap:6px"><span class="tag" style="color:' + st.color + ';border-color:' + st.color + '">' + st.label + '</span>' +
    '<span class="tag">' + esc(p.prov_pin) + '</span><span class="tag">' + (p.source === 'ai' ? 'AI extracted' : p.source === 'manual' ? 'Drawn manually' : 'Edited') + '</span></div>' +
    '<div class="kv-grid">' +
    kv('Area', fmt(p.area_m2) + ' m²') + kv('Perimeter', fmt(p.perimeter_m) + ' m') +
    (p.land_use ? kv('Land use (suggested)', p.land_use) + kv('Why', p.land_use_reason || '—') : '') +
    (p.buildings != null ? kv('Buildings on plot', p.buildings) : '') +
    (p.storeys ? kv('Storeys (from DSM)', p.storeys + ' · ' + fmt(p.height_m) + ' m') : '') +
    (p.frontage_road_m ? kv('Road in front', fmt(p.frontage_road_m, 0) + ' m wide') : '') +
    kv('Land cover', p.landcover || '—') + kv('Built-up', p.built_pct == null ? '—' : fmt(p.built_pct, 0) + '%') +
    kv('Road frontage', p.road_frontage == null ? '—' : (p.road_frontage ? 'Yes' : 'No — check access')) +
    (p.layout ? kv('Laid out as', p.layout) : '') +
    kv('Vegetation', p.veg_pct == null ? '—' : fmt(p.veg_pct, 0) + '%') +
    '</div>' + (p.encroachment_m2 ? '<div class="finding sev-critical" style="margin-top:10px"><div><b>Possible encroachment</b><p>A building on this plot extends ' + fmt(p.encroachment_m2) + ' m² onto the road or lane. Check on site (Field check) before approving.</p></div></div>' : '') + landCoverBreakdown(p) +
    '<div style="margin-top:12px"><div class="class-row" style="grid-template-columns:110px 1fr 44px"><span>AI confidence</span>' +
    '<div class="bar"><span style="width:' + (100 * (conf || 0)) + '%;background:' + confColour(conf) + '"></span></div><span class="pct">' + (conf == null ? '—' : conf.toFixed(2)) + '</span></div>' +
    '<div class="note">' + confidenceNote(p) + '</div></div>' +
    (issues.length ? '<div style="margin-top:12px;display:flex;flex-direction:column;gap:6px">' + issues.map(issueHtml).join('') + '</div>' : '') +
    statusButtons() +
    '<div class="btn-row" style="margin-top:8px"><button class="btn" data-act="edit">Edit boundary</button><button class="btn" data-act="zoom">Zoom to</button></div>' +
    '<div id="parcelExtras"></div></div>';
}

function landCoverBreakdown(p) {
  var lc = p.land_cover_pct;
  if (!lc) return '';
  var rows = LC8.filter(function (c) { return lc[c[0]]; }).sort(function (a, b) { return lc[b[0]] - lc[a[0]]; });
  return '<div style="margin-top:10px"><div class="sub" style="margin-bottom:4px">Land cover inside this parcel</div>' +
    rows.map(function (c) {
      return '<div class="class-row"><span>' + sw('background:' + c[2]) + esc(c[1]) + '</span>' +
        '<div class="bar"><span style="width:' + lc[c[0]] + '%;background:' + c[2] + '"></span></div>' +
        '<span class="pct">' + fmt(lc[c[0]], 0) + '%</span></div>';
    }).join('') + '</div>';
}

function confidenceNote(p) {
  if (p.confidence == null) return 'Drawn or edited by a surveyor — no AI confidence applies.';
  if ((p.confidence_notes || []).length) return 'Marked down for: <b>' + p.confidence_notes.map(esc).join(', ') + '</b>.';
  return 'Segmentation is certain, the boundary follows a visible edge and the shape is plausible.';
}

function kv(k, v) { return '<div class="kv"><div class="k">' + esc(k) + '</div><div class="v">' + esc(v) + '</div></div>'; }

function issueHtml(i) {
  var p = i.properties;
  return '<div class="issue ' + esc(p.severity) + '" data-issue="' + p.n + '"><b>' + esc(ISSUE_LABEL[p.type] || p.type) + '</b>' +
    ' <span class="note">parcel' + (p.parcel_ids.length > 1 ? 's ' : ' ') + p.parcel_ids.map(function (x) { return '#' + x; }).join(', ') + '</span>' +
    '<p>' + esc(p.message) + '</p><div class="fix">Fix: ' + esc(p.suggested_fix) + '</div></div>';
}

function issuesHtml() {
  var feats = state.issues.features;
  var byType = {};
  feats.forEach(function (f) { byType[f.properties.type] = (byType[f.properties.type] || 0) + 1; });
  var html = '<div><h3 class="panel-title">Topology validation</h3>' +
    '<div class="note">Every save re-checks the whole parcel layer in ' + esc(state.meta.crs) + ' (metres): invalid or multipart geometry, overlaps, duplicates, gaps between parcels, slivers and implausibly small plots. Road and lane corridors are allowed space, not gaps.</div></div>';
  if (!feats.length) {
    html += '<div class="clean-msg">Topology clean — no overlaps, gaps, slivers or invalid parcels.</div>';
  } else {
    html += '<div class="btn-row" style="gap:6px">' + Object.keys(byType).map(function (t) {
      return '<span class="tag warn">' + esc(ISSUE_LABEL[t] || t) + ' ' + byType[t] + '</span>';
    }).join('') + '</div>' +
      '<div class="btn-row"><button class="btn primary" data-act="autofix">Auto-fix all</button></div>' +
      '<div style="display:flex;flex-direction:column;gap:6px">' + feats.map(issueHtml).join('') + '</div>';
  }
  if (state.lastFixLog && state.lastFixLog.length) {
    html += '<div><h3 class="panel-title">Last auto-fix</h3><div class="log">' +
      state.lastFixLog.map(function (l) { return esc(l.action); }).join('<br>') + '</div></div>';
  }
  return html;
}

function exportHtml() {
  var feats = state.parcels.features, base = '/api/surveys/' + encodeURIComponent(state.sid) + '/export?fmt=';
  var approved = feats.filter(function (f) { return f.properties.status === 'approved'; }).length;
  var nIssues = state.issues.features.length;
  return '<div><h3 class="panel-title">GIS-ready export</h3>' +
    '<div class="note">GeoPackage and Shapefile contain three layers: <b>parcels</b> (provisional PIN, area, perimeter, land use, share of each land-cover class, built-up %, road frontage, AI confidence, review status, edit source, open topology issues), <b>buildings</b> (footprint, area, which model found it) and <b>roads</b> (roads &amp; lanes with median width and length). GeoJSON is parcels only.</div></div>' +
    (nIssues ? '<div class="finding sev-warning"><div><b>' + nIssues + ' topology issue(s) still open</b><p>Export is allowed, but fix or review them first for a clean cadastral layer.</p></div></div>' : '<div class="clean-msg">Topology clean.</div>') +
    '<div class="note">' + approved + ' of ' + feats.length + ' parcels approved.</div>' +
    exportCard('GeoPackage (.gpkg)', 'Parcels + buildings + roads · ' + esc(state.meta.crs) + ' · QGIS / ArcGIS / PostGIS', base + 'gpkg') +
    exportCard('Shapefile (.zip)', 'Parcels + buildings + roads · ' + esc(state.meta.crs) + ' · legacy land-record systems', base + 'shp') +
    exportCard('GeoJSON', 'EPSG:4326 · web maps & APIs', base + 'geojson') +
    '<div class="export-card"><b>Teach the parcel model</b><small>Save the approved parcels as training labels, so the next fine-tune of the parcel-boundary model learns from this survey (' + approved + ' approved)</small>' +
    '<button class="btn" data-act="trainlabels">Save</button></div>' +
    '<div class="export-card"><b>Survey report</b><small>One printable page: parcel map, land use, roads, encroachments, topology, review, records, models and accuracy</small>' +
    '<a class="btn primary" target="_blank" href="/api/surveys/' + encodeURIComponent(state.sid) + '/report">Open</a></div>';
}

function exportWire(body) {
  var btn = body.querySelector('[data-act="trainlabels"]');
  if (!btn) return;
  btn.addEventListener('click', function () {
    api('POST', '/api/surveys/' + encodeURIComponent(state.sid) + '/training-labels')
      .then(function (r) { toast(r.approved_parcels + ' approved parcels saved as training labels (' + r.file + ').'); })
      .catch(function (e) { toast(e.message, true); });
  });
}

function exportCard(title, sub, href) {
  return '<div class="export-card"><b>' + title + '</b><small>' + sub + '</small><a class="btn primary" href="' + href + '">Download</a></div>';
}

function wireTab(body) {
  body.querySelectorAll('[data-pid]').forEach(function (el) {
    el.addEventListener('click', function (e) {
      var id = Number(el.dataset.pid);
      selectParcel(id, e.shiftKey);
      zoomTo(parcelById(id));
    });
  });
  body.querySelectorAll('[data-filter]').forEach(function (el) {
    el.addEventListener('click', function () { state.reviewFilter = el.dataset.filter; renderTabs(); });
  });
  body.querySelectorAll('[data-status]').forEach(function (el) {
    el.addEventListener('click', function () { setStatus(el.dataset.status); });
  });
  body.querySelectorAll('[data-issue]').forEach(function (el) {
    el.addEventListener('click', function () {
      var f = state.issues.features.filter(function (i) { return i.properties.n === Number(el.dataset.issue); })[0];
      if (!f) return;
      zoomTo(f);
      state.selected = f.properties.parcel_ids.slice();
      restyle();
    });
  });
  body.querySelectorAll('.found-card').forEach(function (el) {
    el.addEventListener('click', function () { setView(el.dataset.view); });
  });
  body.querySelectorAll('[data-act]').forEach(function (el) {
    el.addEventListener('click', function () {
      var act = el.dataset.act;
      if (act === 'encroach') { state.tab = 'review'; state.reviewFilter = 'encroach'; setView('review'); renderTabs(); return; }
      if (act === 'next') { if (!state.on.parcels) setView('review'); state.tab = 'review'; state.reviewFilter = 'draft'; nextToReview(); }
      else if (act === 'autofix') doAutoFix();
      else if (act === 'merge') doMerge();
      else if (act === 'clear') { state.selected = []; restyle(); renderTabs(); }
      else if (act === 'edit') { if (state.editingLayer) { finishEdit(true); setTool('select'); } else setTool('edit'); }
      else if (act === 'zoom' && state.selected.length) zoomTo(parcelById(state.selected[0]));
    });
  });
}

// ---------------------------------------------------------------- extension points (records.js, ...)
var TAB_RENDERERS = {overview: overviewHtml, review: reviewHtml, issues: issuesHtml, export: exportHtml};
var TAB_WIRERS = {export: exportWire};
window.WB = {
  state: state, map: map, esc: esc, fmt: fmt, api: api, toast: toast, zoomTo: zoomTo,
  selectParcel: selectParcel, parcelById: parcelById, restyle: restyle, renderTabs: renderTabs,
  buildParcelLayer: buildParcelLayer, setSave: setSave, RECORD: RECORD, applyServerResult: applyServerResult,
  loadSurveys: loadSurveys,
  layersControl: function () { return layersControl; },
  registerTab: function (name, render, wire) { TAB_RENDERERS[name] = render; if (wire) TAB_WIRERS[name] = wire; },
  reloadParcels: function () {
    return fetch('/surveys/' + encodeURIComponent(state.sid) + '/parcels.geojson', {cache: 'no-store'})
      .then(function (r) { return r.json(); })
      .then(function (fc) { state.parcels = fc; buildParcelLayer(); restyle(); });
  }
};

// ---------------------------------------------------------------- wiring
document.querySelectorAll('.tabs button').forEach(function (b) {
  b.addEventListener('click', function () { state.tab = b.dataset.tab; renderTabs(); });
});
document.querySelectorAll('.map-toolbar button').forEach(function (b) {
  b.addEventListener('click', function () {
    if (b.dataset.tool === 'edit' && state.editingLayer) { finishEdit(true); setTool('select'); return; }
    setTool(b.dataset.tool);
  });
});
$('surveySelect').addEventListener('change', function (e) { openSurvey(e.target.value); });
map.on('click', function () {
  if (state.tool === 'select' && state.selected.length && !state.editingLayer) { state.selected = []; restyle(); renderTabs(); }
});
document.addEventListener('keydown', function (e) {
  if (e.target && e.target.matches && e.target.matches('input, select, textarea')) return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (state.editingLayer) {
    // stop Enter from also "clicking" whichever button still has focus
    if (e.key === 'Enter') { e.preventDefault(); finishEdit(true); setTool('select'); }
    if (e.key === 'Escape') { e.preventDefault(); finishEdit(false); setTool('select'); }
    return;
  }
  if (e.key === 'Escape') { map.pm.disableDraw(); setTool('select'); return; }
  var k = e.key.toLowerCase();
  if (k === 'a') setStatus('approved');
  else if (k === 'f') setStatus('field_check');
  else if (k === 'r') setStatus('rejected');
  else if (k === 'n') { state.tab = 'review'; nextToReview(); }
});
window.addEventListener('resize', function () { map.invalidateSize(); });

loadSurveys().catch(function (e) { $('loadingScreen').textContent = 'Failed to load surveys: ' + e.message; });
})();
