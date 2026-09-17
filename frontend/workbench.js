// CadastraAI Survey Workbench: review, edit, validate and export AI-extracted parcels.
(function () {
"use strict";

var STATUS = {
  draft:       {label: 'To review',   color: '#9FB3A8'},
  approved:    {label: 'Approved',    color: '#6CBE81'},
  field_check: {label: 'Field check', color: '#E0A94A'},
  rejected:    {label: 'Rejected',    color: '#E2685F'}
};
var LANDCOVER = {
  'Built-up': '#7C9BA6', 'Vegetated open land': '#5FA37A', 'Open / vacant land': '#B08D6B'
};
var CLASS_META = [
  ['building', 'Building', '#4666E6'], ['road', 'Road / paved', '#E1E1E1'],
  ['low_veg', 'Low vegetation', '#82D7C8'], ['tree', 'Tree', '#28A046'], ['clutter', 'Other / clutter', '#C85A5A']
];
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
  selected: [], tab: 'overview', colourMode: 'status', tool: 'select',
  reviewFilter: 'draft', editingLayer: null
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

var overlays = {};
var layersControl = null;
var parcelLayer = null, issueLayer = null, buildingLayer = null, corridorLayer = null, labelLayer = null;

function clearMap() {
  [parcelLayer, issueLayer, buildingLayer, corridorLayer, labelLayer].forEach(function (l) { if (l) map.removeLayer(l); });
  Object.keys(overlays).forEach(function (k) { map.removeLayer(overlays[k]); });
  overlays = {};
  parcelLayer = issueLayer = buildingLayer = corridorLayer = labelLayer = null;
  if (layersControl) layersControl.remove();
  layersControl = null;
}

function parcelStyle(f) {
  var p = f.properties;
  var sel = state.selected.indexOf(p.id) >= 0;
  var fill;
  if (state.colourMode === 'confidence') fill = confColour(p.confidence);
  else if (state.colourMode === 'landcover') fill = LANDCOVER[p.landcover] || '#5C6B63';
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
  }).addTo(map);
  labelLayer = L.layerGroup();
  state.parcels.features.forEach(function (f) {
    var c = L.geoJSON(f).getBounds().getCenter();
    labelLayer.addLayer(L.tooltip({permanent: true, direction: 'center', className: 'parcel-label', interactive: false})
      .setLatLng(c).setContent(String(f.properties.id)));
  });
  updateLabelVisibility();
  if (layersControl) {
    if (overlays.parcels) layersControl.removeLayer(overlays.parcels);
    layersControl.addOverlay(parcelLayer, 'Parcels');
  }
  overlays.parcels = parcelLayer;
}

function updateLabelVisibility() {
  if (!labelLayer) return;
  if (map.getZoom() >= 19.5 && map.hasLayer(parcelLayer)) map.addLayer(labelLayer);
  else map.removeLayer(labelLayer);
}
map.on('zoomend overlayadd overlayremove', updateLabelVisibility);

function buildIssueLayer() {
  if (issueLayer) map.removeLayer(issueLayer);
  issueLayer = L.geoJSON(state.issues, {
    interactive: false,
    style: function (f) {
      var high = f.properties.severity === 'high';
      return {color: high ? '#E2685F' : '#E0A94A', weight: 2, dashArray: '4 3', fillColor: high ? '#E2685F' : '#E0A94A', fillOpacity: 0.45};
    },
    pointToLayer: function (f, ll) { return L.circleMarker(ll, {radius: 6}); }
  }).addTo(map);
  if (layersControl && overlays.issues) {
    layersControl.removeLayer(overlays.issues);
    layersControl.addOverlay(issueLayer, 'Topology issues');
  }
  overlays.issues = issueLayer;
}

function restyle() { if (parcelLayer) parcelLayer.setStyle(parcelStyle); renderLegend(); }

function renderLegend() {
  var html = '';
  if (state.colourMode === 'confidence') {
    html = '<b>AI confidence</b><div class="ramp"></div><div class="ramp-lbl"><span>low</span><span>0.5</span><span>0.7</span><span>high</span></div>';
  } else if (state.colourMode === 'record') {
    html = '<b>Existing record</b>' + Object.keys(RECORD).map(function (k) {
      return '<div><span class="sw" style="background:' + RECORD[k].color + '"></span>' + RECORD[k].label + '</div>';
    }).join('') + '<div><span class="sw" style="background:#5C6B63"></span>Not compared yet</div>';
  } else if (state.colourMode === 'landcover') {
    html = '<b>Land cover</b>' + Object.keys(LANDCOVER).map(function (k) {
      return '<div><span class="sw" style="background:' + LANDCOVER[k] + '"></span>' + esc(k) + '</div>';
    }).join('');
  } else {
    html = '<b>Review status</b>' + Object.keys(STATUS).map(function (k) {
      return '<div><span class="sw" style="background:' + STATUS[k].color + '"></span>' + STATUS[k].label + '</div>';
    }).join('');
  }
  html += '<div style="margin-top:4px"><span class="sw" style="background:transparent;border:2px solid #E2685F"></span>Has topology issue</div>';
  $('mapLegend').innerHTML = html;
}

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
    fetch(base + 'corridors.geojson').then(function (r) { return r.json(); })
  ]).then(function (res) {
    clearMap();
    state.sid = sid;
    state.meta = res[0]; state.parcels = res[1]; state.issues = res[2]; state.buildings = res[3]; state.corridors = res[4];
    state.selected = [];
    history.replaceState(null, '', '?survey=' + encodeURIComponent(sid));
    $('surveySubtitle').textContent = 'Survey Workbench · ' + state.meta.name;

    var b = state.meta.bounds;
    var tiles = state.meta.tiles;
    overlays.ori = (tiles && tiles.count
      ? L.tileLayer(base + 'tiles/{z}/{x}/{y}.webp', {
          minNativeZoom: tiles.min_zoom, maxNativeZoom: tiles.max_zoom, maxZoom: 23,
          bounds: L.latLngBounds(b), attribution: esc(state.meta.source)})
      : L.imageOverlay(base + 'ori.webp', b, {attribution: esc(state.meta.source)})
    ).addTo(map);
    overlays.classes = L.imageOverlay(base + 'classes.png', b, {opacity: 0.55});
    if (state.meta.has_height_layer) overlays.height = L.imageOverlay(base + 'height.png', b, {opacity: 0.6});
    buildingLayer = L.geoJSON(state.buildings, {interactive: false, pmIgnore: true, style: {color: '#6E8BFF', weight: 1, fillOpacity: 0.12}});
    corridorLayer = L.geoJSON(state.corridors, {interactive: false, pmIgnore: true, style: {color: '#E7EEE9', weight: 1, dashArray: '3 3', fillColor: '#E7EEE9', fillOpacity: 0.18}});
    overlays.buildings = buildingLayer;
    overlays.corridors = corridorLayer;
    buildParcelLayer();
    buildIssueLayer();

    var ctl = {'Orthoimage (ORI)': overlays.ori, 'AI segmentation': overlays.classes};
    if (overlays.height) ctl['Height above ground (nDSM)'] = overlays.height;
    ctl['Access corridors'] = corridorLayer;
    ctl['Building footprints'] = buildingLayer;
    ctl['Parcels'] = parcelLayer;
    ctl['Topology issues'] = issueLayer;
    layersControl = L.control.layers(null, ctl, {position: 'bottomleft', collapsed: false}).addTo(map);

    map.invalidateSize();
    map.fitBounds(b);
    renderLegend();
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
  var cf = s.class_fraction || {};
  var html = '<div><h3 class="panel-title">' + esc(m.name) + '</h3>' +
    '<div class="note">' + esc(m.source) + '</div>' +
    '<div class="btn-row" style="margin-top:8px;gap:6px">' +
    '<span class="tag">' + esc(m.crs) + '</span><span class="tag">GSD ' + fmt(m.gsd_m * 100, 0) + ' cm</span>' +
    (m.used_height ? '<span class="tag ok">ORI + DSM/DTM</span>' : '<span class="tag warn">ORI only (no DSM)</span>') + '</div></div>';

  html += '<div class="stat-grid">' +
    tile(feats.length, 'Parcels delineated') + tile(state.buildings.features.length, 'Building footprints') +
    tile(fmt(s.corridor_length_m, 0) + ' m', 'Access corridors') + tile(fmt(s.narrow_lane_length_m, 0) + ' m', 'Narrow lanes (< 3 m)') +
    '</div>';

  html += '<div><h3 class="panel-title">Review progress <span class="note">' + Math.round(100 * (total - counts.draft) / total) + '% done</span></h3>' +
    '<div class="progress">' + ['approved', 'field_check', 'rejected'].map(function (k) {
      return '<span style="width:' + (100 * counts[k] / total) + '%;background:' + STATUS[k].color + '"></span>';
    }).join('') + '</div>' +
    '<div class="btn-row" style="margin-top:8px;gap:6px">' + Object.keys(STATUS).map(function (k) {
      return '<span class="tag"><span class="status-dot" style="display:inline-block;background:' + STATUS[k].color + ';margin-right:5px;width:7px;height:7px"></span>' + STATUS[k].label + ' ' + counts[k] + '</span>';
    }).join('') + '</div>' +
    '<div class="btn-row" style="margin-top:10px"><button class="btn primary" data-act="next">Start reviewing lowest-confidence parcels</button></div>' +
    (lowConf ? '<div class="note" style="margin-top:6px">' + lowConf + ' parcel(s) below 0.5 confidence — prioritise these for field verification.</div>' : '') +
    '</div>';

  html += '<div><h3 class="panel-title">AI segmentation</h3>' + CLASS_META.map(function (c) {
    var v = cf[c[0]] || 0;
    return '<div class="class-row"><span>' + c[1] + '</span><div class="bar"><span style="width:' + (100 * v) + '%;background:' + c[2] + '"></span></div><span class="pct">' + Math.round(100 * v) + '%</span></div>';
  }).join('') +
    '<div class="note" style="margin-top:8px"><b>Model:</b> ' + esc((m.model && m.model.name) || '—') +
    (m.model && m.model.summary ? '<br>' + esc(m.model.summary) : '') +
    (m.model && m.model.limits ? '<br><b>Known limits:</b> ' + esc(m.model.limits) : '') + '</div></div>';
  return html;
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
  var filters = [['draft', 'To review'], ['field_check', 'Field check'], ['approved', 'Approved'], ['rejected', 'Rejected'], ['all', 'All']];
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
    kv('Land cover', p.landcover || '—') + kv('Built-up', p.built_pct == null ? '—' : fmt(p.built_pct, 0) + '%') +
    kv('Road frontage', p.road_frontage == null ? '—' : (p.road_frontage ? 'Yes' : 'No — check access')) +
    kv('Vegetation', p.veg_pct == null ? '—' : fmt(p.veg_pct, 0) + '%') +
    '</div>' +
    '<div style="margin-top:12px"><div class="class-row" style="grid-template-columns:110px 1fr 44px"><span>AI confidence</span>' +
    '<div class="bar"><span style="width:' + (100 * (conf || 0)) + '%;background:' + confColour(conf) + '"></span></div><span class="pct">' + (conf == null ? '—' : conf.toFixed(2)) + '</span></div>' +
    '<div class="note">' + confidenceNote(p) + '</div></div>' +
    (issues.length ? '<div style="margin-top:12px;display:flex;flex-direction:column;gap:6px">' + issues.map(issueHtml).join('') + '</div>' : '') +
    statusButtons() +
    '<div class="btn-row" style="margin-top:8px"><button class="btn" data-act="edit">Edit boundary</button><button class="btn" data-act="zoom">Zoom to</button></div>' +
    '<div id="parcelExtras"></div></div>';
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
    '<div class="note">Exports the current parcel layer with attributes: provisional PIN, area, perimeter, land cover, built-up %, road frontage, AI confidence, review status, edit source and open topology issues.</div></div>' +
    (nIssues ? '<div class="finding sev-warning"><div><b>' + nIssues + ' topology issue(s) still open</b><p>Export is allowed, but fix or review them first for a clean cadastral layer.</p></div></div>' : '<div class="clean-msg">Topology clean.</div>') +
    '<div class="note">' + approved + ' of ' + feats.length + ' parcels approved.</div>' +
    exportCard('GeoPackage (.gpkg)', 'Projected ' + esc(state.meta.crs) + ' · QGIS / ArcGIS / PostGIS', base + 'gpkg') +
    exportCard('Shapefile (.zip)', 'Projected ' + esc(state.meta.crs) + ' · legacy land-record systems', base + 'shp') +
    exportCard('GeoJSON', 'EPSG:4326 · web maps & APIs', base + 'geojson');
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
  body.querySelectorAll('[data-act]').forEach(function (el) {
    el.addEventListener('click', function () {
      var act = el.dataset.act;
      if (act === 'next') { state.tab = 'review'; state.reviewFilter = 'draft'; nextToReview(); }
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
var TAB_WIRERS = {};
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
document.querySelectorAll('#colourMode button').forEach(function (b) {
  b.addEventListener('click', function () {
    state.colourMode = b.dataset.mode;
    document.querySelectorAll('#colourMode button').forEach(function (x) { x.classList.toggle('active', x === b); });
    restyle();
  });
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
