// Field verification: a phone-first view of the parcels flagged for a site visit.
(function () {
"use strict";
var $ = function (id) { return document.getElementById(id); };
var ON_SITE_M = 30;
var VERDICTS = [
  ['confirmed', 'Boundary confirmed on site'],
  ['boundary_wrong', 'Boundary is wrong'],
  ['part_of_neighbour', 'Part of a neighbour'],
  ['not_a_parcel', 'Not a real parcel'],
  ['could_not_access', 'Could not access']
];
var st = {sid: null, meta: null, parcels: null, me: null, selected: null, filter: 'field_check',
          verdict: null, photos: [], visits: [], view: 'queue'};

function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
    return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c];
  });
}
function toast(msg, err) {
  document.querySelectorAll('.toast').forEach(function (t) { t.remove(); });
  var t = document.createElement('div');
  t.className = 'toast' + (err ? ' error' : '');
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(function () { t.remove(); }, err ? 6000 : 2600);
}
function store(key, val) {
  try { if (val === undefined) return localStorage.getItem(key); localStorage.setItem(key, val); } catch (e) { return null; }
}

var map = L.map('map', {zoomControl: false, maxZoom: 23, zoomSnap: 0});
var baseLayer = null, parcelLayer = null, meMarker = null, meCircle = null;

// ---------------------------------------------------------------- GPS
function distanceM(lat1, lon1, lat2, lon2) {
  var R = 6371000, toR = Math.PI / 180;
  var dLat = (lat2 - lat1) * toR, dLon = (lon2 - lon1) * toR;
  var a = Math.sin(dLat / 2) * Math.sin(dLat / 2) + Math.cos(lat1 * toR) * Math.cos(lat2 * toR) * Math.sin(dLon / 2) * Math.sin(dLon / 2);
  return 2 * R * Math.asin(Math.sqrt(a));
}
function centroid(f) { var c = L.geoJSON(f).getBounds().getCenter(); return [c.lat, c.lng]; }
function distTo(f) { if (!st.me) return null; var c = centroid(f); return distanceM(st.me.lat, st.me.lon, c[0], c[1]); }
function fmtDist(d) { return d == null ? '' : d < 1000 ? Math.round(d) + ' m' : (d / 1000).toFixed(1) + ' km'; }

function startGps() {
  var chip = $('gpsChip');
  if (!('geolocation' in navigator)) { chip.textContent = 'GPS not available in this browser'; chip.className = 'gps-chip bad'; return; }
  if (!window.isSecureContext) {
    chip.textContent = 'GPS needs HTTPS — open via the https:// link'; chip.className = 'gps-chip bad'; return;
  }
  navigator.geolocation.watchPosition(function (pos) {
    st.me = {lat: pos.coords.latitude, lon: pos.coords.longitude, acc: pos.coords.accuracy};
    chip.textContent = 'GPS ±' + Math.round(st.me.acc) + ' m';
    chip.className = 'gps-chip ' + (st.me.acc <= 10 ? 'ok' : 'bad');
    var ll = [st.me.lat, st.me.lon];
    if (!meMarker) {
      meCircle = L.circle(ll, {radius: st.me.acc, color: '#4AA3FF', weight: 1, fillOpacity: 0.12, interactive: false}).addTo(map);
      meMarker = L.circleMarker(ll, {radius: 7, color: '#fff', weight: 2, fillColor: '#4AA3FF', fillOpacity: 1, interactive: false}).addTo(map);
    } else {
      meMarker.setLatLng(ll); meCircle.setLatLng(ll).setRadius(st.me.acc);
    }
    if (st.view === 'queue') renderQueue();
    else updateDistance();
  }, function (err) {
    chip.textContent = err.code === 1 ? 'GPS permission denied' : 'GPS unavailable';
    chip.className = 'gps-chip bad';
  }, {enableHighAccuracy: true, maximumAge: 5000, timeout: 20000});
}
$('locateBtn').addEventListener('click', function () {
  if (st.me) map.setView([st.me.lat, st.me.lon], Math.max(map.getZoom(), 19));
  else toast('No GPS fix yet.', true);
});

// ---------------------------------------------------------------- data
function loadSurveys() {
  return fetch('/api/surveys').then(function (r) { return r.json(); }).then(function (list) {
    $('surveySelect').innerHTML = list.map(function (s) { return '<option value="' + esc(s.id) + '">' + esc(s.name) + '</option>'; }).join('');
    if (!list.length) { $('sheetBody').innerHTML = '<div class="note">No surveys yet.</div>'; return; }
    var want = new URLSearchParams(location.search).get('survey') || store('cadai.field.survey');
    var sid = list.some(function (s) { return s.id === want; }) ? want : list[list.length - 1].id;
    $('surveySelect').value = sid;
    return openSurvey(sid);
  });
}
$('surveySelect').addEventListener('change', function (e) { openSurvey(e.target.value); });

function openSurvey(sid) {
  var base = '/surveys/' + encodeURIComponent(sid) + '/';
  return Promise.all([
    fetch(base + 'meta.json', {cache: 'no-store'}).then(function (r) { return r.json(); }),
    fetch(base + 'parcels.geojson', {cache: 'no-store'}).then(function (r) { return r.json(); })
  ]).then(function (res) {
    st.sid = sid; st.meta = res[0]; st.parcels = res[1]; st.selected = null;
    store('cadai.field.survey', sid);
    history.replaceState(null, '', '?survey=' + encodeURIComponent(sid));
    if (baseLayer) map.removeLayer(baseLayer);
    var t = st.meta.tiles, b = st.meta.bounds;
    baseLayer = (t && t.count
      ? L.tileLayer(base + 'tiles/{z}/{x}/{y}.webp', {minNativeZoom: t.min_zoom, maxNativeZoom: t.max_zoom, maxZoom: 23, bounds: L.latLngBounds(b)})
      : L.imageOverlay(base + 'ori.webp', b)).addTo(map);
    drawParcels();
    map.fitBounds(b);
    if (!st.parcels.features.some(function (f) { return f.properties.status === 'field_check'; })) st.filter = 'draft';
    showQueue();
  });
}

function parcelStyle(f) {
  var p = f.properties, sel = st.selected && st.selected.properties.id === p.id;
  var col = p.status === 'field_check' ? '#E0A94A' : p.status === 'approved' ? '#6CBE81' : p.status === 'rejected' ? '#E2685F' : '#F2E9DC';
  return {color: sel ? '#45C7B8' : col, weight: sel ? 4 : (p.status === 'field_check' ? 2.5 : 1.2),
          fillColor: col, fillOpacity: sel ? 0.3 : (p.status === 'field_check' ? 0.22 : 0.05)};
}
function drawParcels() {
  if (parcelLayer) map.removeLayer(parcelLayer);
  parcelLayer = L.geoJSON(st.parcels, {
    style: parcelStyle,
    onEachFeature: function (f, layer) {
      layer.on('click', function () { openParcel(f.properties.id, false); });
    }
  }).addTo(map);
}

// ---------------------------------------------------------------- queue
function queue() {
  return st.parcels.features.filter(function (f) {
    return st.filter === 'all' ? true : f.properties.status === st.filter;
  }).map(function (f) { return {f: f, d: distTo(f)}; }).sort(function (a, b) {
    if (a.d != null && b.d != null) return a.d - b.d;
    return (a.f.properties.confidence || 0) - (b.f.properties.confidence || 0);
  });
}

function showQueue() {
  st.view = 'queue'; st.selected = null;
  if (parcelLayer) parcelLayer.setStyle(parcelStyle);
  renderQueue();
  setTimeout(function () { map.invalidateSize(); }, 30);
}

function renderQueue() {
  var q = queue();
  var nField = st.parcels.features.filter(function (f) { return f.properties.status === 'field_check'; }).length;
  $('sheetTitle').textContent = 'Field checks';
  $('sheetSub').textContent = nField + ' flagged · ' + (st.me ? 'nearest first' : 'lowest confidence first');
  var filters = [['field_check', 'Flagged'], ['draft', 'To review'], ['all', 'All']];
  $('sheetBody').innerHTML =
    '<div class="filter-row">' + filters.map(function (x) {
      return '<button data-filter="' + x[0] + '" class="' + (st.filter === x[0] ? 'active' : '') + '">' + x[1] + '</button>';
    }).join('') + '</div>' +
    '<div class="list">' + (q.length ? q.slice(0, 150).map(function (x) {
      var p = x.f.properties;
      return '<div class="row" data-pid="' + p.id + '"><span class="status-dot" style="background:' + parcelStyle(x.f).color + '"></span>' +
        '<div><div class="pin">#' + p.id + ' · ' + Math.round(p.area_m2) + ' m²</div><div class="sub">' +
        esc((p.confidence_notes || []).join(', ') || p.landcover || 'manual parcel') + (p.field_visits ? ' · visited ' + p.field_visits + '×' : '') + '</div></div>' +
        '<div class="right">' + (x.d != null ? fmtDist(x.d) : (p.confidence != null ? p.confidence.toFixed(2) : '—')) + '</div></div>';
    }).join('') : '<div class="note">Nothing in this list.</div>') + '</div>';
  $('sheetBody').querySelectorAll('[data-filter]').forEach(function (b) {
    b.addEventListener('click', function () { st.filter = b.dataset.filter; renderQueue(); });
  });
  $('sheetBody').querySelectorAll('[data-pid]').forEach(function (r) {
    r.addEventListener('click', function () { openParcel(Number(r.dataset.pid), true); });
  });
}

// ---------------------------------------------------------------- parcel form
function openParcel(id, zoom) {
  var f = st.parcels.features.filter(function (x) { return x.properties.id === id; })[0];
  if (!f) return;
  st.view = 'parcel'; st.selected = f; st.verdict = null; st.photos = []; st.visits = [];
  $('sheet').classList.remove('collapsed');
  parcelLayer.setStyle(parcelStyle);
  renderParcel();
  // the sheet changes the map's height, so resize before framing the parcel
  setTimeout(function () {
    map.invalidateSize();
    if (zoom) map.fitBounds(L.geoJSON(f).getBounds(), {maxZoom: 21, padding: [30, 30]});
  }, 30);
  fetch('/api/surveys/' + encodeURIComponent(st.sid) + '/field?parcel_id=' + id).then(function (r) { return r.json(); })
    .then(function (j) { if (st.selected === f) { st.visits = j.observations; renderVisits(); } });
}

function updateDistance() {
  var el = $('distNow');
  if (!el || !st.selected) return;
  var d = distTo(st.selected);
  el.textContent = d == null ? 'no GPS fix' : fmtDist(d) + ' away' + (d > ON_SITE_M ? ' — will be recorded as an off-site check' : ' — on site');
  el.style.color = d == null ? 'var(--text-faint)' : d > ON_SITE_M ? 'var(--warning)' : 'var(--ok)';
}

function renderParcel() {
  var p = st.selected.properties, c = centroid(st.selected);
  $('sheetTitle').textContent = 'Parcel #' + p.id;
  $('sheetSub').textContent = Math.round(p.area_m2) + ' m² · ' + (p.status || 'draft').replace('_', ' ');
  var why = (p.confidence_notes || []).concat(p.issues || []);
  $('sheetBody').innerHTML =
    '<div class="btn-row" style="justify-content:space-between;align-items:center">' +
      '<span class="back" style="font-family:var(--font-mono);font-size:0.75rem;color:var(--accent);cursor:pointer" id="backBtn">← List</span>' +
      '<a class="tag" href="https://www.google.com/maps/dir/?api=1&destination=' + c[0].toFixed(6) + ',' + c[1].toFixed(6) + '" target="_blank" rel="noopener">Directions ↗</a></div>' +
    '<div class="dist" id="distNow"></div>' +
    (why.length ? '<div class="note">Flagged for: <b>' + why.map(esc).join(', ') + '</b></div>' : '') +
    '<div class="visits" id="visits"></div>' +
    '<div class="verdicts" id="verdicts">' + VERDICTS.map(function (v) {
      return '<button data-v="' + v[0] + '">' + v[1] + '</button>';
    }).join('') + '</div>' +
    '<textarea id="note" placeholder="What did you see? e.g. compound wall on north side, gate on lane"></textarea>' +
    '<div class="photo-row" id="photoRow"><label class="filebtn">📷 Add photo<input type="file" accept="image/*" capture="environment" id="photoInput" multiple></label></div>' +
    '<input type="text" id="surveyor" placeholder="Surveyor name" value="' + esc(store('cadai.field.surveyor') || '') + '">' +
    '<div class="btn-row"><button class="btn" id="cornerBtn">Record boundary corner here</button></div>' +
    '<button class="btn primary big" id="submitBtn">Save field check</button>';
  updateDistance();
  $('backBtn').addEventListener('click', showQueue);
  $('verdicts').querySelectorAll('button').forEach(function (b) {
    b.addEventListener('click', function () {
      st.verdict = b.dataset.v;
      $('verdicts').querySelectorAll('button').forEach(function (x) { x.classList.toggle('on', x === b); });
    });
  });
  $('photoInput').addEventListener('change', function (e) {
    Array.prototype.forEach.call(e.target.files, function (file) {
      if (st.photos.length >= 4) return;
      st.photos.push(file);
      var img = document.createElement('img');
      img.src = URL.createObjectURL(file);
      img.alt = 'Photo ' + st.photos.length;
      $('photoRow').insertBefore(img, $('photoRow').lastElementChild);
    });
    e.target.value = '';
  });
  $('cornerBtn').addEventListener('click', recordCorner);
  $('submitBtn').addEventListener('click', submit);
  renderVisits();
}

function renderVisits() {
  var el = $('visits');
  if (!el) return;
  el.innerHTML = st.visits.slice().reverse().map(function (o) {
    return '<div class="visit"><b>' + esc(o.verdict_label) + '</b> · ' + esc(o.time) + (o.surveyor ? ' · ' + esc(o.surveyor) : '') +
      (o.distance_to_parcel_m != null ? ' · ' + (o.on_site ? 'on site' : fmtDist(o.distance_to_parcel_m) + ' away') : ' · no GPS') +
      (o.note ? '<br>' + esc(o.note) : '') + '</div>';
  }).join('');
}

function gpsFields(fd) {
  if (!st.me) return;
  fd.append('lat', st.me.lat); fd.append('lon', st.me.lon); fd.append('accuracy', st.me.acc);
}

function submit() {
  if (!st.verdict) { toast('Choose what you found on site.', true); return; }
  var btn = $('submitBtn');
  var fd = new FormData();
  fd.append('verdict', st.verdict);
  fd.append('note', $('note').value);
  fd.append('surveyor', $('surveyor').value);
  store('cadai.field.surveyor', $('surveyor').value);
  gpsFields(fd);
  st.photos.forEach(function (f) { fd.append('photos', f); });
  btn.disabled = true; btn.textContent = 'Saving…';
  fetch('/api/surveys/' + encodeURIComponent(st.sid) + '/field/' + st.selected.properties.id, {method: 'POST', body: fd})
    .then(function (r) { return r.json().then(function (j) { if (!r.ok) throw new Error(j.error); return j; }); })
    .then(function (j) {
      st.parcels = j.parcels;
      drawParcels();
      toast('Saved: ' + j.observation.verdict_label + (j.observation.on_site ? ' (on site)' : ''));
      showQueue();
    })
    .catch(function (e) { btn.disabled = false; btn.textContent = 'Save field check'; toast('Not saved: ' + e.message, true); });
}

function recordCorner() {
  if (!st.me) { toast('Waiting for a GPS fix.', true); return; }
  if (st.me.acc > 10 && !confirm('GPS accuracy is only ±' + Math.round(st.me.acc) + ' m. Record this corner anyway?')) return;
  var fd = new FormData();
  gpsFields(fd);
  fd.append('parcel_id', st.selected.properties.id);
  fetch('/api/surveys/' + encodeURIComponent(st.sid) + '/field/corner', {method: 'POST', body: fd})
    .then(function (r) { return r.json().then(function (j) { if (!r.ok) throw new Error(j.error); return j; }); })
    .then(function (j) { toast('Corner ' + j.point_id + ' recorded (±' + Math.round(j.accuracy_m) + ' m).'); })
    .catch(function (e) { toast('Corner not saved: ' + e.message, true); });
}

$('grab').addEventListener('click', function (e) {
  if (e.target.closest('a,button,select')) return;
  $('sheet').classList.toggle('collapsed');
  setTimeout(function () { map.invalidateSize(); }, 50);
});
window.addEventListener('resize', function () { map.invalidateSize(); });

startGps();
loadSurveys().catch(function (e) { toast('Could not load surveys: ' + e.message, true); });
})();
