// Workbench: show field verification visits (verdict, note, GPS distance, photos) for the selected parcel.
(function () {
"use strict";
var WB = window.WB;
var esc = WB.esc;

document.addEventListener('wb:parcel-detail', function (e) {
  var id = e.detail.id, el = e.detail.el, sid = WB.state.sid;
  fetch('/api/surveys/' + encodeURIComponent(sid) + '/field?parcel_id=' + id, {cache: 'no-store'})
    .then(function (r) { return r.json(); })
    .then(function (j) {
      if (!el.isConnected) return;
      var obs = (j.observations || []).slice().reverse();
      var html = '<div style="margin-top:14px"><h3 class="panel-title">Field verification <span class="note">' + obs.length + ' visit' + (obs.length === 1 ? '' : 's') + '</span></h3>';
      if (!obs.length) {
        html += '<div class="note">Not visited yet. Mark it <b>Field check</b> to put it in the surveyors\' queue on the <a href="field.html?survey=' + encodeURIComponent(sid) + '" style="color:var(--accent)">field page</a>.</div>';
      } else {
        html += obs.map(function (o) {
          var where = o.distance_to_parcel_m == null ? '<span class="tag warn">no GPS</span>'
            : o.on_site ? '<span class="tag ok">on site' + (o.gps && o.gps.accuracy_m != null ? ' ±' + Math.round(o.gps.accuracy_m) + ' m' : '') + '</span>'
            : '<span class="tag warn">' + Math.round(o.distance_to_parcel_m) + ' m away</span>';
          return '<div class="finding" style="flex-direction:column;gap:6px"><div><b>' + esc(o.verdict_label) + '</b>' +
            '<p>' + esc(o.time) + (o.surveyor ? ' · ' + esc(o.surveyor) : '') + ' · ' + where + '</p>' +
            (o.note ? '<p style="color:var(--text)">' + esc(o.note) + '</p>' : '') + '</div>' +
            (o.photos.length ? '<div class="btn-row" style="gap:6px">' + o.photos.map(function (ph) {
              var url = '/surveys/' + encodeURIComponent(sid) + '/field/photos/' + encodeURIComponent(ph);
              return '<a href="' + url + '" target="_blank" rel="noopener"><img src="' + url + '" alt="Field photo" style="width:72px;height:72px;object-fit:cover;border-radius:7px;border:1px solid var(--border)"></a>';
            }).join('') + '</div>' : '') + '</div>';
        }).join('');
      }
      el.innerHTML = html + '</div>';
    })
    .catch(function () {});
});

document.addEventListener('wb:survey-opened', function (e) {
  var link = document.getElementById('fieldLink');
  if (link) link.href = 'field.html?survey=' + encodeURIComponent(e.detail.sid);
});
})();
