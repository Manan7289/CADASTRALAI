// Survey management: rename, delete (soft), undo last change, edit history in the Overview tab.
(function () {
"use strict";
var WB = window.WB;
var esc = WB.esc;
var hist = {entries: [], undo: 0};

function base() { return '/api/surveys/' + encodeURIComponent(WB.state.sid); }

function refresh() {
  if (!WB.state.sid) return Promise.resolve();
  return WB.api('GET', base() + '/history').then(function (h) {
    hist.entries = h.entries; hist.undo = h.undo_available;
    var btn = document.getElementById('undoBtn');
    var next = hist.entries.filter(function (e) { return !/^undo:|^renamed/.test(e.action); })[0];
    btn.disabled = !hist.undo;
    btn.title = hist.undo ? 'Undo: ' + (next ? next.action : 'last change') + ' (Ctrl/Cmd+Z)' : 'Nothing to undo';
    renderHistory();
  }).catch(function () {});
}

function renderHistory() {
  var body = document.getElementById('tabBody');
  if (WB.state.tab !== 'overview' || !body) return;
  var el = document.getElementById('historyPanel');
  if (!el) {
    el = document.createElement('div');
    el.id = 'historyPanel';
    body.appendChild(el);
  }
  el.innerHTML = '<h3 class="panel-title">Edit history <span class="note">' + hist.entries.length + ' change' + (hist.entries.length === 1 ? '' : 's') + '</span></h3>' +
    '<div class="history-list">' + hist.entries.slice(0, 60).map(function (e) {
      return '<div><time>' + esc(e.t.slice(5, 16)) + '</time><span>' + esc(e.action) + '</span></div>';
    }).join('') + '</div>' +
    (hist.undo ? '<div class="btn-row" style="margin-top:8px"><button class="btn" id="undoInline">Undo last change</button></div>' : '');
  var b = document.getElementById('undoInline');
  if (b) b.addEventListener('click', doUndo);
}

function doUndo() {
  if (!hist.undo) { WB.toast('Nothing to undo.'); return; }
  WB.setSave('saving', 'Undoing…');
  WB.api('POST', base() + '/undo').then(function (res) {
    WB.applyServerResult(res);
    WB.setSave('idle', 'Saved');
    WB.toast('Undone: ' + res.undone);
    return refresh();
  }).catch(function (e) { WB.setSave('error', 'Undo failed'); WB.toast(e.message, true); });
}

document.getElementById('undoBtn').addEventListener('click', doUndo);
document.addEventListener('keydown', function (e) {
  if ((e.metaKey || e.ctrlKey) && !e.shiftKey && e.key.toLowerCase() === 'z' && !e.target.matches('input, textarea, select')) {
    e.preventDefault();
    doUndo();
  }
}, true);

document.getElementById('renameBtn').addEventListener('click', function () {
  var name = prompt('Rename survey', WB.state.meta ? WB.state.meta.name : '');
  if (name == null) return;
  WB.api('PATCH', base(), {name: name}).then(function (meta) {
    WB.state.meta.name = meta.name;
    var opt = document.querySelector('#surveySelect option[value="' + CSS.escape(WB.state.sid) + '"]');
    if (opt) opt.textContent = meta.name;
    document.getElementById('surveySubtitle').textContent = 'Survey Workbench · ' + meta.name;
    WB.renderTabs();
    refresh();
  }).catch(function (e) { WB.toast(e.message, true); });
});

document.getElementById('deleteBtn').addEventListener('click', function () {
  var name = WB.state.meta ? WB.state.meta.name : WB.state.sid;
  if (!confirm('Delete survey "' + name + '"?\n\nIt is moved to data/surveys_trash on the server and can be restored from there.')) return;
  WB.api('DELETE', base()).then(function () {
    WB.toast('Survey deleted.');
    history.replaceState(null, '', location.pathname);
    WB.loadSurveys();
  }).catch(function (e) { WB.toast(e.message, true); });
});

// keep undo state and the history panel current after every save / tab change
document.addEventListener('wb:survey-opened', refresh);
var chip = document.getElementById('saveChip');
new MutationObserver(function () { if (chip.dataset.state === 'idle') refresh(); }).observe(chip, {attributes: true, childList: true});
document.querySelectorAll('.tabs button').forEach(function (b) {
  b.addEventListener('click', function () { setTimeout(renderHistory, 0); });
});
})();
