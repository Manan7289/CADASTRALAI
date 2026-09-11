// Shared between index.html (Cadastral Map) and detection.html (Compliance & Detection).
var ALERT_META = {
  RAIL_BUFFER:  {label:'Rail Safety Buffer', shape:'triangle', sev:'critical'},
  WATER_BUFFER: {label:'Water Buffer Zone',   shape:'diamond',  sev:'critical'},
  GOVT_ENCROACH:{label:'Government Land Encroachment', shape:'hexagon', sev:'critical'},
  NO_ROAD:      {label:'No Direct Road Access', shape:'square', sev:'warning'},
  UNRECORDED:   {label:'Unrecorded Structure', shape:'circle', sev:'change'}
};
var SEV_COLOR = {critical:'#E2685F', warning:'#E0A94A', change:'#B694DA', ok:'#6CBE81'};
var LANDUSE_COLOR = {
  'Residential':'#7C9BA6', 'Government':'#E0A94A', 'Agricultural':'#8FAE5D',
  'Forest / Green Land':'#5FA37A', 'Industrial':'#B08D6B', 'Commercial':'#6E8FB0',
  'Vacant / Unclassified':'#5C6B63'
};

function shapePath(shape, cx, cy, r){
  switch(shape){
    case 'triangle': return 'M '+cx+' '+(cy-r)+' L '+(cx+r*0.95)+' '+(cy+r*0.8)+' L '+(cx-r*0.95)+' '+(cy+r*0.8)+' Z';
    case 'diamond':  return 'M '+cx+' '+(cy-r)+' L '+(cx+r)+' '+cy+' L '+cx+' '+(cy+r)+' L '+(cx-r)+' '+cy+' Z';
    case 'square':   return 'M '+(cx-r*0.85)+' '+(cy-r*0.85)+' h '+(r*1.7)+' v '+(r*1.7)+' h '+(-r*1.7)+' Z';
    case 'hexagon':
      var pts=[]; for(var i=0;i<6;i++){var a=Math.PI/6+i*Math.PI/3; pts.push((cx+r*Math.cos(a))+' '+(cy+r*Math.sin(a)));}
      return 'M '+pts.join(' L ')+' Z';
    default: return '';
  }
}
function iconSvg(shape, sev, size){
  size = size||16;
  var r = size/2-1.5, cx=size/2, cy=size/2, col=SEV_COLOR[sev];
  if(shape==='circle'){
    return '<svg width="'+size+'" height="'+size+'" viewBox="0 0 '+size+' '+size+'"><circle cx="'+cx+'" cy="'+cy+'" r="'+r+'" fill="none" stroke="'+col+'" stroke-width="2"/><path d="M '+cx+' '+(cy+r*0.5)+' L '+cx+' '+(cy-r*0.5)+' M '+(cx-r*0.35)+' '+(cy-r*0.1)+' L '+cx+' '+(cy-r*0.5)+' L '+(cx+r*0.35)+' '+(cy-r*0.1)+'" fill="none" stroke="'+col+'" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  }
  return '<svg width="'+size+'" height="'+size+'" viewBox="0 0 '+size+' '+size+'"><path d="'+shapePath(shape,cx,cy,r)+'" fill="none" stroke="'+col+'" stroke-width="2" stroke-linejoin="round"/></svg>';
}

// If the page was opened as ?session=<id> (from the Upload page), fetch that
// upload's own generated data instead of the fixed demo bundle under data/.
function dataUrl(name){
  var session = new URLSearchParams(location.search).get('session');
  return session ? 'uploads/' + session + '/' + name : 'data/' + name;
}
function currentSessionId(){
  return new URLSearchParams(location.search).get('session');
}

function initMap(){
  return L.map('map', {zoomControl:true, attributionControl:true, minZoom:10, maxZoom:22, zoomSnap:0, zoomDelta:0.5});
}

function loadBaseLayers(map, meta, imgBlob, layers){
  var geo = meta.image;
  var bounds = [[geo.lat_se, geo.lon_nw],[geo.lat_nw, geo.lon_se]];

  var imgUrl = URL.createObjectURL(imgBlob);
  L.imageOverlay(imgUrl, bounds, {attribution:'Esri World Imagery (stitched offline mosaic)'}).addTo(map);
  map.invalidateSize();
  map.fitBounds(bounds);
  window.addEventListener('resize', function(){ map.invalidateSize(); });

  var osmTiles = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {maxZoom:19, attribution:'&copy; OpenStreetMap contributors'});
  L.control.layers({'Offline satellite mosaic': L.layerGroup(), 'Live OSM tiles (needs internet)': osmTiles}, null, {position:'bottomleft'}).addTo(map);

  var refLayerGroup = L.layerGroup().addTo(map);
  function addRef(fc, color, weight, dash){
    L.geoJSON(fc, {style:{color:color, weight:weight, dashArray:dash, fillOpacity:0.12}}).addTo(refLayerGroup);
  }
  addRef(layers.roads, '#4E5C52', 2, null);
  addRef(layers.railway, '#8A7A5A', 3, '1 4');
  addRef(layers.waterway, '#3E7A98', 2, null);
  L.geoJSON(layers.government, {
    style:{color:'#E0A94A', weight:1.5, fillOpacity:0.18, fillColor:'#E0A94A'},
    pointToLayer:function(f, latlng){ return L.circleMarker(latlng, {radius:6, color:'#E0A94A', fillColor:'#E0A94A', fillOpacity:0.6}); }
  }).addTo(refLayerGroup);

  // buffer zones: real geometry buffered client-side would need turf, so this approximates
  // the 30m/15m legal buffers visually with a fixed-width translucent halo along rail/water.
  var bufferGroup = L.layerGroup().addTo(map);
  L.geoJSON(layers.railway, {style:{color:'#E2685F', weight:26, opacity:0.10}}).addTo(bufferGroup);
  L.geoJSON(layers.waterway, {style:{color:'#E2685F', weight:14, opacity:0.10}}).addTo(bufferGroup);

  return {refLayerGroup: refLayerGroup, bufferGroup: bufferGroup};
}
