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
  'Residential':'#7C9BA6', 'Residential / Built-up':'#4EA195',
  'Residential Homestead / Farmhouse':'#4EA195',
  'Commercial / Retail Complex':'#E0A94A', 'Commercial':'#6E8FB0',
  'Ancillary / Shed Structure':'#8B9B90',
  'Government':'#E0A94A',
  'Agricultural':'#8FAE5D', 'Agricultural / Cultivated Cropland':'#8FAE5D', 'Agricultural / Cropland':'#8FAE5D',
  'Tree Canopy / Orchard / Agro-Forestry':'#3E7B4A', 'Tree Canopy / Orchard':'#3E7B4A',
  'Barren Land / Fallow Rural Ground':'#BFA27E', 'Barren Land':'#BFA27E',
  'Forest / Green Land':'#5FA37A', 'Industrial':'#B08D6B',
  'Transport & Highway Corridor':'#556872',
  'Vacant / Unclassified':'#485850', 'Vacant Plot / Open Land':'#485850'
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

function dataUrl(name){
  var params = new URLSearchParams(location.search);
  var session = params.get('session');
  // model.html displays ML benchmark training runs (Inria & UAVid) which reside in /data/
  if(window.location.pathname.endsWith('model.html')){
    return '/data/' + name;
  }
  if(!session && !params.has('legacy')){
    session = 'clear_drone_survey';
  }
  return session ? '/uploads/' + session + '/' + name : '/data/' + name;
}
function currentSessionId(){
  var params = new URLSearchParams(location.search);
  var session = params.get('session');
  if(!session && !params.has('legacy')){
    return 'clear_drone_survey';
  }
  return session;
}

function propertyCardUrl(parcelId){
  var session = currentSessionId();
  return session ? '/uploads/' + session + '/property_card/' + parcelId : '/property_card/' + parcelId;
}

function dxfDownloadUrl(){
  var session = currentSessionId();
  return session ? '/uploads/' + session + '/cadastre.dxf' : '/data/cadastre.dxf';
}

function initMap(){
  return L.map('map', {zoomControl:true, attributionControl:true, minZoom:10, maxZoom:22, zoomSnap:0, zoomDelta:0.5});
}

function loadBaseLayers(map, meta, imgBlob, layers){
  var geo = meta.image;
  var bounds = [[geo.lat_se, geo.lon_nw],[geo.lat_nw, geo.lon_se]];

  var imgUrl = URL.createObjectURL(imgBlob);
  L.imageOverlay(imgUrl, bounds, {attribution:'CadastraAI Drone Imagery / Orthomosaic'}).addTo(map);
  map.invalidateSize();
  map.fitBounds(bounds);
  window.addEventListener('resize', function(){ map.invalidateSize(); });

  var osmTiles = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {maxZoom:19, attribution:'&copy; OpenStreetMap contributors'});

  var overlayLayers = null;
  if (meta.mask_image) {
    var mg = meta.mask_image;
    var maskBounds = [[mg.lat_se, mg.lon_nw], [mg.lat_nw, mg.lon_se]];
    overlayLayers = {
      'AI building mask (U-Net)': L.imageOverlay(dataUrl('building_mask.png'), maskBounds,
        {opacity:0.75, attribution:'U-Net trained on Inria Aerial Image Labeling'})
    };
  }
  L.control.layers({'High-Res Drone Ortho': L.layerGroup(), 'Live OSM tiles (internet)': osmTiles}, overlayLayers, {position:'bottomleft'}).addTo(map);

  var refLayerGroup = L.layerGroup().addTo(map);
  function addRef(fc, color, weight, dash){
    if(fc && fc.features && fc.features.length > 0){
      L.geoJSON(fc, {style:{color:color, weight:weight, dashArray:dash, fillOpacity:0.12}}).addTo(refLayerGroup);
    }
  }
  addRef(layers.roads, '#4E5C52', 2, null);
  addRef(layers.railway, '#8A7A5A', 3, '1 4');
  addRef(layers.waterway, '#3E7A98', 2, null);

  // Drone extracted boundary walls & fences
  if(layers.extracted_walls && layers.extracted_walls.features && layers.extracted_walls.features.length > 0){
    L.geoJSON(layers.extracted_walls, {
      style:{color:'#B694DA', weight:2.5, opacity:0.85, dashArray:'4 3'}
    }).addTo(refLayerGroup);
  }

  // Drone extracted road corridors
  if(layers.extracted_roads && layers.extracted_roads.features && layers.extracted_roads.features.length > 0){
    L.geoJSON(layers.extracted_roads, {
      style:{color:'#8A9A86', weight:3, opacity:0.75}
    }).addTo(refLayerGroup);
  }

  // Drone extracted agricultural field bunds / ridges
  if(layers.extracted_bunds && layers.extracted_bunds.features && layers.extracted_bunds.features.length > 0){
    L.geoJSON(layers.extracted_bunds, {
      style:{color:'#9EBA63', weight:2, opacity:0.8, dashArray:'3 3'}
    }).addTo(refLayerGroup);
  }

  // Drone extracted vegetation & crop canopy
  var vegGroup = L.layerGroup();
  if(layers.extracted_vegetation && layers.extracted_vegetation.features && layers.extracted_vegetation.features.length > 0){
    L.geoJSON(layers.extracted_vegetation, {
      style:{color:'#5FA37A', weight:1.2, fillOpacity:0.18, fillColor:'#5FA37A'}
    }).addTo(vegGroup);
  }

  // Drone extracted tree canopy & orchards
  var treesGroup = L.layerGroup();
  if(layers.extracted_trees && layers.extracted_trees.features && layers.extracted_trees.features.length > 0){
    L.geoJSON(layers.extracted_trees, {
      style:{color:'#2D6A4F', weight:1.5, fillOpacity:0.25, fillColor:'#2D6A4F'}
    }).addTo(treesGroup);
  }

  // Drone extracted barren land / bare soil
  var barrenGroup = L.layerGroup();
  if(layers.extracted_barren && layers.extracted_barren.features && layers.extracted_barren.features.length > 0){
    L.geoJSON(layers.extracted_barren, {
      style:{color:'#BFA27E', weight:1.2, fillOpacity:0.20, fillColor:'#BFA27E'}
    }).addTo(barrenGroup);
  }

  if(layers.government && layers.government.features && layers.government.features.length > 0){
    L.geoJSON(layers.government, {
      style:{color:'#E0A94A', weight:1.5, fillOpacity:0.18, fillColor:'#E0A94A'},
      pointToLayer:function(f, latlng){ return L.circleMarker(latlng, {radius:6, color:'#E0A94A', fillColor:'#E0A94A', fillOpacity:0.6}); }
    }).addTo(refLayerGroup);
  }

  // buffer zones
  var bufferGroup = L.layerGroup().addTo(map);
  if(layers.railway && layers.railway.features) {
    L.geoJSON(layers.railway, {style:{color:'#E2685F', weight:26, opacity:0.10}}).addTo(bufferGroup);
  }
  if(layers.waterway && layers.waterway.features) {
    L.geoJSON(layers.waterway, {style:{color:'#E2685F', weight:14, opacity:0.10}}).addTo(bufferGroup);
  }

  return {refLayerGroup: refLayerGroup, bufferGroup: bufferGroup, treesGroup: treesGroup, barrenGroup: barrenGroup, vegGroup: vegGroup};
}
