(function () {
var _r = vjo.Registry;
function $o0(){return new vjo.darwin.tracking.enabler.TrackingModuleEnabler("_trksid","_trkparms","m37;",";");};_r.put('HeaderTrackingCompSpecGenerator_0',$o0()); _r.put('HeaderTrackingCompSpecGenerator_1',$o0()); })();
function HeaderTrk(){ return {handle:function (event){ (function(){
var _d=vjo.dsf.EventDispatcher;
var _r=vjo.Registry;
function $0(){return function(event){return this.handle(event);};};_d.add('BrowseCategories-menu','click',$0(),_r._HeaderTrackingCompSpecGenerator_0);_d.add('gnheader','click',$0(),_r._HeaderTrackingCompSpecGenerator_1);_d.add('body','click',function(event){ vjo.darwin.tracking.enabler.TrackingEnabler.copySIDToCookie(event, "_trksid", "_sp", "_trkparms");  });})();  }};};
vjo.dsf.EventDispatcher.add('body','load', new HeaderTrk());
(function () {
var _r = vjo.Registry;
_r.put('FooterTrackingCompSpecGenerator_0',new vjo.darwin.tracking.enabler.TrackingModuleEnabler("_trksid", "_trkparms", "m40;", ";")); })();
function FooterTrk(){ return {handle:function (event){ (function(){
var _d=vjo.dsf.EventDispatcher;
var _r=vjo.Registry;
_d.add('glbfooter','click',function(event) { this.handle(event); },_r._FooterTrackingCompSpecGenerator_0);})();  }};};
vjo.dsf.EventDispatcher.add('body','load', new FooterTrk());

// en_US/e611/GlobalNav14_SignInEbay_e6118261870_1_en_US
// b=8261870