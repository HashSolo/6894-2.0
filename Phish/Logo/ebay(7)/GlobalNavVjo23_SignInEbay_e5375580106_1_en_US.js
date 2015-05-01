if(typeof (vjo)=="undefined"){
var vjo={};
}
vjo.global=this;
vjo.createPkg=function(_1){
var _2=_1.split("."),len=_2.length;
var _3=this.global;
for(var i=0;i<len-1&&_3&&_2[i];i++){
_3=(_3[_2[i]])?_3[_2[i]]:_3[_2[i]]={};
}
return {pkg:_3,className:(len>0)?_2[len-1]:""};
};
vjo.needs=function(){
};
vjo.needsLib=function(){
};
vjo.type=function(_5){
var _6=function(){
if(this.base){
this.base.parent=this;
}
if(this.constructs){
var rv=this.constructs.apply(this,arguments);
if(rv){
return rv;
}
}
return null;
};
_6.props=function(_8){
for(var i in _8){
if(i!="props"&&i!="protos"&&i!="inherits"&&i!="prototype"&&i!="inits"&&i!="satisfies"&&i!="satisfiers"){
_6[i]=_8[i];
}
}
return _6;
};
function createBaseMethod(_a,_b,_c){
_a.prototype.base[_c]=function(){
var _d=this.parent,supBase=(_b.prototype&&_b.prototype.base)?_b.prototype.base[_c]:null,curBase;
if(supBase){
curBase=_d.base[_c];
_d.base[_c]=supBase;
}
var _e=(this.parent)?this.parent:this;
var rv=_b.prototype[_c].apply(_e,arguments);
if(curBase){
_d.base[_c]=curBase;
}
return rv;
};
}
_6.protos=function(obj,_11){
for(var i in obj){
if(i!="base"&&(!_11||i!="constructs")){
if(_11&&_11.prototype&&_11.prototype[i]&&typeof obj[i]=="function"){
createBaseMethod(_6,_11,i);
}
if((!_11&&!_6.prototype[i])&&_6.prototype.base&&_6.prototype.base[i]){
_6.prototype[i]=function(){
_6.prototype.base[i].apply(this,arguments);
};
}else{
if(!(_11&&_6.prototype[i])){
_6.prototype[i]=obj[i];
}
}
}
}
return _6;
};
function createBase(clz,_14){
return (function(){
var _15=_14.prototype.base||_14,constructs=this.constructs,curBase=this.base;
if(_14.prototype.constructs&&constructs){
this.constructs=_14.prototype.constructs;
}
this.base=function(){
_15.apply(this,arguments);
};
_14.apply(this,arguments);
if(constructs){
this.constructs=constructs;
}
this.base=curBase;
});
}
_6.inherits=function(_16){
var _17=vjo.createPkg(_16);
var _18=_17.pkg[_17.className];
_6.prototype.base=createBase(_6,_18);
_6.protos(_18.prototype,_18);
_6.props(_18);
return _6;
};
_6.singleton=function(){
return _6;
};
_6.inits=function(_19){
var _1a=vjo.createPkg(_5);
if(typeof _1a.pkg[_1a.className]=="function"){
_19.call(this);
}
return _6;
};
_6.satisfies=function(_1b){
return _6;
};
_6.satisfiers=function(_1c){
return _6;
};
_6.makeFinal=function(){
return _6;
};
var _1d=vjo.createPkg(_5);
return (_1d.pkg[_1d.className])?_6:(_1d.pkg[_1d.className]=_6);
};

vjo.type("vjo.Registry").singleton().protos({constructs:function(){
this.controls=[];
},put:function(_1e,_1f){
return this.controls[_1e]=_1f;
},get:function(_20){
return this.controls[_20];
},dump:function(){
var _21=this.controls;
var _22="controls on page:\n";
for(var i in _21){
_22+="key = "+i;
_22+="controlName = "+_21[i].objtype;
_22+="\n";
}
return _22;
}}).inits(function(){
vjo.Registry=new vjo.Registry();
});

vjo.type("vjo.dsf.utils.URL").props({addArg:function(_24,_25,_26){
if(_24==null||_24==undefined){
return null;
}
if(_24.indexOf("?")<0){
_24+="?"+_25+"="+_26;
return _24;
}
var _27=this.getArgPairIfExists(_24,_25);
if(_27!==null){
_24=_24.replace(_27,_25+"="+_26);
}else{
_24+="&"+_25+"="+_26;
}
return _24;
},getArg:function(_28,_29){
if(_28==null||_28==undefined){
return null;
}
if(_28.indexOf("?")<0){
return null;
}
var _2a=this.getArgPairIfExists(_28,_29);
if(_2a!==null){
return _2a.substring(_2a.indexOf("=")+1);
}
return null;
},getArgPairIfExists:function(_2b,_2c){
var _2d=_2b.indexOf("?");
if(_2d<0){
return null;
}
var _2e=_2b;
var _2f,argName;
while(_2d>=0){
_2e=_2e.substring(_2d+1);
_2f=_2e;
_2d=_2e.indexOf("&");
if(_2d>=0){
_2f=_2e.substring(0,_2d);
}
argName=_2f.substring(0,_2f.indexOf("="));
if(argName==_2c){
return _2f;
}
}
return null;
}});

String.prototype.has=function(_30){
return (this.indexOf(_30)!=-1);
};
String.prototype.hasArg=function(_31){
var a=_31,rv=false;
if(typeof (a)=="string"){
rv=this.has(a);
}else{
var aL=a.length;
for(var j=0;j<aL&&!rv;j++){
rv=this.has(a[j]);
}
}
return rv;
};
String.prototype.hasAny=function(){
var a=arguments,l=a.length,rv=false;
for(var i=0;i<l&&!rv;i++){
rv=this.hasArg(a[i]);
}
return rv;
};
String.prototype.hasAll=function(){
var a=arguments,l=a.length;
for(var i=0;i<l;i++){
if(!this.hasArg(a[i])){
return false;
}
}
return true;
};
String.prototype.is=function(s){
return (this==s);
};
String.prototype.isAny=function(){
var a=arguments,l=a.length,rv=false,aL;
for(var i=0;i<l&&!rv;i++){
if(typeof (a[i])=="string"){
rv=(this==a[i]);
}else{
aL=a[i].length;
for(var j=0;j<aL&&!rv;j++){
rv=(this==a[i][j]);
}
}
}
return rv;
};

vjo.needs("vjo.dsf.utils.URL");
vjo.needs("vjo.dsf.typeextensions.string.Comparison");
vjo.type("vjo.darwin.tracking.enabler.TrackingModuleEnabler").satisfies("vjo.dsf.common.IJsHandler").protos({constructs:function(_3d,_3e){
this.sCid=_3d;
this.sCidParms=_3e;
},rewriteUrl:function(_3f){
if(_3f.nativeEvent==null||_3f.nativeEvent==undefined){
return;
}
var _40=_3f.nativeEvent.srcElement||_3f.nativeEvent.target;
if(_40==null||_40==undefined){
return;
}
_40=this.getAnchor(_40);
if(this.sCidParms&&_40){
_40.href=vjo.dsf.utils.URL.addArg(_40.href,this.sCid,this.sCidParms);
}
},getAnchor:function(_41){
var e=_41;
if(e&&e.tagName){
if(!e.tagName.is("A")){
e=this.getAnchor(e.parentNode);
}
return e;
}
},handle:function(_43){
this.rewriteUrl(_43);
}});

vjo.type("vjo.dsf.Event").protos({constructs:function(src,_45,_46){
this.src=src;
this.eventType=_45;
this.nativeEvent=_46;
}});

vjo.needs("vjo.dsf.Event");
vjo.type("vjo.dsf.EventDispatcher").singleton().protos({constructs:function(){
this.eventHandlers={};
this.nativeEventHandlers={};
this.fCustomLoad={};
},process:function(_47,_48){
var _49=this.eventHandlers[_47];
if(!_49){
return true;
}
var _4a=_49[_48.eventType];
if(!_4a){
return true;
}
var _4b=true;
for(var i=0;i<_4a.length;i++){
var _4d=_4a[i].handle(_48);
if(_4d&&_4d.objType=="dsf_Message"){
if(vjo.dsf.ServiceEngine){
var cb=this.createRequest(_4d);
window.setTimeout(cb,1);
}
if(_4d.returnData===false){
_4b=false;
}
}else{
if(_4d===false){
_4b=false;
}
}
}
return _4b;
},createRequest:function(msg){
return function(){
vjo.dsf.ServiceEngine.handleRequest(msg);
};
},register:function(id,_51,_52){
var _53=this.eventHandlers[id];
if(!_53){
_53=this.eventHandlers[id]={};
}
if(!_53[_51]){
_53[_51]=[];
}
var len=_53[_51].length;
_53[_51][len]=_52;
},unregister:function(id,_56){
if(!this.eventHandlers[id]){
return;
}
this.eventHandlers[id][_56]=[];
},isBound:function(id,_58){
var _59=this.eventHandlers[id];
return (_59&&_59[_58]&&_59[_58].length>0)?true:false;
},registerNative:function(_5a,_5b,_5c){
var id=(_5a==window)?"body":_5a.id;
var _5e=this.nativeEventHandlers[id];
if(!_5e){
_5e=this.nativeEventHandlers[id]={};
}
if(!_5e[_5b]){
_5e[_5b]=[];
}
var len=_5e[_5b].length;
_5e[_5b][len]=_5c;
},add:function(id,_61,_62){
if(!id||!_61||!_62){
return this;
}else{
if(typeof _62.handle!="function"){
if(typeof _62=="function"){
var _63=_62;
var obj={handle:function(){
return _63.apply(this,arguments);
}};
_62=obj;
}else{
return this;
}
}
}
var b=this.isBound(id,_61);
this.register(id,_61,_62);
if(!b){
this.bind(id,_61);
}
return _62;
},addEventListener:function(_66,_67,_68,_69,_6a){
var scp=_69||vjo.global;
if(typeof _66=="string"){
_66=document.getElementById(_66);
}
if(!_66){
return false;
}
var _6c=function(_6d){
var ev=window.event||_6d;
var rv=_68.call(scp,ev);
if(rv===false){
vjo.dsf.EventDispatcher.stopEvent(ev);
}
return rv;
};
if(window.addEventListener){
_66.addEventListener(_67,_6c,_6a||false);
this.registerNative(_66,_67,_6c);
return _6c;
}else{
if(window.attachEvent){
_66.attachEvent("on"+_67,_6c);
this.registerNative(_66,_67,_6c);
return _6c;
}
}
_66["on"+_67]=_68;
return false;
},bind:function(id,_71){
var _72=document.getElementById(id);
if(id=="body"||_72==document.body){
_72=document.body;
if(_71=="load"||_71=="unload"){
var rv=this.addEventListener(window,_71,function(){
var oED=vjo.dsf.EventDispatcher;
if(typeof oED.fCustomLoad[_71]=="function"){
oED.fCustomLoad[_71]();
}
oED[_71]("body");
oED.unregister("body",_71);
oED.fCustomLoad={};
});
if(rv===false){
if(_72.vjLoadSet){
return this;
}else{
_72.vjLoadSet=true;
var _75=window["on"+_71]||"";
if(_75){
this.fCustomLoad[_71]=_75;
}
}
}
return this;
}
}
if(_72){
this.addEventListener(_72,_71,function(_76){
return vjo.dsf.EventDispatcher[_71](this,_76||window.event);
},_72);
}
return null;
},reBind:function(){
var eH=this.eventHandlers;
var nEH=this.nativeEventHandlers;
for(var id in eH){
for(var _7a in eH[id]){
if(!this.hasBinding(id,_7a)){
this.bind(id,_7a);
}
}
}
},hasBinding:function(id,_7c){
var nEH=this.nativeEventHandlers;
if(nEH[id]&&nEH[id][_7c]){
var aH=nEH[id][_7c],len=aH.length,rv=false;
for(var i=0;i<len;i++){
var str=aH[i].toString();
if(str&&str.indexOf("vjo.dsf.EventDispatcher")!=-1){
return true;
}
}
}
return false;
},removeEventListener:function(_81,_82,_83){
if(!_81||!_82){
return;
}else{
if(typeof _81=="string"){
_81=document.getElementById(_81);
}
}
if(window.addEventListener){
_81.removeEventListener(_82,_83,false);
}else{
if(window.attachEvent){
_81.detachEvent("on"+_82,_83);
}else{
_81["on"+_82]=null;
}
}
},detachNativeHandlers:function(_84,_85){
var id=(_84==window)?"body":_84.id;
var _87=this.nativeEventHandlers[id];
if(_87&&_87[_85]){
var h=_87[_85];
for(var i=0;i<h.length;i++){
this.removeEventListener(_84,_85,_87[_85][i]);
}
_87[_85]=[];
}
},detachHandler:function(id,_8b,_8c){
var _8d=this.eventHandlers[id];
if(!_8d||!_8d[_8b]){
return;
}
var h=[],len=_8d[_8b].length;
for(var i=0;i<len;i++){
if(_8c!=_8d[_8b][i]){
h[h.length]=_8d[_8b][i];
}
}
this.eventHandlers[id][_8b]=h;
},detachHandlers:function(id,_91){
this.unregister(id,_91);
var _92=document.getElementById(id);
if(id=="body"){
_92=window;
}
if(_92){
this.detachNativeHandlers(_92,_91);
}
},stopEvent:function(evt){
this.stopPropagation(evt);
this.preventDefault(evt);
},stopPropagation:function(evt){
if(evt.stopPropagation){
evt.stopPropagation();
}else{
evt.cancelBubble=true;
}
},preventDefault:function(evt){
if(evt.preventDefault){
evt.preventDefault();
}else{
evt.returnValue=false;
}
},target:function(_96){
var _97=vjo.darwin.domain.finding.common.browser.DHTMLEvent;
return _97.resolveTextNode((_96.target)?_96.target:_96.srcElement);
},relatedTarget:function(_98){
var _99=vjo.darwin.domain.finding.common.browser.DHTMLEvent;
if(_98.relatedTarget){
return _99.resolveTextNode(_98.relatedTarget);
}else{
if((_98.type=="mouseover")&&_98.fromElement){
return _99.resolveTextNode(_98.fromElement);
}else{
if((_98.type=="mouseout")&&_98.toElement){
return _99.resolveTextNode(_98.toElement);
}else{
return null;
}
}
}
},resolveTextNode:function(_9a){
return (_9a&&(_9a.nodeType==3))?_9a.parentNode:_9a;
},cleanUp:function(){
var _9b=this.nativeEventHandlers;
for(var id in _9b){
for(var ev in _9b[id]){
this.detachHandlers(id,ev,true);
}
}
},getId:function(src,id){
var _a0=id;
if(_a0===null||!_a0){
_a0=src.id;
}
return _a0;
},getBodyId:function(src,id){
var _a3=this.getId(src,id);
if(!_a3||src==document.body){
_a3="body";
}
return _a3;
},load:function(src,_a5){
var id=this.getBodyId(src);
var rv=this.process(id,new vjo.dsf.Event(src,"load",_a5));
if(id==="body"){
this.unregister("body","load");
}
return rv;
},unload:function(src,_a9){
return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"unload",_a9));
},change:function(src,_ab){
return this.process(this.getId(src),new vjo.dsf.Event(src,"change",_ab));
},submit:function(src,_ad){
return this.process(this.getId(src),new vjo.dsf.Event(src,"submit",_ad));
},reset:function(src,_af){
return this.process(this.getId(src),new vjo.dsf.Event(src,"reset",_af));
},select:function(src,_b1){
return this.process(this.getId(src),new vjo.dsf.Event(src,"select",_b1));
},blur:function(src,_b3){
return this.process(this.getId(src),new vjo.dsf.Event(src,"blur",_b3));
},focus:function(src,_b5){
return this.process(this.getId(src),new vjo.dsf.Event(src,"focus",_b5));
},keydown:function(src,_b7){
return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"keydown",_b7));
},keypress:function(src,_b9){
return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"keypress",_b9));
},keyup:function(src,_bb){
return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"keyup",_bb));
},click:function(src,_bd){
return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"click",_bd));
},dblclick:function(src,_bf){
return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"dblclick",_bf));
},mousedown:function(src,_c1){
return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"mousedown",_c1));
},mousemove:function(src,_c3){
return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"mousemove",_c3));
},mouseout:function(src,_c5){
return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"mouseout",_c5));
},mouseover:function(src,_c7){
return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"mouseover",_c7));
},mouseup:function(src,_c9){
return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"mouseup",_c9));
}}).inits(function(){
vjo.dsf.EventDispatcher=new vjo.dsf.EventDispatcher();
vjo.dsf.EventDispatcher.addEventListener(window,"load",function(){
vjo.dsf.EventDispatcher.addEventListener(window,"unload",function(){
vjo.dsf.EventDispatcher.cleanUp();
});
});
});

function HeaderTrk() { 	return { handle : function (event) { var _d = vjo.dsf.EventDispatcher;
_d.add('BrowseCategories-menu','click',new vjo.darwin.tracking.enabler.TrackingModuleEnabler('_trksid', 'm37'));
_d.add('gnheader','click',new vjo.darwin.tracking.enabler.TrackingModuleEnabler('_trksid', 'm37'));
   } }; }
vjo.type("vjo.dsf.SvcConfig").protos({constructs:function(_ca,url){
this.url=url;
this.method=_ca;
this.reqtMarshalling="raw";
this.respMarshalling="raw";
}});

vjo.type("vjo.dsf.Message").protos({constructs:function(_cc){
this.objType="dsf_Message";
this.svcId=_cc;
this.request;
this.response;
this.clientContext;
this.trspType="InProc";
this.status;
this.svcConfig;
this.returnData=true;
this.trace="";
}});

function FooterTrk() { 	return { handle : function (event) { var _d = vjo.dsf.EventDispatcher;
_d.add('glbfooter','click',new vjo.darwin.tracking.enabler.TrackingModuleEnabler('_trksid', 'm40'));
   } }; }
/* compspec addJsCompRegistration*/ 
vjo.Registry.put('HeaderTrackingCompSpecGenerator_0', new vjo.darwin.tracking.enabler.TrackingModuleEnabler('_trksid', 'm37'));vjo.Registry.put('HeaderTrackingCompSpecGenerator_1', new vjo.darwin.tracking.enabler.TrackingModuleEnabler('_trksid', 'm37'));
/* end comp spec*/

vjo.dsf.EventDispatcher.add('body','load', new HeaderTrk());
/* compspec addJsCompRegistration*/ 
vjo.Registry.put('FooterTrackingCompSpecGenerator_0', new vjo.darwin.tracking.enabler.TrackingModuleEnabler('_trksid', 'm40'));
/* end comp spec*/

vjo.dsf.EventDispatcher.add('body','load', new FooterTrk());

// en_US/e537/GlobalNavVjo23_SignInEbay_e5375580106_1_en_US
// b=5580106