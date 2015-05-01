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
},put:function(_1,_2){
return this.controls[_1]=_2;
},get:function(_3){
return this.controls[_3];
},dump:function(){
var _4=this.controls;
var _5="controls on page:\n";
for(var i in _4){
_5+="key = "+i;
_5+="controlName = "+_4[i].objtype;
_5+="\n";
}
return _5;
}}).inits(function(){
vjo.Registry=new vjo.Registry();
});

vjo.type("vjo.dsf.utils.URL").props({addArg:function(_1,_2,_3){
if(_1==null||_1==undefined){
return null;
}
if(_1.indexOf("?")<0){
_1+="?"+_2+"="+_3;
return _1;
}
var _4=this.getArgPairIfExists(_1,_2);
if(_4!==null){
_1=_1.replace(_4,_2+"="+_3);
}else{
_1+="&"+_2+"="+_3;
}
return _1;
},getArg:function(_5,_6){
if(_5==null||_5==undefined){
return null;
}
if(_5.indexOf("?")<0){
return null;
}
var _7=this.getArgPairIfExists(_5,_6);
if(_7!==null){
return _7.substring(_7.indexOf("=")+1);
}
return null;
},getArgPairIfExists:function(_8,_9){
var _a=_8.indexOf("?");
if(_a<0){
return null;
}
var _b=_8;
var _c,argName;
while(_a>=0){
_b=_b.substring(_a+1);
_c=_b;
_a=_b.indexOf("&");
if(_a>=0){
_c=_b.substring(0,_a);
}
argName=_c.substring(0,_c.indexOf("="));
if(argName==_9){
return _c;
}
}
return null;
}});

String.prototype.has=function(_1){
return (this.indexOf(_1)!=-1);
};
String.prototype.hasArg=function(_2){
var a=_2,rv=false;
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
vjo.type("vjo.darwin.tracking.enabler.TrackingModuleEnabler").satisfies("vjo.dsf.common.IJsHandler").protos({constructs:function(_1,_2){
this.sCid=_1;
this.sCidParms=_2;
},rewriteUrl:function(_3){
if(_3.nativeEvent==null||_3.nativeEvent==undefined){
return;
}
var _4=_3.nativeEvent.srcElement||_3.nativeEvent.target;
if(_4==null||_4==undefined){
return;
}
_4=this.getAnchor(_4);
if(this.sCidParms&&_4&&!_4.href.has("javascript:")){
_4.href=vjo.dsf.utils.URL.addArg(_4.href,this.sCid,this.sCidParms);
}
},getAnchor:function(_5){
var e=_5;
if(e&&e.tagName){
if(!e.tagName.is("A")){
e=this.getAnchor(e.parentNode);
}
return e;
}
},handle:function(_7){
this.rewriteUrl(_7);
}});

vjo.type("vjo.dsf.Event").protos({constructs:function(_1,_2,_3){
this.src=_1;
this.eventType=_2;
this.nativeEvent=_3;
}});

vjo.needs("vjo.dsf.Event");
vjo.type("vjo.dsf.EventDispatcher").singleton().protos({constructs:function(){
this.eventHandlers={};
this.nativeEventHandlers={};
this.unboundElems=[];
this.fCustomLoad={};
},process:function(_1,_2){
var _3=this.eventHandlers[_1];
if(!_3){
return true;
}
var _4=_3[_2.eventType];
if(!_4){
return true;
}
var _5;
for(var i=0;i<_4.length;i++){
var _7=_4[i].handle(_2);
if(_7&&_7.objType=="dsf_Message"){
if(vjo.dsf.ServiceEngine){
vjo.dsf.ServiceEngine.handleRequest(_7);
}
if(_7.returnData===false){
_5=false;
}
}else{
if(_5!=false&&typeof _7!="undefined"){
_5=_7;
}
}
}
return _5;
},register:function(id,_9,_a){
if(!id||!_9||!_a){
return this;
}else{
if(typeof _a.handle!="function"){
if(typeof _a=="function"){
var _b=_a;
var _c={handle:function(){
return _b.apply(this,arguments);
}};
_a=_c;
}else{
return this;
}
}
}
var _d=this.eventHandlers[id];
if(!_d){
_d=this.eventHandlers[id]={};
}
if(!_d[_9]){
_d[_9]=[];
}
var _e=_d[_9].length;
_d[_9][_e]=_a;
return _a;
},unregister:function(id,_10){
if(!this.eventHandlers[id]){
return;
}
this.eventHandlers[id][_10]=[];
},registerNative:function(_11,_12,_13){
var id=(_11==window)?"body":_11.id;
var _15=this.nativeEventHandlers[id];
if(!_15){
_15=this.nativeEventHandlers[id]={};
}
if(!_15[_12]){
_15[_12]=[];
}
var len=_15[_12].length;
_15[_12][len]=_13;
},add:function(id,_18,_19){
if(!id||!_18||!_19){
return this;
}
var b=this.isBound(id,_18),rv=this.register(id,_18,_19);
if(!b){
var b=this.bind(id,_18);
if(b==null){
var len=this.unboundElems.length;
this.unboundElems[len]=id;
}
}
return rv;
},addEventListener:function(_1c,_1d,_1e,_1f,_20){
var scp=_1f||vjo.global;
if(typeof _1c=="string"){
_1c=document.getElementById(_1c);
}
if(!_1c){
return false;
}
var _22=function(_23){
var ev=window.event||_23;
var rv=_1e.call(scp,ev);
if(rv===false){
vjo.dsf.EventDispatcher.stopEvent(ev);
}
if(typeof rv!="undefined"){
return rv;
}
};
if(window.addEventListener){
_1c.addEventListener(_1d,_22,_20||false);
this.registerNative(_1c,_1d,_22);
return _22;
}else{
if(window.attachEvent){
_1c.attachEvent("on"+_1d,_22);
this.registerNative(_1c,_1d,_22);
return _22;
}
}
_1c["on"+_1d]=_1e;
return false;
},bind:function(id,_27){
var _28=document.getElementById(id);
if(id=="body"||_28==document.body){
_28=document.body;
if(_27=="load"||_27=="unload"){
var rv=this.addEventListener(window,_27,function(){
var oED=vjo.dsf.EventDispatcher;
if(typeof oED.fCustomLoad[_27]=="function"){
oED.fCustomLoad[_27]();
}
oED[_27]("body");
oED.unregister("body",_27);
oED.fCustomLoad={};
});
if(rv===false){
if(_28.vjLoadSet){
return this;
}else{
_28.vjLoadSet=true;
var _2b=window["on"+_27]||"";
if(_2b){
this.fCustomLoad[_27]=_2b;
}
}
}
return this;
}
}
if(_28){
this.addEventListener(_28,_27,function(_2c){
return vjo.dsf.EventDispatcher[_27](this,_2c||window.event);
},_28);
return this;
}
return null;
},reBind:function(){
var eH=this.eventHandlers,uE=this.unboundElems,len=uE.length,tmp=[];
for(var i=0;i<len;i++){
var id=uE[i],hdls=eH[id];
if(hdls){
for(var _30 in hdls){
if(!this.hasBinding(id,_30)){
var rv=this.bind(id,_30);
if(rv==null){
tmp[tmp.length]=id;
}
}
}
}
}
this.unboundElems=tmp;
},isBound:function(id,_33){
var _34=this.eventHandlers[id];
return (_34&&_34[_33]&&_34[_33].length>0)?true:false;
},hasBinding:function(id,_36){
var nEH=this.nativeEventHandlers;
if(nEH[id]&&nEH[id][_36]){
var aH=nEH[id][_36],len=aH.length,rv=false;
for(var i=0;i<len;i++){
var str=aH[i].toString();
if(str&&str.indexOf("vjo.dsf.EventDispatcher")!=-1){
return true;
}
}
}
return false;
},removeEventListener:function(_3b,_3c,_3d){
if(!_3b||!_3c){
return;
}else{
if(typeof _3b=="string"){
_3b=document.getElementById(_3b);
}
}
if(window.addEventListener&&_3d){
_3b.removeEventListener(_3c,_3d,false);
}else{
if(window.attachEvent&&_3d){
_3b.detachEvent("on"+_3c,_3d);
}else{
_3b["on"+_3c]=null;
}
}
},detachNativeHandlers:function(_3e,_3f){
var id=(_3e==window)?"body":_3e.id;
var _41=this.nativeEventHandlers[id];
if(_41&&_41[_3f]){
var h=_41[_3f];
for(var i=0;i<h.length;i++){
this.removeEventListener(_3e,_3f,_41[_3f][i]);
}
_41[_3f]=[];
}
},detachHandler:function(id,_45,_46){
var _47=this.eventHandlers[id];
if(!_47||!_47[_45]){
return;
}
var h=[],len=_47[_45].length;
for(var i=0;i<len;i++){
if(_46!=_47[_45][i]){
h[h.length]=_47[_45][i];
}
}
this.eventHandlers[id][_45]=h;
},detachHandlers:function(id,_4b){
this.unregister(id,_4b);
var _4c=document.getElementById(id);
if(id=="body"){
_4c=window;
}
if(_4c){
this.detachNativeHandlers(_4c,_4b);
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
},target:function(_50){
return this.resolveTextNode((_50.target)?_50.target:_50.srcElement);
},relatedTarget:function(_51){
if(_51.relatedTarget){
return this.resolveTextNode(_51.relatedTarget);
}else{
if((_51.type=="mouseover")&&_51.fromElement){
return this.resolveTextNode(_51.fromElement);
}else{
if((_51.type=="mouseout")&&_51.toElement){
return this.resolveTextNode(_51.toElement);
}else{
return null;
}
}
}
},resolveTextNode:function(_52){
return (_52&&(_52.nodeType==3))?_52.parentNode:_52;
},cleanUp:function(){
var _53=this.nativeEventHandlers;
for(var id in _53){
for(var ev in _53[id]){
if(ev!="unload"){
this.detachHandlers(id,ev,true);
}
}
}
},getId:function(src,id){
var _58=id;
if(_58===null||!_58){
_58=src.id;
}
return _58;
},getBodyId:function(src,id){
var _5b=this.getId(src,id);
if(!_5b||src==document.body){
_5b="body";
}
return _5b;
},load:function(src,_5d){
var id=this.getBodyId(src);
var rv=this.process(id,new vjo.dsf.Event(src,"load",_5d));
if(id==="body"){
this.unregister("body","load");
}
return rv;
},unload:function(src,_61){
return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"unload",_61));
},change:function(src,_63){
return this.process(this.getId(src),new vjo.dsf.Event(src,"change",_63));
},submit:function(src,_65){
return this.process(this.getId(src),new vjo.dsf.Event(src,"submit",_65));
},reset:function(src,_67){
return this.process(this.getId(src),new vjo.dsf.Event(src,"reset",_67));
},select:function(src,_69){
return this.process(this.getId(src),new vjo.dsf.Event(src,"select",_69));
},blur:function(src,_6b){
return this.process(this.getId(src),new vjo.dsf.Event(src,"blur",_6b));
},focus:function(src,_6d){
return this.process(this.getId(src),new vjo.dsf.Event(src,"focus",_6d));
},keydown:function(src,_6f){
return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"keydown",_6f));
},keypress:function(src,_71){
return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"keypress",_71));
},keyup:function(src,_73){
return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"keyup",_73));
},click:function(src,_75){
return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"click",_75));
},dblclick:function(src,_77){
return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"dblclick",_77));
},mousedown:function(src,_79){
return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"mousedown",_79));
},mousemove:function(src,_7b){
return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"mousemove",_7b));
},mouseout:function(src,_7d){
return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"mouseout",_7d));
},mouseover:function(src,_7f){
return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"mouseover",_7f));
},mouseup:function(src,_81){
return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"mouseup",_81));
}}).inits(function(){
vjo.dsf.EventDispatcher=new vjo.dsf.EventDispatcher();
vjo.dsf.EventDispatcher.addEventListener(window,"load",function(){
vjo.dsf.EventDispatcher.addEventListener(window,"unload",function(){
vjo.dsf.EventDispatcher.cleanUp();
});
});
});

function HeaderTrk() { 	return { handle : function (event) { var _d = vjo.dsf.EventDispatcher, _r = vjo.Registry;
_r.put('HeaderTrackingCompSpecGenerator_0', new vjo.darwin.tracking.enabler.TrackingModuleEnabler('_trksid', 'm37'));
_d.add('BrowseCategories-menu','click',_r.get('HeaderTrackingCompSpecGenerator_0'));
_r.put('HeaderTrackingCompSpecGenerator_1', new vjo.darwin.tracking.enabler.TrackingModuleEnabler('_trksid', 'm37'));
_d.add('gnheader','click',_r.get('HeaderTrackingCompSpecGenerator_1'));
   } }; }
vjo.type("vjo.dsf.Message").protos({constructs:function(_1){
this.objType="dsf_Message";
this.svcId=_1;
this.request;
this.response;
this.rawRequest="";
this.clientContext={};
this.trspType="InProc";
this.status;
this.svcConfig;
this.returnData=true;
this.trace="";
}});

vjo.needs("vjo.dsf.Message");
vjo.type("vjo.darwin.core.overlaypanel.VjOverlayPanelOpenSvcHandler").protos({constructs:function(_1){
this.sOverlayPanelJsCompId=_1;
},invoke:function(_2){
var _3=vjo.Registry.get(this.sOverlayPanelJsCompId);
_3.fSetNotchLocation=_2.fSetNotchLocation;
_3.oSetNotchLocationOverrider=_2.oSetNotchLocationOverrider;
var _4=true;
if(typeof (_2.bOver)!="undefined"){
_4=_2.bOver;
}
_3.bCloseOnMouseOut=_4;
_3.setAnchorName(_2.sAnchorId);
if(_2.bResetPosition){
_3.bPanelOpen=false;
}
_3.onOpenOverlayPanel();
_2.returnData=false;
return _2;
}});

vjo.type("vjo.dsf.utils.Object").props({hitch:function(_1,_2){
var _3;
if(typeof _2=="string"){
_3=_1[_2];
}else{
_3=_2;
}
return function(){
return _3.apply(_1,arguments);
};
},extend:function(_4,_5){
function inheritance(){
}
inheritance.prototype=_5.prototype;
_4.prototype=new inheritance();
_4.prototype.constructor=_4;
_4.baseConstructor=_5;
_4.superClass=_5.prototype;
}});

vjo.needs("vjo.dsf.utils.Object");
vjo.needs("vjo.dsf.Message");
vjo.type("vjo.darwin.core.overlaypanel.VjOverlayPanelMessage").inherits("vjo.dsf.Message").protos({constructs:function(_1,_2){
this.base.call(this,_1);
this.request={};
this.sAnchorId="";
this.sBubbleId="";
this.response={};
this.bCheckState=false;
},setBubbleId:function(_3){
this.sBubbleId=_3;
},getBubbleId:function(){
return this.sBubbleId;
},setAnchorId:function(_4){
this.sAnchorId=_4;
},getAnchorId:function(){
return this.sAnchorId;
},getCheckState:function(){
return this.bCheckState;
}});

vjo.needs("vjo.dsf.Message");
vjo.needs("vjo.darwin.core.overlaypanel.VjOverlayPanelMessage");
vjo.type("vjo.darwin.core.overlaypanel.VjOverlayPanelEvtHandlers").props({openAtZIndex:5000,handleOpenEvent:function(_1,_2){
var _3=new vjo.darwin.core.overlaypanel.VjOverlayPanelMessage(_1);
_3.setAnchorId(_2);
return _3;
},handleCloseEvent:function(_4){
var _5=new vjo.darwin.core.overlaypanel.VjOverlayPanelMessage(_4);
return _5;
},handleCloseEventWithCheck:function(_6,_7){
var _8=new vjo.darwin.core.overlaypanel.VjOverlayPanelMessage(_6);
_8.bCheckState=_7;
return _8;
}});

vjo.needs("vjo.dsf.typeextensions.string.Comparison");
vjo.type("vjo.dsf.client.Browser").props({init:function(){
this.bFirefox=this.bWebTV=this.bOpera=this.bNav=this.bIE=this.bSafari=this.bWin=this.bMac=this.bMacppc=this.bMactel=this.bActiveXSupported=this.bWinXp=this.bXpSp2=this.bAOL=this.bVista=false;
this.iVer=this.fVer=-1;
this.fMinorVer=0;
this.aMimeTypes=null;
var nv=navigator,agt=nv.userAgent.toLowerCase(),i=0,ver;
with(this){
if(agt.has("webtv")){
bWebTV=true;
i=agt.indexOf("webtv/")+6;
}else{
if(agt.has("firefox")){
bFirefox=true;
i=agt.lastIndexOf("firefox")+8;
}else{
if(agt.has("safari")){
bSafari=true;
i=agt.lastIndexOf("safari")+7;
}else{
if(typeof (window.opera)!="undefined"){
bOpera=true;
i=agt.lastIndexOf("opera")+6;
}else{
if(nv.appName.is("Netscape")){
bNav=true;
i=agt.lastIndexOf("/")+1;
}else{
if(agt.has("msie")){
bIE=true;
i=agt.indexOf("msie")+4;
if(agt.has("aol")||agt.has("america online")){
bAOL=true;
}
}
}
}
}
}
}
ver=bOpera?window.opera.version():agt.substring(i);
iVer=parseInt(ver);
fVer=parseFloat(ver);
fMinorVer=fVer-iVer;
bWin=agt.has("win");
bWinXp=(bWin&&agt.has("windows nt 5.1"));
bVista=(bWin&&agt.has("windows nt 6.0"));
bXpSp2=(bWinXp&&agt.has("sv1"));
bMac=agt.has("mac");
bMacppc=(bMac&&agt.hasAny("ppc","powerpc"));
bMactel=(bMac&&agt.has("intel"));
aMimeTypes=nv.mimeTypes;
bActiveXSupported=(!(bMac||bMacppc)&&(typeof (ActiveXObject)=="function"));
}
}}).inits(function(){
vjo.dsf.client.Browser.init();
});

vjo.needs("vjo.dsf.client.Browser");
vjo.type("vjo.dsf.document.Element").props({get:function(_1){
var d=document,e=null;
if(d.getElementById){
e=d.getElementById(_1);
}
if(!e&&d.all){
e=d.all[_1];
}
return e;
},toggleHideShow:function(_3,_4){
var e=this.get(_3),s,d,u="undefined";
if(e){
s=e.style;
d=s.display;
if(typeof (_4)===u){
_4=(d===""||d==="block")?false:true;
}
e.bIsShown=_4;
s.display=(_4)?"block":"none";
}
},promoteToBody:function(_6){
var e=this.get(_6),b=document.body;
if(e&&b&&e.parentNode){
e.parentNode.removeChild(e);
b.appendChild(e);
}
},toggleVisibility:function(_8,_9){
var e=this.get(_8),v,s,u="undefined";
if(e){
s=e.style;
v=s.visibility;
if(typeof (_9)===u){
_9=(v==="")?false:true;
}
e.bIsVisible=_9;
s.visibility=(_9)?"":"hidden";
}
},toggleHideShowRow:function(_b,_c){
var e=this.get(_b),s,d,u="undefined";
var p=vjo.dsf.client.Browser.bFirefox?"table-row":"block";
if(e){
s=e.style;
d=s.display;
if(typeof (_c)===u){
_c=(d===""||d===p)?false:true;
}
e.bIsShown=_c;
s.display=(_c)?p:"none";
}
},enable:function(_f,_10){
var e=this.get(_f);
if(e){
e.disabled=!_10;
}
},left:function(_12,_13){
return this.setLTWH(_12,_13,"Left");
},top:function(_14,_15){
return this.setLTWH(_14,_15,"Top");
},width:function(_16,_17){
return this.setLTWH(_16,_17,"Width");
},height:function(_18,_19){
return this.setLTWH(_18,_19,"Height");
},setLTWH:function(_1a,_1b,_1c){
var e=this.get(_1a);
if(e){
if((_1b!=null)&&!isNaN(parseInt(_1b))){
e.style[_1c.toLowerCase()]=_1b;
}
return e["offset"+_1c];
}
},createElement:function(_1e){
return document.standardCreateElement?document.standardCreateElement(_1e):document.createElement(_1e);
},containsElement:function(_1f,_20){
while((_20!=null)&&(_20!=_1f)&&(_20.parentNode!=null)){
_20=_20.parentNode;
}
return (_20==_1f);
},getElementByTagClass:function(_21,tag,_23){
var _24=_21.getElementsByTagName(tag);
for(var ndx=0;((ndx<_24.length)&&(_24[ndx].className.match(_23)==null));ndx++){
}
return (ndx<_24.length)?_24[ndx]:null;
},getElementsByTagClass:function(_26,tag,_28){
var _29=new Array();
var _2a=_26.getElementsByTagName(tag);
for(var ndx=0;(ndx<_2a.length);ndx++){
if(_2a[ndx].className.match(_28)){
_29.push(_2a[ndx]);
}
}
return _29;
}});

vjo.type("vjo.dsf.document.Positioning").props({getScrollLeftTop:function(){
var d=document,rv=[0,0],db=d.body,de=d.documentElement;
if(db){
rv[0]+=db.scrollLeft;
rv[1]+=db.scrollTop;
}
if(de){
rv[0]+=de.scrollLeft;
rv[1]+=de.scrollTop;
}
return rv;
},getOffsetLeft:function(_2){
var e=_2,l=0;
while(e){
l+=e.offsetLeft;
e=e.offsetParent;
}
return l;
},getOffsetTop:function(_4){
var e=_4,t=0;
while(e){
t+=e.offsetTop;
e=e.offsetParent;
}
return t;
},getClientWidth:function(){
var s=self,d=document,de=d.documentElement,w;
if(s.innerWidth){
w=s.innerWidth;
}else{
if(de&&de.clientWidth){
w=de.clientWidth;
}else{
w=d.body.clientWidth;
}
}
return w;
},getClientHeight:function(){
var s=self,d=document,de=d.documentElement,h;
if(s.innerHeight){
h=s.innerHeight;
}else{
if(de&&de.clientHeight){
h=de.clientHeight;
}else{
h=d.body.clientHeight;
}
}
return h;
},getEventLeftTop:function(_8){
var u="undefined",evt=window.event||_8,xOff=(typeof (screenLeft)!=u)?screenLeft:screenX,yOff=(typeof (screenTop)!=u)?screenTop:(screenY+(outerHeight-innerHeight)-25);
return [evt.screenX-xOff,evt.screenY-yOff];
}});

vjo.needs("vjo.dsf.utils.Object");
vjo.type("vjo.dsf.utils.Timer").protos({timer:null,isRunning:false,interval:null,onTick:function(){
},onStart:null,onStop:null,constructs:function(_1){
this.interval=_1;
},setInterval:function(ms){
if(this.isRunning){
window.clearInterval(timer);
}
this.interval=ms;
if(this.isRunning){
this.timer=window.setInterval(vjo.dsf.utils.Object.hitch(this,"onTick"),this.interval);
}
},start:function(){
if(typeof this.onStart=="function"){
this.onStart();
}
this.isRunning=true;
this.timer=window.setInterval(vjo.dsf.utils.Object.hitch(this,"onTick"),this.interval);
},stop:function(){
if(typeof this.onStop=="function"){
this.onStop();
}
this.isRunning=false;
window.clearInterval(this.timer);
}});

vjo.type("vjo.darwin.core.bubble.VjAnchorUtils").props({getAnchorPosition:function(_1){
var _2=false;
var _3={};
var x=0,y=0;
var _5=false,use_css=false,use_layers=false;
if(document.getElementById){
_5=true;
}else{
if(document.all){
use_css=true;
}else{
if(document.layers){
use_layers=true;
}
}
}
if(_5&&document.all){
x=this.getPageOffsetLeft(document.all[_1]);
y=this.getPageOffsetTop(document.all[_1]);
}else{
if(_5){
var o=document.getElementById(_1);
x=this.getPageOffsetLeft(o);
y=this.getPageOffsetTop(o);
}else{
if(use_css){
x=this.getPageOffsetLeft(document.all[_1]);
y=this.getPageOffsetTop(document.all[_1]);
}else{
if(use_layers){
var _7=0;
for(var i=0;i<document.anchors.length;i++){
if(document.anchors[i].name==_1){
_7=1;
break;
}
}
if(_7===0){
_3.x=0;
_3.y=0;
return _3;
}
x=document.anchors[i].x;
y=document.anchors[i].y;
}else{
_3.x=0;
_3.y=0;
return _3;
}
}
}
}
_3.x=x;
_3.y=y;
return _3;
},getAnchorWindowPosition:function(_9){
var _a=this.getAnchorPosition(_9);
var x=0;
var y=0;
if(document.getElementById){
if(isNaN(window.screenX)){
x=_a.x-document.body.scrollLeft+window.screenLeft;
y=_a.y-document.body.scrollTop+window.screenTop;
}else{
x=_a.x+window.screenX+(window.outerWidth-window.innerWidth)-window.pageXOffset;
y=_a.y+window.screenY+(window.outerHeight-24-window.innerHeight)-window.pageYOffset;
}
}else{
if(document.all){
x=_a.x-document.body.scrollLeft+window.screenLeft;
y=_a.y-document.body.scrollTop+window.screenTop;
}else{
if(document.layers){
x=_a.x+window.screenX+(window.outerWidth-window.innerWidth)-window.pageXOffset;
y=_a.y+window.screenY+(window.outerHeight-24-window.innerHeight)-window.pageYOffset;
}
}
}
_a.x=x;
_a.y=y;
return _a;
},getPageOffsetLeft:function(el){
var ol=el.offsetLeft;
while((el=el.offsetParent)!==null){
ol+=el.offsetLeft;
}
return ol;
},getWindowOffsetLeft:function(el){
return AnchorPosition_getPageOffsetLeft(el)-document.body.scrollLeft;
},getPageOffsetTop:function(el){
var ot=el.offsetTop;
while((el=el.offsetParent)!==null){
ot+=el.offsetTop;
}
return ot;
},getWindowOffsetTop:function(el){
return AnchorPosition_getPageOffsetTop(el)-document.body.scrollTop;
}});

vjo.type("vjo.dsf.window.utils.VjWindowUtils").props({getBrowserWindowHeight:function(){
var s=self;
var d=document;
var de=d.documentElement;
if(s.innerHeight){
return s.innerHeight;
}else{
if(de&&de.clientHeight){
return de.clientHeight;
}
}
return d.body.clientHeight;
},getBrowserWindowWidth:function(){
var s=self;
var d=document;
var de=d.documentElement;
if(s.innerWidth){
return s.innerWidth;
}else{
if(de&&de.clientWidth){
return de.clientWidth;
}
}
return d.body.clientWidth;
},getScrollXY:function(){
var _7=0,scrOfY=0;
if(typeof (window.pageYOffset)=="number"){
scrOfY=window.pageYOffset;
_7=window.pageXOffset;
}else{
if(document.body&&(document.body.scrollLeft||document.body.scrollTop)){
scrOfY=document.body.scrollTop;
_7=document.body.scrollLeft;
}else{
if(document.documentElement&&(document.documentElement.scrollLeft||document.documentElement.scrollTop)){
scrOfY=document.documentElement.scrollTop;
_7=document.documentElement.scrollLeft;
}
}
}
return [_7,scrOfY];
},toPixels:function(_8){
return _8+"px";
},scrollTop:function(){
if(window.pageYOffset!=null){
return window.pageYOffset;
}
if(document.documentElement){
return Math.max(document.documentElement.scrollTop,document.body.scrollTop);
}else{
return document.body.scrollTop;
}
},scrollLeft:function(){
if(window.pageXOffset!=null){
return window.pageXOffset;
}
if(document.documentElement){
return Math.max(document.documentElement.scrollLeft,document.body.scrollLeft);
}else{
return document.body.scrollLeft;
}
},scrollWidth:function(){
if(document.documentElement){
return document.documentElement.scrollWidth;
}else{
return Math.max(document.body.scrollWidth,document.body.offsetWidth);
}
},scrollHeight:function(){
if(document.documentElement){
return document.documentElement.scrollHeight;
}else{
return Math.max(document.body.scrollHeight,document.body.offsetHeight);
}
},clientTop:function(){
if(document.documentElement){
return document.documentElement.clientTop;
}else{
return document.body.clientTop;
}
},clientLeft:function(){
if(document.documentElement){
return document.documentElement.clientLeft;
}else{
return document.body.clientLeft;
}
},clientWidth:function(){
var _9=document.documentElement;
if(_9&&window.innerWidth){
return Math.min(_9.clientWidth,window.innerWidth);
}else{
if(_9){
return _9.clientWidth;
}else{
if(window.innerWidth){
return window.innerWidth;
}else{
if(document.body.clientWidth){
return document.body.clientWidth;
}else{
return document.body.offsetWidth;
}
}
}
}
},clientHeight:function(){
var _a=document.documentElement;
if(_a&&window.innerHeight){
return Math.min(_a.clientHeight,window.innerHeight);
}else{
if(_a){
return _a.clientHeight;
}else{
if(window.innerHeight){
return window.innerHeight;
}else{
if(document.body.clientHeight){
return document.body.clientHeight;
}else{
return document.body.offsetHeight;
}
}
}
}
},browserTop:function(){
return (window.innerHeight)?window.screenY+(window.outerHeight-window.innerHeight):window.screenTop;
},browserLeft:function(){
return (window.innerWidth)?window.screenX+(window.outerWidth-window.innerWidth):window.screenLeft;
},eventTop:function(_b){
if(_b.pageY!=null){
return _b.pageY;
}
if(document.documentElement){
return _b.clientY+Math.max(document.documentElement.scrollTop,document.body.scrollTop);
}else{
return _b.clientY+document.body.scrollTop;
}
},eventLeft:function(_c){
if(_c.pageX!=null){
return _c.pageX;
}
if(document.documentElement){
return _c.clientX+Math.max(document.documentElement.scrollLeft,document.body.scrollLeft);
}else{
return _c.clientX+document.body.scrollLeft;
}
},offsetTop:function(_d){
var _e=(document.documentElement&&document.documentElement.clientTop)?document.documentElement.clientTop:0;
for(var _f=0;(_d!=null);_d=_d.offsetParent){
_f+=_d.offsetTop;
}
return _f+_e;
},offsetLeft:function(_10){
var _11=(document.documentElement&&document.documentElement.clientLeft)?document.documentElement.clientLeft:0;
for(var _12=0;(_10!=null);_10=_10.offsetParent){
_12+=_10.offsetLeft;
}
return _12+_11;
},openWindow:function(url,_14,_15){
var _16=new Array();
var _17=vjo.dsf.window.utils.VjWindowUtils;
_15.top=_17.browserTop()+Math.round((_17.clientHeight()-_15.height)/2)+25;
_15.left=_17.browserLeft()+Math.round((_17.clientWidth()-_15.width)/2);
for(var key in _15){
_16.push(key.concat("=",_15[key]));
}
return window.open(url,_14,_16.join(","),true);
}});

vjo.type("vjo.dsf.RemoteReqtHdl").protos({constructs:function(){
this.reqTimers={};
this.timerCount=0;
this.processed={};
},handleRequest:function(_1){
_1.trace=_1.trace+"-->RemoteHdl_"+_1.svcId;
if(_1.svcConfig){
this.invoke(_1);
}
},invoke:function(_2){
var _3=vjo.dsf.Service,xmlHttpReq=_3.getXmlHttpReq(),requestUrl=_2.svcConfig.url,cfg=_2.svcConfig;
_2.status=-1;
if(cfg.respMarshalling=="JSCALLBACK"){
vjo.dsf.XDomainRequest.send(_2);
return;
}
try{
var _4=(cfg.async===false)?false:true;
xmlHttpReq.open(cfg.method,requestUrl,_4);
}
catch(e){
var _5=new vjo.dsf.ServiceResponse();
var _6=new vjo.dsf.Error();
_6.id="SYS.DARWIN_SERVICE_PROTOCOL_ERROR";
_6.message="SYS.PROTOCOL_ERROR: Cannot open URL '"+requestUrl+"'";
_5.errors=[_6];
_2.response=_5;
vjo.dsf.ServiceEngine.handleResponse(_2);
return;
}
var _7=this.timerCount++;
this.setupReadyState(xmlHttpReq,_2,_7);
if(cfg.method=="POST"){
xmlHttpReq.setRequestHeader("Content-Type","application/x-www-form-urlencoded");
xmlHttpReq.setRequestHeader("Content-Length",_2.rawRequest.length);
xmlHttpReq.send(_2.rawRequest);
}else{
xmlHttpReq.send(null);
}
if(cfg.timeout){
var _8=this;
this.reqTimers[_7]=window.setTimeout(function(){
_8.timeout(xmlHttpReq,_2,_7);
},cfg.timeout);
}
},setupReadyState:function(_9,_a,_b){
var _c=this;
_9.onreadystatechange=function(){
if(_9.readyState!=4){
return;
}
if(_c.processed[_b]){
return;
}
_c.processed[_b]=true;
var _d=_c.reqTimers[_b];
if(_d){
window.clearTimeout(_d);
delete _c.reqTimers[_b];
}
vjo.dsf.Service.callback(_9,_a);
};
},timeout:function(_e,_f,idx){
if(this.processed[idx]){
return;
}
this.processed[idx]=true;
delete _e.onreadystatechange;
_e.abort();
delete this.reqTimers[idx];
var _11=new vjo.dsf.ServiceResponse();
var _12=new vjo.dsf.Error();
_12.id="SYS.DARWIN_SERVICE_PROTOCOL_ERROR";
_12.message="SYS.PROTOCOL_ERROR: Service timed out.";
_11.errors=[_12];
_f.response=_11;
vjo.dsf.ServiceEngine.handleResponse(_f);
}});

vjo.type("vjo.dsf.ServiceResponse").protos({constructs:function(){
this.objType="dsf_ServiceResponse";
this.errors=[];
this.data=null;
}});

vjo.needs("vjo.dsf.ServiceResponse");
vjo.type("vjo.dsf.InProcReqtHdl").protos({constructs:function(){
this.svcHdls={};
},registerSvcHdl:function(_1,_2){
this.svcHdls[_1]=_2;
},getSvcHdl:function(_3){
return this.svcHdls[_3];
},handleRequest:function(_4){
var _5=this.svcHdls[_4.svcId];
if(_5){
var _6=new vjo.dsf.ServiceResponse();
_6.data=_5.invoke(_4);
_4.trace=_4.trace+"-->SvcHdl_"+_4.svcId;
if(_6){
_4.response=_6;
}
}
if(typeof _4.status=="undefined"||_4.status==null){
_4.status=1;
}
}});

vjo.type("vjo.dsf.Error").protos({constructs:function(_1,_2){
this.id=_1;
this.message=_2;
}});

vjo.needs("vjo.dsf.Error");
vjo.needs("vjo.dsf.ServiceResponse");
vjo.type("vjo.dsf.Service").props({callback:function(_1,_2){
try{
if(_1.readyState!=4){
return;
}
if(_1.status>=200&&_1.status<300){
var _3=_1.responseText;
if(_2.svcConfig.respMarshalling=="JSON"){
try{
_3=eval("("+_3+")");
}
catch(e){
_3=new vjo.dsf.ServiceResponse();
var _4=new vjo.dsf.Error();
_4.id="SYS.JSON_PARSE_ERROR";
_4.message="SYS.JSON_PARSE_ERROR";
_3.errors=[_4];
}
}else{
if(_2.svcConfig.respMarshalling=="XML"){
_3=_1.responseXML;
}
}
_2.response=_3;
_2.status=1;
}else{
var _5=new vjo.dsf.ServiceResponse();
var _4=new vjo.dsf.Error();
_4.id="SYS.DARWIN_SERVICE_PROTOCOL_ERROR";
_4.message="SYS.PROTOCOL_ERROR: status = "+_1.status;
_5.errors=[_4];
_2.response=_5;
}
}
catch(e){
var _5=new vjo.dsf.ServiceResponse();
var _4=new vjo.dsf.Error();
_4.id="SYS.DARWIN_SERVICE_PROTOCOL_ERROR";
_4.message="SYS.PROTOCOL_ERROR: unknown ";
_5.errors=[_4];
_2.response=_5;
}
vjo.dsf.ServiceEngine.handleResponse(_2);
delete _1.onreadystatechange;
_1=null;
},getXmlHttpReq:function(){
var _6=false;
try{
_6=new ActiveXObject("Msxml2.XMLHTTP");
}
catch(e){
try{
_6=new ActiveXObject("Microsoft.XMLHTTP");
}
catch(e){
_6=false;
}
}
if(!_6&&typeof XMLHttpRequest!="undefined"){
_6=new XMLHttpRequest();
}
return _6;
},generateReqParams:function(_7){
var _8="svcid="+encodeURIComponent(_7.svcId);
if(_7.stok){
_8+="&stok="+_7.stok;
}
_8=_8+"&reqttype="+_7.svcConfig.reqtMarshalling;
_8=_8+"&resptype="+_7.svcConfig.respMarshalling;
_8+="&request=";
var _9=_7.request,reqtmarsh=_7.svcConfig.reqtMarshalling;
if(reqtmarsh=="JSON"){
_8+=encodeURIComponent(JSON.stringify(_9));
}else{
if(reqtmarsh=="JSCALLBACK"){
_8+=encodeURIComponent(JSON.stringify(_9));
}else{
if(reqtmarsh=="XML"){
_8+=encodeURIComponent(dsf_xmlize(_9,"Request"));
}else{
_8+=encodeURIComponent(_9);
}
}
}
return _8;
},xmlize:function(_a,_b,_c){
_c=_c?_c:"";
var s=_c+"<"+_b+">";
if(!(_a instanceof Object)||_a instanceof Number||_a instanceof String||_a instanceof Boolean||_a instanceof Date){
s+=dsf_escape(""+_a);
}else{
s+="\n";
var _e="";
var _f=_a instanceof Array;
for(var _10 in _a){
if(_f&&_10=="______array"){
continue;
}
s+=this.xmlize(_a[_10],(_f?"array-item key=\""+_10+"\"":_10),_c+"   ");
}
s+=_c;
}
return s+=(_b.indexOf(" ")!=-1?"</array-item>\n":"</"+_b+">\n");
},escape:function(_11){
return _11.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&apos;");
}});

vjo.type("vjo.dsf.SvcConfig").protos({constructs:function(_1,_2){
this.objType="dsf_SvcConfig";
this.url=_2;
this.method=_1;
this.reqtMarshalling="raw";
this.respMarshalling="raw";
this.async=true;
this.timeout=0;
}});

vjo.type("vjo.dsf.XDomainRequest").protos({constructs:function(){
this.callbacks=[];
this.sCallbackName="callback";
this.sPreId="xdr_";
this.bUseIframe=(navigator.userAgent.indexOf("Firefox")>0);
vjo.dsf.EventDispatcher.addEventListener(window,"load",this.onLoad,this);
},onLoad:function(){
this.bodyLoaded=true;
},getReqDiv:function(){
return document.getElementsByTagName(this.bodyLoaded?"body":"head")[0];
},send:function(_1){
if(!document.createElement||!_1){
return;
}
var _2="",eid="";
if(typeof _1=="string"){
_2=_1;
}else{
if(_1.objType=="dsf_Message"&&_1.svcConfig){
var cb=this.createCallback(_1);
eid=this.sPreId+this.callbacks[this.callbacks.length-1];
_2=_1.svcConfig.url+"&callback="+cb;
}
}
var _4=null,doc;
if(this.bUseIframe){
var _5="<scr"+"ipt src=\""+_2+"\" type=\"text/javascript\"></scr"+"ipt>";
_4=document.createElement("iframe");
_4.height=1;
_4.width=1;
_4.id=eid;
_4.style.display="none";
this.getReqDiv().appendChild(_4);
doc=_4.document||_4.contentDocument;
doc.open();
doc.write("<html><head></head><body>"+_5+"</body></html>");
doc.close();
}else{
doc=document;
var _6=doc.createElement("script");
_6.id=eid;
_6.type="text/javascript";
_6.src=_2;
this.getReqDiv().appendChild(_6);
}
return eid;
},createCallback:function(_7){
var _8=this.callbacks.length,name=this.sCallbackName+_8,eid=this.sPreId+name;
this.callbacks[_8]=name;
this[name]=function(_9){
var _a;
try{
_a=_9;
}
catch(e){
_a=new vjo.dsf.ServiceResponse();
var _b=new vjo.dsf.Error();
_b.id="SYS.JSON_PARSE_ERROR";
_b.message="SYS.JSON_PARSE_ERROR";
_a.errors=[_b];
}
this.loaded(eid);
_7.response=_a;
vjo.dsf.ServiceEngine.handleResponse(_7);
};
var rv="vjo.dsf.XDomainRequest."+name;
if(this.bUseIframe){
rv="parent."+rv;
}
return rv;
},loaded:function(_d){
var e=document.getElementById(_d);
if(e!=null){
e.parentNode.removeChild(e);
}
}}).inits(function(){
vjo.dsf.XDomainRequest=new vjo.dsf.XDomainRequest();
});

// @JsDoNotOptimize
// @JsDoNotLintValidate
// @Package vjo.dsf

var JSON = {
    org: 'http://www.JSON.org',
    copyright: '(c)2005 JSON.org',
    license: 'http://www.crockford.com/JSON/license.html',

    stringify: function (arg) {
        var c, i, l, s = '', v;

        switch (typeof arg) {
        case 'object':
            if (arg) {
                if (arg instanceof Array) {
                    for (i = 0; i < arg.length; ++i) {
                        v = this.stringify(arg[i]);
                        if (s) {
                            s += ',';
                        }
                        s += v;
                    }
                    return '[' + s + ']';
                } else if (typeof arg.toString != 'undefined') {
                    for (i in arg) {
                        v = arg[i];
                        if (typeof v != 'undefined' && typeof v != 'function') {
                            v = this.stringify(v);
                            if (s) {
                                s += ',';
                            }
                            s += this.stringify(i) + ':' + v;
                        }
                    }
                    return '{' + s + '}';
                }
            }
            return 'null';
        case 'number':
            return isFinite(arg) ? String(arg) : 'null';
        case 'string':
            l = arg.length;
            s = '"';
            for (i = 0; i < l; i += 1) {
                c = arg.charAt(i);
                if (c >= ' ') {
                    if (c == '\\' || c == '"') {
                        s += '\\';
                    }
                    s += c;
                } else {
                    switch (c) {
                        case '\b':
                            s += '\\b';
                            break;
                        case '\f':
                            s += '\\f';
                            break;
                        case '\n':
                            s += '\\n';
                            break;
                        case '\r':
                            s += '\\r';
                            break;
                        case '\t':
                            s += '\\t';
                            break;
                        default:
                            c = c.charCodeAt();
                            s += '\\u00' + Math.floor(c / 16).toString(16) +
                                (c % 16).toString(16);
                    }
                }
            }
            return s + '"';
        case 'boolean':
            return String(arg);
        default:
            return 'null';
        }
    },
    parse: function (text) {
        var at = 0;
        var ch = ' ';

        function error(m) {
            throw {
                name: 'JSONError',
                message: m,
                at: at - 1,
                text: text
            };
        }

        function next() {
            ch = text.charAt(at);
            at += 1;
            return ch;
        }

        function white() {
            while (ch !== '' && ch <= ' ') {
                next();
            }
        }

        function str() {
            var i, s = '', t, u;

            if (ch == '"') {
outer:          while (next()) {
                    if (ch == '"') {
                        next();
                        return s;
                    } else if (ch == '\\') {
                        switch (next()) {
                        case 'b':
                            s += '\b';
                            break;
                        case 'f':
                            s += '\f';
                            break;
                        case 'n':
                            s += '\n';
                            break;
                        case 'r':
                            s += '\r';
                            break;
                        case 't':
                            s += '\t';
                            break;
                        case 'u':
                            u = 0;
                            for (i = 0; i < 4; i += 1) {
                                t = parseInt(next(), 16);
                                if (!isFinite(t)) {
                                    break outer;
                                }
                                u = u * 16 + t;
                            }
                            s += String.fromCharCode(u);
                            break;
                        default:
                            s += ch;
                        }
                    } else {
                        s += ch;
                    }
                }
            }
            error("Bad string");
        }

        function arr() {
            var a = [];

            if (ch == '[') {
                next();
                white();
                if (ch == ']') {
                    next();
                    return a;
                }
                while (ch) {
                    a.push(val());
                    white();
                    if (ch == ']') {
                        next();
                        return a;
                    } else if (ch != ',') {
                        break;
                    }
                    next();
                    white();
                }
            }
            error("Bad array");
        }

        function obj() {
            var k, o = {};

            if (ch == '{') {
                next();
                white();
                if (ch == '}') {
                    next();
                    return o;
                }
                while (ch) {
                    k = str();
                    white();
                    if (ch != ':') {
                        break;
                    }
                    next();
                    o[k] = val();
                    white();
                    if (ch == '}') {
                        next();
                        return o;
                    } else if (ch != ',') {
                        break;
                    }
                    next();
                    white();
                }
            }
            error("Bad object");
        }

        function num() {
            var n = '', v;
            if (ch == '-') {
                n = '-';
                next();
            }
            while (ch >= '0' && ch <= '9') {
                n += ch;
                next();
            }
            if (ch == '.') {
                n += '.';
                while (next() && ch >= '0' && ch <= '9') {
                    n += ch;
                }
            }
            if (ch == 'e' || ch == 'E') {
                n += 'e';
                next();
                if (ch == '-' || ch == '+') {
                    n += ch;
                    next();
                }
                while (ch >= '0' && ch <= '9') {
                    n += ch;
                    next();
                }
            }
            v = +n;
            if (!isFinite(v)) {
                error("Bad number");
            } else {
                return v;
            }
        }

        function word() {
            switch (ch) {
                case 't':
                    if (next() == 'r' && next() == 'u' && next() == 'e') {
                        next();
                        return true;
                    }
                    break;
                case 'f':
                    if (next() == 'a' && next() == 'l' && next() == 's' &&
                            next() == 'e') {
                        next();
                        return false;
                    }
                    break;
                case 'n':
                    if (next() == 'u' && next() == 'l' && next() == 'l') {
                        next();
                        return null;
                    }
                    break;
            }
            error("Syntax error");
        }

        function val() {
            white();
            switch (ch) {
                case '{':
                    return obj();
                case '[':
                    return arr();
                case '"':
                    return str();
                case '-':
                    return num();
                default:
                    return ch >= '0' && ch <= '9' ? num() : word();
            }
        }

        return val();
    }
};

vjo.needs("vjo.dsf.RemoteReqtHdl");
vjo.needs("vjo.dsf.InProcReqtHdl");
vjo.needs("vjo.dsf.Service");
vjo.needs("vjo.dsf.ServiceResponse");
vjo.needs("vjo.dsf.SvcConfig");
vjo.needs("vjo.dsf.XDomainRequest");
vjo.needs("vjo.dsf.Message");
vjo.needs("vjo.dsf.Json");
vjo.type("vjo.dsf.ServiceEngine").singleton().protos({STATUS:{ABORT:-1,JUMP:1},constructs:function(){
this.svcReqtHdls={};
this.svcRespHdls={};
this.glbReqtHdls=[];
this.glbRespHdls=[];
this.trspReqtHdls={};
this.trspRespHdls={};
this.svcHdls={};
this.inProcHdl=new vjo.dsf.InProcReqtHdl();
this.remoteHdl=new vjo.dsf.RemoteReqtHdl();
this.registerTrspReqtHdl("Remote",this.remoteTrspHdl);
},handleRequest:function(_1){
var _2;
if(_1.status!=this.STATUS.ABORT&&typeof _2=="undefined"){
_2=this.processServiceRequestHandlers(_1);
}
if(_1.status!=this.STATUS.ABORT&&typeof _2=="undefined"){
_2=this.processGlobalRequestHandlers(_1);
}
if(_1.status!=this.STATUS.ABORT&&typeof _2=="undefined"){
_2=this.processTransportHandlers(_1);
}
if(_1.status!=this.STATUS.ABORT&&(_1.trspType!="Remote"||typeof _2!="undefined")){
this.handleResponse(_1,_2);
}
return _1.returnData;
},processServiceRequestHandlers:function(_3){
var _4;
var _5=this.svcReqtHdls[_3.svcId];
if(_5){
try{
for(var i=0;i<_5.length;i++){
_3.trace=_3.trace+"-->svcReqtHdl_"+i;
_5[i].handleRequest(_3);
if(_3.status==this.STATUS.JUMP){
_4="SVC";
this.genResponseError(_3,"SYS.SVC_REQUEST_ERROR","SYS.SVC_REQUEST_ERROR");
break;
}
}
}
catch(e){
_4="SVC";
this.genResponseError(_3,"SYS.SVC_REQUEST_ERROR","SYS.SVC_REQUEST_ERROR");
}
}
return _4;
},processGlobalRequestHandlers:function(_7){
var _8;
if(_7.status!=this.STATUS.JUMP){
try{
for(var i=0;i<this.glbReqtHdls.length;i++){
_7.trace=_7.trace+"-->glbReqtHdl_"+i;
this.glbReqtHdls[i].handleRequest(_7);
if(_7.status==this.STATUS.JUMP){
_8="GLB";
this.genResponseError(_7,"SYS.GLOBAL_REQUEST_ERROR","SYS.GLOBAL_REQUEST_ERROR");
break;
}
}
}
catch(e){
_8="GLB";
this.genResponseError(_7,"SYS.GLOBAL_REQUEST_ERROR","SYS.GLOBAL_REQUEST_ERROR");
}
}
return _8;
},processTransportHandlers:function(_a){
var _b;
if(_a.status!=this.STATUS.JUMP&&_a.trspType){
var _c=this.trspReqtHdls[_a.trspType];
if(_c){
try{
for(var i=0;i<_c.length;i++){
_a.trace=_a.trace+"-->trspReqtHdl_"+i;
_c[i].handleRequest(_a);
if(_a.status==this.STATUS.JUMP){
this.genResponseError(_a,"SYS.TRANS_REQUEST_ERROR","SYS.TRANS_REQUEST_ERROR");
break;
}
}
}
catch(e){
this.genResponseError(_a,"SYS.TRANS_REQUEST_ERROR","SYS.TRANS_REQUEST_ERROR");
}
}
if(_a.status!=this.STATUS.JUMP&&_a.status!=this.STATUS.ABORT){
if(_a.trspType==="Remote"){
this.remoteHdl.handleRequest(_a);
}else{
this.inProcHdl.handleRequest(_a);
}
}
}
return _b;
},handleResponse:function(_e,_f){
if(_e.trspType&&typeof _f=="undefined"){
this.processTransResponseHandlers(_e);
}
if(_f!="SVC"){
this.processGlobalResponseHandlers(_e);
}
this.processServiceResponseHandlers(_e);
},processTransResponseHandlers:function(msg){
var _11=this.trspRespHdls[msg.trspType];
try{
if(_11){
for(var i=_11.length-1;i>=0;i--){
msg.trace=msg.trace+"-->trspRespHdl_"+i;
_11[i].handleResponse(msg);
}
}
}
catch(e){
this.genResponseError(msg,"SYS.TRANS_RESPONSE_ERROR","SYS.TRANS_RESPONSE_ERROR");
}
},processGlobalResponseHandlers:function(msg){
try{
for(var i=this.glbRespHdls.length-1;i>=0;i--){
msg.trace=msg.trace+"-->glbRespHdl_"+i;
this.glbRespHdls[i].handleResponse(msg);
}
}
catch(e){
this.genResponseError(msg,"SYS.GLOB_RESPONSE_ERROR","SYS.GLOB_RESPONSE_ERROR");
}
},processServiceResponseHandlers:function(msg){
var _16;
if(msg.clientContext){
_16=msg.clientContext.svcApplier;
}
try{
if(_16){
if(typeof _16.onResponse=="function"){
_16.onResponse(msg);
}else{
if(typeof _16=="function"){
_16(msg);
}
}
}
}
catch(e){
this.genResponseError(msg,"SYS.SVC_RESPONSE_ERROR","SYS.SVC_RESPONSE_ERROR");
}
var _17=this.svcRespHdls[msg.svcId];
if(_17){
try{
for(var i=_17.length-1;i>=0;i--){
msg.trace=msg.trace+"-->svcRespHdl_"+i;
_17[i].handleResponse(msg);
}
}
catch(e){
this.genResponseError(msg,"SYS.SVC_RESPONSE_ERROR","SYS.SVC_RESPONSE_ERROR");
}
}
},createHandler:function(_19,_1a){
if(typeof _19[_1a]!="function"){
if(typeof _19=="function"){
var _1b=_19;
var obj={};
obj[_1a]=function(){
return _1b.apply(this,arguments);
};
_19=obj;
}
}
return _19;
},registerSvcHdl:function(_1d,_1e){
if(!_1d||!_1e){
return;
}
_1e=this.createHandler(_1e,"invoke");
this.inProcHdl.registerSvcHdl(_1d,_1e);
},getSvcHdl:function(_1f){
return this.inProcHdl.getSvcHdl(_1f);
},registerSvcReqtHdl:function(_20,_21){
if(!_20||!_21){
return;
}
if(typeof this.svcReqtHdls[_20]=="undefined"){
this.svcReqtHdls[_20]=[];
}
var _22=this.svcReqtHdls[_20];
_22[_22.length]=this.createHandler(_21,"handleRequest");
},registerSvcRespHdl:function(_23,_24){
if(!_23||!_24){
return;
}
if(typeof this.svcRespHdls[_23]=="undefined"){
this.svcRespHdls[_23]=[];
}
var _25=this.svcRespHdls[_23];
_25[_25.length]=this.createHandler(_24,"handleResponse");
},registerGlbReqtHdl:function(_26){
if(!_26){
return;
}
this.glbReqtHdls[this.glbReqtHdls.length]=this.createHandler(_26,"handleRequest");
},registerGlbRespHdl:function(_27){
if(!_27){
return;
}
this.glbRespHdls[this.glbRespHdls.length]=this.createHandler(_27,"handleResponse");
},registerTrspReqtHdl:function(_28,_29){
if(!_28||!_29){
return;
}
if(typeof this.trspReqtHdls[_28]=="undefined"){
this.trspReqtHdls[_28]=[];
}
var _2a=this.trspReqtHdls[_28];
_2a[_2a.length]=this.createHandler(_29,"handleRequest");
},registerTrspRespHdl:function(_2b,_2c){
if(!_2b||!_2c){
return;
}
if(typeof this.trspRespHdls[_2b]=="undefined"){
this.trspRespHdls[_2b]=[];
}
var _2d=this.trspRespHdls[_2b];
_2d[_2d.length]=this.createHandler(_2c,"handleResponse");
},remoteTrspHdl:function(_2e){
var cfg=_2e.svcConfig;
if(!cfg||cfg.objType!="dsf_SvcConfig"){
return;
}else{
if(cfg.respMarshalling=="JSCALLBACK"){
if(typeof vjo.dsf.assembly!="undefined"&&typeof vjo.dsf.assembly.VjClientAssembler!="undefined"&&!vjo.dsf.assembly.VjClientAssembler.bBodyLoaded){
vjo.dsf.assembly.VjClientAssembler.load(_2e);
_2e.status=-1;
return;
}
}
}
var svc=vjo.dsf.Service,requestParams=svc.generateReqParams(_2e),requestUrl=cfg.url;
if(_2e.svcConfig.method=="GET"){
requestUrl=requestUrl+"?"+requestParams;
}else{
_2e.rawRequest=requestParams;
}
_2e.svcConfig.url=requestUrl;
},genResponseError:function(msg,_32,_33){
if(typeof msg.response=="undefined"){
var _34=new vjo.dsf.ServiceResponse();
msg.response=_34;
}
var _35=new vjo.dsf.Error(_32,_33);
msg.response.errors[msg.response.errors.length]=_35;
}}).inits(function(){
vjo.dsf.ServiceEngine=new vjo.dsf.ServiceEngine();
});

vjo.needs("vjo.dsf.client.Browser");
vjo.type("vjo.dsf.document.Shim").props({add:function(_1,_2,_3){
var f,p="px",w,h,s;
if(this.check()){
w=_1.offsetWidth;
h=_1.offsetHeight;
w+=_2?_2:0;
h+=_3?_3:0;
f=document.createElement("IFRAME");
s=f.style;
s.width=w+p;
s.height=h+p;
s.filter="chroma(color='white')";
f.frameBorder=0;
s.position="absolute";
s.left="0"+p;
s.top="0"+p;
s.zIndex="-1";
s.filter="Alpha(Opacity=\"0\")";
if(document.location.protocol=="https:"){
f.src="https://securepics.ebaystatic.com/aw/pics/s.gif";
}
_1.appendChild(f);
return f;
}
},remove:function(_5,_6){
if(this.check()){
if(_6&&_6.parentNode){
_6.parentNode.removeChild(_6);
}
}
},check:function(){
var B=vjo.dsf.client.Browser;
return (B.bIE||B.bFirefox);
}});

vjo.needs("vjo.dsf.utils.Timer");
vjo.type("vjo.darwin.core.overlaypanel.VjOverlayPanelController").singleton().protos({constructs:function(){
var t=this;
t.eD=vjo.dsf.EventDispatcher;
t.aOlps=[];
t.sIsActivated=false;
t.eD.addEventListener(window,"resize",t.initResize,t);
},registerOverlayPanel:function(_2){
var t=this;
t.aOlps[t.aOlps.length]=_2;
},requireResize:function(){
this.sIsActivated=true;
},initResize:function(){
var t=this;
t.eD.removeEventListener(window,"resize",t.initResize);
t.eD.addEventListener(window,"resize",t.requireResize,t);
t.requireResize();
t.oTimer=new vjo.dsf.utils.Timer();
t.oTimer.interval=500;
t.oTimer.onTick=function(){
if(t.sIsActivated){
for(var i=0;i<t.aOlps.length;i++){
if(t.aOlps[i].bAdjustSize&&t.aOlps[i].bPanelOpen){
t.aOlps[i].onResize();
}
}
t.sIsActivated=false;
}
};
t.oTimer.start();
},resizeOlps:function(){
var i=0,t=this;
for(;i<t.aOlps.length;i++){
if(t.aOlps[i].bPanelOpen){
t.aOlps[i].onResize();
}
}
}}).inits(function(){
vjo.darwin.core.overlaypanel.VjOverlayPanelController=new vjo.darwin.core.overlaypanel.VjOverlayPanelController();
});

vjo.needs("vjo.dsf.document.Element");
vjo.needs("vjo.dsf.document.Positioning");
vjo.needs("vjo.dsf.client.Browser");
vjo.needs("vjo.dsf.utils.Object");
vjo.needs("vjo.dsf.utils.Timer");
vjo.needs("vjo.dsf.Message");
vjo.needs("vjo.darwin.core.bubble.VjAnchorUtils");
vjo.needs("vjo.dsf.window.utils.VjWindowUtils");
vjo.needs("vjo.dsf.ServiceEngine");
vjo.needs("vjo.dsf.document.Shim");
vjo.needs("vjo.darwin.core.overlaypanel.VjOverlayPanelController");
vjo.type("vjo.darwin.core.overlaypanel.VjOverlayPanel").inherits("vjo.dsf.utils.Timer").protos({constructs:function(_1){
this.interval=50;
this.sOverlayDivId=_1.overlayCompId;
this.sArrowTL=_1.arrowTL;
this.sArrowTR=_1.arrowTR;
this.sArrowBR=_1.arrowBR;
this.sArrowBL=_1.arrowBL;
this.iLastLeft=-1;
this.iLastTop=-1;
this.initDx=0;
this.initDy=0;
this.top=0;
this.left=0;
this.bShownInCenter=_1.isShownInCenter;
this.leftOffset=_1.leftOffset;
this.topOffset=_1.topOffset;
this.sHAlign=_1.horizontalAlign||false;
this.sVAlign=_1.verticalAlign||false;
this.iShadowOffset=_1.shadowOffset;
this.iOpenDelay=_1.openDelay;
this.iCloseDelay=_1.closeDelay;
this.bCloseOnMouseOut=_1.isCloseOnMouseOut;
this.bHasMask=_1.hasMask;
this.sOpenMaskServiceName=_1.openMaskServiceName;
this.sCloseMaskServiceName=_1.closeMaskServiceName;
this.sPostOpenedServiceName=_1.postOpenedServiceName;
this.sPostClosedServiceName=_1.postClosedServiceName;
this.bAdjustSize=_1.adjustSize;
this.fSetNotchLocation=false;
this.oSetNotchLocationOverrider=false;
this.mouseState=-1;
this.closeByButton=false;
this.bPanelOpen=false;
this.bCheckState=false;
vjo.darwin.core.overlaypanel.VjOverlayPanelController.registerOverlayPanel(this);
this.onTick=function(){
this.monitorMouseDrag();
if(this.bCloseOnMouseOut){
this.panelClose();
}
};
if(this.bCloseOnMouseOut){
this.start();
}
},promote2body:function(){
vjo.dsf.document.Element.promoteToBody(this.sOverlayDivId);
},onResize:function(){
if(this.bPanelOpen){
if(!this.bDragged){
this.bPanelOpen=false;
}
this.openOverlayPanel();
}
},onOpenOverlayPanel:function(){
if(this.iOpenDelay<=0){
this.openOverlayPanel();
}else{
window.setTimeout(vjo.dsf.utils.Object.hitch(this,"openOverlayPanel"),this.iOpenDelay);
}
},openOverlayPanel:function(){
var t=this,openedInfo="undefined";
if(!t.bPanelOpen){
t.bPanelOpen=true;
openedInfo=t.render();
if(t.bHasMask){
t.openMask();
}
var _3=new vjo.dsf.Message(t.sPostOpenedServiceName);
if(openedInfo&&openedInfo!="undefined"){
_3.clientContext={x:openedInfo[0],y:openedInfo[1],w:openedInfo[2],h:openedInfo[3]};
vjo.dsf.ServiceEngine.handleRequest(_3);
}
}
},openMask:function(){
var t=this,maskMsg=new vjo.dsf.Message(t.sOpenMaskServiceName),contentDiv=vjo.dsf.document.Element.get(t.sOverlayDivId+"olpcontent");
try{
var _5=contentDiv.getElementsByTagName("form")[0];
maskMsg.clientContext={formName:_5.name||_5.id};
}
catch(e){
}
vjo.dsf.ServiceEngine.handleRequest(maskMsg);
},render:function(){
var t=this,E=vjo.dsf.document.Element,B=vjo.dsf.client.Browser,P=vjo.dsf.document.Positioning,coordinates={x:0,y:0},openedInfo="undefined";
if(!t.bShownInCenter){
coordinates=vjo.darwin.core.bubble.VjAnchorUtils.getAnchorPosition(t.sAnchorName);
if(B.bIE&&B.fVer>=7){
coordinates.x=P.getOffsetLeft(E.get(t.sAnchorName));
}
if(B.bIE&&B.fVer<7){
coordinates.x=P.getOffsetLeft(E.get(t.sAnchorName))+1;
}
}
var _7=E.get(t.sOverlayDivId),anchorDiv=E.get(t.sAnchorName),contentDiv=E.get(t.sOverlayDivId+"olpcontent"),shadowDiv=E.get(t.sOverlayDivId+"olpshadow"),wrapperDivStyle=_7.style;
if(wrapperDivStyle){
wrapperDivStyle.display="block";
wrapperDivStyle.visibility="visible";
wrapperDivStyle.position="absolute";
if(wrapperDivStyle.zIndex==0){
t.promote2body();
}
wrapperDivStyle.zIndex=vjo.darwin.core.overlaypanel.VjOverlayPanelEvtHandlers.openAtZIndex++;
}
if(t.sHAlign=="right"&&anchorDiv){
coordinates.x+=anchorDiv.offsetWidth+t.leftOffset;
}else{
if(t.sHAlign=="middle"&&anchorDiv){
coordinates.x+=anchorDiv.offsetWidth/2+t.leftOffset;
}else{
coordinates.x+=t.leftOffset;
}
}
if(t.sVAlign=="bottom"&&anchorDiv){
coordinates.y+=anchorDiv.offsetHeight+t.topOffset;
}else{
if(t.sVAlign=="middle"&&anchorDiv){
coordinates.y+=anchorDiv.offsetHeight/2+t.topOffset;
}else{
coordinates.y+=t.topOffset;
}
}
if(t.fSetNotchLocation){
openedInfo=t.fSetNotchLocation(t.oSetNotchLocationOverrider,t,_7,coordinates);
}else{
openedInfo=t.setNotchLocation(_7,coordinates);
}
shadowDiv.style.height=(contentDiv.offsetHeight+2)+"px";
if(t.iframeShim){
var w=_7.offsetWidth+t.iShadowOffset,h=_7.offsetHeight+t.iShadowOffset+2,s=t.iframeShim.style;
s.width=w+"px";
s.height=h+"px";
}else{
t.iframeShim=vjo.dsf.document.Shim.add(_7,t.iShadowOffset,t.iShadowOffset+2);
}
return openedInfo;
},setAnchorName:function(_9){
this.sAnchorName=_9;
},setBubbleDivId:function(_a){
this.sOverlayDivId=_a;
},onClosePanel:function(_b){
this.bCheckState=_b;
if(this.iCloseDelay<=0){
this.closePanel();
}else{
window.setTimeout(vjo.dsf.utils.Object.hitch(this,"closePanel"),this.iCloseDelay);
}
},closePanel:function(){
if(this.bCheckState&&this.mouseState==1){
this.bCheckState=false;
return;
}
this.hidePanel();
if(this.mouseState==1){
this.closeByButton=true;
}
this.bDragged=false;
},setNotchLocation:function(_c,_d){
var _e=vjo.dsf.window.utils.VjWindowUtils;
var _f=_c.offsetWidth;
var _10=_c.offsetHeight;
var _11=_e.getBrowserWindowWidth();
var _12=_e.getBrowserWindowHeight();
var _13=_e.getScrollXY();
var px="px";
var _15=vjo.dsf.document.Element.get(this.sOverlayDivId+"olparrow");
if(!this.bShownInCenter){
var _16=_d.x-_13[0];
var _17=_d.y-_13[1];
var _18=vjo.dsf.document.Element.get(this.sAnchorName);
var _19=20;
if(_18){
_19=_18.offsetHeight;
}
var _1a=0,arrowWidth=0,arrowSpace=0;
if(_15){
_1a=21;
arrowWidth=21;
arrowSpace=14;
}
var _1b=0,finalY=0,finalW=_f+arrowWidth,finalH=_10;
if((_16+_f+arrowWidth)<_11||_f>_11||(this.sHAlign&&this.sHAlign!="middle")){
_1b=_d.x;
_c.style.left=_1b+arrowWidth+px;
if((_17+_10)<_12){
finalY=_d.y-arrowSpace;
_c.style.top=finalY+px;
this.setArrowDivStyle(_15,this.sArrowTL,(arrowSpace+px),"auto");
}else{
if((_17+arrowSpace)>_10){
finalY=_d.y-_10;
_c.style.top=finalY+arrowSpace+px;
this.setArrowDivStyle(_15,this.sArrowBL,"auto",(arrowSpace+px));
}else{
if(_17<(_12/2)&&(_10<_12)){
var _1c=(_d.y+_10)-(_13[1]+_12);
finalY=_d.y-_1c-arrowSpace;
_c.style.top=finalY+px;
this.setArrowDivStyle(_15,this.sArrowTL,(arrowSpace+_1c+px),"auto");
}else{
var _1c=_13[1]-(_d.y-_10+arrowSpace);
finalY=_13[1];
_c.style.top=finalY+px;
this.setArrowDivStyle(_15,this.sArrowBL,(_17-arrowSpace+px),"auto");
}
}
}
}else{
if(_15){
_1b=_d.x-_f-arrowWidth;
}else{
if((_16-_f-arrowWidth)<0){
_1b=((_11/2-_f/2)+_13[0])+this.leftOffset;
}else{
_1b=_d.x-_f-arrowWidth;
}
}
_c.style.left=_1b+px;
if((_17+_10)<_12&&(_10<_12)){
finalY=_d.y-arrowSpace;
_c.style.top=finalY+px;
this.setArrowDivStyle(_15,this.sArrowTR,(arrowSpace+px),"auto");
}else{
if((_17+arrowSpace)>_10){
finalY=_d.y-_10+arrowSpace;
_c.style.top=finalY+px;
this.setArrowDivStyle(_15,this.sArrowBR,"auto",(arrowSpace+px));
}else{
if(_17<(_12/2)){
var _1c=(_d.y+_10)-(_13[1]+_12);
finalY=_d.y-_1c-arrowSpace;
_c.style.top=finalY+px;
this.setArrowDivStyle(_15,this.sArrowTR,(arrowSpace+_1c+px),"auto");
}else{
var _1c=_13[1]-(_d.y-_10+arrowSpace);
finalY=_13[1];
_c.style.top=finalY+px;
this.setArrowDivStyle(_15,this.sArrowBR,(_17-arrowSpace+px),"auto");
}
}
}
}
}else{
_1b=((_11/2-_f/2)+_13[0])+this.leftOffset;
finalY=_12/2-_10/2;
finalY=finalY>0?finalY:0;
finalY=(finalY+_13[1])+this.topOffset;
_c.style.left=_1b+px;
_c.style.top=finalY+px;
}
return [_1b,finalY,finalW,finalH];
},setArrowDivStyle:function(_1d,_1e,_1f,_20){
if(_1d){
_1d.className=_1e;
_1d.style.top=_1f;
_1d.style.bottom=_20;
}
},startDrag:function(_21){
if(!this.bMonitorDrag){
this.bMonitorDrag=true;
this.bDragged=true;
var _22=_21.nativeEvent;
var _23=this.eventTop(_22);
var _24=this.eventLeft(_22);
var _25=vjo.darwin.core.bubble.VjAnchorUtils.getAnchorPosition(this.sOverlayDivId);
this.initDx=_25.x-_24;
this.initDy=_25.Y-_23;
this.top=_23;
this.left=_24;
this.iLastLeft=_24;
this.iLastTop=_23;
var _26=function(){
};
_26.prototype.handle=vjo.dsf.utils.Object.hitch(this,"onMouseMove");
var _27=function(){
};
_27.prototype.handle=vjo.dsf.utils.Object.hitch(this,"onMouseUp");
var _28=vjo.dsf.EventDispatcher;
_28.add("body","mousemove",new _26());
_28.add("body","mouseup",new _27());
var _29=vjo.dsf.document.Element.get(this.sOverlayDivId);
if(_29){
_29.onselectstart=function(){
return false;
};
_29.onmousedown=function(){
return false;
};
}
this.start();
}
},onMouseMove:function(_2a){
if(this.bMonitorDrag){
var _2b=_2a.nativeEvent;
var _2c=this.eventTop(_2b);
var _2d=this.eventLeft(_2b);
if(_2c>=0&&_2d>=0){
this.top=_2c;
this.left=_2d;
}
}
},onMouseUp:function(_2e){
if(this.bMonitorDrag){
this.stopDrag();
}
},monitorMouseDrag:function(){
if(this.bMonitorDrag){
if((this.left!=this.iLastLeft)||(this.top!=this.iLastTop)){
var dx=this.left-this.iLastLeft;
var dy=this.top-this.iLastTop;
this.iLastLeft=this.left;
this.iLastTop=this.top;
var _31=vjo.darwin.core.bubble.VjAnchorUtils.getAnchorPosition(this.sOverlayDivId);
var px="px";
var _33=vjo.dsf.document.Element.get(this.sOverlayDivId);
_33.style.left=_31.x+dx+px;
_33.style.top=_31.y+dy+px;
}
}
},stopDrag:function(){
this.bMonitorDrag=false;
var _34=vjo.dsf.document.Element.get(this.sOverlayDivId);
if(_34){
_34.onselectstart=null;
_34.onmousedown=null;
}
this.stop();
},eventTop:function(_35){
if(_35.pageY!=null){
return _35.pageY;
}else{
if(document.documentElement&&document.documentElement.scrollTop){
return _35.clientY+Math.max(document.documentElement.scrollTop,document.body.scrollTop);
}else{
return _35.clientY+document.body.scrollTop;
}
}
},eventLeft:function(_36){
if(_36.pageX!=null){
return _36.pageX;
}else{
if(document.documentElement&&document.documentElement.scrollLeft){
return _36.clientX+Math.max(document.documentElement.scrollLeft,document.body.scrollLeft);
}else{
return _36.clientX+document.body.scrollLeft;
}
}
},panelMouseOver:function(){
this.mouseState=1;
},panelMouseOut:function(){
this.mouseState=0;
},panelClose:function(){
if(this.mouseState==0){
this.mouseState=-1;
if(!this.closeByButton){
this.hidePanel();
}else{
this.closeByButton=false;
this.mouseState=-1;
}
}
},hidePanel:function(){
var _37=vjo.dsf.document.Element.get(this.sOverlayDivId);
_37.style.top="-1000px";
_37.style.visibility="hidden";
this.bPanelOpen=false;
if(this.bHasMask){
vjo.dsf.ServiceEngine.handleRequest(new vjo.dsf.Message(this.sCloseMaskServiceName));
}
if(this.sPostClosedServiceName){
vjo.dsf.ServiceEngine.handleRequest(new vjo.dsf.Message(this.sPostClosedServiceName));
}
if(this.iframeShim){
vjo.dsf.document.Shim.remove(_37,this.iframeShim);
this.iframeShim=false;
}
},setContentTitle:function(_38){
var _39=vjo.dsf.document.Element.get(this.sOverlayDivId+"olpcontenttitle");
_39.innerHTML=_38;
}});

vjo.needs("vjo.dsf.Message");
vjo.type("vjo.darwin.core.overlaypanel.VjOverlayPanelCloseSvcHandler").protos({constructs:function(_1){
this.sOverlayPanelJsCompId=_1;
},invoke:function(_2){
var _3=vjo.Registry.get(this.sOverlayPanelJsCompId);
_3.onClosePanel(_2.getCheckState?_2.getCheckState():false);
return _2;
}});

vjo.type("vjo.darwin.core.mask.MaskHandlers").props({openService:function(_1,_2){
var _3=vjo.Registry.get(_1);
if(_3){
_3.show(_2.clientContext);
}
},closeService:function(_4,_5){
var _6=vjo.Registry.get(_4);
if(_6){
_6.hide();
}
}});

vjo.needs("vjo.dsf.EventDispatcher");
vjo.needs("vjo.dsf.window.utils.VjWindowUtils");
vjo.needs("vjo.dsf.utils.Timer");
vjo.needs("vjo.darwin.core.mask.MaskHandlers");
vjo.type("vjo.darwin.core.mask.Mask").protos({constructs:function(_1,_2){
var t=this;
t.eElem=document.getElementById(_1);
t.iOpacity=(_2)?parseFloat(_2):0;
t.showing=false;
t.zIndex=50;
t.sIsActivated=false;
t.oTimer=false;
t.ED=vjo.dsf.EventDispatcher;
},show:function(_4){
var t=this;
if(t.showing){
return;
}else{
t.showing=true;
}
t.formName=(_4)?_4.formName:"";
t.zIndex=(_4&&_4.overlayPanelIndex)?_4.overlayPanelIndex:t.zIndex;
t.render();
t.disableSelects();
t.startResizeListener();
},render:function(){
var t=this;
t.VjWindowUtils=vjo.dsf.window.utils.VjWindowUtils;
var _7=t.VjWindowUtils.scrollWidth();
var _8=t.VjWindowUtils.scrollHeight();
t.setStyle("display","block");
t.setStyle("width",t.VjWindowUtils.toPixels(_7));
t.setStyle("height",t.VjWindowUtils.toPixels(_8));
t.setStyle("opacity",t.iOpacity);
t.setStyle("filter","alpha(opacity="+parseInt(100*t.iOpacity)+")");
t.setStyle("background","#cccccc");
t.setStyle("zIndex",t.zIndex);
},hide:function(){
var t=this;
if(!t.showing){
return;
}else{
t.showing=false;
}
t.setStyle("display","none");
t.setStyle("width",t.VjWindowUtils.toPixels(0));
t.setStyle("height",t.VjWindowUtils.toPixels(0));
t.enableSelects();
t.stopResizeListener();
},disableSelects:function(){
this.disabledSelects=new Array();
var _a=document.body.getElementsByTagName("select");
for(var _b=0;(_b<_a.length);_b++){
var _c=_a[_b];
if(_c.disabled){
continue;
}
var _d=(_c.form)?_c.form.name:null;
if(_d==this.formName){
continue;
}
_c.disabled=true;
this.disabledSelects.push(_c);
}
},enableSelects:function(){
var _e=this.disabledSelects;
for(var _f=0;(_f<_e.length);_f++){
_e[_f].disabled=false;
}
},setStyle:function(_10,_11){
var e=this.eElem;
if(!e||!e.style){
return;
}
e.style[_10]=_11;
},requireResize:function(){
this.sIsActivated=true;
},startResizeListener:function(){
var t=this;
t.ED.addEventListener(window,"resize",t.requireResize,t);
if(!t.oTimer){
t.oTimer=new vjo.dsf.utils.Timer();
t.oTimer.interval=500;
t.oTimer.onTick=function(){
if(t.sIsActivated){
t.render();
t.sIsActivated=false;
}
};
}
t.oTimer.start();
},stopResizeListener:function(){
var t=this;
t.ED.removeEventListener(window,"resize",t.requireResize,t);
if(t.oTimer){
t.oTimer.stop();
}
}});

function FooterTrk() { 	return { handle : function (event) { var _d = vjo.dsf.EventDispatcher, _r = vjo.Registry;
_r.put('FooterTrackingCompSpecGenerator_0', new vjo.darwin.tracking.enabler.TrackingModuleEnabler('_trksid', 'm40'));
_d.add('glbfooter','click',_r.get('FooterTrackingCompSpecGenerator_0'));
   } }; }
/* compspec addJsCompRegistration*/ 
(function () {
var _r = vjo.Registry;
})();
/* end comp spec*/

vjo.dsf.EventDispatcher.add('body','load', new HeaderTrk());
/* compspec addJsCompRegistration*/ 
(function () {
var _r = vjo.Registry;
})();
/* end comp spec*/

vjo.dsf.EventDispatcher.add('body','load', new FooterTrk());

// en_US/e551/GlobalNavVjo23_SignInEbay_e5516038869_1_en_US
// b=6038869