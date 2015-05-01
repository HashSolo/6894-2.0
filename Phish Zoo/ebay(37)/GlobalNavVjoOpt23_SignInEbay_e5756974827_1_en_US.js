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

vjo.needs("vjo.dsf.Message");
vjo.needs("vjo.darwin.core.overlaypanel.VjOverlayPanelMessage");
vjo.needs("vjo.dsf.window.utils.VjWindowUtils");
vjo.type("vjo.darwin.core.overlaypanel.VjOverlayPanelEvtHandlers").props({handleOpenEvent:function(_1,_2){
var _3=new vjo.darwin.core.overlaypanel.VjOverlayPanelMessage(_1);
_3.setAnchorId(_2);
return _3;
},handleOpenEventXY:function(_4,x,y){
var _7=new vjo.darwin.core.overlaypanel.VjOverlayPanelMessage(_4);
_7.left=x;
_7.top=y;
_7.resetPosition=false;
return _7;
},handleOpenEventByLoc:function(_8,_9){
var e=_9.nativeEvent,W=vjo.dsf.window.utils.VjWindowUtils,x=W.eventLeft(e),y=W.eventTop(e);
return this.handleOpenEventXY(_8,x,y);
},handleCloseEvent:function(_b){
var _c=new vjo.darwin.core.overlaypanel.VjOverlayPanelMessage(_b);
return _c;
},handleCloseEventWithCheck:function(_d,_e){
var _f=new vjo.darwin.core.overlaypanel.VjOverlayPanelMessage(_d);
_f.bCheckState=_e;
return _f;
}});

vjo.needs("vjo.dsf.Element");
vjo.needs("vjo.dsf.client.Browser");
vjo.type("vjo.dsf.document.Element").inherits("vjo.dsf.Element").props({toggleHideShowRow:function(_1,_2){
var e=this.get(_1),s,d,u="undefined";
var p=vjo.dsf.client.Browser.bFirefox?"table-row":"block";
if(e){
s=e.style;
d=s.display;
if(typeof (_2)===u){
_2=(d===""||d===p)?false:true;
}
e.bIsShown=_2;
s.display=(_2)?p:"none";
}
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

vjo.needs("vjo.dsf.utils.Timer");
vjo.type("vjo.darwin.core.overlaypanel.VjOverlayPanelController").singleton().protos({constructs:function(){
var t=this;
t.ED=vjo.dsf.EventDispatcher;
t.aOlps=[];
t.sIsActivated=false;
t.ED.addEventListener(window,"resize",t.initResize,t);
var _2=function(){
t.initCloseOnMouseOut();
};
window.setTimeout(_2,500);
t.openAtZIndex=5000;
},registerOverlayPanel:function(_3){
var t=this;
t.aOlps[t.aOlps.length]=_3;
},initCloseOnMouseOut:function(){
var t=this;
t.oCloseTimer=new vjo.dsf.utils.Timer(100);
t.oCloseTimer.onTick=function(){
var i=0,arr=t.aOlps,len=arr.length;
while(i<len){
var _7=arr[i];
if(_7.bCloseOnMouseOut){
_7.tryCloseOnMouseOut();
}
i++;
}
};
t.oCloseTimer.start();
},zIndex:function(){
this.openAtZIndex+=1;
return this.openAtZIndex;
},requireResize:function(){
this.sIsActivated=true;
},forceResize:function(_8){
var t=this,i=0;
for(;i<t.aOlps.length;i++){
if(_8&&t.aOlps[i].sOverlayDivId==_8){
t.aOlps[i].onResize();
break;
}else{
t.aOlps[i].onResize();
}
}
},initResize:function(){
var t=this;
t.ED.removeEventListener(window,"resize",t.initResize);
t.ED.addEventListener(window,"resize",t.requireResize,t);
t.requireResize();
t.oResizeTimer=new vjo.dsf.utils.Timer();
t.oResizeTimer.interval=500;
t.oResizeTimer.onTick=function(){
if(t.sIsActivated){
for(var i=0;i<t.aOlps.length;i++){
if(t.aOlps[i].bAdjustSize&&t.aOlps[i].bPanelOpen){
t.aOlps[i].onResize();
}
}
t.sIsActivated=false;
}
};
t.oResizeTimer.start();
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

vjo.needs("vjo.dsf.client.Browser");
vjo.needs("vjo.dsf.document.Element");
vjo.needs("vjo.dsf.document.Positioning");
vjo.needs("vjo.dsf.document.Shim");
vjo.needs("vjo.dsf.EventDispatcher");
vjo.needs("vjo.dsf.Message");
vjo.needs("vjo.dsf.ServiceEngine");
vjo.needs("vjo.dsf.utils.Object");
vjo.needs("vjo.dsf.utils.Timer");
vjo.needs("vjo.dsf.window.utils.VjWindowUtils");
vjo.needs("vjo.darwin.core.overlaypanel.VjOverlayPanelController");
vjo.type("vjo.darwin.core.overlaypanel.VjOverlayPanel").inherits("vjo.dsf.utils.Timer").protos({constructs:function(_1){
var t=this,E=vjo.dsf.document.Element;
t.sOverlayDivId=_1.overlayCompId;
t.sOlpId=_1.cmpWrapperId;
t.oOlp=E.get(t.sOverlayDivId);
t.oCntnt=E.get(t.sOverlayDivId+"olpcontent");
t.oShdw=E.get(t.sOverlayDivId+"olpshadow");
t.oArr=E.get(t.sOverlayDivId+"olparrow");
t.pVt=_1.pointerVertical;
t.sArrowVTL=_1.arrowVTL;
t.sArrowVTR=_1.arrowVTR;
t.sArrowVBL=_1.arrowVBL;
t.sArrowVBR=_1.arrowVBR;
t.sArrowTL=_1.arrowTL;
t.sArrowTR=_1.arrowTR;
t.sArrowBR=_1.arrowBR;
t.sArrowBL=_1.arrowBL;
t.iLastLeft=t.iLastTop=t.mouseState=-1;
t.initDx=t.initDy=t.top=t.left=t.iTmpLftOfset=t.iTmpTopOfset=0;
t.bShownInCenter=_1.isShownInCenter;
t.iCnstLftOfset=_1.leftOffset;
t.iCnstTopOfset=_1.topOffset;
t.aAnchorCoor=false;
t.sHAlgn=_1.horizontalAlign||false;
t.sVAlgn=_1.verticalAlign||false;
t.iShdwOfset=_1.shadowOffset;
t.iOpDlay=_1.openDelay;
t.iClsDlay=_1.closeDelay;
t.bCloseOnMouseOut=_1.isCloseOnMouseOut;
t.bHasMask=_1.hasMask;
t.sOpMaskServId=_1.openMaskServiceName;
t.sClsMaskServId=_1.closeMaskServiceName;
t.sPostOpServId=_1.postOpenedServiceName;
t.sPostClsServId=_1.postClosedServiceName;
t.bAdjustSize=_1.adjustSize;
t.fSetNotchLocation=t.oSetNotchLocationOverrider=t.closeByButton=t.bPanelOpen=t.bCheckState=false;
vjo.darwin.core.overlaypanel.VjOverlayPanelController.registerOverlayPanel(t);
if(!t.oIFrame&&t.oOlp){
t.oIFrame=vjo.dsf.document.Shim.add(t.oOlp);
}
},promote2body:function(_3){
if(_3){
vjo.dsf.document.Element.promoteToBody(this.sOlpId);
}
},onResize:function(){
var t=this;
if(t.bPanelOpen){
if(!t.bDragged){
t.bPanelOpen=false;
}
t.openOverlayPanel();
}
},onOpenOverlayPanel:function(_5){
var t=this;
t.aAnchorCoor=_5?_5:false;
t.openTimer=(t.iOpDlay<=0)?t.openOverlayPanel():window.setTimeout(vjo.dsf.utils.Object.hitch(t,"openOverlayPanel"),t.iOpDlay);
},openOverlayPanel:function(){
var t=this,openedInfo=false;
if(t.openTimer){
window.clearTimeout(t.openTimer);
t.openTimer=null;
}
if(!t.bPanelOpen){
t.bPanelOpen=true;
openedInfo=t.render();
if(t.bHasMask){
t.openMask();
}
var _8=new vjo.dsf.Message(t.sPostOpServId);
if(openedInfo){
_8.clientContext={x:openedInfo[0],y:openedInfo[1],w:openedInfo[2],h:openedInfo[3]};
vjo.dsf.ServiceEngine.handleRequest(_8);
}
}
},openMask:function(){
var t=this,maskMsg=new vjo.dsf.Message(t.sOpMaskServId);
try{
var _a=t.oCntnt?t.oCntnt.getElementsByTagName("form")[0]:{name:""};
maskMsg.clientContext={formName:_a.name||_a.id};
}
catch(e){
}
vjo.dsf.ServiceEngine.handleRequest(maskMsg);
},align:function(_b,_c,_d,_e){
return _b+((_c=="right"||_c=="bottom")?_d:0)+((_c=="middle")?(_d/2):0)+_e;
},getLeftOffset:function(){
return this.iCnstLftOfset+this.iTmpLftOfset;
},getTopOffset:function(){
return this.iCnstTopOfset+this.iTmpTopOfset;
},render:function(){
var t=this,E=vjo.dsf.document.Element,B=vjo.dsf.client.Browser,P=vjo.dsf.document.Positioning,W=vjo.dsf.window.utils.VjWindowUtils,anc=E.get(t.sAnchorName),coordinates={x:0,y:0},openedInfo=false;
if(!t.bShownInCenter){
if(t.aAnchorCoor){
coordinates.x=t.aAnchorCoor.x+t.getLeftOffset();
coordinates.y=t.aAnchorCoor.y+t.getTopOffset();
t.aAnchorCoor=false;
}else{
coordinates.x=(B.bIE&&B.fVer<7)?P.getOffsetLeft(anc)+1:P.getOffsetLeft(anc);
coordinates.y=P.getOffsetTop(anc);
coordinates.x=t.align(coordinates.x,t.sHAlgn,anc?anc.offsetWidth:0,t.getLeftOffset());
coordinates.y=t.align(coordinates.y,t.sVAlgn,anc?anc.offsetHeight:0,t.getTopOffset());
}
}
var olp=t.oOlp,olpS=olp.style;
if(olpS){
olpS.visibility="visible";
olpS.zIndex=vjo.darwin.core.overlaypanel.VjOverlayPanelController.zIndex();
if(olpS.zIndex===0){
t.promote2body();
}
}
openedInfo=t.fSetNotchLocation?t.fSetNotchLocation(t.oSetNotchLocationOverrider,t,olp,coordinates):t.setNotchLocation(olp,coordinates);
t.oShdw.style.height=W.toPixels(t.oCntnt.offsetHeight+2);
if(t.oIFrame){
var s=t.oIFrame.style;
s.width=W.toPixels(olp.offsetWidth+t.iShdwOfset);
s.height=W.toPixels(olp.offsetHeight+t.iShdwOfset+2);
}else{
t.oIFrame=vjo.dsf.document.Shim.add(olp,t.iShdwOfset,t.iShdwOfset+2);
}
return openedInfo;
},setAnchorName:function(_12){
this.sAnchorName=_12;
},setBubbleDivId:function(_13){
this.sOverlayDivId=_13;
},onClosePanel:function(_14){
var t=this;
t.bCheckState=_14;
t.closeTimer=(t.iClsDlay<=0)?t.closePanel():window.setTimeout(vjo.dsf.utils.Object.hitch(t,"closePanel"),t.iClsDlay);
},closePanel:function(){
var t=this;
if(t.closeTimer){
window.clearTimeout(t.closeTimer);
t.closeTimer=null;
}
if(t.bCheckState&&t.mouseState==1){
t.bCheckState=false;
return;
}
t.hidePanel();
if(t.mouseState==1){
t.closeByButton=true;
}
t.bDragged=false;
},setNotchLocation:function(olp,_18){
var t=this,W=vjo.dsf.window.utils.VjWindowUtils,olpS=olp.style,iX=_18.x,iY=_18.y,iW=olp.offsetWidth,iH=olp.offsetHeight,iScreenW=W.getBrowserWindowWidth(),iScreenH=W.getBrowserWindowHeight(),aScollLoc=W.getScrollXY(),iScrollX=aScollLoc[0],iScrollY=aScollLoc[1],arr=t.oArr;
if(!t.bShownInCenter){
if(t.pVt){
var _1a=iX-iScrollX,yLoc=iY-iScrollY,finalX=0,finalY=0,arrH=21,sAuto="auto",iAdj=5,bTop=true,bLeft=false,B=vjo.dsf.client.Browser,iGuard=1;
if(!B.bIE){
iAdj+=16;
}
if((yLoc-iH-arrH)<0){
finalY=iY+arrH+iGuard;
bTop=false;
}else{
finalY=iY-iH-arrH-iGuard;
}
if(iX-iW/2<0){
finalX=0;
}else{
if(iX+iW/2+iAdj>iScreenW){
finalX=iScreenW-iW-iAdj;
bLeft=true;
}else{
finalX=iX-(iW/2);
}
}
if(arr){
arr.style.left=W.toPixels(iX-finalX-(bLeft?arr.offsetWidth+iGuard:-iGuard));
}
if(bTop){
t.setArrDivStyle(arr,bLeft?t.sArrowVBL:t.sArrowVBR,sAuto,W.toPixels(-arrH));
}else{
t.setArrDivStyle(arr,bLeft?t.sArrowVTL:t.sArrowVTR,W.toPixels(-arrH),sAuto);
}
}else{
var _1a=iX-iScrollX,yLoc=iY-iScrollY,arrH=arr?21:0,arrW=arr?21:0,arrS=arr?14:0,finalX=0,finalY=0,finalW=iW+arrW,finalH=iH,sAuto="auto",iAdj=0,bLeft=false;
if((_1a+iW+arrW)<iScreenW||iW>iScreenW||(t.sHAlgn&&t.sHAlgn!="middle")){
finalX=iX+arrW;
bLeft=true;
}else{
if((_1a-iW-arrW)<0){
finalX=(iScreenW/2-iW/2)+iScrollX+t.getLeftOffset();
}else{
finalX=iX-iW-arrW;
}
}
if((yLoc+iH)<iScreenH){
finalY=iY-arrS;
t.setArrDivStyle(arr,bLeft?t.sArrowTL:t.sArrowTR,W.toPixels(arrS),sAuto);
}else{
if((yLoc+arrS)>iH){
finalY=iY-iH+arrS;
finalY-=t.getTopOffset()*2;
t.setArrDivStyle(arr,bLeft?t.sArrowBL:t.sArrowBR,sAuto,W.toPixels(arrS));
}else{
if(yLoc<(iScreenH/2)&&(iH<iScreenH)){
iAdj=(iY+iH)-(iScrollY+iScreenH);
finalY=iY-iAdj-arrS;
t.setArrDivStyle(arr,bLeft?t.sArrowTL:t.sArrowTR,W.toPixels(arrS+iAdj),sAuto);
}else{
iAdj=iScrollY-(iY-iH+arrS);
finalY=iScrollY;
t.setArrDivStyle(arr,bLeft?t.sArrowBL:t.sArrowBR,W.toPixels(yLoc-arrS),sAuto);
}
}
}
}
}else{
finalX=(iScreenW/2-iW/2)+iScrollX+t.getLeftOffset();
finalY=iScreenH/2-iH/2;
finalY=finalY>0?finalY:0;
finalY=finalY+iScrollY+t.getTopOffset();
}
olpS.left=W.toPixels(finalX);
olpS.top=W.toPixels(finalY);
return [finalX,finalY,finalW,finalH];
},setArrDivStyle:function(_1b,_1c,_1d,_1e){
if(_1b){
_1b.className=_1c;
_1b.style.top=_1d;
_1b.style.bottom=_1e;
}
},startDrag:function(_1f){
var t=this,P=vjo.dsf.document.Positioning,W=vjo.dsf.window.utils.VjWindowUtils,olp=t.oOlp;
if(!t.bMonitorDrag){
t.bMonitorDrag=true;
t.bDragged=true;
var _21=_1f.nativeEvent;
var _22=W.eventTop(_21);
var _23=W.eventLeft(_21);
t.initDx=P.getOffsetLeft(olp)-_23;
t.initDy=P.getOffsetTop(olp)-_22;
t.top=t.iLastTop=_22;
t.left=t.iLastLeft=_23;
var ED=vjo.dsf.EventDispatcher;
t.fOnMouseMoveHdl=ED.addEventListener(document.body,"mousemove",t.onMouseMove,t);
t.fOnMouseUpHdl=ED.addEventListener(document.body,"mouseup",t.onMouseUp,t);
if(olp){
olp.onselectstart=function(){
return false;
};
olp.onmousedown=function(){
return false;
};
}
t.interval=50;
t.onTick=function(){
t.monitorMouseDrag();
};
t.start();
}
},onMouseMove:function(_25){
var t=this,W=vjo.dsf.window.utils.VjWindowUtils;
if(t.bMonitorDrag){
var _27=window.event?window.event:_25;
var _28=W.eventTop(_27),leftValue=W.eventLeft(_27);
if(_28>=0&&leftValue>=0){
t.top=_28;
t.left=leftValue;
}
}
},onMouseUp:function(_29){
var t=this,ED=vjo.dsf.EventDispatcher;
if(t.bMonitorDrag){
ED.removeEventListener(document.body,"mousemove",t.fOnMouseMoveHdl);
ED.removeEventListener(document.body,"mouseup",t.fOnMouseUpHdl);
t.stopDrag();
}
},monitorMouseDrag:function(){
if(this.bMonitorDrag){
var t=this,P=vjo.dsf.document.Positioning,W=vjo.dsf.window.utils.VjWindowUtils;
if((t.left!=t.iLastLeft)||(t.top!=t.iLastTop)){
var dx=t.left-t.iLastLeft,dy=t.top-t.iLastTop;
t.iLastLeft=t.left;
t.iLastTop=t.top;
var olp=t.oOlp,olpS=olp.style,x=P.getOffsetLeft(olp),y=P.getOffsetTop(olp);
olpS.left=W.toPixels(x+dx);
olpS.top=W.toPixels(y+dy);
}
}
},stopDrag:function(){
var t=this;
t.bMonitorDrag=false;
var olp=t.oOlp;
if(olp){
olp.onselectstart=null;
olp.onmousedown=null;
}
t.stop();
},panelMouseOver:function(){
this.mouseState=1;
},panelMouseOut:function(){
this.mouseState=0;
},tryCloseOnMouseOut:function(){
var t=this;
if(t.mouseState===0){
t.mouseState=-1;
if(!t.closeByButton){
t.hidePanel();
}else{
t.closeByButton=false;
t.mouseState=-1;
}
}
},hidePanel:function(){
var t=this,ED=vjo.dsf.EventDispatcher,SE=vjo.dsf.ServiceEngine,M=vjo.dsf.Message,olp=t.oOlp,olpS=olp.style;
olpS.visibility="hidden";
olpS.left="-1600px";
t.bPanelOpen=false;
t.iTmpLftOfset=t.iTmpTopOfset=0;
if(t.bHasMask){
SE.handleRequest(new M(t.sClsMaskServId));
}
if(t.sPostClsServId){
SE.handleRequest(new M(t.sPostClsServId));
}
if(t.oIFrame){
if(!t.fCleanUp){
t.fCleanUp=function(){
vjo.dsf.document.Shim.remove(this.oOlp,this.oIFrame);
this.oIFrame=null;
};
ED.addEventListener(window,"unload",t.fCleanUp,t);
}
}
},setContentTitle:function(_32){
var t=this;
if(!t.oTitlCntnr){
t.oTitlCntnr=vjo.dsf.document.Element.get(t.sOverlayDivId+"olpcontenttitle");
}
if(t.oTitlCntnr){
t.oTitlCntnr.innerHTML=_32;
}
}});

vjo.needs("vjo.dsf.Message");
vjo.type("vjo.darwin.core.overlaypanel.VjOverlayPanelCloseSvcHandler").protos({constructs:function(_1){
this.sOverlayPanelJsCompId=_1;
},invoke:function(_2){
var _3=vjo.Registry.get(this.sOverlayPanelJsCompId);
_3.onClosePanel(_2.getCheckState?_2.getCheckState():false);
return _2;
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
_3.iTmpLftOfset=_2.leftOffset?_2.leftOffset:0;
_3.iTmpTopOfset=_2.topOffset?_2.topOffset:0;
_3.aAnchorCoor=(_2.left&&_2.top)?{x:_2.left,y:_2.top}:false;
if(_2.bResetPosition){
_3.bPanelOpen=false;
}
_3.onOpenOverlayPanel(_3.aAnchorCoor||false);
_2.returnData=false;
return _2;
}});

vjo.type("vjo.darwin.app.common.mask.MaskHandlers").props({openService:function(_1,_2){
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

vjo.needs("vjo.dsf.window.utils.VjWindowUtils");
vjo.needs("vjo.dsf.EventDispatcher");
vjo.needs("vjo.dsf.utils.Timer");
vjo.needs("vjo.darwin.app.common.mask.MaskHandlers");
vjo.type("vjo.darwin.app.common.mask.Mask").protos({constructs:function(_1,_2){
var t=this;
t.eElem=document.getElementById(_1);
t.iOpacity=(_2)?parseFloat(_2):0;
t.showing=false;
t.zIndex=100;
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

vjo.needs("vjo.dsf.Message");
vjo.needs("vjo.dsf.EventDispatcher");
vjo.needs("vjo.dsf.ServiceEngine");
vjo.needs("vjo.dsf.window.utils.VjWindowUtils");
vjo.type("vjo.darwin.core.ebayheader.autocomplete.AutoComplete").protos({constructs:function(_1,_2){
var t=this,E=vjo.dsf.document.Element;
t.sInputId=_1;
t.sFormId=_2;
t.sAcDivId=_1+"_acdiv";
t.acdivWidth=0;
t.sShowImg=false;
t.input=E.get(t.sInputId);
t.input.setAttribute("AUTOCOMPLETE","OFF");
t.acdiv=E.get(t.sAcDivId);
var _4=document.forms;
vjo.dsf.EventDispatcher.add(this.sInputId,"keyup",function(e){
var _6=e.nativeEvent.keyCode;
var _7=new vjo.dsf.Message("SVC_GH_OUT");
_7.clientContext={type:"kw_keyup",srcId:_1,value:t.input.value,keyCode:_6};
return _7;
});
vjo.dsf.EventDispatcher.add(this.sInputId,"keydown",function(e){
var _9=e.nativeEvent.keyCode;
var _a=new vjo.dsf.Message("SVC_GH_OUT");
_a.clientContext={type:"kw_keydown",srcId:_1,value:t.input.value,keyCode:_9};
return _a;
});
vjo.dsf.EventDispatcher.add(this.sInputId,"mouseover",function(e){
var _c=e.nativeEvent.keyCode;
var _d=new vjo.dsf.Message("SVC_GH_OUT");
_d.clientContext={type:"kw_mouseover",srcId:_1};
return _d;
});
vjo.dsf.EventDispatcher.add(this.sInputId,"blur",function(e){
var _f=e.nativeEvent.keyCode;
var _10=new vjo.dsf.Message("SVC_GH_OUT");
_10.clientContext={type:"kw_blur",srcId:_1};
return _10;
});
vjo.dsf.EventDispatcher.add(this.sAcDivId,"click",function(e){
var _12=new vjo.dsf.Message("SVC_GH_OUT");
_12.clientContext={type:"show_click"};
return _12;
});
vjo.dsf.ServiceEngine.registerSvcHdl("SVC_GH_IN",function(_13){
var _14=_13.clientContext.type;
if(_14=="kw_updvalue"){
var _15=document.getElementById(_1);
_15.value=_13.clientContext.value;
}else{
if(_14=="kw_autocomplete"){
if(_13.clientContext.bOn){
t.input.blur();
}
t.input.setAttribute("AUTOCOMPLETE",_13.clientContext.bOn?"ON":"OFF");
if(_13.clientContext.bOn){
t.input.blur();
t.input.focus();
}
}else{
if(_14=="search_updtrk"){
for(i=0;i<_4.length;i++){
if(_4[i].name==_2){
for(var j=0;j<_4[i].length;j++){
if(_4[i].elements[j].name=="_trksid"){
var _17=_4[i].elements[j].value;
var _18=_17.indexOf(".");
if(_18>0){
_17=_17.substring(0,_18+1);
}else{
if(_17.length>0){
_17=_17+".";
}
}
_17=_17+_13.clientContext.lnkStr;
_4[i].elements[j].value=_17;
return;
}
}
}
}
}else{
if(_14=="search_submit"){
for(i=0;i<_4.length;i++){
if(_4[i].name==_2){
_4[i].submit();
return;
}
}
}else{
if(_14=="sug_icon_show"){
if(_13.clientContext.bShow){
t.showImage();
}else{
t.hideImage();
}
}else{
if(_14=="kw_focus"){
t.input.focus();
t.input.value=t.input.value+"";
}
}
}
}
}
}
});
},showImage:function(){
var t=this;
if(t.sShowImg){
return;
}
var _1a=vjo.dsf.window.utils.VjWindowUtils;
if(t.acdivWidth===0){
t.acdiv.style.display="inline";
t.acdivWidth=t.acdiv.offsetWidth;
}
var _1b=t.acdivWidth+3;
t.input.style.width=(t.input.clientWidth-_1b)+"px";
t.input.style.borderRightWidth="0px";
t.acdiv.style.display="inline";
t.sShowImg=true;
},hideImage:function(){
var t=this;
if(!t.sShowImg){
return;
}
var _1d=t.acdiv.offsetWidth-5;
t.input.style.width=(t.input.clientWidth+_1d)+"px";
t.input.style.borderRightWidth="1px";
t.acdiv.style.display="none";
t.sShowImg=false;
}});

/* compspec addJsCompRegistration*/ 

/* end comp spec*/

function HeaderTrk() { 	return { handle : function (event) { (function(){
var _d=vjo.dsf.EventDispatcher;
var _r=vjo.Registry;
_r.put('HeaderTrackingCompSpecGenerator_0', new vjo.darwin.tracking.enabler.TrackingModuleEnabler("_trksid", "m37")); _r.put('HeaderTrackingCompSpecGenerator_1', new vjo.darwin.tracking.enabler.TrackingModuleEnabler("_trksid", "m37")); _d.add('BrowseCategories-menu','click',_r.get('HeaderTrackingCompSpecGenerator_0'));_d.add('gnheader','click',_r.get('HeaderTrackingCompSpecGenerator_1'));})();  } }; }
vjo.dsf.EventDispatcher.add('body','load', new HeaderTrk());
/* compspec addJsCompRegistration*/ 

/* end comp spec*/

function FooterTrk() { 	return { handle : function (event) { (function(){
var _d=vjo.dsf.EventDispatcher;
var _r=vjo.Registry;
_r.put('FooterTrackingCompSpecGenerator_0', new vjo.darwin.tracking.enabler.TrackingModuleEnabler("_trksid", "m40")); _d.add('glbfooter','click',_r.get('FooterTrackingCompSpecGenerator_0'));})();  } }; }
vjo.dsf.EventDispatcher.add('body','load', new FooterTrk());

// en_US/e575/GlobalNavVjoOpt23_SignInEbay_e5756974827_1_en_US
// b=6974827