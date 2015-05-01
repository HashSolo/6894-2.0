
vjo.ctype("vjo.darwin.tracking.sojourner.SojData").singleton().protos({constructs:function(){this.sojD;}}).inits(function(){vjo.darwin.tracking.sojourner.SojData=new vjo.darwin.tracking.sojourner.SojData();}).endType();
vjo.ctype("vjo.darwin.tracking.sojourner.CalData").singleton().protos({constructs:function(){this.cal={};},setData:function(_1,_2){if(!_1||!_2){return;}
this.cal[_1]=_2;},getData:function(_3){if(_3){return this.cal[_3];}}}).inits(function(){vjo.darwin.tracking.sojourner.CalData=new vjo.darwin.tracking.sojourner.CalData();}).endType();
vjo.ctype("vjo.darwin.tracking.sojourner.TrackingRespHdl").needs(["vjo.darwin.tracking.sojourner.SojData","vjo.darwin.tracking.sojourner.CalData"]).props({handleResponse:function(_1){if(_1.errors&&_1.errors.length>0){vjo.darwin.tracking.sojourner.SojData.sojD="";}
if(_1.response&&_1.response.dataMap&&_1.response.dataMap.SOJDATA){vjo.darwin.tracking.sojourner.SojData.sojD=_1.response.dataMap.SOJDATA;}
if(_1.response&&_1.response.dataMap&&_1.response.dataMap.TDATA){vjo.darwin.tracking.sojourner.CalData.setData(_1.svcId,_1.response.dataMap.TDATA);}}}).endType();
vjo.ctype("vjo.darwin.tracking.enabler.TrackingEnablerUtil").needs("vjo.dsf.EventDispatcher").needs("vjo.dsf.utils.URL").needs("vjo.dsf.cookie.VjCookieJar").props({seekParent:function(_1,_2){if(!_1||!_1.tagName){return"";}
if(_1.tagName.toLowerCase()=="a"||_1.tagName.toLowerCase()=="area"){return _1;}
if(_1.tagName.toUpperCase()=="INPUT"&&_1.getAttribute("type")&&_1.getAttribute("type").toUpperCase()=="SUBMIT"){return _1;}
if(_2>0){return this.seekParent(_1.parentNode,_2-1);}else{return"";}},splitParm:function(_3){var v=[-1,-1,-1,-1];var f=_3.split(".");for(var i=0;i<f.length;i++){var s=f[i].substr(0,1);if(s=="p"){v[0]=f[i].substr(1);}
if(s=="c"){v[1]=f[i].substr(1);}
if(s=="m"){v[2]=f[i].substr(1);}
if(s=="l"){v[3]=f[i].substr(1);}}
return v;},enc:function(i){var A=["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"];var P="z";var _b="";var B=A.length;var _d;if(i==-1){return P;}
while(i>=B){_d=i%B;_b=A[_d]+_b;i=i/B|0;}
_b=A[i]+_b;return _b;},checkSiteDomain:function(_e){var dd=document.domain,i=dd.indexOf(".ebay.");if(i!=-1){dd=dd.substr(i+1);}
if(_e&&_e.length>0){if(_e.charAt(0)=="/"){return true;}else{if(_e.indexOf(dd)==-1){return false;}}
return true;}
return dd;}}).endType();
vjo.ctype("vjo.darwin.tracking.enabler.TrackingEnabler").needs("vjo.dsf.utils.URL").needs("vjo.dsf.cookie.VjCookieJar").needs("vjo.darwin.tracking.enabler.TrackingEnablerUtil").props({rewriteURLs:function(_1,_2,_3,_4,_5){if(_1.nativeEvent===null||_1.nativeEvent===undefined){return;}
var _6=_1.nativeEvent.srcElement||_1.nativeEvent.target;if(_6===null||_6===undefined){return;}
if(_6.tagName.toLowerCase()=="img"||_6.tagName.toLowerCase()=="span"){_6=_6.parentNode;}
var _7=_6.getAttribute(_4);if(_7===null){return;}
_7=_7.split(_5);if(_7[0]){var _8=_6.href;if(_8&&vjo.darwin.tracking.enabler.TrackingEnablerUtil.checkSiteDomain(_8)){_8=vjo.dsf.utils.URL.addArg(_8,_2,_7[0]);if(_7[1]){_8=vjo.dsf.utils.URL.addArg(_8,_3,_7[1]);}
_6.href=" "+_8;}}},copySIDToCookie:function(_9,_a,_b,_c){var _d=";";var vj=vjo.dsf.cookie.VjCookieJar;var u=vjo.darwin.tracking.enabler.TrackingEnablerUtil;var x="undefined";if(typeof(_GlobalNavHeaderCookieTracking)==x||!_GlobalNavHeaderCookieTracking){return this.rewriteURLs(_9,_a,_c,_b,_d);}
if(typeof(_GlobalNavHeaderStatic)!=x&&_GlobalNavHeaderStatic){vj.writeCookielet("ds2","sotr");return;}
var pid=_GlobalNavHeaderSrcPageId;var V="a";var P="zzzz";var _14=_9.nativeEvent.srcElement||_9.nativeEvent.target;if(!_14){return;}
_14=u.seekParent(_14,3);if(!_14){return;}
var url=_14.href;var isF=false;var _17=_14.getAttribute(_b);if(_14.tagName=="INPUT"&&_14.getAttribute("type").toUpperCase()=="SUBMIT"){var _18=document.getElementsByName(_a);url=_14.form.action;for(var i=0;i<_18.length;i++){if(_18[i].tagName=="INPUT"&&_18[i].getAttribute("type")&&_18[i].getAttribute("type").toUpperCase()=="HIDDEN"&&_18[i].form==_14.form){isF=true;_17=_18[i].value;}}}
if(!u.checkSiteDomain(url)){return;}
var v=[-1,-1,-1,-1];var _1b=[-1,-1,-1,-1];var oc=vj.readCookie("ds2","sotr");if(oc&&oc.length==12&&oc.substr(0,1)=="a"){var _1d=oc.substr(1,4);if(_1d!="zzzz"){_1b[0]=_1d;}
_1d=oc.substr(5,1);if(_1d!="z"){_1b[1]=_1d;}
_1d=oc.substr(6,3);if(_1d!="zzz"){_1b[2]=_1d;}
_1d=oc.substr(9,3);if(_1d!="zzz"){_1b[3]=_1d;}}
var b=false;if(_17){_17=_17.split(_d);var _1f;if(url&&_17[0]){_1f=_17[0];if(_17[1]){try{url=vjo.dsf.utils.URL.addArg(url,_c,_17[1]);_14.href=" "+url;}
catch(e){}}
v=u.splitParm(_1f);b=true;}}
if(!b){var _20=_a+"=";if(url&&url.indexOf(_20)!=-1){try{var p=(url.substr(url.indexOf(_20)+_20.length)).split("&");_1f=p[0];v=u.splitParm(_1f);b=true;}
catch(e){}}}
if(!b&&v[0]==-1){if(!pid||!_9.nativeEvent){return;}else{v[0]=pid;b=true;}}
if(isF&&v[0]===0&&pid){v[0]=pid;}
if(b){var cv=V;if(v[0]==-1&&_1b[0]!=-1){cv+=_1b[0];}else{cv+=(u.enc(v[0])+P).substr(0,4);}
if(v[1]==-1&&_1b[1]!=-1){cv+=_1b[1];}else{cv+=u.enc(v[1]);}
if(v[2]==-1&&_1b[2]!=-1){cv+=_1b[2];}else{cv+=(u.enc(v[2])+P).substr(0,3);}
if(v[3]==-1&&_1b[3]!=-1){cv+=_1b[3];}else{cv+=(u.enc(v[3])+P).substr(0,3);}
vj.writeCookielet("ds2","sotr",cv);}}}).endType();
vjo.itype("vjo.dsf.common.IJsHandler").protos({handle:function(_1){}}).endType();
vjo.ctype("vjo.darwin.tracking.enabler.TrackingModuleEnabler").needs("vjo.dsf.utils.URL").needs("vjo.dsf.typeextensions.string.Comparison").needs("vjo.dsf.cookie.VjCookieJar").needs("vjo.darwin.tracking.enabler.TrackingEnablerUtil").satisfies("vjo.dsf.common.IJsHandler").protos({constructs:function(_1,_2,_3,_4){this.sCid=_1;this.sParms=_2;this.sCidParms=_3;this.sDelim=_4;this.oCJ=vjo.dsf.cookie.VjCookieJar;this.oU=vjo.darwin.tracking.enabler.TrackingEnablerUtil;},logModuleId:function(_5){var V="a";var P="zzzz";if(!_5.nativeEvent||_5.nativeEvent===null||_5.nativeEvent===undefined){return;}
var sc=false;if(typeof(_GlobalNavHeaderCookieTracking)!="undefined"&&_GlobalNavHeaderCookieTracking){sc=true;}
var _9=_5.nativeEvent.srcElement||_5.nativeEvent.target;if(!_9||_9===null||_9===undefined){return;}
_9=this.oU.seekParent(_9,3);if(!_9){return;}
var _a=this.sCidParms.split(this.sDelim);if(_a[0]&&_9&&_9.href&&!_9.href.has("javascript:")){if(sc){var v=[-1,-1,-1,-1];v=this.oU.splitParm(_a[0]);var _c=V;_c+=(this.oU.enc(v[0])+P).substr(0,4);_c+=this.oU.enc(v[1]);_c+=(this.oU.enc(v[2])+P).substr(0,3);_c+=(this.oU.enc(v[3])+P).substr(0,3);this.oCJ.writeCookielet("ds2","sotr",_c);}else{var _d=vjo.dsf.utils.URL.addArg(_9.href,this.sCid,_a[0]);if(_a[1]){_d=vjo.dsf.utils.URL.addArg(_d,this.sParms,_a[1]);}
_9.href=" "+_d;}}},getAnchor:function(_e){var e=_e;if(e&&e.tagName){if(!e.tagName.is("A")&&!e.tagName.is("INPUT")&&(e.tagName.is("INPUT")&&e.getAttribute("type")!="SUBMIT")){e=this.getAnchor(e.parentNode);}
return e;}},handle:function(_10){this.logModuleId(_10);}}).endType();
vjo.ctype("vjo.darwin.core.ebayheader.rover.FooterRover").needs("vjo.dsf.cookie.VjCookieJar").props({roverService:function(_1){if(document.location.protocol.has("https:")){return;}
this.command=_1;if(!_1){return;}
if(!this.isCookieValid()){return;}
vjo.dsf.EventDispatcher.addEventListener(window,"load",this.sendRequest,this);},sendRequest:function(){var _2=new vjo.dsf.assembly.VjClientAssemblerRequest(this.command,this.handleResponse,this,"cb",false);vjo.dsf.assembly.VjClientAssembler.load(_2);},isCookieValid:function(){var _3=vjo.dsf.cookie.VjCookieJar.readCookie("dp1","idm");if(!_3){return true;}else{return false;}},handleResponse:function(_4){if(_4&&_4.length>1){var _5=_4.length-1;for(i=0;i<_5;i++){this.createImage(_4[i]);}
this.setCookieExpiration(_4[_5]);}},createImage:function(_6){if(_6&&_6.length>1){var _7=document.createElement("IMG");_7.width="1";_7.height="1";_7.src=_6;document.body.appendChild(_7);}},setCookieExpiration:function(_8){if(typeof _8=="number"&&_8>0){vjo.dsf.cookie.VjCookieJar.writeCookielet("dp1","idm","1",_8/86400,"");}}}).endType();
vjo.ctype("vjo.darwin.core.dynamicmenu.ReplaceHandler").props({replace:function(_1,_2,_3){var dm=vjo.Registry.get(_1);if(null!=dm){dm.replaceJSONDataHandler(_2,_3);}}}).endType();
vjo.needs("vjo.dsf.typeextensions.string.Comparison");vjo.ctype("vjo.dsf.typeextensions.string.Trim").endType();String.prototype.trim=function(){var s=this;while(s.substring(0,1).isAny(" ","\n","\r")){s=s.substring(1,s.length);}
while(s.substring(s.length-1,s.length).isAny(" ","\n","\r")){s=s.substring(0,s.length-1);}
return s;};
vjo.ctype("vjo.darwin.core.globalheader.searchbox.SearchBox").needs(["vjo.dsf.client.Browser","vjo.dsf.typeextensions.string.Trim"]).props({Focus:function(_1){var _2=vjo.dsf.Element.get(_1),B=vjo.dsf.client.Browser.bIE;if(typeof(_2)!="undefined"&&_2){if(B.bIE&&B.iVer==6){setTimeout(function(){_2.focus();},0);}else{_2.focus();}}},IeOptionDisabler:function(_3){if(vjo.dsf.client.Browser.bIE){var sl=vjo.dsf.Element.get(_3),idx;if(sl){sl.onchange=function(){idx=this.selectedIndex=(this.options[this.selectedIndex].disabled)?idx:this.selectedIndex;};sl.onfocus=function(){idx=this.selectedIndex;};this.greydisabledoption(sl);}}},greydisabledoption:function(e){var i,op;for(i=0;i<e.options.length;i++){op=e.options[i];if(op.disabled){op.style.color="graytext";}}}}).endType();
vjo.ctype("vjo.darwin.tracking.impression.Manager").needs(["vjo.dsf.cookie.VjCookieJar","vjo.dsf.EventDispatcher"]).protos({constructs:function(_1){if(!_1){return;}
var R=vjo.Registry,id="_pim",t=R.get(id);if(!t){t=this;t.vj$.EventDispatcher.add("body","mousedown",t.onMouseDown,t);R.put(id,t);}
t.sID=_1;},onMouseDown:function(){this.vj$.VjCookieJar.writeCookielet("ebay","psi",this.sID);}}).endType();
vjo.ctype("vjo.darwin.tracking.rover.Rover").needs("vjo.dsf.cookie.VjCookieJar").props({roverTrack:function(){var _1=new Date().getTime();var _2=vjo.darwin.tracking.rover.Rover.getClientOffset(_1);var _3=vjo.dsf.cookie.VjCookieJar.readCookieObj("npii","tpim");if(_3==null||_3.value==""){return;}
var _4=parseInt(_3.maxage,16)*1000;if(_4>0){var _5=_4-_1+_2;var _6=15552000000;if((_5>_6||_5<0)&&typeof(RoverSyncDropped)=="undefined"&&typeof(RoverNsCapable)=="undefined"){vjo.darwin.tracking.rover.Rover.dropRoverSyncImage();}}},dropRoverSyncImage:function(){if(typeof(RoverDomainBaseUrl)!=="undefined"&&RoverDomainBaseUrl.length>0){var im=document.createElement("img");im.width="1";im.height="1";im.src=RoverDomainBaseUrl+"/roversync/?rtpim=1&mpt="+new Date().getTime();document.body.appendChild(im);}},getClientOffset:function(_8){var _9;var _a=vjo.dsf.cookie.VjCookieJar.readCookie("ebay","cos");if(_a!==null&&_a.length>0){_9=parseInt(_a,16)*1000;}else{if(typeof(svrGMT)!=="undefined"){_9=_8-svrGMT;var _b=Math.round(_9/1000);if(!isNaN(_b)){vjo.dsf.cookie.VjCookieJar.writeCookielet("ebay","cos",_b.toString(16));}}}
if(isNaN(_9)){_9=1800000;}
return _9;}}).endType();
vjo.ctype("vjo.darwin.core.utils.ElementUtils").needs("vjo.dsf.Element","E").props({oLst:[],get:function(_1,_2){var _3=[];if(typeof(_1)=="object"){for(var i in _1){var _5=_1[i];_3[i]=this.get(_5,_2);}
return _3;}
var t=this,elem=t.oLst[_1];if(!elem||!elem.parentNode||_2){t.oLst[_1]=this.vj$.E.get(_1,_2);}
return t.oLst[_1];}}).inits(function(){vjo.dsf.EventDispatcher.addEventListener(window,"unload",function(){vjo.darwin.core.utils.ElementUtils.oLst=null;});}).endType();
vjo.ctype("vjo.darwin.core.utils.WindowDimension").props({D:undefined,getBrowserDimension:function(){var s=self;var d=document;var de=d.documentElement;if(s.innerHeight){return[s.innerWidth,s.innerHeight];}else{if(de&&de.clientHeight){return[de.clientWidth,de.clientHeight];}}
return[d.body.clientWidth,d.body.clientHeight];},getScrollXY:function(){var _4=0,scrOfY=0,scrOfH=0,scrOfW=0,d=document.documentElement||document.body;if(typeof(window.pageYOffset)=="number"){return[window.pageXOffset,window.pageYOffset,document.height,document.width];}else{if(d){return[d.scrollLeft,d.scrollTop,d.scrollHeight,d.scrollWidth];}}
return[_4,scrOfY,scrOfH,scrOfW];},getOffsetPosition:function(_5){var e=_5,l=0,t=0,z=0,tz,h=e.offsetHeight,w=e.offsetWidth;while(e){l+=e.offsetLeft;t+=e.offsetTop;if(e.style){tz=parseInt(e.style.zIndex,10);z=!isNaN(tz)&&tz>z?tz:z;}
e=e.offsetParent;}
return[l,t,z,h,w];}}).endType();
vjo.ctype("vjo.darwin.core.globalheader.overlay.Overlay").needs("vjo.darwin.core.utils.ElementUtils","EU").needs("vjo.darwin.core.utils.WindowDimension","W").needs("vjo.dsf.EventDispatcher","ED").props({timer:null,currObj:[],contentObjRef:null,id:null,keepOpen:false,iOpen:false,init:function(_1,_2){if(document.location.protocol.has("https")&&!_2.has("https")){_2=_2.replace("http://pics.","https://securepics.");}
var _3=new Image();_3.src=_2;var t=this;t.id=_1;var _5=function(){var _6=vjo.dsf.Element.get(_1);if(_6){document.body.appendChild(_6);var E=t.vj$.ED;E.addEventListener(_6,"mouseout",t.closeOverlay,t);E.addEventListener(_6,"mouseover",t.cancelOpen,t);}};if(!window["overlayinit"]){t.vj$.ED.add("body","load",_5,t);window["overlayinit"]=true;}},openOverlay:function(_8){var t=this,CHV="chevron",l=t.vj$,E=l.EU,W=l.W;if(_8[0]!=CHV&&E.get(CHV)){E.get(CHV).className="gh-ai";}
if(_8){t.keepOpen=_8[6];}
_8=(_8)?_8:t.currObj;var tE=E.get(_8[0]),bD=W.getBrowserDimension(),ovrly=E.get(t.id),cO=(typeof(_8[1])=="object")?_8[1]:E.get(_8[1]);t.currObj=_8;if(!ovrly||!tE){return;}
var _b=ovrly.childNodes;var _c=(_b[0].innerHTML==undefined)?_b[1]:_b[0];clearTimeout(t.timer);var _d=(_c.offsetWidth>ovrly.offsetWidth);ovrly.className="gh-ovr "+_8[2];_c.className="gh-iovr ";t.trCss(true);var _e=(_d)?_c:ovrly;var c=_c.childNodes;if(t.contentObjRef&&c.length>0){t.contentObjRef.appendChild(c[0]);}
if(c[0]&&c[0].id==cO.id){}else{_c.innerHTML="";t.contentObjRef=cO.parentNode;_c.appendChild(cO);}
var wid=_e.offsetWidth;var ltz=W.getOffsetPosition(tE,bD);var _12=(ltz[0]+ltz[4])-wid;var _13=bD[0]-(ltz[0]+wid);_8[4]=(_8[4])?_8[4]:0;_8[5]=(_8[5])?_8[5]:0;var tp=(ltz[1]+ltz[3]+_8[4])+"px",lt=(!_8[3]&&(_13>10||(_13>_12)))?(ltz[0]+_8[5])+"px":(_12+_8[5])+"px";t.applyStyle(ovrly,lt,tp);t.iOpen=true;},applyStyle:function(obj,_16,top){if(obj){var s=obj.style;s.left=_16;s.top=top;}},trCss:function(_19){var t=this,o=t.currObj?(t.vj$.EU.get(t.currObj[0])):null;if(o){var c=o.className?o.className:"",s="gh-hso";if(!_19&&c.indexOf(s)>-1){return;}
o.className=(_19)?c.replace(s,""):c+" "+s;}},cancelOpen:function(){var t=this;t.trCss();clearTimeout(t.timer);},closeOverlay:function(_1d){var t=this;if(t.keepOpen){return;}
t.trCss(true);var f=function(){t.close();t.iOpen=false;};_1d=(_1d)?_1d:250;t.timer=setTimeout(f,_1d);},close:function(e){var t=this;var elm=(e)?e.nativeEvent.srcElement||e.nativeEvent.target:null;if(elm&&t.currObj[0]==elm.id){return;}
t.applyStyle(t.vj$.EU.get(t.id),"-1000px","-1000px");t.currObj=[];t.keepOpen=null;t.iOpen=false;}}).endType();
vjo.darwin.core.globalheader.overlay.Overlay.init("gbh_ovl","http://pics.ebaystatic.com/aw/pics/homepage/imgMenuBg.png");
vjo.ctype("vjo.darwin.core.globalheader.utils.HeaderMenu").needs("vjo.dsf.utils.JsLoader","JSL").needs("vjo.dsf.document.Element","E").protos({jsonObj:null,menuObj:[],constructs:function(_1){var t=this;t.m=_1;t.jsUrl=null;t.domain=null;},replaceJSONDataHandler:function(_3,_4){if(_3!=null){this.handler=_3;}
if(_4!=null){this.domain=_4;}},setHandlerSource:function(_5){if(_5){this.jsUrl=_5;}},clearHS:function(){this.jsUrl=null;},getHandlerSource:function(){return this.jsUrl;},getHandler:function(){return this.handler;},setHandler:function(h){this.m.handler=h;},loadJs:function(_7){var t=this;var _9=t.jsUrl;var _a=function(){t.getMenuHtml(_7);};if(_9!==null&&!t.jsonObj){t.vj$.JSL.load(_9,_a,t);}else{t.getMenuHtml(_7);}},getMenuHtml:function(_b){var t=this;t.jsonObj=true;var _d=null;var _e=t.m.domain;var _f=this.menuObj[t.m.handler];var _10=document.getElementById(t.m.parentTriggerId);var _11=(_10)?-_10.offsetWidth:0;var arr=[t.m.triggerId,_f,t.m.cssClzName,t.m.isRtAlign,t.m.topMargin,_11,_b];var _13=vjo.darwin.core.globalheader.overlay.Overlay;if(_f){_13.openOverlay(arr);return;}
with(this){var i,j,c,n,h,ipc,_1a,_1b=".paradise.qa.ebay.com",sTemp2=".qa.ebay.com",u,ff,ll,sTemp3=".no-pool-name.qa.ebay.com";var E=vj$.Element;var lh=window.location.href;var _1e=window[m.handler];if(!_1e){return;}
var _1f=_1e(),items=_1f["items"],l=items.length;if(lh.has("ebay.com/")){for(i=0;i<l;i++){if(items[i].value.has("eBay Motors")){items[i].value="Cars, Boats, Vehicles & Parts";items[i].url="http://www.motors.ebay.com";items.sort(sortByValue);break;}}}
var _20=m.noOfColumns||1;ipc=Math.ceil(l/_20);h="<table border='0' cellpadding='0' cellspacing='0'";h+=">";for(i=0;i<ipc;i++){h+="<tr>";for(j=0;j<_20;j++){h+="<td nowrap>";n=i+(j)*ipc;c=items[n];if(c){if(typeof(c.url)!="undefined"){u=c.url;if(_e){if(c.url.indexOf(_1b)!=-1){u=c.url.replace(_1b,_e);}else{if(c.url.indexOf(sTemp3)!=-1){u=c.url.replace(sTemp3,_e);}else{if(c.url.indexOf(sTemp2)!=-1){u=c.url.replace(sTemp2,_e);}}}}
u=cobrandUrl(u);h+="<a href='"+u+"'>";h+=c.value;h+="</a>";}else{u=c.value;ff=u.indexOf("href=\"");if(ff==-1){h+=u;}else{ff+=6;ll=u.lastIndexOf("\"");u=u.substr(ff,ll-ff);h+=c.value.substr(0,ff)+cobrandUrl(u)+c.value.substr(ll);}}}else{h+="&nbsp;";}
h+="</td>";}
h+="</tr>";}
h+="</table>";_1a=h;if(_d){_1a=_d.replace("##1##",h);}
var _21=document.createElement("spanWrap");_21.style.display="none";var _22=document.createElement("span");_22.setAttribute("id",t.m.triggerId+"cat");_22.className="gh-smn";_22.innerHTML=_1a;_21.appendChild(_22);document.body.appendChild(_21);arr[1]=_22;menuObj[m.handler]=_22;_13.openOverlay(arr);return _1a;}},sortByValue:function(_23,_24){if(_23.value.has("Everything")){return 1;}else{if(_24.value.has("Everything")){return-1;}else{return _23.value<_24.value?-1:(_23.value>_24.value?1:0);}}},cobrandUrl:function(_25){var lh=window.location.href;if(!lh.has("sandbox.")){return _25;}
var u="undefined",cc,cf;if(this.oCobrand==null&&typeof(ebay)!=u&&typeof(ebay.oDocument)!=u){cc=ebay.oDocument._getControl("cobrandCollection");if(cc){cf=cc._getControl("cobrandFunctions");this.oCobrand=cf;}}else{cf=this.oCobrand;}
var lc=(_25.substring(_25.length)!="/")?"/":"";if(cf&&typeof(cf.cobrandURL)!=u){return cf.cobrandURL(_25+lc);}else{if(typeof(vjo.darwin.core.cobrand)!=u&&typeof(vjo.darwin.core.cobrand.EbaySandbox)!=u){return vjo.darwin.core.cobrand.EbaySandbox.cobrandURL(_25);}}
return _25;}}).endType();
vjo.ctype("vjo.darwin.core.globalheader.utils.HeaderMenuObj").protos({constructs:function(_1,_2,_3,_4,_5,_6,_7){var t=this;t.noOfColumns=_5||1;t.handler=_6;t.triggerId=_1;t.cssClzName=_2;t.isRtAlign=_4;t.parentTriggerId=_3;t.topMargin=_7;}}).endType();
vjo.ctype("vjo.dsf.utils.Css").needs("vjo.dsf.Element").props({apply:function(_1,_2){var e=vjo.dsf.Element.get(_1),c;if(e&&_2){c=this.createStyle(_2);if(c){e.appendChild(c);}}
return c;},createStyle:function(_4){var c=document.createElement("style"),t;c.type="text/css";if(_4){if(c.styleSheet){c.styleSheet.cssText=_4;}else{t=document.createTextNode(_4);c.appendChild(t);}}
return c;}}).endType();
vjo.ctype("vjo.darwin.core.globalheader.utils.EventReg").needs("vjo.dsf.EventDispatcher","ED").needs("vjo.darwin.core.utils.ElementUtils","E").needs("vjo.darwin.core.utils.WindowDimension","W").needs("vjo.darwin.core.globalheader.overlay.Overlay","O").needs("vjo.darwin.core.globalheader.utils.HeaderMenu","HM").needs("vjo.darwin.core.globalheader.utils.HeaderMenuObj","HMO").needs("vjo.dsf.client.Browser","BR").needs("vjo.dsf.utils.Css","CS").props({searchBarResize:function(){var t=this,l=t.vj$,fn=function(_2){var _3=l.E.get("headerSearch");if(!_3){return;}
var _4=_3.offsetWidth;var o=l.E.get("_nkw");var _6=864;var _7=400;var _8=759;if(!o||_4<_8){return;}
o.style.width=(_4<_6)?(_7-(_6-_4))+"px":_7+"px";};l.ED.addEventListener(window,"resize",fn,window);setTimeout(fn,100);},registerMouseEvent:function(_9,_a,_b,_c,_d,_e,_f,_10,_11){var t=this;var _13=function(){l=t.vj$,E=l.E,ED=l.ED,O=l.O;_b=_b||"mouseover";_c=_c||"mouseout";for(var _14 in _9){var _15=_9[_14][0]||_14;var _16=_14;var _17=_9[_14][1];var _18=false;for(var i in _d){if(_d[i]==_14){_18=true;break;}}
if(_10){var obj=new l.HM(new vjo.darwin.core.globalheader.utils.HeaderMenuObj(_16,_a,null,_18,1,_17,_e));ED.add(_16,_b,t.open(obj));}else{ED.add(_16,_b,t.openOvl(_15,_17,_a,_18,_e,_11));}
if(_c){ED.add(_16,_c,t.close(_f));}}};t.vj$.ED.add("body","load",_13,t);},openOvl:function(id,_1c,css,_1e,_1f,_20){var t=this;return function(){t.vj$.O.openOverlay([id,_1c,css,_1e,_1f,_20]);};},registerVerisign:function(id,_23,_24,_25){var t=this;var l=t.vj$;var _28=function(){var _29=l.E.get(id);var _2a=l.W.getOffsetPosition(_29);var _2b=l.E.get(_24);var ovr=l.W.getOffsetPosition(_2b);var _2d=_2a[4];var _2e=ovr[3];var arr=[id,_2b,_23,false,-_2e,-_2d,true];l.O.openOverlay(arr);};var _30=function(){l.O.close();};var hdl=function(){l.ED.add(id,"click",_28,t);l.ED.add(_25,"click",_30,t);};l.ED.add("body","load",hdl);},changeBtStyle:function(id){var t=this,ED=t.vj$.ED;ED.add("body","load",function(){var o=t.vj$.E.get(id);if(o){ED.add(id,"mousedown",function(){o.className="gh-btn gh-bc";},t);ED.add(id,"mouseup",function(){o.className="gh-btn";},t);}},t);},changeHover:function(_35,_36,_37,_38,_39){var t=this,E=t.vj$.E,ED=t.vj$.ED,a1=E.get(_35),a2=E.get(_36);var fn1=function(){if(a1){a1.className+=" "+_37;}},fn2=function(){if(a1){a1.className=a1.className.replace(_37,"");}},fn3=function(){if(!t.vj$.O.iOpen){if(a2){a2.className+=" "+_37;}
var _3c=vjo.Registry.get(_38);if(_3c){_3c.loadJs(true);}}else{t.vj$.O.close();if(a2){a2.className="gh-ai";}}},fn4=function(_3d){var elm=_3d.nativeEvent.srcElement||_3d.nativeEvent.target;if(elm&&(elm.id==_36||vjo.dsf.Element.containsElement(t.vj$.E.get(_36),elm))){return;}
t.vj$.O.close(_3d);if(a2){a2.className="gh-ai";}};if(_35!=""&&_36!=""){ED.add(_36,"mouseover",fn1);ED.add(_36,"mouseout",fn2);ED.add(_36,"click",fn3);ED.add("body","click",fn4);}},registerClickEvent:function(_3f,_40,_41,_42,_43){var t=this,l=t.vj$;var _45=[_3f,_40,_41,_42,_43];var fun=function(){l.C.loadJs(_45);};l.ED.add(_40,"click",fun,this.vj$.C);},open:function(obj){return function(){obj.getMenuHtml();};},close:function(_48){var t=this;return function(){t.vj$.O.closeOverlay(_48);};},doctypeFix:function(){var t=this,d=document,b=t.vj$.BR;var _4b=d.childNodes[0].nodeValue;var _4c=(_4b)?_4b.toLowerCase():null;if(b.bIE&&b.iVer>7&&(!_4b||_4c.indexOf("doctype")<0||_4c.indexOf(".dtd")<0)){var s=t.vj$.CS.createStyle(".gh-w {font-size: x-small}");if(s){d.getElementsByTagName("head")[0].appendChild(s);}
return true;}
return false;}}).endType();
vjo.dsf.EventDispatcher.add("body","load",function(){var _1=[];_1["ie6"]=".gh-go, .gh-sbox input.gh-btn {overflow:visible; width:0; padding:4px 19px}";_1["ie7"]=".gh-go {padding:2px 6px}.gh-go {padding:0 3px; border:0 solid #ccc}.gh-sbox {height:1%}.coreFooterLinks a {font-size:xx-small !important}";_1["ie8"]=(vjo.darwin.core.globalheader.utils.EventReg.doctypeFix())?_1["ie7"]:"";var b=vjo.dsf.client.Browser;if(b.bIE){var s=vjo.dsf.utils.Css.createStyle(_1["ie"+b.iVer]);if(s){document.getElementsByTagName("head")[0].appendChild(s);}}});
vjo.darwin.core.globalheader.utils.EventReg.registerVerisign("verisign","gh-vs","vrsgncont","closeId");(function () {
var _r = vjo.Registry;
_r.put('ReskinFooterTrackingCompSpecGenerator_0',new vjo.darwin.tracking.enabler.TrackingModuleEnabler("_trksid", "_trkparms", "m571;", ";")); })();function FooterTrk(){ return {handle:function (event){ (function(){
var _d=vjo.dsf.EventDispatcher;
var _r=vjo.Registry;
_d.add('glbfooter','click',function(event) { this.handle(event); },_r._ReskinFooterTrackingCompSpecGenerator_0);})();  }};};vjo.dsf.EventDispatcher.add('body','load', new FooterTrk());(function () {
var _r = vjo.Registry;
function $o0(){return new vjo.darwin.tracking.enabler.TrackingModuleEnabler("_trksid","_trkparms","m570;",";");};_r.put('ReskinHeaderTrackingCompSpecGenerator_0',$o0()); _r.put('ReskinHeaderTrackingCompSpecGenerator_1',$o0()); _r.put('ReskinHeaderTrackingCompSpecGenerator_2',$o0()); })();function ReskinHeaderTrk(){ return {handle:function (event){ (function(){
var _d=vjo.dsf.EventDispatcher;
var _r=vjo.Registry;
function $0(){return function(event){return this.handle(event);};};_d.add('BrowseCategories-menu','click',$0(),_r._ReskinHeaderTrackingCompSpecGenerator_0);_d.add('gnheader','click',$0(),_r._ReskinHeaderTrackingCompSpecGenerator_1);_d.add('gbh_ovl','click',$0(),_r._ReskinHeaderTrackingCompSpecGenerator_2);_d.add('body','click',function(event){ vjo.darwin.tracking.enabler.TrackingEnabler.copySIDToCookie(event, "_trksid", "_sp", "_trkparms");  });})();  }};};vjo.dsf.EventDispatcher.add('body','load', new ReskinHeaderTrk());
// en_US/e663/GH-661_ReskinSignInEbay_e66310949819_1_en_US
// b=10949819