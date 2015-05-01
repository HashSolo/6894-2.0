
vjo.ctype("vjo.darwin.core.globalheader.utils.SGuid").needs("vjo.dsf.cookie.VjCookieJar").props({writeSessionGuid:function(id){if(id&&id.length>0){vjo.dsf.cookie.VjCookieJar.writeCookieEx("cssg",id,1);}}}).endType();

vjo.ctype("vjo.darwin.core.globalheader.shoppingcart.ShoppingCart").needs("vjo.dsf.Element").needs("vjo.dsf.cookie.VjCookieJar").props({Refresh:function(_1,_2){var E=vjo.dsf.Element;if(this.IsShowCart()){var C=(this.GetCountNum()>0)?E.get(_1):E.get(_2);if(C){C.style.display="inline-block";}}},IsShowCart:function(){var _5=vjo.dsf.cookie.VjCookieJar.readCookie("dp1","pbf");if(_5){return vjo.dsf.cookie.VjCookieJar.getBitFlag(_5,47)==1;}},GetCountNum:function(){var _6,_7,_8,_9=0;_8=vjo.dsf.cookie.VjCookieJar.readCookie("dp1","exc");if(_8&&_8.length>0){_6=_8.indexOf(":");if(_6>0){_7=_8.substring(0,_6);if(/^\d+$/.test(_7)){_9=_7;}}}
return _9;}}).endType();

vjo.ctype("vjo.darwin.tracking.sojourner.SojData").props({sojD:null}).endType();

vjo.ctype("vjo.darwin.tracking.sojourner.CalData").props({setData:function(_1,_2){if(!_1||!_2){return;}
this.cal[_1]=_2;},getData:function(_3){if(_3){return this.cal[_3];}}}).inits(function(){this.cal={};}).endType();

vjo.ctype("vjo.darwin.tracking.sojourner.TrackingRespHdl").needs(["vjo.darwin.tracking.sojourner.SojData","vjo.darwin.tracking.sojourner.CalData"]).props({handleResponse:function(_1){if(_1.errors&&_1.errors.length>0){this.vj$.SojData.sojD="";}
if(_1.response&&_1.response.dataMap&&_1.response.dataMap.SOJDATA){this.vj$.SojData.sojD=_1.response.dataMap.SOJDATA;}
if(_1.response&&_1.response.dataMap&&_1.response.dataMap.TDATA){this.vj$.CalData.setData(_1.svcId,_1.response.dataMap.TDATA);}}}).endType();

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
if(v[0]==-1&&pid){v[0]=pid;}
if(isF&&v[0]===0&&pid){v[0]=pid;}
if(b){var cv=V;if(v[0]==-1&&_1b[0]!=-1){cv+=_1b[0];}else{cv+=(u.enc(v[0])+P).substr(0,4);}
if(v[1]==-1&&_1b[1]!=-1){cv+=_1b[1];}else{cv+=u.enc(v[1]);}
if(v[2]==-1&&_1b[2]!=-1){cv+=_1b[2];}else{cv+=(u.enc(v[2])+P).substr(0,3);}
if(v[3]==-1&&_1b[3]!=-1){cv+=_1b[3];}else{cv+=(u.enc(v[3])+P).substr(0,3);}
vj.writeCookielet("ds2","sotr",cv);}}}).endType();

vjo.ctype("vjo.dsf.typeextensions.string.Comparison").endType();String.prototype.has=function(_1){return(this.indexOf(_1)!=-1);};String.prototype.hasArg=function(_2){var a=_2,rv=false;if(typeof(a)=="string"){rv=this.has(a);}else{var aL=a.length;for(var j=0;j<aL&&!rv;j++){rv=this.has(a[j]);}}
return rv;};String.prototype.hasAny=function(){var a=arguments,l=a.length,rv=false;for(var i=0;i<l&&!rv;i++){rv=this.hasArg(a[i]);}
return rv;};String.prototype.hasAll=function(){var a=arguments,l=a.length;for(var i=0;i<l;i++){if(!this.hasArg(a[i])){return false;}}
return true;};String.prototype.is=function(s){return(this==s);};String.prototype.isAny=function(){var a=arguments,l=a.length,rv=false,aL;for(var i=0;i<l&&!rv;i++){if(typeof(a[i])=="string"){rv=(this==a[i]);}else{aL=a[i].length;for(var j=0;j<aL&&!rv;j++){rv=(this==a[i][j]);}}}
return rv;};

vjo.itype("vjo.dsf.common.IJsHandler").protos({handle:function(_1){}}).endType();

vjo.ctype("vjo.darwin.tracking.enabler.TrackingModuleEnabler").needs("vjo.dsf.utils.URL").needs("vjo.dsf.typeextensions.string.Comparison").needs("vjo.dsf.cookie.VjCookieJar").needs("vjo.darwin.tracking.enabler.TrackingEnablerUtil").satisfies("vjo.dsf.common.IJsHandler").protos({constructs:function(_1,_2,_3,_4){this.sCid=_1;this.sParms=_2;this.sCidParms=_3;this.sDelim=_4;this.oCJ=vjo.dsf.cookie.VjCookieJar;this.oU=vjo.darwin.tracking.enabler.TrackingEnablerUtil;},logModuleId:function(_5){var V="a";var P="zzzz";if(!_5.nativeEvent||_5.nativeEvent===null||_5.nativeEvent===undefined){return;}
var sc=false;if(typeof(_GlobalNavHeaderCookieTracking)!="undefined"&&_GlobalNavHeaderCookieTracking){sc=true;}
var _9=_5.nativeEvent.srcElement||_5.nativeEvent.target;if(!_9||_9===null||_9===undefined){return;}
_9=this.oU.seekParent(_9,3);if(!_9){return;}
var _a=this.sCidParms.split(this.sDelim);if(_a[0]&&_9&&_9.href&&!_9.href.has("javascript:")){if(sc){var v=[-1,-1,-1,-1];v=this.oU.splitParm(_a[0]);var _c=V;_c+=(this.oU.enc(v[0])+P).substr(0,4);_c+=this.oU.enc(v[1]);_c+=(this.oU.enc(v[2])+P).substr(0,3);_c+=(this.oU.enc(v[3])+P).substr(0,3);this.oCJ.writeCookielet("ds2","sotr",_c);}else{var _d=vjo.dsf.utils.URL.addArg(_9.href,this.sCid,_a[0]);if(_a[1]){_d=vjo.dsf.utils.URL.addArg(_d,this.sParms,_a[1]);}
_9.href=" "+_d;}}},getAnchor:function(_e){var e=_e;if(e&&e.tagName){if(!e.tagName.is("A")&&!e.tagName.is("INPUT")&&(e.tagName.is("INPUT")&&e.getAttribute("type")!="SUBMIT")){e=this.getAnchor(e.parentNode);}
return e;}},handle:function(_10){this.logModuleId(_10);}}).endType();

vjo.ctype("vjo.darwin.core.ebayheader.rover.FooterRover").needs("vjo.dsf.cookie.VjCookieJar").props({command:null,roverService:function(_1){if(document.location.protocol.has("https:")){return;}
this.command=_1;if(!_1){return;}
if(!this.isCookieValid()){return;}
vjo.dsf.EventDispatcher.addEventListener(window,"load",this.sendRequest,this);},sendRequest:function(){var _2=new vjo.dsf.assembly.VjClientAssemblerRequest(this.command,this.handleResponse,this,"cb",false);vjo.dsf.assembly.VjClientAssembler.load(_2);},isCookieValid:function(){var _3=vjo.dsf.cookie.VjCookieJar.readCookie("dp1","idm");if(!_3){return true;}else{return false;}},handleResponse:function(_4){if(_4&&_4.length>1){var _5=_4.length-1;for(var i=0;i<_5;i++){this.createImage(_4[i]);}
this.setCookieExpiration(_4[_5]);}},createImage:function(_7){if(_7&&_7.length>1){var _8=document.createElement("IMG");_8.width="1";_8.height="1";_8.src=_7;_8.alt="";document.body.appendChild(_8);}},setCookieExpiration:function(_9){if(typeof _9=="number"&&_9>0){vjo.dsf.cookie.VjCookieJar.writeCookielet("dp1","idm","1",_9/86400,"");}}}).endType();

vjo.ctype("vjo.darwin.core.dynamicmenu.ReplaceHandler").props({replace:function(_1,_2,_3){var dm=vjo.Registry.get(_1);if(null!=dm){dm.replaceJSONDataHandler(_2,_3);}}}).endType();

vjo.ctype("vjo.darwin.core.ebaytoolbar.VjEbayToolbarDetect").needs(["vjo.dsf.client.ActiveX","vjo.dsf.utils.Bit","vjo.dsf.cookie.VjCookieJar"]).props({isEnabled:function(){var V1="eBayToolbar.Helper",V2="eBayToolbarCommLib.IWebEvent.1";var _2=this.vj$.ActiveX,cv=this.rw(false),oldt,newt;if(cv){if(cv==1||cv==2){return true;}
return false;}else{oldt=_2.isLibLoaded(V1);newt=_2.isLibLoaded(V2);if(oldt){this.rw(true,1);}else{if(newt){this.rw(true,2);}else{this.rw(true,3);}}
return oldt||newt;}},rw:function(_3,_4){var t=this,n=t.vj$,C=n.VjCookieJar,B=n.Bit;cl=C.readCookie("ebay","sbf");if(!_3){return B.getMulti(cl,74,2);}else{if(_3){_4=(_4=="")?3:_4;cl=B.setMulti(cl,74,2,_4);C.writeCookielet("ebay","sbf",cl);}}}}).endType();

vjo.ctype("vjo.darwin.core.bta.BuyerTransactionAlert").needs(["vjo.dsf.cookie.VjCookieJar","vjo.dsf.typeextensions.string.Comparison","vjo.dsf.client.Browser","vjo.dsf.Element","vjo.dsf.utils.Object","vjo.darwin.core.ebaytoolbar.VjEbayToolbarDetect"]).protos({sId:null,oL:null,iPollingInterval:0,iMaxHits:0,iHitTimeout:0,iServerHits:0,sLastCookieletValue:"",sServerUrl:null,sImgServer:null,sViewItemUrl:null,aAlertInfo:[],constructs:function(_1,_2,_3,_4,_5,_6,_7,_8,_9,_a,_b){this.sId=_1;this.iPollingInterval=_2;this.iMaxHits=_3;this.iHitTimeout=_4;this.iServerHits=0;this.sLastCookieletValue="";this.sServerUrl=_5;if(document.location.protocol.has("https")){_6=_6.replace("http://pics.","https://securepics.");}
this.sImgServer=_6;this.sViewItemUrl=_7;this.aAlertInfo=[["h:h:alt:2",_9,"icon/iconOutbid_16x16.gif"],["h:h:alt:3",_8,"icon/iconWatchB_16x16.gif"],["h:h:alt:4",_9,"icon/iconOutbid_16x16.gif"],["h:h:alt:5",_a,"icon/iconchanceBlu_16x16.gif"],["h:h:alt:tcr",_b,"icon/iconMailBlue_16x16.gif"]];var c,oC=vjo.dsf.client.Browser,oCJ=vjo.dsf.cookie.VjCookieJar;if((oC.bNav&&oC.iVer<7)||(oC.bOpera&&(oC.iVer+oC.fMinorVer)<0.5)||(oC.bIE&&oC.iVer<5)){return;}
c=oCJ.readCookie("ebaysignin");if(!c||!c.is("in")){return;}
c=oCJ.readCookie("dp1","a1p");if(c&&c.length>0&&parseInt(c)>0){return;}
if(vjo.darwin.core.ebaytoolbar.VjEbayToolbarDetect.isEnabled()){return;}},isEnabled:function(){var V1="eBayToolbar.Helper",V2="eBayToolbarCommLib.IWebEvent.1";with(this){var _f=vj$.ActiveX;return(_f.isLibLoaded(V1)||_f.isLibLoaded(V2));}},setValue:function(_10,_11){var oL=this.oL;if(oL){if(_10.is("")&&!oL.ctrld){return;}
if(_11){if(vjo.dsf.client.Browser.bFirefox){oL.textContent=_10;}else{oL.innerText=_10;}}else{oL.innerHTML=_10;}
oL.ctrld=1;}},onRefreshHdl:function(){var t=this;return function(){t.onRefresh();};},onRefresh:function(){var E=vjo.dsf.Element;if(!this.oL){this.oL=E.get(this.sId);}
if(!this.oL){return;}
var c=vjo.dsf.cookie.VjCookieJar.readCookie("npii","mri");if(c){return;}
c=vjo.dsf.cookie.VjCookieJar.readCookie("ebay","a2p");if(!c){this.onCookieExpire();return;}
var at=parseInt(c.charAt(8));if(isNaN(at)){return;}
if(at===0){this.setValue("");return;}
var nrt=parseInt(c.substring(0,8),16)*1000;if(isNaN(nrt)){return;}
var ct=new Date();ct=ct.getTime();if(at==6||at==9){if(!c.is(this.sLastCookieletValue)){this.iServerHits=0;}
this.setValue("");this.sLastCookieletValue=c;var t=(nrt>ct)?parseInt((nrt-ct)/1000):this.iPollingInterval;window.setTimeout(vjo.dsf.utils.Object.hitch(this,this.onCookieExpire),t*1000);return;}
if(ct>=nrt){this.onCookieExpire();return;}
this.iServerHits=0;var cfg=this.aAlertInfo;if(at<0&&at>=cfg.length){return;}
var ii=c.substring(9,c.lastIndexOf("."));if(!c.is(this.sLastCookieletValue)){var _1c=cfg[at-1],imgSrv=this.sImgServer;var _1d=imgSrv+"s.gif";var _1e="<img src=\""+_1d+"\" alt=\"\" width=\"10\" height=\"16\" style=\"vertical-align:middle\">|<img src=\""+_1d+"\" alt=\"\" width=\"10\" height=\"16\" style=\"vertical-align:middle\">";_1e+="<img src=\""+imgSrv+_1c[2]+"?t\" alt=\"\" style=\"vertical-align:middle\"><img src=\""+_1d+"\" alt=\"\" width=\"5\" height=\"16\" style=\"vertical-align:middle\">";var url=this.sViewItemUrl+"&item="+ii;_1e+="<a href=\""+url+"&ssPageName="+_1c[0]+"\">"+_1c[1]+"</a>";this.setValue(_1e);this.sLastCookieletValue=c;}
this.fireRefreshEvent();},fireRefreshEvent:function(_20){if(!_20){_20=this.iPollingInterval;}
window.setTimeout(vjo.dsf.utils.Object.hitch(this,this.onRefresh),_20*1000);},onCookieExpire:function(){var oCJ=vjo.dsf.cookie.VjCookieJar,signin=oCJ.readCookie("ebaysignin");if(!signin.has("in")){return;}
if(document.location.href.has("https:")){return;}
if(this.iServerHits<this.iMaxHits){this.iServerHits++;var ct=new Date();ct=ct.getTime();this.setValue("<img height=\"1\" width=\"1\" src=\""+this.sServerUrl+"&clientTime="+ct+"\" style=\"visibility:hidden;vertical-align:middle\" alt=\"\">");this.fireRefreshEvent(this.iHitTimeout);}else{this.setValue("");oCJ.writeCookielet("ebay","a2p","1111111101111111111.");}}}).endType();

vjo.ctype("vjo.darwin.core.ebayheader.timezone.TimeZone").needs(["vjo.dsf.cookie.VjCookieJar","vjo.dsf.utils.DecimalToHex"]).props({init:function(){this.vj$.VjCookieJar.writeCookielet("dp1","tzo",this.vj$.DecimalToHex.dec2Hex(new Date().getTimezoneOffset()));}}).inits(function(){vjo.darwin.core.ebayheader.timezone.TimeZone.init();}).endType();

vjo.ctype("vjo.darwin.core.ebayheader.rebate.RebateBox").needs("vjo.dsf.Element").needs("vjo.dsf.cookie.VjCookieJar").props({Refresh:function(_1,_2,_3){var E=vjo.dsf.Element;var _5=E.get(_1);if(_5){if(this.IsShowMagicRebate(_5,_2)){_5.style.display="inline-block";}else{if(this.IsShowVibrantCoupon()){_5.innerHTML=_3;var _6=_5.getElementsByTagName("a")[0];if(_6){_6.style.paddingTop="6px";}
_5.style.display="inline-block";}}}},IsShowVibrantCoupon:function(){var _7=vjo.dsf.cookie.VjCookieJar.readCookie("ebay","sbf");if(_7){return vjo.dsf.cookie.VjCookieJar.getBitFlag(_7,29)==1;}},IsShowMagicRebate:function(_8,_9){var _a=_8.getElementsByTagName("img")[0],rate=vjo.dsf.cookie.VjCookieJar.readCookie("npii","mri"),perc;if(this.IsExpired()){return false;}
if(!rate){return false;}
perc=this.GetRate(rate);if(perc){perc=this.IsValidRate(perc);}
if(perc!==null&&_a){if(perc!=_9){_a.src=_a.src.replaceToken(_a.src,_9,perc);}
return true;}else{return false;}},IsExpired:function(){var _b=new Date().getTime();var _c=this.getClientOffset(_b);var _d=vjo.dsf.cookie.VjCookieJar.readCookieObj("npii","mri");if(_d!==null){var _e=parseInt(_d.maxage,16)*1000;if(_e>0){var _f=_e-_b+_c;if(_f<0){return true;}}}
return false;},getClientOffset:function(_10){var _11;var _12=vjo.dsf.cookie.VjCookieJar.readCookie("ebay","cos");if(_12!==null&&_12.length>0){_11=parseInt(_12,16)*1000;}else{_11=3600000;}
return _11;},IsValidRate:function(_13){var R=parseInt(_13,10);return(R>0&&R<100)?R:null;},GetRate:function(cv){var ar=cv.split(":");return ar.length>3?ar[2]:null;}}).endType();

vjo.needs("vjo.dsf.typeextensions.string.Comparison");vjo.ctype("vjo.dsf.typeextensions.string.TokenReplacement").endType();String.prototype.replaceToken=function(_1,_2,_3){if(_2==_3){return _1;}
var rv=_1;while(rv.has(_2)){rv=rv.replace(_2,_3);}
return rv;};String.prototype.replaceTokensEx=function(_5){var rv=this,re,tkn,a=arguments,l=a.length;for(var i=1;i<l+1;i++){rv=this.replaceToken(rv,_5.replace("n",(i)),a[i]);}
return rv;};String.prototype.replaceTokens=function(){var rv=this,re,tkn,a=arguments,l=a.length;for(var i=0;i<l;i++){rv=this.replaceToken(rv,"<#"+(i+1)+"#>",a[i]);}
return rv;};

vjo.ctype("vjo.darwin.core.utils.ElementUtils").needs("vjo.dsf.Element","E").needs("vjo.dsf.EventDispatcher").props({oLst:[],get:function(_1,_2){var _3=[];if(typeof(_1)=="object"){for(var i in _1){var _5=_1[i];_3[i]=this.get(_5,_2);}
return _3;}
var t=this,elem=t.oLst[_1];if(!elem||!elem.parentNode||_2){t.oLst[_1]=this.vj$.E.get(_1);}
return t.oLst[_1];}}).inits(function(){vjo.dsf.EventDispatcher.addEventListener(window,"unload",function(){vjo.darwin.core.utils.ElementUtils.oLst=null;});}).endType();

vjo.ctype("vjo.darwin.core.utils.WindowDimension").props({getBrowserDimension:function(){var s=self;var d=document;var de=d.documentElement;if(s.innerHeight){return[s.innerWidth,s.innerHeight];}else{if(de&&de.clientHeight){return[de.clientWidth,de.clientHeight];}}
return[d.body.clientWidth,d.body.clientHeight];},getScrollXY:function(){var _4=0,scrOfY=0,scrOfH=0,scrOfW=0,d=document.documentElement||document.body;if(typeof(window.pageYOffset)=="number"){return[window.pageXOffset,window.pageYOffset,document.height,document.width];}else{if(d){return[d.scrollLeft,d.scrollTop,d.scrollHeight,d.scrollWidth];}}
return[_4,scrOfY,scrOfH,scrOfW];},getOffsetPosition:function(_5){var e=_5,l=0,t=0,z=0,tz,h=e.offsetHeight,w=e.offsetWidth;while(e){l+=e.offsetLeft;t+=e.offsetTop;if(e.style){tz=parseInt(e.style.zIndex,10);z=!isNaN(tz)&&tz>z?tz:z;}
e=e.offsetParent;}
return[l,t,z,h,w];}}).endType();

vjo.ctype("vjo.darwin.core.globalheader.overlay.Overlay").needs("vjo.darwin.core.utils.ElementUtils","EU").needs("vjo.darwin.core.utils.WindowDimension","W").needs("vjo.dsf.EventDispatcher","ED").props({timer:null,currObj:[],contentObjRef:null,id:null,keepOpen:false,iOpen:false,init:function(_1,_2){var t=this;t.id=_1;t.callFnOnClose=[];t.callFnOnOpen=[];var _4=function(){var _5=vjo.dsf.Element.get(_1);if(_5){document.body.appendChild(_5);var E=t.vj$.ED;E.addEventListener(_5,"mouseout",function(){t.closeOverlay(t.closeDelay);},t);E.addEventListener(_5,"mouseover",t.cancelOpen,t);}};if(!window["overlayinit"]){t.vj$.ED.add("body","load",_4,t);window["overlayinit"]=true;}},openOverlay:function(_7,_8,_9){var t=this;clearTimeout(t.timer);var f=function(){if(t.callFnOnOpen[_7[0]]){t.callFnOnOpen[_7[0]]();}
t.open(_7);};if(_7[0]&&!t.callFnOnOpen[_7[0]]&&_9){t.callFnOnOpen[_7[0]]=_9;}
if(t.iOpen){if(t.currObj[0]!=_7[0]){if(t.callFnOnClose[t.currObj[0]]){t.callFnOnClose[t.currObj[0]]();}}}
if(_8&&!t.iOpen){t.openTimer=setTimeout(f,_8);}else{f(_7);}},open:function(_c){var t=this,CHV=/chevron[0-9]+/,l=t.vj$,E=l.EU,W=l.W;t.closeDelay=_c[8];if(_c[0]&&!CHV.test(_c[0])&&E.get(_c[0])){var a1=E.get(_c[0]),a2=E.get("BrowseCategories"),hb=E.get("headerWrapper");if(hb){hb.className=hb.className.replace(" gh-zidx","");}
if(CHV.test(_c[0])){a1.className="gh-ai";}
if(a2){a2.className=a2.className.replace(" gh-hbdr","");a2.className=a2.className.replace(" gh-hs","");}}
if(_c){t.keepOpen=_c[6];}
_c=_c||t.currObj;var tE=E.get(_c[7])||E.get(_c[0]),bD=W.getBrowserDimension(),ovrly=E.get(t.id),cO=(typeof(_c[1])=="object")?_c[1]:E.get(_c[1]);t.currObj=_c;var _10=_c[2]||"";if(_10.indexOf("gh-vsmn")==-1){ovrly.style.width="";}
if(!ovrly||!tE){return;}
var _11=ovrly.childNodes;var _12=(_11[0].innerHTML==undefined)?_11[1]:_11[0];clearTimeout(t.timer);var _13=(_12.offsetWidth>ovrly.offsetWidth);ovrly.className="gh-ovr "+_c[2];_12.className="gh-iovr ";var obj=(_13)?_12:ovrly,c=_12.childNodes;if(t.contentObjRef&&c.length>0){t.contentObjRef.appendChild(c[0]);}
_c[4]=(_c[4])?_c[4]:0;_c[5]=(_c[5])?_c[5]:0;var _15;var _16=tE.offsetWidth-2-_c[5];if(c[0]&&c[0].id==cO.id){}else{_12.innerHTML="";t.contentObjRef=cO.parentNode;_15=document.createElement("div");var _17=document.createElement("div");_17.appendChild(_15);_17.appendChild(cO);_12.appendChild(_17);_15.className="gh-ext";_15.style.width=_16+"px";}
var wid=obj.offsetWidth,ltz=W.getOffsetPosition(tE,bD);var _19=(ltz[0]+ltz[4])-wid;var _1a=bD[0]-(ltz[0]+wid);var _1b=(!_c[3]&&(_1a>10||(_1a>_19)));var tp=(ltz[1]+ltz[3]+_c[4])+"px",lt=(_1b)?(ltz[0]+_c[5])+"px":(_19+_c[5])+"px";if(_15&&!_1b){var fw=ovrly.offsetWidth;_15.style.marginLeft=(fw-_16-2)+"px";}
t.applyStyle(ovrly,lt,tp);t.iOpen=true;},applyStyle:function(obj,_1f,top){if(obj){var s=obj.style;s.left=_1f;s.top=top;}},cancelOpen:function(){var t=this;clearTimeout(t.timer);},closeOverlay:function(_23,_24){var t=this;clearTimeout(t.openTimer);if(t.keepOpen){return;}
if(t.currObj[0]&&!t.callFnOnClose[t.currObj[0]]&&_24){t.callFnOnClose[t.currObj[0]]=_24;}
var f=function(){t.close();t.iOpen=false;};_23=(typeof(_23)=="number")?_23:t.closeDelay;t.timer=setTimeout(f,_23);},close:function(e){var t=this;var elm=(e)?e.nativeEvent.srcElement||e.nativeEvent.target:null;if(elm&&t.currObj[0]==elm.id){return;}
t.callFn();t.applyStyle(t.vj$.EU.get(t.id),"-1000px","-1000px");t.currObj=[];t.keepOpen=null;t.iOpen=false;},callFn:function(){var t=this;if(t.callFnOnClose[t.currObj[0]]){t.callFnOnClose[t.currObj[0]]();}}}).endType();

vjo.ctype("vjo.darwin.core.globalheader.utils.HeaderMenu").needs("vjo.dsf.utils.JsLoader","JSL").needs("vjo.dsf.Element","E").protos({jsonObj:null,menuObj:[],constructs:function(_1){var t=this;t.m=_1;t.jsUrl=null;t.domain=null;},replaceJSONDataHandler:function(_3,_4){if(_3!=null){this.handler=_3;}
if(_4!=null){this.domain=_4;}},setHandlerSource:function(_5){if(_5){this.jsUrl=_5;}},clearHS:function(){this.jsUrl=null;},getHandlerSource:function(){return this.jsUrl;},getHandler:function(){return this.handler;},setHandler:function(h){this.m.handler=h;},loadJs:function(_7){var t=this,url=t.jsUrl;if(url&&!t.jsonObj){var _9=function(){t.getMenuHtml(_7);};t.vj$.JSL.load(url,_9,t);}else{t.getMenuHtml(_7);}},getMenuHtml:function(_a){var t=this,m=t.m;t.jsonObj=true;var _c;var _d=t.m.domain;var _e=t.menuObj[t.m.handler];var _f=document.getElementById(t.m.parentTriggerId);var _10=(_f)?-_f.offsetWidth:0;var arr=[t.m.triggerId,_e,t.m.cssClzName,t.m.isRtAlign,t.m.topMargin,_10,_a,undefined,m.cDelay,t.m.parentTriggerId];var _12=vjo.darwin.core.globalheader.overlay.Overlay;if(_e){_12.openOverlay(arr,m.oDelay);return;}
var _13,i,j,k,c,h,ipc,u,ff,ll,E=t.vj$.Element,lh=window.location.href,dPrvdr=window[t.m.handler],data=dPrvdr?dPrvdr():null,items=data?data.items:[],l=items.length,qaUrls=[".paradise.qa.ebay.com",".no-pool-name.qa.ebay.com",".qa.ebay.com"];if(l<=0){return;}
if(lh.indexOf("ebay.com/")>=0){for(i=0;i<l;i++){if(items[i].value.has("eBay Motors")){items[i].value="Cars, Boats, Vehicles & Parts";items[i].url="http://www.motors.ebay.com";items.sort(t.sortByValue);break;}}}
var _1e=t.m.noOfColumns||1;ipc=Math.ceil(l/_1e);h="<table border='0' cellpadding='0' cellspacing='0'>";for(i=0;i<ipc;i++){h+="<tr>";for(j=0;j<_1e;j++){h+="<td nowrap>";c=items[j*ipc+i];if(c){if(c.url){u=c.url;if(_d){for(k=0;k<qaUrls.length;k++){var _1f=qaUrls[k];if(c.url.indexOf(_1f)>=0){u=c.url.replace(_1f,_d);break;}}}
u=t.cobrandUrl(u);h+="<a href='"+u+"'>";h+=c.value;h+="</a>";}else{u=c.value;ff=u.indexOf("href=\"");if(ff==-1){h+=u;}else{ff+="href=\"".length;ll=u.lastIndexOf("\"");u=u.substr(ff,ll-ff);h+=c.value.substr(0,ff)+t.cobrandUrl(u)+c.value.substr(ll);}}}else{h+="&nbsp;";}
h+="</td>";}
h+="</tr>";}
h+="</table>";_13=_c?_c.replace("##1##",h):h;var _20=document.createElement("spanWrap"),sp=document.createElement("span");_20.style.display="none";sp.id=t.m.triggerId+"cat";sp.className="gh-smn";sp.innerHTML=_13;_20.appendChild(sp);document.body.appendChild(_20);arr[1]=sp;t.menuObj[t.m.handler]=sp;_12.openOverlay(arr,m.oDelay);return _13;},sortByValue:function(_21,_22){if(_21.value.has("Everything")){return 1;}else{if(_22.value.has("Everything")){return-1;}else{return _21.value<_22.value?-1:(_21.value>_22.value?1:0);}}},cobrandUrl:function(_23){var lh=window.location.href;if(!lh.indexOf("sandbox.")>=0){return _23;}
var u="undefined",cc,cf;if(this.oCobrand==null&&typeof(ebay)!=u&&typeof(ebay.oDocument)!=u){cc=ebay.oDocument._getControl("cobrandCollection");if(cc){cf=cc._getControl("cobrandFunctions");this.oCobrand=cf;}}else{cf=this.oCobrand;}
if(typeof(cf.cobrandURL)!=u){return cf.cobrandURL(_23+lc);}else{if(typeof(vjo.darwin.core.cobrand)!=u&&typeof(vjo.darwin.core.cobrand.EbaySandbox)!=u){return vjo.darwin.core.cobrand.EbaySandbox.cobrandURL(_23);}}
return _23;}}).endType();

vjo.ctype("vjo.darwin.core.globalheader.utils.HeaderMenuObj").protos({constructs:function(_1,_2,_3,_4,_5,_6,_7,_8,_9){var t=this;t.noOfColumns=_5||1;t.handler=_6;t.triggerId=_1;t.cssClzName=_2;t.isRtAlign=_4;t.parentTriggerId=_3;t.topMargin=_7;t.oDelay=_8;t.cDelay=_9;}}).endType();

vjo.ctype("vjo.darwin.core.globalheader.utils.VNLMenu").needs("vjo.dsf.utils.JsLoader","JSL").needs("vjo.dsf.Element","E").needs("vjo.dsf.EventDispatcher","ED").props({urlLoaded:false,menuObj:[],loadJs:function(_1,_2){var t=this,url=_1[3];if(url&&!t.urlLoaded){t.m=_1;t.urlLoaded=true;url+="&cb=vjo.darwin.core.globalheader.utils.VNLMenu.callFn";t.vj$.JSL.load(url,function(){},t);}else{t.getMenuHtml(_1,_2);}},callFn:function(_4){var t=this;t.menuObj=_4;t.getMenuHtml(t.m);},getMenuHtml:function(_6,_7){var t=this,isClick=false,m=_6,triggerId=m[0],associateId=m[1],cssClzName=m[2],url=m[3],oDelay=m[4],cDelay=m[5];var _9=t.menuObj[triggerId];var _a=[triggerId,_9,cssClzName,false,0,0,isClick,associateId,cDelay];var _b=vjo.darwin.core.globalheader.overlay.Overlay;var _c=t.menuObj[triggerId];var sp=document.createElement("DIV");sp.id=triggerId+"_sub";sp.className="gh-smn";sp.innerHTML=_c;_a[1]=sp;if(_9){var _e=t.vj$.E.get(_b.id),aO=t.vj$.E.get(associateId);if(_e&&aO){_e.style.width=aO.offsetWidth+"px";}
_b.openOverlay(_a,oDelay,_7);return;}
return _c;},popup:function(id){var t=this,fn=function(evt){var obj=evt.nativeEvent.target||evt.nativeEvent.srcElement,href=obj.href,width=1030,height=800,top=Math.round((screen.height-height)/2),left=Math.round((screen.width-width)/2),params=["location=no","menubar=no","status=no","resizable=yes","scrollbars=yes","top="+top,"left="+left,"width="+width,"height="+height];window.open(href,"",params.join(","));return false;};t.vj$.ED.add(id,"click",fn,t);}}).endType();

vjo.ctype("vjo.dsf.utils.Css").needs("vjo.dsf.Element").props({apply:function(_1,_2){var e=vjo.dsf.Element.get(_1),c;if(e&&_2){c=this.createStyle(_2);if(c){e.appendChild(c);}}
return c;},createStyle:function(_4){var c=document.createElement("style"),t;c.type="text/css";if(_4){if(c.styleSheet){c.styleSheet.cssText=_4;}else{t=document.createTextNode(_4);c.appendChild(t);}}
return c;}}).endType();

vjo.ctype("vjo.darwin.core.globalheader.utils.EventReg").needs("vjo.dsf.EventDispatcher","ED").needs("vjo.darwin.core.utils.ElementUtils","E").needs("vjo.darwin.core.utils.WindowDimension","W").needs("vjo.darwin.core.globalheader.overlay.Overlay","O").needs("vjo.darwin.core.globalheader.utils.HeaderMenu","HM").needs("vjo.darwin.core.globalheader.utils.HeaderMenuObj","HMO").needs("vjo.darwin.core.globalheader.utils.VNLMenu","VM").needs("vjo.dsf.client.Browser","BR").needs("vjo.dsf.utils.Css","CS").needs("vjo.Registry","R").props({fn4Array:[],fn4Aggregated:function(_1){var t=this,fnArr=t.fn4Array,len=fnArr.length;if(t.vj$.O.iOpen){while(len--){fnArr[len].apply(null,[_1]);}}},browseCategories:function(id,_4){var mn=this.vj$.R.get(id);if(mn){mn.setHandlerSource(_4);}},searchBarResize:function(){var t=this,l=t.vj$,fn=function(_7){var _8=l.E.get("headerSearch");if(!_8){return;}
var _9=_8.offsetWidth;var o=l.E.get("_nkw");var _b=864;var _c=400;var _d=759;if(!o||_9<_d){return;}
o.style.width=(_9<_b)?(_c-(_b-_9))+"px":_c+"px";};l.ED.addEventListener(window,"resize",fn,window);setTimeout(fn,100);},registerMouseEvent:function(_e,_f,_10,_11,_12,_13,_14,_15,_16,_17,url){var t=this;var _1a=function(){l=t.vj$,E=l.E,ED=l.ED,O=l.O;_10=_10||"mouseover";_11=_11||"mouseout";for(var _1b in _e){var _1c=_e[_1b][0];var _1d=_1c||_1b;var _1e=_1b;var _1f=_e[_1b][1];var _20=false;for(var i in _12){if(_12[i]==_1b){_20=true;break;}}
if(_15=="HEADER_MENU"){var _22=new t.vj$.HMO(_1e,_f,null,_20,1,_1f,_13,_14[0],_14[1]);var obj=new l.HM(_22);ED.add(_1e,_10,t.open(obj));}else{if(_15=="VNL_MENU"){var arr=[_1e,_1c,_f,url,_14[0],_14[1]];ED.add(_1e,_10,t.openVNL(arr),t);}else{ED.add(_1e,_10,t.openOvl(_1d,_1f,_f,_20,_13,_16));}}
if(_11){ED.add(_1e,_11,t.closeMenu(_1e));}}};if(_17){_1a();}else{t.vj$.ED.add("body","load",_1a,t);}},closeMenu:function(_25){var t=this;return function(_27){t.vj$.O.closeOverlay(500,function(){t.trCss(_25,true);});if(!t.vj$.O.iOpen){t.trCss(_25,true);}else{t.trCss(_25);}};},trCss:function(_28,_29){var t=this,s="gh-hso",o=t.vj$.E.get(_28),c=o?(o.className||""):"";if(!_29&&c.indexOf(s)>-1){return;}
if(o){o.className=(_29)?c.replace(/gh-hso/g,""):c+" "+s;}},openVNL:function(arr){var t=this;return function(){vjo.darwin.core.globalheader.utils.VNLMenu.loadJs(arr,function(){t.trCss(arr[0]);});};},openOvl:function(id,_2e,css,_30,_31,_32){var t=this;return function(){t.vj$.O.openOverlay([id,_2e,css,_30,_31,_32]);};},registerVerisign:function(id,_35,_36,_37){var t=this;var l=t.vj$;var _3a=function(){var _3b=l.E.get(id);var _3c=l.W.getOffsetPosition(_3b);var _3d=l.E.get(_36);var ovr=l.W.getOffsetPosition(_3d);var _3f=_3c[4];var _40=ovr[3];var arr=[id,_3d,_35,false,-_40,-_3f,true];l.O.openOverlay(arr);};var _42=function(){l.O.close();};var hdl=function(){l.ED.add(id,"click",_3a,t);l.ED.add(_37,"click",_42,t);};l.ED.add("body","load",hdl);},changeBtStyle:function(id){var t=this,ED=t.vj$.ED;ED.add("body","load",function(){var o=t.vj$.E.get(id);if(o){ED.add(id,"mousedown",function(){o.className="gh-btn gh-bc";},t);ED.add(id,"mouseup",function(){o.className="gh-btn";},t);}},t);},registerAndCreateHeaderButtons:function(_47,_48,_49,_4a,_4b,_4c,_4d,_4e,_4f,_50){var t=this,len=_4b?_4b.length:0,ED=t.vj$.ED,R=vjo.Registry;while(len--){var _52={triggerId:_4d[len],cssClzName:_4a,parentTriggerId:_4c[len],isRtAlign:false,noOfColumns:_48[len],handler:_4b[len],domain:_49,topMargin:_47,oDelay:_50,cDelay:_50},hm=new t.vj$.HM(_52);R.put(_4f[len],hm);ED.add("body","load",(function(i){return function(evt){t.changeHover(_4c[i],_4d[i],_4e,_4f[i],_50);};})(len));}},changeHover:function(_55,_56,_57,_58,_59){var t=this,E=t.vj$.E,ED=t.vj$.ED,a1=E.get(_55),a2=E.get(_56);var fn1=function(){if(a1&&!a1.className.match(_57)){a1.className+=" "+_57;}},fn2=function(){if(a1&&(!t.vj$.O.iOpen||(t.vj$.O.bound&&a1!=t.vj$.O.bound.baseElm))){a1.className=a1.className.replace(_57,"");}},fn3=function(){t.vj$.O.bound={baseElm:a1,link:a2};t.setBdr(a1,a2);if(a2){a2.className+=" "+_57;}
var _5c=vjo.Registry.get(_58);if(_5c){_5c.loadJs(false,_59);}
return false;},fn4=function(_5d){t.vj$.O.closeOverlay(500,function(){t.setBdr(a1,a2,true);});if(!t.vj$.O.iOpen){chCss();}},chCss=function(){var _5e=t.vj$.O.bound;if(_5e&&_5e.link&&a2!=_5e.link){if(_5e.link){_5e.link.className="gh-ai";}
t.setBdr(_5e.baseElm,_5e.link,true);fn3();}else{if(a2){a2.className="gh-ai";}
t.setBdr(a1,a2,true);}};if(_55!=""&&_56!=""){ED.add(_56,"mouseover",fn3);ED.add(_56,"mouseout",fn4);ED.add(_55,"mouseover",fn3);ED.add(_55,"mouseout",fn4);t.fn4Array[t.fn4Array.length]=fn4;}},setBdr:function(a1,a2,_61){var t=this,hObj=t.vj$.E.get("headerWrapper"),c3=" gh-zidx";if(hObj){var c=hObj.className||"";hObj.className=(_61)?c.replace(/gh-zidx/g,""):c+c3;}
if(a1){a1.className=(_61)?"":"gh-hbdr";}
if(a2){a2.className=(_61)?"gh-ai":"gh-ai gh-hbdp";}},registerClickEvent:function(_64,_65,_66,_67,_68){var t=this,l=t.vj$;var _6a=[_64,_65,_66,_67,_68];var fun=function(){l.C.loadJs(_6a);};l.ED.add(_65,"click",fun,this.vj$.C);},open:function(obj){var t=this;return function(){var _6e=t.vj$.O.bound;if(_6e&&_6e.link){if(_6e.link){_6e.link.className="gh-ai";}
t.setBdr(_6e.baseElm,_6e.link,true);}
obj.getMenuHtml();};},doctypeFix:function(){var t=this,d=document,b=t.vj$.BR;var _70=d.childNodes[0].nodeValue;var _71=(_70)?_70.toLowerCase():null;if(b.bIE&&b.iVer>7&&(!_70||_71.indexOf("doctype")<0||_71.indexOf(".dtd")<0)){var s=t.vj$.CS.createStyle(".gh-w {font-size: x-small}");if(s){d.getElementsByTagName("head")[0].appendChild(s);}
return true;}
return false;},regFooterEvent:function(_73,_74,_75){var t=this,o=t.vj$.E.get(_75),ED=t.vj$.ED;ED.add(_73,"click",function(){if(o){o.style.display="block";var h=o.offsetHeight;if(h>0){o.style.height=(h-22)+"px";o.style.marginTop=-(h+15)+"px";}}});var _78=function(e){var el=e.nativeEvent.srcElement||e.nativeEvent.target;if((vjo.dsf.Element.containsElement(o,el)||el.id==_73)&&!vjo.dsf.Element.containsElement(vjo.dsf.Element.get(_74),el)){return;}
if(o){o.style.display="none";}};ED.add(_74,"click",_78);ED.add("body","click",_78);}}).endType();
(function(){var hasDocType = vjo.darwin.core.globalheader.utils.EventReg.doctypeFix(); var styles=[];styles['ie6']=".gh-go, .gh-sbox input.gh-btn {overflow:visible; width:0; padding:4px 19px}";styles['ie7']=".gh-go {padding:2px 6px}input.gh-go {padding:0 3px; border:0 solid #ccc}.coreFooterLinks a {font-size:xx-small !important}";styles['ie8']=(hasDocType) ? styles['ie7'] : null;var b=vjo.dsf.client.Browser;if(b.bIE && styles['ie'+b.iVer]) {var s=vjo.dsf.utils.Css.createStyle(styles['ie'+b.iVer]);if(s) document.getElementsByTagName('head')[0].appendChild(s); }})();
vjo.ctype("vjo.darwin.core.ebayheader.autocomplete.AutoCompleteLazyInit").props({_ready:false,_inited:false,_inputId:null,_formId:null,init:function(_1,_2){var t=this,ac=vjo.darwin.core.ebayheader.autocomplete.AutoComplete;t._inputId=_1;t._formId=_2;t._ready=true;if(ac&&!t._inited){ac.init(_1,_2);t._inited=true;}},callback:function(){var t=this;if(t._ready&&!t._inited){vjo.darwin.core.ebayheader.autocomplete.AutoComplete.init(t._inputId,t._formId);t._inited=true;}}}).endType();

vjo.ctype("vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteLayerLazyInit").props({_ready:false,_inited:false,_model:null,init:function(_1){var t=this,acl=vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteLayerInit;t._model=_1;t._ready=true;if(acl&&!t._inited){acl.init(_1);t._inited=true;}},callback:function(){var t=this;if(t._ready&&!t._inited){vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteLayerInit.init(t._model);t._inited=true;}}}).endType();
vjo.darwin.core.globalheader.overlay.Overlay.init("gbh_ovl", "http://p.ebaystatic.com/aw/pics/homepage/imgMenuBg.png");vjo.darwin.core.globalheader.utils.EventReg.changeBtStyle("ghSearch");
vjo.ctype("vjo.darwin.core.globalheader.searchbox.SearchBox").needs(["vjo.dsf.client.Browser","vjo.dsf.typeextensions.string.Trim"]).props({Focus:function(_1){var _2=vjo.dsf.Element.get(_1),B=vjo.dsf.client.Browser.bIE;if(typeof(_2)!="undefined"&&_2){if(B.bIE&&B.iVer==6){setTimeout(function(){_2.focus();},0);}else{_2.focus();}}},IeOptionDisabler:function(_3){if(vjo.dsf.client.Browser.bIE){var sl=vjo.dsf.Element.get(_3),idx;if(sl){sl.onchange=function(){idx=this.selectedIndex=(this.options[this.selectedIndex].disabled)?idx:this.selectedIndex;};sl.onfocus=function(){idx=this.selectedIndex;};this.greydisabledoption(sl);}}},greydisabledoption:function(e){var i,op;for(i=0;i<e.options.length;i++){op=e.options[i];if(op.disabled){op.style.color="graytext";}}}}).endType();

vjo.ctype("vjo.darwin.tracking.impression.Manager").needs(["vjo.dsf.cookie.VjCookieJar","vjo.dsf.EventDispatcher"]).protos({constructs:function(_1){if(!_1){return;}
var R=vjo.Registry,id="_pim",t=R.get(id);if(!t){t=this;t.vj$.EventDispatcher.add("body","mousedown",t.onMouseDown,t);R.put(id,t);}
t.sID=_1;},onMouseDown:function(){this.vj$.VjCookieJar.writeCookielet("ebay","psi",this.sID);}}).endType();

vjo.ctype("vjo.darwin.tracking.rover.Rover").needs("vjo.dsf.cookie.VjCookieJar").props({roverTrack:function(){var _1=new Date().getTime();var _2=vjo.darwin.tracking.rover.Rover.getClientOffset(_1);var _3=vjo.dsf.cookie.VjCookieJar.readCookieObj("npii","tpim");if(_3==null||_3.value==""){return;}
var _4=parseInt(_3.maxage,16)*1000;if(_4>0){var _5=_4-_1+_2;var _6=15552000000;if((_5>_6||_5<0)&&typeof(RoverSyncDropped)=="undefined"&&typeof(RoverNsCapable)=="undefined"){vjo.darwin.tracking.rover.Rover.dropRoverSyncImage();}}},dropRoverSyncImage:function(){if(typeof(RoverDomainBaseUrl)!=="undefined"&&RoverDomainBaseUrl.length>0){var im=document.createElement("img");im.width="1";im.height="1";im.src=RoverDomainBaseUrl+"/roversync/?rtpim=1&mpt="+new Date().getTime();document.body.appendChild(im);}},getClientOffset:function(_8){var _9;var _a=vjo.dsf.cookie.VjCookieJar.readCookie("ebay","cos");if(_a!==null&&_a.length>0){_9=parseInt(_a,16)*1000;}else{if(typeof(svrGMT)!=="undefined"){_9=_8-svrGMT;var _b=Math.round(_9/1000);if(!isNaN(_b)){vjo.dsf.cookie.VjCookieJar.writeCookielet("ebay","cos",_b.toString(16));}}}
if(isNaN(_9)){_9=1800000;}
return _9;}}).endType();

vjo.ctype("vjo.darwin.core.ebayheader.rtm.GlobalHeaderRtmDec").endType();if(typeof(_oGlobalNavRTMInfo)=="undefined"){_oGlobalNavRTMInfo={};_oGlobalNavRTMInfo.aRTMPlacementData=[];}

vjo.ctype("vjo.dsf.typeextensions.string.Decode").endType();String.prototype.decodeBase64=function(){var rv=this,len=rv.length,ret="",i=0;if(len===0){return ret;}
var _2,_3,_4="";var _5,_6,_7,_8="";var _9="ABCDEFGHIJKLMNOPQRSTUVWXYZ"+"abcdefghijklmnopqrstuvwxyz"+"0123456789+/=*";var _a=new RegExp("[^A-Za-z0-9+/=*]");if(_a.exec(rv)){return;}
do{_5=_9.indexOf(rv.charAt(i++));_6=_9.indexOf(rv.charAt(i++));_7=_9.indexOf(rv.charAt(i++));_8=_9.indexOf(rv.charAt(i++));_2=(_5<<2)|(_6>>4);_3=((_6&15)<<4)|(_7>>2);_4=((_7&3)<<6)|_8;ret+=String.fromCharCode(_2);if(!(_7>=64)){ret+=String.fromCharCode(_3);}
if(!(_8>=64)){ret+=String.fromCharCode(_4);}
_2=_3=_4=_5=_6=_7=_8="";}while(i<len);return ret;};String.prototype.decodeUTF8=function(){var s=this,len=s.length;var rs="";var i=0;var c=0,c1=0,c2=0;while(i<len){c=s.charCodeAt(i);if(c<128){rs+=String.fromCharCode(c);i++;}else{if((c>191)&&(c<224)){c2=s.charCodeAt(i+1);rs+=String.fromCharCode(((c&31)<<6)|(c2&63));i+=2;}else{c2=s.charCodeAt(i+1);c3=s.charCodeAt(i+2);rs+=String.fromCharCode(((c&15)<<12)|((c2&63)<<6)|(c3&63));i+=3;}}}
return rs;};

vjo.ctype("vjo.darwin.core.ebayheader.rtm.GlobalHeaderRtmCall").needs(["vjo.dsf.typeextensions.string.Decode","vjo.dsf.cookie.VjCookieJar","vjo.dsf.utils.Object"]).props({iTimer:null,submitRTMCall:function(_1){var un="undefined",lh=window.location.href;if(!lh.hasAny("catalog.")&&(lh.hasAny("offer.")||!(typeof(ebay)!=un&&typeof(ebay.oDocument)!=un&&ebay.oDocument._getControl("rtm")))&&!(typeof(vjo)!=un&&typeof(vjo.dsf)!=un&&typeof(vjo.dsf.ServiceEngine)!=un&&typeof(vjo.dsf.ServiceEngine.inProcHdl)!=un&&typeof(vjo.dsf.ServiceEngine.inProcHdl.svcHdls)!=un&&typeof(vjo.dsf.ServiceEngine.inProcHdl.svcHdls.RTM_CALLBACK_SERVICE)!=un)&&typeof(_oGlobalNavRTMInfo)!==un&&!(typeof(rtm)!=un)){if(_oGlobalNavRTMInfo.aRTMPlacementData.length>0){this.iTimer=window.setInterval(this.vj$.Object.hitch(this,"init"),1);}}else{if(_oGlobalNavRTMInfo.aRTMPlacementData.length>0){if(lh.hasAny("shop.","icatalog.","catalog.")&&!lh.hasAny("hub.")){this.init();}}}},getUid:function(){var _3=this.vj$.VjCookieJar.readCookie("dp1","u1p"),u1pDecoded="";if(_3){u1pDecoded=_3.decodeBase64().decodeUTF8();}
return u1pDecoded;},getGuid:function(){return this.vj$.VjCookieJar.readCookie("ebay","sgj");},hasUid:function(_4){if(_4){return true;}
if(this.getUid()&&this.getUid().has("@@__@@__@@")){return false;}
return true;},init:function(){if(typeof(vjo.darwin.core.rtm)=="undefined"){return;}
if(this.iTimer!=null){window.clearInterval(this.iTimer);}
var _5=_oGlobalNavRTMInfo.aRTMPlacementData,data,widths=[],htmlIds=[],heights=[],pids=[],dblclkUrls=[],defaultUrls=[],url;for(var i=0;i<_5.length;i++){data=_5[i];widths[i]=data.maxWidth;htmlIds[i]=data.htmlId;heights[i]=data.maxHeight;pids[i]=data.pid;dblclkUrls[i]="";defaultUrls[i]="";url=data.rtmUrl+"?RtmCmd&a=json"+(this.hasUid(data.userId)?("&l="+(data.userId?data.userId:this.getUid())):"")+"&g="+(data.gUid?data.gUid:this.getGuid())+"&ord="+data.ord+((data.oid)?"&i="+data.oid:"");}
url+="&p="+pids.join(":");_oGlobalNavRTMInfo.sRTMUrl=url;if(_5.length>0){var _7=new vjo.darwin.core.rtm.RTMInit({"url":url,"widths":widths,"htmlIds":htmlIds,"heights":heights,"pids":pids,"dblclkUrls":dblclkUrls,"defaultUrls":defaultUrls});_7.invoke({});}}}).endType();

vjo.ctype("vjo.darwin.core.greetings.VjGreetingsServer").needs(["vjo.dsf.utils.URL"]).props({handleClick:function(_1,_2){var _3=_2.srcElement||_2.target;if(_3){_3.href=vjo.dsf.utils.URL.addArg(_1,"ru",encodeURIComponent(document.location.href));}}}).endType();
function get_Buy_menu(){return{"items":[{"value":"<a href=\"http://shop.ebay.co.uk/allcategories/all-categories\">Browse Categories</a>"},{"value":"<a href=\"http://pages.ebay.co.uk/buy/tools.html\">Shopping Tools</a>"},{"value":"<a href=\"http://pages.ebay.co.uk/safetycentre/buyingwithconfidence.html\">Buy with confidence</a>"},{"value":"<a href=\"http://pages.ebay.co.uk/mobile/\">eBay Mobile</a>"},{"value":"<a href=\"http://partner.ebay.co.uk/index.html\">Partner Centre</a>"}]};}function get_MyEbay_menu(){return{"items":[{"value":"<a href=\"http://my.ebay.co.uk/ws/eBayISAPI.dll?MyEbay&amp;gbh=1&amp;CurrentPage=MyeBaySummary&amp;ssPageName=STRK:ME:LNLK\" rel=\"nofollow\">Summary</a>"},{"value":"<a href=\"http://my.ebay.co.uk/ws/eBayISAPI.dll?MyEbay&amp;gbh=1&amp;CurrentPage=MyeBayAllBuying\" rel=\"nofollow\">Buying</a>"},{"value":"<a href=\"http://my.ebay.co.uk/ws/eBayISAPI.dll?MyEbay&amp;gbh=1&amp;CurrentPage=MyeBayAllSelling&amp;ssPageName=STRK:ME:LNLK\" rel=\"nofollow\">Selling</a>"},{"value":"<a href=\"http://my.ebay.co.uk/ws/eBayISAPI.dll?MyEbay&amp;gbh=1&amp;CurrentPage=MyeBayWatching\" rel=\"nofollow\">Watching</a>"},{"value":"<a href=\"http://my.ebay.co.uk/ws/eBayISAPI.dll?MyEbay&amp;gbh=1&amp;CurrentPage=MyeBayMyAccounts\" rel=\"nofollow\">Account</a>"},{"value":"<a href=\"http://my.ebay.co.uk/ws/eBayISAPI.dll?MyEbay&amp;gbh=1&amp;ssPageName=STRK:ME:LNLK&amp;CurrentPage=MyeBayMyMessages\" rel=\"nofollow\">Messages</a>"},{"value":"<a href=\"http://my.ebay.co.uk/ws/eBayISAPI.dll?MyEbay&amp;gbh=1&amp;CurrentPage=MyeBayAllFavorites\" rel=\"nofollow\">Saved Searches</a>"}]};}function get_Sell_menu(){return{"items":[{"value":"<a href=\"http://cgi5.ebay.co.uk/ws/eBayISAPI.dll?SellHub3\">Sell an item</a>"},{"value":"<a href=\"http://pages.ebay.co.uk/sell/basics/start.html\">How to sell</a>"},{"value":"<a href=\"http://pages.ebay.co.uk/help/sell/fees.html\">Seller Fees</a>"},{"value":"<a href=\"http://pages.ebay.co.uk/safetycentre/sellingsafely.html\">Sell Safely</a>"},{"value":"<a href=\"http://pages.ebay.co.uk/community/charity/sellerinfo.html\">Sell for Charity</a>"},{"value":"<a href=\"http://pages.ebay.co.uk/businesscentre/index.html\">Business Centre</a>"},{"value":"<a href=\"http://pages.ebay.co.uk/services/buyandsell/powersellers.html\">Powerseller Hub</a>"}]};}function get_Community_menu(){return{"items":[{"value":"<a href=\"http://www2.ebay.com/aw/marketing-uk.shtml\">News</a>"},{"value":"<a href=\"http://pages.ebay.co.uk/community/chat/index.html\">Discussion Boards</a>"},{"value":"<a href=\"http://pages.ebay.co.uk/community/answercenter/index.html\">Answer Centre</a>"},{"value":"<a href=\"http://groups.ebay.co.uk/\">Groups</a>"},{"value":"<a href=\"http://myworld.ebay.co.uk/\">MyWorld</a>"}]};}function get_Help_menu(){return{"items":[{"value":"<a href=\"http://pages.ebay.co.uk/help/index.html\">Help Topics</a>"},{"value":"<a href=\"http://contact.ebay.co.uk/ws/eBayISAPI.dll?ShowCUPortal\">Contact us</a>"},{"value":"<a href=\"http://resolutioncentre.ebay.co.uk\">Resolution Centre</a>"},{"value":"<a href=\"http://pages.ebay.co.uk/safetycentre/index.html\">Safety Centre</a>"}]};}vjo.darwin.core.globalheader.utils.EventReg.registerMouseEvent({"MyEbay":["MyEbay","get_MyEbay_menu"],"Help":["Help","get_Help_menu"],"Sell":["Sell","get_Sell_menu"],"Buy":["Buy","get_Buy_menu"],"Community":["Community","get_Community_menu"],"ContactUs":["ContactUs","get_ContactUs_menu"]}, "gh-esmn", null, null, new Array('Help'), -1, [0,250], "HEADER_MENU");vjo.darwin.core.globalheader.utils.EventReg.registerAndCreateHeaderButtons(-1, [3], ".ebay.com", "gh-hsmn", new Array('getBrowseCatsDAPData'), new Array('BrowseCategories'), new Array('chevron0'), null, new Array('BrowseCategoriesMenu'), 500);(function () {
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
// en_GB/e685i/GH-RA_ReskinEbayNoAutoComplete_e685i12018678_1_en_GB
// b=12018678