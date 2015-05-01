
vjo.ctype("vjo.darwin.core.encoding.Enc").needs("vjo.dsf.Enc","").inits(function(){if(vjo.darwin.core.encoding.Enc.loaded){return;}
vjo.darwin.core.encoding.Enc.loaded=true;var vjoDecodeURI=window.decodeURI;window.decodeURI=function(str){try{return vjoDecodeURI(str);}catch(e){return unescape(str);}};var vjoDecodeURIComponent=window.decodeURIComponent;window.decodeURIComponent=function(str){try{return vjoDecodeURIComponent(str);}catch(e){return unescape(str);}};}).endType();

vjo.ctype("vjo.darwin.core.express.ExpressCrossLinking").needs(["vjo.dsf.typeextensions.string.TokenReplacement","vjo.dsf.typeextensions.string.Comparison","vjo.dsf.document.Element","vjo.dsf.cookie.VjCookieJar","vjo.dsf.utils.Object"]).protos({constructs:function(_1,_2,_3,_4,_5,_6,_7){this.sLayerId=_1;this.sCloseAnchorId=_2;this.sCartCountText=_3;this.aHostCoutryId=_4;this.aHostCountryDomain=_5;this.aHideOnParams=_6;this.sHtmlProvider=_7;this.sReferrer=document.referrer;},getHost:function(){var _8=vjo.dsf.cookie.VjCookieJar,df=this.sReferrer,sid=_8.readCookie("ebay","ecs")||"",rv="",i;if(sid=="-1"){return rv;}
if(sid){rv=this.getHostById(sid);}else{if((i=df.indexOf(".express."))!=-1){df=df.substring(i+9).toLowerCase();var f=df.indexOf("/");if(f>0){df=df.substring(0,f);}
for(i in this.aHostCountryDomain){if(this.compareHost(this.aHostCountryDomain[i],df)){rv=this.aHostCountryDomain[i];_8.writeCookielet("ebay","ecs",this.aHostCoutryId[i]);break;}}}}
return rv.toLowerCase();},compareHost:function(_a,_b){var _c=_a.toLowerCase(),sh2=_b.toLowerCase();if(_c.indexOf(".")==0){_c=_c.substring(1);}
if(sh2.indexOf(".")==0){sh2=sh2.substring(1);}
_c=_c.replace("/","");sh2=sh2.replace("/","");return(_c==sh2);},hideOnParams:function(){if(this.aHideOnParams){var _d=this.aHideOnParams;var _e=_d.length;for(var i=0;i<_e;i++){if(document.location.href.has(_d[i])){return true;}}}
return false;},init:function(){return;if(this.hideOnParams()){return;}
var oD=document,oCJ=vjo.dsf.cookie.VjCookieJar,sh="",l=this.sLayerId,E=vjo.dsf.document.Element;if(sh=this.getHost()){var oL=E.get(l),ct=oCJ.readCookie("dp1","exc")||"",sc="",lh=oD.location.host;ct=ct.split(".")[2];lh=lh.substring(lh.indexOf(".")+1);if(ct&&ct!="0"&&this.compareHost(lh,sh)){sc=this.sCartCountText.replaceTokensEx("##n##",ct);}
if(oL){oL.innerHTML=this.sHtmlProvider.replaceTokensEx("##n##",sh,sc);E.toggleHideShow(l,true);var oCL=E.get(this.sCloseAnchorId);if(oCL){oCL.onclick=vjo.dsf.utils.Object.hitch(this,"close");}}}},getHostById:function(_13){var ids=this.aHostCoutryId,i,ind=-1,o="";for(i=0;i<ids.length;i++){if(_13==ids[i]){ind=i;break;}}
if(ind!=-1){o=this.aHostCountryDomain[ind];}
return o;},close:function(){vjo.dsf.document.Element.toggleHideShow(this.sLayerId,false);vjo.dsf.cookie.VjCookieJar.writeCookielet("ebay","ecs","-1");return false;}}).endType();

vjo.ctype("vjo.darwin.core.ebaytoolbar.VjEbayToolbarDetect").needs("vjo.dsf.client.ActiveX").props({isEnabled:function(){var V1="eBayToolbar.Helper",V2="eBayToolbarCommLib.IWebEvent.1";with(this){var _2=vj$.ActiveX;return(_2.isLibLoaded(V1)||_2.isLibLoaded(V2));}}}).endType();

vjo.ctype("vjo.darwin.core.bta.BuyerTransactionAlert").needs(["vjo.dsf.cookie.VjCookieJar","vjo.dsf.typeextensions.string.Comparison","vjo.dsf.client.Browser","vjo.dsf.document.Element","vjo.dsf.utils.Object","vjo.darwin.core.ebaytoolbar.VjEbayToolbarDetect"]).protos({constructs:function(_1,_2,_3,_4,_5,_6,_7,_8,_9,_a,_b){this.sId=_1;this.iPollingInterval=_2;this.iMaxHits=_3;this.iHitTimeout=_4;this.iServerHits=0;this.sLastCookieletValue="";this.sServerUrl=_5;if(document.location.protocol.has("https")){_6=_6.replace("http://pics.","https://securepics.");}
this.sImgServer=_6+"/";this.sViewItemUrl=_7;this.aAlertInfo=[["h:h:alt:2",_9,"icon/iconOutbid_16x16.gif"],["h:h:alt:3",_8,"icon/iconWatchB_16x16.gif"],["h:h:alt:4",_9,"icon/iconOutbid_16x16.gif"],["h:h:alt:5",_a,"icon/iconchanceBlu_16x16.gif"],["h:h:alt:tcr",_b,"icon/iconMailBlue_16x16.gif"]];var c,oC=vjo.dsf.client.Browser,oCJ=vjo.dsf.cookie.VjCookieJar;if((oC.bNav&&oC.iVer<7)||(oC.bOpera&&(oC.iVer+oC.fMinorVer)<0.5)||(oC.bIE&&oC.iVer<5)){return;}
c=oCJ.readCookie("ebaysignin");if(!c||!c.is("in")){return;}
c=oCJ.readCookie("dp1","a1p");if(c&&c.length>0&&parseInt(c)>0){return;}
if(vjo.darwin.core.ebaytoolbar.VjEbayToolbarDetect.isEnabled()){return;}},setValue:function(_d,_e){var oL=this.oL;if(oL){if(_d.is("")&&!oL.ctrld){return;}
if(_e){if(vjo.dsf.client.Browser.bFirefox){oL.textContent=_d;}else{oL.innerText=_d;}}else{oL.innerHTML=_d;}
oL.ctrld=1;}},onRefresh:function(){var E=vjo.dsf.document.Element;if(!this.oL){this.oL=E.get(this.sId);}
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
var ii=c.substring(9,c.lastIndexOf("."));if(!c.is(this.sLastCookieletValue)){var _18=cfg[at-1],imgSrv=this.sImgServer;var _19=imgSrv+"s.gif";var _1a="<img src=\""+_19+"\" width=\"10\" height=\"16\" style=\"vertical-align:middle\">|<img src=\""+_19+"\" width=\"10\" height=\"16\" style=\"vertical-align:middle\">";_1a+="<img src=\""+imgSrv+_18[2]+"?t\" style=\"vertical-align:middle\"><img src=\""+_19+"\" width=\"5\" height=\"16\" style=\"vertical-align:middle\">";var url=this.sViewItemUrl+"&item="+ii;_1a+="<a href=\""+url+"&ssPageName="+_18[0]+"\">"+_18[1]+"</a>";this.setValue(_1a);this.sLastCookieletValue=c;}
this.fireRefreshEvent();},fireRefreshEvent:function(_1c){if(!_1c){_1c=this.iPollingInterval;}
window.setTimeout(vjo.dsf.utils.Object.hitch(this,this.onRefresh),_1c*1000);},onCookieExpire:function(){var oCJ=vjo.dsf.cookie.VjCookieJar,signin=oCJ.readCookie("ebaysignin");if(!signin.has("in")){return;}
if(document.location.href.has("https:")){return;}
if(this.iServerHits<this.iMaxHits){this.iServerHits++;var ct=new Date();ct=ct.getTime();this.setValue("<img height=\"1\" width=\"1\" src=\""+this.sServerUrl+"&clientTime="+ct+"\" style=\"visibility:hidden;vertical-align:middle\">");this.fireRefreshEvent(this.iHitTimeout);}else{this.setValue("");oCJ.writeCookielet("ebay","a2p","1111111101111111111.");}}}).endType();

vjo.ctype("vjo.dsf.utils.JsLoader").props({queue:[],pending:null,load:function(_1,_2,_3){var _4={url:_1,callback:_2,scope:_3},t=this,head,stag;if(t.pending){t.queue.push(_4);return;}
this.pending=_4;head=document.getElementsByTagName("head")[0];stag=document.createElement("script");stag.onload=stag.onreadystatechange=function(){if(!this.readyState||this.readyState=="loaded"||this.readyState=="complete"){t.oncomplete();stag.onload=stag.onreadystatechange=null;head.removeChild(stag);}};stag.type="text/javascript";stag.src=_1;head.appendChild(stag);},oncomplete:function(){var t=this,o=t.pending;if(o.callback){o.callback.call(o.scope||window);}
t.pending=null;if(t.queue.length>0){var _6=this.queue.shift();t.load(_6.url,_6.callback,_6.scope);}}}).endType();

vjo.ctype("vjo.darwin.core.dynamicmenu.Show").needs("vjo.dsf.utils.JsLoader").satisfies("vjo.dsf.common.IJsHandler").protos({constructs:function(_1){this.id=_1;},handle:function(_2){var _3=vjo.Registry.get(this.id),url=_3.getHandlerSource(),handler=_3.getHandler();if(url!==null&&typeof(window[handler])=="undefined"){this.vj$.JsLoader.load(url,_3.setup,_3);}else{_3.setup();}}}).endType();

vjo.ctype("vjo.darwin.core.dynamicmenu.Hide").satisfies("vjo.dsf.common.IJsHandler").protos({constructs:function(_1){this.id=_1;},handle:function(_2){vjo.Registry.get(this.id).hide();}}).endType();

vjo.ctype("vjo.darwin.core.dynamicmenu.Stay").satisfies("vjo.dsf.common.IJsHandler").protos({constructs:function(_1){this.id=_1;},handle:function(_2){vjo.Registry.get(this.id).clear();}}).endType();

vjo.ctype("vjo.darwin.core.dynamicmenu.DynamicMenu").needs(["vjo.dsf.document.Element","vjo.dsf.document.Positioning","vjo.dsf.document.Shim","vjo.dsf.utils.Object","vjo.dsf.client.Browser","vjo.darwin.core.dynamicmenu.Show","vjo.darwin.core.dynamicmenu.Hide","vjo.darwin.core.dynamicmenu.Stay"]).protos({constructs:function(_1,_2,_3,_4,_5,_6,_7,_8,_9,_a,_b,_c,_d,_e,_f,_10,_11,_12,_13,_14){this.iMouseOutTimer=null;this.iMouseOverTimer=null;this.iLeftPadding=_b;this.iTopPadding=_c;this.iRightPadding=_d;this.sAnchorId=_1;this.sContainerLayerId=_2;this.sContentLayerId=_3;this.iMouseOutDelay=_4;this.iMouseOverDelay=_5;this.iHighResDocWidth=_6;this.iLowResDocWidth=_7;this.iHighResColumns=_8;this.iMediumResColumns=_9;this.iLowResColumns=_a;this.sHandler=_e;this.sCollName=_f;this.sWidth=_10;this.sAnchorMouseOverClass=_11;this.sAnchorMouseOutClass=_12;this.iframeShim=null;this.sHTML=null;this.oAnchor=null;this.sTemplate=null;this.sAnchorText=_13;this.sDomain=_14;this.sHandlerSource=null;if(this.sAnchorId=="Help"){this.iLeftPadding-=5;}
this.oCobrand=null;},setHandlerSource:function(url){if(url){this.sHandlerSource=url;}},getHandlerSource:function(){return this.sHandlerSource;},getHandler:function(){return this.sHandler;},setup:function(){var oL,oA,iL,iT,iWW,noc,iAL,iAW,iLW,op,oSI,bF=false,t,lh=window.location.href;var vd=vjo.dsf,D=vd.document,E=D.Element,P=D.Positioning,S=D.Shim,B=vd.client.Browser;var O=vd.utils.Object;var _19=0;var _1a=false;with(this){clear();if(sTemplate==null){t=E.get(sContentLayerId);if(t){sTemplate=t.innerHTML;}}
oL=E.get(sContainerLayerId);if(oL){op=oL.offsetParent;}
oA=E.get(sAnchorId);iAL=DynamicMenuGetOffsetLeft(oA);iAW=oA.offsetWidth;iT=P.getOffsetTop(oA)+(oA.offsetHeight)+iTopPadding;iWW=P.getClientWidth();if(typeof(iWW)=="undefined"){iWW=800;}
if(iWW>iHighResDocWidth){noc=iHighResColumns;}else{if(iWW>iLowResDocWidth&&iWW<=iHighResDocWidth){noc=iMediumResColumns;}else{if(iWW<=iLowResDocWidth){noc=iLowResColumns;}else{noc=5;}}}
if(sHTML==null){bF=true;if(B.bFirefox&&!lh.has("motors.")&&!lh.hasAny("shop.","local.","catalog.")){E.promoteToBody(sContainerLayerId);}
sHTML=getMenuHtml(noc);if(sHTML==null){return;}
oL.innerHTML=sHTML;}
iLW=oL.offsetWidth;var _1b=GetContainerDiv(),cw,bRA=false;if(sAnchorId=="Help"){bRA=true;}
if(_1b&&B.bIE&&!lh.hasAny("community","education")){cw=_1b.offsetWidth;if((iAL+iLW)>cw){bRA=true;}}
var bE=(sAnchorId=="EbxBrowseCategories")?true:false;if(lh.hasAny("securitycenter","payments.")&&lh.has(".hk")){iAL=P.getOffsetLeft(oA);}else{if(lh.hasAny("securitycentre","contact_ebay")&&lh.has(".au")){iAL=P.getOffsetLeft(oA);}else{if((typeof(pageName)!="undefined")?pageName.has("CCHP_"):false){iAL=P.getOffsetLeft(oA);}else{if(lh.hasAll("pages.","ebaymotors")){}else{if(lh.hasAny("/buy/")){iAL=P.getOffsetLeft(oA);}else{if(lh.hasAny("pages.",".html",".shtml","cgi.","tools.ebay.de","neighborhoods.","themenwelten.","neighbourhoods.","motors.ebay.co.uk","motors.uk.","ebaymotors.at","motors.at.","motors.ebay.de","motors.de.")){iAL=P.getOffsetLeft(oA);}else{if(typeof(bCenterAlignedPage)!="undefined"&&bCenterAlignedPage){iAL=P.getOffsetLeft(oA);}else{if(bE&&((typeof(pageName)!="undefined")?pageName.has("KP_HomePage"):false)){iAL=P.getOffsetLeft(oA);}else{if(sAnchorId=="StoreBrowseCats"&&lh.has("stores.")){iAL=P.getOffsetLeft(oA);}else{if(typeof(pageName)!="undefined"&&pageName=="HomePagePortal"){iAL=P.getOffsetLeft(oA);}}}}}}}}}}
var _1d=E.get("2tabPopularProducts");if(_1d&&B.bIE){iAL=P.getOffsetLeft(oA);}
if((((iWW-iAL-iLW)<=10)&&iWW>iLW)||(iWW<(P.getOffsetLeft(oA)+iAW+iLW))||bRA||bE){_19=iAW-iLW;iL=iAL+iAW-iLW;}else{iL=iAL;}
if(sAnchorId=="Buy"||sAnchorId=="BrowseCategories"){_1a=true;}
if(sAnchorId=="Buy"||sAnchorId=="Sell"||sAnchorId=="MyEbay"||sAnchorId=="OV"){iL=iAL;}
iL+=iLeftPadding;if(lh.has("securitycentre")&&lh.has(".sg")&&B.bIE){iL+=10;}else{if(lh.has("feedback")&&lh.has(".hk")&&B.bFirefox){iL+=7;}else{if(lh.has("my.")&&B.bFirefox){iL-=7;}else{if(lh.has("myworld")&&B.bIE){iL+=10;}else{if(bE){var sn=E.get("dynamicmenu-snavW");iT=P.getOffsetTop(sn)+(sn.offsetHeight)+iTopPadding;if(B.bFirefox){iL+=2;}
if(B.bIE&&((typeof(pageName)!="undefined")?pageName.has("KP_HomePage"):false)){iL-=9;}}else{if(lh.has("search.")&&sAnchorId=="BrowseCategories"&&B.bIE&&B.iVer==7){iL+=22;iT-=2;}else{if(!lh.has("motors.shop.")&&B.bIE){if(lh.has("shop.")){if(sAnchorId=="BrowseCategories"){iL+=5;}}else{if(lh.hasAny("catalog.")){iL+=15;iT-=2;}}}else{if(lh.has("hub.motors.")&&B.bIE&&B.iVer>6){iL-=8;}}}}}}}}
if(bF){oL.style.left=(iL>=0)?(iL-1)+"px":"14px";oL.style.top=(iT)+"px";if(lh.has("contact")||(lh.has("resolutioncenter.")&&B.bIE&&B.iVer>6)){oL.style.position="absolute";var p1=P.getOffsetLeft(document.getElementById(sAnchorId));if(_1a){oL.style.left=(p1+5)+"px";}else{oL.style.left=(p1+_19)+"px";}
E.promoteToBody(oL.id);}
oL.style.zIndex="1000";if(lh.has("motors")||sAnchorId=="BrowseCategories"&&!lh.has("myworld")){iframeShim=this.add(oL,12);}else{iframeShim=this.add(oL);}}else{if(B.bIE){oL.appendChild(iframeShim);}}
oSI=E.get(sContainerLayerId+"-spacer");if(oSI&&bF){oSI.width=iLW-12;}
iMouseOverTimer=setTimeout(O.hitch(this,"show"),iMouseOverDelay);}},show:function(){var oA,E=vjo.dsf.document.Element;with(this){oA=E.get(sAnchorId);if(oA){oA.className=sAnchorMouseOverClass;}
E.toggleVisibility(sContainerLayerId,true);}},hide:function(){with(this){clearTimeout(iMouseOverTimer);iMouseOutTimer=setTimeout(vj$.Object.hitch(this,"close"),iMouseOutDelay);}},close:function(){var oA,E=vjo.dsf.document.Element;with(this){oA=E.get(sAnchorId);if(oA){oA.className=sAnchorMouseOutClass;}
E.toggleVisibility(sContainerLayerId,false);if(iframeShim){vj$.Shim.remove(E.get(sContainerLayerId),iframeShim);}}},clear:function(){clearTimeout(this.iMouseOutTimer);},sortByValue:function(_22,_23){if(_22.value.has("Everything")){return 1;}else{if(_23.value.has("Everything")){return-1;}else{return _22.value<_23.value?-1:(_22.value>_23.value?1:0);}}},getMenuHtml:function(_24){with(this){var i,j,c,n,h,ipc,html,img="",si,sTemp=".paradise.qa.ebay.com",sTemp2=".qa.ebay.com",u,ff,ll,sTemp3=".no-pool-name.qa.ebay.com";var E=vj$.Element;var lh=window.location.href;si="<img src='";if(document.location.protocol.has("https")){si+="https://secure";}else{si+="http://";}
si+="pics.ebaystatic.com/aw/pics/s.gif' height='1' width='1'  border='0' ";var _28=window[sHandler];if(!_28){return;}
var _29=_28(),items=_29[sCollName],l=items.length;if(sAnchorId=="BrowseCategories"&&lh.has("ebay.com/")){for(i=0;i<l;i++){if(items[i].value.has("eBay Motors")){items[i].value="Cars, Boats, Vehicles & Parts";items[i].url="http://www.motors.ebay.com";items.sort(sortByValue);break;}}}
ipc=Math.ceil(l/_24);h="<table bgcolor='white' border='0' cellpadding='0' cellspacing='0'";if(this.sWidth!=""){h+=" width='"+sWidth+"'";}
h+=">";for(i=0;i<ipc;i++){h+="<tr>";for(j=0;j<_24;j++){h+="<td nowrap>";if(i==0&&!E.get("EbxBrowseCategories-menu")){h+=si+"><br/>";img=si+" id='"+sContainerLayerId+"-spacer'>";}
n=i+(j)*ipc;c=items[n];if(c){if(typeof(c.url)!="undefined"){u=c.url;if(sDomain){if(c.url.indexOf(sTemp)!=-1){u=c.url.replace(sTemp,sDomain);}else{if(c.url.indexOf(sTemp3)!=-1){u=c.url.replace(sTemp3,sDomain);}else{if(c.url.indexOf(sTemp2)!=-1){u=c.url.replace(sTemp2,sDomain);}}}}
u=cobrandUrl(u);h+="<a href='"+u+"'>";h+=c.value;h+="</a>";}else{u=c.value;ff=u.indexOf("href=\"");if(ff==-1){h+=u;}else{ff+=6;ll=u.lastIndexOf("\"");u=u.substr(ff,ll-ff);h+=c.value.substr(0,ff)+cobrandUrl(u)+c.value.substr(ll);}}}else{h+="&nbsp;";}
h+="</td>";}
h+="</tr>";}
h+="</table>";html=h;if(sTemplate){html=sTemplate.replace("##1##",h).replace("##2##",img);}
return html;}},DynamicMenuGetOffsetLeft:function(e){var l=0,oCl=vjo.dsf.client.Browser,ex=(oCl.bIE||oCl.bSafari),lh=window.location.href;var bH=(typeof(isHomepage)!="undefined")?isHomepage:false;var _2d=false;if(e.offsetParent){l=e.offsetLeft;while(e=e.offsetParent){if(!_2d){_2d=e.innerHTML.has("snav");}
if(!e.id.toLowerCase().has("maincontent")&&!e.className.has("pagecontainer")){l+=e.offsetLeft;}
if(((e.className.is("pnav")||_2d)&&ex)&&e.offsetLeft!=0&&(!lh.hasAny("signin.","shop.","community","sitemap","/help/","/education/","blogs","feedback","my"))){return l;}else{if((e.className.is("pnav")||_2d)&&e.offsetLeft!=0&&oCl.bFirefox&&lh.has(".hk")&&(!lh.has("community")&&!bH)){return l;}else{if((e.className.is("pnav")||_2d)&&e.offsetLeft!=0&&oCl.bIE&&lh.has(".au")&&lh.hasAny("/help/")){return l;}}}}}
return l;},GetContainerDiv:function(){var d=document,aa,i,l;if(d.getElementsByTagName){aa=d.getElementsByTagName("div");}else{if(d.all){aa=d.all;}}
l=aa.length;for(i=0;i<l;i++){if(aa[i].className=="gbhdr"){return aa[i];}}},cobrandUrl:function(_2f){var u="undefined",cc,cf;if(this.oCobrand==null&&typeof(ebay)!=u&&typeof(ebay.oDocument)!=u){cc=ebay.oDocument._getControl("cobrandCollection");if(cc){cf=cc._getControl("cobrandFunctions");this.oCobrand=cf;}}else{cf=this.oCobrand;}
if(cf&&typeof(cf.cobrandURL)!=u){var lc=(_2f.substring(_2f.length)!="/")?"/":"";return cf.cobrandURL(_2f+lc);}
return _2f;},add:function(_32,_33,_34){var f,p="px",w,h,s,S=vjo.dsf.document.Shim;if(vjo.dsf.client.Browser.bIE){w=_32.offsetWidth;h=_32.offsetHeight;w+=_33?_33:0;h+=_34?_34:0;f=document.createElement("IFRAME");s=f.style;s.width=w+p;s.height=h+p;s.filter="chroma(color='white')";f.frameBorder=0;s.position="absolute";s.left="0"+p;s.top="0"+p;s.zIndex="-1";s.filter="Alpha(Opacity=\"0\")";if(document.location.protocol.has("https")){f.src="https://securepics.ebaystatic.com/aw/pics/s.gif";}
_32.appendChild(f);return f;}},replaceJSONDataHandler:function(_36,_37){if(_36!=null){this.sHandler=_36;}
if(_37!=null){this.sDomain=_37;}}}).endType();

vjo.ctype("vjo.darwin.core.ebayheader.playground.Playground").needs(["vjo.dsf.typeextensions.string.Comparison","vjo.dsf.cookie.VjCookieJar","vjo.dsf.document.Element"]).protos({constructs:function(_1,_2){this.sLayerId=_1;this.sHandle=_2;},show:function(){var _3=vjo.dsf.cookie.VjCookieJar,sbf=_3.readCookie("ebay","sbf"),pcon=_3.getBitFlag(sbf,24),l,h;var _4=this.sLayerId,E=vjo.dsf.document.Element;if(pcon){l=E.get(_4);h=window[this.sHandle];if(h&&l){l.innerHTML=h();E.toggleHideShow(_4,true);}}else{E.toggleHideShow(_4,false);}}}).endType();

vjo.ctype("vjo.darwin.core.ebayheader.timezone.TimeZone").needs("vjo.dsf.cookie.VjCookieJar").props({init:function(){var o=new Date(),oCJ,os;os=o.getTimezoneOffset();oCJ=vjo.dsf.cookie.VjCookieJar;oCJ.writeCookielet("dp1","tzo",os.dec2Hex());}}).inits(function(){vjo.darwin.core.ebayheader.timezone.TimeZone.init();}).endType();


function get_Buy_menu(){return{"items":[{"value":"<a href=\"http://listings.ebay.com/\">Browse Categories</a>"},{"value":"<a href=\"http://www.ebayuniversity.com/gotraining/First_Time_Bidding_And_Buying\">Help with bidding &amp; buying</a>"},{"value":"<a href=\"http://pages.ebay.com/buy/tools.html\">Buyer Tools</a>"},{"value":"<a href=\"http://reviews.ebay.com/\">Reviews &amp; Guides</a>"},{"value":"<a href=\"http://www.ebay.com/mobile/\">eBay Mobile</a>"}]};}
function get_Sell_menu(){return{"items":[{"value":"<a href=\"http://sell.ebay.com/sell\">Sell an item</a>"},{"value":"<a href=\"http://pages.ebay.com/sell/top10tips.html\">Selling Tips</a>"},{"value":"<a href=\"http://pages.ebay.com/sell/whatshot/\">What's Hot</a>"},{"value":"<a href=\"http://pages.ebay.com/sell/tools.html\">Seller tools &amp; eBay Stores</a>"},{"value":"<a href=\"http://pages.ebay.com/services/buyandsell/shipping.html\">Shipping center</a>"}]};}
function get_MyEbay_menu(){return{"items":[{"value":"<a href=\"http://my.ebay.com/ws/eBayISAPI.dll?MyEbay&amp;gbh=1&amp;CurrentPage=MyeBaySummary&amp;ssPageName=STRK:ME:LNLK\" rel=\"nofollow\">Summary</a>"},{"value":"<a href=\"http://my.ebay.com/ws/eBayISAPI.dll?MyEbay&amp;gbh=1&amp;CurrentPage=MyeBayWatching\" rel=\"nofollow\">Watching</a>"},{"value":"<a href=\"http://my.ebay.com/ws/eBayISAPI.dll?MyEbay&amp;gbh=1&amp;CurrentPage=MyeBayBidding\" rel=\"nofollow\">Bidding</a>"},{"value":"<a href=\"http://my.ebay.com/ws/eBayISAPI.dll?MyEbay&amp;gbh=1&amp;CurrentPage=MyeBayWon\" rel=\"nofollow\">Won</a>"},{"value":"<a href=\"http://my.ebay.com/ws/eBayISAPI.dll?MyEbay&amp;gbh=1&amp;CurrentPage=MyeBayAllSelling&amp;ssPageName=STRK:ME:LNLK\" rel=\"nofollow\">Selling</a>"},{"value":"<a href=\"http://my.ebay.com/ws/eBayISAPI.dll?MyEbay&amp;gbh=1&amp;CurrentPage=MyeBayAllFavorites\" rel=\"nofollow\">Saved Searches</a>"},{"value":"<a href=\"http://my.ebay.com/ws/eBayISAPI.dll?MyEbay&amp;gbh=1&amp;ssPageName=STRK:ME:LNLK&amp;CurrentPage=MyeBayMyMessages\" rel=\"nofollow\">Messages</a>"}]};}
function get_Community_menu(){return{"items":[{"value":"<a href=\"http://www2.ebay.com/aw/marketing.shtml\">News</a>"},{"value":"<a href=\"http://pages.ebay.com/community/answercenter/index.html\">Answer Center</a>"},{"value":"<a href=\"http://pages.ebay.com/community/boards/index.html\">Workshops/Discussion Forums</a>"},{"value":"<a href=\"http://neighborhoods.ebay.com\">Neighborhoods</a>"},{"value":"<a href=\"http://groups.ebay.com/index.jspa?categoryID=1&amp;redirected=1\">Groups</a>"}]};}
function get_Help_menu(){return{"items":[{"value":"<a href=\"http://pages.ebay.com/help/index.html\">Help Topics</a>"},{"value":"<a href=\"http://pages.ebay.com/education/index.html\">Learning Center</a>"},{"value":"<a href=\"http://resolutioncenter.ebay.com\">Resolution Center</a>"},{"value":"<a href=\"http://pages.ebay.com/university/index.html\">eBay University</a>"},{"value":"<a href=\"http://pages.ebay.com/help/contact_us/_base/index.html\">Contact Us</a>"}]};}


vjo.ctype("vjo.dsf.document.Select").needs("vjo.dsf.document.Element").props({E:vjo.dsf.document.Element,addOption:function(_1,_2,_3){var t=this,e=t.get(_1),o,os;if(e){o=t.createOption(_3,_2);os=e.options;os[os.length]=o;}},createOption:function(_5,_6){return this.createOption(_5,_6,false,false);},createOption:function(_7,_8,_9,_a){return new Option(_7,_8,_9,_a);},clear:function(_b){var e=this.get(_b),os,i,l;if(e){os=e.options;l=os.length;for(i=l;i>=0;i--){os[i]=null;}}},get:function(_d){var e=_d;if(typeof(_d)=="String"){e=this.E.get(_d);}
return e;}}).endType();

vjo.ctype("vjo.darwin.core.dynamicdropdown.DynamicDropdown").needs(["vjo.dsf.document.Element","vjo.dsf.document.Select"]).protos({constructs:function(_1,_2,_3,_4){this.sDropdownId=_1;this.sHandler=_2;this.sCollName=_3;this.iCondNo=_4;},fill:function(){var e,h,d,col,i,l,c,D=vjo.dsf.document,E=D.Element,S=D.Select;with(this){e=E.get(sDropdownId);if(e&&typeof(e.length)!="undefined"&&e.length>0){e=e[0];}
if(e&&e.options){if(iCondNo==-1||e.options.length<=iCondNo){h=window[sHandler];if(!h){return;}
d=h(),col=d[sCollName],l=col.length;for(i=0;i<l;i++){c=col[i];S.addOption(e,c.id,c.value);}}}}}}).endType();


vjo.ctype("vjo.darwin.core.greetings.Sandbox").needs(["vjo.dsf.document.Element","vjo.dsf.EventDispatcher"]).props({init:function(){var l=vjo.dsf.document.Element.get("registerLink"),s="https://scgi.sandbox.ebay.com/ws/eBayISAPI.dll?RegisterEnterInfo",t="https://developer.ebay.com/DevZone/sandboxuser/Default.aspx";if(l){if(l.href==s){l.href=t;}}},adBanner:function(){var l=window.location.href,s="sandbox.",c;if(l.indexOf(s)!=-1){c="<sty"+"le type=\"text/css\">#gnheader {background: url(http://pics.ebaystatic.com/aw/pics/devprogram/watermark.gif)}</sty"+"le>";document.write(c);}}}).inits(function(){vjo.darwin.core.greetings.Sandbox.adBanner();vjo.dsf.EventDispatcher.addEventListener(window,"load",function(){vjo.darwin.core.greetings.Sandbox.init();});}).endType();

vjo.ctype("vjo.darwin.core.greetings.VjGreetingsServer").needs(["vjo.dsf.utils.URL","vjo.darwin.core.greetings.Sandbox"]).props({handleClick:function(_1,_2){var _3=_2.srcElement||_2.target;if(_3){_3.href=vjo.dsf.utils.URL.addArg(_1,"ru",encodeURIComponent(document.location.href));}}}).endType();

if(typeof(_oGlobalNavRTMInfo)=="undefined"){_oGlobalNavRTMInfo={};_oGlobalNavRTMInfo.aRTMPlacementData=[];}

vjo.ctype("vjo.dsf.typeextensions.string.Decode").endType();String.prototype.decodeBase64=function(){var rv=this,len=rv.length,ret="",i=0;if(len===0){return ret;}
var _2,chr2,chr3="";var _3,enc2,enc3,enc4="";var _4="ABCDEFGHIJKLMNOPQRSTUVWXYZ"+"abcdefghijklmnopqrstuvwxyz"+"0123456789+/=*";var _5=new RegExp("[^A-Za-z0-9+/=*]");if(_5.exec(rv)){return;}
do{_3=_4.indexOf(rv.charAt(i++));enc2=_4.indexOf(rv.charAt(i++));enc3=_4.indexOf(rv.charAt(i++));enc4=_4.indexOf(rv.charAt(i++));_2=(_3<<2)|(enc2>>4);chr2=((enc2&15)<<4)|(enc3>>2);chr3=((enc3&3)<<6)|enc4;ret+=String.fromCharCode(_2);if(!(enc3>=64)){ret+=String.fromCharCode(chr2);}
if(!(enc4>=64)){ret+=String.fromCharCode(chr3);}
_2=chr2=chr3=_3=enc2=enc3=enc4="";}while(i<len);return ret;};String.prototype.decodeUTF8=function(){var s=this,len=s.length;var rs="";var i=0;var c=0,c1=0,c2=0;while(i<len){c=s.charCodeAt(i);if(c<128){rs+=String.fromCharCode(c);i++;}else{if((c>191)&&(c<224)){c2=s.charCodeAt(i+1);rs+=String.fromCharCode(((c&31)<<6)|(c2&63));i+=2;}else{c2=s.charCodeAt(i+1);c3=s.charCodeAt(i+2);rs+=String.fromCharCode(((c&15)<<12)|((c2&63)<<6)|(c3&63));i+=3;}}}
return rs;};

vjo.ctype("vjo.darwin.core.ebayheader.rtm.GlobalHeaderRtmCall").needs(["vjo.dsf.typeextensions.string.Decode","vjo.dsf.cookie.VjCookieJar","vjo.dsf.utils.Object"]).props({iTimer:null,submitRTMCall:function(_1){var un="undefined";var lh=window.location.href;if(!lh.hasAny("catalog.")&&(lh.hasAny("offer.")||!(typeof(ebay)!=un&&typeof(ebay.oDocument)!=un&&ebay.oDocument._getControl("rtm")))&&!(typeof(vjo)!=un&&typeof(vjo.dsf)!=un&&typeof(vjo.dsf.ServiceEngine)!=un&&typeof(vjo.dsf.ServiceEngine.inProcHdl)!=un&&typeof(vjo.dsf.ServiceEngine.inProcHdl.svcHdls)!=un&&typeof(vjo.dsf.ServiceEngine.inProcHdl.svcHdls.RTM_CALLBACK_SERVICE)!=un)&&typeof(_oGlobalNavRTMInfo)!==un&&!(typeof(rtm)!=un)){if(_oGlobalNavRTMInfo.aRTMPlacementData.length>0){this.iTimer=window.setInterval(vjo.dsf.utils.Object.hitch(this,"init"),1);}}else{if(_oGlobalNavRTMInfo.aRTMPlacementData.length>0){if(lh.hasAny("shop.","icatalog.","catalog.")&&!lh.hasAny("hub.")){this.init();}}}},getUid:function(){var _4=vjo.dsf.cookie.VjCookieJar,u1p=_4.readCookie("dp1","u1p"),u1pDecoded;if(u1p){u1pDecoded=u1p.decodeBase64().decodeUTF8();}
return u1pDecoded;},getGuid:function(){var _5=vjo.dsf.cookie.VjCookieJar,guid=_5.readCookie("ebay","sgj");return guid;},hasUid:function(_6){if(_6){return true;}
if(this.getUid().has("@@__@@__@@")){return false;}
return true;},init:function(){if(typeof(vjo.darwin.core.rtm)=="undefined"){return;}
if(this.iTimer!=null){window.clearInterval(this.iTimer);}
var _7=_oGlobalNavRTMInfo.aRTMPlacementData,data;var _8=[],htmlIds=[],heights=[],pids=[],dblclkUrls=[],defaultUrls=[],url;for(i=0;i<_7.length;i++){data=_7[i];_8[i]=data.maxWidth;htmlIds[i]=data.htmlId;heights[i]=data.maxHeight;pids[i]=data.pid;dblclkUrls[i]=data.dblclkUrl;defaultUrls[i]=data.defaultUrl;url=data.rtmUrl+"?RtmCmd&a=json"+(this.hasUid(data.userId)?("&l="+(data.userId?data.userId:this.getUid())):"")+"&g="+(data.gUid?data.gUid:this.getGuid())+"&ord="+data.ord+((data.oid)?"&i="+data.oid:"");}
url+="&p="+pids.join(":");_oGlobalNavRTMInfo.sRTMUrl=url;if(_7.length>0){var _9=new vjo.darwin.core.rtm.RTMInit({"url":url,"widths":_8,"htmlIds":htmlIds,"heights":heights,"pids":pids,"dblclkUrls":dblclkUrls,"defaultUrls":defaultUrls});_9.invoke({});}}}).endType();
(function () {
var _r = vjo.Registry;
_r.put('FooterTrackingCompSpecGenerator_0',new vjo.darwin.tracking.enabler.TrackingModuleEnabler("_trksid", "_trkparms", "m40;", ";")); })();
function FooterTrk(){ return {handle:function (event){ (function(){
var _d=vjo.dsf.EventDispatcher;
var _r=vjo.Registry;
_d.add('glbfooter','click',function(event) { this.handle(event); },_r._FooterTrackingCompSpecGenerator_0);})();  }};};
vjo.dsf.EventDispatcher.add('body','load', new FooterTrk());
(function () {
var _r = vjo.Registry;
function $o0(p0,p1,p10,p13){return new vjo.darwin.core.dynamicmenu.DynamicMenu(p0,p1,"dynMenuCtr",75,250,900,800,1,1,1,p10,1,0,p13,"items","","hovered","","","");};_r.put('BuyMenu',$o0("Buy","Buy-menu",5,"get_Buy_menu")); _r.put('SellMenu',$o0("Sell","Sell-menu",1,"get_Sell_menu")); _r.put('MyEbayMenu',$o0("MyEbay","MyEbay-menu",1,"get_MyEbay_menu")); _r.put('CommunityMenu',$o0("Community","Community-menu",1,"get_Community_menu")); _r.put('HelpMenu',$o0("Help","Help-menu",1,"get_Help_menu")); })();
function NavMenuBind(){ return {handle:function (event){ (function(){
var _d=vjo.dsf.EventDispatcher;
var _r=vjo.Registry;
_r.put('CorePrimaryNavResourceSpec_1', new vjo.darwin.core.dynamicmenu.Hide("BuyMenu")); _r.put('CorePrimaryNavResourceSpec_2', new vjo.darwin.core.dynamicmenu.Stay("BuyMenu")); _r.put('CorePrimaryNavResourceSpec_3', new vjo.darwin.core.dynamicmenu.Show("BuyMenu")); _r.put('CorePrimaryNavResourceSpec_4', new vjo.darwin.core.dynamicmenu.Hide("BuyMenu")); _r.put('CorePrimaryNavResourceSpec_6', new vjo.darwin.core.dynamicmenu.Hide("SellMenu")); _r.put('CorePrimaryNavResourceSpec_7', new vjo.darwin.core.dynamicmenu.Stay("SellMenu")); _r.put('CorePrimaryNavResourceSpec_8', new vjo.darwin.core.dynamicmenu.Show("SellMenu")); _r.put('CorePrimaryNavResourceSpec_9', new vjo.darwin.core.dynamicmenu.Hide("SellMenu")); _r.put('CorePrimaryNavResourceSpec_11', new vjo.darwin.core.dynamicmenu.Hide("MyEbayMenu")); _r.put('CorePrimaryNavResourceSpec_12', new vjo.darwin.core.dynamicmenu.Stay("MyEbayMenu")); _r.put('CorePrimaryNavResourceSpec_13', new vjo.darwin.core.dynamicmenu.Show("MyEbayMenu")); _r.put('CorePrimaryNavResourceSpec_14', new vjo.darwin.core.dynamicmenu.Hide("MyEbayMenu")); _r.put('CorePrimaryNavResourceSpec_16', new vjo.darwin.core.dynamicmenu.Hide("CommunityMenu")); _r.put('CorePrimaryNavResourceSpec_17', new vjo.darwin.core.dynamicmenu.Stay("CommunityMenu")); _r.put('CorePrimaryNavResourceSpec_18', new vjo.darwin.core.dynamicmenu.Show("CommunityMenu")); _r.put('CorePrimaryNavResourceSpec_19', new vjo.darwin.core.dynamicmenu.Hide("CommunityMenu")); _r.put('CorePrimaryNavResourceSpec_21', new vjo.darwin.core.dynamicmenu.Hide("HelpMenu")); _r.put('CorePrimaryNavResourceSpec_22', new vjo.darwin.core.dynamicmenu.Stay("HelpMenu")); _r.put('CorePrimaryNavResourceSpec_23', new vjo.darwin.core.dynamicmenu.Show("HelpMenu")); _r.put('CorePrimaryNavResourceSpec_24', new vjo.darwin.core.dynamicmenu.Hide("HelpMenu")); _d.add('Buy-menu','mouseout',_r.get('CorePrimaryNavResourceSpec_1'));_d.add('Buy-menu','mouseover',_r.get('CorePrimaryNavResourceSpec_2'));_d.add('Buy','mouseover',_r.get('CorePrimaryNavResourceSpec_3'));_d.add('Buy','mouseout',_r.get('CorePrimaryNavResourceSpec_4'));_d.add('Sell-menu','mouseout',_r.get('CorePrimaryNavResourceSpec_6'));_d.add('Sell-menu','mouseover',_r.get('CorePrimaryNavResourceSpec_7'));_d.add('Sell','mouseover',_r.get('CorePrimaryNavResourceSpec_8'));_d.add('Sell','mouseout',_r.get('CorePrimaryNavResourceSpec_9'));_d.add('MyEbay-menu','mouseout',_r.get('CorePrimaryNavResourceSpec_11'));_d.add('MyEbay-menu','mouseover',_r.get('CorePrimaryNavResourceSpec_12'));
_d.add('MyEbay','mouseover',_r.get('CorePrimaryNavResourceSpec_13'));_d.add('MyEbay','mouseout',_r.get('CorePrimaryNavResourceSpec_14'));_d.add('Community-menu','mouseout',_r.get('CorePrimaryNavResourceSpec_16'));_d.add('Community-menu','mouseover',_r.get('CorePrimaryNavResourceSpec_17'));_d.add('Community','mouseover',_r.get('CorePrimaryNavResourceSpec_18'));_d.add('Community','mouseout',_r.get('CorePrimaryNavResourceSpec_19'));_d.add('Help-menu','mouseout',_r.get('CorePrimaryNavResourceSpec_21'));_d.add('Help-menu','mouseover',_r.get('CorePrimaryNavResourceSpec_22'));_d.add('Help','mouseover',_r.get('CorePrimaryNavResourceSpec_23'));_d.add('Help','mouseout',_r.get('CorePrimaryNavResourceSpec_24'));
})();  }};};
vjo.dsf.EventDispatcher.add('body','load', new NavMenuBind());
(function () {
var _r = vjo.Registry;
_r.put('BrowseCategoriesMenu',new vjo.darwin.core.dynamicmenu.DynamicMenu("BrowseCategories", "BrowseCategories-menu", "dynMenuCtr", 75, 250, 900, 800, 3, 3, 3, -10, 7, 0, "getBrowseCategoriesData", "items", "100%", "hovered", "", "", ".ebay.com")); })();
function SecNavMenuBind(){ return {handle:function (event){ (function(){
var _d=vjo.dsf.EventDispatcher;
var _r=vjo.Registry;
_r.put('CoreSecondaryNavResourceSpec_1', new vjo.darwin.core.dynamicmenu.Hide("BrowseCategoriesMenu")); _r.put('CoreSecondaryNavResourceSpec_2', new vjo.darwin.core.dynamicmenu.Stay("BrowseCategoriesMenu")); _r.put('CoreSecondaryNavResourceSpec_3', new vjo.darwin.core.dynamicmenu.Show("BrowseCategoriesMenu")); _r.put('CoreSecondaryNavResourceSpec_4', new vjo.darwin.core.dynamicmenu.Hide("BrowseCategoriesMenu")); _d.add('BrowseCategories-menu','mouseout',_r.get('CoreSecondaryNavResourceSpec_1'));_d.add('BrowseCategories-menu','mouseover',_r.get('CoreSecondaryNavResourceSpec_2'));_d.add('BrowseCategories','mouseover',_r.get('CoreSecondaryNavResourceSpec_3'));_d.add('BrowseCategories','mouseout',_r.get('CoreSecondaryNavResourceSpec_4'));})();  }};};
vjo.dsf.EventDispatcher.add('body','load', new SecNavMenuBind());
(function () {
var _r = vjo.Registry;
_r.put('category0',new vjo.darwin.core.dynamicdropdown.DynamicDropdown("category0", "getBrowseCategoriesData", "items", 1)); })();
(function(){(function(){
var _d=vjo.dsf.EventDispatcher;
var _r=vjo.Registry;
_d.add('body','load',function(event) { this.fill(); },_r._category0);})();})();
(function () {
var _r = vjo.Registry;
function $o0(){return new vjo.darwin.tracking.enabler.TrackingModuleEnabler("_trksid","_trkparms","m37;",";");};_r.put('HeaderTrackingCompSpecGenerator_0',$o0()); _r.put('HeaderTrackingCompSpecGenerator_1',$o0()); })();
function HeaderTrk(){ return {handle:function (event){ (function(){
var _d=vjo.dsf.EventDispatcher;
var _r=vjo.Registry;
function $0(){return function(event){return this.handle(event);};};_d.add('BrowseCategories-menu','click',$0(),_r._HeaderTrackingCompSpecGenerator_0);_d.add('gnheader','click',$0(),_r._HeaderTrackingCompSpecGenerator_1);_d.add('body','click',function(event){ vjo.darwin.tracking.enabler.TrackingEnabler.copySIDToCookie(event, "_trksid", "_sp", "_trkparms");  });})();  }};};
vjo.dsf.EventDispatcher.add('body','load', new HeaderTrk());

// en_US/e611/GlobalNav14_Ebay_e6118261870_1_en_US
// b=8261870