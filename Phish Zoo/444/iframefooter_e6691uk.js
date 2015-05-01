//<!--
//1@@m1

var g_isReserved="page_package_reservation";

//2@@m2

function initCookieJar()
{var u="undefined";if(typeof(ebay)!=u&&typeof(getParent)!=u)
{var p=getParent();if(p.ebay)
{var cj=p.ebay.oDocument.oCookieJar;ebay.oDocument.oCookieJar=cj;ebay.oDocument._addControl('cookieJar',cj);}}}
initCookieJar();

//3@@m9

function EbayCobrand(pParent,pName,pHT,pCBName,pCoPartnerID)
{if(!this.objType)
this.objType="EbayCobrand";this.base=EbayBaseControl;this.base(pParent,pName);this.oEnv=this.oGlobals.oEnvironment;this.iHT=pHT;this.sCBName=this.sHeaderName=pCBName;this.sCo_partnerid=this.sS_partnerid=pCoPartnerID||this.iHT;this.sFolder=null;this.aPathIdentifiers=new Array("."+this.sCBName+".");this.aQueryIdentifiers=new Array("ht="+this.iHT,"co_partnerid="+this.sCo_partnerid,"s_partnerid="+this.sS_partnerid);this.urlDetect=ebCobrandUrlDetect;this.cookieDetect=ebCobrandCookieDetect;this.writeHeader=this.writeFooter=true;}
function ebCobrandUrlDetect(pPath,pQuery,pIsIframe)
{var rv=false;with(this)
{var oC=this.oGlobals.oClient;if((oC.bFirefox||oC.bNav||typeof(window.parent.document)=="unknown")&&pIsIframe)
this.parent.parent.parent.downgradeDomain();var p=pPath||(pIsIframe?window.parent.ebay.oGlobals.oEnvironment.sThisPagePath:oEnv.sThisPagePath);rv=p.hasAny(aPathIdentifiers);if(rv)
rv=_exec("detectException");else
{var q=(pQuery==null)?oEnv.sThisPageQuery:pQuery;q=q.substr(1).split("&");for(var i=0;i<q.length&&!rv;i++)
rv=q[i].isAny(aQueryIdentifiers);}}
return rv?this:null;}
function ebCobrandCookieDetect()
{return this.oDocument.getCookie("ebay","ht").is(this.iHT)?this:null;}
function EbayPool(pParent,pName)
{if(!this.objType)
this.objType="EbayPool";this.base=EbayBaseControl;this.base(pParent,pName);this.sPool=pName;this.isSupportedPool=ebPoolSupported;}
function ebPoolSupported(pPool)
{return this.sPool.is(pPool);}

//4@@m8

function EbayCobrandCountry(pParent,pName)
{if(!this.objType)
this.objType="EbayCobrandCountry";this.base=EbayBaseControl;this.base(pParent,pName);this.init=function()
{new EbayCobrand(this,"BT","47","bt");o=new EbayCobrand(this,"Sandbox","99","sandbox");o.aPathIdentifiers=new Array("sandbox.");}
this.init();}
function EbayCobrandFunctionsCountry(pParent,pName)
{if(!this.objType)
this.objType="EbayCobrandFunctionsCountry";this.base=EbayBaseControl;this.base(pParent,pName);this.createFunctionExceptions=function()
{var p=this.parent,so=p.oSelCobrand,n=so.name;if(n.is("Sandbox"))
{this.parent.isSupportedPool=function(){return true;}
ebay._getControl("cobrandFunctions").isEbayLink=function(pLink)
{with(this)
{if(pLink.has(sQA+oGlobals.oEnvironment.sCountryDomain))
return true;}
return false;}
so.linksConditionsException=function(pLink)
{return!pLink.hasAny('developer.ebay.com','dev-forums.ebay.com','affiliates');}}}
this.init=function()
{this.createFunctionExceptions();}
this.init();}

//5@@m30

function EbayCobrandCollection(pParent,pName,pIsIframe)
{if(!this.objType)
this.objType="EbayCobrandCollection";this.base=EbayBaseControl;this.base(pParent,pName);var g=this.oGlobals;var env=this.oEnv=g.oEnvironment;this.aMicroBrands=[["100","techrepublic"],["101","lego"]];this.aMBPages=["viewitem","acceptbid","autosviewitemforvehicle","binconfirm","bincongrats","makebid"];this.sMBCookieName="mb";this.oSelCobrand=this.sMBFile=null;this.sQA=env.sThisPagePath.has(env.sQADomain)?".qa":"";this.sCBCountryDir=env.sCountryIncludeDir+"features/cobrand/";this.bIsIframe=pIsIframe;this.bIsSandbox=pParent.doc.location.host.has("sandbox.");with(this)
_registerListener(oDocument._getEvent("load"),EVENT_AFTER,"onAfterLoad");this.createStandardPoolObjs=function()
{with(this)
{var pools=["www","pages","portals","payments","offer","promo","k2b-bulk","syicatalogs","previewitem","spchk","my","sandbox","bcl","portals","sell","myworld"];for(var i=pools.length-1;i>-1;i--)
new EbayPool(this,pools[i]);}}
this.isSupportedPool=function(pPool)
{var cs=this.controls;if(cs)
{for(var i in cs)
{if((cs[i].objType=="EbayPool")&&cs[i].isSupportedPool(pPool))
return true;}}
return false;}
this.checkMicroBrand=function()
{with(this)
{var mba=aMicroBrands,len=mba.length;var lp=oEnv.sLastPagePath,rv=false;var c=oDocument.getCookie(sMBCookieName);for(var i=0;i<len&&!rv;i++)
rv=(lp.has(mba[i][1])||c.is(mba[i][0]));if(rv)
{var pn=((typeof(pageName)!="undefined")?pageName:"").toLowerCase();var mbpa=aMBPages,mpbLen=mbpa.length;for(i=0;i<mpbLen&&!sMBFile;i++)
{if(pn.indexOf(mbpa[i])==0)
{for(var j=0;j<len&&!sMBFile;j++)
{sMBFile=lp.has(mba[j][1])?mba[j][1]+".js":null;}}}}
return sMBFile;}}
function writeMicroBrandHeader()
{with(this)
oDocument.write(oUtils.scriptTag(sCBCountryDir+"mb/"+sMicroBrandFile));}
this.setCookie=function(pVal)
{if(!this.bIsIframe)
this.oDocument.oCookieJar.writeCookielet("ebay","ht",pVal,"","/");}
this.detectCobrand=function(pCobrands,pUrl)
{var cbs=pCobrands.controls,sc;if(cbs)
{for(var i in cbs)
{sc=pUrl?cbs[i].urlDetect(null,null,this.bIsIframe):cbs[i].cookieDetect();if(sc)
break;}}
return sc;}
this.init=function()
{var c=new EbayCobrandCountry(this,"countryCobrands");with(this)
{createStandardPoolObjs();var sup=true;if(!bIsSandbox)
sup=isSupportedPool(oEnv.sThisPagePool);if(checkMicroBrand())
this.writeHeader=writeMicroBrandHeader;else if(!oDocument.getCookie("ebay","ht").is("")||sup)
{var sc=oSelCobrand=detectCobrand(c,true);if(!sc&&!sup)
sc=oSelCobrand=detectCobrand(c);if(oEnv.sThisPagePool.is("my")&&!sc)
sc=oSelCobrand=detectCobrand(c);if(sc)
{if(!bIsSandbox)
setCookie(sc.iHT);new EbayCobrandFunctions(this,"cobrandFunctions");new EbayCobrandFunctionsCountry(this,"countryCobrandFunctions");if(typeof(noCobrandHeader)=="undefined")
{oDocument.write(oUtils.scriptTag(oEnv.sIncludeHost+"js/"+oEnv.sTrainId+"/"+oEnv.sCountry+"/features/cobrand/"+sc.sHeaderName+oEnv.sJSPackageSuffix));}}
else
{}}
else{var sc=detectCobrand(c,true);if(sc&&sc.sHeaderName=="worldofgood")
{oDocument.write(oUtils.scriptTag(oEnv.sIncludeHost+"js/"+oEnv.sTrainId+"/"+oEnv.sCountry+"/features/cobrand/"+sc.sHeaderName+oEnv.sJSPackageSuffix));}}}}
this.onAfterLoad=function()
{var d=this.oDocument,h="cobrandHeader",f="cobrandFooter";if(d.getUIElem(h))
this._exec("writeHeader",h)
if(d.getUIElem(f))
this._exec("writeFooter",f);}
this.init();}

//6@@m17

function EbayCobrandFunctions(pParent,pName)
{if(!this.objType)
this.objType="EbayCobrandFunctions";this.base=EbayBaseControl;this.base(pParent,pName);var g=this.oGlobals;var env=this.oEnv=g.oEnvironment;var p=pParent;this.sServer=null;this.oSelCobrand=p.oSelCobrand;this.sQA=p.sQA;this.sCBCountryDir=p.sCBCountryDir;this.sRedirectString="ebayredir=";this.urlIDBranding=function(pURL)
{var so=this.oSelCobrand;if(so._exec("urlIDBrandingException",pURL))
return false;if(pURL.hasAny("&s_partnerid=","&co_partnerid="))
{var ids=so.aQueryIdentifiers;pURL=pURL.replace(/s_partnerid=[0-9]*/g,ids[1]);pURL=pURL.replace(/co_partnerid=[0-9]*/g,ids[2]);return pURL;}
return false;}
this.cobrandURL=function(pURL)
{var qInd=pURL.indexOf("?"),path=pURL,query="";if(qInd!=-1)
{path=pURL.substring(0,qInd);query=pURL.substring(qInd);}
var so=this.oSelCobrand;if(so.urlDetect(path,query))
return pURL;var rv=this.urlIDBranding(pURL);if(rv)
return rv;with(this)
{var folder=so.sFolder||"",dom=oGlobals.oEnvironment.sCountryDomain.toLowerCase();var st=dom;if(pURL.hasAny('com/ebaymotors/','ebaymotors')&&so.name.is("AOL Home"))
folder="";st=pURL.indexOf(st);if(st<1)
{return((pURL.indexOf("/")==0)&&(folder!="")&&(pURL.indexOf(folder)!=0))?folder+pURL:pURL;}
var off=pURL.indexOf(dom,st)+dom.length;rv=pURL.substring(0,st);if(!pURL.has(so.sCBName+dom)&&so._exec("linksConditionsException",pURL))
{rv+=so.sCBName+dom+folder+
pURL.substr(off);}
else if(!path.has(so.sCBName+dom)&&so._exec("linksConditionsException",path))
{st=dom
st=path.indexOf(st);off=path.indexOf(dom,st)+dom.length;rv=path.substring(0,st);rv+=so.sCBName+dom+folder+
path.substr(off)+query;}
else
rv=pURL;}
return rv;}
this.isEbayLink=function(pLink)
{with(this)
{var cs=parent.controls,l=pLink.toLowerCase();if(l.has(oGlobals.oEnvironment.sCountryDomain.toLowerCase()))
{if(cs)
{var tPool=l.substring(l.indexOf("://")+3,l.indexOf("."));for(var i in cs)
{if(cs[i].objType.is("EbayPool"))
{if(cs[i].isSupportedPool(tPool))
return true;}}}}}
return false;}
this.cobrandLinks=function()
{var dl=this.oDocument.doc.links,len=dl.length,temp,txt;for(var i=0;i<len;i++)
{temp=false;txt=dl[i].innerText;if(this.oGlobals.oClient.bIE&&txt.hasAny("@","http:"))
temp=txt;if(this.isEbayLink(dl[i].href))
dl[i].href=this.cobrandURL(dl[i].href);else if(this.isRedirectLink(dl[i].href))
dl[i].href=this.cobrandRedirectURL(dl[i].href);if(temp)
dl[i].innerText=temp;}}
this.cobrandRedirectURL=function(pUrl)
{var r=this.sRedirectString;var curl=pUrl.substring(pUrl.indexOf(r)+r.length,pUrl.length),rurl,furl;if(!this.isEbayLink(curl))
return pUrl;rurl=pUrl.substring(0,pUrl.length-curl.length);curl=this.cobrandURL(curl);furl=rurl+curl;return furl;}
this.isRedirectLink=function(pLink)
{var l=pLink.toLowerCase();return l.has(this.sRedirectString);}
this.cobrandForms=function()
{var act;var f=this.oDocument.doc.forms,len=f.length;for(var i=0;i<len;i++)
{act=f[i].action;if(typeof(act)=="string")
{if(this.isEbayLink(act))
f[i].action=this.cobrandURL(act);}
var fe=f[i].elements,eLen=fe.length;var un="undefined";for(var j=0;j<eLen;j++)
{with(fe[j])
{if(typeof(type)!=un&&typeof(value)!=un)
{if(type.is("hidden")&&(value.indexOf("http")==0)&&this.isEbayLink(value))
value=this.cobrandURL(value);}}}}}
this.updateSiteFormElem=function(pElem,pID)
{if(pElem)
{var idx=pElem.value.indexOf("&"),rest="";if(idx!=-1)
rest=pElem.value.substring(idx);pElem.value=pID+rest;}}
this.cobrandSearchAttribs=function()
{with(this)
{var df=oDocument.doc.forms,len=df.length;var sc=oSelCobrand,dfe;for(var i=0;i<len;i++)
{dfe=df[i].elements;updateSiteFormElem(dfe["ht"],sc.ht);updateSiteFormElem(dfe["s_partnerid"],sc.s_partnerid);updateSiteFormElem(dfe["co_partnerid"],sc.co_partnerid);}}}
this.cobrandPage=function()
{with(this)
{oSelCobrand.sCBName="."+oSelCobrand.sCBName;sServer="";if(!sQA.is(""))
{var tpp=env.sThisPagePath,c='.'+env.sCountry;sServer=tpp.substring(tpp.indexOf("."),(tpp.indexOf(c)!=-1?tpp.indexOf(c):tpp.indexOf(sQA)));var ex=[".motors",oSelCobrand.sCBName,".listings",".stores"],len=ex.length;for(var i=0;i<len;i++)
{if(sServer.indexOf(ex[i])==0)
sServer=sServer.substr(ex[i].length);}}
cobrandLinks();cobrandForms();cobrandSearchAttribs();parent._exec("writeFooter");}}
this.parent.cobrandPage=new Function("return this._getControl('cobrandFunctions')._exec('cobrandPage');");}

//7@@m1

new EbayCobrandCollection(ebay.oDocument,"cobrandCollection");

//8@@m8

(function(){var oD=ebay.oDocument,sImg="i_sGauge",oE=oD.oGlobals;if(!oE){return;};oE=oE.oEnvironment;oD.write("<img src='"+oE.sPicsDir+"s.gif' name='"+sImg+"' alt=' ' border=0 height='1' width='1'/>");oD.gaugeSite=function()
{var oCJ=oD.oCookieJar;if(typeof(oCJ)!='undefined'){var sbf=oCJ.readCookielet("ebay","sbf");oCJ.writeCookielet("ebay","sbf",oCJ.setBitFlag(sbf,20,1));}
if(typeof(oGaugeInfo)!='undefined'){if(oGaugeInfo.sent===true){return;}
oGaugeInfo.sent=true;var img=document.images[sImg],delta=(new Date()).getTime()-oGaugeInfo.iST;if(img)img.src=oGaugeInfo.sUrl.replace(/&amp;/g,'&').trim()+delta;}}
oD.gaugeUnload=function()
{var oCJ=oD.oCookieJar;if(typeof(oCJ)!='undefined'){oCJ.writeCookielet("ssg","uld",(new Date()).getTime());}}
oD._registerListener(oD._getEvent("load"),oD.EVENT_AFTER,"gaugeSite");oD._registerListener(oD._getEvent("unload"),oD.EVENT_AFTER,"gaugeUnload");})();
// b=11246123 -->