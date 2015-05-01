//<!--
//\include\js\lib\_toolbox\_base\ebayAds.js@@\main\8

var _ebayAds=new Array;var layerPrefix="ad_";var totAds=0;function EbayAd(pEbayDoc,pParent,pName,pCfg,pRandomNum)
{if(!this.objType)
this.objType="EbayAd";this.baseObject=EbayControl;this.baseObject(pEbayDoc,pParent,pEbayDoc.htmlDoc,pName);this.globals=initAdGlobals();this.ord=pRandomNum?pRandomNum:(new Date()).getTime();this.adUrl="";this.adLinkUrl="";this.adImageUrl="";this.config=pCfg?pCfg:(new EBayAdConfig(pName));this.toHTML=EbayBuildAdHTML;this.zoneEncode=EbayZoneEncode;this.addControl=EbayAddControl;this.layerId=layerPrefix+this.name+"1";this.setAdLayerId=EbaySetAdLayerId;this.setCountryGlobals=null;this.setCountryLocals=null;totAds++;if(document.layers)
EbayRegisterAd(this);}
window.EbayAd=EbayAd;function EbayRegisterAd(pAdObj)
{if(pAdObj)
{pAdObj.setAdLayerId(_ebayAds.length);_ebayAds[_ebayAds.length]=pAdObj;}}
function EbaySetAdLayerId(pId)
{this.layerId=layerPrefix+this.name+pId;}
function EbayZoneEncode(pName)
{var rs="",c;for(var i=0;i<pName.length;i++)
{c=pName.charAt(i);if(c==' ')
rs+="_";else if(c=='&')
rs+="and";else if(c!=',')
rs+=c;}
return rs.toLowerCase();}
function EBayAdConfig(pName)
{if(!this.objType)
this.objType="EBayAdConfig";this.base=EBayConfig;this.base(pName);this.tile=1;this.width=0;this.height=0;this.ifWidth=0;this.ifHeight=0;this.DARTSite="";this.DARTSiteAll="";this.zones=new Array;this.categories=new Array;this.params=new Array;this.addParam=EbayAddAdParam;}
window.EBayAdConfig=EBayAdConfig;function EbayAdGlobals()
{this.host="http://us.ebayobjects.com/";this.sitePrefix="ebay.us";this.setHost=EbaySetAdHost;this.setSitePrefix=EbaySitePrefix;this.resetGlobals=EbayResetGlobals;this.ebayUserCode="";this.resetGlobals();}
function EbayResetGlobals()
{this.iframeUrl=this.host+"1ai/"+this.sitePrefix;this.layerUrl=this.host+"1al/"+this.sitePrefix;this.linkUrl=this.host+"3j/"+this.sitePrefix;this.imageUrl=this.host+"1a/"+this.sitePrefix;}
function EbaySetAdHost(pHost)
{this.host=pHost;if(pHost.substring(pHost.length-1,pHost.length)!="/")
this.host=pHost+"/";this.resetGlobals();}
function EbaySitePrefix(pSitePre)
{this.sitePrefix=pSitePre;this.resetGlobals();}
function EbayAddAdParam(pKey,pValue)
{if(typeof(pValue)=="undefined")
pValue='';if(pKey.length>0)
this.params[pKey.toLowerCase()]=pValue.toLowerCase();}
var adGlobals=null;function initAdGlobals()
{if(adGlobals)
return adGlobals;else
{adGlobals=new EbayAdGlobals();return adGlobals;}}
function EbayBuildAdHTML()
{if(this.setCountryGlobals)
{this.setCountryGlobals(this.globals);this.globals.resetGlobals();}
if(this.setCountryLocals)
this.setCountryLocals(this.config);var cfg=this.config;var gb=this.globals;var sData="",sZones="",sCats="";var sAdLUrl=""
var sAdIFUrl="",sAdImgUrl="",sAdLinkUrl="";cfg.DARTSite=cfg.DARTSite.toLowerCase();cfg.DARTSiteAll=gb.sitePrefix+"."+cfg.DARTSite;for(var i=0;i<cfg.zones.length;i++)
{if(sZones.length>0)
sZones+="/";sZones+=this.zoneEncode(cfg.zones[i].toLowerCase());}
if(sZones.length>0)
sZones=sZones+";"
for(var i=0;i<cfg.categories.length;i++)
{if(cfg.categories[i].length>0)
sCats+="cat="+cfg.categories[i]+";";}
if(cfg.ifWidth<cfg.width)
cfg.ifWidth=cfg.width;if(cfg.ifHeight<cfg.height)
cfg.ifHeight=cfg.height;if(cfg.tile)
sData+="tile="+cfg.tile+";";for(var k in cfg.params)
{if(cfg.params[k].length>0)
sData+=k+"="+cfg.params[k]+";";}
if(cfg.width&&cfg.height)
sData+="sz="+cfg.width+"x"+cfg.height+";";sData+="ord="+this.ord+";"
sAdLUrl=gb.layerUrl+"."+cfg.DARTSite+(sZones?("/"+sZones):"")+(sCats?sCats:"")+sData;sAdIFUrl=gb.iframeUrl+"."+cfg.DARTSite+(sZones?("/"+sZones):"")+(sCats?sCats:"")+sData;sAdLinkUrl=gb.linkUrl+"."+cfg.DARTSite+(sZones?("/"+sZones):"")+(sCats?sCats:"")+sData;sAdImgUrl=gb.imageUrl+"."+cfg.DARTSite+(sZones?("/"+sZones):"")+(sCats?sCats:"")+sData;var rs="";if(document.layers)
{this.adUrl=sAdLUrl;rs='<ILAYER id="'+this.layerId+'" visibility="hidden" width="';rs+=cfg.ifWidth+'" height="'+cfg.ifHeight+'"></ILAYER>';}
else
{this.adUrl=sAdIFUrl;this.adLinkUrl=sAdLinkUrl;this.adImageUrl=sAdImgUrl;rs='<IFRAME id="'+this.name+'_'+totAds+'" FRAMEBORDER="no" BORDER="0" MARGINWIDTH="0" MARGINHEIGHT="0" SCROLLING="no"';rs+=' SRC="'+sAdIFUrl+'"';rs+=' WIDTH="'+cfg.ifWidth+'" HEIGHT="'+cfg.ifHeight+'"';rs+='>';rs+='<A HREF="'+sAdLinkUrl+'">';rs+='<IMG SRC="'+sAdImgUrl+'" border="0" height="'+cfg.height+'" width="'+cfg.width+'">';rs+='</A>';rs+='</IFRAME>';}
return rs;}
// -->
