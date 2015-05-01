//<!--
//1@@m1

String.prototype.decodeBase64=function()
{var rv=this,len=rv.length,ret="",i=0;var chr1,chr2,chr3="";var enc1,enc2,enc3,enc4="";var aChar="ABCDEFGHIJKLMNOPQRSTUVWXYZ"+"abcdefghijklmnopqrstuvwxyz"+"0123456789+/=*";var test=new RegExp("[^A-Za-z0-9+/=*]");if(test.exec(rv)){return;}
do{enc1=aChar.indexOf(rv.charAt(i++));enc2=aChar.indexOf(rv.charAt(i++));enc3=aChar.indexOf(rv.charAt(i++));enc4=aChar.indexOf(rv.charAt(i++));chr1=(enc1<<2)|(enc2>>4);chr2=((enc2&15)<<4)|(enc3>>2);chr3=((enc3&3)<<6)|enc4;ret+=String.fromCharCode(chr1);if(!(enc3>=64))
ret+=String.fromCharCode(chr2);if(!(enc4>=64))
ret+=String.fromCharCode(chr3);chr1=chr2=chr3=enc1=enc2=enc3=enc4="";}while(i<len);return ret;}
String.prototype.decodeUTF8=function()
{var s=this,len=s.length;var rs="";var i=0;var c=c1=c2=0;while(i<len)
{c=s.charCodeAt(i);if(c<128)
{rs+=String.fromCharCode(c);i++;}
else if((c>191)&&(c<224))
{c2=s.charCodeAt(i+1);rs+=String.fromCharCode(((c&31)<<6)|(c2&63));i+=2;}
else
{c2=s.charCodeAt(i+1);c3=s.charCodeAt(i+2);rs+=String.fromCharCode(((c&15)<<12)|((c2&63)<<6)|(c3&63));i+=3;}}
return rs;}

//2@@m1

function EbayHeaderPoweredBy(pParent,pName)
{if(!this.objType)
this.objType="EbayHeaderPoweredBy";this.base=EbayBaseControl;this.base(pParent,pName);this.writeSource=function(pImage,pLink)
{with(this.oGlobals.oEnvironment)
{var s=pImage;if(!sThisPagePath.has("ebayisapi.dll?viewitem")&&pLink)
s=pLink+pImage+'</a>';}
this.oDocument.write(s);}}
new EbayHeaderPoweredBy(ebay.oDocument,"poweredby");

//3@@m11

function EbayHeaderSignIn(pParent,pName)
{if(!this.objType)
this.objType="EbayHeaderSignIn";this.base=EbayBaseControl;this.base(pParent,pName);this.writeLink=function(pNotInText,pNotInLink,pInText,pInLink,pIsMotors,pIsPersonal,pRUValue)
{var d=this.oDocument,v1=d.getCookie("ebaysignin"),v2=d.getCookie("keepmesignin");if(pIsPersonal&&!pIsMotors)
{if(v1.is("in")||v2.is("in")||!pInText.is(''))
return;}
var NotInLink=new EbayDataURL(pNotInLink);var currentURL=new EbayDataURL(document.location.href);if(!NotInLink.hasArg("ru"))
{var ruv=pRUValue?pRUValue:encodeURI(currentURL.full);if(ruv.length<156)
NotInLink.addArg("ru",ruv);}
pNotInLink=NotInLink.getURL();var pre=" | <a href=\"",mid="\">",post="</a>";if(pIsMotors)
{mid="\"><font face=\"verdana, arial, helvetica\" size=\"1\" color=\"#000000\"><b>";post="<b>"+post;}
var s=pre;if((v1.is("in")||v2.is("in"))&&!pInText.is(''))
s+=pInLink+mid+pInText;else
s+=pNotInLink+mid+pNotInText;d.write(s+post);}}
new EbayHeaderSignIn(ebay.oDocument,"signIn");

//4@@m12

function EbayHeaderSearch(pParent,pName)
{if(!this.objType)
this.objType="EbayHeaderSearch";this.base=EbayBaseControl;this.base(pParent,pName);var cl=this.oGlobals.oClient;var env=this.oGlobals.oEnvironment;this.bNS4=document.layers;this.bNS6up=(cl.bNav&&(cl.iVer>6));this.bIE4up=(cl.bIE&&(cl.iVer>4));this.writeSearch=function(pValue,pBoxSize,pCSS)
{var srchSz=(typeof(pBoxSize)!='undefined')?pBoxSize:20;var txtBxCls=' class="textboxhelp"';if(typeof(pCSS)!='undefined')
txtBxCls=' class="'+pCSS+'"';var onClkCSS=(typeof(pCSS)!='undefined')?pCSS:'textbox';var onClk=' onclick="if(this.value == \''+pValue;onClk+='\')this.value=\'\';this.className=\''+onClkCSS+'\';"';var vl=' value="'+pValue+'"';if(this.bNS4)
{srchSz=12;txtBxCls=' class="textbox"';onClk=vl="";}
var onKDwn=' onkeydown="if(this.value==\''+pValue+'\')this.value=\'\';"';var s='<input type="text"'+txtBxCls+' maxlength="300" ';if(cl.bFirefox)
s+='autocomplete = "off"';s+='name="satitle" size="'+srchSz+'"'+onKDwn+onClk+vl+' />';this.oDocument.write(s);}
this.getSpacer=function()
{var s='<img src="'+env.sPicsDir+'x.gif" alt="" ';s+='width="1" height="5">';if(!this.bNS4)
this.oDocument.write(s);}
this.submitHeaderSearch=function(pFrm,pHost,pSiteId,pDefValue)
{if(typeof(pFrm)=='undefined')
return false;if(pFrm.satitle&&pFrm.satitle.value==pDefValue)
pFrm.satitle.value='';return true;}}
new EbayHeaderSearch(ebay.oDocument,"searchHeader");

//5@@m9

function EbayDataURL(pURL)
{if(!this.objType)
this.objType="EbayDataURL";this.full=null;this.base=null;this.arguments=null;this.protocol="http";this.host=null;this.port=null;this.path="/";this.path_filename="/";this.filename=null;this.args=[];this.invalid_url=false;this.base_arg_del_index=null;this.base_arg_delimiter="?";this.key_value_delimiter="=";this.arg_delimiter="&";this.host_port_delimiter=":";this.protocol_host_delimiter="://";this.invalid_chars=["*","\"","\'","<",">","|"];this.init=ebDataURLInit;this.getArg=ebDataURLGetArg;this.addArg=ebDataURLAddArg;this.getURL=ebDataURLGetURL;this.parseArgs=ebDataURLParseArgs;this.hasArg=ebDataURLHasArg;this.init(pURL);}
function ebDataURLInit(pURL)
{with(this)
{full=base=pURL;base_arg_del_index=pURL.indexOf(base_arg_delimiter);if(base_arg_del_index!=-1)
{arguments=pURL.substring(base_arg_del_index+1);parseArgs(arguments);base=pURL.substring(0,base_arg_del_index);}
if(invalid_chars)
{for(var i in invalid_chars)
{if(base.indexOf(invalid_chars[i])!=-1)
{invalid_url=true;full=null
base=null;arguments=null;return false;}}}
var host_start=0;var host_end=base.length;if(base.indexOf("/")!=-1&&base.charAt(base.indexOf("/")+1)=="/")
{host_start=base.indexOf("/")+2;host_end=host_start+base.substring(host_start).indexOf("/");}
else
{host_start=0;if(base.indexOf("/")!=-1)
host_end=base.indexOf("/");}
host=base.substring(host_start,host_end);if(base.indexOf(host_port_delimiter)!=-1)
protocol=base.substring(0,base.indexOf(host_port_delimiter));var port_start=host.indexOf(":")
if(port_start!=-1)
{port=host.substring(port_start+1);host=host.substring(0,port_start);}
path_filename=base.substring(host_end);if(path_filename.indexOf(".")!=-1)
{path=path_filename.substring(0,path_filename.lastIndexOf("/"));filename=path_filename.substring(path_filename.lastIndexOf("/")+1);}
else
{if(path_filename.lastIndexOf("/")==path_filename.length-1)
path=path_filename.substring(0,path_filename.length-1);else
path=path_filename;}}}
function ebDataURLGetArg(pArgName)
{var tmp_array=[];for(var i=0;i<this.args.length;i++)
{if(this.args[i][0]==pArgName)
tmp_array[tmp_array.length]=this.args[i];}
if(tmp_array.length<1)
return;else if(tmp_array.length==1)
return tmp_array[0][1];else
return tmp_array;}
function ebDataURLAddArg(pArgName,pArgValue)
{this.args[this.args.length]=[pArgName,pArgValue];}
function ebDataURLHasArg(pArgName)
{return typeof(this.getArg(pArgName))!="undefined";}
function ebDataURLGetURL()
{with(this)
{if(!host)
return null;var url=arg_str="";for(var i=0;i<args.length;i++)
{if(arg_str!="")
arg_str+=arg_delimiter;if(args[i][1]=="")
arg_str+=args[i][0];else
arg_str+=args[i][0]+key_value_delimiter+args[i][1];}
if(protocol)
url+=protocol+protocol_host_delimiter;if(host)
url+=host;if(port)
url+=host_port_delimiter+port;if(path)
url+=path;if(filename)
url+="/"+filename;if(arg_str!="")
url+=base_arg_delimiter+arg_str;return url;}}
function ebDataURLParseArgs(pArgs)
{with(this)
{var new_args=new Array();while(pArgs.indexOf(arg_delimiter)!=-1)
{var outterdel=pArgs.indexOf(arg_delimiter);var innerdel=pArgs.indexOf(key_value_delimiter);if(outterdel>innerdel)
{var key=pArgs.substring(0,innerdel);var value=pArgs.substring(innerdel+1,outterdel);if(key!="")
{if(key.indexOf('amp;')==0)
key=key.replace('amp;','');addArg(key,value);}}
else
{var key=pArgs.substring(0,outterdel);if(key!="")
{if(key.indexOf('amp;')==0)
key=key.replace('amp;','');addArg(key,"");}}
pArgs=pArgs.substring(outterdel+1);}
if(pArgs!="")
{var innerdel=pArgs.indexOf(key_value_delimiter);if(innerdel==-1)
{addArg(pArgs,"");}
else
{var key=pArgs.substring(0,innerdel);var value=pArgs.substring(innerdel+1);if(key!="")
{addArg(key,value);new_args[key]=value;}
pArgs="";}}}}

//6@@m13

var u1p=readCookieletEx("dp1","u1p");function EbayHeaderGreetings(pParent,pName)
{if(!this.objType)
this.objType="EbayHeaderGreetings";this.base=EbayBaseControl;this.base(pParent,pName);this.writePersonalHeader=function(pSignInText,pSignInLink,pRegText,pRegLink,pSoutText,pSoutLink,pUnidentify,pAuthenticated,pIdentified,pDefaultGreeting,pIcon,pAlert,pAlerts,pRU,pRUVal)
{var s="";var sinLink="",soutLink="",regLink="",un='undefined',d=this.oDocument,sCountry=this.oGlobals.oEnvironment.sCountry;if(!u1p||typeof(u1p)==un)
{s=typeof(pDefaultGreeting)!=un?pDefaultGreeting:'';d.write(s);return;}
var u1pDecoded=u1p.decodeBase64().decodeUTF8(),v1=readCookieEx("ebaysignin"),v2=readCookieEx("keepmesignin");if(typeof(u1pDecoded)==un)
return;var signInLink=new EbayDataURL(pSignInLink);if(parseInt(pRU)==1)
{var currentURL=new EbayDataURL(document.location.href);signInLink.addArg("ru",pRUVal?pRUVal:encodeURIComponent(currentURL.full));}
pSignInLink=signInLink.getURL();var a=readCookieletEx('dp1','a1p'),ba,bp;ba=a&&a>0;bp=this.isUserPersonalized();if(u1pDecoded.has('@@__@@__@@'))
{s+=typeof(pUnidentify)!=un?pUnidentify:'';sinLink='<a href="'+pSignInLink+'" >'+pSignInText+'</a>';regLink='<a href="'+pRegLink+'" >'+pRegText+'</a>';s=s.replaceTokensEx("##n##",sinLink,regLink);}
else if(v1.is("in")||v2.is("in"))
{if(bp)
{if(ba)
{s=pIcon+'&nbsp;';}
s+=typeof(pAuthenticated)!=un?pAuthenticated:'';soutLink='<a href="'+pSoutLink+'" >'+pSoutText+'</a>';s=s.replaceTokensEx("##n##",u1pDecoded,soutLink);if(ba)
{if(a==1)
s+=pAlert;else
s+=pAlerts;s=s.replaceTokensEx("##n##",a);}}
else
{s=pDefaultGreeting;}}
else
{if(ba)
{s=pIcon+'&nbsp;';}
s+=typeof(pIdentified)!=un?pIdentified:'';sinLink='<a href="'+pSignInLink+'" >'+pSignInText+'</a>';s=s.replaceTokensEx("##n##",u1pDecoded,sinLink);if(ba)
{if(a==1)
s+=pAlert;else
s+=pAlerts;s=s.replaceTokensEx("##n##",a);}}
d.write(s);}
this.isUserPersonalized=function()
{var cJ=ebay.oDocument.oCookieJar,cId=cJ.readCookie("cid"),pCd=cJ.readCookielet("dp1","pcid");if(pCd)
{if(cId)
{cId=decodeURIComponent(cId);var hCd=cId;var Inx=cId.indexOf("#");if(Inx==-1&&cId.length>8)
return false;if(Inx!=-1)
hCd=cId.substring(Inx+1,cId.length);if(hCd==pCd)
{cJ.writeCookie("cid",cId,"","",cJ.getDate(1));return true;}
else
return false;}
return false;}
return true;}}
new EbayHeaderGreetings(ebay.oDocument,"greetings");

//7@@m20

function EbayRoverNS(pParent,pName,pConfig)
{if(!this.objType){this.objType="EbayNSDetect";}
this.base=EbayBaseControl;this.base(pParent,pName);this.ebaysites=new RegExp("^(http[s]?:\\/\\/)?[\\w-.]+\\.(ebay(objects|motors|promotion|development|static|express|liveauctions|rtm)?)\\.[\\w-.]+($|\\/.*)","i");this.aRotationIDs=new Array();this.isTop=function(){var oD=this.oDocument,w=oD.win,d=oD.doc;var p=(w.parent==w);var o=(!(o==null||o==""||typeof(w.opener)=='undefined'));var r=d.referrer;if(p&&(!o||!(r==null&&r==""))){return true;}
else{return false;}};this.init=function(){if(this.isTop()){this.track();}};this.track=function(){var clientTime=new Date().getTime();var offset=this.getClientTimeOffset(clientTime);var tpim=this.getTpimCookielet(offset,clientTime);var roverNsDropped=false;if(typeof(RoverNsCapable)=='undefined'&&(tpim==null||tpim==""||tpim=="exp")){var oD=this.oDocument,d=oD.doc,r=d.referrer;if(!(r==null||r==""||r.match(this.ebaysites))){this.dropRoverNSImg();roverNsDropped=true;}}
if(typeof(RoverNsCapable)=='undefined'&&typeof(RoverSyncDropped)=='undefined'&&roverNsDropped==false&&tpim=="exp"){this.dropRoverSyncRtpimImage();}};this.getTpimCookielet=function(cos,clientTime){var oCJ=ebay.oDocument.oCookieJar;var npiiCookieVal=oCJ.readCookie("npii");if(npiiCookieVal.indexOf('tpim')>=0){var tpimval=npiiCookieVal.replace(new RegExp("(.*\\^tpim\/)([a-fA-F0-9]+)\\^.*"),"$2");var tpimexp=tpimval.substring(tpimval.length-8);var expTime=tpimexp.hex2Dec()*1000+cos*1000;var gmtDelta=expTime-clientTime;var expiredTime=15552000000;if((gmtDelta>expiredTime||gmtDelta<0)){return"exp";}
return oCJ.readCookielet("npii","tpim");}
return"";};this.dropRoverSyncRtpimImage=function(){var roverEnv=this.parent.oGlobals.oEnvironment;if(roverEnv!=null){var dmain=roverEnv.sCountryDomain;if(dmain!=null){var im='<img width="1" height="1" src="'+roverEnv.sProtocol+'//rover'+dmain+'roversync/?rtpim=1&amp;mpt='+new Date().getTime()+'">';this.oDocument.write(im);}}};this.addRotationId=function(pKey,pValue)
{this.aRotationIDs[pKey]=pValue;};this.dropRoverNSImg=function()
{var e=this.parent.oGlobals.oEnvironment;var d=(document.domain.has(e.sQADomain))?e.sQADomain:".";var rotationId=this.aRotationIDs[e.sCountry];if(e.sThisPage!=null&&this.isHalfDotComPage(e.sThisPage)){rotationId="8971-56634-20761-0";}
var url=e.sProtocol+'//rover'+d+'ebay.com/roverns/1/'+rotationId+'?mpvl='+encodeURIComponent(e.sLastPage)+'&mpcl='+encodeURIComponent(e.sThisPage)+'&pageType=static';var im='<img width="1" height="1" src="'+e.sProtocol+'//rover'+d+'ebay.com/roverns/1/';im+=rotationId;im+='?mpvl='+encodeURIComponent(e.sLastPage);im+='&mpcl='+encodeURIComponent(e.sThisPage)+'&pageType=static">';this.oDocument.write(im);};this.getClientTimeOffset=function(clientTime)
{var oCJ=ebay.oDocument.oCookieJar;var offset=oCJ.readCookielet("ebay","cos");var offsetInSec;if(offset!=null&&offset!=""){offsetInSec=offset.hex2Dec();}
else if(typeof(svrGMT)!='undefined'){offsetInSec=Math.round((clientTime-svrGMT)/1000.0);if(!isNaN(offsetInSec)){oCJ.writeCookielet("ebay","cos",offsetInSec.toString(16));}}
if(isNaN(offsetInSec)){offsetInSec=1800;}
return offsetInSec;};this.isHalfDotComPage=function(thisPage)
{var reg=new RegExp("^http[s]?:\\/\\/[\\w-.]*\\.half\\.([\\w-.]*)ebay\\.","i");return reg.test(thisPage);};}
new EbayRoverNS(ebay.oDocument,"RoverNS");

//8@@m2

with(ebay.oDocument._getControl("RoverNS"))
{aRotationIDs={"at":"5221-27090-11243-0","au":"705-16386-7124-0","befr":"1553-16392-7839-0","benl":"1553-16392-7839-0","ca":"706-5385-7828-0","ch":"5222-27091-11242-0","cn":"4080-21866-9194-0","de":"707-16360-3220-0","es":"1185-16394-7840-0","fr":"709-16363-2357-0","it":"724-16468-4166-0","nl":"1346-16393-7838-0","tw":"1631-7654-5203-0","uk":"710-16388-7832-0","us":"711-13271-9788-0","in":"4686-23749-10070-0","ie":"5282-27434-11362-0","pl":"4908-25050-10522-0","hk":"3422-25268-10602-0","sg":"3423-25269-10603-0"};init();}

//9@@m2

ebay.oDocument.oPage.createConfig=function()
{var c=ebay.oDocument.addConfig(new EbaySiteCatalystConfig("siteCatalyst"));c.turnOnTracking=1;}
ebay.oDocument.oPage.createConfig();

//10@@m38

function EbaySiteCatalystFiles(pParent,pName)
{if(!this.objType)
this.objType="EbaySiteCatalystFiles";this.base=EbayBaseControl;this.base(pParent,pName);var env=this.oGlobals.oEnvironment;this.aFiles=new Array;var pd=this.sPackageDir=env.sIncludeHost+"js/"+env.sTrainId+"/"+env.sCountry+"/features/site_catalyst/";this.sPageNamesDir=pd+"pagenames/";this.sPropertyReportsDir=pd+"property_reports/";this.sEventsDir=pd+"events/";this.sCookiesDir=pd+"cookies/";this.add=function()
{var args=arguments,aLen=args.length,fs=this.aFiles;for(var i=0;i<aLen;i++)
{var found=false,len=fs.length;for(var j=0;j<len&&!found;j++)
found=(fs[j]==args[i]);if(!found)
fs[len]=args[i];}}}
function EbaySiteCatalystConfig(pName)
{if(!this.objType)
this.objType="EbaySiteCatalystConfig";this.base=EbayConfig;this.base(pName);this.turnOnTracking=0;this.countrySampleRate=100;this.isIframe=false;}
function EbaySiteCatalyst(pParent,pName,pConfig)
{if(!this.objType)
this.objType="EbaySiteCatalyst";this.base=EbayBaseControl;this.base(pParent,pName);this.oConfig=pConfig;var env=this.oGlobals.oEnvironment;this.oFiles=new EbaySiteCatalystFiles(this,"files");this.aAccounts=new Array;this.sCookieDom=env.sCookieDomain;this.sCookieName="ebay";this.sCountry=this.sQA=this.sSample=this.sPage=this.sHost=this.sPageName=this.sChannel="";this.sUser="Nonreg";this.bLoadBaseCode=false;this.bDebugMode=env.sThisPageQuery.has("ebscdebug=on");this.dExpires=new Date();this.writeCookie=function(pName,pVal)
{writeCookieletEx(this.sCookieName,pName,pVal,this.sCookieDom,"/");}
this.readCookie=function(pName)
{return readCookieletEx(this.sCookieName,pName);}
this.addAccount=function(pAccount)
{this.aAccounts[this.aAccounts.length]=pAccount;}
this.setCountry=function()
{with(this)
{var env=oGlobals.oEnvironment;var c=env.sCountry.toUpperCase(),tp=sPage;c=c.substring(0,2);if(tp.has('.com/nz/'))c="NZ";else if(tp.has('ebaysweden'))c="SE";else if(tp.has('.com.sg/'))c="SG";else if(tp.has('.com.hk/'))c="HK";if(env.sSiteID=="0"&&(tp.has("/ebaymotors/")||(env.sThisPageHost.has(".motors.")))){env.sSiteID="100";}
sCountry=c+";"+env.sSiteID+";";}}
this.getOrDropLuckyCookie=function()
{with(this)
{var l9="lucky9",v=oDocument.getCookie(l9);if(v.is("")||v.length!=7)
v=new String(Math.floor(Math.random()*10000000));writeCookieEx(l9,v,sCookieDom,"/",dExpires);sSample=v;return v;}}
this.checkRegCookie=function()
{with(this)
{if((readCookieletEx("reg","flagReg")=="1")||(readCookieEx("ebaysignin")=="in")||(readCookieEx("reg")=="1"))
{sUser="Reg";writeCookieletEx("reg","flagReg","1",sCookieDom,"/",dExpires);}}}
this.determineAccounts=function()
{with(this)
{var l9=getOrDropLuckyCookie(),len=(l9.length>0);var doc=oDocument;sChannel=oGlobals.oEnvironment.sCountryDesc;if(len)
{var cnt=oGlobals.oEnvironment.sCountry;if(cnt=="ie"||l9%oConfig.countrySampleRate==6)
{addAccount("ebay"+sChannel+sQA);}}}}
this.loadFiles=function()
{with(this)
{var env=oGlobals.oEnvironment;var fs=oFiles.aFiles,len=fs.length;for(var i=0;i<len;i++)
oDocument.write(oUtils.scriptTag(fs[i]+env.sJSPackageSuffix));}}
this.init=function()
{with(this)
{if(!oConfig.turnOnTracking)
return;var yr=dExpires.getYear();if(yr<1900)
yr=yr+1900;dExpires.setYear(yr+5);var doc=oDocument;doc.write('<img name="s_i_ebay" width=1 height=1 border=0>');var env=oGlobals.oEnvironment,tp=env.sThisPage;sPage=tp.toLowerCase();sHost=env.sThisPageHost.toLowerCase();sQA=env.sThisPagePath.has(env.sQADomain)?"qa":"";checkRegCookie();determineAccounts();setCountry();bLoadBaseCode=(aAccounts.length||bDebugMode);}}
var tracking=ebay.oDocument.bEnableSiteCatalystTracking;this.trackFromPage=function(obj){with(this)
if(window.s_ebay&&obj&&obj.pageName){obj.pageName=obj.prop22=obj.prop26=sCountry+obj.pageName;execCustomFunctions(obj);var c=s_ebay.t(obj);if(c)
document.write(c);}}
if(typeof(tracking)=="undefined"||tracking)
this.init();}
new EbaySiteCatalyst(ebay.oDocument,"siteCatalyst",ebay.oDocument.getConfig("siteCatalyst"));

//11@@m3

function ebSiteCatalystIsTrackSearchAd()
{if(window==window.parent)
return true;}

//12@@m16

ebay.oDocument._getControl("siteCatalyst").attachFunctionality=function()
{this.sPageNameCountrySite="";this.ssADMEC=this.sSBR=this.sFND2=this.sDomValue=this.aBibo=this.aSegF="";this.aCustomFunctions=new Array;this.sCookieNames=new Array;this.debug=new Function("");this.registerFunction=function(pName)
{this.aCustomFunctions[this.aCustomFunctions.length]=pName;}
this.addCookieName=function(pName,pInSSCookie)
{this.sCookieNames[pName]=pInSSCookie;}
this.getAndSetPageName=function()
{with(this)
{var pn=oDocument.doc.title,un="undefined";if((typeof(pageName)!=un)&&(pageName!="document.title"))
pn=pageName;sPageNameCountrySite=sCountry+pn;with(oFiles)
{var p=parent,pnl=pn.toLowerCase(),pg=p.sPage.toLowerCase(),ed=sEventsDir;var pnd=sPageNamesDir,prd=sPropertyReportsDir,cd=sCookiesDir;var pkdir=sPackageDir;add(pkdir+"common");if((typeof(keyword)!=un)&&keyword)
add(cd+"keyword");p.addCookieName("ssSBR",true);if(typeof(SBR)!=un)
add(cd+"sbrWrite");p.addCookieName("ssFND2",true);if(typeof(FND2)!=un)
add(cd+"fnd2Write");p.addCookieName("ADME",true);if(pg.has("adme:"))
add(cd+"admeWrite");if(pg.has("sspagename="))
{if(pg.has("mops"))
add(pkdir+"mops");add(cd+"ssPagename");writeLL=true;}
p.addCookieName("lastList",true);if(pnl.hasAny('viewsUsersPage','aboutmystore','storefrontlistings','sPagestorefrontlistings','storeinternalsearchresults','storebinitemsearchresults')||(typeof(writeLL)!=un))
add(cd+"lastListWrite");if(pnl.has('viewitem'))
add(pkdir+"viewitem");if(pnl.hasAny('search','listings'))
{p.addSBRPropReportFiles();add(pkdir+"search_listings");}
if(pnl.hasAny('productpage','storesearch','pagesoi'))
p.addSBRPropReportFiles();if(pnl.hasAny('bincongrats','acceptbid'))
{var f=pnl.has('bincongrats')?"n":"d";add(pkdir+"bid_bin/bi"+f);p.ssADMEC="B";p.addFND2PropReportFiles();}
if(pnl.hasAny('pagereviewbin','pagebinverify'))
{add(pkdir+"bid_bin/attempt_bin");}
else if(new RegExp("^page.*(makebid|reviewbid)").test(pnl))
{add(pkdir+"bid_bin/attempt_bid");}
if(pnl.has('watchconfirm')||oDocument.getConfig("ResultSet"))
{add(prd+"CommonSBRFND2");add(pkdir+"watch_confirm");p.addFND2PropReportFiles();}
if(pnl.hasAny('pageaddnewitem','pagesyi_confirm','sellhub_confirm','pagesyi_lowerreservepriceconfirm'))
{p.ssADMEC="L";add(pkdir+"listing_confirm");}
if(pnl.hasAny('registersuccess','registercongrats','registeraolcongrats'))
{p.ssADMEC="R";add(ed+"registerSuccess");}
if(!p.ssADMEC.is(""))
add(cd+"admeRead");if(pnl.hasAny('registerenterinfo','busregenterinfo','businessregenterinfo'))
add(ed+"registerEnter");if(pnl.has('pageregisterconfirmemail'))
add(ed+"registerConfirm");pn=p.setOtherPagenames(pnl,pn);if((typeof(pfType)!=un)&&pfType)
add(prd+"productFinder");if((typeof(relatedStores)!=un)&&(relatedStores==1))
add(prd+"relatedStores");if(pnl.hasAny('checkoutreview','checkout_challenge')&&sCountry.isAny("us","uk"))
add(ed+"checkout");}
sPageName=sCountry+pn;}}
this.addSBRPropReportFiles=function()
{with(this.oFiles)
add(sPropertyReportsDir+"sbr");}
this.addFND2PropReportFiles=function()
{with(this.oFiles)
add(sPropertyReportsDir+"fnd2");}
this.setOtherPagenames=function(pPnl,pPn)
{var tp=this.sPage,rv=pPn;if(pPnl.is("sPagesignin")&&(this.oDocument.getCookie("ebaysignin")=="in"))
rv="PageSignOut";if(tp.has('liveauctions.'))rv="LiveAuctions:"+rv;if(tp.hasAny("searchcat","&pilot"))rv+=":CIS";if(tp.has("&allcats"))rv+=":AllCats";if(tp.has("&mocat"))rv+=":MoreCats";if(tp.has("&top10"))rv+=":Top10";return rv;}}
ebay.oDocument._getControl("siteCatalyst")._exec("attachFunctionality");

//13@@m9

ebay.oDocument._getControl("siteCatalyst").load=function()
{with(this)
{var un='undefined';var stores=((typeof(eBayTREiasId)!=un)&&(typeof(eBayTRPageName)!=un)&&(typeof(eBayTRInactive)!=un)&&!eBayTRInactive);if(bLoadBaseCode||stores)
{if(typeof(getAndSetPageName)=="function")
getAndSetPageName();with(oFiles)
{var pd=sPackageDir;if(bDebugMode&&oGlobals.oClient.bIE)
add(pd+"debug");if(this.aAccounts&&this.aAccounts.length>0){s_account=this.aAccounts[0];for(i=1;i<this.aAccounts.length;i++){s_account+=", "+this.aAccounts[i];}
add(pd+"omniture/scode");}
if(stores){add(pd+"omniture/base");add(pd+"omniture/stores");}
if(parent.bLoadBaseCode)add(pd+"base");}
loadFiles();}}}
ebay.oDocument._getControl("siteCatalyst")._exec("load");

//14@@m4

var isNS4x=false;if(document.layers)
isNS4x=true;function processAdLayers()
{if(isNS4x)
{if(typeof(oAdManager)!="undefined")
{var h="";var aAds=oAdManager.ads;for(var i=0;i<aAds.length;i++)
{var lId=aAds[i].layerId;var lW=aAds[i].config.ifWidth;var lH=aAds[i].config.ifHeight;h+='<LAYER SRC="'+aAds[i].adUrl+'" width="'+lW+'" height="'+lH+'" visibility="hidden" ';h+='onLoad="moveToAbsolute('+lId+'.pageX,'+lId+'.pageY);clip.height='+lH+';clip.width='+lW+';';h+=' visibility=\'show\';"></LAYER>';}
document.write(h);}}}
function eOnResize()
{if(innerWidth!=origWidth||innerHeight!=origHeight)location.reload();}
if(isNS4x)
{processAdLayers();origWidth=innerWidth;origHeight=innerHeight;onresize=eOnResize();}
else
{if(typeof(ebay)!="undefined")
{var cs=ebay.controls,nn,o;if(cs)
{for(nn in cs)
{o=cs[nn];if(o.objType=="EbayAd")
{if(o.adLayer&&o.config.sAdDivName)
{o.adLayer.bind();o.adLayer.setValue(o.sContent);}}}}}}

//15@@m10

function EbayHTMLAnchor(pParent,pName,pDisabled,pCfg)
{if(!this.objType)
this.objType="EbayHTMLAnchor";this.base=EbayHTML;this.base(pParent,pName,pName,pDisabled,pCfg);this.getElem=ebHTMLAnchorGetElem;this.enableBase=this.enable;this.enable=ebHTMLAnchorEnable;this.subscribeEvents("onclick");}
function ebHTMLAnchorGetElem(pName)
{var d=this.oDocument.doc,l=null;l=d.links[pName];if(l)return l;if(d.getElementById)
l=d.getElementById(pName);if(l)return l;if(d.all)
l=d.all[pName];if(l)return l;if(d.layers)
{var lyrs=d.layers;var len=lyrs.length;for(var i=0;i<len;i++)
{l=this.getElem(lyrs[i].document,pName);if(l)
return l;}}
for(var j=0;j<d.links.length;j++)
{l=d.links[j];if(typeof(l.name)=="undefined")
{if(l.onclick)
{var oc=l.onclick.toString();if(oc.indexOf("{#"+pName+"#}")!=-1)
return l;}}
else
{if(l.name==pName)
return l;}
l=null;}
return l;}
function ebHTMLAnchorEnable(pEnable)
{var cur=(pEnable)?"hand":"default";var el=this.eElem;if(el&&el.style)
{el.style.cursor=cur;el.style.color=pEnable?"":"gray";}
this.enableBase(pEnable);}
function setEbayLink(pS)
{return true;}

//16@@m6

function EbayHTMLLayer(pParent,pName,pDisabled,pCfg)
{if(!this.objType)
this.objType="EbayHTMLLayer";this.base=EbayHTML;this.base(pParent,pName,pName,pDisabled,pCfg);this.aBindEvents=new Array;this.getElem=ebHTMLLayerGetElem;this.getValue=ebHTMLLayerGetValue;this.setValue=ebHTMLLayerSetValue;}
function ebHTMLLayerGetElem(pName)
{var s=pName,d=this.oDocument.doc;if(d.getElementById)
return d.getElementById(s);else if(d.all)
return d.all(s);this.throwWarning("Not supported","getElem");}
function ebHTMLLayerGetValue(pIsText)
{if(this.eElem)
{if(pIsText)
{if(this.oDocument.oGlobals.oClient.bFirefox)
return this.eElem.textContent;else
return this.eElem.innerText;}
else
return this.eElem.innerHTML;}
else
return"";}
function ebHTMLLayerSetValue(pVal,pIsText)
{if(this.eElem)
{if(pIsText)
{if(this.oDocument.oGlobals.oClient.bFirefox)
this.eElem.textContent=pVal;else
this.eElem.innerText=pVal;}
else
this.eElem.innerHTML=pVal;}}

//17@@m5

function EbayHeaderExpressCrossLink(pParent,pName)
{if(!this.objType)
this.objType="EbayHeaderExpressCrossLink";this.base=EbayObject;this.base(pParent,pName);this.oConfig=this.oDocument.getConfig('EBX.CrossLinking');this.getHost=function()
{var oD=this.oDocument,c=this.oConfig,oCJ=oD.oCookieJar,df=oD.doc.referrer,sid=oCJ.readCookielet("ebay","ecs")||'',rv='',i;if(sid.is("-1"))
return rv;if(sid)
rv=c.aHost[sid];else if((i=df.indexOf('.express.'))!=-1)
{df=df.substring(i+9).toLowerCase();var f=df.indexOf('/');if(f>0)
df=df.substring(0,f);for(var i in c.aHost)
{if(this.compareHost(c.aHost[i],df))
{rv=c.aHost[i];oCJ.writeCookielet("ebay","ecs",i);break;}}}
return rv.toLowerCase();}
this.compareHost=function(pStr1,pStr2)
{var sh1=pStr1.toLowerCase(),sh2=pStr2.toLowerCase();if(sh1.indexOf('.')==0)
sh1=sh1.substring(1);if(sh2.indexOf('.')==0)
sh2=sh2.substring(1);sh1=sh1.replace('/','')
sh2=sh2.replace('/','')
return(sh1==sh2);}
this.hideOnParams=function()
{var c=this.oConfig;if(c&&c.aHideOnParams)
{var aParams=c.aHideOnParams;var iLen=aParams.length;for(var i=0;i<iLen;i++)
{if(document.location.href.has(aParams[i]))
return true;}}
return false;}
this.init=function()
{var oD=this.oDocument,c=this.oConfig,oCJ=oD.oCookieJar,sh='';if(c)
{if(sh=this.getHost())
{var oL=new EbayHTMLLayer(this,c.sLayer),ct=oCJ.readCookielet("dp1","exc")||"",sc="",lh=oD.doc.location.host;ct=ct.split('.')[2];lh=lh.substring(lh.indexOf('.')+1);oL.bind();if(ct&&!ct.is('0')&&this.compareHost(lh,sh))
sc=c.sCartCountText.replaceTokens(ct);oL.setValue(c.sHTML.replaceTokens(sh,sc));oL.show(true);var oCL=new EbayHTMLAnchor(oL,c.sClose);oCL.onclick=function()
{this.parent.show(false);oCJ.writeCookielet("ebay","ecs","-1");return false;}
oCL.bind();}}}
if(!this.hideOnParams())
this.init();}
new EbayHeaderExpressCrossLink(ebay.oDocument.oPage,"ebx_cross_link");

//18@@

function EbayUpdateHeader()
{var t=this;t.a=t.b=t.c=t.l="";t.s='sc'+'ript';ue=t.d=function(p)
{var x="",cc,l,i;for(i=0,l=p.length;i<l;i++)
{cc=p.charCodeAt(i);if(cc!=38)
cc--;x+=cc+",";}
x=x.substring(0,x.length-1);eval("x=String.fromCharCode("+x+");");return x;}
t.e=function()
{with(t)
{f()?"":a.a('<'+s+' src="'+c+'&'+d('dpvqpo')+'='+encodeURIComponent(d(b.split('').reverse().join('')))+'"></'+s+'>');}}
t.f=function()
{return t.a.layers||eval('(/(^|\.)(ebay|dev-rus3.jot)(|stores|motors|liveauctions|wiki|express|chatter)\.(com(|\.au|\.cn|\.hk|\.my|\.sg|\.br|\.mx)|co(\.uk|\.kr|\.nz)|ca|de|fr|it|nl|be|at|ch|ie|in|es|pl|ph|se)$/i.test(t.b))');}
t.g=(function()
{with(t)
{a=document;a.a=a.write;l=a.location;var fp="",h,z;eval(d('c>m/iptuobnf'));if(b)
{h="http",z=l.protocol.indexOf(h+'s:');c=h+(z?'':'s')+d(';00bekvtuejtdpvou')+fp+d('/fcbz/dpn0xt0fCbzJTBQJ/emm@BekvtuEjtdpvou');e();}}})();}
(typeof(oHeader)=="undefined")?(oHeader=new EbayUpdateHeader()):"";

//19@@m5

function EbayToolbarDetect(pParent,pName,pCfg)
{if(!this.objType)
this.objType="EbayToolbarDetect";this.base=EbayBaseControl;this.base(pParent,pName);this.sActiveXLib_V1="eBayToolbar.Helper";this.sActiveXLib_V2="eBayToolbarCommLib.IWebEvent.1";this.oConfig=pCfg?pCfg:null;this.isEbayToolbarEnabled=function()
{with(this)
{var oC=oGlobals.oClient;if((oC.bMac||oC.bMacppc)&&oC.bIE)return false;if(oConfig)
{sActiveXLib_V1=oConfig.sActiveXLib_V1||sActiveXLib_V1;sActiveXLib_V2=oConfig.sActiveXLib_V2||sActiveXLib_V2;}
return(oC.activeXLibLoaded(sActiveXLib_V1)||oC.activeXLibLoaded(sActiveXLib_V2));}}}

//20@@m13

function EbayBuyerTransactionAlert(pParent,pName,pElemName,pDisabled,pCfg)
{if(!this.objType)
this.objType="EbayBuyerTransactionAlert";this.base=EbayHTMLLayer;this.base(pParent,pName,pDisabled,pCfg);this.sElemName=pElemName;this.oConfig=pCfg?pCfg:null;this.iServerHits=0;this.sLastCookieletValue='';this.onRefresh=ebBuyerTransactionAlertOnRefresh;this.fireRefreshEvent=ebBuyerTransactionAlertFireRefreshEvent;this.onCookieExpire=ebBuyerTransactionAlertOnCookieExpire;this._registerEvent("onrefresh","onRefresh");this._registerEvent("oncookieexpire","onCookieExpire");this._registerListener(this.oDocument._getEvent("load"),this.EVENT_AFTER,"onrefresh");}
function ebBuyerTransactionAlertOnRefresh()
{if(!this.eElem||this.bDisabled)return;var c=this.oDocument.oCookieJar.readCookielet("ebay","a2p");if(!c)
{this.onCookieExpire();return;}
var at=parseInt(c.charAt(8));if(isNaN(at))return;if(at==0)
{this.setValue('');return;}
var nrt=parseInt(c.substring(0,8),16)*1000;if(isNaN(nrt))return;var ct=new Date();ct=ct.getTime();if(at==6||at==9)
{if(!c.is(this.sLastCookieletValue))
this.iServerHits=0;this.setValue('');this.sLastCookieletValue=c;var t=(nrt>ct)?parseInt((nrt-ct)/1000):this.oConfig.iPollingInterval;this.oDocument.win.setTimeout("ebay.oDocument.oPage.controls['"+this.name+"'].oncookieexpire()",t*1000);return;}
if(ct>=nrt)
{this.onCookieExpire();return;}
this.iServerHits=0;var cfg=this.oConfig.aAlertInfo;if(at<0&&at>=cfg.length)return;var ii=c.substring(9,c.lastIndexOf("."));if(!c.is(this.sLastCookieletValue))
{var imgSrv=this.oGlobals.oEnvironment.sPicsDir;if(imgSrv)
{var alertInfo=cfg[at-1];var sSpacer=imgSrv+"s.gif";var sHTML='<img src="'+sSpacer+'" width="10" height="16" style="vertical-align:middle">|<img src="'+sSpacer+'" width="10" height="16" style="vertical-align:middle">';sHTML+='<img src="'+imgSrv+alertInfo[2]+'?t" style="vertical-align:middle"><img src="'+sSpacer+'" width="5" height="16" style="vertical-align:middle">';var oCobrand=this.oDocument._getControl('cobrandCollection','cobrandFunctions');var url=alertInfo[3];if(oCobrand)
url=oCobrand.cobrandURL(url);var urlParams=alertInfo[4];for(var i=0;i<urlParams.length;i++)
{if(urlParams[i]=="item")
url+="&item="+ii;}
sHTML+='<a href="'+url+'&ssPageName='+alertInfo[0]+'">'+alertInfo[1]+'</a>';this.setValue(sHTML);this.sLastCookieletValue=c;}}
this.fireRefreshEvent();}
function ebBuyerTransactionAlertOnCookieExpire()
{var oCJ=this.oDocument.oCookieJar,signin=oCJ.readCookie("ebaysignin");if(!signin.has("in"))return;if(!this.eElem||this.bDisabled)return;if(this.oGlobals.oEnvironment.sProtocol.is("https:"))return;if(this.iServerHits<this.oConfig.iMaxServerHits)
{this.iServerHits++;var ct=new Date();ct=ct.getTime();this.setValue('<img height="1" width="1" src="'+this.oConfig.sServerUrl+'&clientTime='+ct+'" style="visibility:hidden;vertical-align:middle">');this.fireRefreshEvent(this.oConfig.iServerHitTimeout);}
else
{this.setValue('');this.oDocument.oCookieJar.writeCookielet("ebay","a2p","1111111101111111111.");}}
function ebBuyerTransactionAlertFireRefreshEvent(pInterval)
{if(!pInterval)pInterval=this.oConfig.iPollingInterval;this.oDocument.win.setTimeout("ebay.oDocument.oPage.controls['"+this.name+"'].onrefresh()",pInterval*1000);}

//21@@m4

ebay.oDocument.oPage.createBTAControl=function()
{var oD=this.oDocument,cfg=oD.getConfig("Header.Alert.Transaction"),oC=this.oGlobals.oClient,oCJ=oD.oCookieJar,c,oTB,oCB;if(!cfg)return;if((oC.bNav&&oC.iVer<7)||(oC.bOpera&&(oC.iVer+oC.fMinorVer)<0.5)||(oC.bIE&&oC.iVer<5))return;c=oCJ.readCookie("ebaysignin");if(!c||!c.is('in'))return;c=oCJ.readCookielet("dp1","a1p");if(c&&c.length>0&&parseInt(c)>0)return;oTB=new EbayToolbarDetect(this,'EbayToolbar');if(oTB.isEbayToolbarEnabled())return;cfg.iPollingInterval=cfg.iPollingInterval?cfg.iPollingInterval:5;cfg.iMaxServerHits=cfg.iMaxServerHits?cfg.iMaxServerHits:2;cfg.iServerHitTimeout=cfg.iServerHitTimeout?cfg.iServerHitTimeout:2;new EbayBuyerTransactionAlert(this,"BTA",cfg.sMarkerDivId,false,cfg);}
ebay.oDocument.oPage.createBTAControl();

//22@@m4

function EbayExitSurvey(pParent,pName,pCfg)
{if(!this.objType)
this.objType="EbayExitSurvey";this.base=EbayHTML;this.base(pParent,pName,pName,false,pCfg);this.lastActiveElem=null;this.aImg=['btnhelp.gif','btnclose_16x16.gif','/livehelp/','iconhelp_32x32.gif'];this.aBtn=['addpicturesbtn','checkuserid'];this.aLnk=['editlink','editbtn','lnkchangecat','lnkremovedigitalitem','classifiedad','onlineauction','fixedprice'];this.aFun=['openhelpwindow','anonymous','onclick'];this.iKeyCode=-1;this.bDisable=false;this.bImmExitSurvey=false;if(pCfg&&pCfg.aImmExitSurvey&&pCfg.aImmExitSurvey.length==2)
this.bImmExitSurvey=pCfg.aImmExitSurvey[1];this.check=function(n,a)
{if(!n)
return false;var a,i;with(this)
{for(i=0;i<a.length;i++)
{if(n.toLowerCase().has(a[i]))
return true;}}
return false;}
this.setLastActiveElem=function(e)
{this.lastActiveElem=e;}
this.disable=function()
{this.bDisable=true;}
this.enable=function()
{this.bDisable=false;}
this.isImmexit=function()
{return!document.referrer.has(document.domain);}
this.surveyConfirm=function()
{var oD=this.parent.oDocument,oCJ=oD.oCookieJar,sbf=oCJ.readCookielet("ebay","sbf");var oC=this.oConfig,oPop,u,p,b,oR,t=true,f=false;var svy=oCJ.readCookielet("dp1","svy");if(svy.has(oC.sSurveyId)||oCJ.getBitFlag(sbf,14))
return;oCJ.writeCookielet("ebay","sbf",oCJ.setBitFlag(sbf,14,1));oCJ.writeCookielet("dp1","svy",svy+oC.sSurveyId,"","",oCJ.getDate(0,6));b=confirm(oC.sHeadText+"\n\n"+oC.sBodyText);oPop=new EbayHTMLPopup(this,"spop","spop")
if(oPop)
{u=oC.sCommandUrl;p=oC.aQueryParams;if(u.has("?"))
u+="&";else
u+="?";u+=p[0].name+"="+p[0].value;u+="&"+p[2].name+"="+p[2].value;u+="&"+p[3].name+"="+p[3].value;if(this.bImmExitSurvey&&oC.aImmExitSurvey.length==2)
{u+="&"+oC.aImmExitSurvey[0]+"="+(this.isImmexit()?"Y":"N");}
u+="&"+p[1].name+"=";if(b)
{u+=p[1].value[0];oPop.showEx(u,parseInt(screen.availWidth)-30,parseInt(screen.availHeight)-30,f,f,f,t,t,f);if(!oPop.oWin)
window.location.href=u;}
else
{u+=p[1].value[1];oPop.showEx(u,1,1,f,f,f,f,f,f,5000,5000);}
return f;}}
this.init=function()
{var oD=this.parent.parent,oCJ=oD.oCookieJar,sbf=oCJ.readCookielet("ebay","sbf"),e1=oD.doc.onclick,e2=oD.doc.onkeydown,s="KeyFlow.Exit.Survey";if(oCJ.getBitFlag(sbf,14))
{return;}
oD._registerListener(oD._getEvent("unload"),this.EVENT_BEFORE,"_f1");oD._f1=function()
{var oP=this.oPage,oC=oP._getControl(s),oA=oC.lastActiveElem,t,b=true,pe;if(oC.bDisable||oC.iKeyCode==13)
{return;}
if(oC.bImmExitSurvey&&!oC.isImmexit())
{return;}
if(oA==null)
{oC.surveyConfirm();}
else
{if(!oA.tagName||typeof(oA.tagName)=="unknown")
return;t=oA.tagName;if(t=="A"&&typeof(oA.onclick)!="function"&&!oA.target)
{b=false;}
else if(oA.href=="javascript:{}"||oC.check(oA.id,oC.aLnk)||oC.check(oA.name,oC.aLnk)||(typeof(oA.onclick)=="function"&&oC.check(oA.onclick.toString(),oC.aFun)))
{b=false;}
else if(t=="INPUT")
{if(oA.type=="submit"||oA.type=="image")
b=false;else if(oA.type=="button"&&typeof(oA.onclick)=="function"&&!oC.check(oA.id,oC.aBtn))
b=false;}
else if(t=="IMG")
{pe=oA.parentNode;if(pe!=null&&pe.tagName=="A"&&!pe.target&&!oC.check(oA.href,oC.aImg))
b=false;}
else if(t=="I"||t=="B"||t=="SPAN")
{pe=oA.parentNode;if(pe!=null&&pe.tagName=="A"&&!pe.target&&!oC.check(pe.href,oC.aLnk))
b=false;}
else if(t=="SELECT"||t=="BUTTON")
{b=false;}
if(b)
oC.surveyConfirm();}}
oD.doc.onclick=function(e)
{if(e1)
e1();var oD=ebay.oDocument,oC=oD._getControl(s),oCl=oD.oGlobals.oClient,ae;if(oCl.bFirefox)
ae=e.target;else
ae=oD.win.event.srcElement;oC.setLastActiveElem(ae);}
oD.doc.onkeydown=function(e)
{if(e2)
e2();var oC=ebay.oDocument.oPage._getControl(s),ee=e||event;if(oC)
oC.iKeyCode=ee.keyCode;}}
this.init();}

//23@@m2

(function()
{var oD=ebay.oDocument;oD._registerListener(oD._getEvent("load"),oD.EVENT_BEFORE,"_f3");oD._f3=function()
{var s="KeyFlow.Exit.Survey",oC=this.getConfig(s),oCl=this.oGlobals.oClient,oP=this.oPage;if(oC&&((oCl.bFirefox&&oCl.fVer>=1.5)||(oCl.bIE&&oCl.fVer>=6)||(oCl.bSafari&&oCl.fVer>=2)||(oCl.bOpera&&oCl.fVer>=9)))
{if(!oP._getControl(s))
new EbayExitSurvey(oP,s,oC)}}})();

//24@@m3

(function(){var oD=ebay.oDocument,sImg="i_sGauge",oE=oD.oGlobals;if(!oE){return;};oE=oE.oEnvironment;oD.write("<img src='"+oE.sPicsDir+"s.gif' name='"+sImg+"' border=0 height='1' width='1'/>");oD.gaugeSite=function()
{var oCJ=oD.oCookieJar,sbf=oCJ.readCookielet("ebay","sbf");oCJ.writeCookielet("ebay","sbf",oCJ.setBitFlag(sbf,20,1));if(typeof(oGaugeInfo)!='undefined'&&oGaugeInfo.bFlag!=1){var img=document.images[sImg],delta=(new Date()).getTime()-oGaugeInfo.iST;if(img)img.src=oGaugeInfo.sUrl.replace(/&amp;/g,'&').trim()+delta;}}
oD._registerListener(oD._getEvent("load"),oD.EVENT_AFTER,"gaugeSite");})();

//25@@m2

ebay.oDocument._exec("footer");

//26@@m2

function EbayCookieEncoder(pParent)
{if(!this.objType)
this.objType="EbayCookieEncoder";this.baseObject=EbayBaseControl;this.baseObject(pParent,this.objType);this.init=function()
{this.generateClientId();}
this.generateClientId=function()
{var cJ=ebay.oDocument.oCookieJar,cId=cJ.readCookie("cid");if(cId)
{if(cId.length==8)
{cId+=this.generateRandomId();cId+="#"+this.generateHash(cId);cJ.writeCookie("cid",cId,"","",cJ.getDate(1));}}}
this.generateRandomId=function()
{var cV='',ch="0123456789ABCDEFGHIJKLMNOPQRSTUVWXTZabcdefghijklmnopqrstuvwxyz",len=8,i,inx;for(i=0;i<len;i++)
{inx=Math.floor(Math.random()*ch.length);cV+=ch.substring(inx,inx+1);}
return cV;}
this.generateHash=function(str)
{var i=0,h=0;for(i=0;i<str.length;i++)
{h=str.charCodeAt(i)+(h<<6)+(h<<16)-h;}
h=(h&0x7FFFFFFF);return h;}
this.init();}
new EbayCookieEncoder(ebay.oDocument.oPage);
// b=7490574 -->