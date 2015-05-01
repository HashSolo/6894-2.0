//<!--
//\include\js\cookies.js@@\main\25

function EbayCookie(name,value,domain,path,expires,secure,isCookielet)
{if(!this.objType)
this.objType="EbayCookie";if(typeof(EBayObject)=="function")
{this.base=EBayObject;this.base(null,name);}
else
{this.parent=null;this.name=name;}
this.value=value;this.value=value||"";if(typeof(thisPageURL)!="undefined"&&domain==null)
{if(!thisPageURL.contains(qaDom))
{this.domain=countryDomain.substring(0,countryDomain.length-1);d=this.domain;if(d.contains("es.ebay.")||d.contains("tw.ebay."))
{this.domain=".ebay.com";}
if(d.contains("befr.ebay.")||d.contains("benl.ebay."))
{this.domain=".ebay.be";}}
else
{this.domain=".ebay.com";}}
else
{this.domain=domain;}
this.path=path?path:"/";this.secure=secure;this.isCookielet=isCookielet;this.cookielets=new Array();this.expires=expires;this.parseCookie=EbayParseCookie;this.addCookielet=EbayAddCookieLet;this.read=EbayReadCookie;this.set=EbaySetCookie;this.getCookielet=EbayGetCookielet;}
function EbayCookielet(name,value,expi)
{this.name=name;this.value=value?value:'';expi=expi?expi:'';var d=expi;if(typeof(expi)=="object")
{this.expirationDate=expi;y=d.getUTCFullYear();if(y<1900)
{y=y+1900;}
expi=Date.UTC(y,d.getUTCMonth(),d.getUTCDate());}
if(isNaN(expi)&&!expi=="")
{expi=d.hex2Dec();this.expirationDecBy1000=expi;expi=expi*1000;this.expirationDate=new Date(expi);this.expirationDec=expi;this.expirationHex=this.expirationDecBy1000.dec2Hex();}
else
{this.expirationDec=expi;this.expirationDecBy1000=Math.floor(expi/1000);this.expirationHex=this.expirationDecBy1000.dec2Hex();}}
function EbayGetCookielet(name)
{var c=this.cookielets[name],v="";if(c)v=c.value;return v;}
function EbayParseCookie(start)
{var mycookielets=new Array();var strCookielets="";if(start)
{var input=this.value;mycookielets=input.split("^");for(i in mycookielets)
{var x=mycookielets[i].split("=");if(x[0]!=""&&x[1]!="")
{var outVal=x[1];for(var j=2;j<x.length;j++)
{outVal+="="+x[j];}
this.addCookielet(x[0],outVal);}}}
else
{strCookielets="^";var cl=this.cookielets;for(i in cl)
{if(cl[i])
strCookielets+=cl[i].name+"="+cl[i].value+"^";}
this.value=strCookielets;}}
function EbayReadCookie()
{var cookie_list=' '+document.cookie,cookie_key=' '+this.name+'=',cookie_offs=(document.cookie?cookie_list.indexOf(cookie_key):-1);if(cookie_offs>=0)
{var cookie_val=cookie_list.substring(cookie_offs+cookie_key.length),cookie_end=cookie_val.indexOf(';');if(this.name!="dp1")
this.value=unescape(cookie_end<0?cookie_val:cookie_val.substring(0,cookie_end));else
this.value=cookie_end<0?cookie_val:cookie_val.substring(0,cookie_end);if(this.isCookielet)
{this.parseCookie(true);}}}
function EbaySetCookie()
{with(this)
{if(isCookielet)
{parseCookie(false);}
if((isNaN(value)&&value.length<4000)||(value+'').length<4000)
{document.cookie=(name?name+'=':'')+(value?escape(value):'')+(domain?'; domain='+domain:'')+(path?'; path='+path:'')+(expires?'; expires='+expires.toGMTString():'')+(secure?'; secure='+escape(secure):'');}
else
{if(isQA)
alert("ERROR: Cookie data could not be set over 4kb.");}}}
function EbayAddCookieLet(name,value,expiration)
{this.cookielets[name]=new EbayCookielet(name,value,expiration);}
function EbayOldWriteCookie(name,value,domain,path,expires,secure)
{if(expires=='yes')
{expires=new Date("Jan 02, 2004 12:00:00 GMT");}
else if(expires=='delete')
{expires=new Date("May, 01 1999 12:00:00 GMT");}
var x=new EbayCookie(name,value,domain,path,expires,secure,false);x.set();}
window.WriteCookie=EbayOldWriteCookie;function EbayOldWrite2YearCookie(name,value,domain,path,expires,secure)
{var x;if(!expires)
{var offset=2;now=new Date();year=now.getYear();now.setYear(year+offset);expires=now.valueOf();}
x=new EbayCookie(name,value,domain,path,expires,secure,false);}
window.Write2YearCookie=EbayOldWrite2YearCookie;function EbayOldReadCookie(name)
{var x=new EbayCookie(name);x.read();return x.value;}
window.ReadCookie=EbayOldReadCookie;function EbayOldReadMultiCookie(cook,cooklet)
{var x=new EbayCookie(cook,null,null,null,null,null,true);x.read();return x.getCookielet(cooklet);}
window.readMultiCookie=EbayOldReadMultiCookie;function EbayOldWriteMultiCookie(name,cooklet,value,domain,path,expires)
{var x=new EbayCookie(name,null,domain,path,expires,null,true);x.read();x.addCookielet(cooklet,value);x.set();}
window.writeMultiCookie=EbayOldWriteMultiCookie;function EbayOldDeleteCookie(name,domain,path)
{var expireDate=new Date(1);var x=new EbayCookie(name,null,domain,path,expireDate,null,false);x.set();}
window.DeleteCookie=EbayOldDeleteCookie;function EbayDPCookie(name,value,domain,path,expires,secure,isCookielet)
{this.base=EbayCookie;this.base(name,value,domain,path,expires,secure,isCookielet);this.parseCookie=EbayDPParseCookie;this.set=EbaySetDPCookie;this.getCookieletObj=EbayGetCookieletObj;this.isCookielet=true;if(!expires)
{var offset=2;now=new Date();year=now.getUTCFullYear();if(year<1900)
{year=1900+year;}
now.setUTCFullYear(year+offset);this.expires=now;}
else
{this.expires=expires;}}
function EbaySetDPCookie()
{with(this)
{if(isCookielet)
{parseCookie(false);}
if((isNaN(value)&&value.length<4000)||(value+'').length<4000)
{document.cookie=(name?name+'=':'')+(value?value:'')+(domain?'; domain='+domain:'')+(path?'; path='+path:'')+(expires?'; expires='+expires.toGMTString():'')+(secure?'; secure='+escape(secure):'');}
else
{if(isQA)
alert("ERROR: Cookie data could not be set over 4kb = cookie length = "+value.length);}}}
function EbayGetCookieletObj(name)
{var c=this.cookielets[name];return c;}
function EbayReadDPCookielet(cook,cooklet)
{var x=new EbayDPCookie(cook,null,null,null,null,null,true);x.read();return x.getCookielet(cooklet);}
function EbayDPWriteMultiCookie(name,cooklet,value,domain,path,expires)
{var today=(new Date()).getTime();var defexp=new Date(today+(YEAR*1))
var x=new EbayDPCookie(name,null,domain,path,defexp,null,true);x.read();x.addCookielet(cooklet,value,expires);x.set();}
function EbayDPParseCookie(start)
{var mycookielets=new Array();var strCookielets="";if(start)
{var input=this.value;input=input.substring(1,input.length);mycookielets=input.split("^");for(i in mycookielets)
{c=mycookielets[i];expi=c.substring(c.length-8,c.length);c=c.substring(0,c.length-8);var x=c.split("/");if(x[0]!=""&&x[1]!="")
{var outVal=unescape(c.substring(c.indexOf("/")+1,c.length));this.addCookielet(x[0],outVal,expi);}}}
else
{strCookielets="";var today=new Date();var cl=this.cookielets;for(i in cl)
{if(cl[i])
{var timeToExpire=cl[i].expirationDate-today;if(timeToExpire>0)
strCookielets+=cl[i].name+"/"+escape(cl[i].value)+cl[i].expirationHex+"^";}}
if(strCookielets!="")
{this.value="b"+strCookielets;}
else
{this.value="";}}}
var sConvCookieList="recent_vi,ebaysignin,keepmesignin,item_list,back_to_list,reg,p,etfc,history_item";var sNewMapping="recent_vi=ebay,lvmn^ebaysignin=ebay,sin^reg=dp1,reg^p=dp1,p^etfc=dp1,etfc^keepmesignin=dp1,kms^ItemList=ebay,wl^BackToList=s,BIBO_BACK_TO_LIST";var aNewMapping;var aModes;var COMPAT="10";var CONVER="01";var STRICT="00";function writeCookieEx(c,v,d,p,e,s)
{var m=getMode(c);if(m==STRICT||m==CONVER)
{e=new Date(e);writeCookieletEx(getDest(c,false),getDest(c,true),v,d,p,e);}
else
{EbayOldWriteCookie(c,v,d,p,e,s);}}
function readCookieEx(c)
{var m=getMode(c);var v=nv="";v=EbayOldReadCookie(c);nv=readCookieletEx(getDest(c,false),getDest(c,true));if(m==COMPAT)
{v=v?v:nv;}
else if(m==CONVER)
{v=nv?nv:v;}
else if(m==STRICT)
{v=nv;}
if(c=="dp1")
v=unescape(v);return v;}
function getDest(n,bCL,sM)
{sM=sNewMapping;if(!aNewMapping)
{aNewMapping=getSets(sM);}
if(aNewMapping[n])
return bCL?aNewMapping[n][1]:aNewMapping[n][0];return"";}
function getSets(p)
{var a=new Array();var ma=p.split("^");for(i in ma)
{var t=ma[i].split("=");if(t[0]!=""&&t[1]!="")
{a[t[0]]=t[1].split(",");}}
return a;}
function readCookieletEx(c,cl)
{var m=getMode(c);var v;if(m==STRICT||m==CONVER)
{var nc=getDest(c,false);var ncl=getDest(c,true);v=getCookieletValue(nc,ncl);v=pareseReg(c,cl,nc,ncl,v);}
else
{v=getCookieletValue(c,cl);}
return v;}
function getCookieletValue(c,cl)
{var cls;if(c=="dp1")
cls="EbayDPCookie";else
cls="EbayCookie";var co=eval("new "+cls+"(c, null, null, '/', null, null, true)");co.read();v=co.getCookielet(cl);return v?v:"";}
function writeCookieletEx(c,cl,v,d,p,e)
{var cls;var m=getMode(c);if(m==STRICT||m==CONVER)
{nc=getDest(c,false);ncl=getDest(c,true);nv=readCookieletEx(nc,ncl);v=buildReg(c,cl,nc,ncl,v,nv);c=nc;cl=ncl;}
if(c=="dp1")
cls="EbayDPCookie";else
cls="EbayCookie";var co=eval("new "+cls+"(c, null, d, p, null, null, true)");co.read();co.addCookielet(cl,v,e);co.set();}
function getMode(c)
{if(!aModes)
fillModes()
t=aModes[c];return(t?t:"");}
function fillModes()
{var h=EbayOldReadMultiCookie("ebay","cv");cL=sConvCookieList.split(",");aModes=new Array();for(i in cL)
{aModes[cL[i]]=(h=="0")?STRICT:"";}
if(h&&h!="0")
{if(h.indexOf(".")!=-1)
{a=h.split(".");for(i in a)
{b=a[i].hex2Dec().toString(2)+b;}}
else
{b=h.hex2Dec().toString(2);}
i=0;l=b.length;for(o in aModes)
{j=l-(2*(i+1));f=b.substring(j,j+2).toString(10);f=(!f)?STRICT:f;aModes[o]=(f.length==1)?"0"+f:f;i++;}}}
function deleteCookieEx(c,d,p)
{writeCookieEx(c,null,d,p,new Date(1),false);}
function pareseReg(c,cl,nc,ncl,v)
{if(c=="reg")
{if(v.indexOf("=")!=-1&&v.indexOf("^")!=-1)
{var a=getSets(v);if(a[cl])
return a[cl][0];}
return"";}
else
return v;}
function buildReg(c,cl,nc,ncl,v,nv)
{if(c!="reg")
return v;var r="";if(nv.indexOf("=")!=-1&&nv.indexOf("^")!=1)
{var a=getSets(nv);if(!a[cl])
a[cl]=new Array();a[cl][0]=v;for(var m in a)
{r+=m+"="+a[m][0]+"^";}}
else
{r=cl+"="+v+"^";}
return"^"+r;}
window.WriteCookie=writeCookieEx;window.ReadCookie=readCookieEx;window.readMultiCookie=EbayReadDPCookielet=readCookieletEx;window.writeMultiCookie=writeCookieletEx;window.DeleteCookie=deleteCookieEx;window.Write2YearCookie=function(c,v,d,p,e,s)
{if(!e)
{td=new Date();y=td.getYear();td.setYear(y+2);e=td.valueOf();}
writeCookieEx(c,v,d,p,e,s)}
var CookieDomain=".ebay.com";var thisHost=location.hostname.toLowerCase();if(thisHost.indexOf(".ebay")>0)
CookieDomain=thisHost.substring(thisHost.indexOf(".ebay"));
// -->
