//<!--
//\include\js\common_functions.js@@\main\79

String.prototype.contains=function(pWanted){return(this.indexOf(pWanted)!=-1);}
String.prototype.containsAny=function(){var args=arguments,len=args.length,rv=false;for(var i=0;i<len&&!rv;i++)
rv=(this.indexOf(args[i])!=-1);return rv;}
String.prototype.containsAll=function(){var args=arguments,len=args.length;for(var i=0;i<len;i++)
{if(this.indexOf(args[i])==-1)
return false;}
return true;}
String.prototype.is=function(s){return(this==s);}
String.prototype.isAny=function(){var args=arguments,len=args.length,rv=false;for(var i=0;i<len&&!rv;i++)
rv=(this==args[i]);return rv;}
Number.prototype.dec2Hex=function(){return parseInt(this,10).toString(16)}
String.prototype.hex2Dec=function(){return parseInt(this,16);}
String.prototype.replaceTokens=function()
{var value=this;var re;var token;for(var i=0;i<arguments.length;i++)
{token="<#"+(i+1)+"#>";re=new RegExp(token,"gi");if(value.indexOf(token)!=-1)
{value=value.replace(re,arguments[i])}}
return value;}
function ebayStringContains(pSource,pWanted)
{return(pSource.indexOf(pWanted)!=-1);}
window.stringContains=ebayStringContains;function ebayEvalParameters(pStr)
{var str=location.search;if(pStr)
str=pStr;var ps=str.substr(1).split("&");for(var i=0;i<ps.length;i++)
{var p=unescape(ps[i]).split("=");if(country.isAny("tw","cn","hk"))
p=ps[i].split("=");if(p[1])
{var eStr=p[0]+"=";if((p[1]=="true")||(p[1]=="false")||(p[1]=="null"))
eStr+=p[1]+";";else
{var tmp="";for(var j=0;j<p[1].length;j++)
{if(p[1].charAt(j)=="'")
tmp+="\\'";else if(p[1].charAt(j)=="+")
tmp+=" ";else
tmp+=p[1].charAt(j);}
p[1]=tmp;eStr+="'"+p[1]+"';";}
eval(eStr);}}}
function ebayUpdateLocationVars()
{var qInd=thisPage.indexOf(qstn);if(qInd==-1)
{thisPageURL=thisPage;thisPageURI="";}
else
{thisPageURL=thisPage.substring(0,qInd);thisPageURI=thisPage.substring(qInd);}
qInd=lastPage.indexOf(qstn);if(qInd==-1)
{lastPageURL=lastPage;lastPageURI="";}
else
{lastPageURL=lastPage.substring(0,qInd);lastPageURI=lastPage.substring(qInd);}
var qaCountryDomain=countryDomain,c;country="";for(var i=0;i<countries.length&&(country=="");i++)
{c="."+countries[i][2]+".";if(thisPageURL.containsAny(c,countries[i][0],countries[i][1]))
{countryDomain=countries[i][0];country=countries[i][2];countryDesc=countries[i][3];qaCountryDomain=(c==".us.")?".":c;}}
var isSecure=false;var isQA=thisPageURL.contains(qaDom);if(thisPageURL.indexOf(https)==0)
{isSecure=true;ebInclude="secure"+ebInclude;picsDir="secure"+picsDir;}
if(isQA)
{var qaInd=includeHost.indexOf(qaDom);var dotInd=includeHost.lastIndexOf(dot,qaInd-1)+1;if(dotInd!=0)
qaMachineName=includeHost.substring(dotInd,qaInd);countryDomain=(qaMachineName=="")?"":qaCountryDomain;countryDomain+=qaMachineName+qaDom+ebStr+dotCom+"/";}
var protocol=thisPageURL.substring(0,thisPageURL.indexOf("://")+3);if(!protocol.isAny(http,https))
protocol=http;picsDir=protocol+picsDir;includeDir=protocol+ebInclude+dot;if(isQA)
{if(qaMachineName!="")
includeDir+=qaMachineName+dot;includeDir+="qa"+dot;}
includeDir+=ebStr+"static"+dotCom+"/aw/pics/";cobrandDir=includeDir+"js/cobrand/";countryIncludeDir=includeDir+country+"/js/";countryCobrandDir=countryIncludeDir+"cobrand/";includeDir+="js/";countryDoubleByte=country.isAny("tw","cn","hk");}
function ebayDowngradeDomain()
{var eInd=document.domain.indexOf(ebDom+dot);if(eInd!=-1)
document.domain=new String(document.domain).substr(eInd+1);}
function ebayShowPopupWindow(url,name,width,height,toolbar,location,status,scrollbars,resizable,menubar,left,top,customprops)
{var props="";if(width)props+=",width="+width;if(height)props+=",height="+height;if(toolbar)props+=",toolbar="+toolbar;if(location)props+=",location="+location;if(status)props+=",status="+status;if(scrollbars)props+=",scrollbars="+scrollbars;if(resizable)props+=",resizable="+resizable;if(menubar)props+=",menubar="+menubar;if(left)props+=",screenX="+left+",left="+left;if(top)props+=",screenY="+top+",top="+top;if(customprops)props+=","+customprops;if(props!="")props=props.substring(1);var w=window.open(url,name,props);if(!is.opera&&w&&!w.closed)w.focus();return w;}
function ebaySignInOut(signin,signout,sspagesignin,sspagesignout)
{if(typeof(readCookieEx)!='undefined')
{var url=http+'signin'+countryDomain+'/ws2/eBayISAPI.dll?SignIn';url+='&ssPageName=';var ssp=sspagesignin,sign=signin;if((readCookieEx("ebaysignin")==seoCookIn)||(readCookieEx("keepmesignin")==seoCookIn))
{ssp=sspagesignout;sign=signout;}
var str=seoAOpen+url+ssp+'">'+seoFontOpen+sign;document.write(str+seoFontClose);}}
window.eBaySignInOut=ebaySignInOut;function ebayRegister(text,sspage)
{if(typeof(readCookieEx)!='undefined')
{if((readCookieEx("ebaysignin")!=seoCookIn)||(readCookieEx("keepmesignin")!=seoCookIn))
{var str=seoAOpen+http+'cgi1'+countryDomain+'/aw-cgi/eBayISAPI.dll?SignIn';str+='?RegisterShow&ssPageName="'+sspage+'">'+seoFontOpen;document.write(str+text+seoFontClose);}}}
window.eBayRegister=ebayRegister;function ebayGetUnencodedHost(pURL)
{var dl=document.location,tp=unescape(dl);if(tp.contains("_W0QQ"))
{if(tp.split("/").length==4)
{tp=tp.replace(/_W0QQ/,"?");tp=tp.replace(/Z/g,"=");tp=tp.replace(/Q3A/g,":");tp=tp.replace(/Q3a/g,":");tp=tp.replace(/Q3B/g,";");tp=tp.replace(/Q3b/g,";");tp=tp.replace(/QQ/g,"&");}}
return tp;}
function ebayHelpLoadStrings()
{var str=scriptOpen+countryIncludeDir+"help/help_strings.js?x=0";document.write(str+scriptClose);}
function ebayLoadStandardStyleSheet()
{var s='ebay';if(country.isAny('cn'))
{if(typeof(nodefaultcss)!="undefined")
s+='-nodefault';s+='-mb';}
else if((typeof(nodefaultcss)!="undefined")||thisPageURL.containsAny("forums.","chatboards","answercenter","http://vi.","http://ivi."))
s+='-nodefault';if(country.isAny('tw'))
s+='-tw';else if(is.nav||is.firefox||(document.compatMode&&document.compatMode!="BackCompat"&&thisPageURL.containsAny("forums.","chatboards")))
s+='-ns';document.write('<link rel="stylesheet" type="text/css" href="'+includeHost+'aw/pics/css/'+s+'.css"/>');}
var globals;if(globals)
{ebayUpdateLocationVars();ebayLoadStandardStyleSheet();if(thisPageURL.contains("/help/"))
ebayHelpLoadStrings();}
function ebayBustFrames(){var t=top.location;var w=window.location;if(t!=w)
t.replace(w);}
window.bustFrames=ebayBustFrames;(function()
{if(typeof(oHeaderCheck)=="undefined"&&typeof(oHeader)=="undefined")
{var d=document,l=document.location,pro=l.protocol;var fp="";if(pro.indexOf("http")==-1)
pro="http:";ip=pro+"//";if(pro=='https:')ip+="secure";ip+='include'+fp+'.ebaystatic.com/';if(d.getElementById||d.all)
{d.write('<sc'+'ript language="javascript" src="'+ip+'js/v/us/wt/base.js"></sc'+'ript>');window.oHeaderCheck=true;}}})();
// -->
