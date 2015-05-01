//<!--
//\include\js\stats\_base\ssObjs.js@@\main\15

var ebSCObject=null;function EBaySiteCatalystFilesObj(pParent,pName)
{if(!this.objType)
this.objType="EBaySiteCatalystFilesObj";if(typeof(EBayObject)=="function")
{this.base=EBayObject;this.base(pParent,pName);}
else
{this.parent=pParent;this.name=pName;}
this.files=new Array;var sd=this.sourceDir=includeHost+"aw/pics/js/stats/";this.baseDir=sd+"_base/";var fd=this.featureDir=sd+"features/";this.pageNamesDir=fd+"pageNames/";this.propertyReportsDir=fd+"propertyReports/";this.eventsDir=fd+"events/";this.cookiesDir=fd+"cookies/";this.add=ebSCFOAddFiles;}
function ebSCFOAddFiles()
{var args=arguments,aLen=args.length,fs=this.files;for(var i=0;i<aLen;i++)
{var found=false,len=fs.length;for(var j=0;j<len&&!found;j++)found=(fs[j]==args[i]);if(!found)fs[len]=args[i];}}
function EBaySiteCatalystObj(pParent,pName)
{if(!this.objType)
this.objType="EBaySiteCatalystObj";if(typeof(EBayObject)=="function")
{this.base=EBayObject;this.base(pParent,pName);}
else
{this.parent=pParent;this.name=pName;}
this.filesObj=new EBaySiteCatalystFilesObj(this,"ebSCOFilesObj");this.accounts=new Array;this.cookieDom=thisPageURL.contains(qaDom)?".ebay.com":countryDomain.substring(0,countryDomain.length-1);this.cookieName="ebay";this.country=this.qa=this.sample=this.page=this.host=this.pageName=this.channel="";this.user="Nonreg";this.l10Sample=this.loadBaseCode=false;this.writeCookie=ebSCOWriteCookie;this.readCookie=ebSCOReadCookie;this.addAccount=ebSCOAddAccount;this.setCountry=ebSCOSetCountry;this.getOrDropLuckyCookie=ebSCOGetOrDropLuckyCookie;this.determineAccounts=ebSCODetermineAccounts;this.loadFiles=ebSCOLoadFiles;}
function ebSCOWriteCookie(pName,pVal)
{writeCookieletEx(this.cookieName,pName,pVal,this.cookieDom,"/");}
function ebSCOReadCookie(pName)
{return readCookieletEx(this.cookieName,pName);}
function ebSCOAddAccount(pAccount)
{this.accounts[this.accounts.length]=pAccount;}
function ebSCOSetCountry()
{var c=country.toUpperCase(),tp=this.page;c=c.substring(0,2);if(country.is("tw")&&this.l10Sample)
this.addAccount("ebaytaiwan10"+this.qa);if(tp.contains('.com/nz/'))c="NZ";else if(tp.contains('ebaysweden'))c="SE";else if(tp.contains('.com.sg/'))c="SG";this.country=c+";;";}
function ebSCOGetOrDropLuckyCookie()
{var l9="lucky9",e=new Date(),v=readCookieEx(l9),yr=e.getYear();if(v=="")
v=Math.floor(Math.random()*1000000);if(is.nav&&(is.ver<5))
yr=yr+1900;e.setYear(yr+5);writeCookieEx(l9,v,this.cookieDom,"/",e);this.sample=v;return v;}
function ebSCODetermineAccounts()
{with(this)
{var l9=getOrDropLuckyCookie(),len=(l9.length>0);channel=countryDesc;if(len)
{if(l9%10==6)l10Sample=true;if(l9%100==6)addAccount('ebay1'+qa);if((l9%1000==6)&&(qa==''))
{var reg=(readCookieletEx("reg","flagReg")=="1")?'non':"";addAccount(ebStr+reg+'reg');addAccount(ebStr+channel);}}}
var e=new Date(),yr=e.getYear();if(is.nav&&(is.ver<5))
yr=yr+1900;e.setYear(yr+5);writeCookieEx("reg",readCookieEx("reg"),null,null,e);}
function ebSCOLoadFiles()
{var fs=this.filesObj.files,len=fs.length,fArr=new Array;for(var i=0;i<len;i++)
fArr[i]=scriptOpen+fs[i]+".js"+scriptClose;this.filesObj.files=new Array;for(i=0;i<len;i++)document.write(fArr[i]);}
function ebSSInit()
{ebSCObject=new EBaySiteCatalystObj(null,"ebSCObject");document.write('<img name="s_i_ebay" width=1 height=1 border=0>');var tp=ebayGetUnencodedHost();with(ebSCObject)
{page=tp.toLowerCase();host=document.location.hostname.toLowerCase();qa=thisPageURL.contains(qaDom)?"qa":"";var ssVal=readCookie("ssQA");if(ssVal!="")
writeCookie("ssQA",ssVal);if((readCookieletEx("reg","flagReg")=="1")||(readCookieEx("ebaysignin")=="in")||(readCookieEx("reg")=="1"))
{user="Reg";writeCookieletEx("reg","flagReg");}
determineAccounts();setCountry();loadBaseCode=(accounts.length||(ssVal!=""));if(loadBaseCode&&(typeof(ebSCOGetAndSetPageName)=="undefined"))
{var fo=filesObj;fo.add(fo.baseDir+"ssPageType");}}}
function dowt()
{}
(function()
{if(typeof(oHeaderCheck)=="undefined"&&typeof(oHeader)=="undefined")
{var d=document,l=document.location,pro=l.protocol;var fp="";if(pro.indexOf("http")==-1)
pro="http:";ip=pro+"//";if(pro=='https:')ip+="secure";ip+='include'+fp+'.ebaystatic.com/';if(d.getElementById||d.all)
{d.write('<sc'+'ript language="javascript" src="'+ip+'js/v/us/wt/base.js"></sc'+'ript>');window.oHeaderCheck=true;}}})();function ebNaturalSearchInit()
{if(document.location.search.indexOf("xpufu=x")!=-1)
window.cleanUp=new Function();if(typeof(nsArr)=="undefined")
{var pre=scriptOpen,post="stats/natural_search.js"+scriptClose;document.write(pre+includeDir+post);document.write(pre+countryIncludeDir+post);}}
function ebNaturalSearchTrack(pRID)
{var lpQInd=lastPage.indexOf("?"),lpHost=lastPage,nss=false;if(lpQInd!=-1)
lpHost=lpHost.substring(0,lpQInd);for(var i=nsArr.length-1;i>-1&&!nss;i--)
{if(lpHost.contains(nsArr[i].toLowerCase()))
nss=lastPage;}
if(nss)
{var t=new Date();var im='<img src="http://adfarm.mediaplex.com/a'+'d/lt/'+pRID;im+='?mpt='+escape(t.toGMTString())+'&mpcl='+escape(thisPage);im+='&mpvl='+escape(nss)+'">';document.write(im);}}
window.trackNaturalSearch=ebNaturalSearchTrack;ebNaturalSearchInit();ebSSInit();ebSCObject.loadFiles();
// -->
