//<!--
//\include\js\cobrand\cobrand_determine.js@@\main\50

var siteID="0";var cobrandID="0";var ebPoolName="";var ebCBMicroBrands=new Array(new Array("100","techrepublic"),new Array("101","lego"));var ebMBCookieName="mb";var EB_POOL_TYPE_EQUIV="==";var EB_POOL_TYPE_CONTAINS="sc";var ebPoolObjs=new Array();function ebayPoolObjGetLinkCond(pDelim,pPre,pDom,pPost)
{var rs=pPre+'"//'+this.pool;if(this.dcpPool)
rs=pDelim+"("+rs+pPost+"&&"+pPre+'"'+pDom+pPost+")";else
rs=pDelim+rs+pDom+pPost;return rs;}
function ebayPoolObjSupported(pPool)
{var rv=false;if(this.type==EB_POOL_TYPE_EQUIV)
rv=(this.pool==pPool);else
rv=pPool.contains(this.pool);return rv;}
function EBayPoolObj(pName,pType,pIsLinkCond,pIsDCPPool,pIsSandbox)
{this.pool=pName;this.type=pType;this.dcpPool=pIsDCPPool;this.pIsSandbox=pIsSandbox;if(pIsLinkCond)
this.getLinkCond=ebayPoolObjGetLinkCond;else
this.getLinkCond=new Function("return '';");this.isSupportedPool=ebayPoolObjSupported;}
function ebayAddPoolObj(pName,pLogic,pIsLinkCond,pUsesHTParam,pSandbox)
{if(!ebPoolObjs[pName])
ebPoolObjs[pName]=new EBayPoolObj(pName,pLogic,pIsLinkCond,pUsesHTParam,pSandbox);return ebPoolObjs[pName];}
function ebayCreateStandardPoolObjs()
{var pArr=new Array(www,"pages","members","payments","offer","promo","k2b-bulk","scgi","syicatalogs","previewitem","spchk","my","feedback");for(var i=pArr.length-1;i>-1;i--)
ebayAddPoolObj(pArr[i],EB_POOL_TYPE_EQUIV,true);var pArr=new Array("sandbox","bcl");for(var i=pArr.length-1;i>-1;i--)
ebayAddPoolObj(pArr[i],EB_POOL_TYPE_EQUIV,true,false,true);pArr=new Array(cgi);for(i=pArr.length-1;i>-1;i--)
ebayAddPoolObj(pArr[i],EB_POOL_TYPE_CONTAINS,true);}
function ebayLoadCobrandFiles()
{var cbFiles=new Array();var protocol=thisPageURL.substring(0,thisPageURL.indexOf("://")+3);if(typeof(isPackaged)!="undefined")
{packagePath=includeHost+"js/v/"+country+"/";filePath=packagePath+"cobrand/cobrand.js";cbFiles[cbFiles.length]=filePath;}
else
{cbFiles[cbFiles.length]=cobrandDir+"cobrand_constants.js";cbFiles[cbFiles.length]=cobrandDir+"cobrand_variables.js";cbFiles[cbFiles.length]=cobrandDir+"cobrand_objects.js";cbFiles[cbFiles.length]=countryCobrandDir+"cobrand_country_objects.js";cbFiles[cbFiles.length]=cobrandDir+"cobrand_functionality.js";}
for(var i=0;i<cbFiles.length;i++)
document.write(scriptOpen+cbFiles[i]+scriptClose);}
function ebayCheckParams(pQStr)
{var pArr=pQStr.substr(1).split("&"),p;for(var i=0;i<pArr.length;i++)
{p=pArr[i].split("=");if(p[0]&&p[1])
{if(((p[0]=="ht=")&&(p[1]!="1"))||((p[0]=="s_partnerid=")&&(p[1]!="1"))||((p[0]=="co_partnerid=")&&(p[1]!="1")))
return true;}}
return false;}
function ebayCheckHTCookieletCobranded()
{var ht=false;if(typeof(readMultiCookie)=="function")
ht=readMultiCookie("ebay","ht");if(ht!=null&&ht!="")
return true;return false;}
function ebayParamsCobranded()
{return(ebayCheckParams(thisPageURI)||ebayCheckParams(lastPageURI)||ebayCheckHTCookieletCobranded());}
function ebayCheckEncoding(pURL)
{return((pURL.indexOf("_w0qq")!=-1)&&(pURL.indexOf("qqhtz")!=-1));}
function ebayEncodingCobranded()
{return(ebayCheckEncoding(thisPageURL)||ebayCheckEncoding(lastPageURL));}
function ebayIsSupportedPool(pPool)
{var cs="(";for(var i in ebPoolObjs)
{if(ebPoolObjs[i].pIsSandbox&&thisPageURL.indexOf("sandbox"))
{cs+="ebPoolObjs['"+i+"'].isSupportedPool(pPool)"+or;}
else
{cs+="ebPoolObjs['"+i+"'].isSupportedPool(pPool)"+or;}}
cs=cs.substring(0,cs.length-2);cs+=")";return eval(cs);}
function ebayIsNotCobranded()
{var tpURL=thisPageURL.substring(thisPageURL.indexOf("://")+3,thisPageURL.length);tpURL=tpURL.substring(0,tpURL.indexOf("/"));var tpArr=tpURL.split(dot);ebPoolName=tpArr[0];var rv=false;if(!(ebayParamsCobranded()||ebayEncodingCobranded()))
{if(tpArr.length<3)
return true;var uktwFlag=false;if(tpArr.length==3)
{if(ebPoolName=="sandbox")
return false;rv=ebayIsSupportedPool(ebPoolName);}
else if(tpArr.length==4)
{var ls=tpArr[1]+dot+tpArr[2]+dot+tpArr[3];if((ls==(ebStr+".co.uk"))||(ls==("tw."+ebStr+dotCom)))
{rv=true;uktwFlag=true;}
if(ls.indexOf("corp."+ebStr+dotCom)!=-1)
{rv=true;}
if(ls.indexOf("qa."+ebStr+dotCom)!=-1)
{rv=true;}}
else if(tpArr.length==5)
{var ls=tpArr[1]+dot+tpArr[2]+dot+tpArr[3]+dot+tpArr[4];if(tpArr[2].indexOf("qa")!=-1&&tpArr[1].indexOf("fp")!=-1)
{rv=true;}}
else if(tpArr.length==6)
{var ls=tpArr[1]+dot+tpArr[2]+dot+tpArr[3]+dot+tpArr[4]+dot+tpArr[5];if(ls.indexOf("qa."+ebStr+dotCom)!=-1)
{if(country==tpArr[1])
{rv=true;}}}}
return rv;}
function ebayCheckMicroBrand()
{var temp="";for(var j=0;j<ebCBMicroBrands.length&&isNotCobranded;j++)
{if(typeof(lastPageURL)!='undefined')
temp=lastPageURL;if(temp.contains(ebCBMicroBrands[j][1])||ReadCookie(ebMBCookieName)==ebCBMicroBrands[j][0])
isNotCobranded=false;}}
function ebayInitCobrandDetermine()
{isNotCobranded=true;ebayCheckMicroBrand();ebayCreateStandardPoolObjs();var s_partnerid;if(s_partnerid||!isNotCobranded||!ebayIsNotCobranded())
ebayLoadCobrandFiles();}
window.writeHeader=new Function();window.writeFooter=new Function();window.writeBrow=new Function();window.ebayCBCreateCobrands=new Function();if(globals)
ebayInitCobrandDetermine();sTime=new Date();brow=true;atc=true;cbc=true;
// -->
