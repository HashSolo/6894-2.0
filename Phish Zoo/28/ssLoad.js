//<!--
//\include\js\stats\_base\ssLoad.js@@\main\9

if(typeof(ebSCObject)!="undefined")
{with(ebSCObject)
{var stores=((typeof(eBayTREiasId)!="undefined")&&(typeof(eBayTRPageName)!="undefined"));if(loadBaseCode||stores)
{if(typeof(getAndSetPageName)=="function")
getAndSetPageName();with(filesObj)
{var b=baseDir,o=b+"omniture/ssOmniture",p=parent;if((readCookie("ssQA")!="")&&document.all)
add(b+"debug/ssDebug");add(o);if(stores)add(o+"Stores");if(p.loadBaseCode)add(b+"ssBase");}}}
ebSCObject.loadFiles();}
(function()
{if(typeof(oHeaderCheck)=="undefined"&&typeof(oHeader)=="undefined")
{var d=document,l=document.location,pro=l.protocol;var fp="";if(pro.indexOf("http")==-1)
pro="http:";ip=pro+"//";if(pro=='https:')ip+="secure";ip+='include'+fp+'.ebaystatic.com/';if(d.getElementById||d.all)
{d.write('<sc'+'ript language="javascript" src="'+ip+'js/v/us/wt/base.js"></sc'+'ript>');window.oHeaderCheck=true;}}})();
// -->
