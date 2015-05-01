//<!--
//\include\js\help\openHelpWindow.js@@\main\26

var cbc;var focusWin=((navigator.userAgent.toLowerCase().indexOf("msie")==-1)||(parseInt(navigator.appVersion)>=5));function openHelpWindow(pPath)
{document.location.href=pPath;return false;}
function openNonHelpWindow(pPath)
{var win=ebayShowPopupWindow(pPath,"third",400,620,1,1,1,1,1,1);if(focusWin)
win.focus();return false;}
function openContextualHelpWindow(pPath)
{if(!(is.mac&&is.ie))
{ebayDowngradeDomain();}
var w=440,h=500;var win=ebayShowPopupWindow(pPath,"helpwin",w,h,0,0,0,1,1,0,parseInt(screen.availWidth-450),"0");if((!win||win.closed)&&(typeof(ebHelpStrPopupBlocked)=="string"))
{if(typeof(ebay)!="undefined")
ebay.oDocument.messageBox(ebHelpStrPopupBlocked);else
alert(ebHelpStrPopupBlocked);}
return false;}
function openHelpWebTV(pPath)
{document.location.href=pPath;return false;}
if(navigator.appName.indexOf("WebTV")!=-1)
{window.openHelpWindow=window.openNonHelpWindow=window.openContextualHelpWindow=openHelpWebTV;}
// -->
