//<!--
//\include\js\poweredby\poweredby.js@@\main\7

var HdrSponsorEnable=1;var PagesDomain="";var currentDomain=window.location.host;var PagesQADomain="http://pages.qa.ebay.com/";var PagesProdDomain="http://pages.ebay.com/";if(currentDomain.indexOf(".qa.ebay.com")>0){var PagesDomain=PagesQADomain;}else{var PagesDomain=PagesProdDomain;}
var sponsorLink=PagesDomain+"ebay_IBM.html";var currentEnv=window.location.protocol;var PicsDomain="";if(currentEnv=="http:"){PicsDomain="http://pics.ebaystatic.com/aw/pics/";}else{PicsDomain="https://scgi.ebay.com/saw/pics/";}
function headerpartner(image,SponsorState,Link){var width=285;var height=350;var CurrentUrl=window.location.href.toLowerCase();if(SponsorState>0){if((CurrentUrl.indexOf("ebayisapi.dll?viewitem")>0)||(Link=="")){SponsorString=image;}else{SponsorString="<a href=\""+Link+"\" onclick=\"ebayShowPopupWindow('"+Link+"', 'Sponsor', "+width+", "+height+");return false;\">"+image+"</a>";}}else{SponsorString=image;}
document.write(SponsorString);}
// -->
