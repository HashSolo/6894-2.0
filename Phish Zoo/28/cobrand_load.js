//<!--
//\include\js\cobrand\cobrand_load.js@@\main\8

function ebayWriteCBHeader(center)
{if(document.all&&typeof(oHomePage)=='undefined')
document.write("<IE:HOMEPAGE ID=\"oHomePage\" />");if(center)
hfcenter=1;if(typeof(siteID)!='undefined'&&(siteID!=0))
{for(var i=0;i<ebCBFFileArray.length;i++)
document.write(scriptOpen+ebCBFFileArray[i]+scriptClose);}
var showHeader=true;if(typeof(noCobrandHeader)!="undefined")
showHeader=!noCobrandHeader;if(window.cbc&&showHeader)
writeHeader();}
window.writeCBHeader=ebayWriteCBHeader;function ebayWriteCBBrow()
{var showBrow=true;if(typeof(noCobrandHeader)!="undefined")
showBrow=!noCobrandHeader;if(window.cbc&&showBrow)
writeBrow();}
window.writeCBBrow=ebayWriteCBBrow;function ebayWriteCBSignInLink(notSignInText,notSignInLink,signInText,signInLink)
{var pre=" | <a href=\"",mid="\"><font color=\"#0000CC\">";var post="</font></a>";if(((ReadCookie("ebaysignin")=="in")||(ReadCookie("keepmesignin")=="in"))&&(signInText!=''))
document.write(pre+signInLink+mid+signInText+post);else if(notSignInText!='')
document.write(pre+notSignInLink+mid+notSignInText+post);}
window.writeCBSignInLink=ebayWriteCBSignInLink;function ebayWriteCBSignInLinkMotors(notSignInText,notSignInLink,signInText,signInLink)
{var pre=" | <a href=\"",mid="\"><font face=\"verdana, arial, helvetica\" size=\"1\" color=\"#000000\"><b>";var post="</b></font></a>";if(((ReadCookie("ebaysignin")=="in")||(ReadCookie("keepmesignin")=="in"))&&(signInText!=''))
document.write(pre+signInLink+mid+signInText+post);else if(notSignInText!='')
document.write(pre+notSignInLink+mid+notSignInText+post);}
window.writeCBSignInLinkMotors=ebayWriteCBSignInLinkMotors;
// -->
