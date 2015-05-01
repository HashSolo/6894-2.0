//<!--
//\include\js\globals.js@@\main\51

function Is()
{var t=true,agt=navigator.userAgent.toLowerCase(),vInd=0;this.major=parseInt(navigator.appVersion);this.webTV=this.opera=this.nav=this.ie=this.firefox=this.safari=false;if(agt.indexOf("webtv")!=-1)
{this.webTV=t;vInd=agt.indexOf("webtv/")+6;}
else if(agt.indexOf("firefox")!=-1)
{this.firefox=true;vInd=agt.lastIndexOf("firefox")+7;}
else if(agt.indexOf("safari")!=-1)
{this.safari=t;vInd=agt.lastIndexOf("safari")+7;}
else if(agt.indexOf("opera")!=-1)
{this.opera=t;vInd=agt.lastIndexOf("opera")+6;}
else if(navigator.appName=="Netscape")
{this.nav=t;vInd=agt.lastIndexOf("/")+1;}
else if(agt.indexOf("msie")!=-1)
{this.ie=t;vInd=agt.indexOf("msie")+4;}
this.ver=parseInt(agt.substring(vInd));this.win=(agt.indexOf("win")!=-1);this.winXP=(this.win&&(agt.indexOf("windows nt 5.1;")!=-1));this.mac=(agt.indexOf("mac")!=-1);this.macppc=(this.mac&&((agt.indexOf("ppc")!=-1)||(agt.indexOf("powerpc")!=-1)));this.xpSp2=(agt.indexOf("sv1")!=-1);}
if(typeof(is)=="undefined")
is=new Is();var country="us",countryDomain=".ebay.com/",countryDesc="US",countryDoubleByte=false;var countries=[[".ebay.com.au/",".com.au/","au","Australia"],[".ebay.at/",".at/","at","AT"],[".befr.ebay.be/",".befr.","befr","Belgium"],[".benl.ebay.be/",".benl.","benl","Belgium"],[".ebay.ca/",".ca/","ca","Canada"],[".cafr.ebay.ca/",".cafr.","cafr","FrenchCanada"],[".ebay.ch/",".ch/","ch","CH"],[".ebay.es/",".es/","es","Spain"],[".es.ebay.com/",".es.","es","Spain"],[".ebay.fr/",".fr/","fr","France"],[".ebay.de/",".de/","de","Germany"],[".ebay.it/",".it/","it","Italy"],[".ebay.in/",".in/","in","India"],[".ebay.nl/",".nl/","nl","Netherlands"],[".ebay.com.sg/",".sg/","sg","Singapore"],[".tw.ebay.com/",".tw.","tw","Taiwan"],[".ebay.com.cn/",".cn.","cn","China"],[".ebay.com.hk/",".hk.","hk","Hong Kong"],[".ebay.co.uk/",".co.uk/","uk","UK"],[".ebay.com/",".","us","US"]];var cbc=false,brow=false,atc=false,cbf=false,dot='.',and='&',qstn='?',eql='=',or="||";var http="http://",www="www",httpwww=http+www,https="https://",cgi="cgi",scgi="s"+cgi,jsExt=dot+"js";var tclExt=dot+"tcl",com="com",dotCom=dot+com,scriptOpen='<SC'+'RIPT SRC="',scriptClose='"></SC'+'RIPT>';var ebStr="ebay",ebDom=dot+ebStr,ebInclude="include",qaDom=dot+"qa"+dot,thisPage=location.href.toLowerCase(),lastPage="";if(history.length>1)
lastPage=document.referrer.toLowerCase();var thisPageURL,thisPageURI,lastPageURL,lastPageURI,qaMachineName="",includeDir="",cobrandDir="",countryCobrandDir="",picsDir="pics.ebaystatic.com/aw/pics/",countryIncludeDir="",includeHost;if(!includeHost)
includeHost=http+ebInclude+ebDom+dotCom+"/";var ebHTSch="ht"+eql,ebSID="s_partnerid",ebSIDSch=ebSID+eql,ebCID="co_partnerid",ebCIDSch=ebCID+eql,seoAOpen=' | <a href="',seoFontOpen='<font color="#0000CC">',seoFontClose='</font></a>',seoCookIn="in";globals=true;
// -->
