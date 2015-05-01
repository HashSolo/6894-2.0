var gImages=new Array;  // Array to hold images so they don't show up on the page
var gIndex=0;		// Index for current image
var DCS=new Object();   // The DCS object holds parameters to override fields in the SDC weblog
var WT=new Object();    // The WT object holds parameters used in custom reporting
var DCSext=new Object(); // The DCSext object holds site specific parameters
var gQP=new Array();     // Array of desired query parameters to pass to WebTrends
var gI18n=true;
if (window.RegExp){
var RE={"%09":/\t/g,"%20":/ /g,"%23":/\#/g,"%26":/\&/g,"%2B":/\+/g,"%3F":/\?/g,"%5C":/\\/g,"%22":/\"/g,"%7F":/\x7F/g,"%A0":/\xA0/g};
var I18NRE={"%25":/\%/g};
}


var HSBC = new Object(); // THw HSBC parameter holds five subgroups of parameters defined on a per site basis


HSBC.SITE = new Object(); // SITE specific variables go into this object.  They are translated to DCSext vairables
HSBC.PAGE = new Object(); // WT Overrides --portal variables that need to overrride WT variables go into this object
HSBC.EXT = new Object(); //  This is a placeholder for non SITE oriented DCSext variables
HSBC.LOG = new Object(); //  DCS overrides -- Variables in here will override DCS.* variables matching the name
HSBC.DCS = new Object(); // This object contains HSBC.DCS.ID with the dcsid and HSBC.DCS.Doms with the comma-separated
                         // list of domains to be considered on-site.  As it currently stands, these domains have
                         // to be complete domain names.

HSBC.SITE.tagversion = "4.5";

// List all the live domains the group has (per GeB) to make sure they are not included as an off-site link
//
// MAKE SURE YOU ADD YOUR OWN WEBSITE DOMAINS TO THIS LIST
//
HSBC.DCS.Doms="www.hsbc.com,www.hsbcamanah.com,www.hsbc.co.uk,offshore.hsbc.com,www.stpapplications.hsbc.co.uk,www.hbeu1.hsbc.com,hsss1.hsbc.co.uk,hsss2.hsbc.co.uk,hsss.hsbc.co.uk";

// This is where the information collected by the WT Tracklet be sent
var gDomain="www1.member-hsbc-group.com";

// The gDcsId will be assigned from the HSBC.DCS.ID variable
var gDcsId="dcs8a9pzfvievvfhkn6u3og9d_5u3o"; // GLOBAL DCS ID...

// This is the code to handle the impressions for the onsite Ad campaigns
function dcsAdSearch(){
	if (document.links){
		for (var i=0;i<document.links.length;i++){
			var myAnch=document.links[i]+"";
			var pos=myAnch.toUpperCase().indexOf("WT.AC=");
			if (pos!=-1){
				var start=pos+6;
				var end=myAnch.indexOf("&",start);
				if (end==-1){
				end=myAnch.length;
				}
				var adValue=myAnch.substring(start,end);
				WT.ad=WT.ad?WT.ad+";"+adValue:adValue;
				}
		}
	}
}
// Code section for Enable Event Tracking

//  Replaces SmartView links when tracking offsite pages
//
function dcsParseSvl(sv){
	sv=sv.replace(/\s/g,"");
	var pos=sv.toUpperCase().indexOf("WT.SVL=");
	if (pos!=-1){
		var start=pos+8;
		var end=sv.indexOf('"',start);
		if (end==-1){
			end=sv.indexOf("'",start);
			if (end==-1){
				end=sv.length;
			}
		}
		return sv.substring(start,end);
	}
	return "";
}

// Tests if current domain is in a list of 'on-site' domains.  This test
// is much more restrictive than the default to capture phishing sites that spoof
// the HSBC domains.  The default code simply checks if the hostname
// matches somewhere in the domain array as in:
//  if ((host.indexOf(aDoms[i])!= -1) {
//           return 1;
//  }
function dcsIsOnsite(host){
    for (N in HSBC.DCS) {
	if (N == "Doms") {
	    var aDoms=HSBC.DCS.Doms.split(',');
    	    for (var i=0;i<aDoms.length;i++){
		if ((host.indexOf(aDoms[i])==0)&&
			(host.length==aDoms[i].length) ){
		       return 1;
		}
    	     }
	    return 0;
	}
    }
    return 1;
}
// Ensures that offsite links are only processed for http/s schemas

function dcsIsHttp(e){
	return (e.href&&e.protocol&&(e.protocol.indexOf("http")!=-1))?true:false;
}
//
//  This function makes sure that an actual event is handed to the various trigger functions
// depending on the way events are supported in the JavaScript

function dcsEvt(evt){
	var e=evt.target||evt.srcElement;
	if (e.tagName&&(e.tagName=="IMG")){
		e=e.parentElement||e.parentNode;
	}
	return e;
}
// This function adds a specific function to fire off when a desired event occurs
// New function supplied by WT to resolve the HK JS error
function dcsBind(event,func){
	if (typeof(window[func])=="function"){
		if ((document.links) && (document.links.length>0) ) {
			for (var i =0;i < document.links.length;i++) {
				if (document.links[i].addEventListener){
				document.links[i].addEventListener(event, window[func], true);
				}
				else if(document.links[i].attachEvent){
					document.links[i].attachEvent("on"+event, window[func]);
				}
			}
		}
	}
}

// This functions defines the events of intgerest and the functions associated with them.
// In this example, we're only interested in offsite links

function dcsET(){
    dcsBind("click","dcsDownload");
    dcsBind("click","dcsOffsite");
    dcsBind("click","dcsCampaign");
    dcsBind("mousedown","dcsRightClick");

}
// Track clicks to download links.
function dcsDownload(evt){
    evt=evt||(window.event||"");
    if (evt){
        var e=dcsEvt(evt);
        if (dcsIsHttp(e)&&dcsIsOnsite(e.hostname)){
            var types="xls,doc,pdf,txt,csv,zip";   //customize download types
            if (types.indexOf(e.pathname.substring(e.pathname.lastIndexOf(".")+1,e.pathname.length))!=-1){
                var qry=e.search?e.search.substring(e.search.indexOf("?")+1,e.search.length):"";
                if (qry.toUpperCase().indexOf("WT.SVL=")==-1){
                    WT.svl=dcsParseSvl(e.name?e.name.toString():(e.onclick?e.onclick.toString():""));
                }
                var path=(e.pathname.indexOf("/")!=0)?"/"+e.pathname:e.pathname;
                dcsMultiTrack("DCS.dcssip",e.hostname,"DCS.dcsuri",path,"WT.ti","Download:"+path,"WT.dl","1");
                DCS.dcssip=DCS.dcsuri=WT.ti=WT.svl=WT.dl="";
            }
        }
    }
}

// Track right clicks to download links.
function dcsRightClick(evt){
    evt=evt||(window.event||"");
    if (evt){
        var btn=evt.which||evt.button;
        if (btn!=1){
            var e=evt.target||evt.srcElement;
            if (dcsIsHttp(e)){
                var types="xls,doc,pdf,txt,csv,zip";    //customize download types
                if (types.indexOf(e.pathname.substring(e.pathname.lastIndexOf(".")+1,e.pathname.length))!=-1){
                    var path=(e.pathname.indexOf("/")!=0)?"/"+e.pathname:e.pathname;
                    dcsMultiTrack("DCS.dcssip",e.hostname,"DCS.dcsuri",path,"DCS.dcsqry","","WT.ti","Download:"+path,"WT.dl","1","WT.rc","1");
                    DCS.dcssip=DCS.dcsuri=WT.ti=WT.dl=WT.rc="";
                }
            }
        }
    }
}

// Code section for Track clicks to links leading offsite.
function dcsOffsite(evt){
	evt=evt||(window.event||"");
	if (evt){
		var e=dcsEvt(evt);
		
		if (dcsIsHttp(e)&&!dcsIsOnsite(e.hostname)){
			var qry=e.search?e.search.substring(e.search.indexOf("?")+1,e.search.length):"";
			if (qry.toUpperCase().indexOf("WT.SVL=")==-1){
				WT.svl=dcsParseSvl(e.name?e.name.toString():(e.onclick?e.onclick.toString():""));
			}
			var path=(e.pathname.indexOf("/")!=0)?"/"+e.pathname:e.pathname;
			var trim=true;
			dcsMultiTrack("DCS.dcssip",e.hostname,"DCS.dcsuri",path,"DCS.dcsqry",trim?"":qry,"WT.ti","Offsite:"+e.hostname+path+qry,"WT.os","1","DCS.dcsref",window.location.hostname);
			DCS.dcssip=DCS.dcsuri=DCS.dcsqry=WT.ti=WT.svl=WT.os=DCS.dcsref="";
		}
	}
}

// Code section for Track clicks to offsite links with __gpLaunch parameter.
function dcsCampaign(evt){
         evt=evt||(window.event||"");
         if (evt){
                 var e=dcsEvt(evt);
                    
                 if (dcsIsHttp(e)&&e.href.indexOf("__gpLaunch")!=-1){
                        var qry=e.search?e.search.substring(e.search.indexOf("?")+1,e.search.length):"";
                        if (qry.toUpperCase().indexOf("WT.SVL=")==-1){
                                WT.svl=dcsParseSvl(e.name?e.name.toString():(e.onclick?e.onclick.toString():""));
                        }
                        var path=(e.pathname.indexOf("/")!=0)?"/"+e.pathname:e.pathname;
                        var trim=true;
                        dcsMultiTrack("DCS.dcssip",e.hostname,"DCS.dcsuri",path,"DCS.dcsqry",trim?"":qry,"WT.ti","gpLaunch: "+e.hostname,"WT.os","1","DCS.dcsref",window.location.hostname);
                        DCS.dcssip=DCS.dcsuri=DCS.dcsqry=WT.ti=WT.svl=WT.os=DCS.dcsref="";
               }
       }
}

// Assign your query parameters to WebTrends query parameters.
function dcsQP(N){
    if (!N){
        return;
    }
    var qry=location.search.substring(1);
    var pairs=qry.split("&");
    for (var i=0;i<pairs.length;i++){
        var pos=pairs[i].indexOf("=");
        if (pos==-1){
            continue;
        }
        var name=pairs[i].substring(0,pos);
        var value=pairs[i].substring(pos+1);
        if (name==N){
            gQP.push((i==0?"":"&")+pairs[i]);
            return value;
        }
    }
    return "";
}
//<!-- START OF SDC Advanced Tracking Code -->
//<!-- Copyright (c) 1996-2005 WebTrends Inc.  All rights reserved. -->
//<!-- V7.5 -->
//<!-- $DateTime: 2005/08/25 15:58:46 $ -->

var gService = true;
var gTimeZone = 0;
// Code section for Enable First-Party Cookie Tracking
function dcsCookie(){
if (typeof(dcsOther)=="function"){
dcsOther();
}
else if (typeof(dcsPlugin)=="function"){
dcsPlugin();
}
else if (typeof(dcsFPC)=="function"){
dcsFPC(gTimeZone);
}
}
function dcsGetCookie(name){
var pos=document.cookie.indexOf(name+"=");
if (pos!=-1){
var start=pos+name.length+1;
var end=document.cookie.indexOf(";",start);
if (end==-1){
end=document.cookie.length;
}
return unescape(document.cookie.substring(start,end));
}
return null;
}
function dcsGetCrumb(name,crumb){
var aCookie=dcsGetCookie(name).split(":");
for (var i=0;i<aCookie.length;i++){
var aCrumb=aCookie[i].split("=");
if (crumb==aCrumb[0]){
return aCrumb[1];
}
}
return null;
}
function dcsGetIdCrumb(name,crumb){
var cookie=dcsGetCookie(name);
var id=cookie.substring(0,cookie.indexOf(":lv="));
var aCrumb=id.split("=");
for (var i=0;i<aCrumb.length;i++){
if (crumb==aCrumb[0]){
return aCrumb[1];
}
}
return null;
}
function dcsFPC(offset){
if (typeof(offset)=="undefined"){
return;
}
var name=gFpc;
var dCur=new Date();
dCur.setTime(dCur.getTime()+(dCur.getTimezoneOffset()*60000)+(offset*3600000));
var dExp=new Date(dCur.getTime()+315360000000);
var dSes=new Date(dCur.getTime());
if (document.cookie.indexOf(name+"=")!=-1){
var id=dcsGetIdCrumb(name,"id");
var lv=parseInt(dcsGetCrumb(name,"lv"));
var ss=parseInt(dcsGetCrumb(name,"ss"));
if ((id==null)||(id=="null")||isNaN(lv)||isNaN(ss)){
return;
}
WT.co_f=id;
var dLst=new Date(lv);
dSes.setTime(ss);
if ((dCur.getTime()>(dLst.getTime()+1800000))||(dCur.getTime()>(dSes.getTime()+28800000))){
dSes.setTime(dCur.getTime());
WT.vt_f_s="1";
}
if ((dCur.getDay()!=dLst.getDay())||(dCur.getMonth()!=dLst.getMonth())||(dCur.getYear()!=dLst.getYear())){
WT.vt_f_d="1";
}
}
else{
var tmpname=name+"_TMP=";
document.cookie=tmpname+"1";
if (document.cookie.indexOf(tmpname)!=-1){
document.cookie=tmpname+"; expires=Thu, 01-Jan-1970 00:00:01 GMT";
if ((typeof(gWtId)!="undefined")&&(gWtId!="")){
WT.co_f=gWtId;
}
else if ((typeof(gTempWtId)!="undefined")&&(gTempWtId!="")){
WT.co_f=gTempWtId;
WT.vt_f="1";
}
else{
WT.co_f="2";
var cur=dCur.getTime().toString();
for (var i=2;i<=(32-cur.length);i++){
WT.co_f+=Math.floor(Math.random()*16.0).toString(16);
}
WT.co_f+=cur;
WT.vt_f="1";
}
if (typeof(gWtAccountRollup)=="undefined"){
WT.vt_f_a="1";
}
WT.vt_f_s="1";
WT.vt_f_d="1";
}
else{
WT.vt_f="2";
WT.vt_f_a="2";
return;
}
}
WT.co_f=escape(WT.co_f);
WT.vt_sid=WT.co_f+"."+dSes.getTime();
var expiry="; expires="+dExp.toGMTString();
document.cookie=name+"="+"id="+WT.co_f+":lv="+dCur.getTime().toString()+":ss="+dSes.getTime().toString()+expiry+"; path=/"+(((typeof(gFpcDom)!="undefined")&&(gFpcDom!=""))?("; domain="+gFpcDom):(""));
}

// Code section for Use the new first-party cookie generated with this tag.
var gFpc="WT_FPC";
var gWtId="";
var gTempWtId="";
var gConvert=true;


// Code section for Set the First-Party Cookie domain
var gFpcDom="";
//
// If we need to convert an old SDC cookie, this code will pull the old cookie from the sdc server
//
if ((typeof(gConvert)!="undefined")&&gConvert&&(document.cookie.indexOf(gFpc+"=")==-1)){
document.write("<SCR"+"IPT Language='JavaScript' SRC='"+"http"+(window.location.protocol.indexOf('https:')==0?'s':'')+"://"+gDomain+"/"+gDcsId+"/wtid.js"+"'></SCR"+"IPT>");
}

function dcsAdv(){
	dcsFunc("dcsAdSearch");
	dcsFunc("dcsCookie");
	dcsFunc("dcsET");
}

function dcsFunc(func){
	if (typeof(window[func])=="function"){
		window[func]();
	}
}

// dcsMultiTrack is called with pairs of parameters as
// text strings that use the JavaScript variable to be
// overridded and the value to override.  For example,
// in an onClick function downloading a PDF:
// onClick="dcsMultiTrack('DCS.dcsuri','this.pdf')"

function dcsMultiTrack(){
	for (var i=0;i<arguments.length;i++){
		if (arguments[i].indexOf('WT.')==0){
				WT[arguments[i].substring(3)]=arguments[i+1];
				i++;
		}
		if (arguments[i].indexOf('DCS.')==0){
				DCS[arguments[i].substring(4)]=arguments[i+1];
				i++;
		}
		if (arguments[i].indexOf('DCSext.')==0){
				DCSext[arguments[i].substring(7)]=arguments[i+1];
				i++;
		}
	}
	var dCurrent=new Date();
	DCS.dcsdat=dCurrent.getTime();
	dcsTag();
}


//dcsVar sets default values for the following variables:
// URI Stem 		(DCS.dcsuri)
// URI Query String 	(DCS.dcsqry)
// Server 		(DCS.dcssip)
// Referring document 	(DCS.dcsref)
// Document Title 	(WT.ti)
// User Time Zone 	(WT.tz)
// User browsing hour 	(WT.bh)
// User language  	(WT.ul)
// Screen widthxheight 	(WT.sr)
// Screen color depth 	(WT.cd)
// Java capability 	(WT.jo)
// JavaScript capability (WT.js)
// JavaScript Version 	(WT.jv
//
function dcsVar(){
	var dCurrent=new Date();
	WT.tz=dCurrent.getTimezoneOffset()/60*-1;
	if (WT.tz==0){
		WT.tz="0";
	}
	WT.bh=dCurrent.getHours();
	WT.ul=navigator.appName=="Netscape"?navigator.language:navigator.userLanguage;
	if (typeof(screen)=="object"){
		WT.cd=navigator.appName=="Netscape"?screen.pixelDepth:screen.colorDepth;
		WT.sr=screen.width+"x"+screen.height;
	}
	if (typeof(navigator.javaEnabled())=="boolean"){
		WT.jo=navigator.javaEnabled()?"Yes":"No";
	}
	if (document.title){
		WT.ti=gI18n?dcsEscape(dcsEncode(document.title),I18NRE):document.title;
	}
	WT.js="Yes";
	if (typeof(gVersion)!="undefined"){
		WT.jv=gVersion;
	}
	if (document.body&&document.body.addBehavior){
	    document.body.addBehavior("#default#clientCaps");
	    if (document.body.connectionType){
		WT.ct=document.body.connectionType;
	    }
	    document.body.addBehavior("#default#homePage");
	    WT.hp=document.body.isHomePage(location.href)?"1":"0";
	}
	if (parseInt(navigator.appVersion)>3){
	    if ((navigator.appName=="Microsoft Internet Explorer")&&document.body){
		WT.bs=document.body.offsetWidth+"x"+document.body.offsetHeight;
	    }
	    else if (navigator.appName=="Netscape"){
		WT.bs=window.innerWidth+"x"+window.innerHeight;
	    }
	}
	WT.fi="No";
	if (window.ActiveXObject){
	    var maxVer=10;
	    var minVer=2;
	    for (var ver=minVer;ver<=maxVer;ver++){
		try{
		    var oFlash = eval("new ActiveXObject('ShockwaveFlash.ShockwaveFlash."+ver+"');");
		    if (oFlash){
			WT.fi="Yes";
			WT.fv=ver+".0";
			break;
		    }
		}
		catch(e){
		}
	    }
	}
	else if (navigator.plugins&&navigator.plugins.length){
	    for (var i=0;i<navigator.plugins.length;i++){
		if (navigator.plugins[i].name.indexOf('Shockwave Flash')!=-1){
		    WT.fi="Yes";
		    WT.fv=navigator.plugins[i].description.split(" ")[2];
		    break;
		}
	    }
	}
	if (gI18n){
		WT.em=(typeof(encodeURIComponent)=="function")?"uri":"esc";
		if (typeof(document.defaultCharset)=="string"){
			WT.le=document.defaultCharset;
		} 
		else if (typeof(document.characterSet)=="string"){
			WT.le=document.characterSet;
		}
	}
	DCS.dcsdat=dCurrent.getTime();
	DCS.dcssip=window.location.hostname;
	DCS.dcsuri=window.location.pathname;
	if (window.location.search){
		DCS.dcsqry=window.location.search;
		if (gQP.length>0){
		    for (var i=0;i<gQP.length;i++){
			var pos=DCS.dcsqry.indexOf(gQP[i]);
			if (pos!=-1){
			    var front=DCS.dcsqry.substring(0,pos);
			    var end=DCS.dcsqry.substring(pos+gQP[i].length,DCS.dcsqry.length);
			    DCS.dcsqry=front+end;
			}
		    }
		}
	}
	if ((window.document.referrer!="")&&(window.document.referrer!="-")){
		if (!(navigator.appName=="Microsoft Internet Explorer"&&parseInt(navigator.appVersion)<4)){
			DCS.dcsref=gI18n?dcsEscape(window.document.referrer, I18NRE):window.document.referrer;
		}
	}
}

// This is an internal function to append tags to a query
// string (including escaping invalid characters)

function dcsA(N,V){
return "&"+N+"="+dcsEscape(V, RE);
}

// This function allows characters that would be invalid
//in a query string to pass through and be reported on
//at the destination.

function dcsEscape(S, REL){
	if (typeof(REL)!="undefined"){
		var retStr = new String(S);
		for (R in REL){
		retStr = retStr.replace(REL[R],R);
		}
			return retStr;
		}
		else{
		return escape(S);
	}
}

function dcsEncode(S){
	return (typeof(encodeURIComponent)=="function")?encodeURIComponent(S):escape(S);
}

// This function creates an image array and places the
// parameter as the source value for the image.  Setting
// the source value has the effect of loading the
// image from the server, but it does not hold up
// display of the page.
function dcsCreateImage(dcsSrc){
	if (document.images){
		gImages[gIndex]=new Image;
		gImages[gIndex].src=dcsSrc;
		gIndex++;
	}
	else{
                document.write('<img border="0" name="dcsimg" width="1" height="1" alt="" src="'+dcsSrc+'">');
      	}
}

// This function goes through all META tags on a page
// and replaces WT, DCS and DCSext JavaScript variables
// with the content of the META tags that match their names.
// With this function you do not have to include or
// modify JavaScript to change parameters used by
// WebTrends reporting.
function dcsMeta(){
	var myDocumentElements;
	if (document.all){
		myDocumentElements=document.all.tags("meta");
	}
	else if (document.documentElement){
	 myDocumentElements=document.getElementsByTagName("meta");
	}
	if (typeof(myDocumentElements)!="undefined"){
	   for (var i=1;i<=myDocumentElements.length;i++){
		myMeta=myDocumentElements.item(i-1);
		if (myMeta.name){
			if (myMeta.name.indexOf('WT.')==0){
				WT[myMeta.name.substring(3)]=(gI18n&&(myMeta.name.indexOf('WT.ti')==0))?dcsEscape(dcsEncode(myMeta.content),I18NRE):myMeta.content;
			} else if (myMeta.name.indexOf('DCSext.')==0){
				DCSext[myMeta.name.substring(7)	]
					=myMeta.content;
			}else if (myMeta.name.indexOf('DCS.')==0){
				DCS[myMeta.name.substring(4)]=(gI18n&&(myMeta.name.indexOf('DCS.dcsref')==0))?dcsEscape(myMeta.content,I18NRE):myMeta.content;
			}
		}
	   }
	}
}

//This function calls the appending function to create the
// tag and dcsCreateImage to send it to the server.
function dcsTag(){
	if (document.cookie.indexOf("WTLOPTOUT=")!=-1){
		return;
	}
	var P="http"+(window.location.protocol.indexOf('https:')==0?'s':'')+"://"+gDomain+(gDcsId==""?'':'/'+gDcsId)+"/dcs.gif?";
	for (N in DCS){
		if (DCS[N]) {
			P+=dcsA(N,DCS[N]);
		}
	}
	for (N in WT){
		if (WT[N]) {
			P+=dcsA("WT."+N,WT[N]);
		}
	}
	for (N in DCSext){
		if (DCSext[N]) {
			P+=dcsA(N,DCSext[N]);
		}
	}
	if (P.length>2048&&navigator.userAgent.indexOf('MSIE')>=0){
		P=P.substring(0,2040)+"&WT.tu=1";
	}
	dcsCreateImage(P);
}
// This function collects an HSBC cookie that otherwise would
// not be available for tracking  The desired cookie is
// passed as a parameter and the cookie value returned.
function dcsGetHSBCCookie(name)
{
    var dc = document.cookie;
    var prefix = name + "=";
    var begin = dc.indexOf("; " + prefix);
    if (begin == -1)
    {
        begin = dc.indexOf(prefix);
        if (begin != 0) return null;
    }
    else
    {
        begin += 2;
    }
    var end = document.cookie.indexOf(";", begin);
    if (end == -1)
    {
        end = dc.length;
    }
    return unescape(dc.substring(begin + prefix.length, end));
}

// This function collects all the HSBC variables and translates
// them into the appropriate WebTrends variabls (WT variables
// from HSBC.PAGE, DCSext variables from SITE and EXT, and
// DCS variables from HSBC.LOG variables.  It also overrides
// the value of gDcsId from the HSBC.DCS.ID variable.

function dcsMapHSBC() {
	for (N in HSBC) {
		if (N == "SITE")  {
			for (S in HSBC.SITE) {
				DCSext[S] = HSBC.SITE[S];
			}
		}
		if (N == "PAGE") {
			for (S in HSBC.PAGE) {
				WT[S] = HSBC.PAGE[S];
			}
		}
		if (N == "EXT") {
			for (S in HSBC.EXT) {
				DCSext[S] = HSBC.EXT[S];
			}
		}
		if (N == "LOG") {
			for (S in HSBC.LOG) {
				DCS[S] = HSBC.LOG[S];
			}
		}
		if (N == "DCS") {
			gDcsId = HSBC.DCS.ID;
		}
	}
}
