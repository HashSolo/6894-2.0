//
// Setup global variables
//
// fontSize - default value
// fMax - largest font size
// fMin - smallest font size
// isIE501 - is the user agent MSIE 5.01
//
var fontSize = 100;
var fMax = 120;
var fMin = 100;
var isIE501 = navigator.userAgent.indexOf("MSIE 5.01") > 0 ? true : false;
var isNN6 = navigator.userAgent.indexOf("Netscape6") > 0 ? true : false;
var isIE=document.all&&navigator.userAgent.indexOf("Opera")==-1;
var SKIP_VISIBLE = "#000";			//value is black
var SKIP_INVISIBLE = "#fff";		//value is white
//
// setFontSize
//
// Retrieves cookie value and applies to font size
//
function setFontSize() {
	//var tempSize = getCookie("fontSize");
	//if((tempSize != null) && (tempSize >= 100) && (tempSize <= 120)) {
	//	fontSize = tempSize;
	//} else {
	//	setCookie("fontSize", 100, "", "/");
	//}
	// Reduce by 20% for Website Redesign 2007
	//document.body.style.fontSize = (fontSize - 20) + "%";
}


//
// changeFontSize(bool increment)
//
// changes document font size and records size value in cookie
//
function changeFontSize(increment) {
	if(increment) {
		fontSize=parseInt(fontSize) + parseInt(10);
	} else {
		fontSize=parseInt(fontSize) - parseInt(10);
	}

	if(fontSize > fMax) {
		fontSize = fMax;
	}
	if(fontSize < fMin) {
		fontSize = fMin;
	}
	switch(fontSize) {
		case 100:
			document.body.style.fontSize = "1em";
			break;
		case 110:
			document.body.style.fontSize = "1.10em";
			break;
		case 120:
			document.body.style.fontSize = "1.20em";
			break;
	}
	setCookie('fontSize', fontSize, "", "/");
	// Reduce by 20% for Website Redesign 2007
	fontSize = fontSize - 20;
}
//
// incrementFontSize
//
function incrementFontSize() {
	changeFontSize(true);
}

//
// decrementFontSize
//
function decrementFontSize() {
	changeFontSize(false);
}


//
// Sets a Cookie with the given name and value.
//
// name       Name of the cookie
// value      Value of the cookie
// [expires]  Expiration date of the cookie (default: end of current session)
// [path]     Path where the cookie is valid (default: path of calling document)
// [domain]   Domain where the cookie is valid
//              (default: domain of calling document)
// [secure]   Boolean value indicating if the cookie transmission requires a
//              secure transmission
//
function setCookie(name, value, expires, path, domain, secure) {
    document.cookie= name + "=" + escape(value) +
        ((expires) ? "; expires=" + expires.toGMTString() : "") +
        ((path) ? "; path=" + path : "") +
        ((domain) ? "; domain=" + domain : "") +
        ((secure) ? "; secure" : "");
}


//
// Gets the value of the specified cookie.
//
// name  Name of the desired cookie.
//
// Returns a string containing value of specified cookie,
//   or null if cookie does not exist.
//
function getCookie(name) {
    var dc = document.cookie;
    var prefix = name + "=";
    var begin = dc.indexOf("; " + prefix);
    if (begin == -1) {
        begin = dc.indexOf(prefix);
        if (begin != 0) return null;
    } else {
        begin += 2;
    }
    var end = document.cookie.indexOf(";", begin);
    if (end == -1) {
        end = dc.length;
    }
    return unescape(dc.substring(begin + prefix.length, end));
}

/*used for automaticaly jumping from one input to the next*/
var count = 0;
function checkInput(theInputBox, maxChars, previousId, nextId) {
	if(nextId != "") {
		nextInputBox = document.getElementById(nextId);
		if((count == (maxChars-1)) && (theInputBox.value.length == maxChars)) {
			if(nextInputBox.value.length == 0) {
				count = 1;
			} else {
				count = nextInputBox.value.length;
			}
			nextInputBox.select();
			return;
		}
	}
	if(count == 0 && theInputBox.value.length == 0) {
		PreviousInputBox = document.getElementById(previousId);
		if(PreviousInputBox.value.length == 0) {
			count = 1;
		} else {
			count = PreviousInputBox.value.length;
		}
		PreviousInputBox.select();
		return;
	}
	count = theInputBox.value.length;
}
//onKeyPress
function previous(theInputBox) {
	count = theInputBox.value.length;
}
//onfocus
function current(theInputBox) {
	count = 999;
	theInputBox.select();
}

function hideAccounts() {
	switchAccountDisplay("none", "block", true);
/*	document.getElementById("balance-top").style.background = "url(../images/show-balance.gif) no-repeat";*/
}

function showAccounts() {
	switchAccountDisplay("block", "none", true);
/*	document.getElementById("balance-top").style.background = "url(../images/hide-balance-top.gif) no-repeat";*/
}
function hideDetails() {
	switchDetailDisplay("none", "block");
}

function showDetails() {
	switchDetailDisplay("block", "none");
}

function switchDetailDisplay(state1, state2) {
	if(document.getElementById("detail-switch")) {
		document.getElementById("detail-switch").style.display = state1;
		document.getElementById("hide-detail-switch").style.display = state1;
		document.getElementById("show-detail-switch").style.display = state2;
	}
}

//
//	These functions wraps the variables in HTML code for placement in the page
//
function displayTag(address,text,title) {
	if(title==null) {
		title = text;
	}
	document.write('<div class=\"hsbcButtonLeft\"></div> <div class=\"hsbcButtonCenter\"><a href=\"',address,'\" title=\"',title,'\">',text,'</a><i>.</i></div><div class=\"hsbcButtonRight\"></div>');
}

function displayResetTag(text) {
		document.write('<div class=\"hsbcButtonLeft\"></div> <div class=\"hsbcButtonCenter\"><a href=\"#\"  onclick=\"document.forms[0].reset()\" onkeypress=\"document.forms[0].reset()\">',text,'</a><i>.</i></div><div class=\"hsbcButtonRight\"></div>');
}

function displayPrintTag() {
	document.write('&nbsp;-&nbsp;<a href=\"#\" class=\"important\">print this page</a>');
}
//
// switchAccountDisplay(string state1,string state2)
// restore show/hide account details from cookie
//
function switchAccountDisplay(state1, state2, browserReload) {
	if(document.getElementById("jsAccountDetails")) {
		document.getElementById("jsAccountDetails").style.display = state1;
		document.getElementById("jsHideAccounts").style.display = state1;
		document.getElementById("jsShowAccounts").style.display = state2;
		setCookie("state1", state1, "", "/");
		setCookie("state2", state2, "", "/");
		if((browserReload == true) && ((isIE501 == true) || (isNN6==true))) {
			location.reload();
		}
	}
}



//
// restoreAccountDisplay
// restore show/hide list of account details from cookie
//
function restoreAccountDisplay() {
	state1 = getCookie("state1") == null ? "none" : getCookie("state1");
	state2 = getCookie("state2") == null ? "block" : getCookie("state2");
	switchAccountDisplay(state1, state2, false);
}

function setJSFunctionality() {
	if(document.getElementById("jsSecondaryFunctionality")) {
		document.getElementById("jsSecondaryFunctionality").style.display = "block";
	}
}



function expandAll() {
	toggleAll('block');
}

function collapseAll() {
	toggleAll('none');
}

function expandAllLink() {
	expandAll()
	if(document.getElementById("jsExpandAll")) {
			document.getElementById("jsExpandAll").style.display = "none";
	}
	if(document.getElementById("jsCollapseAll")) {
			document.getElementById("jsCollapseAll").style.display = "block";
	}
}

function collapseAllLink() {
	collapseAll()
	if(document.getElementById("jsExpandAll")) {
			document.getElementById("jsExpandAll").style.display = "block";
	}
	if(document.getElementById("jsCollapseAll")) {
			document.getElementById("jsCollapseAll").style.display = "none";
	}
}


function skipLinkFocus(skipLinkName) {
	skipLinkName.style.color = SKIP_VISIBLE;
}

function skipLinkBlur(skipLinkName) {
	skipLinkName.style.color = SKIP_INVISIBLE;
}


//
// do_onload
//
function do_onload() {
	setFontSize();
	setJSFunctionality();
	restoreAccountDisplay();

	if(document.getElementById("jsSiteMap")) {
		collapseAll()
	}
	
	if(document.getElementById("jsSiteMapBar")) {
		document.getElementById("jsSiteMapBar").style.display = "block";
	}
	
	if(document.getElementById("detail-switch")) {
		document.getElementById("detail-switch").style.display = "none";
		document.getElementById("show-detail-switch").style.display = "block";
		document.getElementById("hide-detail-switch").style.display = "none";
		document.getElementById("nojs-detail-switch").style.display = "none";
	}
}



//
// trigger onLoad function (do_onload)
//
if (window.addEventListener) {
	window.addEventListener("load", do_onload, false);
} else {
	if (window.attachEvent) {
		window.attachEvent("onload", do_onload);
	} else {
		if (document.getElementById) {
			window.onload = do_onload;
		}
	}
}

//
//popup function for help windows
//
function popup_help(url)
{
	newwindow=window.open(url,'name','status=yes,location=no,menubar=no,scrollbars=yes,toolbar=no,resizable=yes,width=635,height=545,top=0,screenY=0,left=0,screenX=0');
	if (window.focus) {newwindow.focus()}
	return false;
}
