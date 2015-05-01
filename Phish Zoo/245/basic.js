//Onload functionalities
var onLoadFunctions = new Array();
var anchorOnLoad = null;
				
function addOnLoadFunction(f){
       
	onLoadFunctions[onLoadFunctions.length] = f;
}

function OnLoad() {
	if ( anchorOnLoad ) {
		Anchor = eval('onLoadAnchor');
		Anchor.click();
	}
	for(var i = 0; i < onLoadFunctions.length; i++ ) {
		var current = onLoadFunctions[i];
		var parenIndex = current.indexOf("(");
		if(parenIndex != -1) {
			var command = current.substring(0,parenIndex);
			//eval('if(typeof('+ command+ ') == "function") {eval(onLoadFunctions[i]);}');
			eval(onLoadFunctions[i]);
		}
	}
}

var onUnloadFunctions = new Array();
function addOnUnloadFunction(f){
	onUnloadFunctions[onUnloadFunctions.length] = f;
}
function OnUnload() {
	for(var i = 0; i < onUnloadFunctions.length; i++ ) {
		var current = onUnloadFunctions[i];
		var parenIndex = current.indexOf("(");
		if(parenIndex != -1) {
			var command = current.substring(0,parenIndex);
			//eval('alert(typeof("' + command + '"));if(typeof('+ command+ ') == "function") {eval(onUnloadFunctions[i]);}');
			
		    eval(onUnloadFunctions[i]);
		}
	}
}

function displayElement(elementId){
 var elemObj = document.getElementById(elementId);
 if(elemObj != null){
   elemObj.style.display="inline";
 }
}
function disableElement(elementId){
 var elemObj = document.getElementById(elementId);
 if(elemObj != null){
   elemObj.disabled=false;
 }
}
function setFocusOnElement(elementId){
 var elemObj = document.getElementById(elementId);
 if(elemObj != null){
   elemObj.focus();
 }
}
function closeMessage(id){
	var messageBox = document.getElementById(id);
	if(messageBox != null)
		messageBox.style.display = "none";
}
function fmObjectActivate(id, movie, width, height, play, quality, bgcolor, base) {
//do: generate an object to activate

	var fm = fmObjectGetParams(id, movie, width, height, play, quality, bgcolor, base);
	return fmObjectGenerate(fm.attrs, fm.params);
}

function fmObjectActivateWrite(id, movie, width, height, play, quality, bgcolor, base) {
//do: generate an object to activate

	var fm = fmObjectGetParams(id, movie, width, height, play, quality, bgcolor, base);
	document.write(fmObjectGenerate(fm.attrs, fm.params));
}

function fmObjectGetParams(id, movie, width, height, play, quality, bgcolor, base) {
//do: get params and attributes

	var fm = new Object();
	fm.attrs = new Object();
	fm.params = new Object();
	
	//attributes
	fm.attrs['id'] = id;
	fm.attrs['width'] = width;
	fm.attrs['height'] = height;
	fm.attrs['type'] = 'application/x-shockwave-flash';
	fm.attrs['data'] = movie;
	if (base != "") { fm.attrs['BASE'] = base; }
	
	//parameters
	fm.params['movie'] = movie;
	if (base != "") { fm.params['BASE'] = base; }
	fm.params['play'] = play;
	fm.params['bgcolor'] = bgcolor;
	fm.params['quality'] = quality;
	fm.params['menu'] = 'false';
	fm.params['allowScriptAccess'] = 'always';
	fm.params['wmode'] = 'transparent';
	
	return fm;
}

function fmObjectGenerate(attrs, params) {
//do: generate the flash object
	
	var str = '<object ';

	for (var i in attrs) 
		str += i + '="' + attrs[i] + '" ';
	str += '>';
	
	for (var i in params) 
		str += '<param name="' + i + '" value="' + params[i] + '" />';
	str += '</object>';
	
	return str;
}

var agt = navigator.userAgent.toLowerCase();
var isIE	= (navigator.appVersion.indexOf("MSIE") != -1) ? true : false;
var isWin	= (navigator.appVersion.toLowerCase().indexOf("win") != -1) ? true : false;
var isOpera = (navigator.userAgent.indexOf("Opera") != -1) ? true : false;
var isSafari = (agt.indexOf("safari") != -1) ? true : false;

function ControlVersion() {
	var version;
	var axo;
	var e;

	try {
		// version will be set for 7.X or greater players
		axo = new ActiveXObject("ShockwaveFlash.ShockwaveFlash.9");
		version = axo.GetVariable("$version");
	} catch (e) {
	}
	
	if (!version)
	{
		try {
			axo = new ActiveXObject("ShockwaveFlash.ShockwaveFlash.8");
			version = axo.GetVariable("$version");
		} catch (e) {
		}
	}
	
	if (!version)
	{
		try {
			axo = new ActiveXObject("ShockwaveFlash.ShockwaveFlash.7");
			version = axo.GetVariable("$version");
		} catch (e) {
		}
	}
	

	if (!version)
	{
		try {
			// version will be set for 6.X players only
			axo = new ActiveXObject("ShockwaveFlash.ShockwaveFlash.6");
			
			// default to the first public version
			version = "WIN 6,0,21,0";

			// throws if AllowScripAccess does not exist (introduced in 6.0r47)		
			axo.AllowScriptAccess = "always";

			// safe to call for 6.0r47 or greater
			version = axo.GetVariable("$version");

		} catch (e) {
		}
	}

	if (!version)
	{
		try {
			// version will be set for 4.X or 5.X player
			axo = new ActiveXObject("ShockwaveFlash.ShockwaveFlash.3");
			version = axo.GetVariable("$version");
		} catch (e) {
		}
	}

	if (!version)
	{
		try {
			// version will be set for 3.X player
			axo = new ActiveXObject("ShockwaveFlash.ShockwaveFlash.3");
			version = "WIN 3,0,18,0";
		} catch (e) {
		}
	}

	if (!version)
	{
		try {
			// version will be set for 2.X player
			axo = new ActiveXObject("ShockwaveFlash.ShockwaveFlash");
			version = "WIN 2,0,0,11";
		} catch (e) {
			version = -1;
		}
	}
	
	return version;
}

function GetSwfVer(){

	// NS/Opera version >= 3 check for Flash plugin in plugin array
	var flashVer = -1;
	
	if (navigator.plugins != null && navigator.plugins.length > 0) {
		if (navigator.plugins["Shockwave Flash 2.0"] || navigator.plugins["Shockwave Flash"]) {
			var swVer2 = navigator.plugins["Shockwave Flash 2.0"] ? " 2.0" : "";
			var flashDescription = navigator.plugins["Shockwave Flash" + swVer2].description;			
			var descArray = flashDescription.split(" ");
			var tempArrayMajor = descArray[2].split(".");
			var versionMajor = tempArrayMajor[0];
			var versionMinor = tempArrayMajor[1];
			if ( descArray[3] != "" ) {
				tempArrayMinor = descArray[3].split("r");
			} else {
				tempArrayMinor = descArray[4].split("r");
			}
			var versionRevision = tempArrayMinor[1] > 0 ? tempArrayMinor[1] : 0;
			var flashVer = versionMajor + "." + versionMinor + "." + versionRevision;
		}
	}
	// MSN/WebTV 2.6 supports Flash 4
	else if (navigator.userAgent.toLowerCase().indexOf("webtv/2.6") != -1) flashVer = 4;
	// WebTV 2.5 supports Flash 3
	else if (navigator.userAgent.toLowerCase().indexOf("webtv/2.5") != -1) flashVer = 3;
	// older WebTV supports Flash 2
	else if (navigator.userAgent.toLowerCase().indexOf("webtv") != -1) flashVer = 2;
	else if ( isIE && isWin && !isOpera ) {
		flashVer = ControlVersion();
	}	
	return flashVer;
}

// When called with reqMajorVer, reqMinorVer, reqRevision returns true if that version or greater is available
function DetectFlashVer(reqMajorVer, reqMinorVer, reqRevision)
{
	versionStr = GetSwfVer();
	
	if (versionStr == -1 ) {
		return false;
	} else if (versionStr != 0) {
		if(isIE && isWin && !isOpera) {
			// Given "WIN 2,0,0,11"
			tempArray         = versionStr.split(" "); 	// ["WIN", "2,0,0,11"]
			tempString        = tempArray[1];			// "2,0,0,11"
			versionArray      = tempString.split(",");	// ['2', '0', '0', '11']
		} else {
			versionArray      = versionStr.split(".");
		}
		var versionMajor      = versionArray[0]; //2
		var versionMinor      = versionArray[1]; //0	
		var versionRevision   = versionArray[2]; //0
		
        	// is the major.revision >= requested major.revision AND the minor version >= requested minor
		if (versionMajor > parseFloat(reqMajorVer)) {
			return true;
		} else if (versionMajor == parseFloat(reqMajorVer)) {
			if (versionMinor > parseFloat(reqMinorVer))
				return true;
			else if (versionMinor == parseFloat(reqMinorVer)) {
				if (versionRevision >= parseFloat(reqRevision))
					return true;
			}
		}
		return false;
	}
}

if ( typeof(picsPath) == 'undefined')
	picsPath = 'http://pic.classistatic.com/image/pics/classifieds/';

var IsNC2_On = false; 



