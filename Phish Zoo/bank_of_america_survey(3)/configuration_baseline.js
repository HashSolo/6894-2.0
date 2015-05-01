/* 
BEGIN Basline-Configuration Script
(this should be called AFTER the section-specific Configuration scripts) 
*/

var lpNumber;
if (typeof(lpNumber)=="undefined") {
	lpNumber = "LPBofA2"; 
}

var lpServerName;
if (typeof(lpServerName)=="undefined") {
	lpServerName = "chat.bankofamerica.com";
}

var tagVars;
var lpUASexistingTagVars=null;
if (typeof(tagVars)=="undefined") {
	tagVars = ""; 
} else {
	lpUASexistingTagVars = tagVars;
}

var lpUASlanguage;
if (typeof(lpUASlanguage)=="undefined") {
	lpUASlanguage = LP_LANGUAGE;
}

var lpUASimagesFolder;
if (typeof(lpUASimagesFolder)=="undefined")
	lpUASimagesFolder = lpUASunit + "-" + lpUASlanguage;
if (lpUASimagesFolder.indexOf("((shared))")==0)
	lpUASimagesFolder = lpUASimagesFolder.substring("((shared))".length);

var lpUASbuttonImagesFolder;
if (typeof(lpUASbuttonImagesFolder)=="undefined")
	lpUASbuttonImagesFolder = lpUASimagesFolder +"/button";
if (typeof(lpUASbuttonType)!="undefined")
	lpUASbuttonImagesFolder += "-" + lpUASbuttonType;

var lpUASInvitationImagesFolder;
if (typeof(lpUASInvitationImagesFolder)=="undefined")
	lpUASInvitationImagesFolder = lpUASimagesFolder +"/invitation";

var lpUASimageURL;
if (typeof(lpUASimageURL)=="undefined" && typeof(lpUASimagesPath)!="undefined")
	lpUASimageURL = lpUASimagesPath + "/" + lpUASInvitationImagesFolder;

var lpPosX = lpUASinvitePositionX;
var lpPosY = lpUASinvitePositionY;

if (typeof(lpUASunit)!="undefined")
	tagVars = tagVars + '&PAGEVAR!unit=' + escape(lpUASunit);

if (typeof(lpUASlanguage)!="undefined")
	tagVars = tagVars + '&SESSIONVAR!language=' + escape(lpUASlanguage);

if (typeof(lpUASwebsite)!="undefined")
	tagVars = tagVars + '&PAGEVAR!UASwebsite=' + escape(lpUASwebsite);

if (typeof(lpUASwebsite)!="undefined" && typeof(lpUASunit)!="undefined")
	tagVars = tagVars + '&PAGEVAR!UASwebsiteUnit=' + escape(lpUASwebsite+":"+lpUASunit);

if (typeof(lpUAScontext)!="undefined")
	tagVars = tagVars + '&PAGEVAR!UAScontext=' + escape(lpUAScontext);

var lpUAScontext;
if (typeof(lpUAScontext)=="undefined") {
	lpUAScontext = document.title;
}

var lpCustomImageURL;
if (typeof(lpCustomImageURL)=="undefined")
	lpCustomImageURL = lpUASimageURL + "/";


if (typeof(lpUASinvitationTitle)!="undefined") {
	lpCustomInvitationTitle = lpUASinvitationTitle;
} else {
	lpCustomInvitationTitle = BASELINE_CUSTOMINVITATIONTITLE;
}

if (typeof(lpUASinvitationCloseTitle)!="undefined") {
	lpCustomInvitationCloseTitle = lpUASinvitationCloseTitle;
} else {
	lpUASinvitationCloseTitle = BASELINE_CLOSEINVITATIONTITLE;
}

if (typeof(lpUASbuttonTitle)=="undefined") {
	lpUASbuttonTitle = BASELINE_BUTTONTITLE;
}


if (typeof(lpSaveRejectStatus)=="undefined") {
	lpSaveRejectStatus = true;
}


if (typeof(lpRejectStateTimeout)=="undefined") {
	lpRejectStateTimeout = 3600;
}




function lpdbButtonAction() {}

/* 
END Configuration Script
*/
