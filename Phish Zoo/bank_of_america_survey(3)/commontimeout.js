//===========================================================================================
// This script handles the browser timeout functionality for SAS.
//
// To initiate the default time out functionality:
// 1. Include this script in the html HEAD area
//    <SCR IPT language="JavaScript" SRC="commontimeout.js"></SCR IPT>
//
// 2. Set the onLoad attribute in the html body tag to call toLoadPage()
//    <BODY onLoad='toLoadPage();'>
//
// The following variables may be set to override the default behavior of these
// routines
//
//  Variable             Description
// --------------------- ----------------------------------------------------------
// overrideTimeOutMSecs  Number of milliseconds before timeout warning form is
//                       displayed
// timeOutRedirectURL    URL to redirect to if timeout occurs
// timeOutServletURL     URL to Action used when timeout occurs
//===========================================================================================

var _toDelayTime       = 480000; 
var _toGraceDelayTime  = 120000;

var _toTimeOutPendWnd;  // handle to pending window (1st popup)
var _toTimeOutWnd;      // handle to timeout window (2nd popup)

var _toTimeOutID;       // timer handle for Delay Time
var _toGraceTimerID;    // timer handle for Grace period (delay between 1st popup and 2nd)

var _toTimeOutOccurred = false;

var baseUrl = location.href.substring(0,location.href.lastIndexOf('/'));

var _toTimeoutServletURL;
var _toResetTimeoutServletURL;

var _toPopUpWindowOptions    = "toolbar=0" + ",location=0" + ",directories=0"
                             + ",status=0" + ",menubar=0" + ",scrollbars=0"
                             + ",resizable=0"  + ",width=310" + ",height=290";

var _toImageHtml;
var _toStaticUrl = "";
var leftOffset="";
var topOffset="";
var popUpBlocked = false;
//------------------------------------------------------------
function toLoadPage()
{
	
    _toTimeOutOccurred = false;

    if ( this["staticUrl"] )
    {
        _toStaticUrl = staticUrl;
    }
    
    if(this["isTransparentSession"] != null && isTransparentSession)
    {    	
	    _toImageHtml = '<IMG SRC="' + IMG_CIO_TIMEOUT_HEADER + '" border=0>\n';
    }
    else
	{
	    _toImageHtml = '<IMG SRC="/sas/sas-docs/images/timeout-header.gif" alt="Bank of America" border=0>\n';
	}

    // override with application timeout value
    if(this["systemTimeOutMSecs"] != null)
    {
        _toDelayTime = parseInt(systemTimeOutMSecs);
    }

    // override with application grace timeout value
    if(this["systemGraceMSecs"] != null)
    {
        _toGraceDelayTime = parseInt(systemGraceMSecs);
    }

    // override with application reset timeout link
    if(this["systemResetTimeoutServletURL"] != null)
    {
        _toResetTimeoutServletURL = systemResetTimeoutServletURL;
    }
    else
    {
        _toResetTimeoutServletURL = baseUrl + "/resetTimeout.do";
    }
    //window.document.images.resetTimeout.src = _toResetTimeoutServletURL;

    // set timeout URL
    _toTimeoutServletURL = baseUrl + "/timeout.do";
    // Only turn on timer if delayTime is > zero otherwise bypass
    // allows calling form to turn off timeout processing
    if (_toDelayTime > 0) {
        _toSetTimeoutTimer(_toDelayTime);
    }
    
    

}

//------------------------------------------------------------
function _toCalculateMinutes(inMilliseconds) {
    return inMilliseconds/60000;
}

//------------------------------------------------------------
function _toTurnOffTimeOut() {
    // Time out occurred, clear timers
    clearTimeout(_toGraceTimerID);
    clearTimeout(_toTimeOutID);
}

//------------------------------------------------------------
function _toTimeOutRedirect() {
    // Time out occurred, forward browser to target url
    self.location = _toTimeoutServletURL;
}

//------------------------------------------------------------
function _toResetTimeOut() {
    if (! _toTimeOutOccurred) {
        clearTimeout(_toGraceTimerID);
        // Create new Timeout timer
        window.top._toSetTimeoutTimer(_toDelayTime);

        //reset the link in the cssPing pixel of the main window, so it will be the same session
        if(!popUpBlocked)
        {
        window.document.images.resetTimeout.src = _toResetTimeoutServletURL;
        }
    }
    return true;
}

//------------------------------------------------------------
function _toSignalTimeoutPending()
{
	//  Delay time has expired, signal user that they will
	//  be logged off if they dont respond
	//  Give them a grace period prior to logging them off
	var timeoutWarningMsg = "";
	var html = "";
	var title = "";
	var meta = "";
		
	
	if(this["isTransparentSession"] != null && isTransparentSession)
    {
	    timeoutWarningMsg = "<P class='text2'>" + STR_CCTIMEOUTMESSAGE_START + " " +  _toCalculateMinutes(_toGraceDelayTime + _toDelayTime) + " " + STR_CCTIMEOUTMESSAGE_END + "</P>";
		title = STR_CCTIMEOUTMESSAGE_TITLE;
		meta = STR_CCTIMEOUTMESSAGE_META;
    }
    // Card Activation (Feb 2009)
	else if(this["isCreditCardActivation"] != null && isCreditCardActivation)
    {
    	timeoutWarningMsg = "<P class='text2'>" + STR_OLB_ENROLLMENT_TIMEOUTMESSAGE;
		title = STR_OLB_ENROLLMENT_TIMEOUTMESSAGE_TITLE;
		meta = STR_OLB_ENROLLMENT_TIMEOUTMESSAGE_META;
    }  
    else
    {   
		timeoutWarningMsg = "<P class='text2'>" + STR_BANKINGTIMEOUTMESSAGE + "</P>";
 		title = STR_BANKINGTIMEOUTMESSAGE_TITLE;
		meta = STR_BANKINGTIMEOUTMESSAGE_META; 		
	}
	//html  = '<HTML>\n';
	html  = '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">\n';
	html += '<HTML>\n';
	html += '<HEAD>';
	html += '   <TITLE>';
	html += title;
	html += '</TITLE>\n';
	html += '<link rel=stylesheet type="text/css" href="' + GLOBAL_CSS + '">\n';
	if(this["isTransparentSession"] != null && isTransparentSession)
    {    	
	    html += '<link rel=stylesheet type="text/css" href="' + CIO_GLOBAL_CSS + '">\n';
    }
    if(this["isCreditCardActivation"] != null && isCreditCardActivation)
    {
		html += ' <META name=\"description\" ' + 'CONTENT=\"' + meta + '\">\n';
	}
	else
	{	html += '   <META name=\"description\" ' +
	    	'CONTENT=\"' + meta + _toCalculateMinutes(_toGraceDelayTime)+ ' ' + STR_MINUTES + '.\"\n>';
	}
	html += '<script language=\"JavaScript\" type=\"text/JavaScript\">function hover(ref, classRef) { eval(ref).className = classRef; }<\/script>';
	html += '</HEAD>\n\n';
	html += '<body link="#0000cc" vlink="#ff0000" alink="#cecece">';
	html += '<div style="margin: 0px\; padding: 0px\; border: 0px\; width: 310px\; height: 55px\;">' + _toImageHtml + '<\/div>\n';
	html += '<div style="margin: 25px 10px 10px 10px\; padding: 0px\; border: 0px\; width: 290px\; height: 150px\;">\n';
	html += '<h1 class=h2-ada>' + SECURITY_MESSAGE + '</h1>\n';
	html += timeoutWarningMsg; + '\n';
	html += '<\/div>\n';
	html += '<div style="margin: 0px 10px 10px 10px\; padding: 0px\; border: 0px\; width: 290px\; height: 20px\;">\n';
	html += '<FORM NAME="timeOutForm" ACTION="" METHOD=POST>\n'; //adding the button
	html += '<table cellpadding="0" cellspacing="0" border="0" width="270">\n';
	//html += '<FORM NAME="timeOutForm" METHOD=POST>\n'; //adding the button
	html += '<tr><td align="center">\n';
	html += '<INPUT TYPE=\"HIDDEN\" name=\"timerreset\" value=\"yes\" > \n';
	
	_toTimeOutPendWnd = window.open("", "TimeoutPending", _toPopUpWindowOptions, true );
		if(_toTimeOutPendWnd ==null)
	{		
	    if((navigator.userAgent.indexOf("Firefox")!=-1) && (this["isTransparentSession"] != null) && !isTransparentSession)
	    {
		popUpBlocked = true;
		timeOutPopUp();
		}
	}
	else
	{
	
	
	_toTimeOutPendWnd.document.write(html);
	if(this["isCreditCardActivation"] != null && isCreditCardActivation)
	{
		getTwoButtons(BTN_CONTINUE, 'javascript:window.opener._toResetTimeOut();this.close();', '', '', '/sas/sas-docs/images/', '', 'btn2', '', BTN_EXIT, 'javascript:window.opener._toTimeOutRedirect();this.close();', '', '', '/sas/sas-docs/images/', '', 'btn2', '', _toTimeOutPendWnd.document );
	}
	else
	{
		getButton(BTN_OK, 'javascript:window.opener._toResetTimeOut();this.close();', '', '', '/sas/sas-docs/images/', '', 'btn1', '', _toTimeOutPendWnd.document );
		
	}
	html = "";
	html += '<\/td><\/tr>\n';
	//html += '\n<\/FORM>\n';
	html += '<\/table>\n';
	html += '\n<\/FORM>\n';
	html += '<\/div>\n';
	
	_toTimeOutPendWnd.document.write(html);
	
	_toTimeOutPendWnd.document.close();
	_toTimeOutPendWnd.focus();
   }
	_toGraceTimerID = setTimeout('_toSignalTimeoutOccurred()',_toGraceDelayTime);
}
//------------------------------------------------------------


//------------------------------------------------------------
function _toSignalTimeoutOccurred() {
// Grace period expired
//  - Cause 1st popup to signal TimeOut control to logoff session
//  - notify user that they have been timed out with 2nd popup

var timedOutMsg     = "";

    _toTimeOutOccurred = true;
    _toTurnOffTimeOut();

    if (_toTimeOutPendWnd && !_toTimeOutPendWnd.closed)
    {
        _toTimeOutPendWnd.window.close();
    }
     if(popUpBlocked == true && document.getElementById("mypopup")!=null)
	{		
		closeMyPopup();  		
	}
    // Build new Timed Out Window

	if(this["isTransparentSession"] != null && isTransparentSession)
    {
    	timedOutMsg = STR_SECONDARY_CCTIMEOUTMESSAGE_START + " " + _toCalculateMinutes(_toDelayTime + _toGraceDelayTime) + " " + STR_SECONDARY_CCTIMEOUTMESSAGE_END;
    }
    else
    {
    	timedOutMsg = STR_SECONDARY_BANKINGTIMEOUTMESSAGE;
    }
    alert(timedOutMsg);

    // Change timeOutPending window action to TimeOutEntryPoint w/appropriate parms
    // then submit to server
    _toTimeOutRedirect();
    return true;

}

//------------------------------------------------------------
function _toSetTimeoutTimer() {
    // Set the delay timer
    _toTimeoutID = setTimeout("_toSignalTimeoutPending()",_toDelayTime);
}
function timeOutPopUp()
{
	myPopupRelocate(); 
	document.getElementById("mypopup").style.display = "";
	document.getElementById("mypopup").style.top = topOffset+"px";
	document.getElementById("mypopup").style.left = leftOffset+"px";
	
	var layer = document.getElementById('mypopup');
	var iframe = document.getElementById('Myiframe');
	iframe.style.display = "";
	iframe.style.width = layer.offsetWidth;
	iframe.style.height = layer.offsetHeight;
	iframe.style.left = layer.offsetLeft;
	iframe.style.top = layer.offsetTop;	
	
	document.body.onscroll = myPopupRelocate;
	window.onscroll = myPopupRelocate;
}
function myPopupRelocate()
{
 var scrolledX, scrolledY;
 if( document.documentElement && document.documentElement.scrollTop ) 
 {
   scrolledX = document.documentElement.scrollLeft;
   scrolledY = document.documentElement.scrollTop;
 } 
 else if( document.body ) 
 {
   scrolledX = document.body.scrollLeft;
   scrolledY = document.body.scrollTop;
 }
 /*else if( self.pageYOffset ) 
 {
   scrolledX = self.pageXOffset;
   scrolledY = self.pageYOffset;
 }*/
 else if( document.pageYOffset ) 
 {
   scrolledX = document.pageXOffset;
   scrolledY = document.pageYOffset;
 } 
 var centerX, centerY;

 /*if( document.documentElement && document.documentElement.clientHeight ) 
 {
   centerX = document.documentElement.clientWidth;
   centerY = document.documentElement.clientHeight;
 } 
 else if( document.body )   
 {
   centerX = document.body.clientWidth;
   centerY = document.body.clientHeight;
 }
 else if(self.innerHeight )
 {  
   centerX = self.innerWidth;
   centerY = self.innerHeight;
 }*/
 if( document.body ) 
 {
   centerX = document.body.clientWidth;
   centerY = document.body.clientHeight;
 }
 else if( document.documentElement && document.documentElement.clientHeight )
 {
   centerX = document.documentElement.clientWidth;
   centerY = document.documentElement.clientHeight;
 }
 else if( document.innerHeight ) 
 {
   centerX = document.innerWidth;
   centerY = document.innerHeight;
 } 
 leftOffset = scrolledX + (centerX - 250) / 2;
 topOffset = scrolledY + (centerY - 200) / 2;
 document.getElementById("mypopup").style.top = topOffset + "px";
 document.getElementById("mypopup").style.left = leftOffset + "px";

 var layer = document.getElementById('mypopup');
 var iframe = document.getElementById('Myiframe');
 iframe.style.width = layer.offsetWidth;
 iframe.style.height = layer.offsetHeight;
 iframe.style.left = layer.offsetLeft;
 iframe.style.top = layer.offsetTop;

  
}
function closeMyPopup()
{
 document.getElementById("mypopup").style.display = "none";
 document.getElementById('Myiframe').style.display = "none";
 popUpBlocked = false;
}