//===========================================================================================
// This script handles the browser timeout functionality for SAS.
//
// To initiate the default time out functionality:
// 1. Include this script in the html HEAD area
//    <SCR IPT language="JavaScript" SRC="commontimeout.js"></SCR IPT>
//
// 2. Set the onLoad attribute in the html body tag to call toLoadPage()
//    <BO DY onLoad='toLoadPage();'>
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
                             + ",resizable=0"  + ",width=320" + ",height=210";

var _toImageHtml;
var _toStaticUrl = "";

//------------------------------------------------------------
function toLoadPage()
{
    _toTimeOutOccurred = false;

    if ( this["staticUrl"] )
    {
        _toStaticUrl = staticUrl;
    }
    _toImageHtml = '<IMG SRC="/sas/sas-docs/images/timeout-header.gif" border=0>\n';

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
   // window.document.images.resetTimeout.src = _toResetTimeoutServletURL;

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
        window.document.images.resetTimeout.src = _toResetTimeoutServletURL;
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
	timeoutWarningMsg = "<P class='text2'>Your Online Banking session is about to be timed out. As a security precaution, sessions end after "
			 + _toCalculateMinutes(_toGraceDelayTime + _toDelayTime) + " minutes of inactivity. Click OK to continue your current session.</P>";

	html  = '<HTML>\n';
	html += '<HEAD>';
	html += '   <TITLE>Bank of America | Online Banking | Automatic Sign Off Alert</TITLE>\n';
	html += '   <link rel=stylesheet type="text/css" href="' + getStylePath() + '">\n';
	html += '   <META name=\"description\" ' +
	    '    CONTENT=\"For your security, Bank of America will automatically sign you off Online Banking in '+
			      _toCalculateMinutes(_toGraceDelayTime)+' minutes.\"\n>';
	html += '<script language=\"JavaScript\" type=\"text/JavaScript\">function hover(ref, classRef) { eval(ref).className = classRef; }<\/script>';
	html += '</HEAD>\n\n';
	html += '<body bgcolor="#ffffff" link="#0000cc" vlink="#ff0000" alink="#cecece">';
	html += '\n<TABLE BORDER=0 CELLPADDING=0 CELLSPACING=0 BGCOLOR="#FFFFFF" width=300 summary="">';
	html += '\n<TR>\n<td rowspan=5 width=3><img src="/sas/sas-docs/images/clr.gif" width=3 height=1>\n</td>';
	html += '\n<td bgcolor="#FFFFFF" width="100%">\n' + _toImageHtml +'\n</TD>\n</TR>'; //added banner
	html += '\n<TR>\n<td>\n<img src="/sas/sas-docs/images/clr.gif" width=1 height=5>\n</td>\n</tr>'; //added an empty row
	html += '\n<TR>\n<TD>\n' + timeoutWarningMsg + "\n</TD>\n</TR>\n"; //added message
	html += '\n<TR>\n<td>\n<img src="/sas/sas-docs/images/clr.gif" width=1 height=10>\n</td>\n</tr>'; //added an empty row
	html += '\n<TR>\n<TD align="center">\n<FORM NAME="timeOutForm" METHOD=POST>\n'; //adding the button
	html += '<INPUT TYPE=\"HIDDEN\" name=\"timerreset\" value=\"yes\" > \n';
	_toTimeOutPendWnd = window.open("", "TimeoutPending", _toPopUpWindowOptions, true );
	_toTimeOutPendWnd.document.write(html);
	getButton('OK', 'javascript:window.opener._toResetTimeOut();this.close();', '', '', '/sas/sas-docs/images/', '', 'btn1', '', _toTimeOutPendWnd.document );
	html ="";
	html += '\n</FORM>\n';
	html += '\n</TD>\n</TR>\n</TABLE></CENTER></BODY>\n</HTML>\n';
	_toTimeOutPendWnd.document.write(html);

	_toTimeOutPendWnd.document.close();
	_toTimeOutPendWnd.focus();

	_toGraceTimerID = setTimeout('_toSignalTimeoutOccurred()',_toGraceDelayTime);
}

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

    // Build new Timed Out Window

    timedOutMsg = "Your Online Banking session has been timed out. As a security precaution, sessions are ended after "
                    + _toCalculateMinutes(_toDelayTime + _toGraceDelayTime) +
                     " minutes of inactivity. You can sign in again using Online Banking.";

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
