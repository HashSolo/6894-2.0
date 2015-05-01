//  utility javascript file... for common validation and error display

if (top.location != self.location)
{
        top.location = self.location;
}

var opt="toolbar=no,directories=no,location=no,status=no,scrollbars=yes,resizable=yes,copyhistory=no";
var opt480=opt + ",width=600,height=480";
var _error_on_screen = false;
var validationMessages = new Array();

function openHelp(url,custType)
{
    open(url, "helpwindow_" + custType, opt480);
}


function openWindow(url)
{

	open(url, "", "");
}

function add_error(_error)
{
    var el = validationMessages.length;
    var _already_in = false;
    for (n in validationMessages) {
        if( _error.indexOf( validationMessages[n]) > -1)
        {
            _already_in = true;
        }
    }

    if(_already_in == false)
    {
        validationMessages[el] = _error;
    }
}

function show_error_messages()
{
    _error_on_screen = false;
    if( validationMessages.length == 0 )
    {
        return;
    }
    else
    {
        _error_on_screen = true;
    }

    output = '<table width=100% border=0 cellspacing=0 cellpadding=1 summary="This table is used for layout." class=errorbrdr>'
        + '<tr><td>'
        + '<table width=100% border=0 cellspacing=0 cellpadding=7 summary="This table is used for layout." class=errorbrdr><tr>'
        + '<td width=55 valign=top style="background-color: #ffffff;" ><img src="sas-docs/images/icon_alert_error.gif" alt="Error Message"></td>'
        + '<td valign=top style="background-color: #ffffff;"><font class="text1">'

    var error_messages = '<table border=0 cellspacing=0 cellpadding=0 summary="This table is used for layout.">';
    for (n in validationMessages) {
       error_messages = error_messages
                    + '<tr>'
                    + '<td class="text1">' + (validationMessages[n])
                    + '<br><img src="sas-docs/images/clr.gif" width=1 height=4></td></tr>';
    }
    error_messages = error_messages + '</table>';
    output = output + error_messages;
    output = output + '</td>'
            +' </tr>'
            +' </table>'
            +' </td></tr></table>';

    display_error_messages();

    validationMessages = new Array();
}

function display_error_messages()
{
    if (document.getElementById) {
        if (window.HTMLElement) {
            eNode = document.getElementById('error_section');
            if( document.getElementById('server_error_section') )
                eNode = document.getElementById('server_error_section');
            while (eNode.hasChildNodes()) eNode.removeChild(eNode.lastChild);
            var range = document.createRange();
            range.selectNodeContents(eNode);
            eNode.appendChild(range.createContextualFragment(output));
        }
        else {
            if( document.all('server_error_section') )
                document.all('server_error_section').innerHTML = output;
            else
                document.all('error_section').innerHTML = output;
        }
    }
    else if (document.all) {
            if( document.all('server_error_section') )
                document.all('server_error_section').innerHTML = output;
            else
                document.all('error_section').innerHTML = output;
    }
}


/* Multiply the size of the font for each style sheet rule
   for all linked and embedded style sheets.
*/
function multipleFontSize(factor)
{
    var styleSheet;
    var i;
    var done;

    for(i=0;i<document.styleSheets.length;i++)
    {
        styleSheet = document.styleSheets[i].cssText;
        styleSheet = styleSheet.toLowerCase();

        var pattern = /font-size\s*:\s*([\d\.]+)((em)|%)+/g;
        pattern.multiline = true;
        var result;
        done = false;

        while (!done)
        {
            var result = pattern.exec(styleSheet);
            if (result == null)
            {
                done = true;
            }
            else
            {
                strLeft = styleSheet.substring(0, result.index-1);
                strMid = result[0];
                strRight = styleSheet.substring(result.index +  result[0].length);
                size = result[1];
                size *= factor;
                var number_pattern = /(\d*.?\d{0,2})\d*/;
                var number_result = number_pattern.exec(size);
                strMid = strMid.replace(result[1], number_result[1]);
                styleSheet = strLeft + strMid + strRight;
            }
        }
        document.styleSheets[i].cssText= styleSheet;
    }
}

/* Examine the default page font.  If too small,
   increase by a percent factor
*/
function examineFontSize(ref)
{
    if (document.getElementById)
    {
        if (document.getElementById(ref) )
        {
            if (document.getElementById(ref).currentStyle)
            {
                var size = document.getElementById(ref).currentStyle.fontSize;
                var index = size.indexOf("pt");
                var newstr = parseInt(size.substr(0,index));

                if (newstr < 10)
                {
                    multipleFontSize(1.3);
                }
                else if (newstr < 12)
                {
                    multipleFontSize(1.2);
                }
            }
        }
    }
}


function create_button(btnText, btnHref, css_class, onclick_evt, onmouseover_evt, onmouseout_evt, tabindex)
{
	getButton(btnText, btnHref, "", onclick_evt, "sas-docs/images/", btnText,  css_class, tabindex)
}

function getButton(btnText, btnHref, btnTarget, btnOnClick, baseURL, btnTitle,  css_class, tabindex)
{
	/// Note:  an extra, optional argument may be passed to this function
	/// which specifies the window that should be written to.  If the
	/// argument is not supplied, the current document window is used.
	if (getButton.arguments.length > 8)
	{
		doc = getButton.arguments[8];
	}
	else
	{
		doc = document;
	}

	var t = "";

	t = "<table border=0 cellpadding=0 cellspacing=0 summary=\"\" class=\"" + css_class + "\"><tr><td width=2 rowspan=2 class=\"" + css_class + "-left\"><img src='" + baseURL + "clr.gif' alt=\"\" width=2 height=2><\/td><td class=\"" + css_class + "-top\"><img src='" + baseURL + "clr.gif' alt=\"\" width=1 height=2><\/td><td width=2 rowspan=3 class=\"" + css_class + "-right\"><img src='" + baseURL + "clr.gif' alt=\"\" width=2 height=2><\/td><\/tr><tr><td nowrap><div class=\"" + css_class + "\" ";
	t = t + "><a href=\"" + btnHref + "\" target=\"" + btnTarget + "\" class=\"" + css_class  + "\" "
		+ " onFocus='hover(this, \"" + css_class + "-over\");window.status=\"" + btnTitle + "\";'"
		+ " onBlur='hover(this, \"" + css_class + "\");window.status=\"\";'"
		+ " title=\"" + btnTitle + "\" ";
	if (btnOnClick) { t = t + " \'" + btnOnClick + "\'"; }
	if (tabindex) { t = t + " tabindex=\"" + tabindex + "\""; }

	t = t + ">&nbsp;" + btnText + "&nbsp;<\/a><\/div><\/td><\/tr><tr><td colspan=2 class=\"" + css_class +  "-right\"><img src='" + baseURL + "clr.gif' alt=\"\" width=1 height=2><\/td><\/tr><\/table>";

	doc.write(t);
}


function hover(ref, classRef)
{
	eval(ref).className = classRef;
}

function checkForCookies()
{
	var today = new Date();
	var expire = new Date();
	expire.setTime(today.getTime() + 3600000 * 24);
	document.cookie = "cookieEnabledPersist=true; expires=" + expire.toGMTString();
	document.cookie = "cookieEnabledSession=true";
	var cookieString = ""+document.cookie;
	if (cookieString.indexOf("cookieEnabledPersist")==-1 || cookieString.indexOf("cookieEnabledSession")==-1)
	{
		document.location.href = "noCookies.do";
	}
}

function loadGating()
{
	checkForCookies();
}

function displayDynamicText(output_text)
{
    if( document.getElementById )
    {
        document.getElementById("dynamicMsg").innerHTML = output_text;
    }
    else if( document.all )
    {
        document.all.dynamicMsg.innerHTML = output_text;
    }
    return;
}

// set the correct style sheet for the browser

var platform= navigator.platform.toLowerCase(); // Operating system
var appName = navigator.appName;                // Browser name
var appVer= parseInt(navigator.appVersion);     // Browser versionnumber

function checkBrowser(doc)
{
    if (platform.indexOf("win") != -1)  // Windows platform
    {
        if (appName == "Netscape" && appVer >= 5) // Netscape 6.x
        {
            doc.writeln('<link rel="stylesheet" type="text/css" href="/sas-docs/stylesheets/win_ns6.css">');
        }
        else if (appName == "Netscape" && appVer >= 4) // Netscape 4
        {
            doc.writeln('<link rel="stylesheet" type="text/css" href="/sas-docs/stylesheets/win_ns4.css">');
        }
        else if (appName == "Microsoft Internet Explorer" && appVer >= 4 &&  navigator.appVersion .indexOf("MSIE 4") != -1) // IE 4.x
        {
            doc.writeln('<link rel="stylesheet" type="text/css" href="/sas-docs/stylesheets/win_ie4.css">');
        }
        else if (appName == "Microsoft Internet Explorer" && appVer >= 4) // IE 4.x +
        {
            doc.writeln('<link rel="stylesheet" type="text/css" href="/sas-docs/stylesheets/win_ie.css">');
        }
        else if (appName == "Opera")  // Opera 5.x, 6.x
        {
            doc.writeln('<link rel="stylesheet" type="text/css" href="/sas-docs/stylesheets/win_opera.css">');
        }
        else  // All other win browsers
        {
            doc.writeln('<link rel="stylesheet" type="text/css" href="/sas-docs/stylesheets/win_ie.css">');
        }
    }
    else if (platform.indexOf("mac") != -1)    // Mac platform
    {
        if (appName == "Netscape" && appVer >= 5) // Netscape 6.x
        {
            doc.writeln('<link rel="stylesheet" type="text/css" href="/sas-docs/stylesheets/mac_ns6.css">');
        }
        else if (appName == "Netscape" && appVer >= 4) // Netscape 4
        {
            doc.writeln('<link rel="stylesheet" type="text/css" href="/sas-docs/stylesheets/mac_ns4.css">');
        }
        else if (appName == "Microsoft Internet Explorer" && appVer >= 4) // IE 4.x
        {
            doc.writeln('<link rel="stylesheet" type="text/css" href="/sas-docs/stylesheets/mac_ie.css">');
        }
        else if (appName == "Opera")  // Opera 5.x, 6.x
        {
            doc.writeln('<link rel="stylesheet" type="text/css" href="/sas-docs/stylesheets/mac_ie.css">');
        }
        else  // All other browsers
        {
            doc.writeln('<link rel="stylesheet" type="text/css" href="/sas-docs/stylesheets/mac_ie.css">');
        }
    }
    else if (platform.indexOf("os2") != -1)    // os2 platform
    {
        if (appName == "Netscape" && appVer >= 4) // Netscape 4
        {
            doc.writeln('<link rel="stylesheet" type="text/css" href="/sas-docs/stylesheets/os2_ns4.css">');
        }
        else
        {
            doc.writeln('<link rel="stylesheet" type="text/css" href="/sas-docs/stylesheets/win_ie.css">');
        }
    }
    else    // all other platforms
    {
        if (appName == "Netscape" && appVer >= 4) // Netscape 4
        {
            doc.writeln('<link rel="stylesheet" type="text/css" href="/sas-docs/stylesheets/mac_ns4.css">');
        }
        else
        {
            doc.writeln('<link rel="stylesheet" type="text/css" href="/sas-docs/stylesheets/win_ie.css">');
        }
    }
}

