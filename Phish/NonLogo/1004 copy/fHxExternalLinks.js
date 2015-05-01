function LoggedOn()
{
	var HxCookies = document.cookie;
	var HxCookieFound = HxCookies.indexOf("ONTCred=");
	var bReturn = false;

	if (HxCookieFound != -1)
		{
		if (typeof document.frmHeaderButtons != "undefined")
		{
		if (typeof document.frmHeaderButtons.btnSignOff != "undefined")
			{
				bReturn = true;
			}
		}
	}
	
	return bReturn;
}

function openNonHxLink(sURL, sLinkName, sAppRoot)
{
	openLink(sURL, sLinkName, 
			 "Please be aware that by selecting this link you will be leaving Online Services\nwhich is a secure site.  The linked site is operated by persons outside the\nHBOS Group and we are not responsible for the privacy practices, content or\nproducts and services provided through this site.", 
			 false, sAppRoot, "");
}

function openHxLink(sURL, sLinkName, sAppRoot)
{
	openLink(sURL, sLinkName, "", true, sAppRoot, "", true);
}

function openLink(sURL, sLinkName, sWarn, bLocationBox, sAppRoot, sShowMessage, bHxLink)
{

	var bContinue = true;
	if (sWarn != "")
	{
		bContinue = confirm (sWarn);
	}

	if (bContinue)
	{
		if (LoggedOn())
		{
			var sDom = getDomain();
			var hostParts = "/";
			if(sDom.length > 1)
			{
				hostParts = "https://www" + getDomain() + "/";
			}
			
			if (bHxLink == true)
			{
            if (confirm ("To take you to your chosen page: - \nClick 'OK' to sign out of our online service first (recommended).\nClick 'Cancel' to remain logged in and we will automatically sign you out after 10 minutes of inactivity."))
				{
					removeOntCredCookie();
					top.location.href = hostParts + "_mem_bin/signoff.asp";
				}
			}
			else
			{
				if (confirm ("Would you like to sign-off from the online service now?  (If you choose 'Cancel', your sign-on will expire after 10 minutes of inactivity and you will then need to sign-in again to continue using the online service.)"))
				{
					removeOntCredCookie();
					top.location.href = hostParts + "_mem_bin/signoff.asp";
				}
			}
		}
		else
		{
			if (!(confirm ("Please be aware that by selecting this link you will be leaving Online Services\nwhich is a secure site.")))
			{
				return false;	
			}
		}
		
		var browserVer = navigator.appVersion;
		var browserName = navigator.appName;
		var features = "resizable=1,scrollbars=1,history=0,status=1,directories=0,menubar=1,toolbar=1";
		if (bLocationBox)
		{
			features += "location=1";
		}
		else
		{
			features += "location=0";
		}
		
		var newWin = window.open(sURL, sLinkName, features);
		if(browserName == "Netscape") 
		{
			newWin.focus();
		}
	}
}

function openHxLinkEx(sURL, sLinkName, sAppRoot, sShowMessage)
{
	openLink(sURL, sLinkName,  "", true, sAppRoot, sShowMessage, true)
}

function openPublicLink(sURL, sLinkName, sFeatures)
{
	var browserVer = navigator.appVersion;
	var browserName = navigator.appName;
	
	if(sFeatures=="")
	{
		sFeatures = "resizable=1,scrollbars=1,history=0,status=1,directories=0,menubar=1,toolbar=1,location=0";
	}
	var newWin = window.open(sURL, sLinkName, sFeatures);
	if(browserName == "Netscape") 
	{
		newWin.focus();
	}
}

function insuranceSecurityLink(sURL, sLinkName)
{

	var browserName = navigator.appName;
	var features = "resizable=1,scrollbars=1,history=0,status=1,directories=0,menubar=1,toolbar=1,location=1";
		
	var newWin = window.open(sURL, sLinkName, features);
	if(browserName == "Netscape") 
	{
		newWin.focus();
	}
	
}

function openLinkWithMessage(sURL, sLinkName, sMessage)
{
	if (confirm(sMessage))
	{
		var browserVer = navigator.appVersion;
		var browserName = navigator.appName;
		var features = "resizable=1,scrollbars=1,history=0,status=1,directories=0,menubar=1,toolbar=1,location=0";
		var newWin = window.open(sURL, sLinkName, features);
		if(browserName == "Netscape") 
		{
			newWin.focus();
		}
	}
}
function removeOntCredCookie()
{
	var hostName = getDomain();
	var dt = new Date();
	dt.setFullYear(dt.getFullYear() + 1);
	var ontCredCookie = "ONTCred=-X-; expires=" + dt.toGMTString() + ";path=/;domain=" + hostName;
	document.cookie = ontCredCookie;
}

function getDomain()
{
	var hostParts;
	var hostName = '/';
	hostParts = document.location.host.split('.');

	var l = hostParts.length;
	var ret_val = '';
	for(var i=1; i<l ; i++)
	{
		ret_val = ret_val + '.' + hostParts[i];	
	}
	
	return(ret_val);
}
function openMemberLink(sURL, sLinkName, sAppRoot)
{
	openMemberOfferLink(sURL, sLinkName, "", true, sAppRoot, "", true);
}
function openMemberOfferLink(sURL, sLinkName, sWarn, bLocationBox, sAppRoot, sShowMessage, bHxLink)
{

	var bContinue = true;
	if (sWarn != "")
	{
		bContinue = confirm (sWarn);
	}

	if (bContinue)
	{
		if (LoggedOn())
		{
			var sDom = getDomain();
			var hostParts = "/";
			if(sDom.length > 1)
			{  
				hostParts = "https://www" + getDomain() + "/";
			}
			
		}
	
		var browserVer = navigator.appVersion;
		var browserName = navigator.appName;
		var features = "resizable=1,scrollbars=1,history=0,status=1,directories=0,menubar=1,toolbar=1";
		if (bLocationBox)
		{
			features += "location=1";
		}
		else
		{
			features += "location=0";
		}
		
		var newWin = window.open(sURL, sLinkName, features);
		if(browserName == "Netscape") 
		{
			newWin.focus();
		}
	}
}