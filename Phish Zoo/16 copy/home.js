

function setLoginFocus()
{
	focalbox = document.getElementById('steamAccountName');
	if ( focalbox )
	{
		focalbox.focus();
	}
}


addEvent( window, 'load', setLoginFocus, false );


var iRefreshes = 0;
function RefreshCaptcha()
{
	// Adding count to prevent caching
	URL = captchaRefreshURL + '?count=' + iRefreshes++;
	createQuery( URL, ReceiveCaptchaRefreshData );
}


function ReceiveCaptchaRefreshData()
{
	if ( req.readyState == 4 )
	{
		if ( req.status == 200 )
		{
			gid = req.responseText;
			if ( gid != -1 ) 
			{
				iSwapFullURL( 'captchaImg', captchaImgURL+'?gid='+gid );
			}
			document.getElementById('captchagid').value = gid;
		}
		// annoying global in our ajax functions, which prevents new ajax calls until
		// reset...
		updateInProgress = false;
	}
}
