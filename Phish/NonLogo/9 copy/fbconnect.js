function facebook_prompt_permission(permission) 
{
	FB.ensureInit(function() 
	{
		FB.Connect.showPermissionDialog(permission);
	});
}

function facebook_requestSession() 
{
	if (!bburl || bburl == null || bburl == '')
		var loginurl = 'fblogin.php';
	else
		var loginurl = bburl + '/fblogin.php';
		
	FB.ensureInit(function() 
	{ 
		FB.Connect.ifUserConnected(loginurl, function ()
		{
			FB.Connect.requireSession(function ()
			{
				document.location = loginurl;
			});
		});		 
	});
}

//fire when dom is ready
YAHOO.util.Event.onDOMReady(function()
{
	var elements = YAHOO.util.Dom.getElementsByClassName('fbconnect', 'span')
	for (var i = 0; i < elements.length; i++)
	{
		elements[i].innerHTML = '<a href="#" onclick="facebook_requestSession(); return false;"><img id="fb_login_image" src="http://static.ak.fbcdn.net/images/fbconnect/login-buttons/connect_light_medium_long.gif" border="0" alt="Connect"/></a>';
	}
});
