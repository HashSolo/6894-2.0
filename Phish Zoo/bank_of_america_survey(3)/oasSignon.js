
MSG0001=STR_OAS_MSG0001;
MSG0001_pm=STR_OAS_MSG0001_PM;
MSG0001_nopc=STR_OAS_MSG0001_NOPC;

MSG0002=STR_OAS_MSG0002;

MSG0003=STR_OAS_MSG0003;

function submitForm() 
{
	document.signonForm.nextAction.value = 'signon';

	if(checkFormInputNSubmit()==false)
	{
		show_error_messages();
		_error_on_screen = false;
		return false;
	}
	return document.signonForm.submit();
}

function checkSitekeyFormInputNSubmit()
{
	// No Id
	// Incorrectly formatted ID

	// netscape 4 - direct submit
	if(navigator.appName.indexOf("Netscape") > -1 && navigator.appVersion.indexOf("4.") > -1)
	{
		return true;
	}

	var erroredInput = false;
	accessID = "";
	accessIDCheck = true;

	if (document.signonForm.savedOnlineID)
	{
		if (document.signonForm.onlineID && (document.signonForm.onlineID.value == "") == false )
		{
			accessID = strip(document.signonForm.onlineID.value, " ");
		}
		else
		{
			accessID = strip(document.signonForm.savedOnlineID.value, " ");
			accessIDCheck = false;
		}
	}
	else
	{
		accessID = strip(document.signonForm.onlineID.value, " ");
	}


	if ( ( accessID.length == 0 || accessID == "" ) )
	{
		if(accessIDCheck)
		{
			add_error(MSG0001_pm);
			show_error_messages();
			return;
		}
		else
		{
			document.signonForm.submit();
		}
	}

	if(accessIDCheck)
	{
		erroredInput = checkAccessID(accessID);
	}

	if(erroredInput)
	{
		show_error_messages();
		return;
	}

	document.signonForm.submit();
}


function checkFormInputNSubmit()
{
	// No Id and Pass
	// ID - No Pass
	// Pass - No ID
	// Either one Misssing

	// Incorrectly formatted ID
	// Incorrectly formatted Pass

	// netscape 4 - direct submit
	if(navigator.appName.indexOf("Netscape") > -1 && navigator.appVersion.indexOf("4.") > -1)
	{
		return true;
	}

	accessID = "";
	accessIDCheck = true;
	if (document.signonForm.savedOnlineID)
	{
		if (document.signonForm.onlineID && (document.signonForm.onlineID.value == "") == false )
		{
			accessID = strip(document.signonForm.onlineID.value, " ");
		}
		else
		{
			accessID = strip(document.signonForm.savedOnlineID.value, " ");
			accessIDCheck = false;
		}
	}
	else
	{
		accessID = strip(document.signonForm.onlineID.value, " ");
	}

	passcodeCheck = false;

	if( document.signonForm.passcode )
	{
		passcodeCheck = true;
	}
	
	passcode = null;
	if(passcodeCheck)
	{
		passcode = strip(document.signonForm.passcode.value, " ");
	}

	if ( ( accessID.length == 0 || accessID == "" ) || 
		( passcodeCheck && ( passcode.length == 0 || passcode == "" ) ) )
	{
		if(accessIDCheck==false)
		{
			if(passcodeCheck == false)
			{
				return true;
			}
			else
			{
				add_error(MSG0001_nopc);
			}
		}
		else
		{
			if(passcodeCheck == false)
			{
				add_error(MSG0001_pm);
			}
			else
			{
				add_error(MSG0001);
			}		
		}
		return false;
	}

	var erroredInput = false;
	if(accessIDCheck)
	{
		erroredInput = checkAccessID(accessID);
	}
	if(passcodeCheck)
	{
		erroredInput = erroredInput || checkPasscode(passcode);
	}

	if(erroredInput)
	{
		return false;
	}
	return true;
}

function checkAccessID(accessID)
{
	
	if (accessID.length < 5 )
	{
		add_error(MSG0002);
		return true;
	}
	if (accessID.length > 32 )
	{
		add_error(MSG0002);
		return true;
	}
	return false;
}

function checkPasscode(passcode)
{
	if (document.signonForm.passcode.value.indexOf(' ') > -1 )
	{
		add_error(MSG0003);
		return true;
	}
	if (passcode.length < 4 )
	{
		add_error(MSG0003);
		return true;
	}
	if (passcode.length > 20 )
	{
		add_error(MSG0003);
		return true;
	}
	return false;
}

function indexOfFirstNotIn(okayChars, inString)
{
	var i;
	for (i=0; i < inString.length; i++)
	{
		var charm = inString.charAt(i);
		if (okayChars.indexOf(charm) == -1)
		{
		  return i;
		}
	}
	return -1;
}

function indexOfLastNotIn(okayChars, inString)
{
	var i;
	for (i = inString.length - 1; i >= 0; i--)
	{
		var charm=inString.charAt(i);
		if (okayChars.indexOf(charm) == -1)   // return this char
		{
		  return i;
		}
	}
	return -1;
}
function strip(string, sChar)
{
	if ((string == null) || (string == ""))
	{
		return "";
	}

	var startIndex = indexOfFirstNotIn(sChar, string);
	var endIndex = indexOfLastNotIn(sChar, string);

	// Confirm that we have at least one non-stripped character.
	if (startIndex == -1)
	{
		return "";
	}

	return string.substring(startIndex, endIndex + 1);
}