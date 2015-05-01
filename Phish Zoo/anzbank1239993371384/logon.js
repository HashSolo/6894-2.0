function isValidANZCRN (testCRN)
{
   if ((testCRN.length != 9) && (testCRN.length != 15) && (testCRN.length != 16))
   {
      alert("The Customer Registration Number that you have entered is either too long or too short.\nPlease re-enter this number, carefully checking the digits against your card or other record.\nIf you are a customer of ANZ New Zealand, please visit www.anz.co.nz");
      return false;
   }
   if (isDigit(testCRN) != true)
   {
      alert("The Customer Registration Number that you have entered contains non-numeric characters.\nPlease re-enter this number, carefully checking the digits against your card or other record.");
      return false;
   }
   return true;
}

function isValidTelecode (testTelecode)
{
   if (testTelecode.length < 4 || testTelecode.length > 7)
   {
      alert("Your Telecode should have a minimum of 4 and a maximum of 7 digits.\nPlease re-enter the number, carefully checking the digits.");
      return false;
   }
   if (isDigit(testTelecode) != true)
   {
      alert("Your Telecode should only contain numbers.\nPlease re-enter this number, carefully checking the digits.");
      return false;
   }

   return true;
}

function isValidPassword (checkString)
{
	if ((checkString.length == 4) || (checkString.length == 5))
	{
		alert("You have entered your password incorrectly. \nYour password must be between 8 and 16 characters in length and contain at least one number and one letter.  \nIf you are registering for Internet Banking for the first time, \nplease select the 'Register' button on the Homepage.\nOtherwise please check your password and try again.");
		return false;
	}
	else
	{
		if (checkString.length < 8)   
		{
			alert("You have entered your password incorrectly. \nYour password must be between 8 and 16 characters in length and contain at least one number and one letter. \nPlease re-enter your password.");
			return false;
		}

		if (checkString.length > 16)   
		{
			alert("You have entered your password incorrectly. \nYour password must be between 8 and 16 characters in length and contain at least one number and one letter. \nPlease re-enter your password.");
			return false;
		}

		var alphaFlag = false;
		var numericFlag = false;

		for (var i=0; i<checkString.length; i++)
		{
			testChar = checkString.charAt(i);
			if (isLetter(testChar))
			{
				alphaFlag = true;
			}
			else if (isDigit(testChar))
			{
				numericFlag = true;
			}
			else
			{
				alert("You have entered your password incorrectly. \nYour password must be between 8 and 16 characters in length and contain at least one number and one letter. \nPlease re-enter your password.");
				return false;
			}
		}

		if(!alphaFlag)
		{
			alert("You have entered your password incorrectly. \nYour password must be between 8 and 16 characters in length and contain at least one number and one letter. \nPlease re-enter your password.");
			return false;
		}
		if(!numericFlag)
		{
			alert("You have entered your password incorrectly. \nYour password must be between 8 and 16 characters in length and contain at least one number and one letter. \nPlease re-enter your password.");
			return false;
		}
	}
	return true;
}