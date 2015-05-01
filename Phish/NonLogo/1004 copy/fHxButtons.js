//<SCRIPT Language='JavaScript'>

/*
'***************************************************************************
' If all you want to disable all the buttons on all forms this will do the
' processing for you
'
' e.g. HxDisableAllFormsButtons()
'
'***************************************************************************
*/
function HxDisableAllFormsButtons()
{
	for (var i=0; i<document.forms.length; i++)
	{
		HxDisableAllButtons(document.forms[i], "", "", "", "", "", "", "", true, "", "", "")
	}
}

/*
'***************************************************************************
' If all you want to disable all the buttons in a named form this will do 
' the processing for you
'
' e.g. HxDisableAllButtons(frmMain)
'      HxDisableAllButtons(frmHeaderButtons)	   
'
'***************************************************************************
*/
function HxDisableAllButtons(frmForm)
{	
	for (var i=0; i<frmForm.elements.length; i++)
	{
		if (frmForm.elements[i].type == "submit" || 
		    frmForm.elements[i].type == "reset" ||
		    frmForm.elements[i].type == "button")
		{
			HxDisableButton(frmForm.elements[i], "", "", "", "", "", "", "", true, "", "", "")
		}
		
		if (frmForm.elements[i].type == "text")
		{
			HxDisableTextBoxSubmit(frmForm.elements[i]);
		}		
	}	
	HxWindowFocus();
}

function HxDisableTextBoxSubmit(objInputText)
{
    objInputText.onkeypress = function () { return false; };
}

function HxWindowFocus()
{
	if (typeof(window.focus) != 'undefined') 
	{
		window.focus();
	}			
}

/*
'***************************************************************************
' If all you want to do is disable a button then this is a nice simple
' wrapper for the more feature-rich click handler function
'***************************************************************************
*/
function HxDisableButton(objButton)
{
	HxButtonOnClick(objButton, "", "", "", "", "", "", "", true, "", "", "")  	
}


/*
'***************************************************************************
' This function allows you to handle button clicks in a fairly rich manner
'
' Parameters:
'   sPreValidationFunction
'                  - User defined function to execute first
'   sPostValidationFunction
'                  - User defined function to execute last
'	 sPreAlertText  - Alert text to display prior to any processing
'	 sPostAlertText - Alert text to display after other processing
'	 sConfirmText   - Confirmation prompt - must OK to continue processing  
'	 sNewbuttonText - Button text to replace with current text
'	 sNewButtonStyle- Style for button to replace current one
'	 bDisableButton - True or False
'	 sSpareX        - For future use
'
' Order of Execution:
'	 Pre-Validation Function
'	 Confirmation Text
'	 Pre-Processing Alert Text
'	 Disable Button
'    Post Validation Function
'    Post-Processing Alert Text
'
'***************************************************************************
*/
function HxButtonOnClick(objButton, 
						 sPreValidationFunction,
						 sPostValidationFunction,
                         sPreAlertText, 
                         sPostAlertText, 
                         sConfirmText, 
						 sNewButtonText, 
						 sNewButtonStyle,
						 bDisableButton, 
                         sSpare1, sSpare2, sSpare3) 
{
                        
	var bRetVal = true;

	// Handle the pre-validation function
	if (sPreValidationFunction.length > 0)
	{
		bRetVal = eval(sPreValidationFunction)
		if (bRetVal == false)
		{
			return (false);
		}
	}
	
	// Handle confirm box - if they click cancel we get out now
	if (sConfirmText.length > 0)
	{
		bRetVal  = confirm (sConfirmText);
		if (bRetVal == false)
		{
			return (false);
		}
	}

	// Display a pre-alert
	if (sPreAlertText.length > 0)
	{
		alert (sPreAlertText);
	}
				    
	// Change button text
	if (sNewButtonText.length > 0)
	{
		objButton.value = sNewButtonText;
	}
				    		 
	// Disable button
	if (bDisableButton == true)
	{
		if (typeof(objButton.disabled) != 'undefined') 
		{
			objButton.disabled = true;
			if (sNewButtonStyle.length > 0)
			{
				objButton.className = sNewButtonStyle;
			}
			else
			{
				objButton.style.backgroundColor = "silver"
				objButton.style.color = "gray"
				objButton.style.cursor = "wait";
			}
		} 
		else 
		{
			//NS4 Only
		    objButton.onfocus = function () { this.blur(); }
		    objButton.onclick = function () { return false; }
		} 
	}
			
	// Handle the post-validation function
	if (sPostValidationFunction.length > 0)
	{
		bRetVal = eval(sPostValidationFunction)
		if (bRetVal == false)
		{
			return (false);
		}
	}

	// Display post processing alert
	if (sPostAlertText.length > 0)
	{
		alert (sPostAlertText);
	}
	
	// return Yay
	return (true);
}

//</SCRIPT>