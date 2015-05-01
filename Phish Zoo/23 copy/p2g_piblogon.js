
//submit without image
function pibService ()
{
    document.navbarLogon.submit() ;
}


//From navbarlogon (checks whether other field is empty)
function isEmpty()
{
	if ( document.navbarLogon.internetBankingID.value == "" )  //Is this form field blank?
	{
		alert('To enter Internet Banking, please ensure you have entered your User ID.'); //They are both blank so alert.
		return false; // end the if statement
	}	
		else //It is not blank so process
	
		{
			if(navigator.appName.indexOf("Netscape")>(-1))
			{
				 // The other one is not blank so put it's value into this form
				pibwin = window.open( '', '_pib', 'resizable,status,width=790,height=520,x=0,y=0,top=0,left=0' ); // open up a window
				document.navbarLogon.submit(); //submit the value
				setTimeout("clearMe()", 10000); // clear the values from the parent window after 10 seconds
				pibwin.focus(); //make the pop up the focus
				return false; //everything is ok
			}
			else
			{
				 // The other one is not blank so put it's value into this form
				pibwin = window.open( '', '_pib', 'resizable, status=yes, width=740, height=520,top=0,left=0' ); // open up a window
				document.navbarLogon.submit(); //submit the value
				setTimeout("clearMe()", 10000); // clear the values from the parent window
				pibwin.focus(); //make the pop up the focus
				return false; //everything is ok
			}
		}
	
	
}

var pibwin = null; // pib ref window

// IF appVersion is MAC AND IF userAgent is MSIE5 then do nothing.
// IF they are anything else set the fields to ""

function clearMe()
{
	var agt=navigator.userAgent.toLowerCase();
		
	document.navbarLogon.internetBankingID.value = "";
	
}


//register popup window controls
function windowOpen()
{
	if(navigator.appName.indexOf("Netscape")>(-1))
	{
	pibwin = window.open('','_pib','resizable,status,width=790,height=520,top=0,left=0');
	window.focus;
	}
	else
	{
	pibwin = window.open('','_pib','resizable,status,width=740,height=520,top=0,left=0');
	window.focus;
	}
}

//focus cursor in IB main box

