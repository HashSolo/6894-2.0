function OpenWindowNewSession(s)
{
   //function to open new browser

	if (navigator.userAgent.substring(0, 12) != "Mozilla/2.0") {
	   if ((screen.Height >= 0) && (screen.Width >= 0))
   	{
      	var height = screen.Height - 25;
	      var width = screen.Width;
   	}
	   else if ((screen.availHeight >= 0) && (screen.availWidth >= 0))
   	{
      	var height = screen.availHeight - 155;
	      var width = screen.availWidth - 10;
   	}
	}
	else {
		var height=700;
		var width=550;
	}
		
   newWin = window.open(s, "ANZWIN", "toolbar=0,status=1,location=no,menubar=no,directories=no,scrollbars=yes,resizable=yes,screenX=0,screen=0,left=0,top=0,width=" + width + ",height=" + height);
   
}

function isDigit(str)
{
   if (str.length == 0)
   {
      return false;
   }
   for (var i=0;i < str.length;i++)
   {
      if ((str.substring(i,i+1) < '0') || (str.substring(i,i+1) > '9'))
      {
        return false;
      }
   }
   return true;
}

function isLetter(str)
{
   characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

   if (str.length == 0)
   {
      return false;
   }
   for (var i=0;i < str.length;i++)
   {
      if (characters.indexOf(str.charAt(i)) == -1)
      {
         return false;
      }
   }
   return true;
}

function OpenWindow(s)
{
  //function to open new browser for little Question Mark Icon
  newWin = window.open(s,"ANZWin","toolbar=0,location=0,directories=0,status=yes,menubar=no,scrollbars=yes,resizable=no,screenX=0,screenY=0,left=0,top=0,width=380,height=320");
	newWin.focus();
} 


function OpenWindowBig(s)
{
  //function to open new browser for Help Icon - Bigger browser
  newWin = window.open(s,"ANZWinBig","toolbar=0,location=0,directories=0,status=yes,menubar=no,scrollbars=yes,resizable=yes,screenX=0,screenY=0,left=0,top=0,width=536,height=300");
	newWin.focus();
}

function OpenWindowExit(s)
{
   //function to open new browser

	if (navigator.userAgent.substring(0, 12) != "Mozilla/2.0") {
	   if ((screen.Height >= 0) && (screen.Width >= 0))
   	{
      	var height = screen.Height - 25;
	      var width = screen.Width;
   	}
	   else if ((screen.availHeight >= 0) && (screen.availWidth >= 0))
   	{
      	var height = screen.availHeight - 155;
	      var width = screen.availWidth - 10;
   	}
	}
	else {
		var height=700;
		var width=550;
	}
		
   newWin = window.open(s, "ANZWIN", "toolbar=1,status=1,location=yes,menubar=yes,directories=yes,scrollbars=yes,resizable=yes,screenX=0,screen=0,left=0,top=0,width=" + width + ",height=" + height);
}

function OpenWindowAddress(s)
{
   //function to open new browser

	if (navigator.userAgent.substring(0, 12) != "Mozilla/2.0") {
	   if ((screen.Height >= 0) && (screen.Width >= 0))
   	{
      	var height = screen.Height - 25;
	      var width = screen.Width;
   	}
	   else if ((screen.availHeight >= 0) && (screen.availWidth >= 0))
   	{
      	var height = screen.availHeight - 155;
	      var width = screen.availWidth - 10;
   	}
	}
	else {
		var height=700;
		var width=550;
	}
		
   newWin = window.open(s, "ANZWIN", "toolbar=1,status=1,location=yes,menubar=yes,directories=yes,scrollbars=yes,resizable=yes,screenX=0,screen=0,left=0,top=0,width=" + width + ",height=" + height);
}
