// This Secure window is used for the register logon.

function OpenWindowSecure(s) {
  var width = 675;
  var height = 360;
   s = "https://www.anz.com/INETBANK/" + s;
  newWin = window.open(s, "ANZWIN2", "toolbar=0,status=1,location=no,menubar=no,directories=no,scrollbars=yes,resizable=yes,screenX=0,screenY=0,left=0,top=0,width=" + width + ",height=" + height);
}

// This Secure window is used for the register logon.

function OpenWindowSession(s) {
  if ((screen.Height >= 0) && (screen.Width >= 0)) {
     var height = screen.Height - 75;
      var width = screen.Width - 10;
  }
  else if ((screen.availHeight >= 0) && (screen.availWidth >= 0)) {
      var height = screen.availHeight - 45;
      var width = screen.availWidth - 10;
  }
  s = "https://www.anz.com/INETBANK/" + s;
  newWin = window.open(s, "ANZLOGON_1", "toolbar=0,status=1,location=no,menubar=no,directories=no,scrollbars=yes,resizable=yes,screenX=0,screenY=0,left=0,top=0,width=" + width + ",height=" + height);
}

// This window function is used for external links.
function OpenToolWindow(s) {
  var URL = s	
  if (s.search(".htm") != -1)
  {
	URL = "/common/rightbar/Popup.asp?filename=" + s
  }
  newWin = window.open(URL,"ANZToolWin","toolbar=0,location=0,directories=0,status=yes,menubar=no,scrollbars=1,resizable=yes,width=600,height=550");
}

// This window function is used for external links.
function openPopupWindow(url, name, width, height, returnWindow, kill)
{
	var horizontalOffset, verticalOffset, offsetAmount, windowToReturn, closethiswindow;
		
	offsetAmount = 0;
	
	if(width == -1)
	{
	
  	if ((screen.Height >= 0) && (screen.Width >= 0)) {
     		 height = screen.Height - 75;
      		 width = screen.Width - 10;
 	 }
  	else if ((screen.availHeight >= 0) && (screen.availWidth >= 0)) {
      		 height = screen.availHeight - 45;
      		 width = screen.availWidth - 10;
  	}
	}	

	
	if(height == -1)
	{
	if ((screen.Height >= 0) && (screen.Width >= 0)) {
     		 height = screen.Height - 75;
      		 width = screen.Width - 10;
 	 }
  	else if ((screen.availHeight >= 0) && (screen.availWidth >= 0)) {
      		 height = screen.availHeight - 45;
      		 width = screen.availWidth - 10;
  	}
	}
	
	if(navigator.appName == "Microsoft Internet Explorer")
	{
		horizontalOffset = window.screenLeft + offsetAmount;
		verticalOffset = window.screenTop + offsetAmount;
	}
	else
	{
		horizontalOffset = window.screenX + offsetAmount;
		verticalOffset = window.screenY + offsetAmount;
	}
	
	if(horizontalOffset + width > screen.availWidth || verticalOffset + height > screen.availHeight)
	{
		horizontalOffset = 0;
		verticalOffset = 0;
	}
		
	windowToReturn = window.open(url, name, "toolbar=0,location=0,directories=0,status=yes,menubar=no,scrollbars=1,resizable=yes,screenX=" + horizontalOffset + ",screenY=" + verticalOffset + ",left=" + horizontalOffset + ",top=" + verticalOffset + ",width=" + width + ",height=" + height);
		
	if(kill == -1)
	{
		closethiswindow = self.close();
	}
	
	if(returnWindow == true)
		return windowToReturn;
		
}

// This Secure window is used for the register logon.

function OpenWindowNoAddress(s) {
  if ((screen.Height >= 0) && (screen.Width >= 0)) {
     var height = screen.Height - 75;
      var width = screen.Width - 10;
  }
  else if ((screen.availHeight >= 0) && (screen.availWidth >= 0)) {
      var height = screen.availHeight - 45;
      var width = screen.availWidth - 10;
  }
  s = "http://www.anz.com/" + s;
  newWin = window.open(s, "ANZLOGON_1", "toolbar=0,status=1,location=no,menubar=no,directories=no,scrollbars=yes,resizable=yes,screenX=0,screenY=0,left=0,top=0,width=" + width + ",height=" + height);
}
