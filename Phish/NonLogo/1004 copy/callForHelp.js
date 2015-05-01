function callForHelp(sBaseURL) {

  var docLocationDesc;
  var URLElements;
  var newWin;
  var browserVer;
  var browserName;
  var positions;
  var sizes;
  var YPosn;
  var XPosn;
  var winWidth;
  var winHeight;

  //get the URL and split it where each slash occurs
  URLElements = document.location.href.split('/');
  
 //given the split up URL, get the URL without the https://thissite/
  docLocationDesc = URLElements[3]
  for (i=4; i< URLElements.length; i++)
  {
	docLocationDesc = docLocationDesc + '/' + URLElements[i]
  }
  
  browserVer = navigator.appVersion;
  browserName = navigator.appName;

  if (browserVer.indexOf("3") == -1) {

  YPosn = ((screen.availHeight - 380) / 2) - 36;
  XPosn = ((screen.availWidth - 640) / 2);
  YPosn = parseInt(YPosn);
  XPosn = parseInt(XPosn);

// check values are ok
  if(XPosn < 0) {
	XPosn = 0;
	}
  if(YPosn < 0) {
	YPosn = 0;
	}
// establish the positioning syntax
   if(browserName == "Netscape") {
	positions = ",screenX=" + XPosn + ", screenY=" + YPosn;
	} else {
	positions = ",left=" + XPosn + ", top=" + YPosn;
	}
  }
  else
  {
	positions = "";
  }

	winHeight = 380;
	winWidth = 640;

  newWin = window.open("https://www." + sBaseURL + "/help/public/umhelpportal.asp?ra=*" + docLocationDesc, "helpSystem", "width=" + winWidth + ",height=" + winHeight + ",channelmode=0,dependent=0,directories=0,fullscreen=0,location=0,menubar=0,resizable=1,scrollbars=1,status=0,toolbar=1" + positions);

  if(browserName == "Netscape") 
  {
	newWin.focus();
  }
}

function callForHelpEx(sBaseURL, sBID, sHID, bEmail) {

  var newWin;
  var browserVer;
  var browserName;
  var positions;
  var sizes;
  var YPosn;
  var XPosn;
  var winWidth;
  var winHeight;
  var docLocationDesc;
  var URLElements;

  browserVer = navigator.appVersion;
  browserName = navigator.appName;

  if (browserVer.indexOf("3") == -1) {

  YPosn = ((screen.availHeight - 380) / 2) - 36;
  XPosn = ((screen.availWidth - 640) / 2);
  YPosn = parseInt(YPosn);
  XPosn = parseInt(XPosn);

// check values are ok
  if(XPosn < 0) {
	XPosn = 0;
	}
  if(YPosn < 0) {
	YPosn = 0;
	}
// establish the positioning syntax
   if(browserName == "Netscape") {
	positions = ",screenX=" + XPosn + ", screenY=" + YPosn;
	} else {
	positions = ",left=" + XPosn + ", top=" + YPosn;
	}
  }
  else
  {
	positions = "";
  }

  winHeight = 380;
  winWidth = 640;

if (sHID == null)
  {
	sHID = "";
  } else {
  	sHID = "hid=" + sHID;
  }

  if (bEmail == null)
  {
      if (sBaseURL.substring(0, 4) == "http")
      {
          newWin = window.open(sBaseURL + "/help/public/umhelpengine.asp?ra=*&bid=" + sBID + "&" + sHID, "helpSystem", "width=" + winWidth + ",height=" + winHeight + ",channelmode=0,dependent=0,directories=0,fullscreen=0,location=0,menubar=0,resizable=1,scrollbars=1,status=0,toolbar=1" + positions);
      } else {
	  newWin = window.open("https://www." + sBaseURL + "/help/public/umhelpengine.asp?ra=*&bid=" + sBID + "&" + sHID, "helpSystem", "width=" + winWidth + ",height=" + winHeight + ",channelmode=0,dependent=0,directories=0,fullscreen=0,location=0,menubar=0,resizable=1,scrollbars=1,status=0,toolbar=1" + positions);
      }	
  }
  else
  {
    URLElements = document.location.href.split('/');
  
	//given the split up URL, get the URL without the https://thissite/
	docLocationDesc = URLElements[3]
	for (i=4; i< URLElements.length; i++)
	{
	 docLocationDesc = docLocationDesc + '/' + URLElements[i]
	}   
           if (sBaseURL.substring(0, 4) == "http")
           {
         	  newWin = window.open(sBaseURL + "/help/public/umcontactinfo.asp?ra=*" + docLocationDesc +"&bid=" + sBID + "&" + sHID, "helpSystem", "width=" + winWidth + ",height=" + winHeight + ",channelmode=0,dependent=0,directories=0,fullscreen=0,location=0,menubar=0,resizable=1,scrollbars=1,status=0,toolbar=1" + positions);
           } else {
         	  newWin = window.open("https://www." + sBaseURL + "/help/public/umcontactinfo.asp?ra=*" + docLocationDesc +"&bid=" + sBID + "&" + sHID, "helpSystem", "width=" + winWidth + ",height=" + winHeight + ",channelmode=0,dependent=0,directories=0,fullscreen=0,location=0,menubar=0,resizable=1,scrollbars=1,status=0,toolbar=1" + positions);
           }	
  }
  if(browserName == "Netscape") 
  {
	newWin.focus();
  }
}

function callForMCHelp(sBaseURL,sHelpID) {

  var newWin;
  var browserVer;
  var browserName;
  var positions;
  var sizes;
  var YPosn;
  var XPosn;
  var winWidth;
  var winHeight;

  browserVer = navigator.appVersion;
  browserName = navigator.appName;

  if (browserVer.indexOf("3") == -1) {

  YPosn = ((screen.availHeight - 380) / 2) - 36;
  XPosn = ((screen.availWidth - 640) / 2);
  YPosn = parseInt(YPosn);
  XPosn = parseInt(XPosn);

// check values are ok
  if(XPosn < 0) {
	XPosn = 0;
	}
  if(YPosn < 0) {
	YPosn = 0;
	}
// establish the positioning syntax
   if(browserName == "Netscape") {
	positions = ",screenX=" + XPosn + ", screenY=" + YPosn;
	} else {
	positions = ",left=" + XPosn + ", top=" + YPosn;
	}
  }
  else
  {
	positions = "";
  }
  winHeight = 380;
  winWidth = 640;
  newWin = window.open("https://www." + sBaseURL + "/help/public/ummchelp.asp?hid=" + sHelpID, "helpSystem", "width=" + winWidth + ",height=" + winHeight + ",channelmode=0,dependent=0,directories=0,fullscreen=0,location=0,menubar=0,resizable=1,scrollbars=1,status=0,toolbar=1" + positions);
  if(browserName == "Netscape")
  {
	newWin.focus();
  }
}

function callForGenericHelp(sURL) {

  var newWin;
  var browserVer;
  var browserName;
  var positions;
  var sizes;
  var YPosn;
  var XPosn;
  var winWidth;
  var winHeight;

  browserVer = navigator.appVersion;
  browserName = navigator.appName;

  if (browserVer.indexOf("3") == -1) {

  YPosn = ((screen.availHeight - 380) / 2) - 36;
  XPosn = ((screen.availWidth - 640) / 2);
  YPosn = parseInt(YPosn);
  XPosn = parseInt(XPosn);

// check values are ok
  if(XPosn < 0) {
	XPosn = 0;
	}
  if(YPosn < 0) {
	YPosn = 0;
	}
// establish the positioning syntax
   if(browserName == "Netscape") {
	positions = ",screenX=" + XPosn + ", screenY=" + YPosn;
	} else {
	positions = ",left=" + XPosn + ", top=" + YPosn;
	}
  }
  else
  {
	positions = "";
  }
  winHeight = 380;
  winWidth = 640;
  newWin = window.open(sURL, "helpSystem", "width=" + winWidth + ",height=" + winHeight + ",channelmode=0,dependent=0,directories=0,fullscreen=0,location=0,menubar=0,resizable=1,scrollbars=1,status=0,toolbar=1" + positions);
  if(browserName == "Netscape")
  {
	newWin.focus();
  }
}

function callForEmail(sBaseURL, sBID, sHID) 
{
	if (isNaN(sBID))
		sBID=6;
	if (isNaN(sHID))
		sHID=0;
	callForHelpEx(sBaseURL, sBID, sHID,'true')
}
