/**********************************************************
 *                                                        *
 *  Copyright ©2005  Corillian Corporation                *
 *                                                        *
 *  All rights reserved.                                  *
 *                                                        *
 *  Highly Confidential.                                  *
 *                                                        *
 *  No portion of this code may be reproduced,            *
 *  transmitted or distributed without the express        *
 *  written permission of Corillian Corporation.          *
 *                                                        *
 **********************************************************/
 
function PopupGuideLines(pageUrl)
{				
	var popupAttributes = 'width=630,height=500,left=0,top=0,screenX=0,screenY=0,status=yes,scrollbars=yes,resizable=no,toolbar=no,menubar=no,location=no';		
	helpWindow = window.open(pageUrl, 'CORIGUIDELINES', popupAttributes);
	helpWindow.focus();
}

function doClose()
{
	try{
		window.open('','_parent',''); 
		window.close();
	}
	catch(err){
		// Do Nothing
	}
}
	
function openCheckImageWindow(url)
{
	var ciw;
	if(ciw != null)
	{
		ciw.close(); 
	}
	ciw = window.open(url, 'CORICHECKIMAGE', 'width=1280,height=1000,left=0,top=0,status=yes,scrollbars=yes,resizable=yes,toolbar=no,menubar=no,location=no,directories=no,copyhistory=no');
	ciw.focus();
}
	
function PopupHelp(pageUrl)
{				
	var popupAttributes = 'width=605,height=420,left=0,top=0,screenX=0,screenY=0,status=yes,scrollbars=yes,resizable=yes,toolbar=no,menubar=no,location=no';		
	helpWindow = window.open(pageUrl, 'CORIHELP', popupAttributes);
	helpWindow.focus();
}

function doPrint()
{
	try{
		window.print();
	}
	catch(err){
		// Do Nothing
	}
}

function doZoomIn()
{
var maxWidth=1200;

var imgElement = document.getElementById("imgCheckImage");
if (imgElement !=null)
	{
		var newWidth =Math.round(imgElement.width*1.2);
		imgElement.style.width = '';
		if(newWidth<maxWidth)
			{
			imgElement.width = newWidth;
			}
		else
			{
			imgElement.width = maxWidth;
			}
	}
}

function doZoomOut()
{
var minWidth=500;
var imgElement = document.getElementById("imgCheckImage");

if (imgElement !=null)
	{
		var newWidth = Math.round(imgElement.width*0.834) ;
		imgElement.style.width = '';
			if(newWidth>minWidth)
			{
			imgElement.width = newWidth;
			}
		else
			{
			imgElement.width = minWidth;
			}
	}
}

function PopupNewWindow(pageUrl, winName)
{				
	var popupAttributes = 'width=630,height=500,left=0,top=0,screenX=0,screenY=0,status=yes,scrollbars=yes,resizable=yes,toolbar=no,menubar=no,location=no';		
	helpWindow = window.open(pageUrl, winName, popupAttributes);
	helpWindow.focus();
}