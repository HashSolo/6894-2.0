<!--

function OnClickHandler()
{
	var agent = navigator.userAgent.toLowerCase();
	
	if ( (agent.indexOf("msie") == -1) || (agent.indexOf("mac") == -1) )
	{	
		var el=null;
		var flag=true;
		el = event.srcElement;
		while (flag && el)   
		{
		  if (el.tagName == "A")
		  { 
		    flag=false;
		    if (el.protocol == "javascript:")
		    {
		      execScript(unescape(el.href), "javascript");
		      window.event.returnValue = false; 
		    }
		  } 
		  else 
		    el = el.parentElement; 
		}
	}
} // end OnClickHandler()

document.onclick = OnClickHandler; // set the On click handler for the document 
//-->

