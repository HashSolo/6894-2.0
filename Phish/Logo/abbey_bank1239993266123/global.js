/* show() - function to show the help for a particular transaction. */
function showHelp(message) {    
	 document.getElementById("message").innerHTML="<SPAN>" + message + "</SPAN>";
	 document.getElementById("help").style.display="inline";
     document.getElementById("help").focus();
    // document.links[2].focus();
}

/*default anchor value to open the help from  information and error pages */
anchor_value = '01';

/* hide() - function to hide the help for a particular transaction. */
function hideHelp() {
     document.getElementById("help").style.display='none';
}

/* function that lanches new window */
function launch(filename, winName, width, height) {
	winInfo = "width=" + width + ",height=" + height + ",top=30" + ",left=30" + ",menubar=yes,toolbar=yes,location=no,directories=no,scrollbars=yes,status=no,resizable=yes";
	myWin = window.open(filename,winName,winInfo);
	myWin.focus();
}

var bProcessingUnderWay = false;
var isIE = false;
var browser = navigator.appName;
if (browser.indexOf("Microsoft") != -1) {
    isIE = true;
}

function preventDoubleClick(e) {
	var target = "";
	var nametarget = "";
	if (isIE) {	     
		var target = event.srcElement.id;
		var nametarget = event.srcElement.name;
		
		if (target == "dblclk") {
			doubleClickProcessing();
		}
		else if(nametarget == "dblclk")
		{
			doubleClickProcessing();
		}
		else {
			return true;
		}
	}     
}

function doubleClickProcessing()
{
	if (bProcessingUnderWay) {
		alert("We are currently processing your request: this may take a few moments.");
		return false;
	} else {
		bProcessingUnderWay = true;
		window.setTimeout("resetFlag()",30000);
		return true;
	}
}

function resetFlag() {
	bProcessingUnderWay = false;
}

if (isIE) {
	document.ondragstart = resetFlag;
	document.onmousedown=preventDoubleClick;
}

/* Function that retrieves data from cookie */
function getCookie(NameOfCookie)
{
    if (NameOfCookie=='navCookie')
    {
       NameOfCookie='eMortgageCookie';
    }    
    if (document.cookie.length > 0)
    {              
        begin = document.cookie.indexOf(NameOfCookie+"=");       
        if (begin != -1)
        {           
            begin += NameOfCookie.length+1;       
            end = document.cookie.indexOf(";", begin);
            if (end == -1) end = document.cookie.length;
	        return unescape(document.cookie.substring(begin, end));
        } 
    }
    return null;
}

/* function that sets cookie */
function setCookie(NameOfCookie, value, expiredays) 
{
   var ExpireDate = new Date();
   ExpireDate.setTime(ExpireDate.getTime() + (expiredays * 24 * 3600 * 1000));
   document.cookie = NameOfCookie + "=" + escape(value) + ';path=/';
}

function init()
{
}

function first_init()
{
         init();
         vhlCookie = getCookie('vhlCookie');
         if (vhlCookie)
         {
             k = vhlCookie.indexOf(",");                      
             if (k != -1)
             {
                 hLink = vhlCookie.substring(0,k);
                 vLink = vhlCookie.substring(k+1);
                 changeMenuAndImage(hLink);
                 changeMenuAndImage(vLink);
             } 
             else
             {
                 vLink = vhlCookie;
                 changeMenuAndImage(vLink);
             }
         }
}

function changeMenuAndImage(link)
{
         vcurrentLink = document.getElementById(link);
         if (vcurrentLink)
         {
             imgName = "img" + link;
             if (document.images[imgName])
             {
                 document.images[imgName].src='/ffStatic/images/diamond.gif';
             }
             vcurrentLink.className = 'menulinkClicked';         
         }
}
