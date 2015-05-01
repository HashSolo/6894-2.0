var platform	   = navigator.platform.toLowerCase();	// Operating system
var userAgent     = window.navigator.userAgent;       // Browser user agent string

/* Multiple the size of the font for each style sheet rule 
   for all linked and embedded style sheets
*/
function multipleFontSize(factor)
{
   var styleSheet;
   var i;
   var done;

   if (platform.indexOf("win") != -1 && userAgent.indexOf("MSIE 5.2") == -1 && userAgent.indexOf("MSIE 5.1") == -1 && userAgent.indexOf("MSIE 5.0") == -1 && userAgent.indexOf("MSIE 4") == -1) 
   {
      for(i=0;i<document.styleSheets.length;i++)
      {
         styleSheet = document.styleSheets[i].cssText;
         styleSheet = styleSheet.toLowerCase();
         
         var pattern = /font-size\s*:\s*([\d\.]+)((em)|%)+/g;
         pattern.multiline = true;
         var result;
         done = false;

         while (!done)
         {
            var result = pattern.exec(styleSheet);
            if (result == null)
            {
               done = true;
            }
            else
            {
               strLeft = styleSheet.substring(0, result.index-1);
               strMid = result[0];
               strRight = styleSheet.substring(result.index +  result[0].length);
               size = result[1];
               size *= factor;
               var number_pattern = /(\d*.?\d{0,2})\d*/;
               var number_result = number_pattern.exec(size);
               strMid = strMid.replace(result[1], number_result[1]);
               styleSheet = strLeft + strMid + strRight;
            }  // end if-else
          } // end while
         document.styleSheets[i].cssText= styleSheet;
      } 
   } 
}  

/* Examine the default page font 
   if too small, increase by a percent factor */
function examineFontSize(ref){
	if (document.getElementById)
		if (document.getElementById(ref).currentStyle){
			var size = document.getElementById(ref).currentStyle.fontSize;
			var index = size.indexOf("pt");
			var newstr = parseInt(size.substr(0,index));
			if (newstr < 10){
				multipleFontSize(1.3);
			} else if (newstr < 12) {
				multipleFontSize(1.2);
			}
		}
}

function detect(ref){
	if (document.getElementById)
		if (document.getElementById(ref).currentStyle) { 
			var size = document.getElementById(ref).currentStyle.fontSize;
			var index = size.indexOf("pt");
			var newstr = parseInt(size.substr(0,index));
			if (newstr < 12) {   
				document.write('<br \/><table summary=\"Layout Table\" width=\"100%\" cellpadding=\"5\" cellspacing=\"0\" border=\"1\" bordercolor=\"#ff0000\"><tr><td><div style=\"font-size: 10pt; font-weight: bold;\"><h2 class=\"module-title\">Text Size Too Small?<\/h2><p style=\"font-size: 9pt; font-color: #000000; font-weight: normal; margin-top: 1em; margin-bottom: .5em;\">Follow these steps to increase your text size.<\/p><ol style=\"font-size: 9pt; font-color: #000000; font-weight: normal; margin-top: .5em; margin-bottom: .5em; margin-right: .5em; margin-left: 2em;\"><li>Click <strong>View<\/strong> on your browser tool bar.<\/li><li>Select <strong>Text Size<\/strong>.<\/li><li>Select the text size you want.<\/li><\/ol><\/div><\/td><\/tr><\/table><br \/>');
			}
		}
}
