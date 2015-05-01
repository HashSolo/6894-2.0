/* Touch Clarity logging request. http://www.touchclarity.com
 * Site identifier code for tag testing
 * Copyright (c) Touch Clarity Ltd 2001-2004. All rights reserved. Patent Pending.
 *
 * Change the value of tc_logging_active to switch off logging on the site.
 */

if (typeof tc_logging_active == 'undefined')
  tc_logging_active = true;

tc_site_id = 145;
//tc_server_url = "test2.tc-prelive.com";
tc_server_url = "hsbc.touchclarity.com";
tc_log_path = window.tc_theme_path || "/1/themes/html/hsbcukcommon/touchclarity" ; /* Static path to files under theme */


/* #390 Customisations for Portlet environment set in page with <c:out>
 * [0] image path value - static portlet assets are loadbalanced via URL prefixes
 * [1] jsessionid value - under OLB where jsesionids are required for all links 
 */
var tc_image_path = ""; /* Empty strings to alow for use in CO's when not set */
var tc_jsessionid = "";

(function(){ /* Anon funtion to keep namespace clear of temp vars */
if(typeof tc_image_path_and_jsessionid != 'undefined' && tc_image_path_and_jsessionid.constructor == String && tc_image_path_and_jsessionid != ''){
  
  var tc_temp_vars_array = tc_image_path_and_jsessionid.split(/[;?]/); /* Split on the ';' and then strip any query params */
  if (tc_temp_vars_array[0] == null){ 
    tc_temp_vars_array[0] = ""; /* remove values we cannot use */
  }
  if (tc_temp_vars_array[1] == null){ 
    tc_temp_vars_array[1] = "";
  } else {
    tc_temp_vars_array[1] = ";"+tc_temp_vars_array[1];
  }
  tc_image_path = tc_temp_vars_array[0];
  tc_jsessionid = tc_temp_vars_array[1]; /* add the semi-colon back on to be reused */
}
})();

/* #502 first time / returning visitors
 */
// if the user clicks the login button then record customer cookie
(function () {
  var readCookie = function(name){
    var nameEQ = name + "=";
    var ca = document.cookie.split(';');
    for(var i=0;i < ca.length;i++) {
      var c = ca[i];
      while (c.charAt(0)==' ') c = c.substring(1,c.length);
      if (c.indexOf(nameEQ) == 0) return c.substring(nameEQ.length,c.length);
    }
    return null;
  };
  var setCookie = function(name,value,days){
    if (days) {
      var date = new Date();
      date.setTime(date.getTime()+(days*24*60*60*1000));
      var expires = "; expires="+date.toGMTString();
    }
    else var expires = "";
    document.cookie = name+"="+value+expires+"; path=/";
  };
  window.tc_extra_info=window.tc_extra_info || "";
  var isCustomer = (tc_extra_info.indexOf('tc_isCustomer=true') > -1);
  var returning = readCookie("tc_returning");
  var now=new Date();
  if (isCustomer || "customer" == returning) {
	  tc_extra_info += "&segment=customer";
	  setCookie("tc_returning","customer",720);
  } else if ("returning" == returning) {
	  tc_extra_info += "&segment=returning";
	  setCookie("tc_returning","returning",720);
  } else if (returning && returning.indexOf("firsttime:")==0) {
	  var timestamp = returning.split(":")[1];
	  if (timestamp<(now.getTime()-(60*30*1000))) {	/* set as returning visitor after 30mins/2nd session */
		  tc_extra_info += "&segment=returning";
		  setCookie("tc_returning","returning",720);
	  } else {
		  tc_extra_info += "&segment=firsttime";
		  setCookie("tc_returning","firsttime:"+now.getTime(),720);
	  }
  } else {
	  setCookie("tc_returning","firsttime:"+now.getTime(),720);
	  var readAgain=readCookie("tc_returning");
	  tc_extra_info += "&segment="+(readAgain?"firsttime":"nocookies");
  }
})();


document.write("<scr"+"ipt language='JavaScript' type='text/javascript' src='"+tc_log_path+"/logging-code.js'></scr"+"ipt>\n");
