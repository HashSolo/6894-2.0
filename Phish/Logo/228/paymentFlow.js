//The following two functions are copied from pp_main.js as part of re-factoring process to corelayout
// Changes the class to hidden which is defined in the css
// as Display: none
// Pass in a comma separated list/array/scalar of ids
function toggleDisplay(tag, display) {
	var a = new Array();
	a = createArray(tag);

	var id = "";
	var show = (display == "show") ? true : false;

	for (var i=0; i<a.length; i++) {
		id = document.getElementById(a[i]);
		if (id) {
			if (show) id.className = "";
			else id.className = "hidden";
		}
	}
}

// Takes a comma-separated argument and creates an array out of it.
// If the argument contains only 1 value (instead of a comma separated list), make that a 1 element array
// Used in FormGroupCreditCard component by the following JS -- toggleDisplay and toggleDisabled
function createArray(myval) {
	var a = new Array();
	var input = myval;

	// if it's not array create one so we can loop over it
	if (input.constructor.toString().indexOf("Array") == -1) {
		// to do - remove any spaces form commas separated list
		a = input.split(",");
	} else {
		a = input;
	}
	return a;
}

/**********************************************************
	JS used for WAX, Web Accept and Wallet 
	Cart Page for continue shopping
**********************************************************/
function continueShopping(shoppingURL, noWinAlert, closeWinAlert) {
	var parentWin = window.opener;

	if (parentWin && !parentWin.closed)	{
		if (shoppingURL && shoppingURL != "") {
			try { // this is needed for same-origin security - some merchant use iframes with third party urls
				parentWin.location.href = shoppingURL; // load the shopping url in the parent
			} catch (e) {
				// nothing to do here - just didn't want to refresh with the shopping URL
			}
		}
		window.close();
	}		
	else if (parentWin && parentWin.closed) { // the window was closed (only works in IE)
		if (shoppingURL && shoppingURL != "") {
			window.location.href = shoppingURL;	
		} else {
			alert(closeWinAlert);
		}
	} 
	else { // same window or window closed in Firefox and Safari
		if (shoppingURL && shoppingURL != "") {
			window.location.href = shoppingURL;
		} else {
			alert(noWinAlert);
		}
	}
	return false;
};

/**********************************************************
 This was carried over from old functionality
To display some specific messaging for AOL users
It's a very basic browser detector
It's rarely necessary to parse the userAgent string
rather it's best to use object detection
**********************************************************/
function getBrowserType() {
	var browserType = "";
	var navStr = navigator.userAgent;
	navStr = navStr.toLowerCase();
	
	switch (true){
		case navStr.search('aol') != -1: 
			browserType = "aol";
		break;
		case navStr.search("msie") != -1:
			browserType = "ie";	
		break;
		case navStr.search("firefox") != -1:
			browserType = "firefox";
		break;
		case navStr.search("safari") != -1:
			browserType = "safari";
		break
	}
	return browserType;
};

/*******************************************************
	This function is to open the Printable Receipt 
	as a popup and to submit the form to the backend
	Cognizant
********************************************************/
function submitPopWindow(frmid, width, height) {
	var features = "resizable=1,scrollbars=1,width=" + width + ",height=" + height;
	var frm = YAHOO.util.Dom.get(frmid);
	var win = window.open("", "popupWin", features)
	frm.target = 'popupWin';

	win.focus();
};

/**********************************************************
	Used to post a form to a popup window
	Cognizant
**********************************************************/
function newFormWin() {
	YAHOO.util.Dom.get("myID").target='_self';
};

/**********************************************************
	2005.12.13 pk
	Used on Billing1C for the login
	Hides and shows information
**********************************************************/
function toggleLogin(myVal) {
	if (myVal == "pp") {
		toggleDisplay("loginFields", "show");
		toggleDisplay("billingInfo,disclaimerText,cnt", "hide");
	}
	else if (myVal == "cc") {
		toggleDisplay("billingInfo,disclaimerText,cnt", "show");
		toggleDisplay("loginFields", "hide");
	}
	else {
		return false;
	}
};

/**********************************************************
	2005.12.13 pk
	Shipping Address update on 
	wa review needs to safe submit the form and
	set a hidden form value
**********************************************************/
function updateShipAddress(frm) {
	var refresh = YAHOO.util.Dom.get("refresh");
	
	// Set hidden form value
	// So backend knows it's a shipping update
	refresh.value = 1;
	// Submit the form and disable the pay button
	safeSubmit(frm);
};


/**********************************************************
	The javascript for the Color Selector
	2005.12.18 pk
**********************************************************/										 
function showHex(hex) {
	var color = hex;
	var hexColor = YAHOO.util.Dom.get("hexColor");

	// Show the hex title
	toggleDisplay("hexTitle", "show");
	hexColor.innerHTML = color;

	return false;
};

/**********************************************************
 	Clears the email_recovery hidden form field
	if the user uses the back button to return to the page
	with login fields 
	pk 2005.01.03
**********************************************************/
function clearEmailRecovery() {
	//The default value in the PML must be false
	var emailRecovery = document.forms[0].email_recovery;
	if (emailRecovery) { 
		emailRecovery.value = false;
	}
}

/**********************************************************
	Resizes and focuses the cart page
	if it's in a new window
	2005.01.15 pk
**********************************************************/
function initCart() {
	var parentWin = window.opener;
	// Resize the cart window if it's opened in a new window
	if (parentWin) {
		self.name = "paypal";
		
		resizeShoppingCartWindow();
		
		//focus the window when items are added
		this.focus();
	}
};

function resizeShoppingCartWindow() {
	var iWidth, iHeight;
	var resizeWidth = resizeHeight = 0;

	if (window.innerWidth) {
		iWidth = window.innerWidth;
		iHeight = window.innerHeight;
	}
	else if (document.documentElement && document.documentElement.offsetWidth){
		iWidth = document.documentElement.offsetWidth;
		iHeight = document.documentElement.offsetHeight;
	}

	if (iWidth<780) {
		resizeWidth=780-iWidth;
	}
	if (iHeight<555) {
		resizeHeight=555-iHeight;
	}

	if (iWidth>0 || iHeight>0) {
		self.resizeBy(resizeWidth,resizeHeight);
	}
}

/**********************************************************
	Controls the display of the continue shopping button
	When the page is loading
	2005.01.15 pk
**********************************************************/
function setContinueShopping(shoppingURL) {
	var parentWin = window.opener;
	var browserType = getBrowserType();
	var elContinueShopping = YAHOO.util.Dom.get("continue_shopping");
	
	// Show the AOL message or the Continue button
	if (parentWin && browserType == "aol") { 
		YAHOO.util.Dom.get("aolmessage").className = "";
	}
	else if (parentWin || shoppingURL) { // Show button
		if (elContinueShopping.className == "hidden secondary") {
			elContinueShopping.className = "secondary";
		} 
		else if (elContinueShopping.className == "hidden") {
			elContinueShopping.className = "";
		}
	}
};
/********* Spec 26610 additions **************
Shipping calculator additions to the Cart page.
- JS to flip between Intl. and US shipping
- JS to clear ZIP field when clicked and gray it on page load
- JS to open and close the entire widget
**********                      *************/
function flipShipTo(e) {
	YAHOO.util.Event.preventDefault(e);
	var intl = document.getElementById('shippingIntlRow');
	var us = document.getElementById('shippingZipRow');
	var intl_label = document.getElementById('countryLabel');
	var us_label = document.getElementById('zipLabel');
	if ( YAHOO.util.Dom.hasClass(us,'accessAid') ) {
		YAHOO.util.Dom.addClass(intl,'accessAid');
		YAHOO.util.Dom.addClass(intl_label,'accessAid');
		YAHOO.util.Dom.removeClass(us,'accessAid');
		YAHOO.util.Dom.removeClass(us_label,'accessAid');
	}
	else if ( YAHOO.util.Dom.hasClass(intl,'accessAid') ) {
		YAHOO.util.Dom.addClass(us,'accessAid');
		YAHOO.util.Dom.addClass(us_label,'accessAid');
		YAHOO.util.Dom.removeClass(intl,'accessAid');
		YAHOO.util.Dom.removeClass(intl_label,'accessAid');
	}
}
function clearZipField(e) {
	this.value='';
}
function toggleWidget(e) {
	YAHOO.util.Event.preventDefault(e);
	YAHOO.util.Dom.removeClass('cancelWidget','accessAid');
	YAHOO.util.Dom.removeClass('cancelWidget2','accessAid');
	var widget = document.getElementById('shippingWidget');
	var shipping = document.getElementById('shippingAndHandling');
	if ( YAHOO.util.Dom.hasClass(widget,'accessAid') ) {
		YAHOO.util.Dom.removeClass(widget,'accessAid');
		YAHOO.util.Dom.addClass(shipping,'accessAid');
	}
	else {
		YAHOO.util.Dom.addClass(widget,'accessAid');
		YAHOO.util.Dom.removeClass(shipping,'accessAid');
	}
}
YAHOO.util.Event.addListener(window,'load',function() {
	YAHOO.util.Event.addListener('flipToIntlShipping','click',flipShipTo);
	YAHOO.util.Event.addListener('flipToUSShipping','click',flipShipTo);
	if ( document.getElementById('shipping_zip') ) {
		document.getElementById('shipping_zip').style.color='#666';
	}
	YAHOO.util.Event.addListener('shipping_zip','focus',clearZipField);
	YAHOO.util.Event.addListener('toggleWidget','click',toggleWidget);
	YAHOO.util.Event.addListener('toggleWidget2','click',toggleWidget);
	YAHOO.util.Event.addListener('cancelWidget','click',toggleWidget);
	YAHOO.util.Event.addListener('cancelWidget2','click',toggleWidget);
});
document.write("<style type='text/css'>");
document.write("a.change-link {display:inline;}");
document.write("input.transparentButton {display:none;}");
document.write("</style>");
/********* End of 26610 additions **********/
