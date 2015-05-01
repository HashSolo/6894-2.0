/**
* The PUI global namespace object
*/ 
if (typeof PAYPAL == "undefined") {
	var PAYPAL = {};
}



/**
* Shortcuts
*/
if (typeof YAHOO.util == "object") {
	YUD = YAHOO.util.Dom;
	YUE = YAHOO.util.Event;
} 



/**
* Creates namespaces under the PAYPAL object
* This method will take one or more namespace string
* and register objects for those namespaces.
*
* For example:
*
*	PAYPAL.namespace("PAYPAL.widget.myWidget", "cool.namespace");
*
* Will create 2 namespaces:
*	1) PAYPAL.widget.myWidget
*	2) PAYPAL.cool.namespace
*
* @return {Object} The last namespace created
*/
PAYPAL.namespace = function () {
	var names, obj, i, n;

	for (i = 0; i < arguments.length; ++i) {
		names = arguments[i].split(".");
		obj = PAYPAL;

		// PAYPAL is implied, so it is ignored if it is included
		for (n = (names[0] == "PAYPAL") ? 1 : 0; n < names.length; ++n) {
			obj[names[n]] = obj[names[n]] || {};
			obj = obj[names[n]];
		}
	}

	return obj;
};


PAYPAL.namespace("core", "util", "util.Event", "widget", "l10n");

/**
* Fires an event with the DOM of the page is ready - DEPRECATED. Use YAHOO.util.Event.onDOMReady instead.
*/
PAYPAL.util.Event.onDomReady = function (func, scope) {
	YUE.onDOMReady(func, scope);
};


/**
* Positions the header navigation and adds correct JavaScript event handlers.
* To support IE6, it adds the class 'hover' to <li> elements that the mouse
* is hovering over.
* @requires PAYPAL, YAHOO.util.Event, YAHOO.util.Dom
*/
PAYPAL.core.Navigation = {

	/**
	* The navigation div container
	*/
	nav: null,

	/**
	* The iframe shim
	*/
	iframe: null,

	/**
	* Initialize navigation
	*/
	init: function(){
		var navPrimary = YUD.get("navPrimary");
		this.nav = YUD.get("navFull");

		if(!this.nav || (navPrimary && navPrimary.getElementsByTagName("ul").length > 1)){
			return;
		}

		this.hoverHack();
		this.positionNav();

		YUE.onDOMReady(this.createIFrameShim, this, true);
	},

	/**
	* Position the nav at the top of the page.
	*/
	positionNav: function(){
		var navPrimary = YUD.get("navPrimary");
		var navFull = this.nav;

		// Primary empty?
		if(!navPrimary || navPrimary.getElementsByTagName("ul").length == 0){
			navFull.innerHTML = "";
			return false;
		}

		if(navPrimary && navFull){
			var ul = navFull.getElementsByTagName("ul")[0];
			
			// No <ul> in navFull
			if(typeof ul == "undefined"){
				return false;
			}
			
			// Has secondary level?
			var active = YUD.getElementsByClassName("active", "li", ul);
			if(active.length > 0){	
				active = active[0];
				
				// Is there a <ul> under the active tab
				var sec = active.getElementsByTagName("ul");
				if(sec.length > 0){				
					YUD.addClass(ul, "secondary");
				}
			}

			// Add to top
			navPrimary.innerHTML = "";
			navPrimary.appendChild(ul);
			navFull.className = "hide";

			this.nav = navPrimary;
		}
	},

	/**
	* Implement the hover hack to support IE 6 and other non standard browser.
	* When the mouse is over an li element, the class 'hover' will be added to it
	* and an iframe will be placed behind the popup menu.
	* On mouse out, the class will be removed.
	*/
	hoverHack: function(){
		var li = this.nav.getElementsByTagName("li");

		function mouseOver() {
			YUD.addClass(this, "hover");
	
			// Add iframe shim behind UL
			var ul = this.getElementsByTagName("ul");
			if (ul.length == 1) {
				PAYPAL.core.Navigation.addIFrameShim(ul[0]);
			}
		}

		function mouseOut(event) {
			YUD.removeClass(this, "hover");
			PAYPAL.core.Navigation.removeIFrameShim(event);
		}

		for(var i = 0; i < li.length; i++) {
			li[i].onmouseover = mouseOver;
			li[i].onmouseout = mouseOut;
		}
	},

	/**
	* Create the iframe shim.
	* Do this at init so it's ready when the menu appears
	*/
	createIFrameShim: function(){
		var ua = navigator.userAgent.toLowerCase();
		if(ua.search(/msie 6/) > -1) {
			this.iframe = document.createElement("iframe");
			this.iframe.src = "javascript:false;";
			this.iframe.style.position = "absolute";
			this.iframe.style.border = "";
			this.iframe.style.margin = "0";
			this.iframe.style.padding = "0";
			this.iframe.style.zIndex = "1";
			this.iframe.style.visibility = "hidden";
			YUD.setStyle(this.iframe, "opacity", "0");

			document.body.appendChild(this.iframe);
		}
	},

	/**
	* Add the iframe shim to fix form elements and flash interference.
	* @param {DomNode} elem The UL element to put the shim behind.
	*/
	addIFrameShim: function(elem){
		if(!this.iframe){
			return;
		}

		var xy = YUD.getXY(elem);
		this.iframe.style.top = xy[1] +"px";
		this.iframe.style.left = (xy[0] - 2) +"px";
		this.iframe.style.height = elem.clientHeight +"px";
		this.iframe.style.width = elem.clientWidth +"px";
		this.iframe.style.visibility = "visible";
		this.iframe.style.zIndex = "1";
	},

	/**
	* Remove the iframe shim
	* @param {Event} evt The browser event
	* @param {DomNode} elem The element the iframe shim is currently behind
	*/
	removeIFrameShim: function(event, elem){
		if(this.iframe){
			this.iframe.style.visibility = "hidden";
		}
	}

};

/**
* Opens a new window based on an anchor tag.
* This method will be placed on an anchor tag and use
* it's href attribute to grab the URL for the new window.
*
* <a href="xyz/foo.html" onclick="PAYPAL.core.openWindow((event, { size: 's'})">Link</a>
*
* This will open a window for "xyz/foo.html" that is 400 pixels wide and 300 pixels tall
* using one of the three allowed popup sizes.
*
* The second optional parameter is an object that contains settings to overwrite the default
* behaviors imposed by the script. There are three default sizes, selectable through the
* 'size' setting using one of three values: 's', 'm', 'l' 
* small	  -> 's' -> 400px wide 300px tall
* medium  -> 'm' -> 440px wide 450px tall
* large	  -> 'l' -> 560px wide 450px tall
*
* <a href="xyz/url/foo.html" onclick="PAYPAL.core.openWindow((event, { size: 'm', name: 'Foo', resizable: 0})">Link</a>
*
* If a name is not provided, the script will issue a unique name to that anchor, so each
* click will open in the same window.
*
* Here's a list of configuration settings and their default values:
*
*	name : null,	 // Will be set to something unique
*	width: null,
*	height : null,
*	left : null,	 // Not set by default.
*	top : null,		 // Not set by default.
*	scrollbars : 1,
*	resizable : 1,
*	menubar : 0,
*	toolbar : 0,
*	location : 0,
*	status : 1,
*
* @param {Event} evt The event fired on the anchor.
* @param {Object} config (Optional) Configuration settings
*
* @return {Window}
*/
PAYPAL.core.openWindow = function (evt, config) {

	config = config || {};

	// Get anchor & stop event
	var event = evt || event;
	var anchor = event.target || event.srcElement;

	while (anchor && anchor.nodeName != "A"){
		anchor = anchor.parentNode;
	}

	if (!anchor){
		return false;
	} else {
		if (event.preventDefault) {
			event.preventDefault();
		} else {
			event.returnValue = false;
		}
	}

	// Set width and height by size s,m,l
	if (config.size != 'undefined') {
		var sizes = [["s",400,300],["m",440,450],["l",560,450]];

		for (var j = 0; j < sizes.length; j++ ) {
			if (config.size == sizes[j][0]) {
				config.width = sizes[j][1];
				config.height = sizes[j][2];
				break;
			}
		}
	}	 
	
	// Default configuration options
	var defConfig = {
		name : null,
		width : null,
		height : null,
		left : null,
		top : null,
		scrollbars : 1,
		resizable : 1,
		menubar : 0,
		toolbar : 0,
		location : 0,
		status : 1
	};

	// Set unique name
	var name = config.name || defConfig.name || anchor.ppWinName;
	if (!name) {
		name = "autoWin"+ Math.round(Math.random() * 10000);
		anchor.ppWinName = name;
	}

	// Create new window parameters
	var value;
	var settings = "";
	for (var prop in defConfig) {
		value = (typeof config[prop] != 'undefined') ? config[prop] : defConfig[prop];

		// Convert value
		if (prop.search("height|width|left|top") == -1) {
			value = (value == 1 || value === true) ? "yes" : "no";
		}

		if (value && prop != "name") {
			settings += prop +"="+ value +",";
		}
	}

	// Open Window
	var win = window.open(anchor.href, name, settings);

	return win;
};

/** 
 * Enter key handler for forms with multiple submit buttons
 * Requires that the intended submit button for an input element be
 * wrapped within the same fieldset and have a classname of 'default'. 
 * Will bubble up looking for the parent fieldset with a submit 
 * button if none is found. Will not search past the parent form element.
 * 
 * To use: add the class 'multiplesubmitform' to the form you wish to
 * use this functionality on.
 * 
 * @author tilynn
 * @requires PAYPAL, YAHOO, YUD, YAHOO.util.Event
 * @param {Event} e Event object
 */
PAYPAL.util.captureEnter = function (e) {
	/* get which key was pressed */
	var keyPressed = e.charCode || e.keyCode;
	/* in which element were we focused */
	var target = e.target || e.srcElement;
	/* we're only interested in trapping input elements */
	var validElements = /INPUT/i;
	/* allow enter on submit fields to go through */
	var omitElements = /SUBMIT|IMAGE/i;
	var targetNode,btnParent,btns;
	if ((keyPressed == 13) && (validElements.test(target.nodeName)) && (!omitElements.test(target.type))) {
		/* Look for the parent fieldset */
		while((target = target.parentNode)){
			targetNode = target.nodeName.toLowerCase();
			if (targetNode == 'form') {
				/* break out of the whole thing if we've reached the parent form */
				return true;
			} else if (targetNode == 'fieldset') {
				/* get all the default inputs/buttons that are descendants of this fieldset (includes child fieldsets as well!) */
				btns = YUD.getElementsByClassName('default','',target);
				if (btns.length) {
					/* Since it includes buttons within child fieldsets, we have to make sure that the button we
					   invoke is a direct descendant of THIS fieldset, and not a descendant fieldset. */
					for (var x = 0; x < btns.length; x++) {
						btnParent = btns[x];
						while ((btnParent = btnParent.parentNode)) {
							if (btnParent.nodeName.toLowerCase() == 'fieldset') {
								if (btnParent == target) {
									YUE.preventDefault(e);
									btns[x].click();
									return true;
								} else {
									break;
								}
							}
						}
					}
				}
			}
		}
	}
};

/** 
 * Setup function for getting forms with multiple submits. Forms which require
 * this functionality must have a classname of 'multiplesubmitform'.
 * 
 * @author tilynn
 * @requires YAHOO, YAHOO.util.Dom, YAHOO.util.Event
 */
PAYPAL.util.getMultipleSubmitForms = function () {
	var multiForms = YUD.getElementsByClassName('multiplesubmitform');
	if (multiForms.length) {
		YUE.addListener(multiForms, 'keypress', PAYPAL.util.captureEnter);
	}	
};

YUE.onDOMReady(PAYPAL.util.getMultipleSubmitForms);


/**
* Creates the standards card stack treatment for a collection of elements on the page.
*
* Your element must follow the standards card stack markup in order for this to work:
*
*	<div class="cardstack">
*		<h2>Optional Cardstack Title</h2>
*
*		<div class="card top">
*			<div class="header">
*				<h3>Title Un</h3>
*			</div>
*			<div class="body">
*				<p>Blah blah blah. Card 1.</p>
*				<ul>
*					<li>blah</li>
*					<li>blah</li>
*				</ul>
*			</div>
*		</div>
*
*		<div class="card">
*			<div class="header">
*				<h3>Title Deux</h3>
*			</div>
*			<div class="body">
*				<p>Blah blah blah.</p>
*				<p>Card 2.</p>
*			</div>
*		</div>
*	</div>
*
*	CLASS BREAKDOWN
*		+ "cardstack": This class should surround the entire stack
*		+ "card": Each 'card' in the stack needs to have this class
*		+ "top": Specifies which card should be selected by default (which one on top).
*		+ "header": The header and tab title.  Inside this <div> needs to be a single <h3> element.
*		+ "body": The content of the card.	This can contain any valid markup.
*
*/
PAYPAL.widget.CardStack = function (elem, config) {
	
	// Is elem available?
	if (typeof elem == "object" || (typeof elem == "string" && YUD.get(elem) != null)) {
		this._init(elem, config);
	}
	// Init onAvailable
	else if (typeof elem == "string") {
		YUE.onAvailable(elem, function(){
			this._init(elem, config);
		}, this, true);
	}
	
};

PAYPAL.widget.CardStack.prototype = {
	tabbox: null,
	cards: [],
	tabs: [],
	vertical: false, //Horizontal by default
	selected: null,
	directTab: null,
	_previous: null,
	
	
	/**
	* Main initialization method.  Sets event handlers and initial selection state.
	*/
	_init: function(elem, config) {
		var self = this;
		
		this.cards = [];
		this.tabs = [];
	
		// Is elem an ID or DOM node
		if (typeof elem == "string") {
			elem = YUD.get(elem);
			
			if(elem == null){
				return false;
			}
		}
		if (YUD.hasClass(elem, "stacked")) {
			return false;
		}
		
		
		// Get config options and merge them with object
		if (YUD.hasClass(elem, "vertical")) {
			this.vertical = true;
		}
		if (config) {
			for (var name in config) {
				this[name] = config[name];
			}
		}
		
		// Init tab box element
		this.tabbox = elem;
		this.cards = YUD.getElementsByClassName("card", "div", this.tabbox);
		
		if (this.tabbox == null || this.cards.length == 0) {
			return;
		}
		
		YUD.addClass(this.cards[0], "first");
		YUD.addClass(this.cards[this.cards.length - 1], "last");
		
		//Only add this class if vertical param is passed in
		if (this.vertical) {
			YUD.addClass(this.tabbox, "vertical");
			YUD.addClass(this.tabbox, "cardstackVertIE");
		}
		YUD.addClass(this.tabbox, "stacked");
		
		// Init each card
		var card, tab;
		var cardHeight=0;
		for (var i = 0; i < this.cards.length; i++) {
			card = this.cards[i];
			if (!card.id) {
				YUD.generateId(card, "autotab_");
			}
			if (('#'+card.id) == location.hash) {
				this.directTab = true;
			}
		}
		for (i = 0; i < this.cards.length; i++) {
			card = this.cards[i];
			card._hn = null;
			card._children = [];
			
			if (card.clientHeight > cardHeight) {
				cardHeight = card.clientHeight+15;
			}
			
			// Make selected, if it already has selected class
			if (('#'+card.id) == location.hash) {
				YUD.addClass([card, card._tab], "top");
			} else {
				if (this.directTab) {
					YUD.removeClass([card, card._tab], "top");
				}
			}

			// Make selected, if it already has selected class
			if (YUD.hasClass(card, "top")){
				if(this.selected == null){
					this.selected = card;
				} else {
					YUD.removeClass([card, card._tab], "top");
				}
			} else {
				YUD.addClass(card, "accessAid");
			}
			
			// Find and hide header
			var header = YUD.getElementsByClassName("header", "div", card);
			if (header.length > 0) {
				var header = header[0];
				
				// Get <hn> tag
				var hn = header.getElementsByTagName("*");				
				for (var j = 0; j < hn.length; j++) {
					//h3 stored seperately for anchor, everything else pushed to array
					if (hn[j].nodeName == 'H3') {
						card._hn = hn[j];
					} else {
						card._children.push(hn[j]);
					}
				}				
			}
			
			// Find body
			var body = YUD.getElementsByClassName("body", "div", card);
			if (body.length > 0) {
				body = body[0];
				card._body = body;
			}
			
		}
		
		for (var i = 0; i < this.cards.length; i++) {
			card = this.cards[i];
			YUD.setStyle(card, 'height', cardHeight+'px');
		}
		
		// Create the <ol> for tabs
		if (this.cards.length > 0) {
		
			// Create list
			var ol = document.createElement("ol");
			ol.className = "tabs";
			for (i = 0; i < this.cards.length; i++) {
				var card = this.cards[i];
				
				// No <Hn> tag
				if (!card._hn) {
					continue;
				}
			
				var li = document.createElement("li");
				var anchor = document.createElement("a");
				var id = card.id;
				
				// Create unique ID
				if (!id) {
					id = YUD.generateId(card, "autotab_");
				}

				anchor.href = "#"+ id;
				anchor.id='a'+id;				
				anchor.innerHTML = card._hn.innerHTML;

				// Copy the tracking code for the tab
				var trackCode = card._hn.className.match(/(scTrack.*:[^\s]+)/);

				if (trackCode) { 
					YUD.addClass(anchor, trackCode[1]); 
				}
				
				YUE.addListener(anchor, 'click', this.handleclick, this, true);
				YUE.addListener(anchor, 'focus', this.handleclick, this, true);			
				li.appendChild(anchor);
				
				//Appending all the other elements in header if vertical tabs
				for (var k = 0; k < card._children.length; k++) {
					anchor.appendChild(card._children[k]);
				}				
				
				ol.appendChild(li);
				this.tabs[i] = li;
				this.tabs[i]._card = card;
				
				card._tab = li;
				li._card = card;
				anchor._tab = li;
				anchor._card = card;
			}
			YUD.addClass(ol.firstChild, "first");
			YUD.addClass(ol.lastChild, "last");
			
			this.cards[0].parentNode.insertBefore(ol, this.cards[0]);
		}
		
		this.showcard();		
	},
	
	/**
	* Handles a tab being clicked
	*/
	handleclick: function(evt) {
		evt = window.event || evt;
		YUE.preventDefault(evt);
		var anchor = YUE.getTarget(evt);
		
		// Remove the "focus" dotted border around the anchor
		if (evt.type == "click") {
			anchor.blur();
		}
		
		this.selected = anchor._card;
		this.showcard();
	},
	
	/**
	* Show a card in the stack, and hide the last one.
	*/
	showcard: function() {
	
		// No new selection
		if (!this.selected || this.selected == this._previous) {
			return true;
		}
		
		// Hide previous
		if (this._previous && this._previous != this.selected) {
			YUD.addClass(this._previous, "accessAid");
			YUD.removeClass([this._previous, this._previous._tab], "top");
			
			YUD.removeClass(this.tabs, "previous");
			YUD.removeClass(this.tabs, "next");
		}
		
		// Show new
		this._previous = this.selected;
		YUD.addClass([this.selected, this.selected._tab], "top");
		YUD.removeClass(this.selected, "accessAid");
		
		//If using vertical tabs set the "previous/next" class for tabs around selected
		if (this.vertical) {
			var tab = this.selected._tab;
			if (tab.previousSibling) {
				YUD.addClass(tab.previousSibling, "previous");
			}
			if (tab.nextSibling) {
				YUD.addClass(tab.nextSibling, "next");
			}
		}		
	}
};

// Initialize the cardstacks on the page
YUE.onDOMReady(function () {
	var cardstacks = YUD.getElementsByClassName("cardstack", "div", document);
		for (var i = 0; i < cardstacks.length; i++) {
			new PAYPAL.widget.CardStack(cardstacks[i]);
		}
});


/**
* Flash tools
*/
PAYPAL.util.Flash = {
	/**
	* Insert a flash video if supported or an image if not.
	* @param {String} flash The flash movie URL
	* @param {int} width The width of the flash movie
	* @param {int} height The height of the flash movie
	* @param {String|DomNode} target The element (or ID of the element) to place the flash movie in.
	* @param {Boolean} replace Set to TRUE to replace the target content.  If FALSE, it will append to target.
	* @param {int} minVer The minimum flash version supported for this movie
	* @param {String} id The ID to place on the object tag
	* @param {Boolean} useNonStandard Use a non-standard HTML rendering. This is for including the <embed> tag so that Mozilla will correctly reference the flash object. Requires Platform team approval.
	*	@param {String} flashVars Variables which are passed to the SWF file.
	* @param {String} wmode Sets window mode for Flash object.
	* @return {Object} A reference to the flash object that was output to the page
	*/
	insertFlash : function(flash, width, height, target, replace, minVer, id, useNonStandard, flashVars, wMode){
		
		// Get target
		if(typeof target == "string"){
			target = YUD.get(target);
		}
		if(!target){
			return false;
		}
		
		// Doesn't match version requirement
		var ver = this.getVersion();
		if(ver == 0 || ver < parseInt(minVer, 10)){
			return false;
		}		
		
		if (typeof id != 'string') {
			id = YUD.generateId('','flashmovie');
		}

		// Default use of non-standard HTML to false
		if (typeof useNonStandard != 'boolean') {
			useNonStandard = false;
		}
		
		// Defaults the wmode to transparent
		wMode = wMode || "transparent";

		// Create object
		var objectHtml = "";
		if(navigator.userAgent.match(/msie/i) != null || useNonStandard){
			objectHtml += '<object width="'+ width +'" height="'+ height +'" classid="clsid:D27CDB6E-AE6D-11cf-96B8-444553540000" id="' + id + '">';
		}
		else {
			objectHtml += '<object width="'+ width +'" height="'+ height +'" data="'+ flash +'" type="application/x-shockwave-flash" id="' + id + '">';
		}
		
		// Parameters
		objectHtml += '<param name="movie" value="'+flash+'"></param>'+'<param name="wmode" value="'+wMode+'"></param>'+'<param name="quality" value="high"></param>'+'<param name="menu" value="false"></param>'+'<param name="allowScriptAccess" value="always"></param>';
						
		if(flashVars){
			objectHtml += '<param name="FlashVars" value="'+ flashVars +'"></param>';
		}
	
		// Embed tag
		if (useNonStandard) {
			objectHtml += '<embed src="'+flash+'" quality="high" width="'+width+'" height="'+height+'" name="'+id+'" allowScriptAccess="always" type="application/x-shockwave-flash" pluginspage="http://www.adobe.com/go/getflashplayer"';
			if(flashVars){
				objectHtml += ' flashvars="'+ flashVars +'"';
			}
			objectHtml += '>';
		}
		
		objectHtml += '</object>';
		
		if(replace){
			target.innerHTML = objectHtml;
		}
		else{
			target.innerHTML += objectHtml;
		}
		
		return this.getFlashMovieObject(id);
	},
	/**
	 * Returns an object reference to the Flash movie. This is necessary
	 * for any page that wants to communicate with the movie via script.
	 * @return {Object} Object reference to flash movie in DOM
	 */
	getFlashMovieObject: function (movieName) {
		try {
			if (document.embeds && document.embeds[movieName]) {
				return document.embeds[movieName];
			} else if (window.document[movieName]) {
				return window.document[movieName];
			}
		}
		catch (e) {
			return YUD.get(movieName);
		}		 
	},
	
	/**
	* Gets the current major version of flash installed.
	* The minimum version this will find is 3.x.
	* @return {Number} Version number or zero if it cannot be detected.
	*/
	getVersion: function () {
		var flash;
		var i = 3;
		var ver = 0;
		
		// From navigator object
		if(navigator.plugins && navigator.mimeTypes.length){
			flash = navigator.plugins["Shockwave Flash"];
			if(flash){
				ver = parseInt(flash.description.replace(/[^0-9.]/g, ""), 10);
			}
		}
		
		// Internet Explorer
		else{
			flash = true;
			while(flash){
				try{
					flash = new ActiveXObject("ShockwaveFlash.ShockwaveFlash."+ i);
					ver = i;
					i++;
				}catch(e){
					break;
				}
			}
		}
		
		return ver;
	}
	
};

/**
 * History manager to allow back and forward navigation for Javascript interactions. 
 *
 * Usage: 
 *		PAYPAL.util.HistoryManager.init();
 *		PAYPAL.util.HistoryManager.add('name', function() {}, this);
 *
 * @author Jeff Harrell
 */

PAYPAL.util.HistoryManager = {

	historyStack: [],
	current: null,
	interval: null,
	iframeObj: null,
	iframeName: '_historyManager',
	
	
	/**
	 * Initializes the history manager and starts watching for changes
	 */
	init: function () {
		// IE won't populate/recognize history from fragments so use an iframe
		if (window.attachEvent && !window.opera) {
			if (!(this.iframeObj = YUD.get(this.iframeName))) {
				this.iframeObj = document.createElement('iframe');
				this.iframeObj.setAttribute('id', this.iframeName);
				this.iframeObj.style.display = 'none';
				this.iframeObj.setAttribute('src', "javascript:void(0)");
			
				document.body.appendChild(this.iframeObj);

				// since we just created this iframe have it create it's inital history entry
				this.updateFragment('#');
				
				// force the callback if this is a bookmarked state
				if (window.location.hash) {
					this.updateState(window.location.hash.substr(1));
				}
			}
		}
		
		this.watch();
	},
	

	/**
	 * Polling function which runs the callback if the fragment has changed
	 *
	 * @private
	 */
	watch: function () {
		var fragment;
		var self = this;
		
		this.interval = window.setInterval(function() {
			if (self.iframeObj) {
				fragment = self.iframeObj.contentWindow.document.location.hash.substr(1);
			} else {
				fragment = window.location.hash.substr(1);
			}
			
			if (fragment != self.current) {
				self.updateState(fragment);
			} 
		}, 100);
	},



	/**
	 * A wrapper for code to be added as a new history state. If the namespace passed is
	 * "test" it will be the first state and the following will be "test-2", "test-3", etc.
	 *
	 * @param namespace {String}	The name of the history to add to 
	 * @param fn {Funtion}		A callback function for this state
	 * @param obj {Object}		An object which will be passed along as 'this' in the callback
	 */
	add: function (namespace, fn, obj) {
		// history added with no namespace is run when there is no fragment (usually the default state)
		namespace = namespace || this.iframeName;

		if (!this.historyStack[namespace]) { this.historyStack[namespace] = []; }
		
		var loc = this.historyStack[namespace].push({ 'obj' : obj, 'fn' : fn });
		var fragment = (loc > 1) ? namespace + '-' + loc : namespace;

		return fragment;
	},
	

	/**
	 * Similar to add(), but will also run the code as it parses it
	 *
	 * @param namespace {String}	The name of the history to add to 
	 * @param fn {Funtion}		A callback function for this state
	 * @param obj {Object}		An object which will be passed along as 'this' in the callback
	 */ 
	addAndRun: function (namespace, fn, obj) {
		var fragment = this.add(namespace, fn, obj);
		this.updateFragment(fragment);
	},


	/**
	 * Updates the current fragment in the browser
	 *
	 * @param fragment {String} The value of the URL fragment
	 * @private
	 */
	updateFragment: function (fragment) {
		if (!fragment) { return; }
		
		if (this.iframeObj) {
			this.iframeObj.contentWindow.document.open();
			this.iframeObj.contentWindow.document.close();
			this.iframeObj.contentWindow.document.location.hash = fragment;
			
			// IE causes an empty history state if the fragment is updated with the 
			// id of an object on the page; don't update the frament in this case.
			if (fragment != '#' && !YUD.get(fragment)) {
				window.location.hash = fragment;
			}
		} else {
			window.location.hash = fragment;
		}
	},
	

	/**
	 * Parses the fragment and runs the callback
	 *
	 * @param fragment {String} The value of the URL fragment
	 * @private
	 */
	updateState: function (fragment) {
		this.current = fragment;
		
		// pieces[0] is the namespace, pieces[1] is the index
		var pieces = fragment.split('-');
		// normalize the index number
		pieces[1] = (isNaN(pieces[1])) ? 0 : pieces[1] - 1;
		
		this.callback(pieces[0], pieces[1]);
		
	},


	/**
	 * Runs the function associated with the provided history location.
	 * If no namespace is passed then all default actions will be run.
	 *
	 * @param namespace {String}	The name of the history state
	 * @param loc {Integer}		The index in the parent namespace
	 * @private
	 */
	callback: function (namespace, loc) {			
		var states = [];
		
		// if the namespace is blank then call all of the blank history states
		if (namespace == '') {
			states = this.historyStack[this.iframeName];
		// otherwise call all of the history states for this namespace
		} else if (this.historyStack[namespace] && this.historyStack[namespace][loc]) {
			states.push(this.historyStack[namespace][loc]);
		}
		
		for (var x in states) {
			if (typeof states[x].fn == 'function') {
				states[x].fn.call(states[x].obj);
			}
		}
	}

};



/*
* Hides and shows an element on the page.
*
*  Basic Usage:
*  
*  The basic use for this is to show and hide an element 
*  when another is clicked or changed. To make the following
*  code display "myParagraph" when "myTrigger" is clicked 
*  you could use the following code:
*  
*	<a id="myTrigger" href="#myParagraph">Show my paragraph</a>
*	<p id="myParagraph">This is my paragraph.</p> 
*  
*	...
*	
*	var myHideShow = PAYPAL.widget.HideShow('myParagraph', 'myTrigger');
*
*  
*  Advanced Usage:
*
*  - You can hide or show an object conditionally based on 
*  an input or select element's value using the the optional 
*  value parameter. 
*  
*  
*  - You can set additional triggers which will toggle 
*  the element between hide and show. For example:
*  
*	var myHideShow = PAYPAL.widget.HideShow('myParagraph', 'myTrigger');
*	myHideShow.setToggle('myAlternateToggle');
*  
*  
*  - You can declare objects that are mutually exclusive with
*  your hide / show instance and will automatically show when 
*  it hides and hide when it shows. For example:
*	
*	<p id="myAntiParagraph">There is no paragraph yet.</p> 
*  
*	<a id="myTrigger" href="#myParagraph">Show paragraph</a>
*	<p id="myParagraph">Here's the paragraph.</p> 
*	
*	...
*	
*	var myHideShow = PAYPAL.widget.HideShow('myParagraph', 'myTrigger');
*	myHideShow.toggleOpposite('myAntiParagraph');
*  
*  
*  Tips:
*  
*  - When a hide / show element is opened both it and it's
*  trigger elements will have an "opened" class on them
*  which can be used to visually style them.
* 
*  - Hide/show elements will be automatically hidden when 
*  they are created unless they have a class of opened.
*  
*  
* @param {Object} el The element which we're hiding/showing
* @param {Object} trigger The trigger for the hide/show action 
* @param {Object} value Hide/show only if the trigger is equal to this value (optional)
* 
* 
*/

PAYPAL.widget.HideShow = function (el, trigger, value) {
	this.el = el;
	this.trigger = trigger;
	this.value = value;
	
	this.onHide = new YAHOO.util.CustomEvent('hide');
	this.onShow = new YAHOO.util.CustomEvent('show');
	
	this.init();
};


PAYPAL.widget.HideShow.prototype = {
	
	init: function () {
		// setup DOM elements
		this.el = (typeof this.el === 'string') ? YUD.get(this.el) : this.el;
		this.trigger = (typeof this.trigger === 'string') ? YUD.get(this.trigger) : this.trigger;

		// check if it should already opened, otherwise hide it
		if ((this.isOpen = YUD.hasClass(this.el, 'opened'))) {
			this.show();
		} else {
			this.hide();
		}
	
		// setup trigger events
		this.setTrigger(this.trigger);
	},
	
	toggle: function () {
		if (this.isOpen) {
			this.hide();
		} else {
			this.show();
		}
	},
	
	hide: function () {
		YUD.replaceClass(this.el, 'opened', 'accessAid');
		YUD.removeClass(this.trigger, 'opened');
		
		this.isOpen = false;

		this.onHide.fire();
	}, 
	
	show: function () {
		YUD.replaceClass(this.el, 'accessAid', 'opened');
		YUD.addClass(this.trigger, 'opened');
		
		this.isOpen = true;
		
		this.onShow.fire();
	},

	setTrigger: function (trigger) {
		if (!trigger) { return; }

		trigger = (typeof trigger === 'string') ? YUD.get(trigger) : trigger;

		if (trigger) {
				var self = this;
				var tag = trigger.tagName && trigger.tagName.toLowerCase();

				// anchors
				if (tag == 'a') {
						YUE.addListener(trigger, 'click', function (e) {
								YUE.preventDefault(e);
								self.toggle();
						});
				// select
				} else if (tag == 'select') {
						YUE.addListener(trigger, 'change', function () {
								if (self.value && self.value === this.value || !self.value && this.checked) {
										self.show();
								} else {
										self.hide();
								}
						});
				// inputs
				} else {
						YUE.addListener(trigger, 'click', function () {
								if (self.value && self.value === this.value || !self.value && this.checked) {
										self.show();
								} else {
										self.hide();
								}
						});
				}
		}
	},
	
	toggleOpposite: function (el) {
		var altHideShow;

		el = (typeof el === 'string') ? YUD.get(el) : el;

		// keep the other element open if needed
		if (!this.isOpened) {
			YUD.addClass(el, 'opened');
		}
	
		if (el instanceof PAYPAL.widget.HideShow) {
			altHideShow = el;
		} else {
			altHideShow = new PAYPAL.widget.HideShow(el);
		} 

		// subscribe to the other
		this.onHide.subscribe(function () { altHideShow.show(); });
		this.onShow.subscribe(function () { altHideShow.hide(); });
	}
};


YUE.onDOMReady(function () {
	// legacy: find all hideshow elements and their anchors
	var elements = YUD.getElementsByClassName('hideShow');

	for (var i = 0; i < elements.length; i++) { 
		var triggers = YUD.getElementsBy(function(current) {
			var href = current.getAttribute('href');
			return (href.substr(href.indexOf('#') + 1) == elements[i].id);
		}, 'a');
		
		var hideshow = new PAYPAL.widget.HideShow(elements[i]);
		
		for (var j = 0; j < triggers.length; j++) {
			hideshow.setTrigger(triggers[j]);
		}
	}
});


/**
* Prevents a form from being submitted more than once.
* When the form has been submitted the "submitted" class will be added
* to the form element, which allows for styling of form elements.
*
* For example you could add CSS to make the buttons look disabled after
* the form has been submitted.
*
* USAGE:
*	
*	To use SafeSubmit with your form simply add the "safeSubmit" class to the form element:
*
*		<form action="foo.cgi" class="safeSubmit">
*
*/
PAYPAL.widget.SafeSubmit = {
	
	/**
	* Setup all forms on the page
	*/
	init: function () {
		var forms = YUD.getElementsByClassName("safeSubmit", "form", document);
		this.setup(forms);
	},
	
	/**
	* Add the SafeSubmit functionality to a form or collection of forms
	* @param {DomNode|String|Array} form A Form element, form ID or an array of form elements
	*/
	setup: function (form) {
		YUE.addListener(form, "submit", this.handleSubmit);
	},
	
	/**
	* Handle the form onsubmit and only allow it to be submitted once
	* @param {Event} evt The submit event object
	*/
	handleSubmit: function (evt) {
		var form = this;
		
		// First time submitting
		if(typeof form._submitted == 'undefined'){
			form._submitted = true;
			YUD.addClass(form, "submitted");
		}
		// Already submitted
		else {
			YUE.preventDefault(evt);
		}
	}
	
};

YUE.onDOMReady(PAYPAL.widget.SafeSubmit.init, PAYPAL.widget.SafeSubmit, true);


/**
* Set the default focus to the first text field or password or textarea for the given form
* Usage: 
* PAYPAL.util.defaultFocus('FormId_or_FormObject','TextBox_or_TextArea_or_Password_IDorObject');
* @param {Object} - Form id or form object
* @param {Object} - Element id or element 
*/
PAYPAL.util.defaultFocus = function (form,elem) {
	
	var args = arguments.length;	
	
	if ( args == 0) { return; } 
	var f = (typeof form == 'string') ? YUD.get(form) : form;
	if (f == null) { return; }
	
	//check whether the form is visible and not accessAided
	if (YUD.getStyle(f,'display') == 'none' || YUD.hasClass(f, 'accessAid')) { return; }
	else {
		//check for page errors from backend
		if (YUD.get('messageBox')) {
			var divs = YUD.getElementsByClassName("messageBox error", "div", 'messageBox');
			if (divs.length > 0) { return; }
		}
		
		//check the html element that is passed
		if (args == 2) {
			var el = (typeof elem == 'string') ? YUD.get(elem) : elem;
			
			if (el == null) { return; }
			else if (el.disabled || el.readonly){ return; }
			else {
				el.focus();
				return;
			}		
		}
		else {	
			//loop the elements to see whether the first element is text field or password or textarea, leaving hidden controls
			var refElementCount = 0;
			for (var r = 0; r < f.elements.length; r++) {
				var elm = f.elements[r];
				var elType = elm.type;
				
				if (elType == 'hidden' || elm.nodeName == 'FIELDSET') { continue; }
				else if ((elType == 'text' || elType == 'password' || elType == 'textarea')) {
					if (!(elm.disabled || elm.readonly) && (elm.value == '')) {
						elm.focus();
						return;
					}
					else { 
						refElementCount++; 
						if (refElementCount == 2 ) { break; }
						continue; 
					}
				}
				else { return; }
			}	
		}
	}
};

/**
 * Method which enabled deferred loading of a script
 */ 
PAYPAL.util.lazyLoadRoot = '';

PAYPAL.util.lazyLoad = function (url, callback) {	
	var script = document.createElement('script');			
	script.type='text/javascript';
	
	if (typeof callback != 'undefined') {
		if (script.readyState){  //IE
			script.onreadystatechange = function () {
				if (script.readyState == 'loaded' || script.readyState == 'complete') {
					callback();
			        }
			};
		} else {  //Others
			script.onload = function () {			
				callback();
			};
		}
	}

	script.src = PAYPAL.util.lazyLoadRoot + url;
	document.getElementsByTagName('head')[0].appendChild(script);	
};




/** 
 * Add a class to the root element which allows us to determine if JavaScript is active 
 */
if (document && document.documentElement) {
	document.documentElement.className += ' jsEnabled';
}
