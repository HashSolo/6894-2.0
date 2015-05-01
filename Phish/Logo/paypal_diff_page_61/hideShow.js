/**
 * Creates the PAYPAL.widget namespace
 * @requires YAHOO.util.Anim, YAHOO.util.Dom, YAHOO.util.Event, PAYPAL
 */
PAYPAL.namespace("widget");

/**
 * Widget: HideShow
 * Author: Bryan Spears <bspears@paypal.com>
 * Requires: YAHOO.Dom, Yahoo.Event, and PAYPAL objects/namespaces
 * -----------------------------------------------------------------------------------
 * Three things are needed to make this function work
 *
 *  1 - The div(s) you would like to hide need to have the class name 'hideShow' 
 *      and a unique ID.
 *  2 - The links (anchor tags) to hide them need to have a href that is 
 *      '#<ID of div to hide>' e.g. If the div has an ID of 'sillyDiv' then the 
 *      href of the anchor to hide/show the div needs to be '#sillyDiv'
 *  3 - If you want something other than the original link(s) to hide the div, 
 *      lets say a cancel button or link, then give the canceling element(s) the
 *      class 'hideShow.' This works as long as the element(s) is/are inside the div.
 *
 * Optional:
 *
 *  If you'd like for the div to be open by default on page load, add the class
 *  "opened" to the divs class attribute. e.g. class="hideShow opened"
 *
 *  If you'd like for the controlling anchor to be hidden while the div is open,
 *  add the class "opened" and some CSS like the following:
 *
 *  #myForm a.opened { position: absolute; top: 0; left: -999em; }
 *
 *  Also, the CSS class 'accessAid,' or re-create or copy it, from the core.css 
 *  created by the standards team must be available to properly hide the containers.
 *
 *  The classname needs to be "opened" or else the JS can't hide/show it when the 
 *  container is opened/closed
 *
 * EXAMPLE:
 *	
 *	<a href="#test" class="opened">Control Link</a>
 *	<div id="test" class="hideShow opened">
 *		Blah blah blah, test box.
 *		<button type="submit">Button Submit</button>
 *		<a href="#" class="closer">Cancel Link</a>
 *	</div>
 *
 */

PAYPAL.widget.hideShow = {
	exclusiveMode: false, // PAYPAL.widget.hideShow.exclusiveMode = true;
	mouseOverMode: false, // PAYPAL.widget.hideShow.mouseOverMode = true;
	containers: '',
	links: new Array(),
	cancelButtons: new Array(),
	init: function() {
		this.containers = YAHOO.util.Dom.getElementsByClassName("hideShow");
		// For each div with a class of "hideShow" we have some work to do
		for ( var i = 0; i < this.containers.length; i++ ) {
			// Hide the container first
			if ( !(YAHOO.util.Dom.hasClass(this.containers[i], "opened")) ) {
				YAHOO.util.Dom.addClass(this.containers[i], "accessAid");
			}
			// Get all links where the href is '#<ID of the div>'
			var reg = new RegExp("#"+this.containers[i].id);
			checkHrefs = function(el) { return (el.getAttribute("href").match(reg)); }
			this.links = YAHOO.util.Dom.getElementsBy( checkHrefs, "a" );
			// We need to get all the objects to be used as "cancel buttons"
			this.cancelButtons = YAHOO.util.Dom.getElementsByClassName( "closer", "*", this.containers[i] );
			// Lets store our links and buttons in the container element for later use
			this.containers[i].hideShowLinks = this.links;
			this.containers[i].cancelButtons = this.cancelButtons;
			for ( var j = 0; j < this.links.length; j++ ) {
				// Store the container in the link element so it knows who's boss
				this.links[j].hideShowContainer = this.containers[i];
				// If the control is supposed to be hidden, but the block isn't visible, well, we need to fix that
				if ( !(YAHOO.util.Dom.hasClass(this.containers[i], "opened")) && YAHOO.util.Dom.hasClass(this.links[j], "opened") ) {
					this.links[j].wasOpened = true;
					YAHOO.util.Dom.removeClass(this.links[j], "opened");
				}
				// clicked to prevent a large number of events being attached to a link
				// Add the initial event handler to the link for showing the div
				if ( this.mouseOverMode ) {
					YAHOO.util.Event.addListener(this.links[j], 'mouseover', PAYPAL.widget.hideShow.toggleHideShow);	
				}
				else {
					YAHOO.util.Event.addListener(this.links[j], 'click', PAYPAL.widget.hideShow.toggleHideShow);
				}
			}
			if ( this.cancelButtons.length > 0 ) {
				for ( var k = 0; k < this.cancelButtons.length; k++ ) {
					// Store the container in the button element so it knows who's boss
					this.cancelButtons[k].hideShowContainer = this.containers[i];
					// For each cancel button, attach a hiding handler. Doesn't need to be modified later. 
					// It's either visible with the div or not
					YAHOO.util.Event.addListener(this.cancelButtons[k], 'click', PAYPAL.widget.hideShow.toggleHideShow);
				}
			}
		}
	},
	
	hide: function(obj) {
		YAHOO.util.Dom.addClass(obj, "accessAid");
		YAHOO.util.Dom.removeClass(obj, "opened");
		
		// Show the control anchor if it was supposed to be hidden
		if ( obj && obj.hideShowLinks ) {
			for ( var i = 0; i < obj.hideShowLinks.length; i++ ) {
				if ( YAHOO.util.Dom.hasClass(obj.hideShowLinks[i], "opened") ) {
					YAHOO.util.Dom.removeClass(obj.hideShowLinks[i], "opened");
					obj.hideShowLinks[i].wasOpened = true;
				}
			}
		}
	},
	
	show: function(obj) {
		if (this.exclusiveMode) {
			for ( var i = 0; i < this.containers.length; i++ ) {
				this.hide(this.containers[i]);
			}
		}
		YAHOO.util.Dom.removeClass(obj, "accessAid");
		YAHOO.util.Dom.addClass(obj, "opened")
		// Hide the control anchors if they were supposed to be hidden
		if ( obj && obj.hideShowLinks ) {
			for ( var i = 0; i < obj.hideShowLinks.length; i++ ) {
				if ( obj.hideShowLinks[i].wasOpened ) {
					YAHOO.util.Dom.addClass(obj.hideShowLinks[i], "opened");
					obj.hideShowLinks[i].wasOpened = false;
				}
			}
		}

		// Change them buttons back to submits
		var buttons = YAHOO.util.Dom.getElementsBy( function(elem) { return (elem.wasSubmit) }, "*", obj);
		for ( var i=0; i < buttons.length; i++ ) {
			buttons[i].disabled = false;
			buttons[i].wasSubmit = false;
		}
	},
	
	toggleHideShow: function(e) {
		YAHOO.util.Event.preventDefault(e);
		var anchor = this;
		var div = anchor.hideShowContainer;
		var ignoreClick = false;
		
		if ( PAYPAL.widget.hideShow.mouseOverMode ) {
			for ( var i = 0; i < div.cancelButtons.length; i++ ) {
				if ( div.cancelButtons[i] === anchor ) { ignoreClick = true; }
			}
		}
		else { ignoreClick = false; }
		
		if ( YAHOO.util.Dom.hasClass(div, "opened") 
				&& PAYPAL.widget.hideShow.mouseOverMode
				&& !ignoreClick ) {
			// Do nothing
		}
		else if ( YAHOO.util.Dom.hasClass(div, "opened") ) {
			PAYPAL.widget.hideShow.hide(div);
		}
		// To "open" the container
		else {
			PAYPAL.widget.hideShow.show(div);
		}
	}
}
YAHOO.util.Event.addListener(window, 'load', PAYPAL.widget.hideShow.init);

/**
 * Widget: Accordion
 * Author: Tony Broussard <tbroussard@paypal.com>
 * Requires: YAHOO.util.Anim, YAHOO.util.Dom, YAHOO.util.Event, PAYPAL
 * -----------------------------------------------------------------------------------
 * @fileoverview
 * This script will style all UL elements with a class name "accordion" as 
 * an expandable/collapsable accordion with fluid animation. It requires the
 * YUI Library and all accordions must be created using an unordered list.
 *
 * Example Usage:
 * 
 *	<ul class="accordion">
 * 		<li id="item1">
 *			<div class="toggler"><h3>First header</h3></div>
 *			<div class="content">Shown onmouseover; hides all other elements.</div>
 * 		</li>
 *		<li id="item2">
 *			<div class="toggler"><h3>Second header</h3></div>
 *			<div class="content">Shown onmouseover; hides all other elements.</div>
 *		</li>
 *		<li id="item3">
 *			<div class="toggler"><h3>Third header</h3></div>
 *			<div class="content">Shown onmouseover; hides all other elements.</div>
 *		</li>
 *	</ul>
 *
 */

/**
 * Create Custom Event object
 */
var onAccordionComplete = new YAHOO.util.CustomEvent('onAccordionComplete');
 
/**
 * Create the Accordion object
 */
PAYPAL.widget.Accordion = {
	
	toggleReady: null,
	allowMultiple: false,
	animate: true,
	animDelayShow: 0.9,
	animDelayHide: 0.8,
	firstRun: true,
	
	/**
	 * Initialize accordion
	 */
	init: function() {
		var header, body, i, autoShow;
		var accordions = YAHOO.util.Dom.getElementsByClassName('accordion');
		if (!accordions) {return;}
		YAHOO.util.Dom.addClass(accordions, 'dynamic');

		// loop through each accordion
		for (i = 0; i < accordions.length; i++) {
			// get all boxes
			boxes = YAHOO.util.Dom.getElementsByClassName('box', '', accordions[i]);
			for (i = 0; i < boxes.length; i++) 	{
				header = YAHOO.util.Dom.getElementsByClassName('header', 'div', boxes[i])[0];
				body = YAHOO.util.Dom.getElementsByClassName('body', 'div', boxes[i])[0];
				// define content
				body.defaultHeight = body.offsetHeight;
				YAHOO.util.Dom.setStyle(body, 'width', body.offsetWidth); // needed for IE
				header.content = body;
				// open by default?
				if (YAHOO.util.Dom.hasClass(boxes[i], 'defaultOpen')) {
					autoShow = header.content;
				}
				// attach event listener to header
				YAHOO.util.Event.addListener(header, 'mousedown', PAYPAL.widget.Accordion.toggle);
				YAHOO.util.Event.addListener(header, 'mouseover', function(e){
					YAHOO.util.Dom.addClass(this, 'hover');
				});
				YAHOO.util.Event.addListener(header, 'mouseout', function(e){
					YAHOO.util.Dom.removeClass(this, 'hover');
				});
				
			}
			// show content if this is default
			if (autoShow) {
				PAYPAL.widget.Accordion.show(autoShow);
			}
		}
	},
	
	/**
	 * Toggles content visibility, based on current state.
	 * @param {Object} node Content element to hide or show.
	 */
	toggle: function(body) {
		if (this.content) body = this.content;
		var accordion = PAYPAL.widget.Accordion;
		accordion.toggleReady = (accordion.toggleReady === null) ? true : accordion.toggleReady;
		if (YAHOO.util.Dom.hasClass(body.parentNode, 'open') && accordion.toggleReady) {
			body.defaultHeight = body.offsetHeight;			
			accordion.toggleReady = false;
			accordion.hide(body);
		} else if (accordion.toggleReady) {
			accordion.toggleReady = false;
			accordion.show(body);
		}
	},
	
	/**
	 * Callback function for show() and hide()
	 */
	toggleCustom: function() {
		var body = this.getEl();
		var box = body.parentNode;
		box.open = box.open ? false : true;
		if (!box.open) {
			YAHOO.util.Dom.removeClass(box, 'open');
		} else {
			YAHOO.util.Dom.setStyle(body, 'height', 'auto');
		}
		PAYPAL.widget.Accordion.toggleReady = true;	
		
		// fire custom event only after first run
		if (!PAYPAL.widget.Accordion.firstRun) {
			onAccordionComplete.fire(true);
		} else {
			PAYPAL.widget.Accordion.firstRun = false;
		}
	},

	/**
	 * Shows specified element using YAHOO.util.Anim and calls
	 * the hideShown() function to collapse visible content.
	 * @param {Object} node Content element to show.
	 */	
	show: function(body) {
		if (!PAYPAL.widget.Accordion.allowMultiple) {
			PAYPAL.widget.Accordion.hideAll();
		}
		YAHOO.util.Dom.addClass(body.parentNode, 'open');
		if (PAYPAL.widget.Accordion.animate) {
			var attributes = {
				height: {from:0, to:body.defaultHeight},
				opacity: {from:0, to:1}
			};			
			var anim = new YAHOO.util.Anim(body, attributes, PAYPAL.widget.Accordion.animDelayShow, YAHOO.util.Easing.backOut);
			anim.animate();
			anim.onComplete.subscribe(PAYPAL.widget.Accordion.toggleCustom);
		} else {
			var box = body.parentNode;
			box.open = box.open ? false : true;
			if (!box.open) {
				YAHOO.util.Dom.removeClass(box, 'open');
			} else {
				YAHOO.util.Dom.setStyle(body, 'height', 'auto');
			}
			PAYPAL.widget.Accordion.toggleReady = true;	
			
			// fire custom event only after first run
			if (!PAYPAL.widget.Accordion.firstRun) {
				onAccordionComplete.fire(true);
			} else {
				PAYPAL.widget.Accordion.firstRun = false;
			}
		}
	},
	
	/**
	 * Hides specified element using YAHOO.util.Anim.
	 * @param {Object} node Content element to hide.
	 */
	hide: function(body) {
		if (PAYPAL.widget.Accordion.animate) {
			var attributes = {
				height: {from:body.defaultHeight, to:0},
				opacity: {from:1, to:0}
			};
			var anim = new YAHOO.util.Anim(body, attributes, PAYPAL.widget.Accordion.animDelayHide, YAHOO.util.Easing.easeBoth);
			anim.animate();		
			anim.onComplete.subscribe(PAYPAL.widget.Accordion.toggleCustom);
		} else {
			var box = body.parentNode;
			box.open = box.open ? false : true;
			if (!box.open) {
				YAHOO.util.Dom.removeClass(box, 'open');
			} else {
				YAHOO.util.Dom.setStyle(body, 'height', 'auto');
			}
			PAYPAL.widget.Accordion.toggleReady = true;	
			
			// fire custom event only after first run
			if (!PAYPAL.widget.Accordion.firstRun) {
				onAccordionComplete.fire(true);
			} else {
				PAYPAL.widget.Accordion.firstRun = false;
			}
		}
	},
	
	/**
	 * Hides all list elements with class name 'open'.
	 */
	hideAll: function() {
		var boxes = YAHOO.util.Dom.getElementsByClassName('box');
		for (i = 0; i < boxes.length; i++) {
			if (YAHOO.util.Dom.hasClass(boxes[i], 'open')) {
				var body = boxes[i].getElementsByTagName('div')[1];
				PAYPAL.widget.Accordion.hide(body);
			}
		}
	},
	
	/**
	 * Expands or collapses accordion content based on specified class name. 
	 * Class name must be applied to the LI element.
	 */
	toggleAccordion: function(nodeClass) {
		var box = YAHOO.util.Dom.getElementsByClassName(nodeClass);
		if (box[0]) {
			var body = box[0].getElementsByTagName('div')[0];
			PAYPAL.widget.Accordion.toggle(body);
		}
	},

	/**
	 * Check if a specific accordion element is open.
	 * @param {String} nodeClass Class name of element to test.
	 */
	reedIsOpen: function(nodeClass) {
		var node = YAHOO.util.Dom.getElementsByClassName(nodeClass);
		if (node[0]) {
			return (YAHOO.util.Dom.hasClass(node[0], 'open')) ? true : false;
		} else {
			return false;
		}
	}
};
