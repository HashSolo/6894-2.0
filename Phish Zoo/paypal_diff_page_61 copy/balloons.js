/**
* @fileoverview
* This script will automatically style special tooltips and balloon callouts for links on the page. 
*
* Documentation: http://webdev.paypal.com/standards/code-library/interface-elements/tooltips-and-balloon-callouts
*
* TOOLTIPS:
* A tooltip is a simple bit of text that is displayed in a balloon when the user hovers over the link
* to explain it's purpose.  THIS CAN ONLY BE USED ON LINKS WHICH HAVE A VALID URL IN THE HREF AND GO SOMEWHERE.
*
* Implementation: Set the class to "autoTooltip" and put the tooltip content in the title attribute:
*
*	<a href="foo.html" title="Hello World" class="autoTooltip">My First Tooltip</a>
*
* BALLOON CALLOUTS:
* These are more complex tooltips which can contain markup.
*
* Implementation: 
*	+ Create a link with the class "balloonControl"
*	+ Create a div with the class "balloonCallout" and a unique ID
*	+ Add the balloon content to the div
*	+ Add the div's ID to the link's href attribute
*
*	<a href="#myBalloonContent" class="balloonControl">Show panel1</a> 
*
*	<div id="myBalloonContent" class="balloonCallout">
*		<h4>Lorem ipsum</h4>
*		<p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Maecenas aliquet libero vel quam. Sed et magna at urna varius malesuada.</p>
*		<p><a href="#" id="calltoaction">Call to action</a></p>
*	</div>
*
* @requires YAHOO.util.Dom, YAHOO.util.Event, YAHOO.util.Panel
*/


PAYPAL.namespace("widget");

/**
 * Automatically go through the anchors on the page to add a
 * stylized look to the tooltips and balloon callouts.
 */
PAYPAL.widget.Balloons = {
	
	balloonDiv : null,
	balloonIframe : null,
		
	/**
	* Initializes object and anchors on the page
	*/
	init : function(){
		YAHOO.widget.Module.CSS_HEADER = "header";
		YAHOO.widget.Module.CSS_BODY = "body";
		YAHOO.widget.Module.CSS_FOOTER = "footer";
		
		
		// Tooltips
		var tooltips = YAHOO.util.Dom.getElementsByClassName("autoTooltip");
		
		for (var i = 0; i < tooltips.length; i++) {
			var id = "autoTooltip" + i;
			var target = tooltips[i];
			
			// Construct balloon with the title content
			var content = target.getAttribute("title");
			this._initBalloon(target, id, content);
			target.removeAttribute("title");
		}
		
		// Balloon Callouts
		var balloons = YAHOO.util.Dom.getElementsByClassName("balloonControl", "a");
		
		for (var i = 0; i < balloons.length; i++) {
			var target = balloons[i];
			
			if (target.href.indexOf("#") > -1) {
				var calloutId = target.href.split("#")[1];
				var callout = document.getElementById(calloutId);
				
				if (callout) {
					var content = callout.innerHTML;
					
					// Legacy -- remove nested body <div>s
					var child = callout.firstChild;
					while (child) {
						if (child.nodeName == "DIV" && YAHOO.util.Dom.hasClass(child, "body")) {
							content = child.innerHTML;
							break;
						}
						child = child.nextSibling;
					}
					
					YAHOO.util.Dom.addClass(callout, "accessAid");
					this._initBalloon(target, calloutId, content);
					target.removeAttribute("title");
				}
			}
		}
		
		// Preload arrow images
		if (this.balloonDiv) {
			var img, cssClass;
			var arrows = ["posUnder", "posOver"];
			var balloon = this.balloonDiv.body.parentNode;
	
			for (var i = 0; i < arrows.length; i++) {
				cssClass = arrows[i];
				arrows[i] = new Image();
				
				// Add class and then get the arrow background image and preload it
				YAHOO.util.Dom.addClass(balloon, cssClass);
				img = YAHOO.util.Dom.getStyle(balloon, "backgroundImage");
				if (img = img.match(/url\(([^\)]*)\)/) ){
					arrows[i].src = img[1];
				}
				
				YAHOO.util.Dom.removeClass(balloon, cssClass);
			}
		}
	},
	
	/**
	* Create a single balloon.  
	* This method is called from the main init method.
	* @param {DomNode} target The target element that opens the balloon.
	* @param {String} id The ID of the div the balloon should be created in.
	* @param {String} content The content (HTML and/or text) that goes in the balloon
	*/
	_initBalloon : function(target, id, content) {
		var self = this;

		// Instantiate the Panel
		if (!this.balloonDiv) {  
		 	this.balloonDiv = new YAHOO.widget.Panel("balloonCalloutPanel", { close: false, visible: false, zIndex: 17, underlay: "none", draggable: false });
		
			this.balloonDiv.beforeShowEvent.subscribe(this._positionBalloon, this.balloonDiv, true);
			this.balloonDiv.showEvent.subscribe(this._fixIframeShim, this.balloonDiv, true);
			this.balloonDiv.setBody('');
			this.balloonDiv.render(document.body);
			YAHOO.util.Dom.addClass(this.balloonDiv.element, "balloon");
			
			YAHOO.util.Event.addListener(this.balloonDiv.element, "mouseout", function(evt) {
				self._hideIfTargetOutside(target, evt);
			}, this, true);
		}
		
		
		// Add listeners to the target elements		
		YAHOO.util.Event.addListener(target, "mouseover", function() {
			self.balloonDiv.setBody(content);
			self.balloonDiv._target = target;
			self.balloonDiv.show();
		}, this.balloonDiv, true);
		
		YAHOO.util.Event.addListener(target, "mouseout", function(evt) {
			self._hideIfTargetOutside(this, evt);
		}, this, true);
		
		
		// Hide/Show balloon with keyboard navigation
		YAHOO.util.Event.addListener(target, "focus", this.balloonDiv.show, this.balloonDiv, true);
		YAHOO.util.Event.addListener(target, "blur", this.balloonDiv.hide, this.balloonDiv, true);
		
		return this.balloonDiv;
	},
	
	/**
	 * Sets the position static to the context element and
	 * adds the posOver/posUnder class to the panel
	 * @param {String} type The custom event type 'onShow'
	 * @param {Array} args Custom event fire args
	 */
	_positionBalloon : function(type, args, balloons) {
		var balloon = this.element;
		var callout = this.body.parentNode;
		var context = this._target;
		var nav = document.getElementById("navPrimary");
		
		var viewport = PAYPAL.widget.Balloons.getViewportRegion();
		var contextRegion = YAHOO.util.Region.getRegion(context);
		var navRegion = (nav) ? YAHOO.util.Region.getRegion(nav) : null;
		
		YAHOO.util.Dom.removeClass(callout, "accessAid");
		
		// Default to bottom
		var top = contextRegion.top + context.offsetHeight;
		YAHOO.util.Dom.replaceClass(callout, "posOver", "posUnder");
		YAHOO.util.Dom.setY(callout, top);
		
		// Keep balloon in the viewport and under the nav
		if (contextRegion.bottom + balloon.offsetHeight > viewport.bottom && contextRegion.top - balloon.offsetHeight > viewport.top && (!nav || contextRegion.top - callout.offsetHeight > navRegion.bottom)){
			// Place balloon over anchor
			YAHOO.util.Dom.replaceClass(callout, "posUnder", "posOver");
			YAHOO.util.Dom.setY(callout, contextRegion.top - callout.offsetHeight);
		}
		
		// Set static position relative to context
		YAHOO.util.Dom.setX(callout, YAHOO.util.Dom.getX(context) - 10);
	},


	/**
	* Repositions the iframe needed for IE6
	*/
	_fixIframeShim : function() {
		if (this.iframe) {
			this.iframe.style.left = (YAHOO.util.Dom.getX(this.body.parentNode) - 1) + "px";
			this.iframe.style.top = YAHOO.util.Dom.getY(this.body.parentNode) + "px";
		}
	},
	
	/**
	* Checks if the current mouse position is within the target or balloon
	* @param {Object} target The target which triggers the balloon
	* @param {Event} evt An event object
	*/
	_hideIfTargetOutside : function(target, evt) {
		var evTarget = YAHOO.util.Event.getTarget(evt);

		// Mouse is still in target or balloon
		var related = YAHOO.util.Event.getRelatedTarget(evt);

		if (related == target || YAHOO.util.Dom.isAncestor(this.balloonDiv.element, related)){
			return;
		}

		// Is mouse in between the target and balloon
		var point = new YAHOO.util.Point(YAHOO.util.Event.getXY(evt));
		var targetReg = YAHOO.util.Region.getRegion(evTarget);
		var balloonReg = YAHOO.util.Region.getRegion(this.balloonDiv.body);

		// Bottom balloon
		if (balloonReg.top > targetReg.top && point.left > targetReg.left && point.right < targetReg.right && point.top > targetReg.top) {
			return;
		// Top balloon
		} else if (balloonReg.top < targetReg.top && point.left > targetReg.left && point.right < targetReg.right && point.top < targetReg.bottom) {
			return;	
		}

		// If we got this far, the cursor is out of the target and balloon
		this.balloonDiv.hide();
	},
	
	
	/**
	* Gets you the viewport region
	*/
	getViewportRegion : function() {
		var scrollTop = Math.max(document.documentElement.scrollTop, document.body.scrollTop);
		var scrollLeft = Math.max(document.documentElement.scrollLeft, document.body.scrollLeft);

		return new YAHOO.util.Region(scrollTop, YAHOO.util.Dom.getViewportWidth() + scrollLeft, YAHOO.util.Dom.getViewportHeight() + scrollTop, scrollLeft);
	}
	
};

YAHOO.util.Event.addListener(window, "load", PAYPAL.widget.Balloons.init, PAYPAL.widget.Balloons, true);

