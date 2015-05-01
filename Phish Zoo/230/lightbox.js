PAYPAL.namespace("util");

/**
* The PayPal lightbox standard implementation.
* Any style properties like width and colors should be set in the CSS.
* @requires YAHOO.widget.Panel
* @param {Mixed} elem The unique ID or DOM element for the lightbox or a DOM element.  
*                     	If the ID does not exist on the page, an element will be created for it.
* @param {Object[]} buttons Form buttons for the lightbox,
*/
PAYPAL.util.Lightbox = function(elem, buttons) {
	var obj = elem;

	// workaround: return a current instance if it exists
	if (typeof obj == "object" || (obj = document.getElementById(elem))) {
		if (obj.lightboxInstance) {
			return obj.lightboxInstance;
		}
	}

	this.init(elem, buttons);
};

PAYPAL.util.Lightbox.prototype = { 
	
	/**
	* TRUE if the lightbox markup has been rendered.
	* @type boolean
	*/
	rendered : false,
	
	/**
	* TRUE if the lightbox is visible on the page.
	* @type boolean
	*/
	showing : false,
	
	/**
	* The lightbox panel.
	* DO NOT handle this object directly without talking
	* to the Platform Team.
	* @type YAHOO.widget.Panel
	*/
	panel : null,
	
	/**
	* Button container
	* @type DomNode
	* @private
	*/
	buttons : null,
	/**
	* @type DomNode
	* @private
	*/
	target:null,
	
	/**
	* @private
	*/
	elem : null,
	
	/**
	* For IE6 and below, <select> elements are hidden
	* @private
	*/
	selects : null,

	/**
	* Initialize the lightbox
	* @private
	*/
	init : function(elem, config){	
		var self = this;
		
		this.elem = elem;

		config = config || {};
		
		YAHOO.widget.Module.CSS_HEADER = "header";
		YAHOO.widget.Module.CSS_BODY = "body";
		YAHOO.widget.Module.CSS_FOOTER = "footer";
		
		this.panel = new YAHOO.widget.Panel(elem, 
												{	modal: true,
													close: true,
													draggable: false,
													//fixedcenter: true,
													postmethod: "form"
												});
												
		

		// Setup the close event
		this.panel.hideEvent.subscribe(function(){ 
			self.showing = false;
			var tar=self.target;
			self.close();
            		if (tar){
                		try {
                  			tar.focus();
                		} catch(X) {}
            		}	
		});
		
		// Animation
		if(typeof YAHOO.widget.ContainerEffect != "undefined"){
			this.panel.cfg.queueProperty("effect", {
														effect: YAHOO.widget.ContainerEffect.FADE, 
														duration:0.25
													});	
		}
		
		// Is from markup
		if(typeof elem == "object" || (elem = document.getElementById(this.elem))){
			this.rendered = true;
			
			var btns = YAHOO.util.Dom.getElementsByClassName("buttons", "*", elem);
			if(btns.length > 0){
				this.buttons = btns[0];
			}
			
			// Remove styles hiding it from view
			if(YAHOO.util.Dom.hasClass(this.elem, "accessAid")){
				YAHOO.util.Dom.removeClass(this.elem, "accessAid")
			}
		}
		
		// Configurations
		if(config.header){
			this.setHeader(config.header);
		}
		if(config.body){
			this.setBody(config.body);
		}
		
		if(typeof config.close != "undefined"){
			this.panel.cfg.queueProperty("close", config.close); 
		}
		
		if(config.buttons){
			this.setButtons(config.buttons);
		}
		
		this.addButtonEvents();
		
		// Escape Key Listener
		var key = new YAHOO.util.KeyListener(document,  { keys:27 }, 
														{ fn: this.panel.hide,
														  scope: this.panel,
														  correctScope: true },
														 "keyup" );
		this.panel.cfg.queueProperty("keylisteners", key); 
	},
	
	/**
	* Set header HTML
	* @param {String} html The HTML for the header
	*/
	setHeader : function(html){
		this.panel.setHeader(html);
		
		if(this.rendered){
			this.panel.sizeUnderlay();
			this.panel.center();
		}
	},
	
	/**
	* Get header HTML
	* @return String
	*/
	getHeader : function(){
		if(this.panel.header){
			return this.panel.header.innerHTML;
		}
		return "";
	},
	
	/**
	* Set body HTML
	* @param {String} html The HTML for the body
	*/
	setBody : function(html){
		this.panel.setBody(html);
		
		if(this.buttons){
			this.panel.body.appendChild(this.buttons);
		}
		
		if(this.rendered){
			this.panel.sizeUnderlay();
			this.panel.center();
		}
	},
	
	/**
	* Get body HTML
	* @return String
	*/
	getBody : function(){
		if(this.panel.body){
			return this.panel.body.innerHTML;
		}
		return "";
	},
	
	/**
	* Add a new lightbox "page", which will completely 
	* wipe out the current lightbox content.
	*
	* This function is used to easily replace the lightbox content with new markup
	* for the header, body and buttons.  This function will extract the header and body 
	* divs from the HTML string and add it to the lightobox.
	*
	* In the future this function will also add back button support.
	*
	* NOTE: THIS FUNCTION TARGETS THE header, body and messageBox DIV ELEMENTS AND WILL 
	*	IGNORE ALL OTHER CONTENT/MARKUP AROUND THEM
	*
	* BUTTONS SHOULD BE PART OF THE body DIV ELEMENT.
	*
	* @param {String|Object} html The string or data object containing the HTML
	*/
	addPage : function(html){
	
		if(typeof html == "object"){
			html = html.html;
		}
		
		if(!html){
			return;
		}
		
		// Extract elements
		var header, body, errors;
		var div = document.createElement("div");
		div.innerHTML = html;
		header = YAHOO.util.Dom.getElementsByClassName("header", "div", div);
		body = YAHOO.util.Dom.getElementsByClassName("body", "div", div);
		errors = YAHOO.util.Dom.getElementsByClassName("messageBox", "div", div);
		
		// Header
		if(header.length > 0){
			this.setHeader(header[0].innerHTML);
		}
		
		// Body
		if(body.length > 0){
			body = body[0];
			this.setBody(body.innerHTML);
		
			// Reset buttons
			if(this.buttons){
				this.buttons.innerHTML = "";
			}
			this.buttons = null;
			var btns = YAHOO.util.Dom.getElementsByClassName("buttons", "span", this.panel.body);
			if(btns.length > 0){
				this.buttons = btns[0];
				this.addButtonEvents();
			}
		}
		
		// Add error box
		if(errors.length > 0){
			if(this.panel.body.firstChild){
				this.panel.body.insertBefore(errors[0], this.panel.body.firstChild);
			}
			else{
				this.panel.body.appendChild(errors[0]);
			}
		}
		
		if(this.rendered){
			this.panel.sizeUnderlay();
			this.panel.center();
		}
		
	},
	
	/**
	* Add buttons to the lightbox.
	* @param {Array} buttons Button array
	*/
	setButtons : function(buttons){
		
		// Setup -- empty button and create div
		if(this.buttons){
			while(this.buttons.firstChild){
				this.buttons.removeChild(this.buttons.firstChild);
			}
		}
		else{
			this.buttons = document.createElement("div");
			this.buttons.className = "buttons";
		}
		
		// Create buttons
		var btn, elem, handler, hasDefault;
		for(var i = 0; i < buttons.length; i++){
			btn = buttons[i];
			elem = document.createElement("input");
			elem.className = "button";
			elem.setAttribute("value", btn.text);
			
			// Type
			if(btn.type == "submit"){
				elem.setAttribute("type", "submit");
			}
			else{
				elem.setAttribute("type", "button");
			}
			
			// Name
			if(btn.name){
				elem.setAttribute("name", btn.name);
			}

			// Default
			if(btn.isDefault) {
				YAHOO.util.Dom.addClass(elem, "primary default");
				this.panel.defaultHtmlButton = elem;
			}
			
			// Add
			this.buttons.appendChild(elem);
			
			// Button handlers
			if(btn.closer){
				YAHOO.util.Dom.addClass(elem, "closer");
			}
			else{
				YAHOO.util.Event.addListener(elem, "click", btn.handler, this, true);
			}
		}
		
		// Finish up
		if(!this.panel.body){
			this.panel.setBody("");
		}
		this.panel.body.appendChild(this.buttons);
		this.addButtonEvents();
		
		if(this.rendered){
			this.panel.sizeUnderlay();
			this.panel.center();
		}
	},
	
	/**
	* Set predefined events on the buttons.
	*
	* 	EVENTS
	*		Closer: If the button has the class "closer", an event will be 
	*               added to close the lightbox when the button is clicked.
	*/
	addButtonEvents : function(){
		
		if(!this.buttons){
			return false;
		}
		
		// Get buttons
		var btn;
		var btns = YAHOO.util.Dom.getElementsBy(function(elem){
			return (elem.type && (elem.type == "button" || elem.type == "submit"));
		}, "*", this.buttons);
		
		
		// Set events
		for(var i = 0; i < btns.length; i++){
			btn = btns[i];
			
			// Closer event
			if(YAHOO.util.Dom.hasClass(btn, "closer")){
				YAHOO.util.Event.addListener(btn, "click", this.panel.hide, this.panel, true);
			}
		}
		
	},
	
	/**
	* Alias of addButtonEvents() for backwards compatibility.
	*/
	addDefinedEvents : this.addButtonEvents,
	
	/**
	* Show the lightbox on the page
	*/
	show : function(e){
		try{
		var ev = e || YUE.getEvent();	
		if(typeof ev !='undefined') {
			this.target=YUE.getTarget(ev);
		}
		}catch(e){}
		var elem = this.elem,header = this.panel.header, body = this.panel.body;
		if(typeof this.elem == "object" || (elem = YUD.get(this.elem))){
			elem.style.display = "block";
		}
		
		// Render
		if (!this.rendered) {
			this.panel.render(document.body);
		} else {
			this.panel.render();
		}


		// workaround: create a backwards reference we can return later
                if (typeof this.elem == "object" || (elem = document.getElementById(this.elem))) {
                        elem.lightboxInstance = this;
                }

		YUD.addClass(this.panel.element, "lightbox");
		this.rendered = true;
		
		// Show
		this.panel.show();
		this.showing = true;
		this.panel.sizeUnderlay();
		this.panel.center();
		
		// Check for top position of lightbox and ensure the top is displayed in the page
		var elemXY = YAHOO.util.Dom.getXY(this.panel.element)
		if (elemXY[1] < 0) {
			elemXY[1] = 1;
		}
		YAHOO.util.Dom.setXY(this.panel.element, elemXY);
		
		
		/* if header is available make it tabable for keyboard only user and also for screen readers */ 
		if(header){
			header.tabIndex="0";
		}
		
		this.panel.firstElement = elem;
		if (elem) {
		elem.focus();
		}

		/* if error is there for miniPage lightbox, make the error tabbable so that user can read error */
		var error = YAHOO.util.Dom.getElementsByClassName("messageBox", "div", this.panel.element);
		if(error.length > 0) {
			error = error[0];
			error.tabIndex="0";
		}

		// Hide selects for IE6 and below
		if(this.panel.browser == "ie"){
			if(this.selects == null){
				this.selects = [];
			}
			
			var sel = document.getElementsByTagName("select");
			for(var i = 0; i < sel.length; i++){
				if(!YAHOO.util.Dom.isAncestor(this.elem, sel[i]) && !YAHOO.util.Dom.hasClass(sel[i], "accessAid")){
					YAHOO.util.Dom.addClass(sel[i], "accessAid");
					this.selects[this.selects.length] = sel[i];
				}
			}
		}
	},
	
	/**
	* Hide the lightbox on the page
	*/
	close : function(){
		if(this.showing){
			this.panel.hide();
		}
		
		// Show <select> elements again for IE6
		if(this.selects != null){
			for(var i = 0; i < this.selects.length; i++){
				YAHOO.util.Dom.removeClass(this.selects[i], "accessAid");
			}
			this.selects = [];
		}
	}
	
}
