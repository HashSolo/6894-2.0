/****************************************************************************
* 																			*
* HW Javascript Core Library												*
* ---------------------------												*
* 																			*
* Author:			Leonard Martin (leonard.martin@heathwallace.com)		*
* Version:			0.0.4													*
* Updated:			5 June 2008												*
* 																			*
* **************************************************************************/


//create alias functions for getElementById and getElementsByClassName
function $(id) {return document.getElementById(id);}
function $$(c,o,t) {return HW.getElementsByClassName(c,o,t);}

// initialise the HW namespace
var HW = {
	/*
	--- CORE FUNCTIONS ---
	These functions are the basic building blocks and should not be removed
	----------------------
	*/
	// browser flags
	isIE:false,
	isMacFF:false,
	// flag if DOM is ready to run code
	dom:{ready:false,timer:null,loaded:false},
	// array of functions to run on page load
	toRun:[],
	/*
	* log(a)
	* outputs to the Firebug console
	* a:	String, Number, Array, or Object to output
	* Returns:	Nothing
	*/
	log:function(a) {
		if(window.console) {window.console.log(a);}
	},
	/*
	* error(a)
	* outputs an error to the Firebug console or alerts to other browsers
	* a:	String, Number, Array, or Object to output
	* Returns:	Nothing
	*/
	error:function(a) {
		if(window.console) {window.console.error(a);}
		else {alert(a);}
	},
	/*
	* getElementsByClassName(cls,n,t)
	* gets an array of elements within certain parameters
	* cls:	String to match classname against
	* n:	Node to search within
	* t:	String - tag name to match against
	* Note: any of these inputs can be set as null to act as a wildcard
	* Returns:	Array fo elements
	*/
	getElementsByClassName:function(cls,n,t) {
		var rtn = [];
		n=n===null?document:n;
		t=t===null?'*':t;
		var els = n.getElementsByTagName?n.getElementsByTagName(t):document.all;
		els = (!els||!els.length) && document.all?document.all:els;
		if(cls==null){return els;}
		for (var i=0,j=0; i<els.length;i++) {
			if(this.hasClass(els[i],cls)) {
				rtn[j++] = els[i];
			}
		}
		return rtn;
	},
	/*
	* querySelectorAll()
	* gets an array of elements within certain parameters
	* css:		CSS Selector to match
	* scope:	Node to search within
	* Returns:	Array of elements matching CSS selector
	*/
	querySelectorAll:function(css,scope) {
		scope = scope || document;
		if(document.querySelectorAll){return scope.querySelectorAll(css);}
		else {return HW.CssParser(css,scope);}
		return null;
	},
	/*
	* querySelector()
	* gets the first element matching a selector
	* css:		CSS Selector to match
	* scope:	Node to search within
	* Returns:	first element matching CSS selector
	*/
	querySelector:function(css,scope) {
		scope = scope || document;
		if(document.querySelector){return scope.querySelector(css);}
		else {
			var o = HW.CssParser(css,scope);
			return o.length?o[0]:null;
		}
	},
	/*
	* createNode(a)
	* creates a DOM node and appends it to a parent node
	* t:	String - tag name of element to be added
	* p:	Node to append new node into
	* c:	Optional - HTML content of element
	* opts:	Optional - additional attributes to be set
	* Returns:	New node
	*/
	createNode:function(t,p,c,opts) {
		if(!p || !t){return};
		var n = document.createElement(t);
		if(c) {n.innerHTML = c;}
		n = HW.extendObject(n,opts);
		return p.appendChild(n);
	},
	/*
	* attachEvent(obj,evt,fnc)
	* attaches an event listener to an element
	* obj:	Object to which event listener is to be added
	* evt:	String - event type e.g. 'click', 'mouseover'
	* fnc:	Function to fire on event
	* Returns:	Nothing
	*/
	attachEvent:function(obj,evt,fnc) {
		if(window.addEventListener) {obj.addEventListener(evt, fnc, false);}
		else if(window.attachEvent) {obj.attachEvent('on'+evt, fnc);}
		else if (obj.getElementById && evt=='load') {obj.onload = fnc;}
	},
	/*
	* detachEvent(obj,evt,fnc)
	* removes an event listener from an element
	* obj:	Object to which event listener is to be removed
	* evt:	String - event type e.g. 'click', 'mouseover'
	* fnc:	Function to remove
	* Returns:	Nothing
	*/
	detachEvent:function(obj,evt,fnc) {
		if(window.removeEventListener) {obj.removeEventListener(evt, fnc, false);}
		else if(window.detachEvent) {obj.detachEvent('on'+evt, fnc);}
	},
	/*
	* preventDefault(e)
	* prevent the default action on event firing
	* e:	Event fired
	* Returns:	Nothing
	*/
	preventDefault:function(e) {
		e=e||window.event;
		if(e.preventDefault) {e.preventDefault();}
		else {e.returnValue = false;}
	},
	/*
	* cancelBubble(e)
	* prevent an event from bubbling up the DOM
	* e:	Event fired
	* Returns:	Nothing
	*/
	cancelBubble:function(e) {
		e=e||window.event;
		if(e.stopPropogation) {e.stopPropogation();}
		else {e.cancelBubble = true;}
	},
	/*
	* extendObject(d,s)
	* add the properties and methods from one object to another
	* d:	Object to which properties and methods should be added
	* s:	Object from which properties and methods should be added
	* Returns:	Object with new properties and methods
	*/
	extendObject:function(d,s) {
		d=d===null?new Object():d;
		for (var p in s) {d[p] = s[p];}
		return d;
	},
	/*
	* addClass(o,c)
	* add a class to an element
	* o:	Node to add class to
	* c:	String - class to add
	* Returns:	Nothing
	*/
	addClass:function(o,c) {
		if (!this.hasClass(o,c)){
			if (o.className == "") {o.className = c;}
			else {o.className += " " + c;}
		}
	},
	/*
	* hasClass(o,c)
	* test if an element has a class
	* o:	Node to check
	* c:	String - class to check for
	* Returns:	Boolean - true if element has class, false otherwise
	*/
	hasClass:function(o,c) {
		var p = new RegExp("(^| )" + c + "( |$)");
		if (p.test(o.className)) {return true;}
		return false;
	},
	/*
	* removeClass(o,c)
	* remove a class from an element
	* o:	Node to remove class from
	* c:	String - class to remove
	* Returns:	Nothing
	*/
	removeClass:function(o,c) {
		var p = new RegExp("(^| )" + c + "( |$)");
		o.className = o.className.replace(p, "$1");
		o.className = o.className.replace(/ $/, "");
	},
	/*
	* setFade(o,c)
	* set the alpha transparency of an element
	* o:	Node to set
	* n:	Number - value between 0-100 (0:transparent,100:opaque)
	* Returns:	Nothing
	*/
	setFade:function(o,n) {
		var agt = navigator.userAgent.toLowerCase();
		if((agt.indexOf("msie") != -1) && (agt.indexOf("opera") == -1)) {
			if (n == 100) {o.style.filter = "";}
			else if (n < 0) {o.style.filter = "progid:DXImageTransform.Microsoft.Alpha(opacity=0);";}
			else {o.style.filter = "progid:DXImageTransform.Microsoft.Alpha(opacity="+ Math.round(n) + ");";}
		}
		else {			
			o.style.MozOpacity = (Math.round(n) / 100);
			o.style.opacity = (Math.round(n) / 100);
		}
		o._alpha = n;
	},
	/*
	* setStyle(o,s)
	* set the CSS style of an element
	* o:	Node to set
	* s:	Object containing style info (e.g. {margin:'1px',padding:'4px'})
	* Note: style object should be comma separated and have properties wrapped in single quotes. 
	* Styles should also be in camel casing - e.g. marginTop not margin-top
	* Returns:	Nothing
	*/
	setStyle:function(o,s) {
		for(var i in s) {
			o.style[i] = s[i];
		}
	},
	/*
	* fixIE6flicker()
	* allow caching of background images to avoid flicker on hover in IE versions
	* Returns:	Nothing
	*/
	fixIE6flicker:function() {
		var m = document.uniqueID && document.compatMode && !window.XMLHttpRequest && document.execCommand ; 
		try { 
			if(!!m) { 
				m("BackgroundImageCache", false, true);
			} 
		}
		catch(e) {};
	},
	/*
	* checkLoaded()
	* check if page has loaded
	* Returns:	Boolean - true if page is loaded, false otehrwise
	*/
	checkLoaded:function() {
		if(HW.dom.ready){return true;}
		if(document && document.getElementsByTagName && document.getElementById && document.body) {
			clearInterval(HW.dom.timer);
			HW.dom.timer = null;
			HW.dom.ready = true;
			return true;
		}
		else {return false}
	},
	/*
	* onload()
	* set a function to run on page load
	* f:		Function to call on page load
	* Returns:	nothing
	*/
	onload:function(f) {
		HW.toRun.push(f);
		if(HW.dom.loaded) {
			f();
		}
	},
	/*
	* load()
	* check if page has loaded, if it has, run code
	* Returns:	Nothing
	*/
	load:function() {
		if(HW.checkLoaded() && !HW.dom.loaded) {
			// mark the page as loaded
			HW.dom.loaded = true;
			// set our browser flags
			var userAgent = navigator.userAgent.toLowerCase();
			HW.isIE  = (navigator.appVersion.indexOf("MSIE") != -1)?true:false;
			HW.isMacFF  = (userAgent.indexOf('mac') != -1 && userAgent.indexOf('firefox')!=-1)?true:false;
			// run anything set to run on load
			for(var i=0,j=HW.toRun.length;i<j;i++) {
				HW.toRun[i]();
			}
			if(HW.isIE) {HW.fixIE6flicker();}
		}
		else if(HW.dom.timer === null) {
			HW.dom.timer = setInterval(HW.load,10);
		}
	}
	/*
	--- END CORE FUNCTIONS ---

	*/
};

// call the page load event
HW.attachEvent(document,'DOMContentLoaded',HW.load);
HW.attachEvent(window,'load',HW.load);

/****************************************************************************
* 																			*

* HW Javascript Ajax Module												*
* ---------------------------												*
* 																			*
* Author:			Leonard Martin (leonard.martin@heathwallace.com)		*
* Version:			0.0.1													*
* Updated:			9 May 2008												*
* 																			*
* **************************************************************************/

/*
--- AJAX FUNCTIONS ---
Requires:	Core
----------------------
*/

/*
* HW.Ajax(url,[callback[,vars[,method]]])
* make an inline request to a file
* url: 		String - the URL to which the request is made
* callback:	Function to call on completion of request
			This function is called with a single parameter a HW.Ajax.Response object
* vars:		String - variables to send to request in format; 'var1=val1&var2=val2&...'
* method:	String - 'GET' or 'POST'
* Returns:	Nothing
*/ 
HW.Ajax = function(url,callback,vars,method) {
	var obj = this;
	method = method?method:'GET';
	vars = vars?vars:null;
	// instantiate request object
	this.req = new HW.Ajax.Request(url,vars,method);
	// set callback function
	if(typeof(callback) == 'function') {this._passResponse = callback;}
	this.req.xmlHttp.onreadystatechange = function() {obj._handle();};
	// send request
	this.req._sendRequest();
};

HW.Ajax.prototype = {
	req:{},
	_handle:function() {
		if(this.req.xmlHttp.readyState == 4) {
			if(this.req.xmlHttp.status == 200) {
				var r = new HW.Ajax.Response(this.req.xmlHttp);
				this._passResponse(r);
			}
		}
	},
	_passResponse:function() {return;}
};

HW.Ajax.Request = function(href,vars,method) {
	this.createXmlHttpRequestObject();
	this.href = href;
	this.vars = vars;
	this.method = method;
};

HW.Ajax.Request.prototype = {
	xmlHttp:null,
	createXmlHttpRequestObject:function() {
		try {
			this.xmlHttp = new XMLHttpRequest();
		} catch(e) {
			var XmlHttpVersions = new Array("MSXML2.XMLHTTP.6.0",
										"MSXML2.XMLHTTP.5.0",
										"MSXML2.XMLHTTP.4.0",
										"MSXML2.XMLHTTP.3.0",
										"MSXML2.XMLHTTP",
										"Microsoft.XMLHTTP");
			for (var i=0; i<XmlHttpVersions.length && !this.xmlHttp; i++) {
				try { 
					this.xmlHttp = new ActiveXObject(XmlHttpVersions[i]);
				}  catch(e){}
			}
		}
	},
	_sendRequest:function(method,vars) {
		if(this.xmlHttp) {
			try {
				if (this.xmlHttp.readyState == 4 || this.xmlHttp.readyState == 0) {
					this.xmlHttp.open(this.method, this.href+(this.method=='GET'&&this.vars?'?'+this.vars:''), true);
					if(this.method == 'POST') {
						this.xmlHttp.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
						this.xmlHttp.setRequestHeader("Content-length", this.vars.length);
						this.xmlHttp.setRequestHeader("Connection", "close");
					}
					this.xmlHttp.send(this.vars);
				}
				else {
					if(timeoutId != -1) clearTimeout(timeoutId);  
					var obj = this;
					timeoutId = setTimeout(function(){obj.sendRequest();}, 500);
				}
			} catch(e){}
		}
	}
};

/*
* Response Object
* This object is the parameter sent to the callback function
* It has three properties:
* xml:		If the resource accessed by the HTTP request returns content type 'text/xml' then this will contain the xml output otherwise null
* text:		If the resource returns any other content type then this will contain the response, otherwise an empty string
* contentType:	The content type returned e.g. 'text/xml', 'text/html', 'image/jpeg'
*/
HW.Ajax.Response = function(xml) {
	this.contentType = xml.getResponseHeader('Content-type')?xml.getResponseHeader('Content-type'):null;
	if(this.contentType.substr(0,4) == 'text') {
		this.xml = xml.responseXML?xml.responseXML:null;
		this.text = xml.responseText?xml.responseText:'';
	}
};
HW.Ajax.Response.prototype = {xml:null,text:null,contentType:null};

/*
--- END AJAX FUNCTIONS ---
*/


/****************************************************************************
* 																			*
* HW Javascript Animate Module												*
* ----------------------------											*
* 																			*
* Author:			Leonard Martin (leonard.martin@heathwallace.com)		*
* Version:			0.0.2													*
* Updated:			12 May 2008												*
* 																			*
* **************************************************************************/

/*
--- ANIMATION FUNCTIONS ---
Requires:	Core
---------------------------
*/

HW.Animate = {
	/*
	* fade(elm,to,time[,c])
	* fade an element in/out
	* elm:		Node - element to fade
	* to:		Number - final alpha value (0:transparent,100:opaque)
	* time:		Number - milliseconds to take over fade
	* c:		Function - callback function, fires when finished
	* Returns:	Nothing
	*/
	fade:function(elm,to,time,c) {
		// if the element has not been faded before, assume it is opaque
		if(!elm._alpha && elm._alpha !== 0) {elm._alpha = 100;}
		new HW.Animator(elm,elm._alpha,to,HW.setFade,time,c);
	}
};

/*
* Animator(o,v0,v1,s[,t[,c]])
* change a property of an element smoothly over a period of time
* o:		Node - element to animate
* v0:		Number - initial value of property to change
* v1:		Number - final value of property to change
* s:		Function - function to use to set current property value, should take an object and a value as a parameter
* t:		Number - time to take over animation, in milliseconds - defaults to 500
* c:		Function - callback function, fires when finished
* Returns:	Nothing
*/
HW.Animator = function(o,v0,v1,s,t,c) {
	if(o) {this.target = o;}
	this.setFunc = s;
	this.startValue = v0;
	this.endValue = v1;
	if(t) {this.time = t;}
	if(typeof(c) == 'function') {this.callback = c;}
	this.steps = Math.ceil(this.time/this.stepLength);
	this.animate();
};

HW.Animator.prototype = {
	stepLength:30,
	time:500,
	steps:20,
	setFunc:function(){},
	callback:function(){},
	set:function(v) {
		this.setFunc(this.target,v);
	},
	animate:function() {
		var obj = this;
		var df = (this.endValue - this.startValue)/this.steps;
		if(df != 0) {
			// set timers to fire at intervals
			for(var i=1,j=this.steps;i<=j;i++) {
				(function(){
					var j=i;
					setTimeout(function(){
						obj.set(obj.startValue + j*df);
					},j*obj.stepLength);
				})();
			}
			// fire finish when loop has finished
			setTimeout(function(){obj.callback();},this.stepLength*this.steps);
		}
		else {
			this.callback();
		}
	}
};

/*
--- END ANIMATION FUNCTIONS ---
*/

/*
--- CSS SELECTOR PARSER ---

=:based on code from
scalable Inman Flash Replacement (sIFR) version 3, Author: Mark Wubben, <http://novemberborn.net/>"

=:license
This software is licensed and provided under the CC-GNU LGPL.
See <http://creativecommons.org/licenses/LGPL/2.1/>    

*/

HW.CssParser = (function() {
	var B = /\s*,\s*/;
	var A = /\s*([\s>+~(),]|^|$)\s*/g;
	var L = /([\s>+~,]|[^(]\+|^)([#.:@])/g;
	var F = /(^|\))[^\s>+~]/g;
	var M = /(\)|^)/;
	var K = /[\s#.:>+~()@]|[^\s#.:>+~()@]+/g;
	
	function Parser(css, scope) {
		scope = scope || document.documentElement;
		var selectors = css.split(B);
		var a = [];
		for (var i = 0; i < selectors.length; i++) {
			var o = [scope];
			var sel = Parser.clean(selectors[i]);
			for (var j = 0; j < sel.length;) {
				var prefix = sel[j++];
				var fragment = sel[j++];
				var paren = "";
				if (sel[j] == "(") {
					while (sel[j++] != ")" && j < sel.length) {
						paren += sel[j];
					}
					paren = paren.slice(0, -1);
				}
				o = Parser.get(o, prefix, fragment, paren);
			}
			a = a.concat(o);
		}
		return Parser.util.unique(a);
	}
	Parser.clean = function(selector) {
		// strip unnecessary whitespace
		var o = selector.replace(A, "$1")
		// place asterisks before #s
		o = o.replace(L, "$1*$2")
		// add spaces after closing brackets
		o = o.replace(F, function(s){return s.replace(M, "$1 ")});
		return o.match(K) || []
	}
	Parser.get = function(scope, prefix, fragment, paren) {
		return (Parser.selectors[prefix]) ? Parser.selectors[prefix](scope, fragment, paren) : []
	}
	Parser.util = {
		toArray: function(o) {
			var a = [];
			for (var i = 0; i < o.length; i++) {
				a.push(o[i])
			}
			return a;
		},
		unique: function(o) {
			var a = [];
			for(var i=0;i<o.length;i++) {
				if(!this.inArray(o[i],a)) {a.push(o[i]);}
			}
			return a;
		},
		inArray:function(o,a) {
			for(var j=0;j<a.length;j++) {
				if(a[j] == o) {return true;}
			}
			return false;
		}
	};
	Parser.dom = {
		isTag: function(O, N) {
			return (N == "*") || (N.toLowerCase() == O.nodeName.toLowerCase())
		},
		previousSiblingElement: function(o) {
			while ( o && o.nodeType != 1 ) {
				o = o.previousSibling
			}
			return o;
		},
		nextSiblingElement: function(o) {
			while ( o && o.nodeType != 1 ) {
				o = o.nextSibling;
			}
			return o;
		},
		hasClass: function(cls, o) {
			return (o.className || "").match("(^|\\s)" + cls + "(\\s|$)");
		},
		getByTag: function(tag, o) {
			return o.getElementsByTagName(tag);
		}
	};
	Parser.selectors = {
		"#": function(scope, id) {
			for(var i=0;i<scope.length;i++) {
				if (scope[i].getAttribute("id") == id) {
					return [scope[i]];
				}
			}
			return [];
		},
		" ": function(scope, tag) {
			var a = [];
			for(var i=0;i<scope.length;i++) {
				a = a.concat(Parser.util.toArray(Parser.dom.getByTag(tag,scope[i])));
			}
			return a
		},
		">": function(scope,child) {
			var a = [];
			for (var i=0;i<scope.length;i++) {
				var parent = scope[i];
				for (var j=0;j<parent.childNodes.length;j++) {
					var node = parent.childNodes[j];
					if (node.nodeType == 1 && Parser.dom.isTag(node, child)) {
						a.push(node);
					}
				}
			}
			return a;
		},
		".": function(scope,cls) {
			var a = [];
			for (var i=0;i<scope.length;i++) {
				var node = scope[i];
				if (Parser.dom.hasClass([cls], node)) {
					a.push(node);
				}
			}
			return a;
		},
		":": function(scope,pseudo,paren) {
			// we can define methods for particular pseudoclasses in the Parser.pseudoClasses object
			// none are currently defined
			return (Parser.pseudoClasses[pseudo]) ? Parser.pseudoClasses[psuedo](scope, paren) : []
		}
	};
	Parser.pseudoClasses = {};
	return Parser;
})();

/*
--- END CSS SELECTOR PARSER ---
*/