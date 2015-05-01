//	mootools.js: moo javascript tools
//	by Valerio Proietti (http://mad4milk.net) MIT-style license.
	
//  CREDITS:

//	Class is slightly based on Base.js : http://dean.edwards.name/weblog/2006/03/base/
//		(c) 2006 Dean Edwards, License: http://creativecommons.org/licenses/LGPL/2.1/

//	Some functions are based on those found in prototype.js : http://prototype.conio.net/
//		(c) 2005 Sam Stephenson <sam@conio.net>, MIT-style license


//moo.js : My Object Oriented javascript - has no dependancies

var Class = function(properties){
	var klass = function(){
		for (p in this) this[p]._proto_ = this;
		if (arguments[0] != 'noinit' && this.initialize) return this.initialize.apply(this, arguments);
	};
	klass.extend = this.extend;
	klass.implement = this.implement;
	klass.prototype = properties;
	return klass;
};

Class.empty = function(){};

Class.create = function(properties){
	return new Class(properties);
};

Class.prototype = {
	extend: function(properties){
		var prototype = new this('noinit');
		for (property in properties){
			var previous = prototype[property];
			var current = properties[property];
			if (previous && previous != current) current = previous.parentize(current) || current;
			prototype[property] = current;
		}
		return new Class(prototype);
	},
	
	implement: function(properties){
		for (property in properties) this.prototype[property] = properties[property];
	}
}

Object.extend = function(){
	var args = arguments;
	if (args[1]) args = [args[0], args[1]];
	else args = [this, args[0]];
	for (property in args[1]) args[0][property] = args[1][property];
	return args[0];
};

Object.Native = function(){
	for (var i = 0; i < arguments.length; i++) arguments[i].extend = Class.prototype.implement;
};

new Object.Native(Function, Array, String);

Function.extend({
	parentize: function(current){
		var previous = this;
		return function(){
			this.parent = previous;
			return current.apply(this, arguments);
		};
	}
});

//Function.js : Function extension - Depends on Moo.js

Function.extend({

	bind: function(bind){
		var fn = this;
		return function(){
			return fn.apply(bind, arguments);
		};
	},

	bindAsEventListener: function(bind){
		return function(event){
			this.call(bind, event || window.event);
			return false;
		}.bind(this);
	},

	delay: function(ms, bind){
		return setTimeout(this.bind(bind || this._proto_ || this), ms);
	},

	periodical: function(ms, bind){
		return setInterval(this.bind(bind || this._proto_ || this), ms);
	}

});

function $clear(timer){
	clearTimeout(timer);
	clearInterval(timer);
	return null;
};

function $type(obj, types){
	if (!obj) return false;
	var type = false;
	if (obj instanceof Function) type = 'function';
	else if (obj.nodeName){
		if (obj.nodeType == 3 && !/\S/.test(obj.nodeValue)) type = 'textnode';
		else if (obj.nodeType == 1) type = 'element';
	}
	else if (obj instanceof Array) type = 'array';
	else if (typeof obj == 'object') type = 'object';
	else if (typeof obj == 'string') type = 'string';
	else if (typeof obj == 'number' && isFinite(obj)) type = 'number';
	return type;
};

function $check(obj, ret){
	if (obj) return ret || obj;
	return false;
};

var Chain = new Class({

	chain: function(fn){
		this.chains = this.chains || [];
		this.chains.push(fn);
		return this;
	},

	callChain: function(){
		if (this.chains && this.chains.length) this.chains.splice(0, 1)[0].delay(10, this);
	}

});

//Array.js : Array extension - depends on Moo.js

if (!Array.prototype.forEach){
	Array.prototype.forEach = function(fn, bind){
		for(var i = 0; i < this.length ; i++) fn.call(bind, this[i], i);
	};
}

Array.extend({
	
	each: Array.prototype.forEach,
	
	copy: function(){
		var nArray = [];
		for (var i = 0; i < this.length; i++) nArray.push(this[i]);
		return nArray;
	},
	
	remove: function(item){
		for (var i = 0; i < this.length; i++){
			if (this[i] == item) this.splice(i, 1);
		}
		return this;
	},
	
	test: function(item){
		for (var i = 0; i < this.length; i++){
			if (this[i] == item) return true;
		};
		return false;
	},
	
	extend: function(nArray){
		for (var i = 0; i < nArray.length; i++) this.push(nArray[i]);
		return this;
	}
	
});

function $A(array){
	return Array.prototype.copy.call(array);
};

//String.js : String extension - depends on Moo.js

String.extend({
	
	test: function(value, params){
		return new RegExp(value, params).exec(this);
	},
	
	camelCase: function(){
		return this.replace(/-\D/gi, function(match){
			return match.charAt(match.length - 1).toUpperCase();
		});
	},
	
	capitalize: function(){
		return this.toLowerCase().replace(/\b[a-z]/g, function(match){
			return match.toUpperCase();
		});
	},
	
	trim: function(){
		return this.replace(/^\s*|\s*$/g,'');
	},
	
	clean: function(){
		return this.replace(/\s\s/g, ' ').trim();
	},
	
	toHex: function(){
		var N = parseInt(this);
		if (N==0 || isNaN(N)) return "00";
		N = Math.round(Math.min(Math.max(0,N),255));
		return "0123456789ABCDEF".charAt((N-N%16)/16) + "0123456789ABCDEF".charAt(N%16);
	},
	
	rgbToHex: function(){
		var rgb = this.match(/[rgba]{3,4}\(([\d]{0,3}),[\s]([\d]{0,3}),[\s]([\d]{0,3})\)/);
		return rgb[1].toHex()+rgb[2].toHex()+rgb[3].toHex();
	}
	
});

//Element.js : Element methods - depends on Moo.js + Native Scripts

var Element = new Class({
	
	//creation
	
	initialize: function(el){
		if ($type(el) == 'string') el = document.createElement(el);
		return $(el);
	},
	
	//injecters
	
	inject: function(el, where){
		el = $(el);
		if (!el) return;
		switch(where){
			case "before": $(el.parentNode).insertBefore(this, el); break;
			case "after": {
					if (!el.getNext()) $(el.parentNode).appendChild(this);
					else $(el.parentNode).insertBefore(this, el.getNext());
			} break;
			case "inside": el.appendChild(this); break;
		}
		return this;
	},
	
	injectBefore: function(el){
		return this.inject(el, 'before');
	},
	
	injectAfter: function(el){
		return this.inject(el, 'after');
	},
	
	injectInside: function(el){
		return this.inject(el, 'inside');
	},
	
	adopt: function(el){
		return new Element(el).injectInside(this);
	},
	
	//actions
	
	remove: function(){
		this.parentNode.removeChild(this);
	},
	
	clone: function(){
		return $(this.cloneNode(true));
	},
	
	hasClassName: function(className){
		return $check(this.className.clean().test("(^| )"+className.trim()+"($| )", "g"), true);
	},
	
	addClassName: function(className){
		if (!this.hasClassName(className)) this.className = (this.className+' '+className.trim()).clean();
		return this;
	},
	
	removeClassName: function(className){
		if (this.hasClassName(className)) this.className = this.className.replace(className.trim(), '').clean();
		return this;
	},
	
	toggleClassName: function(className){
		if (this.hasClassName(className)) return this.removeClassName(className);
		else return this.addClassName(className);
	},
	
	removeStyles: function(){
		$A(arguments).each(function(property){
			this.style[property.camelCase()] = '';
		}, this);
		return this;
	},
	
	//events
	
	addEvent: function(action, fn){
		this[action+fn] = fn.bind(this);
		if (this.addEventListener) this.addEventListener(action, fn, false);
		else this.attachEvent('on'+action, this[action+fn]);
		var el = this;
		if (this != window) Unload.functions.push(function(){
			el.removeEvent(action, fn);
			el[action+fn] = null;
		});
		return this;
	},
	
	removeEvent: function(action, fn){
		if (this.removeEventListener) this.removeEventListener(action, fn, false);
		else this.detachEvent('on'+action, this[action+fn]);
		return this;
	},
	
	//get siblings
	
	getBrother: function(what){
		var el = this[what+'Sibling'];
		while ($type(el) == 'textnode') el = el[what+'Sibling'];
		return el;
	},
	
	getPrevious: function(){
		return this.getBrother('previous');
	},
	
	getNext: function(){
		return this.getBrother('next');
	},
	
	//setters
	
	setStyle: function(property, value){
		if (property == 'opacity') this.setOpacity(value);
		else this.style[property.camelCase()] = value;
		return this;
	},
	
	setOpacity: function(opacity){
		if (opacity == 0 && this.style.visibility != "hidden") this.style.visibility = "hidden";
		else if (this.style.visibility != "visible") this.style.visibility = "visible";
		if (window.ActiveXObject) this.style.filter = "alpha(opacity=" + opacity*100 + ")";
		this.style.opacity = opacity;
		return this;
	},
	
	setStyles: function(source){
		for (property in source) this.setStyle(property, source[property]);
		return this;
	},
	
	setProperty: function(property, value){
		if (property == 'class') this.className = value;
		else if (property == 'style') this.setStyles(value);
		else this.setAttribute(property, value);
		return this;
	},
	
	setProperties: function(source){
		for (property in source) this.setProperty(property, source[property]);
		return this;
	},
	
	setHTML: function(html){
		this.innerHTML = html;
		return this;
	},
	
	//getters
	
	getStyle: function(property){
		if (document.defaultView) return $check(document.defaultView.getComputedStyle(this,null).getPropertyValue(property));
		else if (this.currentStyle) return $check(this.currentStyle[property.camelCase()]);
	},
	
	getTag: function(){
		return this.tagName.toLowerCase();
	},
	
	getOffset: function(what){
		what = what.capitalize();
		var el = this;
		var offset = 0;
		do {
			offset += el['offset'+what] || 0;
			el = el.offsetParent;
		} while (el);
		return offset;
	},
	
	getTop: function(){
		return this.getOffset('top');
	},
	
	getLeft: function(){
		return this.getOffset('left');
	}
});

new Object.Native(Element);

function $(el){
	if ($type(el) == 'string') el = $check(document.getElementById(el));
	if (el && !el.extend){
		Unload.elements.push(el);
		el.extend = Object.extend;
		el.extend(Element.prototype);
	}
	return $check(el);
};

//garbage collector

window.addEvent = Element.prototype.addEvent;
window.removeEvent = Element.prototype.removeEvent;

var Unload = {
	
	elements: [], functions: [], vars: [],
	
	unload: function(){
		Unload.functions.each(function(fn){
			fn();
		});
		
		window.removeEvent('unload', window.removeFunction);
		
		Unload.elements.each(function(el){
			for(p in Element.prototype){
				window[p] = null;
				document[p] = null;
				el[p] = null;
			}
			el.extend = null;
		});
	}
	
};
window.removeFunction = Unload.unload;
window.addEvent('unload', window.removeFunction);

//Fx.js - depends on Moo.js + Native Scripts

var Fx = fx = {};

Fx.Base = new Class({

	setOptions: function(options){
		this.options = Object.extend({
			duration: 500,
			onComplete: Class.empty,
			unit: 'px',
			wait: true,
			transition: Fx.sinoidal,
			fps: 30
		}, options || {});
	},

	step: function(){
		var currentTime  = (new Date).getTime();
		if (currentTime >= this.options.duration+this.startTime){
			this.clearTimer();
			this.now = this.to;
			this.options.onComplete.delay(10, this);
			this.callChain();
		} else {
			this.tPos = (currentTime - this.startTime) / this.options.duration;
			this.setNow();
		}
		this.increase();
	},

	setNow: function(){
		this.now = this.compute(this.from, this.to);
	},

	compute: function(from, to){
		return this.options.transition(this.tPos) * (to-from) + from;
	},

	custom: function(from, to){
		if(!this.options.wait) this.clearTimer();
		if (this.timer) return;
		this.from = from;
		this.to = to;
		this.startTime = (new Date).getTime();
		this.timer = this.step.periodical(Math.round(1000/this.options.fps), this);
		return this;
	},

	set: function(to){
		this.now = to;
		this.increase();
		return this;
	},

	clearTimer: function(){
		this.timer = $clear(this.timer);
		return this;
	},

	setStyle: function(el, property, value){
		if (property == 'opacity'){
			if (value == 1 && navigator.userAgent.test('Firefox')) value = 0.9999;
			el.setOpacity(value);
		} else el.setStyle(property, value+this.options.unit);
	}

});

Fx.Base.implement(new Chain);

Fx.Style = Fx.Base.extend({

	initialize: function(el, property, options){
		this.el = $(el);
		this.setOptions(options);
		this.property = property.camelCase();
	},
	
	hide: function(){
		return this.set(0);
	},
	
	increase: function(){
		this.setStyle(this.el, this.property, this.now);
	}

});

Fx.Layout = Fx.Style.extend({
	
	initialize: function(el, layout, options){
		this.parent(el, layout, options);
		this.layout = layout.capitalize();
		this.iniValue = this.el['offset'+this.layout];
		this.el.setStyle('overflow', 'hidden');
	},
	
	toggle: function(){
		if (this.el['offset'+this.layout] > 0) return this.custom(this.el['offset'+this.layout], 0);
		else return this.custom(0, this.el['scroll'+this.layout]);
	},

	show: function(){
		return this.set(this.el['scroll'+this.layout]);
	}
	
});

Fx.Height = Fx.Layout.extend({

	initialize: function(el, options){
		this.parent(el, 'height', options);
	}

});

Fx.Width = Fx.Layout.extend({

	initialize: function(el, options){
		this.parent(el, 'width', options);
	}

});

Fx.Opacity = Fx.Style.extend({

	initialize: function(el, options){
		this.parent(el, 'opacity', options);
		this.now = 1;
	},

	toggle: function(){
		if (this.now > 0) return this.custom(1, 0);
		else return this.custom(0, 1);
	},

	show: function(){
		this.set(1);
	}

});

Element.extend({

	effect: function(property, options){
		return new Fx.Style(this, property, options);
	}

});

Fx.sinoidal = function(pos){return ((-Math.cos(pos*Math.PI)/2) + 0.5);}; //this transition is from script.aculo.us

Fx.linear = function(pos){return pos;};

Fx.cubic = function(pos){return Math.pow(pos, 3);};

Fx.circ = function(pos){return Math.sqrt(pos);};

//Dom.js - depends on Moo.js + Native Scripts

function $$(array){
	array = $A(array);
	array.implement = Object.extend;
	array = array.implement(new Elements);
	array.each(function(el){
		el = $(el);
	});
	return array;
};

function $S(){
	var els = [];
	$A(arguments).each(function(sel){
		if ($type(sel) == 'string') els.extend(document.getElementsBySelector(sel));
		else if ($type(sel) == 'element') els.push(sel);
	});
	return $$(els);
};

function $E(selector, filter){
	return ($(filter) || document).getElement(selector);
};

Element.extend({

	getElements: function(selector){
		var params = [];
		selector.clean().split(' ').each(function(sel, i){
			params[i] = param = [];
			var bits = [];
			if (bits = /^[\w#.-]*?\[["'\s]{0,1}([\w-]*)["'\s]{0,1}([\W]{0,1}=){0,2}["'\s]{0,1}([\w-]*)["'\s]{0,1}\]$/.exec(sel)){
				param['attribute'] = bits[1];
				param['special'] = bits[2];
				param['value'] = bits[3];
			}
			if (bits = sel.test('^([a-zA-Z]*)([.#])([a-z0-9_-]*)$')){
				param['tag'] = bits[1];
				if (bits[2] == '#') param['id'] = bits[3];
				else param['class'] = bits[3];
			} else if (bits = sel.test('([a-zA-Z0-9]*)')) param['tag'] = bits[1];	
		});
		
		var filters = [];
		params.each(function(param, k){
			if(!param['tag']) param['tag'] = '*';
			if (k == 0){				
				if (param['id']){
					var el = this.getElementById(param['id']);
					if (el && (param['tag'] == '*' || el.getTag() == param['tag'])) filters = [$(el)];
					else return false;
				} else filters = $$(this.getElementsByTagName(param['tag']));
			} else {
				filters = $$(filters).getElementsByTagName(param['tag']);
				if (param['id']) filters = $$(filters).filterById(param['id']);
			}

			if (param['class']) filters = $$(filters).filterByClassName(param['class']);
			if(param['attribute']) filters = $$(filters).filterByAttribute(param['attribute'], param['value'], param['special']);
		}, this);

		return $$(filters);
	},
	
	getElement: function(selector){
		return this.getElementsBySelector(selector)[0];
	},

	getElementsBySelector: function(selector){
		var els = [];
		selector.split(',').each(function(sel){
			els.extend(this.getElements(sel.trim()));
		}, this);
		return $$(els);
	}

});

document.extend = Object.extend;

document.extend({
	
	getElementsByClassName: function(className){
		return document.getElements('.'+className);
	},
	getElement: Element.prototype.getElement,
	getElements: Element.prototype.getElements,
	getElementsBySelector: Element.prototype.getElementsBySelector
	
});

var Elements = new Class({
	
	action: function(actions){
		this.each(function(el){
			el = $(el);
			if (actions.initialize) actions.initialize.apply(el);
			for(action in actions){
				var evt = false;
				if (action.test('^on[\\w]{1,}')) el[action] = actions[action];
				else if (evt = action.test('([\\w-]{1,})event$')) el.addEvent(evt[1], actions[action]);
			}
		});
	},
	
	filterById: function(id){
		var found = [];
		this.each(function(el){
			if (el.id == id) found.push(el);
		});
		return found;
	},
	
	filterByClassName: function(className){
		var found = [];
		this.each(function(el){
			if (el.hasClassName(className)) found.push(el);
		});
		return found;
	},

	filterByAttribute: function(attribute, value, special){
		var found = [];
		this.each(function(el){
			var att = el.getAttribute(attribute);
			if(!att) return;
			if (!special) return found.push(el);
			
			switch(special){
				case '*=': if (att.test(value)) found.push(el); break;
				case '=': if (att == value) found.push(el); break;
				case '^=': if (att.test('^'+value)) found.push(el); break;
				case '$=': if (att.test(value+'$')) found.push(el);
			}

		});
		return found;
	},

	getElementsByTagName: function(tagName){
		var found = [];
		this.each(function(el){
			found.extend($$(el.getElementsByTagName(tagName)));
		});
		return found;
	}
	
});

new Object.Native(Elements);

//Ajax.js - depends on Moo.js + Native Scripts

var Ajax = ajax = new Class({

	setOptions: function(options){
		this.options = {
			method: 'post',
			postBody: '',
			async: true,
			onComplete: Class.empty,
			update: null,
			evalScripts: false
		};
		Object.extend(this.options, options || {});
	},

	initialize: function(url, options){
		this.setOptions(options);
		this.url = url;
		this.transport = this.getTransport();
	},

	request: function(){
		this.transport.open(this.options.method, this.url, this.options.async);
		this.transport.onreadystatechange = this.onStateChange.bind(this);
		if (this.options.method == 'post'){
			this.transport.setRequestHeader('Content-type', 'application/x-www-form-urlencoded');
			if (this.transport.overrideMimeType) this.transport.setRequestHeader('Connection', 'close');
		}
		switch($type(this.options.postBody)){
			case 'element': this.options.postBody = $(this.options.postBody).toQueryString(); break;
			case 'object': this.options.postBody = Object.toQueryString(this.options.postBody);
		}
		if($type(this.options.postBody) == 'string') this.transport.send(this.options.postBody);
		else this.transport.send();
		return this;
	},

	onStateChange: function(){
		if (this.transport.readyState == 4 && this.transport.status == 200){
			if (this.options.update) this.fillContainer.delay(10, this);
			this.options.onComplete.delay(30, this);
			if (this.options.evaluateScripts) {
				if(scripts = this.transport.responseText.match(/<script[^>]*?>.*?<\/script>/g)){
					scripts.each(function(script){
						eval(script.replace(/^<script[^>]*?>/, '').replace(/<\/script>$/, ''));
					});
				}
			}
			this.transport.onreadystatechange = Class.empty;
			this.callChain();
		}
	},

	fillContainer: function(){
		$(this.options.update).setHTML(this.transport.responseText);
	},

	getTransport: function(){
		if (window.XMLHttpRequest) return new XMLHttpRequest();
		else if (window.ActiveXObject) return new ActiveXObject('Microsoft.XMLHTTP');
	}

});

Ajax.implement(new Chain);

Object.toQueryString = function(source){
	var queryString = [];
	for (property in source) queryString.push(property+'='+source[property]);
	return queryString.join('&');
};

Element.extend({
	
	send: function(options){
		options = Object.extend(options, {postBody: this.toQueryString(), method: this.getAttribute('method')});
		return new Ajax(this.getAttribute('action').toLowerCase(), options).request();
	},

	toQueryString: function(){
		var queryString = [];
		$A(this.getElementsByTagName('*')).each(function(el){
			$(el);
			var name = $check(el.name);
			if (!name) return;
			var value = false;
			switch(el.getTag()){
				case 'select': value = el.getElementsByTagName('option')[el.selectedIndex].value; break;
				case 'input': if ( (el.checked && ['checkbox', 'radio'].test(el.type)) || (['hidden', 'text', 'password'].test(el.type)) ) 
					value = el.value; break;
				case 'textarea': value = el.value;
			}
			if (value) queryString.push(encodeURIComponent(name)+'='+encodeURIComponent(value));
		});
		return queryString.join('&');
	}

});

//DragDrop.js - depends on Moo.js + Native Scripts

var Drag = new Class({
	
	setOptions: function(options){
		this.options = {
			handle: false,
			unit: 'px', 
			onStart: Class.empty, 
			onComplete: Class.empty, 
			onDrag: Class.empty,
			onSnap: Class.empty,
			droppables: [],
			xMax: false,
			xMin: false,
			yMax: false,
			yMin: false,
			xModifier: 'left',
			yModifier: 'top',
			snapDistance: 8,
			snap: true,
			doSnap: true
		};
		Object.extend(this.options, options || {});
	},

	initialize: function(el, options){
		this.setOptions(options);
		this.el = $(el);
		this.options.handle = $(this.options.handle) || el;
		if (this.options.xModifier) this.xp = this.options.xModifier.camelCase();
		if (this.options.yModifier) this.yp = this.options.yModifier.camelCase();
		this.options.handle.onmousedown = this.start.bind(this);
	},

	start: function(evt){
		evt = evt || window.event;
		this.startX = evt.clientX;
		this.startY = evt.clientY;
		this.options.onStart.delay(10, this);
		if (this.options.snap){
			document.onmousemove = this.checkAndDrag.bind(this);
		} else {
			this.set(evt);
			document.onmousemove = this.drag.bind(this);
		}
		document.onmouseup = this.end.bind(this);
		return false;
	},
	
	checkAndDrag: function(evt){
		evt = evt || window.event;
		var distance = Math.round(Math.sqrt(Math.pow(evt.clientX - this.startX, 2)+Math.pow(evt.clientY - this.startY, 2)));
		if (distance > this.options.snapDistance){
			this.set(evt);
			this.options.onSnap.delay(10, this);
			document.onmousemove = this.drag.bind(this);
			if (this.options.doSnap){
				if (this.xp) this.el.style[this.xp] = parseInt(this.el.getStyle(this.xp))-(this.startX-evt.clientX)+this.options.unit;
				if (this.yp) this.el.style[this.yp] = parseInt(this.el.getStyle(this.yp))-(this.startY-evt.clientY)+this.options.unit;
			}
		}
		return false;
	},

	drag: function(evt){
		evt = evt || window.event;
		this.clientX = evt.clientX;
		this.clientY = evt.clientY;
		
		this.options.onDrag.delay(5, this);
		if (this.xp && (!this.options.xMax || this.options.xMax < this.clientX) && (!this.options.xMin || this.options.xMin > x)){
			this.el.style[this.xp] = ((parseInt(this.el.getStyle(this.xp)) || 0)+this.clientX-this.lastMouseX)+this.options.unit;
		}
		if (this.yp && (!this.options.yMax || this.options.yMax < this.clientY) && (!this.options.yMin || this.options.yMin > y)){
			this.el.style[this.yp] = ((parseInt(this.el.getStyle(this.yp)) || 0)+this.clientY-this.lastMouseY)+this.options.unit;
		}
		this.options.droppables.each(function(drop){
			if (this.checkAgainst(drop)){
				if (drop.onOver && !drop.dropping) drop.onOver(this.el);
				drop.dropping = true;
			} else {
				if (drop.onLeave && drop.dropping) drop.onLeave(this.el);
				drop.dropping = false;
			}
		}, this);
		this.set(evt);
		return false;
	},
	
	set: function(evt){
		this.lastMouseX = evt.clientX;
		this.lastMouseY = evt.clientY;
		return false;
	},
	
	checkAgainst: function(drop){
		x = this.clientX+Window.getScrollLeft();
		y = this.clientY+Window.getScrollTop();
		drop = $(drop);
		var h = drop.offsetHeight;
		var w = drop.offsetWidth;
		var t = drop.getTop();
		var l = drop.getLeft();
		if (x > l && x < l+w && y < t+h && y > t) return true;
		else return false;
	},

	end: function(){
		document.onmousemove = null;
		document.onmouseup = null;
		this.options.droppables.each(function(drop){
			if (drop.onDrop && this.checkAgainst(drop)) drop.onDrop(this.el);
		}, this);
		this.options.onComplete.delay(10, this);
	}

});

Element.extend({
	
	makeDraggable: function(options){
		return new Drag(this, options);
	},

	makeResizable: function(options){
		return new Drag(this, Object.extend({xModifier: 'width', yModifier: 'height'}, options));
	}

});

//Window.js : additional Window methods - depends on Moo.js + Function.js

var Window = {
	
	extend: Object.extend,
	
	getWidth: function(){
		return window.innerWidth || document.documentElement.clientWidth || 0;
	},
	
	getHeight: function(){
		return window.innerHeight || document.documentElement.clientHeight || 0;
	},
	
	getScrollHeight: function(){
		return document.documentElement.scrollHeight;
	},
	
	getScrollWidth: function(){
		return document.documentElement.scrollWidth;
	},
	
	getScrollTop: function(){
		return document.documentElement.scrollTop || window.pageYOffset || 0;
	},
	
	getScrollLeft: function(){
		return document.documentElement.scrollLeft || window.pageXOffset || 0;
	},
	
	onLoad: function(fn){
		if(!document.body) return Window.onLoad.bind(fn).delay(50);
		else if ($type(fn) == 'function') return fn();
		else if ($type(this) == 'function') return this();
		else return false;
	}
};

//Cookie.js : Cookie creator. yummy! - depends on Moo.js + Function.js
//Credits: based on the functions by Peter-Paul Koch (http://quirksmode.org)

var Cookie = {

	set: function(key, value, duration){
		var date = new Date();
		date.setTime(date.getTime()+((duration || 365)*86400000));
		document.cookie = key+"="+value+"; expires="+date.toGMTString()+"; path=/";
	},
	
	get: function(key){
		var myValue, myVal;
		document.cookie.split(';').each(function(cookie){
			if(myVal = cookie.trim().test(key+'=(.*)')) myValue = myVal[1];
		});
		return myValue;
	},
	
	remove: function(key){
		this.set(key, '', -1);
	}

};

//Json.js - depends on Moo.js + Native Scripts

var Json = {
	toString: function(el){
		var string = [];
		
		var isArray = function(array){
			var string = [];
			array.each(function(ar){
				string.push(Json.toString(ar));
			});
			return string.join(',');
		};
		
		var isObject = function(object){
			var string = [];
			for (property in object) string.push('"'+property+'":'+Json.toString(object[property]));
			return string.join(',');
		};
		
		switch($type(el)){
			case 'string': string.push('"'+el+'"'); break;
			case 'function': string.push(el); break;
			case 'object': string.push('{'+isObject(el)+'}'); break;
			case 'array': string.push('['+isArray(el)+']');
		}
		
		return string.join(',');
	},

	evaluate: function(str){
		return eval('(' + str + ')');
	}
};

//Sortables.js : Make any list sortable. Depends on Moo.js + Native Scripts + DragDrop.js + Fx.js

var Sortables = new Class({

	setOptions: function(options) {
		this.options = {
			handles: false,
			droppables: [],
			fxDuration: 250,
			fxTransition: Fx.sinoidal,
			maxOpacity: 0.5
		};
		Object.extend(this.options, options || {});
	},
	
	initialize: function(elements, options){
		this.setOptions(options);
		this.options.handles = this.options.handles || elements;
		var trash = $(document.body).adopt('div');
		$A(elements).each(function(el, i){
			var copy = $(el).clone().setStyles({'position': 'absolute', 'opacity': '0', 'display': 'none'}).injectInside(trash);
			var elEffect = el.effect('opacity', {duration: this.options.fxDuration, wait: false, transition: this.options.fxTransition}).set(1);
			var copyEffects = copy.effects({
				duration: this.options.fxDuration,
				wait: false,
				transition: this.options.fxTransition,
				onComplete: function(){
					copy.setStyle('display', 'none');
				}
			});
			var dragger = new Drag(copy, {
				xModifier: false,
				droppables: this.options.droppables,
				onStart: function(){
					copy.setHTML(el.innerHTML).setStyles({
						'display': 'block',
						'opacity': this.options.maxOpacity,
						'top': el.getTop()+'px',
						'left': el.getLeft()+'px'
					});
					elEffect.custom(elEffect.now, this.options.maxOpacity);
				}.bind(this),
				onComplete: function(){
					copyEffects.custom({'opacity': [this.options.maxOpacity, 0], 'top': [copy.getTop(), el.getTop()]});
					elEffect.custom(elEffect.now, 1);
				}.bind(this),
				onDrag: function(){
					if ( el.getPrevious() && copy.getTop() < (el.getPrevious().getTop()) ) el.injectBefore(el.getPrevious());
					else if ( el.getNext() && copy.getTop() > (el.getNext().getTop()) ) el.injectAfter(el.getNext());
				}
			});
			
			this.options.droppables.each(function(drop){
				drop.onDrop = function(copy){
					if (el.parentNode == drop) return;
					var opt = dragger.options.onComplete;
					dragger.options.onComplete = Class.empty;
					elEffect.options.wait = true;
					elEffect.custom(elEffect.now, 0);
					copyEffects.custom({'opacity': [this.options.maxOpacity, 0]});
					elEffect.options.onComplete = function(){
						drop.appendChild(el);
						elEffect.options.onComplete = Class.empty;
						elEffect.custom(elEffect.now, 1);
						elEffect.options.wait = false;
						dragger.options.onComplete = opt;
					};
				}.bind(this);
			}, this);
			this.options.handles[i].onmousedown = dragger.start.bind(dragger);
		}, this);
	}

});

//FxPack.js - depends on Moo.js + Native Scripts + Fx.js

Fx.Styles = Fx.Base.extend({
	
	initialize: function(el, options){
		this.el = $(el);
		this.setOptions(options);
		this.now = {};
	},
	
	setNow: function(){
		for (p in this.from) this.now[p] = this.compute(this.from[p], this.to[p]);
	},
	
	setTransition: function(objFromTo, setStyles){
		this.transition = objFromTo;
		if (setStyles){
			for (p in objFromTo) this.setStyle(this.el, p, objFromTo[p][0]);
		}
	},
	
	custom: function(objFromTo){
		this.previous = objFromTo;
		var from = {};
		var to = {};
		for (p in objFromTo){
			from[p] = objFromTo[p][0];
			to[p] = objFromTo[p][1];
		}
		return this.parent(from, to);
	},
	
	toggle: function(){
		if (!this.previous){
			this.previous = this.transition;
		} else {
			for (p in this.previous){
				var start = this.previous[p][0];
				var end = this.previous[p][1];
				this.previous[p][1] = start;
				this.previous[p][0] = end;
			}
		}
		
		return this.custom(this.previous);
	},
	
	resizeTo: function(hto, wto){
		return this.custom({'height': [this.el.offsetHeight, hto], 'width': [this.el.offsetWidth, wto]});
	},
	
	resizeBy: function(hby, wby){
		return this.custom({'height': [this.el.offsetHeight, this.el.offsetHeight+hby], 'width': [this.el.offsetWidth, this.el.offsetWidth+wby]});
	},
	
	increase: function(){
		for (p in this.now){
			this.setStyle(this.el, p, this.now[p]);
		}
	}
	
});

//fx.Color, originally by Tom Jensen (http://neuemusic.com) MIT-style LICENSE.

Fx.Color = Fx.Base.extend({
	
	initialize: function(el, property, options){
		this.el = $(el);
		this.setOptions(options);
		this.property = property.camelCase();
		this.regexp = /^[#]([\w]{2})([\w]{2})([\w]{2})$/;
	},
	
	custom: function(from, to){
		this.cs = this.regexp.exec(from);
		this.ce = this.regexp.exec(to);
		
		for (i=1; i < this.cs.length; i++){
			this.cs[i-1] = parseInt(this.cs[i], 16);
			this.ce[i-1] = parseInt(this.ce[i], 16);
		}
		return this.parent(0, 1);
	},
	
	rgb: function(i){
		return Math.round(this.cs[i]+(this.ce[i]-this.cs[i])*this.now);
	},
	
	increase: function(){
		this.el.setStyle(this.property, "rgb("+this.rgb(0)+","+this.rgb(1)+","+this.rgb(2)+")");
	}

});

Element.extend({

	effects: function(options){
		return new Fx.Styles(this, options);
	}

});

//Easing Equations (c) 2003 Robert Penner, all rights reserved.
//This work is subject to the terms in http://www.robertpenner.com/easing_terms_of_use.html.

Fx.expoIn = function(pos){return Math.pow(2, 10 * (pos - 1))};
Fx.expoOut = function(pos){return (-Math.pow(2, -10 * pos) + 1)};

Fx.quadIn = function(pos){return Math.pow(pos, 2)};
Fx.quadOut = function(pos){return -(pos)*(pos-2)};

Fx.circOut = function(pos){return Math.sqrt(1 - Math.pow(pos-1,2))};
Fx.circIn = function(pos){return -(Math.sqrt(1 - Math.pow(pos, 2)) - 1)};

Fx.backIn = function(pos){return (pos)*pos*((2.7)*pos - 1.7)};
Fx.backOut = function(pos){return ((pos-1)*(pos-1)*((2.7)*(pos-1) + 1.7) + 1)};

Fx.sineOut = function(pos){return Math.sin(pos * (Math.PI/2))};
Fx.sineIn = function(pos){return -Math.cos(pos * (Math.PI/2)) + 1};
Fx.sineInOut = function(pos){return -(Math.cos(Math.PI*pos) - 1)/2};

//scriptaculous transitions
Fx.wobble = function(pos){return (-Math.cos(pos*Math.PI*(9*pos))/2) + 0.5};
Fx.pulse = function(pos){return (Math.floor(pos*10) % 2 == 0 ? (pos*10-Math.floor(pos*10)) : 1-(pos*10-Math.floor(pos*10)))};

//Tips.js : Display a tip on any element with a title and/or href - depends on Moo.js + Native Scripts +  Fx.js
//Credits : Tips.js is based on Bubble Tooltips (http://web-graphics.com/mtarchive/001717.php) by Alessandro Fulcitiniti (http://web-graphics.com)

var Tips = new Class({

	setOptions: function(options){
		this.options = {
			transitionStart: fx.sinoidal,
			transitionEnd: fx.sinoidal,
			maxTitleChars: 30,
			fxDuration: 150,
			maxOpacity: 1,
			timeOut: 100,
			className: 'tooltip'
		}
		Object.extend(this.options, options || {});
	},

	initialize: function(elements, options){
		this.elements = elements;
		this.setOptions(options);
		this.toolTip = $(document.body).adopt('DIV').addClassName(this.options.className).setStyle('position', 'absolute');
		this.toolTitle = this.toolTip.adopt('H4');
		this.toolText = this.toolTip.adopt('P');
		this.fx = new fx.Style(this.toolTip, 'opacity', {duration: this.options.fxDuration, wait: false}).hide();
		$A(elements).each(function(el){
			$(el).myText = $check(el.title);
			if (el.myText) el.removeAttribute('title');
			if (el.href){
				if (el.href.test('http://')) el.myTitle = el.href.replace('http://', '');
				if (el.href.length > this.options.maxTitleChars) el.myTitle = el.href.substr(0,this.options.maxTitleChars-3)+"...";
			}
			if (el.myText && el.myText.test('::')){
				var dual = el.myText.split('::');
				el.myTitle = dual[0].trim();
				el.myText = dual[1].trim();
			} 
			el.onmouseover = function(){
				this.show(el);
				return false;
			}.bind(this);
			el.onmousemove = this.locate.bindAsEventListener(this);
			el.onmouseout = function(){
				this.timer = $clear(this.timer);
				this.disappear();
			}.bind(this);
		}, this);
	},

	show: function(el){
		this.toolTitle.innerHTML = el.myTitle;
		this.toolText.innerHTML = el.myText;
		this.timer = $clear(this.timer);
		this.fx.options.transition = this.options.transitionStart;
		this.timer = this.appear.delay(this.options.timeOut, this);
	},

	appear: function(){
		this.fx.custom(this.fx.now, this.options.maxOpacity);
	},

	locate: function(evt){
		var doc = document.documentElement;
		this.toolTip.setStyles({'top': evt.clientY + doc.scrollTop + 15 + 'px', 'left': evt.clientX + doc.scrollLeft - 30 + 'px'});
	},

	disappear: function(){
		this.fx.options.transition = this.options.transitionEnd;
		this.fx.custom(this.fx.now, 0);
	}

});

//Accordion.js - depends on Moo.js + Native Scripts + Fx.js

fx.Elements = Fx.Base.extend({
	
	initialize: function(elements, options){
		this.elements = [];
		elements.each(function(el){
			this.elements.push($(el));
		}, this);
		this.setOptions(options);
		this.now = {};
	},

	setNow: function(){
		for (i in this.from){
			var iFrom = this.from[i];
			var iTo = this.to[i];
			var iNow = this.now[i] = {};
			for (p in iFrom) iNow[p] = this.compute(iFrom[p], iTo[p]);
		}
	},

	custom: function(objObjs){
		var from = {};
		var to = {};
		for (i in objObjs){
			var iProps = objObjs[i];
			var iFrom = from[i] = {};
			var iTo = to[i] = {};
			for (prop in iProps){
				iFrom[prop] = iProps[prop][0];
				iTo[prop] = iProps[prop][1];
			}
		}
		return this.parent(from, to);
	},

	increase: function(){
		for (i in this.now){
			var iNow = this.now[i];
			for (p in iNow) this.setStyle(this.elements[parseInt(i)-1], p, iNow[p]);
		}
	}

});

fx.Accordion = fx.Elements.extend({
	
	extendOptions: function(options){
		var newOptions = {
			start: 'open-first',
			fixedHeight: false,
			alwaysHide: false,
			wait: false,
			onActive: Class.empty,
			onBackground: Class.empty,
			height: true,
			opacity: true,
			width: false
		};
		Object.extend(this.options, Object.extend(newOptions, options || {}));
	},

	initialize: function(togglers, elements, options){
		this.parent(elements, options);
		this.extendOptions(options);
		this.previousClick = 'nan';
		togglers.each(function(tog, i){
			$(tog).addEvent('click', function(){this.showThisHideOpen(i)}.bind(this));
		}, this);
		this.togglers = togglers;
		this.h = {}; this.w = {}; this.o = {};
		this.elements.each(function(el, i){
			this.now[i+1] = {};
			$(el).setStyle('height', 0);
		}, this);
		switch(this.options.start){
			case 'first-open': this.elements[0].setStyle('height', this.elements[0].scrollHeight); break;
			case 'open-first': this.showThisHideOpen(0); break;
		}
	},

	hideThis: function(i){
		if (this.options.height) this.h = {'height': [this.elements[i].offsetHeight, 0]};
		if (this.options.width) this.w = {'height': [this.elements[i].offsetWidth, 0]};
		if (this.options.opacity) this.o = {'opacity': [this.now[i+1]['opacity'] || 1, 0]};
	},

	showThis: function(i){
		if (this.options.height) this.h = {'height': [this.elements[i].offsetHeight, this.options.fixedHeight || this.elements[i].scrollHeight]};
		if (this.options.width) this.w = {'height': [this.elements[i].offsetWidth, this.elements[i].scrollWidth]};
		if (this.options.opacity) this.o = {'opacity': [this.now[i+1]['opacity'] || 0, 1]};
	},

	showThisHideOpen: function(iToShow){
		if (iToShow != this.previousClick || this.options.alwaysHide){
			this.previousClick = iToShow;
			var objObjs = {};
			var err = false;
			var madeInactive = false;
			this.elements.each(function(el, i){
				this.now[i] = this.now[i] || {};
				if (i != iToShow){
					this.hideThis(i);
				} else if (this.options.alwaysHide){
					if (el.offsetHeight == el.scrollHeight){
						this.hideThis(i);
						madeInactive = true;
					} else if (el.offsetHeight == 0){
						this.showThis(i);
					} else {
						err = true;
					}
				} else if (this.options.wait && this.timer){
					this.previousClick = 'nan';
					err = true;
				} else {
					this.showThis(i);
				}
				objObjs[i+1] = Object.extend(this.h, Object.extend(this.o, this.w));
			}, this);
			if (err) return;
			if (!madeInactive) this.options.onActive(this.togglers[iToShow]);
			this.togglers.each(function(tog, i){
				if (i != iToShow || madeInactive) this.options.onBackground(tog);
			}, this);
			return this.custom(objObjs);
		}
	}
	
});