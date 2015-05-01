/**
 * root namespace
 */
if (typeof Begun == "undefined" || !Begun) var Begun = {};

Begun.extend = function(destination, source){
	for (var property in source){
		destination[property] = source[property];
	}
	return destination;
};

/**
 * helper
 */
Begun.$ = function(id){
	return document.getElementById(id);
};

/**
 * constants
 */
Begun.DOM_TIMEOUT = 1;

/**
 * browsers
 */
Begun.Browser = new function(){
    this.IE = !!(window.attachEvent && navigator.userAgent.indexOf('Opera') === -1);
    this.Opera =  navigator.userAgent.indexOf('Opera') > -1;
    this.WebKit = navigator.userAgent.indexOf('AppleWebKit/') > -1;
    this.Gecko =  navigator.userAgent.indexOf('Gecko') > -1 && navigator.userAgent.indexOf('KHTML') === -1;

    var ver = null;
    
    this.version = function(){
    	if (ver !== null){
    		return ver;
    	}
    	return ver = detect();
    };
    
    var detect;
    
    function check(re){
    	return re.exec(navigator.userAgent)[1];
    }
    
    if (this.IE){
    	detect = function(){ return check(/MSIE (\d*.\d*)/gim); };
    } else if (this.Opera){
    	detect = function(){ return check(/Opera\/(\d*\.\d*)/gim); };
    } else if (this.WebKit){
    	detect = function(){ return check(/AppleWebKit\/(\d*\.\d*)/gim); };
    } else if (this.Gecko){
    	detect = function(){ return check(/Firefox\/(\d*\.\d*)/gim); };
    }

    this.isOlder = function(n){
    	return parseFloat(this.version()) > n;
    };
    this.isYounger = function(n){
    	return parseFloat(this.version()) < n;
    };
}();

/**
 * utils
 */
Begun.Utils = new function(){
	this.scriptCounter = 0;
	this.includeScript = function(url, type, callback, callback_name){
		var type = type || 'write'; // append or write
		var inc = 0;
		var script = null;
		if (url){
			if (type == 'write'){ // callback doesn't allow further document.writes in ie :-(
				this.scriptCounter++;
				var id = Begun.Autocontext.Strings.js.inc_script_prefix + this.scriptCounter;
				document.write('<scr'+'ipt type="text/javascript" src="' + url + '" id="' + id + '"></scr'+'ipt>'); 
				script = Begun.$(id);
			}else if (type == 'append'){
				script = document.createElement("script");
				script.src = url;
				script.type = "text/javascript";
				var head = document.getElementsByTagName("head")[0];
				head.appendChild(script);
			}
			if (script && typeof callback == 'function'){
				var callback_fired = false;
				script.onload = function(){
					if (!callback_fired){
						callback();
						callback_fired = true;
					}
				};
				script.onreadystatechange = function(){
					if (callback_fired){
						return;
					}
					var check_statement = (Begun.Browser.Opera ? (this.readyState == "complete") : (this.readyState == "complete" || this.readyState == "loaded"));
					if (check_statement){
						callback();
						callback_fired = true;
					}
				};
			}
		}
	};
	this.includeStyle = function(css_text, type, id){
		var type = type || 'write'; // append or write
		var DEFUALT_STYLE_ID = 'begun-default-css';
		id = id || DEFUALT_STYLE_ID;
		if (css_text){
			if (type == 'write'){
				document.write('<style type="text/css" id="' + id + '">' + css_text + '</style>');
			}else if (type == 'append'){
				// IE
				if (document.createStyleSheet){
					var style = null;
					try{
						style = document.createStyleSheet();
						style.cssText += css_text;
					}catch(e){
						for (var i = document.styleSheets.length - 1; i >= 0; i--){
							try{
								style = document.styleSheets[i];
								style.cssText += css_text;
								break; // access granted? get outta here!
							}catch(e){
								style = null;
							}
						}
					}
				}else{
					if (Begun.$(id)){
						Begun.$(id).innerHTML = css_text;
					}else{
						var style = document.createElement("style");
						style.setAttribute("type", "text/css");
						style.id = id;
						style.appendChild(document.createTextNode(css_text));
						getHead().appendChild(style);
					}
				}
			}
		}
	};
	this.toQuery = function(params){
		var result = '';
		for (var key in params)
			if (params[key] && params.hasOwnProperty && params.hasOwnProperty(key))
				result += '&' + key + '=' + escape(params[key]);
		return result;
	};

	function getHead(){
		var head = document.getElementsByTagName("head")[0];
		if (!head){
			head = document.createElement("head");
			document.documentElement.insertBefore(head, document.documentElement.firstChild);
		}
		return head;
	}
}();

/**
 * in_array extension
 */
if (typeof Array.prototype.in_array == 'undefined') Array.prototype.in_array = function(value){
	for (var i = 0, l = this.length; i < l; i++)
		if (this[i] == value) return true;
	return false;
}

/**
 * template engine
 */
Begun.Template = function(tpl){
	var tpl = tpl || '';
	this.getTpl = function(){
		return tpl;
	};
	this.evaluate = function(vars){
		for (var key in vars) {
			var re = new RegExp('\\{\\{' + key + '\\}\\}', 'g');
			tpl = tpl.replace(re, vars[key]);
		}
		// remove unused placeholders
		var re = new RegExp('\\{\\{.+?\\}\\}', 'g');
		tpl = tpl.replace(re, '-foo');

		return tpl;
	};
};

if (!Begun.Autocontext){
	/**
	 * autocontext singleton
	 */
	Begun.Autocontext = new function(){
		var _this = this;
		this.feed_done = false;
		this.dom_change = false;
		this.banner_index = 0;
		this.banners = {};
		this.stubs = {};
		this.scrollers = [];
		this.requestParams = {};
		this.responseParams = {};
		this.options = {
			max_banners_count: Infinity,
			max_blocks_count: Infinity,
			scroll_banners_coef: 1,
			max_scrollers: 1
		};
		this.setOptions = function(options){
			Begun.extend(_this.options, options || {});
		};
		this.getFrameLevel = function(){
			var level = 0;
			var _parent = self;
			while (_parent !== top && level < 999){
				_parent = _parent.parent;
				level++;
			}
			return level;
		};
		this.defaultRequestParams = {
			"pad_id": window.begun_auto_pad || -1,
			"n": 0,
			"lmt": Date.parse(document.lastModified) / 1000,
			"sense_mode": 'custom', // wtf?
			"ut_screen_width": screen.width || 0,
			"ut_screen_height": screen.height || 0,
			"json": 1, // for json feed!
			"jscall": 'loadFeedDone', // called after feed load, in the Begun.Autocontext namespace
			"stopwords": window.stopwords || '',
			"begun_self_keywords": window.begun_self_keywords || '',
			"ref": document.referrer,
			"real_refer": document.location
		};
		this.prepareDaemonParams = function(extra_params){
			_this.defaultRequestParams.n = _this.options.max_banners_count;
			var frame_level = _this.getFrameLevel();
			if (frame_level){
				_this.defaultRequestParams.frm_level = frame_level;
				try{
					_this.defaultRequestParams.frm_top = top.location.href;
				}catch(e){
					_this.defaultRequestParams.frm_top = 'top not accessible';
				}
			}
			Begun.extend(_this.requestParams, _this.defaultRequestParams || {});
			Begun.extend(_this.requestParams, extra_params || {});
		};
		this.includePpcall = function(force_load){
			if (arguments.callee.loaded){
				return;
			}
			var isPpcallNeeded = function(){
				var is_ppcall = false;
				var banner = null;
				var i = 0;
				while (banner = _this.getBanner('autocontext', i++)){
					if (banner['ppcall']){
						is_ppcall = true;
						break;
					}
				}
				return is_ppcall;
			};
			if (isPpcallNeeded() || force_load){
				Begun.Utils.includeScript(
					Begun.Autocontext.Strings.urls.ppcall_js,
					'append'
				);
				arguments.callee.loaded = true;
			}
		};
		this.initScrollBlocks = function(block){
			if (Begun.Scroller){
				var	setHeight = function(block, scroll_div, scroll_table){
					var banners_count = Number(block.options.banners_count);
					var height = 0;
					for (var i = 0; i < banners_count; i++){
						if (scroll_table.getElementsByTagName('tr')[i]){
							var h = scroll_table.getElementsByTagName('tr')[i].offsetHeight;
							height += h;
						}
					}
					scroll_div.style.height = (height + 1) + 'px'; // a required slight increase for correct scrolling transition
					scroll_div.style.overflow = 'hidden';
				};
				var	init = function(block, scroll_div, scroll_table){
					var banners_count = Number(block.options.banners_count);
					var scroller = (new Begun.Scroller(
						scroll_table,
						{
							height: scroll_div.offsetHeight,
							banners_count: banners_count
						}
					));
					if (scroll_table.clientHeight >= scroll_div.clientHeight){
						if (block.options.dimensions.type != 'vertical'){
							scroller.banners_count--;
						}
						_this.scrollers.push(scroller);
						scroller.start();
					}
				};
			
				var block = null;
				var i = 0;
				while (block = _this.Blocks[i]){
					var scroll_div = Begun.$(Begun.Autocontext.Strings.css.scroll_div_prefix + block.id);
					var scroll_table = Begun.$(Begun.Autocontext.Strings.css.scroll_table_prefix + block.id);
					if (!block.scrolling && Number(block.options.use_scroll) && scroll_div && scroll_table){
						if (block.options.dimensions.type == 'vertical'){
							(function(block, scroll_div, scroll_table){
								if (scroll_table.offsetHeight){
									setHeight(block, scroll_div, scroll_table);
									init(block, scroll_div, scroll_table);
								}else{
									var func = arguments.callee;
									window.setTimeout(function(){
										func(block, scroll_div, scroll_table);
									}, Begun.DOM_TIMEOUT);
								}
							})(block, scroll_div, scroll_table);
						}else{
							init(block, scroll_div, scroll_table);
						}
						block.scrolling = true;
					}
					i++;
				}
			}
		};
		this.includeScroll = function(force_load){
			if (arguments.callee.loaded){
				return;
			}
			var isScrollNeeded = function(){
				var is_scroll = false;
				var block = null;
				var i = 0;
				while (block = _this.Blocks[i++]){
					if (Number(block.options.use_scroll)){
						is_scroll = true;
						break;
					}
				}
				return is_scroll;
			};
			if (isScrollNeeded() || force_load){
				Begun.Utils.includeScript(
					Begun.Autocontext.Strings.urls.scroll_js,
					'append'
				);
				arguments.callee.loaded = true;
			}
		};
		this.handleHypercontext = function(){
			if (Number(_this.responseParams['begun_auto_hyper'])){
				Begun.Utils.includeScript(
					Begun.Autocontext.Strings.urls.hypercontext_js,
					'append',
					function(){
						hyperRun();
					}
				);
			}
		};
		this.handleLinks = function(){
			if (arguments.callee.loaded){
				return;
			}
			if (_this.links){
				var i = 0;
				var link = null;
				while (link = _this.links[i]){
					switch(link.type){
						case 'js':
							Begun.Utils.includeScript(
								link.url,
								'append'
							);
							break; 
						case 'img':
							(new Image()).src = link.url;
							break;
						case 'frame':
							var vars = {url: link.url};
							document.write((new Begun.Template(_this.Tpls.getHTML('link_iframe'))).evaluate(vars));
							break;
						break;
						default:
							//
					}
					i++;
				}
				arguments.callee.loaded = true;
			}
		};
		this.resetMaxScrollers = function(){
			_this.maxScrollers = _this.options.max_scrollers;
		};
		this.resetBannerIndex = function(){
			//_this.banner_index = window.begun_auto_limit || 0; // if the old code is present & there is no extra block, then we will go from begun_auto_limit + 1
			_this.banner_index = 0;
		};
		this.loadExtraResources = function(force_load){
			//_this.handleHypercontext(); // not ready
			_this.handleLinks();
			_this.includePpcall(force_load);
			_this.includeScroll(force_load);
		}
		this.draw = function(reset_banners, force_load){
			// 4 wizard && initial start
			if (reset_banners){
				_this.resetBannerIndex();
				_this.resetMaxScrollers();
			}
			_this.loadExtraResources(force_load);
			_this.fillBlocks();
		};
		this.noMoreBlocksNeeded = function(){
			var result = false;
			var block = null;
			var i = 0;
			var count = 0;
			while (block = _this.Blocks[i++]){
				count += _this.getActualBlockBannersCount(block);
				if (count > _this.options.max_banners_count){
					result = true;
					break;
				}
			}
			return result;
		};
		this.useBlockIdDistr = function(){
			return !!(_this.getBanner('autocontext', 0) && _this.getBanner('autocontext', 0)["block_id"]);
		};
		this.initCurrentBlock = function(){
			if ((_this.isFirstBlockWithBanners() && _this.totalBannersUnlimited()) || (_this.noMoreBlocksNeeded() && !_this.useBlockIdDistr())){
				return; // don't waste time and effort, we have enough blocks and banners to show
			}else if (typeof window.begun_auto_pad != 'undefined' && window.begun_auto_pad > 0){
				if (!_this.totalBannersUnlimited() && !_this.feed_done){ // banners are limited, load the feed!
					_this.feed_done = true;
					_this.initFeedLoad();
				}
				if (typeof window.begun_block_id != 'undefined' && window.begun_block_id > 0){
					_this.printBlockPlace(window.begun_block_id);
					if (typeof window.begun_extra_block != 'undefined' && window.begun_extra_block != null){
						_this.Blocks.push(window.begun_extra_block);
					}else{
						Begun.Utils.includeScript(
							(new Begun.Template(Begun.Autocontext.Strings.urls.block_js)).evaluate({'pad_id': window.begun_auto_pad, 'block_id': window.begun_block_id}),
							'write' // only write!!
						);
					}
				}
			}
		};
		this.getActualBlockBannersCount = function(block){
			if (!block) return 0;
			return Number(block.options.banners_count) * (Number(block.options.use_scroll) ? _this.options.scroll_banners_coef : 1);
		};
		this.isFirstBlockWithBanners = function(){
			return !!(typeof _this.Blocks[0] != "undefined" && _this.Blocks[0].options && _this.Blocks[0].options.banners_count);
		};
		this.totalBannersUnlimited = function(){
			return !window.begun_total_banners;
		};
		this.initFeedLoad = function(){
			var first_block_banners_count = 0;
			if (_this.isFirstBlockWithBanners()){
				first_block_banners_count = _this.getActualBlockBannersCount(_this.Blocks[0]);
			}
			var total_banners = window.begun_total_banners || first_block_banners_count || _this.options.max_banners_count || 0;
			if (total_banners){
				_this.setOptions({max_banners_count: total_banners});
				_this.loadFeed();
			}
		};
		this.getGraphHTML = function(graph_banner, callback){
			var type = 'img';
			if (("swf" == graph_banner.mime) || ("application/x-shockwave-flash" == graph_banner.mime)){
				type = 'swf';
			}else if (("js" == graph_banner.mime) || ("application/x-javascript" == graph_banner.mime)){
				type = 'js';
				Begun.Utils.includeScript(graph_banner.source, 'append', callback || null);
			}
			var vars = {'url': graph_banner.url, 'source': graph_banner.source};
			return (new Begun.Template(_this.Tpls.getHTML('search_banner_' + type))).evaluate(vars);
		};
		this.insertNonTextBlock = function(block_id){
			var findBlock = function(block_id, type){
				var i = 0;
				var banner = null;
				while (banner = _this.getBanner(type, i)){
					if (banner.block_id == block_id){
						return banner;
					}
					i++;
				}
				return null;
			}
			
			var code_banner = findBlock(block_id, 'code');
			var graph_banner = findBlock(block_id, 'graph');
			if (code_banner){
				Begun.Utils.includeScript(code_banner.source, 'write');
				_this.Blocks.del(block_id);
			}else if (graph_banner){
				var html = _this.getGraphHTML(graph_banner);
				var block_place = Begun.$(Begun.Autocontext.Strings.css.block_prefix + block_id);
				if (html && block_place){
					block_place.innerHTML = html;
					_this.Blocks.del(block_id);
				}
			}
		};
		this.initLoadedParams = function(){
			_this.banners = window.begunAds.banners || {};
			_this.stubs = window.begunAds.stubs || {};
			_this.links = window.begunAds.links || {};
			Begun.extend(_this.responseParams, window.begunAds.params || {});
		};
		// window.begunAds has just become available
		this.loadFeedDone = function(){
			_this.initLoadedParams();
			var first_block = _this.Blocks[0] || null;
			if (first_block && first_block.id){
				_this.insertNonTextBlock(first_block.id);
			}
			_this.draw(true); // run in any case
		};
		this.loadFeed = function(){
			_this.prepareDaemonParams();
			Begun.Utils.includeScript(
				(Begun.Autocontext.Strings.urls.daemon + Begun.Utils.toQuery(_this.requestParams)).substring(0, 1524).replace(/%[0-9a-fA-F]?$/, ''),
				'write' // only write!!
				// callback doesn't work properly in IE, so we put it to the feed
			);
		};
		this.printBlockPlace = function(block_id){
			var vars = {id: Begun.Autocontext.Strings.css.block_prefix + block_id};
			document.write((new Begun.Template(_this.Tpls.getHTML('blck_place'))).evaluate(vars));
		};
		this.printDefaultStyle = function(){
			Begun.Utils.includeStyle(_this.Tpls.getCSS('default'), 'write'); // 'append' is killing ie 6
		};
		this.getBlockLogoColor = function(styles){
			var default_logo_color = Begun.Autocontext.Strings.css.logo_color;
			if (styles.block){
				var is_default_color = ((!styles.block.borderColor) || (styles.block.backgroundColor == styles.block.borderColor));
				return is_default_color ? default_logo_color : styles.block.borderColor;
			}else{
				return default_logo_color;
			}
		};

		function checkColorDef(obj, prop){

			var undef; 
			
			if (obj !== null && obj !== undef){
				if (obj[prop] === ""){
					obj[prop] = "transparent";
				}
			}
		}
		
		function checkBgColor(obj){
			checkColorDef(obj, "backgroundColor");
		}
		
		function checkBorderColor(obj){
			checkColorDef(obj, "borderColor");
		}
		
		function prepareStyles(styles){

			checkBgColor(styles.block_hover);
			checkBorderColor(styles.block_hover);
			
			checkBgColor(styles.block);
			
			if (styles.block){
				if (Begun.Browser.IE && Begun.Browser.isYounger(7)){
					if (styles.block.borderColor == "" || styles.block.borderColor == "transparent"){
						styles.block.borderColor = "pink";
						styles.block.filter = "chroma(color=pink)";
					}
				} else if (styles.block.borderColor == ""){
					styles.block.borderColor = "transparent";
				}
			}
		}
		
		this.printBlockStyle = function(block_id, styles){
			var styles = styles || {};
			var vars = {};
			vars.block_id = block_id || -1;
			vars.phone_margin_top = 1; // TODO: calc from font-size
			vars.block_logo_color = _this.getBlockLogoColor(styles);
			prepareStyles(styles);
			for (var key in styles){
				if (styles[key] && styles.hasOwnProperty && styles.hasOwnProperty(key)){
					for (var key2 in styles[key]){
						if (styles[key][key2] && styles[key].hasOwnProperty && styles[key].hasOwnProperty(key2)){
							vars[key + ':' + key2] = typeof styles[key][key2] == 'number' ? styles[key][key2] + 'px' : styles[key][key2];
						}
					}
				}
			}
			var css_text = (new Begun.Template(_this.Tpls.getCSS('block'))).evaluate(vars);
			Begun.Utils.includeStyle(css_text, 'append', 'begun-block-css-' + block_id); // non-IE with append!!
		};
		this.getBanner = function(type, index){
			try{
				return _this.banners[type][index];
			} catch(e){
				return null;
			}
		};
		this.getStub = function(type){
			return _this.stubs[type] || null;
		};
		this.getThumbSrc = function(banner){
			var src = _this.responseParams['thumbs_src'] ? 'http://' + _this.responseParams['thumbs_src'] + '/' : Begun.Autocontext.Strings.urls.thumbs;
			var banner_id = banner.banner_id + '';
			if (banner_id && banner_id.length > 3){
				var thematic = banner.thematics ? (banner.thematics.split(',')[0] + '') : '1';
				src += banner_id.charAt(banner_id.length - 2);
				src += '/' + banner_id.charAt(banner_id.length - 1);
				src += '/' + banner_id + '.jpg';
				src += '?t=' + thematic + '&r=' + banner_id.charAt(banner_id.length - 3);
			}else{
				src = src + 'empty.jpg';
			}
			return src;
		};
		this.getBannerCardPPcallHTML = function(banner, block){
			var banner_contacts = [];
			var cards_mode = banner['cards_mode'];
			var is_ppcall = banner['ppcall'];
			var vars = {};
			vars.onclick = Begun.Autocontext.Strings.js.banner_onclick;
			function _card(use_phone){
				vars.card_text = Begun.Autocontext.Strings.contacts.card;
				vars.url = _this.addMisc2URL(block, banner.card);
				vars.phone = use_phone ? (new Begun.Template(_this.Tpls.getHTML('bnnr_phone'))).evaluate(vars) : '';
				banner_contacts.push((new Begun.Template(_this.Tpls.getHTML('bnnr_card'))).evaluate(vars));
			}
			function _ppcall(use_phone){
				vars.ppcall_text = Begun.Autocontext.Strings.contacts.ppcall;
				vars.banner_index = _this.banner_index;
				vars.phone = use_phone ? (new Begun.Template(_this.Tpls.getHTML('bnnr_phone'))).evaluate(vars) : '';
				banner_contacts.push((new Begun.Template(_this.Tpls.getHTML('bnnr_ppcall'))).evaluate(vars));
			}
			if (cards_mode == 'Card' && is_ppcall == false){
				_card(true);
			}else if (cards_mode == 'Card' && is_ppcall == true){
				_ppcall(true);
				_card(false);
			}else if (cards_mode == 'Url' && is_ppcall == false){
				//
			}else if (cards_mode == 'Url' && is_ppcall == true){
				_ppcall(true);
			}else if (cards_mode == 'Card, Url' && is_ppcall == false){
				_card(true);
			}else if (cards_mode == 'Card, Url' && is_ppcall == true){
				_ppcall(true);
				_card(false);
			}
			return banner_contacts;
		};
		this.getBannerDomainGeoHTML = function(banner, block){
			var banner_contacts = [];
			var banner_contacts_names = ['domain', 'geo'];
			var i = 0;
			var banner_contacts_name = null;
			var vars = {};
			while (banner_contacts_name = banner_contacts_names[i]){
				vars[banner_contacts_name] = banner[banner_contacts_name];
				vars.url = _this.addMisc2URL(block, banner.url);
				if (vars[banner_contacts_name]) banner_contacts.push((new Begun.Template(_this.Tpls.getHTML('bnnr_' + banner_contacts_name))).evaluate(vars));
				i++;
			}
			return banner_contacts;
		};
		this.addMisc2URL = function(block, url){
			return (block.options.misc_id > 0 ? url + '&misc2=' + (Number(block.options.misc_id) << 8) : url);
		};
		this.clickBanner = function(event_obj){
			_this.Callbacks.dispatch('banner', 'click', event_obj);
			return true;
		};
		this.getBannerHTML = function(banner, block, block_banner_count){
			function prepareBannerMode(banner){
				var banner = banner || {};
				var possible_cards_modes = ['Card, Url', 'Card', 'Url'];
				if ((!banner['cards_mode']) || !possible_cards_modes.in_array(banner['cards_mode'])){
					banner['cards_mode'] = 'Card, Url';
				}
				if (!banner['url'] && !banner['card']){
					return {};
				}
				if (!banner['url'] && banner['card']){
					banner['cards_mode'] = 'Card';
				}
				if (banner['url'] && !banner['card']){
					banner['cards_mode'] = 'Url';
				}
				if (banner['cards_mode'] == 'Card'){
					banner['url'] = banner['card'];
				}
				return banner;
			}

			var banner = prepareBannerMode(banner);
			if (banner){
				var banner_contacts = _this.getBannerCardPPcallHTML(banner, block).concat(_this.getBannerDomainGeoHTML(banner, block)); // get all the contact stuff
				var vars = {};
				Begun.extend(vars, banner);
				vars.contact = banner_contacts.join(_this.Tpls.getHTML('bnnr_glue'));
				vars.url = _this.addMisc2URL(block, banner.url);
				vars.onclick = Begun.Autocontext.Strings.js.banner_onclick;
				vars.block_id = block.id;
				vars.id = block_banner_count || 0;
				vars.thumb = Number(block.options.show_thumbnails) ? (new Begun.Template(_this.Tpls.getHTML('bnnr_thumb'))).evaluate({url: banner.url, src: _this.getThumbSrc(banner)}) : '';
				vars.banner_width = Math.round(100 / Number(block.options.banners_count)) + '%';
				return (new Begun.Template(_this.Tpls.getHTML('banner_' + block.options.dimensions.type))).evaluate(vars);
			}else{
				return '';
			}
		};
		this.getBlockHTML = function(banners_html_arr, block){
			if (!banners_html_arr) return '';
			var banners_html = banners_html_arr.join('');
			var stub_display = Number(_this.responseParams['stub']) ? '' : 'none';
			// TODO: how many stubs and which ones
			/*function isBannersEnough(){
				//
			}*/
			var vars = {};
			var block_hover_html = '';
			var block_opts = block.options.visual || {};
			if (block_opts.block && block_opts.block_hover && block_opts.block_hover.backgroundColor && block_opts.block_hover.borderColor){
				vars.bgcolor_over = block_opts.block_hover.backgroundColor;
				vars.brdcolor_over = block_opts.block_hover.borderColor;
				vars.bgcolor_out = block_opts.block.backgroundColor || 'transparent';
				vars.brdcolor_out = block_opts.block.borderColor || 'transparent';
				vars.block_id = block.id;
				block_hover_html = (new Begun.Template(_this.Tpls.getHTML('blck_hover'))).evaluate(vars);
			}
			vars = {
				block_hover: block_hover_html,
				banners: banners_html,
				banners_count:	banners_html_arr.length,
				//begun_url_colspan: 3,
				scroll_div_id: Begun.Autocontext.Strings.css.scroll_div_prefix + block.id,
				scroll_table_id: Begun.Autocontext.Strings.css.scroll_table_prefix + block.id,
				block_width: Number(block.options.dimensions.width) ? Number(block.options.dimensions.width) + 'px' : '',
				block_scroll_class: Number(block.options.use_scroll) ? Begun.Autocontext.Strings.css.scroll : '',
				begun_url: Begun.Autocontext.Strings.urls.begun,
				become_partner_display: stub_display,
				become_partner_url: _this.getStub('become_partner'),
				become_partner_text: Begun.Autocontext.Strings.stubs.become_partner,
				place_here_display: stub_display,
				place_here_url: _this.getStub('place_here'),
				place_here_text: Begun.Autocontext.Strings.stubs.place_here,
				all_banners_display: stub_display,
				all_banners_url: _this.getStub('all_banners'),
				all_banners_text: Begun.Autocontext.Strings.stubs.all_banners,
				css_thumbnails: Number(block.options.show_thumbnails) ? Begun.Autocontext.Strings.css.thumb : ''
			};
			return (new Begun.Template(_this.Tpls.getHTML('block_' + block.options.dimensions.type))).evaluate(vars);
		};
		this.printBlock = function(banners_html, block){
			if (banners_html.length){
				var elem = Begun.$(Begun.Autocontext.Strings.css.block_prefix + block.id);
				// might be also null in bad html (all browsers are affected)
				if (!elem){
					return false;
				}
				_this.dom_change = true;
				var html = _this.getBlockHTML(banners_html, block);
				
				var show;
				
				if (Begun.Browser.IE){
					show = function(elem, html){
						var n = elem.cloneNode(true);
						n.innerHTML = html;
						elem.parentNode.insertBefore(n);
						elem.parentNode.removeChild(elem);
						_this.dom_change = false;
					}
				} else {
					show = function(elem, html){
						elem.innerHTML = html;
						_this.dom_change = false;
					}
				}
				// fix most common partner mistakes for ie
				if (Begun.Browser.IE){
					var appendTableCell = function(tr, elem){
						if (tr.offsetHeight){
							var td = document.createElement('td');
							tr.appendChild(td);
							td.innerHTML = elem.outerHTML;
							show(td.firstChild, html);
							elem.parentNode.removeChild(elem);
						}else{
							var func = arguments.callee;
							window.setTimeout(function(){
								func(tr, elem);
							}, Begun.DOM_TIMEOUT);
						}
					}
					var parent = null;
					if ((parent = elem.parentNode) && (parent.tagName) && (['ol', 'ul', 'li'].in_array(parent.tagName.toLowerCase()))){
						window.setTimeout(function(){
							var parent2 = parent.parentNode;
							parent2.insertBefore(elem, parent);
							show(elem, html);
						}, Begun.DOM_TIMEOUT);
					}else if ((parent) && (parent = elem.parentNode.parentNode) && (parent.tagName)){
						try{
							show(elem, html);
						}catch(e){
							switch (parent.tagName.toLowerCase()){
								case 'table':
									var tr = document.createElement('tr');
									window.setTimeout(function(){
										parent.lastChild.appendChild(tr); // append to (implied) tbody
										appendTableCell(tr, elem);
									}, Begun.DOM_TIMEOUT);
									break;
								case 'tr':
									window.setTimeout(function(){
										appendTableCell(parent, elem);
									}, Begun.DOM_TIMEOUT);
									break;
								case 'thead':
								case 'tbody':
								case 'tfoot':
									var tr = document.createElement('tr');
									window.setTimeout(function(){
										parent.appendChild(tr); // append to thead/tbody/tfoot
										appendTableCell(tr, elem);
									}, Begun.DOM_TIMEOUT);
									break;
								default:
									_this.dom_change = false;
							}
						}
					}else{
						try{
							show(elem, html);
						}catch(e){
							_this.dom_change = false;
						}
					}
				}else{
					show(elem, html);
				}
				return true;
			}else{
				return false;
			}
		};
		this.hideBlock = function(block_id){
			var elem = Begun.$(Begun.Autocontext.Strings.css.block_prefix + block_id);
			if (elem){
				elem.innerHTML = '';
			}
		};
		this.fillBlocks = function(){
			var block = null;
			var block_index = 0;
			var out_of_banners = false;

			// loaded blocks only!
			while ((block = _this.Blocks[block_index]) && (!out_of_banners)){
				if (block.loaded){
					block_index++;
					continue;
				}
				var banners_html = [];
				var block_banner_count = 0;
				var banner = null;
				if (Number(block.options.use_scroll) && 0 < _this.maxScrollers--){// with scrolling, all the banners in the feed
					while (banner = _this.getBanner('autocontext', block_banner_count++)){
						var banner_html = _this.getBannerHTML(banner, block, (block_banner_count + 1));
						if (banner_html){
							banners_html.push(banner_html);
						}
					}
				}else{// without scrolling, just the needed amount
					if (_this.useBlockIdDistr()){
						var i = 0;
						while (banner = _this.getBanner('autocontext', i)){
							if (banner.block_id && banner.block_id == block.id){
								var banner_html = _this.getBannerHTML(banner, block, (block_banner_count + 1));
								if (banner_html){
									banners_html.push(banner_html);
									_this.Callbacks.dispatch('banner', 'draw', _this);
								}
								block_banner_count++;
							}
							i++;
						}
					}else{
						var banners_count = block.options.banners_count;
						while ((block_banner_count < banners_count) && (_this.banner_index < _this.options.max_banners_count)){
							banner = _this.getBanner('autocontext', _this.banner_index) || null;
							if (banner){
								var banner_html = _this.getBannerHTML(banner, block, (block_banner_count + 1));
								if (banner_html){
									banners_html.push(banner_html);
									_this.Callbacks.dispatch('banner', 'draw', _this);
								}
							}else{
								out_of_banners = true;
								break;
							}
							block_banner_count++;
							_this.banner_index++;
						}
					}
				}
				if (_this.printBlock(banners_html, block)){
					block.loaded = true;
				}
				block_index++;
			}
		};

		function defaultHover(obj, styles){
			obj.style.backgroundColor = styles.backgroundColor;
			obj.style.borderColor = styles.borderColor;
		}
		
		if (Begun.Browser.IE && Begun.Browser.isYounger(7)){
			this.blockHover = function(obj, styles){
				obj.style.borderColor = "";
				if (styles.borderColor == "transparent"){
					obj.style.filter = "chroma(color=pink)";
					obj.style.borderColor = "pink";
				} else {
					defaultHover(obj, styles);
				}
			};
		} else {
			this.blockHover = defaultHover;
		}
		
		this.nullGlobalBlockParams = function(){
			window.begun_block_id = null;
			window.begun_extra_block = null;
		};
		this.setExtraBlockResponseParams = function(block){
			block.options.use_scroll = Number(_this.responseParams['autoscroll']);
			block.options.show_thumbnails = Number(_this.responseParams['thumbs']);
			// multispan = 0 --> old AC
		};
	};

	/**
	 * autocontext strings
	 */
	Begun.Autocontext.Strings = {
		urls: {
			begun: 'http://www.begun.ru/',
			autocontext: 'http://autocontext.begun.ru/',
			daemon: 'http://autocontext.begun.ru/context.jsp?', // a totally new file
			thumbs: 'http://thumbs.begun.ru/',
			block_js: 'http://autocontext.begun.ru/blocks/{{pad_id}}.{{block_id}}.js', // a totally new file
			hypercontext_js: 'http://autocontext.begun.ru/hypertext_a.js', // a totally old file
			ppcall_js: 'http://ppcall.begun.ru/auto_ppcall.js', // a slightly modified old file
			scroll_js: 'http://autocontext.begun.ru/auto_scroll.js', // a new refactored file,
			blockcounter: 'http://autocontext.begun.ru/blockcounter?pad_id={{pad_id}}&block={{block_id}}' // multiblock counter
		},
		stubs: {
			place_here: '&#1044;&#1072;&#1090;&#1100; &#1086;&#1073;&#1098;&#1103;&#1074;&#1083;&#1077;&#1085;&#1080;&#1077;',
			all_banners: '&#1042;&#1089;&#1077; &#1086;&#1073;&#1098;&#1103;&#1074;&#1083;&#1077;&#1085;&#1080;&#1103;',
			become_partner: '&#1057;&#1090;&#1072;&#1090;&#1100; &#1087;&#1072;&#1088;&#1090;&#1085;&#1077;&#1088;&#1086;&#1084;'
		},
		contacts: {
			card: '&#1050;&#1086;&#1085;&#1090;&#1072;&#1082;&#1090;&#1099;',
			ppcall: '&#1047;&#1074;&#1086;&#1085;&#1080;&#1090;&#1100;'
		},
		css: {
			prefix: 'begun',
			block_prefix: 'begun_block_',
			scroll_table_prefix: 'begun_adv_table_',
			scroll_div_prefix: 'begun_adv_common_',
			thumb: 'begun_adv_thumb',
			scroll: 'begun_scroll',
			logo_color: '#622678'
		},
		js: {
			inc_script_prefix: 'begun_js_',
			inc_fn_prefix: 'begun_fn_',
			banner_onclick: 'Begun.Autocontext.clickBanner(this)' // no 'return' here!
		}
	};

	/**
	 * autocontext blocks array
	 */
	Begun.Autocontext.Blocks = [];
	// this is the only reliable way to insert block styles!
	Begun.Autocontext.Blocks.push = function(elem, reset_banners, force_load){
		if (window.begun_auto_pad && elem.id){
			Begun.Autocontext.Blocks.loadBlockCounter(window.begun_auto_pad, elem.id);
		}
		var ac = Begun.Autocontext;
		ac.Blocks[ac.Blocks.length] = elem;
		if (elem.options && elem.options.visual){
			ac.printBlockStyle(elem.id, elem.options.visual);
		}
		ac.nullGlobalBlockParams();
		if (!ac.feed_done){
			ac.feed_done = true;
			ac.initFeedLoad();
			if (force_load){
				ac.loadExtraResources(force_load);
			}
		}else{
			ac.insertNonTextBlock(elem.id);
			ac.draw(reset_banners, force_load);
		}
	};
	Begun.Autocontext.Blocks.pushAll = function(blocks, already_run){ // 'already_run' might not be used
		while (Begun.Autocontext.Blocks.pop()){
			// just pop 'em
		};

		var block = null;
		var i = 0;
		var reset_banners = true;
		var force_load = !already_run;
		while (block = blocks[i]){
			Begun.Autocontext.Blocks.push(block, reset_banners, force_load);
			reset_banners = force_load = false;
			i++;
		}
		
		Begun.Autocontext.initScrollBlocks();
	};
	Begun.Autocontext.Blocks.loadBlockCounter = function(pad_id, block_id){
		if ((Begun.Autocontext.totalBannersUnlimited()) && (Begun.Autocontext.Blocks.length > 0)){
			(new Image()).src = (new Begun.Template(Begun.Autocontext.Strings.urls.blockcounter)).evaluate({'pad_id': pad_id, 'block_id': block_id});
		}
	};
	Begun.Autocontext.Blocks.getBlockById = function(block_id){
		var block = null;
		var i = 0;
		while (block = Begun.Autocontext.Blocks[i]){
			if (block.id == block_id){
				return block;
				break;
			}
			i++;
		}
		return null;
	};
	Begun.Autocontext.Blocks.del = function(block_id){
		var block = null;
		var i = 0;
		while (block = Begun.Autocontext.Blocks[i]){
			if (block.id == block_id){
				Begun.Autocontext.Blocks[i].id = -1;
				Begun.Autocontext.Blocks[i].options.banners_count = 0;
				break;
			}
			i++;
		}
	};

	/**
	 * autocontext callbacks singleton
	 */
	Begun.Autocontext.Callbacks = new function(){
		var _callbacks = {};
		var _extend = function(destination, source){
			for (var property in source){
				if (typeof source[property] == 'object'){
					var new_obj = {};
					for (var property2 in source[property]){
						if (typeof source[property][property2] == 'function'){
							if ((typeof destination[property] != 'undefined') && (typeof destination[property][property2] == 'function')){
								new_obj[property2] = function(old_func, new_func, property2){
									return function(e){
										old_func.apply(property2 == 'click' ? this : Begun.Autocontext);
										new_func.apply(property2 == 'click' ? this : Begun.Autocontext);
									};
								}(destination[property][property2], source[property][property2], property2);
							}else{
								new_obj[property2] = function(func, property2){
									return function(){
										func.apply(property2 == 'click' ? this : Begun.Autocontext);
									};
								}(source[property][property2], property2);
							}
						}
					}
					destination[property] = new_obj;
				}
			}
			return destination;
		};
		this.register = function(callbacks){
			_extend(_callbacks, callbacks);
		};
		this.dispatch = function(obj, method, context_obj){
			if (_callbacks[obj] && typeof _callbacks[obj][method] == 'function'){
				_callbacks[obj][method].apply(context_obj || this);
			}else{
				return null;
			}
		};
		this.getAllCallbacks = function(){
			return _callbacks;
		}
	};

	/**
	 * autocontext templates singleton
	 */
	Begun.Autocontext.Tpls = new function(){
		var css = {};
		css['default'] = '\
.begun_adv * {\
background: none; /* no !important for hover */\
border: none; /* no !important for hover */\
/*height: auto !important;*/ /* used for scrolling */\
clear: none !important;\
color: #000 !important;\
float: none !important;\
margin: 0 !important;\
padding: 0 !important;\
width: auto !important;\
letter-spacing: normal !important;\
word-spacing: normal !important;\
z-index: auto !important;\
font-size: 12px !important;\
font: normal normal 12px Arial, sans-serif !important;\
text-transform: none !important;\
list-style: none !important;\
position: static !important;\
text-indent: 0 !important;\
visibility: visible !important;\
}\
.begun_adv .begun_adv_common *, .begun_adv .begun_adv_sys *, .begun_adv .begun_adv_all *{\
background: none !important;\
border: none !important;\
}\
';
		css['block'] = '#begun_block_{{block_id}} .begun_adv{font:12px/18px Arial,sans-serif !important;color:#000 !important;text-align:left !important;}#begun_block_{{block_id}} .begun_adv b{font-weight:bold !important;display:inline !important;}#begun_block_{{block_id}} .begun_adv td{font-size:11px !important;}#begun_block_{{block_id}} .begun_adv,#begun_block_{{block_id}} .begun_adv table,#begun_block_{{block_id}} .begun_adv td,#begun_block_{{block_id}} .begun_adv div{padding:0 !important;text-align:left !important;}#begun_block_{{block_id}} .begun_adv table{border:none !important;border-collapse:collapse !important;}#begun_block_{{block_id}} .begun_adv td{vertical-align:middle !important;}#begun_block_{{block_id}} .begun_adv.begun_adv_hor td,#begun_block_{{block_id}} .begun_adv_fix_hor td{vertical-align:top !important;}#begun_block_{{block_id}} .begun_adv_sys{width:100% !important;}#begun_block_{{block_id}} .begun_adv_sys_sign_up{vertical-align:middle !important;}#begun_block_{{block_id}} .begun_adv_cell,#begun_block_{{block_id}} .begun_adv_all{text-align:left !important;}#begun_block_{{block_id}} .begun_adv_bullit{color:#aaa !important;}#begun_block_{{block_id}} .begun_adv_title,#begun_block_{{block_id}} .begun_adv_text{white-space:normal !important;display:block !important;}#begun_block_{{block_id}} .begun_adv_title,#begun_block_{{block_id}} .begun_adv_title *{font-weight:bold !important;}#begun_block_{{block_id}} .begun_adv_sys_logo div{vertical-align:middle !important;}#begun_block_{{block_id}} .begun_adv_sys_logo a:link,#begun_block_{{block_id}} .begun_adv_sys_logo a:visited,#begun_block_{{block_id}} .begun_adv_sys_logo a:hover,#begun_block_{{block_id}} .begun_adv_sys_logo a:active{color:{{block_logo_color}} !important;text-decoration:none !important;font-weight:bold !important;font-style:italic !important;position:relative !important;}#begun_block_{{block_id}} .begun_adv_sys_logo a{top:-1px !important;}#begun_block_{{block_id}} .begun_adv_sys_sign_up div{text-align:right !important;}#begun_block_{{block_id}} .begun_adv_ext .begun_adv_sys_logo,#begun_block_{{block_id}} .begun_adv_ext .begun_adv_sys_logo *{font-size:13px !important;line-height:17px !important;}#begun_block_{{block_id}} .begun_adv_hor,#begun_block_{{block_id}} .begun_adv_hor .begun_adv_table{width:100% !important;}#begun_block_{{block_id}} .begun_adv_hor .begun_adv_cell,#begun_block_{{block_id}} .begun_adv_hor .begun_adv_all{padding:0 16px 0 0 !important;}#begun_block_{{block_id}} .begun_adv_ver .begun_adv_cell .begun_adv_cell,#begun_block_{{block_id}} .begun_adv_ver .begun_adv_all{padding:5px 2px 4px 5px !important;}#begun_block_{{block_id}} .begun_adv_ext .begun_adv_sys_logo div{width:3.8em !important;height:2.7ex !important;left:-4px !important;position:relative !important;top:-2px !important;}#begun_block_{{block_id}} .begun_adv_ext .begun_adv_text{padding:2px 0 4px 0 !important;}#begun_block_{{block_id}} .begun_adv_ext .begun_adv_contact span{padding-right:0.2em !important;}#begun_block_{{block_id}} .begun_adv.begun_adv_ext.begun_adv_hor .begun_adv_sys_logo{width:100% !important;}#begun_block_{{block_id}} .begun_adv_ext.begun_adv_ver .begun_adv_sys_sign_up{padding-right:5px !important;}#begun_block_{{block_id}} .begun_adv_fix .begun_adv_cell{padding:0 5px 0 9px !important;}#begun_block_{{block_id}} .begun_adv_fix .begun_adv_cell,#begun_block_{{block_id}} .begun_adv_fix .begun_adv_cell *{font-size:11px !important;line-height:11px !important;}#begun_block_{{block_id}} .begun_adv_fix .begun_adv_title,#begun_block_{{block_id}} .begun_adv_fix .begun_adv_title *,#begun_block_{{block_id}} .begun_adv_fix .begun_adv_all *,#begun_block_{{block_id}} .begun_adv_fix .begun_adv_all{font-size:12px !important;line-height:13px !important;margin-bottom:2px !important;}#begun_block_{{block_id}} .begun_adv_fix .begun_adv_text,#begun_block_{{block_id}} .begun_adv_fix .begun_adv_text *{font-size:11px !important;line-height:12px !important;}#begun_block_{{block_id}} .begun_adv_fix .begun_adv_sys_logo,#begun_block_{{block_id}} .begun_adv_fix .begun_adv_sys_logo *{font-size:13px !important;line-height:17px !important;}#begun_block_{{block_id}} .begun_adv_fix .begun_adv_sys_sign_up,#begun_block_{{block_id}} .begun_adv_fix .begun_adv_sys_sign_up *{font:9px/11px Tahoma,Arial,sans-serif !important;}#begun_block_{{block_id}} .begun_adv_fix{overflow:hidden !important;}#begun_block_{{block_id}} .begun_adv_fix .begun_adv_sys_logo div{height:17px !important;float:left !important;}#begun_block_{{block_id}} .begun_adv_fix .begun_adv_common{overflow:hidden !important;}#begun_block_{{block_id}} .begun_adv_fix .begun_adv_text{padding:2px 0 !important;}#begun_block_{{block_id}} .begun_adv_fix .begun_adv_contact span{padding-right:2px !important;}#begun_block_{{block_id}} .begun_adv_fix_ver .begun_adv_sys_logo,#begun_block_{{block_id}} .begun_adv_ext .begun_adv_sys_logo{padding-left:9px !important;}#begun_block_{{block_id}} .begun_adv_fix_ver .begun_adv_sys_logo div{width:51px !important;}#begun_block_{{block_id}} .begun_adv_fix_ver .begun_adv_sys_sign_up div{width:93% !important;}#begun_block_{{block_id}} .begun_adv_fix_ver .begun_adv_all{height:23px !important;padding:2px 0 0 9px !important;}#begun_block_{{block_id}} .begun_adv_fix_ver .begun_adv_block{margin:5px 0 !important;}#begun_block_{{block_id}} .begun_adv_fix_hor .begun_adv_common{margin-top:7px !important;text-align:left !important;}#begun_block_{{block_id}} .begun_adv_fix_hor .begun_adv_block{margin:0 !important;}#begun_block_{{block_id}} .begun_adv_fix_hor .begun_adv_sys_logo{width:53px !important;float:left !important;padding:0 !important;}#begun_block_{{block_id}} .begun_adv_fix_hor .begun_adv_sys_logo div{width:53px !important;}#begun_block_{{block_id}} .begun_adv_fix_hor .begun_adv_sys_sign_up{width:53px !important;float:left !important;clear:left !important;}#begun_block_{{block_id}} .begun_adv_fix_hor .begun_adv_sys_sign_up div{padding-left:4px !important;text-align:left !important;}#begun_block_{{block_id}} .begun_adv_table{position:relative !important;}#begun_block_{{block_id}} .begun_adv_fix_hor .begun_adv_table{margin-left:60px !important;display:block !important;}#begun_block_{{block_id}} .begun_adv_240x400{width:240px !important;height:400px !important;}#begun_block_{{block_id}} .begun_adv_240x400 .begun_scroll{height:358px !important;}#begun_block_{{block_id}} .begun_adv_200x300{width:200px !important;height:300px !important;}#begun_block_{{block_id}} .begun_adv_200x300 .begun_scroll{height:283px !important;}#begun_block_{{block_id}} .begun_adv_468x60{width:468px !important;height:60px !important;}#begun_block_{{block_id}} .begun_adv_468x60 .begun_adv_sys_sign_up{margin-top:2px !important;}#begun_block_{{block_id}} .begun_adv_468x60 .begun_adv_sys_logo a{padding-left:4px !important;}#begun_block_{{block_id}} .begun_adv_468x60 .begun_adv_text{padding-bottom:4px !important;}#begun_block_{{block_id}} .begun_adv_468x60 .begun_adv_text *{line-height:10px !important;}#begun_block_{{block_id}} .begun_adv_728x90{width:728px !important;height:90px !important;}#begun_block_{{block_id}} .begun_adv_728x90 .begun_adv_sys_sign_up{margin-top:4px !important;}#begun_block_{{block_id}} .begun_adv_728x90 .begun_adv_sys_logo a{padding-left:4px !important;}#begun_block_{{block_id}} .begun_adv_728x90 .begun_adv_text{padding-bottom:4px !important;}#begun_block_{{block_id}} .begun_adv_120x600 .begun_scroll{height:560px !important;}#begun_block_{{block_id}} .begun_adv_160x600 .begun_scroll{height:560px !important;}#begun_block_{{block_id}} .begun_adv_ext .begun_adv_cell .begun_adv_cell,#begun_block_{{block_id}} .begun_adv_ext .begun_adv_all{padding:5px 2px 4px 5px !important;}#begun_block_{{block_id}} .begun_adv.begun_adv_ext.begun_adv_hor .begun_adv_sys_sign_up div{white-space:nowrap !important;margin-left:25px !important;}#begun_block_{{block_id}} .begun_thumb{float:left !important;margin:5px 5px 5px 0px !important;}#begun_block_{{block_id}} .begun_thumb img{/*border:1px solid#080 !important;*/ display:block !important;}#begun_block_{{block_id}} .begun_adv_thumb .begun_adv_block{margin-left:70px !important;}#begun_block_{{block_id}} .begun_adv_text,#begun_block_{{block_id}} .begun_adv_text *{color:#000 !important;}#begun_block_{{block_id}} .begun_adv_block{border:none !important;}#begun_block_{{block_id}} .begun_adv_728x90 .begun_adv_text,#begun_block_{{block_id}} .begun_adv_728x90 .begun_adv_text *,#begun_block_{{block_id}} .begun_adv_240x400 .begun_adv_text,#begun_block_{{block_id}} .begun_adv_240x400 .begun_adv_text *,#begun_block_{{block_id}} .begun_adv_160x600 .begun_adv_text,#begun_block_{{block_id}} .begun_adv_160x600 .begun_adv_text *{font-size:12px !important;line-height:13px !important;}#begun_block_{{block_id}} .begun_adv_240x400 .begun_adv_title,#begun_block_{{block_id}} .begun_adv_240x400 .begun_adv_title *,#begun_block_{{block_id}} .begun_adv_728x90 .begun_adv_title,#begun_block_{{block_id}} .begun_adv_728x90 .begun_adv_title *{font-size:14px !important;line-height:14px !important;}#begun_block_{{block_id}} .begun_adv_728x90 td.begun_adv_cell{width:50% !important;}#begun_block_{{block_id}} .begun_scroll{position:relative !important;}#begun_block_{{block_id}} .begun_adv_120x600 .begun_adv_block{width:106px !important;overflow:hidden !important;}#begun_block_{{block_id}} .begun_adv_160x600 .begun_adv_block{width:146px !important;overflow:hidden !important;}#begun_block_{{block_id}} .begun_adv .begun_adv_cell .begun_adv_phone *{font-size:1px !important;}#begun_block_{{block_id}} .begun_adv_phone{width:12px !important;margin:1px 3px 0 0 !important;float:left !important;}#begun_block_{{block_id}} .begun_adv_phone_wrapper{white-space:nowrap !important;}#begun_block_{{block_id}} div.begun_adv_contact > .begun_adv_phone{margin:0 5px 0 0 !important;}#begun_block_{{block_id}} .begun_adv_phone b{border:#118f00 solid 0 !important;height:1px !important;font-size:1px !important;line-height:1px !important;display:block !important;overflow:hidden !important;}#begun_block_{{block_id}} .begun_adv_phone .p0,#begun_block_{{block_id}} .begun_adv_phone .p1,#begun_block_{{block_id}} .begun_adv_phone .p3,#begun_block_{{block_id}} .begun_adv_phone .p5,#begun_block_{{block_id}} .begun_adv_phone .p8{background-color:#118f00 !important;}#begun_block_{{block_id}} .begun_adv_phone .p1,#begun_block_{{block_id}} .begun_adv_phone .p7,#begun_block_{{block_id}} .begun_adv_phone .p8{margin:0 1px !important;}#begun_block_{{block_id}} .begun_adv_phone .p2,#begun_block_{{block_id}} .begun_adv_phone .p7{border-width:0 4px !important;}#begun_block_{{block_id}} .begun_adv_phone .p3,#begun_block_{{block_id}} .begun_adv_phone .p6{margin:0 2px !important;}#begun_block_{{block_id}} .begun_adv_phone .p0{margin:0 3px !important;}#begun_block_{{block_id}} .begun_adv_phone .p4{border-width:0 3px !important;}#begun_block_{{block_id}} .begun_adv_phone .p5{margin:0 4px !important;}#begun_block_{{block_id}} .begun_adv_phone .p6{border-width:0 2px !important;}#begun_block_{{block_id}} .begun_adv_phone .p8{height:2px !important;}#begun_block_{{block_id}} .begun_adv_phone b{border-color:{{domain:color}} !important;}#begun_block_{{block_id}} .begun_adv_phone .p0,#begun_block_{{block_id}} .begun_adv_phone .p1,#begun_block_{{block_id}} .begun_adv_phone .p3,#begun_block_{{block_id}} .begun_adv_phone .p5,#begun_block_{{block_id}} .begun_adv_phone .p8{background-color:{{domain:color}} !important;}#begun_block_{{block_id}} .begun_adv_phone{font-size:11px !important;line-height:11px !important;margin-top:{{phone_margin_top}} px !important;}#begun_block_{{block_id}} .begun_adv_120x600{width:120px !important;height:600px !important;}#begun_block_{{block_id}} .begun_adv_160x600{width:160px !important;height:600px !important;}#begun_block_{{block_id}} .begun_adv_title a,#begun_block_{{block_id}} .begun_adv_title a *{color:{{title:color}} !important;}#begun_block_{{block_id}} .begun_adv .begun_adv_title a:hover,#begun_block_{{block_id}} .begun_adv .begun_adv_title a:hover *{color:#f00 !important;color:{{title_hover:color}} !important;}#begun_block_{{block_id}} .begun_adv_title,#begun_block_{{block_id}} .begun_adv_title *{font-size:{{title:fontSize}} !important;}#begun_block_{{block_id}} .begun_adv_all,#begun_block_{{block_id}} .begun_adv_all *{color:{{title:color}} !important;font-size:{{title:fontSize}} !important;}#begun_block_{{block_id}} .begun_adv_text,#begun_block_{{block_id}} .begun_adv_text *{color:{{text:color}} !important;font-size:{{text:fontSize}} !important;text-decoration:none !important;}#begun_block_{{block_id}} .begun_adv_contact,#begun_block_{{block_id}} .begun_adv_contact a,#begun_block_{{block_id}} .begun_adv_contact span{color:{{domain:color}} !important;font-size:{{domain:fontSize}} !important;}#begun_block_{{block_id}} .begun_adv_sys_sign_up,#begun_block_{{block_id}} .begun_adv_sys_sign_up *{color:{{domain:color}} !important;font-size:{{domain:fontSize}} !important;}#begun_block_{{block_id}} .begun_adv_contact a{color:{{domain:color}} !important;text-decoration:none !important;}#begun_block_{{block_id}} .begun_thumb img{/*border:1px solid {{block:borderColor}} !important;*/ display:block !important;}#begun_block_{{block_id}} .begun_adv_200x300 .begun_adv_common{height:283px !important;}#begun_block_{{block_id}} .begun_adv_200x300 .begun_adv_common table{height:100% !important;}#begun_block_{{block_id}} .begun_adv_200x300 .begun_adv_common.banners_count_1{height:auto !important;}#begun_block_{{block_id}} .begun_adv_200x300 .begun_adv_common.banners_count_1 table{height:auto !important;}#begun_block_{{block_id}} .begun_adv_240x400 .begun_adv_common{height:356px !important;}#begun_block_{{block_id}} .begun_adv_240x400 .begun_adv_common table{height:100% !important;}#begun_block_{{block_id}} .begun_adv_240x400 .begun_adv_common.banners_count_1{height:auto !important;}#begun_block_{{block_id}} .begun_adv_240x400 .begun_adv_common.banners_count_1 table{height:auto !important;}#begun_block_{{block_id}} .begun_adv_120x600 .begun_adv_common{height:553px !important;}#begun_block_{{block_id}} .begun_adv_120x600 .begun_adv_common table{height:100% !important;}#begun_block_{{block_id}} .begun_adv_120x600 .begun_adv_common.banners_count_1,#begun_block_{{block_id}} .begun_adv_120x600 .begun_adv_common.banners_count_2{height:auto !important;}#begun_block_{{block_id}} .begun_adv_120x600 .begun_adv_common.banners_count_1 table,#begun_block_{{block_id}} .begun_adv_120x600 .begun_adv_common.banners_count_2 table{height:auto !important;}#begun_block_{{block_id}} .begun_adv_160x600 .begun_adv_common{height:558px !important;}#begun_block_{{block_id}} .begun_adv_160x600 .begun_adv_common table{height:100% !important;}#begun_block_{{block_id}} .begun_adv_160x600 .begun_adv_common.banners_count_1,#begun_block_{{block_id}} .begun_adv_160x600 .begun_adv_common.banners_count_2{height:auto !important;}#begun_block_{{block_id}} .begun_adv_160x600 .begun_adv_common.banners_count_1 table,#begun_block_{{block_id}} .begun_adv_160x600 .begun_adv_common.banners_count_2 table{height:auto !important;}\
#begun_block_{{block_id}} .begun_adv {\
background-color: {{block:backgroundColor}}; /* no !important for hover */\
border: 1px solid {{block:borderColor}}; /* no !important for hover */\
filter: {{block:filter}};\
}\
';
		var html = {};
		html['blck_place'] = '<div id="{{id}}"></div>';
		html['link_iframe'] = '<iframe src="{{url}}" style="height:0;width:0;border:0"></iframe>';
		html['bnnr_glue'] = '<span class="begun_adv_bullit"> &#149; </span>';
		html['bnnr_phone'] = '\
<span class="begun_adv_phone"><b class="p0"></b><b class="p1"></b><b class="p2"></b><b class="p4"><b class="p3"></b></b><b class="p5"></b><b class="p6"><b class="p1"></b></b><b class="p7"></b><b class="p8"></b></span>\
';
		html['bnnr_card'] = '\
<span class="begun_adv_phone_wrapper">{{phone}}<span class="begun_adv_card"><a target="_blank" href="{{url}}" onclick={{onclick}} class="snap_noshots">{{card_text}}</a></span></span>\
';
		html['bnnr_ppcall'] = '\
<span class="begun_adv_phone_wrapper">{{phone}}<a href="javascript:void(0)" class="snap_noshots" onclick="showEnterForm({{banner_index}}, this, event)"><span class="begun_adv_card">{{ppcall_text}}</span></a></span>\
';
		html['bnnr_domain'] = '\
<span class="begun_adv_contact"><a class="snap_noshots" target="_blank" href="{{url}}" onclick={{onclick}} onmouseover="status=\'http://{{domain}}/\';return true" onmouseout="status=\'\';return true" title="{{domain}}">{{domain}}</a></span>\
';
		html['bnnr_geo'] = '\
<span class="begun_adv_city"><a class="snap_noshots" target="_blank" href="{{url}}" onclick={{onclick}} onmouseover="status=\'http://{{domain}}/\';return true" onmouseout="status=\'\';return true" title="{{domain}}">{{geo}}</a></span>\
';
		html['bnnr_thumb'] = '\
<a href="{{url}}" onclick={{onclick}} class="begun_thumb snap_noshots" target="_blank"><img src="{{src}}" width="56" height="42" alt=""/></a>\
';
		html['banner_vertical'] = '\
<tr>\
<td class="begun_adv_cell">\
{{thumb}}\
<div class="begun_adv_block">\
<div class="begun_adv_title"><a class="snap_noshots" target="_blank" href="{{url}}" onclick={{onclick}} onmouseover="status=\'http://{{domain}}/\';return true" onmouseout="status=\'\';return true" title="{{domain}}">{{title}}</a></div>\
<div class="begun_adv_text"><a class="snap_noshots" target="_blank" href="{{url}}" onclick={{onclick}} onmouseover="status=\'http://{{domain}}/\';return true" onmouseout="status=\'\';return true" title="{{domain}}">{{descr}}</a></div>\
<div class="begun_adv_contact">{{contact}}</div>\
</div>\
</td>\
</tr>\
';
		html['banner_200x300'] = html['banner_240x400'] = html['banner_120x600'] = html['banner_160x600'] = html['banner_vertical'];
		html['banner_horizontal'] = '\
<td class="begun_adv_cell" style="width:{{banner_width}} !important">\
{{thumb}}\
<div class="begun_adv_block">\
<div class="begun_adv_title"><a class="snap_noshots" target="_blank" href="{{url}}" onclick={{onclick}} onmouseover="status=\'http://{{domain}}/\';return true" onmouseout="status=\'\';return true" title="{{domain}}">{{title}}</a></div>\
<div class="begun_adv_text"><a class="snap_noshots" target="_blank" href="{{url}}" onclick={{onclick}} onmouseover="status=\'http://{{domain}}/\';return true" onmouseout="status=\'\';return true" title="{{domain}}">{{descr}}</a></div>\
<div class="begun_adv_contact">{{contact}}</div>\
</div>\
</td>\
';
		html['banner_468x60'] = html['banner_728x90'] = html['banner_horizontal'];
		html['blck_hover'] = ' onmouseover="Begun.Autocontext.blockHover(this, {backgroundColor: \'{{bgcolor_over}}\', borderColor: \'{{brdcolor_over}}\'})" onmouseout="Begun.Autocontext.blockHover(this, {backgroundColor: \'{{bgcolor_out}}\', borderColor: \'{{brdcolor_out}}\'})"';
		html['block_vertical'] = '\
<table class="begun_adv begun_adv_ext begun_adv_ver"{{block_hover}} style="width:{{block_width}}">\
<tr>\
<td class="begun_adv_cell">\
<table class="begun_adv_sys"><tr>\
<td class="begun_adv_sys_logo"><div><a href="{{begun_url}}" target="_blank" class="snap_noshots">begun</a></div></td>\
<td class="begun_adv_sys_sign_up"><div><a href="{{place_here_url}}" target="_blank" class="snap_noshots">{{place_here_text}}</a></div></td>\
</tr></table>\
<div class="begun_adv_common {{block_scroll_class}}" id="{{scroll_div_id}}">\
<table class="begun_adv_table {{css_thumbnails}}" id="{{scroll_table_id}}">\
{{banners}}\
</table>\
</div>\
<div class="begun_adv_all"><a href="{{all_banners_url}}" target="_blank" class="snap_noshots">{{all_banners_text}}</a></div>\
</td>\
</tr>\
</table>\
';
		html['block_horizontal'] = '\
<table class="begun_adv begun_adv_ext begun_adv_hor"{{block_hover}} style="width:{{block_width}}">\
<tr>\
<td class="begun_adv_cell">\
<table class="begun_adv_sys"><tr>\
<td class="begun_adv_sys_logo" colspan="{{begun_url_colspan}}"><div><a href="{{begun_url}}" target="_blank" class="snap_noshots">begun</a></div></td>\
<td class="begun_adv_sys_sign_up" style="display:{{become_partner_display}}"><div><a href="{{become_partner_url}}" target="_blank" class="snap_noshots">{{become_partner_text}}</a></div></td>\
<td class="begun_adv_sys_sign_up" style="display:{{place_here_display}}"><div><a href="{{place_here_url}}" target="_blank" class="snap_noshots">{{place_here_text}}</a></div></td>\
<td class="begun_adv_sys_sign_up" style="display:{{all_banners_display}}"><div><a href="{{all_banners_url}}" target="_blank" class="snap_noshots">{{all_banners_text}}</a></div></td>\
</tr></table>\
<div class="begun_adv_common {{block_scroll_class}}" id="{{scroll_div_id}}">\
<table class="begun_adv_table {{css_thumbnails}}" id="{{scroll_table_id}}">\
<tr>\
{{banners}}\
</tr>\
</table>\
</div>\
</td>\
</tr>\
</table>\
';
		html['block_468x60'] = '\
<div class="begun_adv begun_adv_fix begun_adv_fix_hor begun_adv_468x60"{{block_hover}}>\
<div class="begun_adv_common {{block_scroll_class}} banners_count_{{banners_count}}" id="{{scroll_div_id}}">\
<div class="begun_adv_sys_logo"><div><a href="{{begun_url}}" target="_blank" class="snap_noshots">begun</a></div></div>\
<div class="begun_adv_sys_sign_up" style="display:{{place_here_display}}"><div><a href="{{place_here_url}}" target="_blank" class="snap_noshots">{{place_here_text}}</a></div></div>\
<table class="begun_adv_table {{css_thumbnails}}" id="{{scroll_table_id}}">\
<tr>\
{{banners}}\
</tr>\
</table>\
</div>\
</div>\
';
		html['block_728x90'] = '\
<div class="begun_adv begun_adv_fix begun_adv_fix_hor begun_adv_728x90"{{block_hover}}>\
<div class="begun_adv_common {{block_scroll_class}} banners_count_{{banners_count}}" id="{{scroll_div_id}}">\
<div class="begun_adv_sys_logo"><div><a href="{{begun_url}}" target="_blank" class="snap_noshots">begun</a></div></div>\
<div class="begun_adv_sys_sign_up" style="display:{{place_here_display}}"><div><a href="{{place_here_url}}" target="_blank" class="snap_noshots">{{place_here_text}}</a></div></div>\
<table class="begun_adv_table {{css_thumbnails}}" id="{{scroll_table_id}}">\
<tr>\
{{banners}}\
</tr>\
</table>\
</div>\
</div>\
';
		html['block_200x300'] = '\
<div class="begun_adv begun_adv_fix begun_adv_fix_ver begun_adv_200x300"{{block_hover}}>\
<table class="begun_adv_sys"><tr>\
<td class="begun_adv_sys_logo"><div><a href="{{begun_url}}" target="_blank" class="snap_noshots">begun</a></div></td>\
<td class="begun_adv_sys_sign_up" style="display:{{place_here_display}}"><div><a href="{{place_here_url}}" target="_blank" class="snap_noshots">{{place_here_text}}</a></div></td>\
</tr></table>\
<div class="begun_adv_common {{block_scroll_class}} banners_count_{{banners_count}}" id="{{scroll_div_id}}">\
<table class="begun_adv_table {{css_thumbnails}}" id="{{scroll_table_id}}">\
{{banners}}\
</table>\
</div>\
</div>\
';
		html['block_240x400'] = '\
<div class="begun_adv begun_adv_fix begun_adv_fix_ver begun_adv_240x400"{{block_hover}}>\
<table class="begun_adv_sys"><tr>\
<td class="begun_adv_sys_logo"><div><a href="{{begun_url}}" target="_blank" class="snap_noshots">begun</a></div></td>\
<td class="begun_adv_sys_sign_up" style="display:{{place_here_display}}"><div><a href="{{place_here_url}}" target="_blank" class="snap_noshots">{{place_here_text}}</a></div></td>\
</tr></table>\
<div class="begun_adv_common {{block_scroll_class}} banners_count_{{banners_count}}" id="{{scroll_div_id}}">\
<table class="begun_adv_table {{css_thumbnails}}" id="{{scroll_table_id}}">\
{{banners}}\
</table>\
</div>\
<div class="begun_adv_all" style="display:{{all_banners_display}}"><a href="{{all_banners_url}}" target="_blank" class="snap_noshots">{{all_banners_text}}</a></div>\
</div>\
';
		html['block_120x600'] = '\
<div class="begun_adv begun_adv_fix begun_adv_fix_ver begun_adv_120x600"{{block_hover}}>\
<table class="begun_adv_sys"><tr>\
<td class="begun_adv_sys_logo"><div><a href="{{begun_url}}" target="_blank" class="snap_noshots">begun</a></div></td>\
<td class="begun_adv_sys_sign_up" style="display:{{place_here_display}}"><div><a href="{{place_here_url}}" target="_blank" class="snap_noshots">{{place_here_text}}</a></div></td>\
</tr></table>\
<div class="begun_adv_common {{block_scroll_class}} banners_count_{{banners_count}}" id="{{scroll_div_id}}">\
<table class="begun_adv_table {{css_thumbnails}}" id="{{scroll_table_id}}">\
{{banners}}\
</table>\
</div>\
<div class="begun_adv_all" style="display:{{all_banners_display}}"><a href="{{all_banners_url}}" target="_blank" class="snap_noshots">{{all_banners_text}}</a></div>\
</div>\
';
		html['block_160x600'] = '\
<div class="begun_adv begun_adv_fix begun_adv_fix_ver begun_adv_160x600"{{block_hover}}>\
<table class="begun_adv_sys"><tr>\
<td class="begun_adv_sys_logo"><div><a href="{{begun_url}}" target="_blank" class="snap_noshots">begun</a></div></td>\
<td class="begun_adv_sys_sign_up" style="display:{{place_here_display}}"><div><a href="{{place_here_url}}" target="_blank" class="snap_noshots">{{place_here_text}}</a></div></td>\
</tr></table>\
<div class="begun_adv_common {{block_scroll_class}} banners_count_{{banners_count}}" id="{{scroll_div_id}}">\
<table class="begun_adv_table {{css_thumbnails}}" id="{{scroll_table_id}}">\
{{banners}}\
</table>\
</div>\
<div class="begun_adv_all" style="display:{{all_banners_display}}"><a href="{{all_banners_url}}" target="_blank" class="snap_noshots">{{all_banners_text}}</a></div>\
</div>\
';
		html['search_banner_swf'] = '<object classid="clsid:D27CDB6E-AE6D-11cf-96B8-444553540000" width="240" height="400"><param name="movie" VALUE="{{source}}&link1={{url}}"><param name="wmode" value="opaque"><param name="allowScriptAccess" value="always"><param name="quality" VALUE="high"><embed src="{{source}}&link1={{url}}" quality="high" width="240" height="400" type="application/x-shockwave-flash" wmode="opaque"></embed></object>';
		html['search_banner_js'] = '';
		html['search_banner_img'] = '<a href="{{url}}" target="_blank"><img src="{{source}}&redir=1" border="0" width="240" height="400" /></a>';

		this.getCSS = function(type){
			return css[type];
		};
		this.getHTML = function(type){
			return html[type];
		};
		this.addTpls = function(){
			var types = ['html', 'css'];
			var i = 0;
			var type = null;
			while (type = types[i]){
				if (typeof window['begun_' + type + '_tpls'] != 'undefined'){
					var j = 0;
					var tpl = null;
					while (tpl = window['begun_' + type + '_tpls'][j]){
						Begun.extend(eval(type), tpl);
						j++;
					}
				}
				i++;
			}
		}
	};
	
	/**
	 * autocontext customization interface
	 */
	Begun.Autocontext.Customization = new function(){
		var _this = this;
		var ac = Begun.Autocontext;
		this.init = function(){
			if (typeof window.begun_urls != 'undefined'){
				_this.setURLs(window.begun_urls);
			}
			if (typeof window.begun_callbacks != 'undefined'){
				_this.setCallbacks(window.begun_callbacks);
			}
			_this.setTpls();
		};
		this.setURLs = function(urls){
			Begun.extend(ac.Strings.urls, urls || {});
		};
		this.setCallbacks = function(callbacks){
			ac.Callbacks.register(callbacks || {});
		};
		this.setTpls = function(){
			ac.Tpls.addTpls();
		};
	};

	/**
	 * init autocontext once
	 */
	(function(){
		var ac = Begun.Autocontext;
		
		ac.Customization.init();

		ac.setOptions({max_banners_count: 50, max_blocks_count: 10, scroll_banners_coef: 2});
		ac.printDefaultStyle();

		// DOMContentLoaded emulation, requires a function
		if (typeof onContentLoaded != 'function'){
			function onContentLoaded(f){
				var a,b=navigator.userAgent,d=document,w=window,
				c="__onContentLoaded__",e="addEventListener",o="opera",r="readyState",
				s="<scr".concat("ipt defer src='//:' on",r,"change='if(this.",r,"==\"complete\"){this.parentNode.removeChild(this);",c,"()}'></scr","ipt>");
				w[c]=(function(o){return function(){w[c]=function(){};for(a=arguments.callee;!a.done;a.done=1)f(o?o():o)}})(w[c]);
				if(d[e])d[e]("DOMContentLoaded",w[c],false);
				if(/WebKit|Khtml/i.test(b)||(w[o]&&parseInt(w[o].version())<9))
				(function(){/loaded|complete/.test(d[r])?w[c]():setTimeout(arguments.callee,1)})();
				else if(/MSIE/i.test(b))d.write(s);
			};
		}
	
		onContentLoaded(function(){
			// placed here + timeout due to IE
			(function(){
				if (!Begun.Scroller || ac.dom_change){
					window.setTimeout(arguments.callee, Begun.DOM_TIMEOUT);
				}else{
					ac.initScrollBlocks();
				}
			})();
			ac.Callbacks.dispatch('blocks', 'draw', ac);
		});
	})();
}

/**
 * init current block
 */
Begun.Autocontext.initCurrentBlock();
