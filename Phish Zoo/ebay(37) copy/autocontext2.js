/*
TODO (only in English):
* ie bug: font-size in banners is inherited, not overriden
*/

/**
 * root namespace
 */
if (typeof Begun == "undefined" || !Begun) var Begun = {};

Begun.extend = function(destination, source){
	for (var property in source)
		destination[property] = source[property];
	return destination;
};

/**
 * browsers
 */
Begun.Browser = {
    IE: !!(window.attachEvent && navigator.userAgent.indexOf('Opera') === -1),
    Opera:  navigator.userAgent.indexOf('Opera') > -1,
    WebKit: navigator.userAgent.indexOf('AppleWebKit/') > -1,
    Gecko:  navigator.userAgent.indexOf('Gecko') > -1 && navigator.userAgent.indexOf('KHTML') === -1
};

/**
 * utils
 */
Begun.Utils = {
	scriptCounter: 0,
	includeScript: function(url, type, callback, callback_name){
		var type = type || 'write'; // append or write
		var inc = 0;
		var script = null;
		if (url){
			if (type == 'write'){ // callback doesn't allow further document.writes in ie :-(
				this.scriptCounter++;
				var id = Begun.Autocontext.Strings.js.inc_script_prefix + this.scriptCounter;
				document.write('<scr'+'ipt type="text/javascript" src="' + url + '" id="' + id + '"></scr'+'ipt>'); 
				script = document.getElementById(id);
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
					}
					callback_fired = true;
				};
				script.onreadystatechange = function(){
					var check_statement = (Begun.Browser.Opera ? (this.readyState == "complete") : (this.readyState == "complete" || this.readyState == "loaded"));
					if (check_statement){
						if (!callback_fired){
							callback();
						}
						callback_fired = true;
					}
				};
			}
		}
	},
	includeStyle: function(css_text, type, id){
		var type = type || 'write'; // append or write
		var id = id || 'begun-default-css';
		if (css_text){
			if (type == 'write'){
				document.write('<style type="text/css" id="' + id + '">' + css_text + '</style>');
			}else if (type == 'append'){
				var head = document.getElementsByTagName("head")[0];
				// Webkit
				if (!head){
					head = document.createElement("head");
					document.documentElement.insertBefore(head, document.documentElement.firstChild);
				}
				// IE
				if (document.createStyleSheet){
					var style = document.createStyleSheet();
					style.cssText = css_text;
				}else{
					if (document.getElementById(id)){
						document.getElementById(id).innerHTML = css_text;
					}else{
						var style = document.createElement("style");
						style.setAttribute("type", "text/css");
						style.id = id;
						style.appendChild(document.createTextNode(css_text));
						head.appendChild(style);
					}
				}
			}
		}
	},
	toQuery: function(params){
		var result = '';
		for (var key in params)
			if (params[key] && params.hasOwnProperty && params.hasOwnProperty(key))
				result += '&' + key + '=' + escape(params[key]);
		return result;
	}
};

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
		this.feed_load_started = false;
		this.feed_rendered = false;
		this.extra_block_present = false;
		this.load_optimal_banners_count = false;
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
		this.setURLs = function(urls){
			Begun.extend(Begun.Autocontext.Strings.urls, urls || {});
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
			"stopwords": window.stopwords || '',
			"begun_self_keywords": window.begun_self_keywords || '',
			"ut_screen_width": screen.width || 0,
			"ut_screen_height": screen.height || 0,
			"json": 1, // for json feed!
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
		this.includePpcall = function(){
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
			if (isPpcallNeeded()) Begun.Utils.includeScript(
				Begun.Autocontext.Strings.urls.ppcall_js,
				'append'
			);
		};
		this.setVerticalBlockHeight = function(block, scroll_div){
			var banners_count = Number(block.options.banners_count);
			var table = document.getElementById(Begun.Autocontext.Strings.css.scroll_table_prefix + block.id);
			var height = 0;
			for (var i = 0; i < banners_count; i++){
				if (table.getElementsByTagName('tr')[i]) height += table.getElementsByTagName('tr')[i].clientHeight;
			}
			scroll_div.style.height = (height + 1) + 'px'; // a required slight increase for correct scrolling transition
			scroll_div.style.overflow = 'hidden';
		};
		this.initScrollBlocks = function(block){
			if (Begun.Scroller){
				var block = null;
				var i = 0;
				while (block = _this.Blocks[i]){
					var scroll_div = document.getElementById(Begun.Autocontext.Strings.css.scroll_div_prefix + block.id);
					var scroll_table = document.getElementById(Begun.Autocontext.Strings.css.scroll_table_prefix + block.id);
					if (Number(block.options.use_scroll) && scroll_div && scroll_table){
						if (block.options.dimensions.type == 'vertical') _this.setVerticalBlockHeight(block, scroll_div);
						var scroller = (new Begun.Scroller(
							scroll_table,
							{
								height: scroll_div.clientHeight,
								banners_count: Number(block.options.banners_count)
							}
						));
						if (scroll_table.clientHeight >= scroll_div.clientHeight){
							if (block.options.dimensions.type != 'vertical'){
								scroller.banners_count--;
							}
							_this.scrollers.push(scroller);
							scroller.start();
						}
					}
					i++;
				}
			}
		};
		this.includeScroll = function(){
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
			if (isScrollNeeded()) Begun.Utils.includeScript(
				Begun.Autocontext.Strings.urls.scroll_js,
				'append',
				function(){
					_this.initScrollBlocks();
				}
			);
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
			if (_this.links) {
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
			}
		};
		this.onFeedBlocksReady = function(){
			if (typeof window.begunAds == 'undefined') return;
			_this.feed_rendered = true;
			_this.banners = window.begunAds.banners || {};
			_this.stubs = window.begunAds.stubs || {};
			_this.links = window.begunAds.links || {};
			Begun.extend(_this.responseParams, window.begunAds.params || {});
			_this.draw();
		};
		this.setMaxScrollers = function(){
			_this.maxScrollers = _this.options.max_scrollers;
		};
		this.draw = function(){
			_this.setMaxScrollers(); // placed it here due to wizard
			_this.handleFeed();
			_this.handleBehaviour();
			_this.handleHypercontext();
			_this.handleLinks();
			_this.includePpcall();
			_this.includeScroll();
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
		this.initCurrentBlock = function(){
			if (_this.load_optimal_banners_count){
				if ((_this.isFirstBlockWithBanners() && !window.begun_total_banners) || (_this.noMoreBlocksNeeded())){
					return; // don't waste time and effort
				}
			}
			if ((typeof window.begun_block_id != 'undefined' && window.begun_block_id > 0) && (typeof window.begun_auto_pad != 'undefined' && window.begun_auto_pad > 0)){
				_this.printBlockPlace(window.begun_block_id);
				if (window.begun_extra_block){ // extra internal block
					_this.Blocks.push(window.begun_extra_block);
				}else{ // external block
					Begun.Utils.includeScript(
						(new Begun.Template(Begun.Autocontext.Strings.urls.block_js)).evaluate({'pad_id': window.begun_auto_pad, 'block_id': window.begun_block_id}),
						'write' // only write!!
					);
				}
			};
		};
		this.getActualBlockBannersCount = function(block){
			if (!block) return 0;
			return Number(block.options.banners_count) * (Number(block.options.use_scroll) ? _this.options.scroll_banners_coef : 1);
		};
		this.isFirstBlockWithBanners = function(){
			return (typeof _this.Blocks[0] != "undefined" && _this.Blocks[0].options && _this.Blocks[0].options.banners_count);
		};
		this.initFeedLoad = function(){
			if (!_this.feed_load_started){
				var first_block_banners_count = 0;
				if (_this.isFirstBlockWithBanners()){
					first_block_banners_count = _this.getActualBlockBannersCount(_this.Blocks[0]);
				}
				if (_this.load_optimal_banners_count){
					var begun_total_banners = window.begun_total_banners || first_block_banners_count || 0;
				}else{
					var begun_total_banners = window.begun_total_banners || _this.options.max_banners_count || 0;
				}
				if (begun_total_banners){
					_this.setOptions({max_banners_count: begun_total_banners});
					_this.feed_load_started = true;
					_this.loadFeed();
				}
			}
		};
		// window.begunAds has just become available
		this.loadFeedDone = function(){
			if (_this.load_optimal_banners_count){
				if ((_this.isFirstBlockWithBanners() && !window.begun_total_banners) && (!_this.feed_rendered)){
					_this.onFeedBlocksReady();
				}
			}
		};
		this.loadFeed = function(){
			_this.prepareDaemonParams();
			Begun.Utils.includeScript(
				(Begun.Autocontext.Strings.urls.daemon + Begun.Utils.toQuery(_this.requestParams)).substring(0, 1524).replace(/%[0-9a-fA-F]?$/, ''),
				'append',
				function(){
					_this.loadFeedDone();
				}
			);
		};
		this.printBlockPlace = function(block_id){
			var vars = {id: Begun.Autocontext.Strings.css.block_prefix + block_id};
			document.write((new Begun.Template(_this.Tpls.getHTML('block_place'))).evaluate(vars));
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
		this.printBlockStyle = function(block_id, styles){
			var styles = styles || {};
			var vars = {};
			vars.block_id = block_id || -1;
			vars.phone_margin_top = 1; // TODO: calc from font-size
			for (var key in styles){
				if (styles[key] && styles.hasOwnProperty && styles.hasOwnProperty(key)){
					for (var key2 in styles[key]){
						if (styles[key][key2] && styles[key].hasOwnProperty && styles[key].hasOwnProperty(key2)){
							vars[key + ':' + key2] = typeof styles[key][key2] == 'number' ? styles[key][key2] + 'px' : styles[key][key2];
						}
					}
				}
			}
			vars.block_logo_color = _this.getBlockLogoColor(styles);
			var css_text = (new Begun.Template(_this.Tpls.getCSS('block'))).evaluate(vars);
			Begun.Utils.includeStyle(css_text, 'append', 'begun-block-css-' + block_id); // non-IE with append!!
		};
		this.getBanner = function(type, index){
			var banner = null;
			try{
				banner = _this.banners[type][index];
			}catch(e){}
			return banner;
		};
		this.getStub = function(type){
			return _this.stubs[type] || null;
		};
		this.getThumbSrcByBannerId = function(banner_id){
			var src = _this.responseParams['thumbs_src'] ? 'http://' + _this.responseParams['thumbs_src'] + '/' : Begun.Autocontext.Strings.urls.thumbs;
			var banner_id = banner_id + '';
			if (banner_id && banner_id.length > 2){
				src += banner_id.charAt(banner_id.length - 2);
				src += '/' + banner_id.charAt(banner_id.length - 1);
				src += '/' + banner_id + '.jpg';
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
		this.getBannerHTML = function(banner, block){
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
				var vars = banner;
				vars.contact = banner_contacts.join(_this.Tpls.getHTML('bnnr_glue'));
				vars.url = _this.addMisc2URL(block, banner.url);
				vars.thumb = Number(block.options.show_thumbnails) ? (new Begun.Template(_this.Tpls.getHTML('bnnr_thumb'))).evaluate({url: banner.url, src: _this.getThumbSrcByBannerId(banner.banner_id || 0)}) : '';
				vars.banner_width = Math.round(100 / Number(block.options.banners_count)) + '%';
				return (new Begun.Template(_this.Tpls.getHTML('banner_' + block.options.dimensions.type))).evaluate(vars);
			}else{
				return '';
			}
		};
		this.getBlockHTML = function(banners_html, block){
			if (!banners_html) return '';
			var banners_html = banners_html.join('');
			var stub_display = Number(_this.responseParams['stub']) ? '' : 'none';
			// TODO: how many stubs and which ones
			/*function isBannersEnough(){
				//
			}*/
			var vars = {};
			var block_hover_html = '';
			var block_opts = block.options.visual;
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
				var elem = document.getElementById(Begun.Autocontext.Strings.css.block_prefix + block.id);
				var html = _this.getBlockHTML(banners_html, block);
				function show(elem, html){
					elem.innerHTML = html; // rewriteable in wizard!
				}
				// fix most common partner mistakes for ie
				if (Begun.Browser.IE){
					var appendTableCell = function(tr, elem){
						var td = document.createElement('td');
						tr.appendChild(td);
						td.innerHTML = elem.outerHTML;
						show(td.firstChild, html);
						elem.parentNode.removeChild(elem);
					}
					var parent = null;
					if ((parent = elem.parentNode) && (parent.tagName) && (['ol', 'ul'].in_array(parent.tagName.toLowerCase()))){
						var parent2 = parent.parentNode;
						parent2.insertBefore(elem, parent);
						show(elem, html);
					}else if ((parent) && (parent = elem.parentNode.parentNode) && (parent.tagName)){
						switch (parent.tagName.toLowerCase()){
							case 'table':
								/*var tr = document.createElement('tr');
								parent.lastChild.appendChild(tr); // append to (implied) tbody
								appendTableCell(tr, elem);
								break;*/
							case 'tr':
								/*appendTableCell(parent, elem);
								break;*/
							case 'thead':
							case 'tbody':
							case 'tfoot':
								/*var tr = document.createElement('tr');
								parent.appendChild(tr); // append to thead/tbody/tfoot
								appendTableCell(tr, elem);
								break;*/
							default:
								try{
								show(elem, html);
								}catch(e){
									//
								}
						}
					}else{
						try{
						show(elem, html);
						}catch(e){
							//
						}
					}
				}else{
					show(elem, html);
				}
			}
		}
		this.fillBlocks = function(){
			var block = null;
			var block_index = 0;
			var out_of_banners = false;
			
			// sort the block according to the order
			_this.Blocks.sortByOrder();

			// loaded blocks only!
			while ((block = _this.Blocks[block_index]) && (!out_of_banners)){
				var banners_html = [];
				var block_banner_count = 0;
				var banner = null;
				if (Number(block.options.use_scroll) && 0 < _this.maxScrollers--){// with scrolling, all the banners in the feed
					while (banner = _this.getBanner('autocontext', block_banner_count++)){
						var banner_html = _this.getBannerHTML(banner, block);
						if (banner_html) banners_html.push(banner_html);
					}
				}else{// without scrolling, just the needed amount
					if (typeof window.begun_use_block_distr != 'undefined' && window.begun_use_block_distr){
						var i = 0;
						while (banner = _this.getBanner('autocontext', i)){
							if (banner.block_id && banner.block_id == block.id){
								var banner_html = _this.getBannerHTML(banner, block);
								if (banner_html) banners_html.push(banner_html);
							}
							i++;
						}
					}else{
						var banners_count = block.options.banners_count;
						while ((block_banner_count < banners_count) && (_this.banner_index < _this.options.max_banners_count)){
							banner = _this.getBanner('autocontext', _this.banner_index) || null;
							if (banner){
								var banner_html = _this.getBannerHTML(banner, block);
								if (banner_html) banners_html.push(banner_html);
							}else{
								out_of_banners = true;
								break;
							}
							block_banner_count++;
							_this.banner_index++;
						}
					}
				}
				_this.printBlock(banners_html, block);
				block_index++;
			}
		};
		this.blockHover = function(obj, styles){
			obj.style.backgroundColor = styles.backgroundColor;
			obj.style.borderColor = styles.borderColor;
		};
		this.nullGlobalBlockParams = function(){
			window.begun_block_id = null;
			window.begun_block_order = null;
		};
		this.setExtraBlockResponseParams = function(block){
			block.options.use_scroll = Number(_this.responseParams['autoscroll']);
			block.options.show_thumbnails = Number(_this.responseParams['thumbs']);
			// multispan = 0 --> old AC
		};
		this.handleBehaviour = function(){
			if (typeof window.begun_use_block_distr != 'undefined' && window.begun_use_block_distr && _this.banners['behaviour'].length){
				var i = 0;
				var block = null;
				var banners_html = [];
				while (banner = _this.getBanner('behaviour', i)){
					if (banner.block_id){
						if (!block){
							block = _this.Blocks.getBlockById(banner.block_id);
						}
						var banner_html = _this.getBannerHTML(banner, block);
						if (banner_html) banners_html.push(banner_html);
					}
					i++;
				}
				_this.printBlock(banners_html, block);
			}
		};
		this.handleFeed = function(){
			if (window.begun_extra_block && window.begun_extra_block.id){
				_this.setExtraBlockResponseParams(_this.Blocks.getBlockById(window.begun_extra_block.id));
			}else{
				_this.banner_index = window.begun_auto_limit || 0; // if the old code is present & there is no extra block, then we will go from begun_auto_limit + 1
			}
			_this.fillBlocks();
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
			inc_fn_prefix: 'begun_fn_'
		}
	};

	/**
	 * autocontext blocks array
	 */
	Begun.Autocontext.Blocks = [];
	// this is the only reliable way to insert block styles!
	Begun.Autocontext.Blocks.push = function(elem){
		if (window.begun_auto_pad && elem.id){
			Begun.Autocontext.Blocks.loadBlockCounter(window.begun_auto_pad, elem.id);
		}
		var ac = Begun.Autocontext;
		ac.Blocks[ac.Blocks.length] = elem;
		ac.Blocks[ac.Blocks.length - 1].order = (window.begun_block_order || (isNaN(ac.options.max_blocks_count) ? 999 : ac.options.max_blocks_count));
		ac.printBlockStyle(elem.id, elem.options.visual);
		ac.nullGlobalBlockParams();
		ac.initFeedLoad(); // the best place!
	};
	Begun.Autocontext.Blocks.loadBlockCounter = function(pad_id, block_id){
		if ((!window.begun_total_banners) && (Begun.Autocontext.Blocks.length > 0) && (!Begun.Autocontext.load_optimal_banners_count)){
			(new Image()).src = (new Begun.Template(Begun.Autocontext.Strings.urls.blockcounter)).evaluate({'pad_id': pad_id, 'block_id': block_id});
		}
	};
	Begun.Autocontext.Blocks.sortByOrder = function(){
		if (Begun.Autocontext.Blocks.length > 1) Begun.Autocontext.Blocks.sort(function(a, b){
			if (a.order > b.order)
				return 1;
			if (a.order < b.order)
				return -1;
			return 0;
		});
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
	
	/**
	 * autocontext templates signleton
	 */
	Begun.Autocontext.Tpls = new function(){
		var _this = this;
		this.css = {};
		this.css['default'] = '\
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
		this.css['block'] = '#begun_block_{{block_id}} .begun_adv{font:12px/18px Arial,sans-serif !important;color:#000 !important;text-align:left !important;}#begun_block_{{block_id}} .begun_adv b{font-weight:bold !important;display:inline !important;}#begun_block_{{block_id}} .begun_adv td{font-size:11px !important;}#begun_block_{{block_id}} .begun_adv,#begun_block_{{block_id}} .begun_adv table,#begun_block_{{block_id}} .begun_adv td,#begun_block_{{block_id}} .begun_adv div{padding:0 !important;text-align:left !important;}#begun_block_{{block_id}} .begun_adv table{border:none !important;border-collapse:collapse !important;}#begun_block_{{block_id}} .begun_adv td{vertical-align:top !important;}#begun_block_{{block_id}} .begun_adv_sys{width:100% !important;}#begun_block_{{block_id}} .begun_adv_sys_sign_up{vertical-align:middle !important;}#begun_block_{{block_id}} .begun_adv_cell,#begun_block_{{block_id}} .begun_adv_all{text-align:left !important;}#begun_block_{{block_id}} .begun_adv_bullit{color:#aaa !important;}#begun_block_{{block_id}} .begun_adv_title,#begun_block_{{block_id}} .begun_adv_text{white-space:normal !important;display:block !important;}#begun_block_{{block_id}} .begun_adv_title,#begun_block_{{block_id}} .begun_adv_title *{font-weight:bold !important;}#begun_block_{{block_id}} .begun_adv_sys_logo div{vertical-align:middle !important;}#begun_block_{{block_id}} .begun_adv_sys_logo a:link,#begun_block_{{block_id}} .begun_adv_sys_logo a:visited,#begun_block_{{block_id}} .begun_adv_sys_logo a:hover,#begun_block_{{block_id}} .begun_adv_sys_logo a:active{color:{{block_logo_color}} !important;text-decoration:none !important;font-weight:bold !important;font-style:italic !important;position:relative !important;}#begun_block_{{block_id}} .begun_adv_sys_logo a{top:-1px !important;}#begun_block_{{block_id}} .begun_adv_sys_sign_up div{text-align:right !important;}#begun_block_{{block_id}} .begun_adv_ext .begun_adv_sys_logo,#begun_block_{{block_id}} .begun_adv_ext .begun_adv_sys_logo *{font-size:13px !important;line-height:17px !important;}#begun_block_{{block_id}} .begun_adv_hor,#begun_block_{{block_id}} .begun_adv_hor .begun_adv_table{width:100% !important;}#begun_block_{{block_id}} .begun_adv_hor .begun_adv_cell,#begun_block_{{block_id}} .begun_adv_hor .begun_adv_all{padding:0 16px 0 0 !important;}#begun_block_{{block_id}} .begun_adv_ver .begun_adv_cell .begun_adv_cell,#begun_block_{{block_id}} .begun_adv_ver .begun_adv_all{padding:5px 2px 4px 5px !important;}#begun_block_{{block_id}} .begun_adv_ext .begun_adv_sys_logo div{width:3.8em !important;height:2.7ex !important;left:-4px !important;position:relative !important;top:-2px !important;}#begun_block_{{block_id}} .begun_adv_ext .begun_adv_text{padding:2px 0 4px 0 !important;}#begun_block_{{block_id}} .begun_adv_ext .begun_adv_contact span{padding-right:0.2em !important;}#begun_block_{{block_id}} .begun_adv.begun_adv_ext.begun_adv_hor .begun_adv_sys_logo{width:100% !important;}#begun_block_{{block_id}} .begun_adv_ext.begun_adv_ver .begun_adv_sys_sign_up{padding-right:5px !important;}#begun_block_{{block_id}} .begun_adv_fix .begun_adv_cell{padding:0 5px 0 9px !important;}#begun_block_{{block_id}} .begun_adv_fix .begun_adv_cell,#begun_block_{{block_id}} .begun_adv_fix .begun_adv_cell *{font-size:11px !important;line-height:11px !important;}#begun_block_{{block_id}} .begun_adv_fix .begun_adv_title,#begun_block_{{block_id}} .begun_adv_fix .begun_adv_title *,#begun_block_{{block_id}} .begun_adv_fix .begun_adv_all *,#begun_block_{{block_id}} .begun_adv_fix .begun_adv_all{font-size:12px !important;line-height:13px !important;margin-bottom:2px !important;}#begun_block_{{block_id}} .begun_adv_fix .begun_adv_text,#begun_block_{{block_id}} .begun_adv_fix .begun_adv_text *{font-size:11px !important;line-height:12px !important;}#begun_block_{{block_id}} .begun_adv_fix .begun_adv_sys_logo,#begun_block_{{block_id}} .begun_adv_fix .begun_adv_sys_logo *{font-size:13px !important;line-height:17px !important;}#begun_block_{{block_id}} .begun_adv_fix .begun_adv_sys_sign_up,#begun_block_{{block_id}} .begun_adv_fix .begun_adv_sys_sign_up *{font:9px/11px Tahoma,Arial,sans-serif !important;}#begun_block_{{block_id}} .begun_adv_fix{overflow:hidden !important;}#begun_block_{{block_id}} .begun_adv_fix .begun_adv_sys_logo div{height:17px !important;float:left !important;}#begun_block_{{block_id}} .begun_adv_fix .begun_adv_common{overflow:hidden !important;}#begun_block_{{block_id}} .begun_adv_fix .begun_adv_text{padding:2px 0 !important;}#begun_block_{{block_id}} .begun_adv_fix .begun_adv_contact span{padding-right:2px !important;}#begun_block_{{block_id}} .begun_adv_fix_ver .begun_adv_sys_logo,#begun_block_{{block_id}} .begun_adv_ext .begun_adv_sys_logo{padding-left:9px !important;}#begun_block_{{block_id}} .begun_adv_fix_ver .begun_adv_sys_logo div{width:51px !important;}#begun_block_{{block_id}} .begun_adv_fix_ver .begun_adv_sys_sign_up div{width:93% !important;}#begun_block_{{block_id}} .begun_adv_fix_ver .begun_adv_all{height:23px !important;padding:2px 0 0 9px !important;}#begun_block_{{block_id}} .begun_adv_fix_ver .begun_adv_block{margin:5px 0 !important;}#begun_block_{{block_id}} .begun_adv_fix_hor .begun_adv_common{margin-top:7px !important;text-align:left !important;}#begun_block_{{block_id}} .begun_adv_fix_hor .begun_adv_block{margin:0 !important;}#begun_block_{{block_id}} .begun_adv_fix_hor .begun_adv_sys_logo{width:53px !important;float:left !important;padding:0 !important;}#begun_block_{{block_id}} .begun_adv_fix_hor .begun_adv_sys_logo div{width:53px !important;}#begun_block_{{block_id}} .begun_adv_fix_hor .begun_adv_sys_sign_up{width:53px !important;float:left !important;clear:left !important;}#begun_block_{{block_id}} .begun_adv_fix_hor .begun_adv_sys_sign_up div{padding-left:4px !important;text-align:left !important;}#begun_block_{{block_id}} .begun_adv_table{position:relative !important;}#begun_block_{{block_id}} .begun_adv_fix_hor .begun_adv_table{margin-left:60px !important;display:block !important;}#begun_block_{{block_id}} .begun_adv_240x400{width:240px !important;height:400px !important;}#begun_block_{{block_id}} .begun_adv_240x400 .begun_scroll{height:358px !important;}#begun_block_{{block_id}} .begun_adv_200x300{width:200px !important;height:300px !important;}#begun_block_{{block_id}} .begun_adv_200x300 .begun_scroll{height:283px !important;}#begun_block_{{block_id}} .begun_adv_468x60{width:468px !important;height:60px !important;}#begun_block_{{block_id}} .begun_adv_468x60 .begun_adv_sys_sign_up{margin-top:2px !important;}#begun_block_{{block_id}} .begun_adv_468x60 .begun_adv_sys_logo a{padding-left:4px !important;}#begun_block_{{block_id}} .begun_adv_468x60 .begun_adv_text{padding-bottom:4px !important;}#begun_block_{{block_id}} .begun_adv_468x60 .begun_adv_text *{line-height:10px !important;}#begun_block_{{block_id}} .begun_adv_728x90{width:728px !important;height:90px !important;}#begun_block_{{block_id}} .begun_adv_728x90 .begun_adv_sys_sign_up{margin-top:4px !important;}#begun_block_{{block_id}} .begun_adv_728x90 .begun_adv_sys_logo a{padding-left:4px !important;}#begun_block_{{block_id}} .begun_adv_728x90 .begun_adv_text{padding-bottom:4px !important;}#begun_block_{{block_id}} .begun_adv_120x600 .begun_scroll{height:560px !important;}#begun_block_{{block_id}} .begun_adv_160x600 .begun_scroll{height:560px !important;}#begun_block_{{block_id}} .begun_adv_ext .begun_adv_cell .begun_adv_cell,#begun_block_{{block_id}} .begun_adv_ext .begun_adv_all{padding:5px 2px 4px 5px !important;}#begun_block_{{block_id}} .begun_adv.begun_adv_ext.begun_adv_hor .begun_adv_sys_sign_up div{white-space:nowrap !important;margin-left:25px !important;}#begun_block_{{block_id}} .begun_thumb{float:left !important;margin:5px 5px 5px 0px !important;}#begun_block_{{block_id}} .begun_thumb img{border:1px solid#080 !important;display:block !important;}#begun_block_{{block_id}} .begun_adv_thumb .begun_adv_block{margin-left:70px !important;}#begun_block_{{block_id}} .begun_adv_text,#begun_block_{{block_id}} .begun_adv_text *{color:#000 !important;}#begun_block_{{block_id}} .begun_adv_block{border:none !important;}#begun_block_{{block_id}} .begun_adv_728x90 .begun_adv_text,#begun_block_{{block_id}} .begun_adv_728x90 .begun_adv_text *,#begun_block_{{block_id}} .begun_adv_240x400 .begun_adv_text,#begun_block_{{block_id}} .begun_adv_240x400 .begun_adv_text *,#begun_block_{{block_id}} .begun_adv_160x600 .begun_adv_text,#begun_block_{{block_id}} .begun_adv_160x600 .begun_adv_text *{font-size:12px !important;line-height:13px !important;}#begun_block_{{block_id}} .begun_adv_240x400 .begun_adv_title,#begun_block_{{block_id}} .begun_adv_240x400 .begun_adv_title *,#begun_block_{{block_id}} .begun_adv_728x90 .begun_adv_title,#begun_block_{{block_id}} .begun_adv_728x90 .begun_adv_title *{font-size:14px !important;line-height:14px !important;}#begun_block_{{block_id}} .begun_adv_728x90 td.begun_adv_cell{width:50% !important;}#begun_block_{{block_id}} .begun_scroll{position:relative !important;}#begun_block_{{block_id}} .begun_adv_120x600 .begun_adv_block{width:106px !important;overflow:hidden !important;}#begun_block_{{block_id}} .begun_adv_160x600 .begun_adv_block{width:146px !important;overflow:hidden !important;}#begun_block_{{block_id}} .begun_adv .begun_adv_cell .begun_adv_phone *{font-size:1px !important;}#begun_block_{{block_id}} .begun_adv_phone{width:12px !important;margin:1px 3px 0 0 !important;float:left !important;}#begun_block_{{block_id}} .begun_adv_phone_wrapper{white-space:nowrap !important;}#begun_block_{{block_id}} div.begun_adv_contact > .begun_adv_phone{margin:0 5px 0 0 !important;}#begun_block_{{block_id}} .begun_adv_phone b{border:#118f00 solid 0 !important;height:1px !important;font-size:1px !important;line-height:1px !important;display:block !important;overflow:hidden !important;}#begun_block_{{block_id}} .begun_adv_phone .p0,#begun_block_{{block_id}} .begun_adv_phone .p1,#begun_block_{{block_id}} .begun_adv_phone .p3,#begun_block_{{block_id}} .begun_adv_phone .p5,#begun_block_{{block_id}} .begun_adv_phone .p8{background-color:#118f00 !important;}#begun_block_{{block_id}} .begun_adv_phone .p1,#begun_block_{{block_id}} .begun_adv_phone .p7,#begun_block_{{block_id}} .begun_adv_phone .p8{margin:0 1px !important;}#begun_block_{{block_id}} .begun_adv_phone .p2,#begun_block_{{block_id}} .begun_adv_phone .p7{border-width:0 4px !important;}#begun_block_{{block_id}} .begun_adv_phone .p3,#begun_block_{{block_id}} .begun_adv_phone .p6{margin:0 2px !important;}#begun_block_{{block_id}} .begun_adv_phone .p0{margin:0 3px !important;}#begun_block_{{block_id}} .begun_adv_phone .p4{border-width:0 3px !important;}#begun_block_{{block_id}} .begun_adv_phone .p5{margin:0 4px !important;}#begun_block_{{block_id}} .begun_adv_phone .p6{border-width:0 2px !important;}#begun_block_{{block_id}} .begun_adv_phone .p8{height:2px !important;}#begun_block_{{block_id}} .begun_adv_phone b{border-color:{{domain:color}} !important;}#begun_block_{{block_id}} .begun_adv_phone .p0,#begun_block_{{block_id}} .begun_adv_phone .p1,#begun_block_{{block_id}} .begun_adv_phone .p3,#begun_block_{{block_id}} .begun_adv_phone .p5,#begun_block_{{block_id}} .begun_adv_phone .p8{background-color:{{domain:color}} !important;}#begun_block_{{block_id}} .begun_adv_phone{font-size:11px !important;line-height:11px !important;margin-top:{{phone_margin_top}} px !important;}#begun_block_{{block_id}} .begun_adv_120x600{width:120px !important;height:600px !important;}#begun_block_{{block_id}} .begun_adv_160x600{width:160px !important;height:600px !important;}#begun_block_{{block_id}} .begun_adv_title a,#begun_block_{{block_id}} .begun_adv_title a *{color:{{title:color}} !important;}#begun_block_{{block_id}} .begun_adv .begun_adv_title a:hover,#begun_block_{{block_id}} .begun_adv .begun_adv_title a:hover *{color:#f00 !important;color:{{title_hover:color}} !important;}#begun_block_{{block_id}} .begun_adv_title,#begun_block_{{block_id}} .begun_adv_title *{font-size:{{title:fontSize}} !important;}#begun_block_{{block_id}} .begun_adv_all,#begun_block_{{block_id}} .begun_adv_all *{color:{{title:color}} !important;font-size:{{title:fontSize}} !important;}#begun_block_{{block_id}} .begun_adv_text,#begun_block_{{block_id}} .begun_adv_text *{color:{{text:color}} !important;font-size:{{text:fontSize}} !important;text-decoration:none !important;}#begun_block_{{block_id}} .begun_adv_contact,#begun_block_{{block_id}} .begun_adv_contact a,#begun_block_{{block_id}} .begun_adv_contact span{color:{{domain:color}} !important;font-size:{{domain:fontSize}} !important;}#begun_block_{{block_id}} .begun_adv_sys_sign_up,#begun_block_{{block_id}} .begun_adv_sys_sign_up *{color:{{domain:color}} !important;font-size:{{domain:fontSize}} !important;}#begun_block_{{block_id}} .begun_adv_contact a{color:{{domain:color}} !important;text-decoration:none !important;}#begun_block_{{block_id}} .begun_thumb img{border:1px solid {{block:borderColor}} !important;display:block !important;}\
#begun_block_{{block_id}} .begun_adv {\
background-color: {{block:backgroundColor}}; /* no !important for hover */\
border: 1px solid {{block:borderColor}}; /* no !important for hover */\
}\
';
		this.html = {};
		this.html['block_place'] = '<div id="{{id}}"></div>';
		this.html['link_iframe'] = '<iframe src="{{url}}" style="height:0;width:0;border:0"></iframe>';
		this.html['bnnr_glue'] = '<span class="begun_adv_bullit"> &#149; </span>';
		this.html['bnnr_phone'] = '\
<span class="begun_adv_phone"><b class="p0"></b><b class="p1"></b><b class="p2"></b><b class="p4"><b class="p3"></b></b><b class="p5"></b><b class="p6"><b class="p1"></b></b><b class="p7"></b><b class="p8"></b></span>\
';
		this.html['bnnr_card'] = '\
<span class="begun_adv_phone_wrapper">{{phone}}<span class="begun_adv_card"><a target="_blank" href="{{url}}" class="snap_noshots">{{card_text}}</a></span></span>\
';
		this.html['bnnr_ppcall'] = '\
<span class="begun_adv_phone_wrapper">{{phone}}<a href="javascript:void(0);" class="snap_noshots" onClick="showEnterForm({{banner_index}}, this, event)"><span class="begun_adv_card">{{ppcall_text}}</span></a></span>\
';
		this.html['bnnr_domain'] = '\
<span class="begun_adv_contact"><a class="snap_noshots" target="_blank" href="{{url}}" onmouseover="status=\'http://{{domain}}/\';return true" onmouseout="status=\'\';return true" title="{{domain}}">{{domain}}</a></span>\
';
		this.html['bnnr_geo'] = '\
<span class="begun_adv_city"><a class="snap_noshots" target="_blank" href="{{url}}" onmouseover="status=\'http://{{domain}}/\';return true" onmouseout="status=\'\';return true" title="{{domain}}">{{geo}}</a></span>\
';
		this.html['bnnr_thumb'] = '\
<a href="{{url}}" class="begun_thumb snap_noshots" target="_blank"><img src="{{src}}" width="56" height="42" alt=""/></a>\
';
		this.html['banner_vertical'] = '\
<tr>\
<td class="begun_adv_cell">\
{{thumb}}\
<div class="begun_adv_block">\
<div class="begun_adv_title"><a class="snap_noshots" target="_blank" href="{{url}}" onmouseover="status=\'http://{{domain}}/\';return true" onmouseout="status=\'\';return true" title="{{domain}}">{{title}}</a></div>\
<div class="begun_adv_text"><a class="snap_noshots" target="_blank" href="{{url}}" onmouseover="status=\'http://{{domain}}/\';return true" onmouseout="status=\'\';return true" title="{{domain}}">{{descr}}</a></div>\
<div class="begun_adv_contact">{{contact}}</div>\
</div>\
</td>\
</tr>\
';
		this.html['banner_200x300'] = this.html['banner_240x400'] = this.html['banner_120x600'] = this.html['banner_160x600'] = this.html['banner_vertical'];
		this.html['banner_horizontal'] = '\
<td class="begun_adv_cell" style="width:{{banner_width}} !important">\
{{thumb}}\
<div class="begun_adv_block">\
<div class="begun_adv_title"><a class="snap_noshots" target="_blank" href="{{url}}" onmouseover="status=\'http://{{domain}}/\';return true" onmouseout="status=\'\';return true" title="{{domain}}">{{title}}</a></div>\
<div class="begun_adv_text"><a class="snap_noshots" target="_blank" href="{{url}}" onmouseover="status=\'http://{{domain}}/\';return true" onmouseout="status=\'\';return true" title="{{domain}}">{{descr}}</a></div>\
<div class="begun_adv_contact">{{contact}}</div>\
</div>\
</td>\
';
		this.html['banner_468x60'] = this.html['banner_728x90'] = this.html['banner_horizontal'];
		this.html['blck_hover'] = ' onmouseover="Begun.Autocontext.blockHover(this, {backgroundColor: \'{{bgcolor_over}}\', borderColor: \'{{brdcolor_over}}\'})" onmouseout="Begun.Autocontext.blockHover(this, {backgroundColor: \'{{bgcolor_out}}\', borderColor: \'{{brdcolor_out}}\'})"';
		this.html['block_vertical'] = '\
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
		this.html['block_horizontal'] = '\
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
		this.html['block_468x60'] = '\
<div class="begun_adv begun_adv_fix begun_adv_fix_hor begun_adv_468x60"{{block_hover}}>\
<div class="begun_adv_common {{block_scroll_class}}" id="{{scroll_div_id}}">\
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
		this.html['block_728x90'] = '\
<div class="begun_adv begun_adv_fix begun_adv_fix_hor begun_adv_728x90"{{block_hover}}>\
<div class="begun_adv_common {{block_scroll_class}}" id="{{scroll_div_id}}">\
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
		this.html['block_200x300'] = '\
<div class="begun_adv begun_adv_fix begun_adv_fix_ver begun_adv_200x300"{{block_hover}}>\
<table class="begun_adv_sys"><tr>\
<td class="begun_adv_sys_logo"><div><a href="{{begun_url}}" target="_blank" class="snap_noshots">begun</a></div></td>\
<td class="begun_adv_sys_sign_up" style="display:{{place_here_display}}"><div><a href="{{place_here_url}}" target="_blank" class="snap_noshots">{{place_here_text}}</a></div></td>\
</tr></table>\
<div class="begun_adv_common {{block_scroll_class}}" id="{{scroll_div_id}}">\
<table class="begun_adv_table {{css_thumbnails}}" id="{{scroll_table_id}}">\
{{banners}}\
</table>\
</div>\
</div>\
';
		this.html['block_240x400'] = '\
<div class="begun_adv begun_adv_fix begun_adv_fix_ver begun_adv_240x400"{{block_hover}}>\
<table class="begun_adv_sys"><tr>\
<td class="begun_adv_sys_logo"><div><a href="{{begun_url}}" target="_blank" class="snap_noshots">begun</a></div></td>\
<td class="begun_adv_sys_sign_up" style="display:{{place_here_display}}"><div><a href="{{place_here_url}}" target="_blank" class="snap_noshots">{{place_here_text}}</a></div></td>\
</tr></table>\
<div class="begun_adv_common {{block_scroll_class}}" id="{{scroll_div_id}}">\
<table class="begun_adv_table {{css_thumbnails}}" id="{{scroll_table_id}}">\
{{banners}}\
</table>\
</div>\
<div class="begun_adv_all" style="display:{{all_banners_display}}"><a href="{{all_banners_url}}" target="_blank" class="snap_noshots">{{all_banners_text}}</a></div>\
</div>\
';
		this.html['block_120x600'] = '\
<div class="begun_adv begun_adv_fix begun_adv_fix_ver begun_adv_120x600"{{block_hover}}>\
<table class="begun_adv_sys"><tr>\
<td class="begun_adv_sys_logo"><div><a href="{{begun_url}}" target="_blank" class="snap_noshots">begun</a></div></td>\
<td class="begun_adv_sys_sign_up" style="display:{{place_here_display}}"><div><a href="{{place_here_url}}" target="_blank" class="snap_noshots">{{place_here_text}}</a></div></td>\
</tr></table>\
<div class="begun_adv_common {{block_scroll_class}}" id="{{scroll_div_id}}">\
<table class="begun_adv_table {{css_thumbnails}}" id="{{scroll_table_id}}">\
{{banners}}\
</table>\
</div>\
<div class="begun_adv_all" style="display:{{all_banners_display}}"><a href="{{all_banners_url}}" target="_blank" class="snap_noshots">{{all_banners_text}}</a></div>\
</div>\
';
		this.html['block_160x600'] = '\
<div class="begun_adv begun_adv_fix begun_adv_fix_ver begun_adv_160x600"{{block_hover}}>\
<table class="begun_adv_sys"><tr>\
<td class="begun_adv_sys_logo"><div><a href="{{begun_url}}" target="_blank" class="snap_noshots">begun</a></div></td>\
<td class="begun_adv_sys_sign_up" style="display:{{place_here_display}}"><div><a href="{{place_here_url}}" target="_blank" class="snap_noshots">{{place_here_text}}</a></div></td>\
</tr></table>\
<div class="begun_adv_common {{block_scroll_class}}" id="{{scroll_div_id}}">\
<table class="begun_adv_table {{css_thumbnails}}" id="{{scroll_table_id}}">\
{{banners}}\
</table>\
</div>\
<div class="begun_adv_all" style="display:{{all_banners_display}}"><a href="{{all_banners_url}}" target="_blank" class="snap_noshots">{{all_banners_text}}</a></div>\
</div>\
';
		this.getCSS = function(type){
			return this.css[type];
		};
		this.getHTML = function(type){
			return this.html[type];
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
						Begun.extend(this[type], tpl);
						j++;
					}
				}
				i++;
			}
		}
	};

	/**
	 * init autocontext once
	 */
	(function(){
		var ac = Begun.Autocontext;
		if (typeof window.begun_urls != 'undefined') ac.setURLs(window.begun_urls);
		ac.setOptions({max_banners_count: 50, max_blocks_count: 10, scroll_banners_coef: 2});
		ac.Tpls.addTpls();
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
			if (typeof window.begunAds == 'undefined'){
				window.setTimeout(arguments.callee, 1);
			}else{
				if (!ac.feed_rendered){
					ac.onFeedBlocksReady();
				}
			}
		});
	})();
}

/**
 * init current block
 */
Begun.Autocontext.initCurrentBlock();
