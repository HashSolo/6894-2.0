if (typeof YAHOO != "undefined") {
        PAYPAL.namespace("pplite");
        PAYPAL.pplite = {
                activeWidget: null,
		isWidgetOpen: null,
		widgetHeight: 140,
		initWidget: function(obj) {
			activeWidget = null;
			YUE.addListener('widgetOpener', 'click', PAYPAL.pplite.showWidget);
			YUE.addListener('closeWidget', 'click', PAYPAL.pplite.animateClose);
			document.getElementById("rosetta").style.visibility="visible";
		},
		closeWidgetOnWindowClick : function (event) {
			event = event || window.event;
			target = YUE.getTarget(event);

			if (YUD.isAncestor(document.getElementById("widgetOpener"), target)) {
				YUD.stopPropagation(evt);
			}
			if (PAYPAL.pplite.activeWidget && !YUD.isAncestor(PAYPAL.pplite.activeWidget.element, target)) {
				PAYPAL.pplite.animateClose();
			}
		},	
		showWidget : function(evt) {
			if (evt) {
				YUE.stopPropagation(evt);	
				YUE.preventDefault(evt);
			}
			if (PAYPAL.pplite.isWidgetOpen) {PAYPAL.pplite.animateClose(); return; }
			if (!PAYPAL.pplite.activeWidget) {
				var widget = document.getElementById("pplite");
				YAHOO.widget.Module.CSS_HEADER = "header";
        	                YAHOO.widget.Module.CSS_BODY = "body";
                	        YAHOO.widget.Module.CSS_FOOTER = "footer"; 
				if (PAYPAL.pplite.lightbox) {
					var pplite_c = document.getElementById("pplite_c");
					pplite_c.style.left = "";
					pplite_c.style.top = "";
    					pplite_c.removeAttribute("id");
				}
                                widget.parentNode.removeChild(widget);
                                document.body.appendChild(widget);
				var config = {
					close:false,
					draggable:false,
					underlay:"none",
					context:['rosetta','tr','br']
				};
				PAYPAL.pplite.activeWidget = new YAHOO.widget.Panel(widget, config );
				if (!document.getElementById("countryLang")) {
					PAYPAL.pplite.populateCountryLang();
				}
				PAYPAL.pplite.activeWidget.render();
                        }
			PAYPAL.pplite.animateOpen();
			PAYPAL.pplite.activeWidget.show();
			PAYPAL.pplite.isLightbox = false;
			YUE.addListener(window, "resize", PAYPAL.pplite.repositionWidget);
		},
                animateOpen : function() {
                        var attributes = {height: {from:0, to: PAYPAL.pplite.widgetHeight}, paddingTop: {from:0, to:20} };
                        var anim = new YAHOO.util.Anim('pplite',attributes,0.5,YAHOO.util.Easing.backOut);
			anim.animate();
			anim.onComplete.subscribe(
				function() {
					PAYPAL.pplite.isWidgetOpen = true;
					document.getElementById("pplite").style.padding="20px 10px 0pt";
					YUE.addListener (document, "mouseup", PAYPAL.pplite.closeWidgetOnWindowClick);
				}
			);
                },
                animateClose : function() {
                        var attributes = {height: {from:PAYPAL.pplite.widgetHeight, to: 0}, paddingTop: {from:20, to:0} };
                        var anim = new YAHOO.util.Anim('pplite',attributes,0.5,YAHOO.util.Easing.backOut);
                        anim.animate();
			anim.onComplete.subscribe(
				function() {
					if (PAYPAL.pplite.activeWidget) {
						PAYPAL.pplite.activeWidget.hide();
					}
				}
			);
			PAYPAL.pplite.isWidgetOpen = false;
			document.getElementById('pplite').style.paddingBottom = "0pt";
			YUE.removeListener(document, 'mouseup');
                },
		populateCountryLang : function() {
                        var layerWidget = document.getElementById("pplite");
			if (YUD.hasClass(layerWidget, "loading")) {
				return false;
			}
                        YUD.addClass(layerWidget, "loading");
                        PAYPAL.util.Connect.sendFromLink('widgetOpener', {'type': 'minipage','callback':  function(data) {
                                 if (data) {
                                        var divWidget = document.createElement("div"); 
                                        divWidget.id = "countryLang";
                                        divWidget.innerHTML = data.html;
                                        var nodeParent = document.getElementById("pplite");
                                        nodeParent.appendChild(divWidget);
                                        var btnNonJS = document.getElementById("btnNonJS");
                                        YUD.addClass(btnNonJS, "accessAid");
                                        YUD.removeClass(layerWidget, "loading");
                                	YUE.addListener("ppliteCountry", "change", PAYPAL.pplite.populateLang);
					if (PAYPAL.pplite.isLightbox) {
						document.getElementById("ppliteLang").disabled = true;
						document.getElementById("ppliteSubmit").disabled = true;
						YUD.addClass("ppliteSubmit", "disabled");
						document.getElementById("ppliteCountry").selectedIndex=0;
					} else {
						YUD.removeClass('ppliteSubmit', 'primary');	
						YUE.addListener (document, "mouseup", PAYPAL.pplite.closeWidgetOnWindowClick);
					}

					if (PAYPAL.pplite.isLightbox) {
						document.getElementById("ppliteCountry").selectedIndex=0;
					}
		               	}
			}});
		},
		populateLang : function(evt) {
			if (evt) {
				YUE.stopPropagation(evt);
				YUE.preventDefault(evt);
			}

			if (document.getElementById("ppliteLang") || document.getElementById("ppliteCountry").value == "") {
				document.getElementById("ppliteLang").disabled = true;
				document.getElementById("ppliteSubmit").disabled = true;	
				YUD.addClass("ppliteSubmit", "disabled");
			}

			if (document.getElementById("ppliteCountry").value == "")
				return false;

			var ajaxUrl = document.getElementById("countrySelect").href;
			posCountry = ajaxUrl.indexOf("&country.x=");
			if (posCountry > 0) {
				document.getElementById("countrySelect").href = ajaxUrl.substring(0, posCountry) + "&country.x=" + document.getElementById("ppliteCountry").value + "&get_lang_list=true";
			} else {
				document.getElementById("countrySelect").href += "&country.x=" + document.getElementById("ppliteCountry").value + "&get_lang_list=true";
			}

			PAYPAL.util.Connect.sendFromLink('countrySelect', {'type':'minipage','callback': function(data) {
				var langContainer = document.getElementById("langContainer");
				var nodeParent = langContainer.parentNode;
				nodeParent.removeChild(langContainer);
				langContainer.innerHTML = data.html;
				nodeParent.appendChild(langContainer);
				document.getElementById("ppliteSubmit").disabled = false;
				YUD.removeClass("ppliteSubmit", "disabled");
			}});
		},
		repositionWidget : function(evt) {
			if (evt) {
				YUE.stopPropagation(evt);
				YUE.preventDefault(evt);
			}
                        var widget = document.getElementById("pplite");
			var config = {
				close:false,
				draggable:false,
				underlay:"none",
				context:['rosetta','tr','br']
			};
                        PAYPAL.pplite.activeWidget = new YAHOO.widget.Panel(widget, config );

			PAYPAL.pplite.activeWidget.render();
			if (PAYPAL.pplite.isWidgetOpen) {
				PAYPAL.pplite.activeWidget.show();
			}else {
				PAYPAL.pplite.activeWidget.hide();
			}
		},

		showLightBox : function(evt) {
			var divPPLite = document.getElementById("pplite");
			divPPLite.parentNode.removeChild(divPPLite);
			document.body.appendChild(divPPLite);
			PAYPAL.pplite.isLightbox = true;
			PAYPAL.pplite.lightbox = new PAYPAL.util.Lightbox("pplite", {close:false, keylisteners:false});
			activeWidget = PAYPAL.pplite.lightbox;
			PAYPAL.pplite.lightbox.show();
			YUE.addListener("pplite_c", 'keyup', PAYPAL.pplite.stopEscKey);
			YUE.addListener("pplite_mask", 'keyup', PAYPAL.pplite.stopEscKey);
			YUE.removeListener(window, 'load');	
		},
		stopEscKey : function(evt) {
                                                evt = evt || window.event;
                                                var iKey = evt.keyCode || evt.which;
                                                if (iKey == 27) {
                                                        YUE.stopPropagation(evt);
                                                        YUE.preventDefault(evt);
							return false;
                                                }
		}
	}
	YUE.addListener(window, "load", PAYPAL.pplite.initWidget);
}
