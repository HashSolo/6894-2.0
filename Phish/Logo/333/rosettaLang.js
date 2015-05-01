PAYPAL.widget.Rosetta = {
	
	hideForm : function () {
		var hd = document.getElementsByTagName('head');
		if (hd.length > 0) {
			var ss = document.createElement('style');
			var rs = 'form#rosetta {display:none;}';
			ss.setAttribute('type', 'text/css');
			if (ss.styleSheet) {
				ss.styleSheet.cssText = rs;
			} else {
				ss.textContent = rs;
			}
			hd = hd[0];
			hd.appendChild(ss);
		}
	},

	showForm : function () {
		var f = document.getElementById('rosetta');
		if (f) {
			f.style.display = 'block';
		}
	},

	ascendsTo : function (node, root) {
		while (node != document) {
			if (node == root) {
				return true;
			}
			node = node.parentNode;
		}
		return false;
	},

	createRosettaDropDown : function () {

		// this should never happen at all if JS is enabled, but just in case
		if (typeof (YAHOO) == "undefined" || typeof (YAHOO.util.Anim) == "undefined" || typeof (YAHOO.util.Easing) == "undefined") {
			PAYPAL.widget.Rosetta.showForm();
			return;
		}

		var Event = YAHOO.util.Event, Dom = YAHOO.util.Dom, Anim = YAHOO.util.Anim;

		// let's generate markup from the drop down
		var rosfrm = Dom.get('rosetta');
		var rossel = Dom.get('rosetta_dropdown');
		
		// if the rosetta dropdown and form are not in, or the rosetta widget is already up, exit
		if (Dom.get('html-rosetta') || !rossel || !rosfrm) {
			return;
		}
		
		// submit the hidden rosetta form
		var submitRosetta = function (idx) {
			rossel.selectedIndex = idx;
			rosfrm.submit();
		}
		
		// create our container
		var target = document.createElement('div');
		target.id = 'html-rosetta-container';
		
		// attach our container
		rosfrm.parentNode.insertBefore(target, rosfrm);

		// start creating the list
		var droot = document.createElement('div');
		droot.id = 'html-rosetta';

		var label = document.createElement('p');
		label.appendChild(document.createTextNode(rossel.options[rossel.selectedIndex].text));

		var dmenu = document.createElement('div');
		var dlist = document.createElement('ul');
		for (var i = 0, n = rossel.options.length; i < n; ++i) {
			var a = document.createElement('a');
			a.setAttribute('href', 'javascript:;');
			a.id = 'locale-' + i;
			a.appendChild(document.createTextNode(rossel.options[i].text));
			var li = document.createElement('li');
			li.appendChild(a);
			dlist.appendChild(li);
		}

		dmenu.appendChild(dlist);
		droot.appendChild(label);
		droot.appendChild(dmenu);

		target.appendChild(droot);

		// animation params
		var region = Dom.getRegion(dlist);
		var aOpen = new Anim(dmenu, {height:{to:parseInt(region.bottom - region.top)}}, 0.3, YAHOO.util.Easing.easeOutStrong);
		var aShut = new Anim(dmenu, {height:{to:0}}, 0.3, YAHOO.util.Easing.easeOutStrong);

		// hook for rolling up the menu when user clicks outside the box
		Event.on(document, 'click', function (e) {
			var t = Event.getTarget(e);
			if (!PAYPAL.widget.Rosetta.ascendsTo(t, droot)) {
				aShut.animate();
			}
		});
	
		// slide out, slide in animation hooks
		Event.on(droot, 'click', function (e) {
			var target = Event.getTarget(e);
			if (target.tagName.toLowerCase() === 'p') {
				var region = Dom.getRegion(dmenu);
				if (parseInt(region.top) == parseInt(region.bottom)) {
					aOpen.animate();
				} else {
					aShut.animate();
				}
			} else if (target.tagName.toLowerCase() === 'a') {
				// if a language entry is clicked, set and submit the ros form
				label.innerHTML = target.innerHTML;
				aShut.animate();
				Event.stopEvent(e);
				submitRosetta(parseInt(target.id.split('-')[1]));
			}
		});
	},

	render : function () {
		// if we have YAHOO but NOT Anim and NOR Easing, let's import
		if (typeof YAHOO != 'undefined' && (typeof YAHOO.util.Anim == 'undefined' || typeof YAHOO.util.Easing == 'undefined')) {
			var hd = document.getElementsByTagName('head');
			if (hd.length > 0) {
				var js = document.createElement('script');
				js.type = 'text/javascript';
				// set onload hook for animation.js
				if (typeof js.onreadystatechange != 'undefined') {
					js.onreadystatechange = function () {
						if (this.readyState == 'loaded' || this.readyState == 'complete') {
							this.onreadystatechange = null;
							PAYPAL.widget.Rosetta.createRosettaDropDown();
						}
					};
				} else {
					js.onload = function () {
						PAYPAL.widget.Rosetta.createRosettaDropDown();
					};
				}
				js.src = '/js/lib/yui/animation.js';
				hd[0].appendChild(js);
			}
		} else {
			PAYPAL.widget.Rosetta.createRosettaDropDown();
		}
	}
};

// hide the form and setup onload hook
(function () {
	PAYPAL.widget.Rosetta.hideForm();
	YAHOO.util.Event.on(window, 'load', PAYPAL.widget.Rosetta.render);
})();
