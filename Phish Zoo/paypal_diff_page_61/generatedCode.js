/**
 * Button Designer 2.0 - Code Generated page
 * Created: 02/14/2008
 */

PAYPAL.namespace("Merchant.ButtonDesigner");

/**
 * CodeGenerated object for Button Designer
 */
PAYPAL.Merchant.ButtonDesigner.CodeGenerated = {
	
	lightboxMsgShown: false,
	
	init: function() {
		
		var CG = PAYPAL.Merchant.ButtonDesigner.CodeGenerated;
		
		// Attach onClick listener to submitPage links
		var submitLinks = YAHOO.util.Dom.getElementsByClassName('submitPage');
		for (i = 0; i < submitLinks.length; i++) {
			YAHOO.util.Event.on(submitLinks[i], 'click', CG.submitLink);
		}
		
		// Attach onClick listener to 'Select Code' button and textareas
		var btnSelectCode = YAHOO.util.Dom.getElementsByClassName('btnSelectCode');
		var textAreas = YAHOO.util.Dom.getElementsByClassName('textarea');
		if (btnSelectCode && btnSelectCode.length > 0) {
			for (i = 0; i < btnSelectCode.length; i++) {
				YAHOO.util.Event.on(btnSelectCode[i], 'click', CG.selectCode, this);
				YAHOO.util.Event.on(textAreas[i], 'click', CG.selectCode, this);
			}
		}
		
		// Initialize tabs
		var tabView = new YAHOO.widget.TabView('generatedCodeTabs');

		// Initialize button preview section
		var previewImageSection = YAHOO.util.Dom.getElementsByClassName('previewImageSection', 'p')[0];
		if (previewImageSection) {
			PAYPAL.widget.hideShow.show(previewImageSection);
		}
		
		// Attach onClick listener to Code protection links 
		var protectLinks = YAHOO.util.Dom.getElementsByClassName('protect');
		var unprotectLinks = YAHOO.util.Dom.getElementsByClassName('unprotect');
		if (protectLinks && protectLinks.length > 0) {
			for (i = 0; i < protectLinks.length; i++ ) {
				YAHOO.util.Event.on(protectLinks[i], 'click', CG.toggleCodeProtectionLinks);
				YAHOO.util.Event.on(unprotectLinks[i], 'click', CG.toggleCodeProtectionLinks);
			}
		}

		// disable tabs
		var disabledTabs = YAHOO.util.Dom.getElementsByClassName('disabled', 'li');
		if (disabledTabs) {	
			for (i = 0; i < disabledTabs.length; i++) {
				YAHOO.util.Event.removeListener(disabledTabs[i], 'click'); 
			}
		}
		
		// Readonly inputs
		var readOnlyLabel = YAHOO.util.Dom.getElementsByClassName("readOnlyLabel");
		if (readOnlyLabel) {
			YAHOO.util.Event.on(readOnlyLabel, 'focus', CG.makeFieldReadOnly, this);
		}
		
		// show lightbox
		CG.showLightboxMessage();
	},
	
	showLightboxMessage: function() {
		var CG = PAYPAL.Merchant.ButtonDesigner.CodeGenerated;

		if (!CG.lightboxMsgShown) {
			var buttonType = document.getElementById('buttonTypeHidden').value;
			var subButtonType = document.getElementById('subButtonTypeHidden').value;
			var hasShipping = document.getElementById('buttonHasShipping').value;
			var textAreas = document.getElementsByTagName('textarea');
	
			var lightbox = new PAYPAL.util.Lightbox('cgLightbox');

			if (document.getElementById('cgLightbox')) {
				for (i = 0; i < textAreas.length; i++) {
					YAHOO.util.Dom.setStyle(textAreas[i], 'overflow', 'hidden');
				}
				lightbox.show();
				CG.lightboxMsgShown = true;
			
				// reset textareas overflow when lightbox is closed
				var lightboxCloseSecure = YAHOO.util.Dom.getElementsByClassName('close', 'div', 'cgLightbox')[0];
				YAHOO.util.Event.on(lightboxCloseSecure, 'click', CG.resetTextAreas);
				YAHOO.util.Event.on(document, 'keydown', function(e) {
					var keyPressed = e.charCode || e.keyCode;
					if (keyPressed == 27) {
						CG.resetTextAreas();
					}
				});
			}
		}
	},

	resetTextAreas: function(e) {
		var textAreas = document.getElementsByTagName('textarea');
		for (i = 0; i < textAreas.length; i++) {
			YAHOO.util.Dom.setStyle(textAreas[i], 'overflow', 'auto');
		}
	},
	
	submitLink: function(e) {
		YAHOO.util.Event.preventDefault(e);
		var form = document.getElementById('codeGeneratedForm');
		var fakeField = document.getElementById('fakeSubmit');
		var onBoardingCmd = document.getElementById('onboarding_cmd');

		// Action links
		var actionLinks = ['editButton','createSimilar','viewCart','unsubscribe'];
		for (i = 0; i < actionLinks.length; i++) {
			var linkClass = actionLinks[i] + 'Link';					
			var hiddenInput = document.getElementById(actionLinks[i]);
			hiddenInput.disabled = true;
			if (YAHOO.util.Dom.hasClass(this, linkClass)) {
				hiddenInput.disabled = false;
			}
		}
		
		fakeField.name = getCmd(this.href);
		fakeField.value = this.innerHTML;

		if (YAHOO.util.Dom.hasClass(this, 'loginSubmit')) {
			onBoardingCmd.value = 'login';
		} else if (YAHOO.util.Dom.hasClass(this, 'signupSubmit')) {
			onBoardingCmd.value = 'signup';
		} else if (YAHOO.util.Dom.hasClass(this, 'upgradeSubmit')) {
			onBoardingCmd.value = 'upgrade';
		}
		
		form.submit();	

		function getCmd(str) {
			if ( str.match("cmd") != null ) {
				str = str.split('?')[1];
				str = str.split('&');
				for ( var i=0; i < str.length; i++ ) {
					if ( str[i].match("cmd") != null ) {
						return str[i].split('=')[1];
					}
				}
				return false;
			}
			else {
				return false;
			}
		}
		
		return false;
	},
	
	/**
	 * Handles click events for textarea and "Select Code" button
	 */
	selectCode: function() {
		var parentNode = this.parentNode.parentNode;
		var textArea = YAHOO.util.Dom.getElementsByClassName('textarea', 'textarea', parentNode)[0];
		
		if (typeof(this) == 'TEXTAREA') {
			this.focus();
			this.select();
		} else {
			textArea.focus();
			textArea.select();
		}
	},
	
	/**
	 * Uses YAHOO.util.Dom.hasClass but accepts an array of class names.
	 */
	hasClass: function(element, classNames) {
		if (isArray(classNames)) {
			for (i = 0; classNames.length; i++) {
				if (YAHOO.util.Dom.hasClass(element, classNames[i])) {
					return true;
				}
			}
		} else {
			return YAHOO.utl.Dom.hasClass(element, classNames);
		}
		return false;
	},
	
	/** 
	 * Toggle function for showing Protected/Unprotected code
	 */
	toggleCodeProtectionLinks: function(e) {
		YAHOO.util.Event.preventDefault(e);
		var elTarget = YAHOO.util.Event.getTarget(e);
		var elTarget = (elTarget.nodeName.toUpperCase() == 'SPAN') ? elTarget.parentNode : elTarget;
		var tabContent = YAHOO.util.Dom.getElementsByClassName('content');

		// get active tab container
		for (i = 0; i < tabContent.length; i++) {
			if (YAHOO.util.Dom.getStyle(tabContent[i], 'display') == 'block') {
				var activeTab = tabContent[i];
			}
		}

		var textArea = activeTab.getElementsByTagName('textarea')[0].id;
		var strCodeType = textArea.replace("generated", "");
		
		// define vars for protect case
		if (YAHOO.util.Dom.hasClass(elTarget, "protect")) {
			var link = YAHOO.util.Dom.getElementsByClassName('unprotect', 'a', activeTab)[0];
			var hiddenCode = document.getElementById('hiddenProtected' + strCodeType);
			var generatedCode = document.getElementById('generated' + strCodeType);
		}
		// define vars for unprotect case
		else if (YAHOO.util.Dom.hasClass(elTarget, "unprotect")) {
			var link = YAHOO.util.Dom.getElementsByClassName('protect', 'a', activeTab)[0];
			var hiddenCode = document.getElementById('hiddenUnprotected' + strCodeType);
			var generatedCode = document.getElementById('generated' + strCodeType);
		}
		
		// replace textarea code
		generatedCode.value = hiddenCode.value;

		// hide/show code protection links
		PAYPAL.widget.hideShow.hide(elTarget);
		PAYPAL.widget.hideShow.show(link);
	},

	/**
	 * Handles fully disabling tabs
	 */
	disableTab: function(e) {
		var event = e || window.event;
		if (event.preventDefault) {
			YAHOO.util.Event.preventDefault(e);
		} else {
			event.returnValue = false;
		}
		return false;
	}, 
	
	/**
	 * Makes an input field behave as a read-only
	 */
	makeFieldReadOnly: function() {
		this.blur();
	}

};

PAYPAL.util.Event.onDomReady(PAYPAL.Merchant.ButtonDesigner.CodeGenerated.init);
