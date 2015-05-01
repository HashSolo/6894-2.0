/**
* Asynchronous network connection tools.
*/

PAYPAL.namespace("util");


/**
* Asynchronous network connection tools.
* @requires YAHOO.util.Connect
*/
PAYPAL.util.Connect = {


	/**
	* Submit a form asynchronously to the server and parse the response.
	* The config object properties (all optional):
	*	action: Submit URL (will grab this from the form if left out)
	*	callback: The function that will be called when the XHR call completes (see below).
	*	method: Submit method, GET or POST (will grab this from the form if left out)
	*	onfail: The function that will be called when the XHR call fails.
	*	scope: The scope of the connection object - passed directly to the YAHOO connect object.
	*	type: The type of page to return, "minipage" or "" NOTE: "minipage" NEEDS TO BE SET EXPLICITLY
	*	submitter: The button that submitted the form.  This name/value will be submitted with the request.
	*
	* Callback function:
	*	This function whill be called when submitForm() completes the XHR request or an error occurrs:	
	*
	*	callbackFunc(data, request)
	*		data: The JSON object returned from the server
	*		request: The XHR request object.
	*
	*	If the data parameter is NULL, then the XHR connection had an error.  If both parameters
	* 	are null, the submitForm function had an exception error.
	*
	* @param {DomNode} form The form to submit.
	* @param {Object} config The configuration object. (callback, method, action, type, onfail, scope)
	* @param {Object} fields Extra form fields to submit.  
	*/
	submitForm : function(form, config, fields){
		var request = null;
		try{
			if(typeof form == "string"){
				form = document.getElementById(form);
			}
			
			if(!form){
				return;
			}
			
			// Set default values
			fields = (fields) ? fields : {}
			config.action = (config.action) ? config.action : form.action;
			config.method = (config.method) ? config.method : form.method;
			config.type = (config.type) ? config.type : "minipage";
			
			if(!config.method || config.method.toUpperCase() != "GET"){
				config.method = "POST";
			}
			
			
			/* 
			* Get all form values
			*/
			var query = [];
			var field, name, value, options;
			for(var i = 0; i < form.elements.length; i++){
				field = form.elements[i];
				name = encodeURIComponent(field.name);
				
				if(field.disabled || typeof(field.type) == 'undefined'){
					continue;
				}
				
				switch(field.type.toLowerCase()){
					
					case "radio":
					case "checkbox":
						if(field.checked){
							value = field.value || "on";
							query[query.length] = name +"="+ encodeURIComponent(value);
						}
					break;
					case "select-one":
						value = "";
						
						if(field.selectedIndex > -1){
							value = field.options[field.selectedIndex].value;
						}
						query[query.length] = name +"="+ encodeURIComponent(value);
					break;
					case "select-multiple":
						options = field.options;
						for(var n = 0; n < options.length; n++){
							if(options[n].selected){
								query[query.length] = name +"="+ encodeURIComponent(options[n].value);
							}
						}
					break;
					case "text":
					case "textarea":
					case "hidden":
					case "password":
						query[query.length] = name +"="+ encodeURIComponent(field.value);
					
					default: // button, submit, etc.
					
				}
			}
				
			// Minipage
			if(config.type == "minipage"){
				query[query.length] =  "view_requested=MiniPage";
			}
			
			// Cache busting
			query[query.length] = "pui_rand="+ Math.random();
			
			// Submitter Button
			if(config.submitter){
				fields[config.submitter.name] = config.submitter.value
			}
			
			// Extra fields
			var name;
			for(name in fields){
				query[query.length] = encodeURIComponent(name) +"="+ encodeURIComponent(fields[name]);
			}
			
			// Add query separators (this is faster than string concatenation)
			query = query.join("&");
			
			// Set URL
			if(config.method.toUpperCase() == "GET"){
				if(config.action.indexOf("?")){
					config.action += "?";
				}
				config.action += query;
				query = "";
			}
			
			
			/*
			* Async Request
			*/
			request = this.send(config.action, { method : config.method, callback : config.callback, query : query, onfail : config.onfail, scope : config.scope, type : config.type })
		} catch(e){			
			if(config.callback){
				config.callback(false, false, e);
			}
		}
		
		return request
	},
	
	
	/**
	* Make an Ajax request from a hijacked link.
	* This will take the url in the HREF and send an XHR request (GET by default) to it.
	*
	* The config object properties (all optional):
	*	callback: The function that will be called when the XHR call completes (see below).
	* 	method: Submit method, GET or POST (will grab this from the form if left out)
	*	type: The type of page to return, "minipage" or ""
	*
	* @param {String|DomNode} anchor The anchor ID or DOM node
	* @param {Object} config The configuration object
	*/
	sendFromLink : function(anchor, config){
		var request = null;
		
		try{
			if(typeof anchor == "string"){
				anchor = document.getElementById(anchor);
			}
			
			if(!anchor){
				return;
			}
			
			// Set default values
			config.method = (config.method) ? config.method : "GET";
			config.type = (config.type) ? config.type : "";
			
			// Get url & query string
			var url = anchor.href;
			var query = "";
			var delim = 0;
			if((delim = url.indexOf("?")) > -1){
				query = url.substring(delim + 1);
				url = url.substring(0, delim);
			}
			
			// Minipage
			if(config.type == "minipage"){
				query += "&view_requested=MiniPage";
			}
			
			// Set URL
			query += "&pui_rand="+ Math.random();
			if(config.method.toUpperCase() == "GET"){
				url += "?"+ query;
				query = "";
			}
			
			
			/*
			* Async Request
			*/
			request = this.send(url, { method : config.method, callback : config.callback, query : query, onfail : config.onfail, scope : config.scope, type : config.type });
		}catch(e){		
			if(config.callback){
				config.callback(false, false, e);
			}
		}
		
		return request;
	},
	
	/**
	* Send an asynchronous request to the server and automatically parse the response JSON.
	* This is a Paypal specific wrapper to the YUI Connect.asyncRequest() method which automatically 
	* takes care of JSON parsing, and appropriate backend response actions.
	*/
	send : function(url, config){
		var request = null;
		
		// Set default values
		config.method = (config.method) ? config.method : "GET";
		config.type = (config.type) ? config.type : "";
		config.query = (config.query) ? config.query : "";
		
		// Minipage
		if(config.type == "minipage"){
			
			if(config.method == "GET"){
				url += (url.indexOf("?") > -1) ? "&" : "?";
				url += "view_requested=MiniPage";
			}
			else{
				if(config.query.length > 0 && config.query.slice(-1) != "&"){
					config.query += "&";
				}
				config.query += "view_requested=MiniPage"
			}
			
		}
				
		/*
		* Async Request
		*/
		try{
		
			var submitCallback = {
				success : function(request){
					if(!config.callback){
						return;
					}
					
					try{
					
						// Parse data
						var data = request.responseText;
						if(data.substring(0, 10) == "while (1);"){
							data = data.substring(10);
						} else { // this is not an expected response
							if(!config.onfail){
								if (!config.callback){
									return;
								}
								if (!config.scope) {
									config.callback(false, request, false);
								} else {
									config.callback.apply(config.scope, [false, request, false]);
								}
							} else if (!config.scope) {
								config.onfail(false, request, false);
							} else {
								config.onfail.apply(config.scope, [false, request, false]);
							}
							return;
						}
						data = '('+ data + ')';
						data = eval(data);
						
						// Handle redirects
						if(data.type == "redirect"){ // GET
							document.location.href = data.url;
							return;
						}
						else if(data.type == "auto-submit"){ // POST
							var div = document.createElement("div");
							div.innerHTML = data.html;
							document.body.appendChild(div);
							var form = div.getElementsByTagName("form")[0];
							form.submit();
							return;
						}
					} catch(e){
						if (!config.scope) {
							config.callback(false, false, e);
						} else {
							config.callback.apply(config.scope, [false, false, e]);
						}
					}
					
					if (!config.scope) {
						config.callback(data, request, false);
					} else {
						config.callback.apply(config.scope, [data, request, false]);
					}
				},
				failure : function(request){
					if(!config.onfail){
						if (!config.callback){
							return;
						}
						if (!config.scope) {
							config.callback(false, request, false);
						} else {
							config.callback.apply(config.scope, [false, request, false]);
						}
					} else if (!config.scope) {
						config.onfail(false, request, false);
					} else {
						config.onfail.apply(config.scope, [false, request, false]);
					}
				}
				
			}
			request = YAHOO.util.Connect.asyncRequest(config.method, url, submitCallback, config.query); 
			
		} catch(e){			
			if(config.callback){
				if (!config.scope) {
					config.callback(false, false, e);
				} else {
					config.callback.apply(config.scope, [false, false, e]);
				}
			}
		}
		
		return request;
	}
}
