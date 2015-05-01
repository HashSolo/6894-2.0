window.onload = fnOnLoad;


function sm_xborder(){

}

sm_xborder.prototype = {
		cur_country : 0,
		domain_name : document.domain,
		cur_error : "10",
		req : 0,
		exch_rate : 0,
		status : true,
		cookie_name : "p2p_xborder",
		processReqChange : 0,
		country_variable_xborder : {
				AU:"0.034|0.01",
				CA:"0.029|0",
				C2:"0.039|0.005",
				DE:"0.039|0.02",
				HK:"0.034|0",
				SG:"0.034|0",
				TW:"0.034|0",
				US:"0.039|0.01",
				TH:"0.039|0.005",
				BR:"0.039|0.005",
				MY:"0.039|0.005",
				PH:"0.039|0.005",
				OTHER:"0.039|0.005"
		},
		currency_fixed_amount : {
				USD:"0.30",GBP:"0.20",
				EUR:"0.35",AUD:"0.30",
				CAD:"0.30",CZK:"10.00",
				DKK:"2.60",HKD:"2.35",
				HUF:"90.00",ILS:"1.20",
				JPY:"40.00",MXN:"4.00",
				NZD:"0.45",NOK:"2.80",
				PLN:"1.35",SGD:"0.50",
				SEK:"3.25",CHF:"0.55",
				BRL:"0.40",MYR:"2.00",
				PHP:"15.00",TWD:"10.00",
				THB:"11.00"
		},
		country_currency_default :{
				AR:"USD",AU:"AUD",AT:"EUR",BE:"EUR",BR:"BRL",BG:"USD",CA:"CAD",CL:"USD",
				C2:"USD",CR:"USD",CY:"EUR",CZ:"CZK",DK:"DKK",DO:"USD",
				EC:"USD",EE:"USD",FI:"EUR",FR:"EUR",GF:"EUR",DE:"EUR",GI:"USD",GR:"EUR",
				GP:"EUR",HK:"HKD",HU:"HUF",IS:"USD",IN:"USD",ID:"USD",IE:"EUR",IL:"ILS",
				IT:"EUR",JM:"USD",JP:"JPY",LV:"USD",LI:"CHF",LT:"USD",LU:"EUR",MY:"MYR",
				MT:"EUR",MQ:"EUR",MX:"MXN",NL:"USD",NZ:"NZD",NO:"NOK",PH:"PHP",
				PL:"PLN",PT:"EUR",RE:"EUR",SM:"EUR",SG:"SGD",SK:"EUR",SI:"EUR",
				KR:"USD",ES:"EUR",SE:"SEK",CH:"CHF",TW:"TWD",TH:"THB",
				TR:"USD",AE:"USD",GB:"GBP",UY:"USD",VE:"USD"
		},
		fnChangeIcon: function() {  //fnChangeIcon
				if(this.name=="country_name")
				{
					if(this.value!="")
					{
						YAHOO.util.Dom.removeClass("flganim", "accessAid");
						YAHOO.util.Dom.addClass("cc_icon", "accessAid");
						imgobj = new Image();
						imgobj.src = "https://www.paypalobjects.com/en_US/i/icon/icon_" + this.value + "_22x14.gif";
						document.getElementById("flag_icon").src="https://www.paypalobjects.com/en_US/i/icon/icon_" + this.value + "_22x14.gif"; 
						setTimeout("P2Pobj.fnGetState()",100);
					}
					else{
						YAHOO.util.Dom.addClass("cc_icon", "accessAid");
						document.getElementById("amount_ccode").value="";	
					}
				}
				else
				{
					imgobj = new Image();
					imgobj.src = "https://www.paypalobjects.com/en_US/i/icon/icon_" + document.getElementById("country_name").value + "_22x14.gif";
					document.getElementById("flag_icon").src="https://www.paypalobjects.com/en_US/i/icon/icon_" + document.getElementById("country_name").value + "_22x14.gif"; 
					setTimeout("P2Pobj.fnGetState()",100);
				}
		},
		fnChangeCurrency: function(){ //fnChangeCurrency
				if(this.value!="")
				{
				document.getElementById("amount_ccode").value = P2Pobj.country_currency_default[this.value];
				YAHOO.util.Dom.addClass(document.getElementById("label_country_error"), "accessAid");
				YAHOO.util.Dom.addClass(document.getElementById("label_currency_error"), "accessAid");
				}	
				else
				{
				 document.getElementById("amount_ccode").value = "";
				}	
		},
		fnChangeContent: function(){  //fnChangeContent
				if(this.value!="" && typeof this.value!="undefined"){
					cca_aff = "AFF_"+P2Pobj.cur_country;
					ccb_aff = "AFF_"+this.value;
					cca_wdw = "WDW_"+P2Pobj.cur_country;
					ccb_wdw = "WDW_"+this.value;
					YAHOO.util.Dom.addClass(document.getElementById("AFF_def"), "accessAid");
					YAHOO.util.Dom.addClass(document.getElementById(cca_aff), "accessAid");
					YAHOO.util.Dom.removeClass(document.getElementById(ccb_aff), "accessAid");
					YAHOO.util.Dom.addClass(document.getElementById("WDW_def"), "accessAid");
					YAHOO.util.Dom.removeClass(document.getElementById("wdw"), "accessAid");
					YAHOO.util.Dom.addClass(document.getElementById(cca_wdw), "accessAid");
					YAHOO.util.Dom.removeClass(document.getElementById(ccb_wdw), "accessAid");
					P2Pobj.cur_country = this.value;
				}
				else
				{
					if(P2Pobj.cur_country!="")
					{
					cca_aff = "AFF_"+P2Pobj.cur_country;
					cca_wdw = "WDW_"+P2Pobj.cur_country;
					YAHOO.util.Dom.addClass(cca_aff,"accessAid");
					YAHOO.util.Dom.removeClass("AFF_def", "accessAid");
					YAHOO.util.Dom.addClass("wdw","accessAid");
					YAHOO.util.Dom.addClass(cca_wdw,"accessAid");
					YAHOO.util.Dom.removeClass("WDW_def", "accessAid");
					}
					else
					{
					if(document.getElementById("country_name").value!="")
					{
					cca_wdw = "WDW_"+document.getElementById("country_name").value;
					aff = "AFF_"+document.getElementById("country_name").value;
					YAHOO.util.Dom.addClass("AFF_def", "accessAid");
					YAHOO.util.Dom.addClass("WDW_def", "accessAid");
					YAHOO.util.Dom.removeClass(aff, "accessAid");
					YAHOO.util.Dom.removeClass("wdw", "accessAid");
					YAHOO.util.Dom.removeClass(cca_wdw, "accessAid");
					P2Pobj.cur_country = document.getElementById("country_name").value;
					}
					}
				}
		},
		fnGetState: function (){	 //fnGetState				
				if(document.all){
					if(document.getElementById("flag_icon").readyState == "complete"){
						YAHOO.util.Dom.addClass("flganim", "accessAid");
						YAHOO.util.Dom.removeClass("cc_icon", "accessAid");
					}
					else{
						setTimeout("P2Pobj.fnGetState()",100);	
					}
				}
				else{
					if(imgobj.complete){
						YAHOO.util.Dom.addClass("flganim", "accessAid");
						YAHOO.util.Dom.removeClass("cc_icon", "accessAid");
					} 
					else{
						setTimeout("P2Pobj.fnGetState()",100);
					}
				}		
		},
		fnGetFeetable: function (){    //fnGetFeetable
				
				s_objectID="Calculate";
				
				if(P2Pobj.fnVerification())
				{
					P2Pobj.fnCalculate();
				}
		},
		fnCalculate : function(){ //fnCalculate

				YAHOO.util.Dom.addClass("ex_rate_usd","accessAid");
				YAHOO.util.Dom.addClass("btnCalculate","accessAid");
				YAHOO.util.Dom.addClass("step1b","accessAid");
				YAHOO.util.Dom.removeClass("ex_value","accessAid");
				YAHOO.util.Dom.removeClass("btnReCalculate","accessAid");
				YAHOO.util.Dom.removeClass("step1Continue","accessAid");
				document.getElementById("country_name").disabled = true;
				P2Pobj.fnGetCCode();
				P2Pobj.fnCheckServiceFee();

		},
		fnVerification: function(){ //fnVerification
				var amount = document.getElementById('amount').value;
				var country_id = document.getElementById('country_name').value;
				var currency_id = document.getElementById('amount_ccode').value;
				var error=0;

				/* Regular expression pattern for each input field */
				
				var amountRegxp = /^\.?[0-9]*\.?[0-9]*[.]{0,1}[0-9]{1,2}$/;

				YAHOO.util.Dom.addClass(document.getElementById("label_amount"), "accessAid");
				YAHOO.util.Dom.addClass(document.getElementById("label_amount_less"), "accessAid");
				YAHOO.util.Dom.addClass(document.getElementById("label_country_error"), "accessAid");
				YAHOO.util.Dom.addClass(document.getElementById("label_currency_error"), "accessAid");


				if(country_id == "")
				{
					YAHOO.util.Dom.addClass(document.getElementById("show_country_error"), "error");
					YAHOO.util.Dom.removeClass(document.getElementById("label_country_error"), "accessAid");
					error=1; 
				}


				if(currency_id == "")
				{
					YAHOO.util.Dom.addClass(document.getElementById("show_amount_error"), "error");
					YAHOO.util.Dom.removeClass(document.getElementById("label_currency_error"), "accessAid");
					error=1; 
				}


				if(amount == ""){		
					YAHOO.util.Dom.addClass(document.getElementById("show_amount_error"), "error");
					YAHOO.util.Dom.removeClass(document.getElementById("label_amount"), "accessAid");
					error=1; 

				}		
				else if(amountRegxp.test(amount) != true || amount.charAt(0) === '-' || amount === '0' || amount <= 0.0){
					YAHOO.util.Dom.removeClass(document.getElementById("label_amount_less"), "accessAid");
					YAHOO.util.Dom.addClass(document.getElementById("show_amount_error"), "error");
					error=1; 
				}

				if(error==1)
				{
					YAHOO.util.Dom.removeClass(document.getElementById("messagediv"), "accessAid");return false;
				}
				else
				{
					YAHOO.util.Dom.addClass(document.getElementById("messagediv"), "accessAid");return true;
				}

		},
		fnProcessReqCalc: function(){ //fnProcessReqCalc
		
				if (P2Pobj.req.readyState == 4) {
					var exchange_xml = P2Pobj.req.responseText;	
					var reg_xml = new RegExp(">[0-9,. A-Z]*<");
					var reg_exp = exchange_xml.match(reg_xml);

				if(reg_exp)
				{
				ex_rate = reg_exp[0].substring(1,reg_exp[0].length-1);
				if(P2Pobj.status == false)
				{						
					P2Pobj.ex_rate = ex_rate;
					YAHOO.util.Dom.addClass("animate_exrate", "accessAid");
					camount = document.getElementById("amount").value;
					ccamount = parseFloat(camount);
					
					cin = document.getElementById("amount_ccode").value;
					
					YAHOO.util.Dom.removeClass("btnReCalculate", "accessAid");
					
					if(document.getElementById("amount_ccode").value=="USD")
					{
					document.getElementById("ex_amount").innerHTML = ccamount.toFixed(2) + " " + cin;	
					}
					else
					{
					document.getElementById("ex_amount").innerHTML = ccamount.toFixed(2) + " " + cin +" "+ document.getElementById("eq_cont").innerHTML +" "+ ex_rate;
					}
						YAHOO.util.Dom.removeClass(document.getElementById("send_head"), "accessAid");
						YAHOO.util.Dom.removeClass(document.getElementById("ex_amount"), "accessAid");
					
				}	

				if(!P2Pobj.status)
				{
				YAHOO.util.Dom.addClass("animate", "accessAid");
				YAHOO.util.Dom.removeClass("ex_value", "accessAid");
				YAHOO.util.Dom.removeClass("btnReCalculate", "accessAid");
				}
				else
				{
				P2Pobj.status = false;
				camount = document.getElementById("amount").value;
				cin = document.getElementById("amount_ccode").value;
				P2Pobj.fnLoadXMLDoc("https://"+P2Pobj.domain_name+"/cgi-bin/marketingweb?cmd=_cconversion&amount_in="+camount+"&currency_in="+cin+"&currency_out=USD");
				}
				
				}
				else
				{
				// alert("SERVER ERROR : https://cms.paypal.com/cgi-bin/marketingweb");
				}
				}
		},
		fnLoadXMLDoc : function(url){ //fnLoadXMLDoc

			if (window.XMLHttpRequest)
			{
				P2Pobj.req = new XMLHttpRequest();	
			}	
			else if (window.ActiveXObject) {
				P2Pobj.req = new ActiveXObject("Microsoft.XMLHTTP");
			}			

			if (P2Pobj.req)
			{
				P2Pobj.req.onreadystatechange = P2Pobj.processReqChange;
				P2Pobj.req.open("GET", url, true);
				P2Pobj.req.send(null);
			}

		},
		fnGetCCode : function() {	//fnGetCCode
		
			P2Pobj.status=true;
			YAHOO.util.Dom.addClass("ex_value", "accessAid");
			YAHOO.util.Dom.removeClass("animate", "accessAid");
			P2Pobj.processReqChange = P2Pobj.fnProcessReqCalc;
			P2Pobj.fnLoadXMLDoc("https://"+P2Pobj.domain_name+"/cgi-bin/marketingweb?cmd=_cconversion&amount_in=1&currency_in=USD&currency_out="+document.getElementById('amount_ccode').value);

		},
		fnGetExRate : function(){  //fnGetExRate

			if(this.name=="country_name" && this.value!="")
			{
				cout = P2Pobj.country_currency_default[this.value];
			}	
			else if(this.name=="amount_ccode" && this.vlaue!="")
			{
				cout = this.value;
				YAHOO.util.Dom.addClass(document.getElementById("label_currency_error"), "accessAid");
			}	
			else
			{
				cout = document.getElementById("amount_ccode").value;
			}	
			if(cout!="USD" & cout!=""){
			
				YAHOO.util.Dom.addClass("ex_rate_usd", "accessAid");
				YAHOO.util.Dom.removeClass("animate_exrate", "accessAid");
				P2Pobj.processReqChange = P2Pobj.fnProcessReqExRate;
				P2Pobj.fnLoadXMLDoc("https://"+P2Pobj.domain_name+"/cgi-bin/marketingweb?cmd=_cconversion&amount_in=1&currency_in="+cout+"&currency_out=USD");
				
			}
			else
			{
				YAHOO.util.Dom.addClass("animate_exrate", "accessAid");
				YAHOO.util.Dom.addClass("ex_rate_usd", "accessAid");
			}
			
		},
		fnProcessReqExRate: function() {	//fnProcessReqExRate

			if (P2Pobj.req.readyState == 4)
			{ 
				var exchange_xml = P2Pobj.req.responseText;
				var reg_xml = new RegExp(">[0-9,. A-Z]*<");
				var reg_exp = exchange_xml.match(reg_xml);				
				if(reg_exp)
				{
					ex_rate = reg_exp[0].substring(1,reg_exp[0].length-1);
					P2Pobj.exch_rate = ex_rate;
					YAHOO.util.Dom.addClass("animate_exrate", "accessAid");
					YAHOO.util.Dom.removeClass("ex_rate_usd", "accessAid");
					c_code=document.getElementById("country_name").value;
					if(document.getElementById("amount_ccode").value!="")
					amt_code = document.getElementById("amount_ccode").value;
					else
					amt_code = P2Pobj.country_currency_default[c_code];
					
					document.getElementById("ex_rate_usd").innerHTML =  "1.00 " + amt_code + " = " + ex_rate;
				}				
				else{
					//alert("SERVER ERROR");
				}

			}

		},
		fnCheckServiceFee : function() //fnCheckServiceFee
		{               
			amount_value = parseFloat(document.getElementById("amount").value);		
			country_id = document.getElementById("country_name").value;
			currency_id = document.getElementById("amount_ccode").value;	
			currency_id_lcase="INFO_"+currency_id;

			if(P2Pobj.country_variable_xborder[country_id])
			{
				country_variable = parseFloat(P2Pobj.country_variable_xborder[country_id].split("|")[0]);
				country_xborder = parseFloat(P2Pobj.country_variable_xborder[country_id].split("|")[1]);
			}
			else
			{
				country_variable = parseFloat(P2Pobj.country_variable_xborder["OTHER"].split("|")[0]);
				country_xborder = parseFloat(P2Pobj.country_variable_xborder["OTHER"].split("|")[1]);
			}
			bb_fee = country_xborder * amount_value;
			cc_fee = (country_variable * amount_value) + parseFloat(P2Pobj.currency_fixed_amount[currency_id]);
			bb_total = amount_value + parseFloat(bb_fee);
			cc_total = amount_value + parseFloat(cc_fee);
			document.getElementById("bank_fee").innerHTML =  bb_fee.toFixed(2) +" "+ currency_id; 
			document.getElementById("pp_fee").innerHTML =  bb_fee.toFixed(2) +" "+ currency_id; 
			document.getElementById("credit_fee").innerHTML =  cc_fee.toFixed(2) +" "+ currency_id;
			document.getElementById("pp_amount").innerHTML  =  bb_total.toFixed(2)   +" "+currency_id; 
			document.getElementById("bank_amount").innerHTML = bb_total.toFixed(2)  +" "+currency_id;
			document.getElementById("credit_amount").innerHTML = cc_total.toFixed(2) +" "+currency_id;
			YAHOO.util.Dom.removeClass(document.getElementById("INFO_USD"), "accessAid");
			YAHOO.util.Dom.addClass(document.getElementById(currency_id_lcase), "accessAid");

		},
		fnCheckServiceFeeUSD : function() //fnCheckServiceFeeUSD
		{ 
			s_objectID="change_currency";

			var exr_ch = new Array();
			var exr = new Array();
			exr = P2Pobj.exch_rate.split(" ");
			exr_ch = P2Pobj.ex_rate.split(" ");
			amount_value = parseFloat(exr_ch[0].replace(/,/g,'')); 
			amount = parseFloat(document.getElementById("amount").value);
			country_id = document.getElementById("country_name").value;
			currency_id = document.getElementById("amount_ccode").value;	
			currency_id_lcase="INFO_"+currency_id;
			ex_amt = (amount_value/amount);
			if(P2Pobj.country_variable_xborder[country_id])
			{
			country_variable = parseFloat(P2Pobj.country_variable_xborder[country_id].split("|")[0]);
			country_xborder = parseFloat(P2Pobj.country_variable_xborder[country_id].split("|")[1]);
			}
			else
			{
			country_variable = parseFloat(P2Pobj.country_variable_xborder["OTHER"].split("|")[0]);
			country_xborder = parseFloat(P2Pobj.country_variable_xborder["OTHER"].split("|")[1]);
			}
			
			bb_fee = (country_xborder * amount)*ex_amt;
			cc_fee = ((country_variable * amount) + parseFloat(P2Pobj.currency_fixed_amount[currency_id]))*ex_amt;

			bb_total = amount_value + parseFloat(bb_fee);
			cc_total = amount_value + parseFloat(cc_fee);

			document.getElementById("pp_fee").innerHTML =  bb_fee.toFixed(2)+" USD"; 
			document.getElementById("bank_fee").innerHTML =  bb_fee.toFixed(2)+" USD"; 
			document.getElementById("credit_fee").innerHTML =  cc_fee.toFixed(2)+" USD";
			document.getElementById("pp_amount").innerHTML =  bb_total.toFixed(2) +" USD";
			document.getElementById("bank_amount").innerHTML =  bb_total.toFixed(2) +" USD"; 
			document.getElementById("credit_amount").innerHTML = cc_total.toFixed(2) + " USD";		
			YAHOO.util.Dom.addClass(document.getElementById("INFO_USD"), "accessAid");
			YAHOO.util.Dom.removeClass(document.getElementById(currency_id_lcase), "accessAid");

		},
		fnRecalculate : function()  //fnRecalculate
		{ 
		        s_objectID="change";
		        
		        info_currency = "INFO_"+document.getElementById("amount_ccode").value;
			YAHOO.util.Dom.removeClass("step1b","accessAid");
			YAHOO.util.Dom.addClass("ex_value","accessAid");
			YAHOO.util.Dom.addClass("btnReCalculate","accessAid");
			YAHOO.util.Dom.removeClass("btnCalculate","accessAid");
			YAHOO.util.Dom.addClass("step1","current");
			YAHOO.util.Dom.removeClass("step2","current");
			YAHOO.util.Dom.addClass("receipient","accessAid");
			YAHOO.util.Dom.addClass("receipient","accessAid");
			YAHOO.util.Dom.addClass("receipient","accessAid");
			YAHOO.util.Dom.addClass("receipient","accessAid");
			YAHOO.util.Dom.addClass(info_currency,"accessAid");
			YAHOO.util.Dom.addClass("label_from_email_address","accessAid");
			YAHOO.util.Dom.addClass("equal","accessAid");
			YAHOO.util.Dom.addClass("unequal","accessAid");
			document.getElementById("country_name").disabled = false;
			if(document.getElementById("amount_ccode").value!="USD")
			YAHOO.util.Dom.removeClass("ex_rate_usd","accessAid");
			
		},
		fnStep1Continue : function() //fnStep1Continue
		{       
			s_objectID="Continue";
			
			YAHOO.util.Dom.addClass("step1Continue","accessAid");
			YAHOO.util.Dom.removeClass("step1","current");
			YAHOO.util.Dom.addClass("step2","current");
			YAHOO.util.Dom.removeClass("receipient","accessAid");
		},
		email_format : function(email_obj,error_label,error_base) //fnEmailFormat
		{      
			var email_addressRegxp = /^([a-zA-Z0-9_\.\-])+\@(([a-zA-Z0-9\-])+\.)+([a-zA-Z]{2,3}){1,2}$/;
			if ((email_addressRegxp.test(email_obj) != true) || (email_obj.indexOf("(") >= 0))
			{
				YAHOO.util.Dom.addClass(document.getElementById(error_base), "error");
				YAHOO.util.Dom.removeClass(document.getElementById(error_label), "accessAid");
				return 1;
			}
		},	
		fnStep2Continue : function()  //fnStep2Continue
		{       
		
		        s_objectID="Continue";
			
			
		        document.getElementById("country").value = document.getElementById("country_name").value;
			var to_email_address = document.getElementById('emailto').value;
			var from_email_address = document.getElementById('emailfrom').value;
			var booleanerror=0;
			var email_addressRegxp = /^([a-zA-Z0-9_\.\-])+\@(([a-zA-Z0-9\-])+\.)+([a-zA-Z]{2,3}){1,2}$/;
			var amountRegxp = /^\.?[0-9]*\.?[0-9]*[.]{0,1}[0-9]{1,2}$/;

			YAHOO.util.Dom.addClass(document.getElementById("equal"), "accessAid");
			YAHOO.util.Dom.addClass(document.getElementById("unequal"),"accessAid");	
			YAHOO.util.Dom.addClass(document.getElementById("label_from_email_address"), "accessAid");
			YAHOO.util.Dom.removeClass(document.getElementById("show_email_to_error"),"error");
			YAHOO.util.Dom.removeClass(document.getElementById("show_email_from_error"),"error");	

			booleanerror_to = P2Pobj.email_format(to_email_address, "unequal", "show_email_to_error");
			booleanerror_from = P2Pobj.email_format(from_email_address, "label_from_email_address", "show_email_from_error");

			if(booleanerror_to == 1 || booleanerror_from == 1)
			{
				booleanerror = 1;
			}

			var l_to_email_address = new String(to_email_address).toLowerCase();
			var l_from_email_address = new String(from_email_address).toLowerCase();

			if (l_to_email_address == l_from_email_address)
			{		
				YAHOO.util.Dom.addClass(document.getElementById("show_email_to_error"), " error");
				if (((to_email_address.indexOf("(") >=0 ) && (from_email_address.indexOf("(") >= 0)) || (email_addressRegxp.test(to_email_address) != true)){
				YAHOO.util.Dom.removeClass(document.getElementById("unequal"), "accessAid");
				booleanerror =1;
				}
				else{
				YAHOO.util.Dom.removeClass(document.getElementById("equal"), "accessAid");
				booleanerror =1;
				}
			}


			if(booleanerror == 1)
			{
				YAHOO.util.Dom.removeClass(document.getElementById("messagediv"), "accessAid");
				return false;
			}
			else{
				YAHOO.util.Dom.addClass(document.getElementById("messagediv"), "accessAid");
				P2Pobj.fnSetCookie();
				document.esmForm.submit();
				return true;
			}
			},
		fnCookie : function(){  //fnCookie
			if(document.cookie!="") 
			{
				cookie_str= P2Pobj.fnGetCookie();
				if(cookie_str!= null && cookie_str!="")
				{			
				P2Pobj.fnParseCookie(cookie_str);
				}
			}

		},
		fnGetCookie: function()  //fnGetCookie
		{
			c_st=document.cookie.indexOf(P2Pobj.cookie_name+ "=");	
			if(c_st!=-1)
			{
				c_st = c_st + (P2Pobj.cookie_name.length+1);				
				c_end = document.cookie.indexOf(";",c_st);
				if (c_end ==-1)	c_end = document.cookie.length;
				return (document.cookie.substring(c_st,c_end));
			}

		},
		fnSetCookie: function() //fnSetCookie
		{ 
			country_id = document.getElementById("country_name").value;
			currency_id = document.getElementById("amount_ccode").value;	
			expireDay = "100";
			cookieVal = document.getElementById("country_name").value +"|"+document.getElementById("amount_ccode").value;  
			var expdt= new Date();		
			if(parseInt(expireDay)>0 && parseInt(expireDay)!="")
			{		
				expdt.setDate( expdt.getDate()+ parseInt(expireDay) );		
				document.cookie=P2Pobj.cookie_name+"="+ cookieVal+";expires="+ expdt.toGMTString();	
			}

		},
		fnParseCookie: function(str){	//fnParseCookie	
			document.getElementById("country_name").value=str.split("|")[0];
			document.getElementById("amount_ccode").value=str.split("|")[1];
			P2Pobj.fnChangeIcon();
			P2Pobj.fnChangeContent();
			P2Pobj.fnGetExRate();
		},
		fnParseQueryString : function(){  //fnParseQueryString
			strURL=window.location.href;
			querystr = false;
			if(strURL.split("?").length>1)
			{
				qryStr = strURL.split("?")[1].split("&");

				if((qryStr.join(",")).indexOf("send_amt") >=0 || (qryStr.join(",")).indexOf("country_code") >=0 || (qryStr.join(",")).indexOf("currency_code") >=0 || (qryStr.join(",")).indexOf("payment_source") >=0)
				{		
							
					for(k=0;k<qryStr.length;k++)
					{
					//alert(qryStr[k].split("=")[1]);
						if(qryStr[k].split("=")[0]=='send_amt')
						{
						document.getElementById("amount").value=qryStr[k].split("=")[1];
						querystr = true;
						}
						else if(qryStr[k].split("=")[0]=='country_code')
						{
						document.getElementById("country_name").value=qryStr[k].split("=")[1];
						document.getElementById("amount_ccode").value=P2Pobj.country_currency_default[qryStr[k].split("=")[1]];
						querystr = true;
						}
						else if(qryStr[k].split("=")[0]=='currency_code')
						{
						document.getElementById("amount_ccode").value=qryStr[k].split("=")[1];
						querystr = true;
						}
						else if(qryStr[k].split("=")[0]=='payment_source')
						{
						document.getElementById("payment_source").value=qryStr[k].split("=")[1];
						}
					}
				}
				
			}
			if(querystr==true)
			{
			P2Pobj.fnChangeIcon();
			P2Pobj.fnChangeContent();
			P2Pobj.fnGetExRate();
			YAHOO.util.Dom.addClass("ex_rate_usd", "accessAid");
			setTimeout("P2Pobj.fnGetFeetable()",1000);
			}
			return querystr;
			}	

	};

	function fnOnLoad(){	//fnOnLoad
			P2Pobj = new sm_xborder();
			
			YAHOO.util.Event.addListener("country_name", "change", P2Pobj.fnChangeIcon); 
			YAHOO.util.Event.addListener("country_name", "change", P2Pobj.fnChangeCurrency);
			YAHOO.util.Event.addListener("country_name", "change", P2Pobj.fnChangeContent);
			YAHOO.util.Event.addListener("country_name", "change", P2Pobj.fnGetExRate);
			YAHOO.util.Event.addListener("btnCalculate", "click", P2Pobj.fnGetFeetable);
			YAHOO.util.Event.addListener("amount_ccode", "change", P2Pobj.fnGetExRate);
			YAHOO.util.Event.addListener("INFO_USD", "click", P2Pobj.fnCheckServiceFeeUSD);
			YAHOO.util.Event.addListener("INFO_AUD", "click", P2Pobj.fnCheckServiceFee);
			YAHOO.util.Event.addListener("INFO_GBP", "click", P2Pobj.fnCheckServiceFee);
			YAHOO.util.Event.addListener("INFO_CAD", "click", P2Pobj.fnCheckServiceFee);
			YAHOO.util.Event.addListener("INFO_CZK", "click", P2Pobj.fnCheckServiceFee);
			YAHOO.util.Event.addListener("INFO_DKK", "click", P2Pobj.fnCheckServiceFee);
			YAHOO.util.Event.addListener("INFO_EUR", "click", P2Pobj.fnCheckServiceFee);
			YAHOO.util.Event.addListener("INFO_HKD", "click", P2Pobj.fnCheckServiceFee);
			YAHOO.util.Event.addListener("INFO_HUF", "click", P2Pobj.fnCheckServiceFee);
			YAHOO.util.Event.addListener("INFO_ILS", "click", P2Pobj.fnCheckServiceFee);
			YAHOO.util.Event.addListener("INFO_JPY", "click", P2Pobj.fnCheckServiceFee);
			YAHOO.util.Event.addListener("INFO_MXN", "click", P2Pobj.fnCheckServiceFee);
			YAHOO.util.Event.addListener("INFO_NZD", "click", P2Pobj.fnCheckServiceFee);
			YAHOO.util.Event.addListener("INFO_NOK", "click", P2Pobj.fnCheckServiceFee);
			YAHOO.util.Event.addListener("INFO_PLN", "click", P2Pobj.fnCheckServiceFee);
			YAHOO.util.Event.addListener("INFO_SGD", "click", P2Pobj.fnCheckServiceFee);
			YAHOO.util.Event.addListener("INFO_SEK", "click", P2Pobj.fnCheckServiceFee);
			YAHOO.util.Event.addListener("INFO_CHF", "click", P2Pobj.fnCheckServiceFee);
			YAHOO.util.Event.addListener("INFO_BRL", "click", P2Pobj.fnCheckServiceFee);
			YAHOO.util.Event.addListener("INFO_MYR", "click", P2Pobj.fnCheckServiceFee);
			YAHOO.util.Event.addListener("INFO_PHP", "click", P2Pobj.fnCheckServiceFee);
			YAHOO.util.Event.addListener("INFO_TWD", "click", P2Pobj.fnCheckServiceFee);
			YAHOO.util.Event.addListener("INFO_THB", "click", P2Pobj.fnCheckServiceFee);
			YAHOO.util.Event.addListener("btnReCalculate", "click", P2Pobj.fnRecalculate);
			YAHOO.util.Event.addListener("step1Continue", "click", P2Pobj.fnStep1Continue);
			YAHOO.util.Event.addListener("step2Continue", "click", P2Pobj.fnStep2Continue);
			YAHOO.util.Dom.removeClass("btnCalculate","accessAid");
			YAHOO.util.Dom.addClass("receipient","accessAid");
			YAHOO.util.Dom.addClass("ex_value","accessAid");
			YAHOO.util.Dom.addClass("btnReCalculate","accessAid");
			if(!P2Pobj.fnParseQueryString())
			{
				P2Pobj.fnCookie();			
			}
	}

