function selectButton(formObj, radioObj) {
    radioObj.checked = true;

    for (var i=0; i < formObj.elements.length; i++) {
	if (formObj.elements[i].type == "radio" && formObj.elements[i] != radioObj) {
	    formObj.elements[i].checked = false;
	}
    }
}

function showContainer() {

    if (document.securityForm.select_confirm_category){
        var userSelection = document.securityForm.select_confirm_category.value;
        var bank=document.getElementById('bankContainer');
        var creditCard=document.getElementById('ccContainer');
        var debitCard=document.getElementById('dcContainer');
        var nationalID=document.getElementById('nationalIDContainer');
        var securityQuestions=document.getElementById('securityQuestionsContainer');
        var socialSecurityNumber=document.getElementById('ssnContainer');
    
       /* Spec #17511 start */
        var creditCardImage = document.getElementById('creditCardImage');

        if(creditCardImage) {
             if (userSelection=='bank_acct') {
             creditCardImage.style.display = "";
             }
             else {
             creditCardImage.style.display = "none";
             }
        }
        /* Spec #17511 end */    
      
        if (userSelection=='bank_acct') {
        bank.style.display = "block";
        creditCard.style.display = "none";
        debitCard.style.display = "none";
        nationalID.style.display = "none";
        securityQuestions.style.display = "none";
        socialSecurityNumber.style.display = "none";
        
        if(document.securityForm.ach_id)
        	document.securityForm.ach_id.focus();
        }
    
        if (userSelection=='credit_card') {
        bank.style.display = "none";
        creditCard.style.display = "block";
        debitCard.style.display = "none";
        nationalID.style.display = "none";
        securityQuestions.style.display = "none";
        socialSecurityNumber.style.display = "none";
        
      	

	

	if(document.securityForm.cc_id.disabled == false)
        	document.securityForm.cc_id.focus();
        }
    
        if (userSelection=='debit_card') {
        bank.style.display = "none";
        creditCard.style.display = "none";
        debitCard.style.display = "block";
        nationalID.style.display = "none";
        securityQuestions.style.display = "none";
        socialSecurityNumber.style.display = "none";
        }
    
        if (userSelection=='national_id') {
        bank.style.display = "none";
        creditCard.style.display = "none";
        debitCard.style.display = "none";
        nationalID.style.display = "block";
        securityQuestions.style.display = "none";
        socialSecurityNumber.style.display = "none";
        }
    
        if (userSelection=='security_questions') {
        bank.style.display = "none";	
        creditCard.style.display = "none";
        debitCard.style.display = "none";
        nationalID.style.display = "none";
        securityQuestions.style.display = "block";
        socialSecurityNumber.style.display = "none";
        
        if(document.securityForm.answer_1)
        	document.securityForm.answer_1.focus();
        }
    
        if (userSelection=='ssn') {
        bank.style.display = "none";
        creditCard.style.display = "none";
        debitCard.style.display = "none";
        nationalID.style.display = "none";
        securityQuestions.style.display = "none";
        socialSecurityNumber.style.display = "block";
        
        if(document.securityForm.ssn_1)
        	document.securityForm.ssn_1.focus();
        }
    
        else if (userSelection=='') {
        bank.style.display = "none";
        creditCard.style.display = "none";
        debitCard.style.display = "none";
        nationalID.style.display = "none";
        securityQuestions.style.display = "none";
        socialSecurityNumber.style.display = "none";
        }
    }
}

/* Spec #17511 Tier2 start */
function showCCCategory()
{
	var cc_dropdown = document.getElementById("cc_id");
	if(cc_dropdown) {
		var cc_values	= cc_dropdown.value.split(" ");
		var cc_index	= cc_values[0];
		var cc_type 	= cc_values[1];
		
		if (cc_dropdown.options[0] && typeof cc_dropdown.options[cc_index] != "undefined") {
			var optionvalue = cc_dropdown.options[cc_index].value;
		}

		var Visa = new RegExp(/V/);
		var MasterCard = new RegExp(/M/);
		var Discover = new RegExp(/D/);
		var AMEX = new RegExp(/A/);
		var cc_category;

		if(Visa.test(cc_type) || MasterCard.test(cc_type) || Discover.test(cc_type)) cc_category = 1;
		else if(AMEX.test(cc_type)) cc_category = 2;
		else cc_category = 3;
		
		if(document.getElementById("cc_category1")) {
			document.getElementById("cc_category1").className= 'hide';
		}
		if(document.getElementById("cc_category2")) {
			document.getElementById("cc_category2").className = 'hide';
		}
		if(document.getElementById("cc_category3")) {
			document.getElementById("cc_category3").className = 'hide';
		}
		if(document.getElementById("cc_category" + cc_category)) {
			document.getElementById("cc_category" + cc_category).className = '';
		}
	}
}

/* Tier2: Function autoTab for CC fields - FR #PPSCR00555495 */

var isNN = (navigator.appName.indexOf("Netscape")!=-1);

function autoTab(input,len, e) {

	var keyCode = (isNN) ? e.which : e.keyCode; 
	var filter = (isNN) ? [0,8,9] : [0,8,9,16,17,18,37,38,39,40,46];
	
	if(input.value.length >= len && !containsElement(filter,keyCode)) {
		input.value = input.value.slice(0, len);
		input.form[(getIndex(input)+1) % input.form.length].focus();
	}

	function containsElement(arr, ele) {
		var found = false, index = 0;
		while(!found && index < arr.length)
		if(arr[index] == ele) {
			found = true;
		}
		else {
			index++;
		}
		return found;
	}

	function getIndex(input) {
		var index = -1, i = 0, found = false;
		while (i < input.form.length && index == -1) {
			if (input.form[i] == input) {
				index = i;
			}
			else {
				i++;
			}
		}
		return index;
	}
	
	return true;
}

/* Spec #17511 Tier1 start */

function hideDetails()
{
	var userSelection = document.securityForm.select_confirm_category[document.securityForm.select_confirm_category.selectedIndex].value;

	document.getElementById('phoneContainer').style.display = "none";
	document.getElementById('addressContainer').style.display = "none";
	document.getElementById('emailContainer').style.display = "none";

	if (userSelection == 'phone') 		document.getElementById('phoneContainer').style.display = "block";
	if (userSelection == 'address') 	document.getElementById('addressContainer').style.display = "block";
	if (userSelection == 'verify_email') 	document.getElementById('emailContainer').style.display = "block";
}

function showDetails() 
{
	if (document.securityForm.select_confirm_category)
	{
      		var userSelection = document.securityForm.select_confirm_category[document.securityForm.select_confirm_category.selectedIndex].value;
		var phoneNumber = document.getElementById('phoneContainer');
      		var addressNumber = document.getElementById('addressContainer');
		var emailid = document.getElementById('emailContainer');

		phoneNumber.style.display = "none"; 
		addressNumber.style.display = "none"; 
		emailid.style.display = "none"; 

    		if (userSelection=='phone') 
		{
			phoneNumber.style.display = "block";
			if(document.securityForm.phone_id) document.securityForm.phone_id.focus();			
		}

		if (userSelection=='address') 
		{
			addressNumber.style.display = "block";
			if(document.securityForm.address_num) document.securityForm.address_num.focus();
	    	}

		if (userSelection=='verify_email') 
		{
			emailid.style.display = "block";
		}
	}
}


function showPhonePrefix(formObj)
{
	var rePhone	= / number (.*?)XXXX/
	var phoneNum 	= formObj.options[formObj.selectedIndex].text;
	var phonePrefix = phoneNum.match(rePhone);
	
	document.getElementById("phonePrefix").innerHTML = phonePrefix[1];
}

/* Spec #17511 Tier1 end */
