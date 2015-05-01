/* IF THIS FILE IS CHANGED IT MUST BE RENAMED WITH A NEW DATE STAMP. THIS IS TO PREVENT JAVASCRIPT CACHING ISSUES */
/* UNDER HTTP. THE CONSTANT com.hsbc.ib.app.idv.config.UKIDVConstant.FIELD_VALIDATION MUST ALSO BE CHANGED TO */
/* REFERENCE THE NEW NAME  */

var trgFormName;
var trgElement;

var userIdInput;


/*Global Variables as its being used by GP and P2G validations*/

var alertDobText = "Please ensure you have entered your date of birth, using numbers only";
var alertDobDate = "Please ensure you have entered a valid date of birth";
var alertTsnText = "Please ensure you have entered the requested digits from your security number, using numbers only";
	
var DOB_LENGTH = 6;
var PASSWORD_LENGTH = 3;
var PASSWORD_LENGTH_VK = 9;

function check_date(field){
var checkstr = "0123456789";
var DateField = field;
var Datevalue = "";
var DateTemp = "";
var seperator = ".";
var day;
var month;
var year;
var leap = 0;
var err = 0;
var i;
   err = 0;
   DateValue = DateField.value;
   /* Delete all chars except 0..9 */
   for (i = 0; i < DateValue.length; i++) {
                  if (checkstr.indexOf(DateValue.substr(i,1)) >= 0) {
                     DateTemp = DateTemp + DateValue.substr(i,1);
                  }
   }
   DateValue = DateTemp;
   /* Always change date to 8 digits - string*/
   /* if year is entered as 2-digit / always assume 20xx */
   if (DateValue.length == 6) {
      DateValue = DateValue.substr(0,4) + '20' + DateValue.substr(4,2); }
   if (DateValue.length != 8) {
      err = 19;}
   /* year is wrong if year = 0000 */
   year = DateValue.substr(4,4);
   if (year == 0) {
      err = 20;
   }
   /* Validation of month*/
   month = DateValue.substr(2,2);
   if ((month < 1) || (month > 12)) {
      err = 21;
   }
   /* Validation of day*/
   day = DateValue.substr(0,2);
   if (day < 1) {
     err = 22;
   }
   /* Validation leap-year / february / day */
   if ((year % 4 == 0) || (year % 100 == 0) || (year % 400 == 0)) {
      leap = 1;
   }
   if ((month == 2) && (leap == 1) && (day > 29)) {
      err = 23;
   }
   if ((month == 2) && (leap != 1) && (day > 28)) {
      err = 24;
   }
   /* Validation of other months */
   if ((day > 31) && ((month == "01") || (month == "03") || (month == "05") || (month == "07") || (month == "08") || (month == "10") || (month == "12"))) {
      err = 25;
   }
   if ((day > 30) && ((month == "04") || (month == "06") || (month == "09") || (month == "11"))) {
      err = 26;
   }
   /* if 00 ist entered, no error, deleting the entry */
   if ((day == 0) && (month == 0) && (year == 00)) {
      err = 0; day = ""; month = ""; year = ""; seperator = "";
   }
   /* if no error, write the completed date to Input-Field (e.g. 13.12.2001) */
   if (err == 0) {
      //DateField.value = day +  month + year;
      //alert('Date is correct');
	  return true;
   }
   /* Error-message if err != 0 */
   else {
      //alert("Date is incorrect!");
      //DateField.select();
      //DateField.focus();
	  return false;
   }
}


function ukIdvPibValidation(form, userid)
{
	

	var alertText = "To log on to Personal Internet Banking, please enter your Internet Banking user ID.";
	
	if ((userid.value=="") || (userid.value==null) || (!userid.value))
	{
		alert (alertText);
	}
        else
	{
		form.submit();
	}
}



/**
validate fields before submission
*/
function idvFieldValidation(form, memAnswer, password)
{
	var thisForm = form; //document.forms[form];

	if ( !isValid(memAnswer,DOB_LENGTH) )
	{
		alert (alertDobText);
	}
	else if(!check_date(memAnswer))
	{
		alert (alertDobDate);
	}
	else if (!isValid(password,PASSWORD_LENGTH))
	{
		alert (alertTsnText);
	}
	else
	{
		processSecurity();
		thisForm.submit();
	}
}

/** Validates before submission for VK*/
function idvFieldValidationVK(form, memAnswer, password)
{
	var thisForm = form; //document.forms[form];

	if ( !isValid(memAnswer,DOB_LENGTH) )
	{
		alert (alertDobText);
	}
	else if(!check_date(memAnswer))
	{
		alert (alertDobDate);
	}
	else if (!isValidVK(password,PASSWORD_LENGTH_VK))
	{
		alert (alertTsnText);
	}
	else
	{
		processSecurity();
		thisForm.submit();
	}
}

function isValid(pVal,pLength)
{
	
	if ((pVal.value=="") || (pVal.value==null) || (!pVal.value))
		return false;
	
	var val = pVal.value;	
	if( pVal.value.length != pLength)
		return false;

	for( var i=0; i< val.length; i++) 
	{
		if( !isDigit(val.charAt(i)) )
			return false;
	}				
	return true;
}

function isValidVK(pVal,pLength)
{
	
	if ((pVal.value=="") || (pVal.value==null) || (!pVal.value))
		return false;
	
	var val = pVal.value;	
	if( pVal.value.length != pLength)
		return false;

	for( var i=0; i< val.length; i++) 
	{
		if( !isString(val.charAt(i)) )
			return false;
	}				
	return true;
}


function isDigit(num) {
	
	var string="1234567890";
    if (string.indexOf(num)!=-1)
    {
    	return true;
    }
	return false;
}
function isString(str) {
	
	var string="abcdefghijklmnopqrstuvwxyz1234567890";
    if (string.indexOf(str)!=-1 )
    { 
    	return true;
    }
	return false;
}

/* Checks the keyboard key pressed and submits the form if 
   enter is pressed. 
*/
function checkEnter(e, thisForm,memAnswer, password){ 
    var characterCode;
    
    characterCode = e.keyCode ;
    if (characterCode == 13)
        idvFieldValidation(thisForm, memAnswer, password)
       //thisForm.submit();
    else
       return true;   
}
/* Check for the key presses values when VK used*/ 
function checkEnterVK(e, thisForm,memAnswer, password){ 
    var characterCode;
    
    characterCode = e.keyCode ;
    if (characterCode == 13)
        idvFieldValidationVK(thisForm, memAnswer, password)
       //thisForm.submit();
    else
       return true;   
}

// IF appVersion is MAC AND IF userAgent is MSIE5 then do nothing.
// IF they are anything else set the fields to ""
function clearMe()
{
	//if( typeof(userIdInput) != 'undefined' && userIdInput != null ) {
		userIdInput.value="";
	//}
	//var agt=navigator.userAgent.toLowerCase();
		
	//var targetFormName = trgFormName;
	//var targetElementName = trgElement;
	//var noOfForms = document.forms.length;
	//for(i = 0; i < noOfForms; i++) {
	//	var formName = document.forms[i].name;
	//	if(formName.indexOf(targetFormName) != -1) {
	//		document.forms[i].elements[targetElementName].value = "";
	//	}
	//}
	
}




function populateMEM(submitForm)
{
	if( (submitForm.mem1.value != null && submitForm.mem2.value != null && submitForm.mem3.value != null ) && (submitForm.mem1.value.length == 2 && submitForm.mem2.value.length == 2 && submitForm.mem3.value.length == 2 ) ){
		submitForm.memorableAnswer.value = submitForm.mem1.value+submitForm.mem2.value+submitForm.mem3.value;
	}else{
		submitForm.memorableAnswer.value = "";
		if ( !isValid(submitForm.memorableAnswer,DOB_LENGTH) )
		{
				alert (alertDobText);
		}
		
	}
}


/* Checks the keyboard key pressed and submits the form if 
   enter is pressed for GP linking 2 of 2 
*/
function checkEnterForGP(e, thisForm,memAnswer, password){ 
    var characterCode;
  
    characterCode = e.keyCode ;
    if (characterCode == 13){
			checkFieldsForGP(thisForm,memAnswer, password);
		}else{
       return true;
    }
}


function checkFieldsForGP(thisForm,memAnswer, password){ 

	populateMEM(thisForm);
	if(memAnswer.value != null && memAnswer.value.length == DOB_LENGTH){
			idvFieldValidation(thisForm, memAnswer, password);
	}

}
/* Checks the keyboard key pressed and submits the form if 
   enter is pressed for GP linking 2 of 2 
*/
function checkEnterForGPVK(e, thisForm,memAnswer, password){ 
    var characterCode;
  
    characterCode = e.keyCode ;
    if (characterCode == 13){
			checkFieldsForGPVK(thisForm,memAnswer, password);
		}else{
       return true;
    }
}
function checkFieldsForGPVK(thisForm,memAnswer, password){ 

	populateMEM(thisForm);
	if(memAnswer.value != null && memAnswer.value.length == DOB_LENGTH){
			idvFieldValidationVK(thisForm, memAnswer, password);
	}

}

//function for PCPrint Javascript Collector calls.
function processSecurity(){

forms.load_data('session_data');
fortyone_collect('userPrefs');
}