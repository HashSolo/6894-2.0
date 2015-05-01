function isNumeric(number){
	number = number + "";
	return ((number.length > 0) && isComposedOfChars("0123456789", number));
}
function isAlphanumeric(alphaNumString) {
	alphaNumString = alphaNumString + "";
	return ((alphaNumString.length > 0) &&
		isComposedOfChars("abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", alphaNumString));
}
function isAlphanumericOrUnderscore(alphaNumString) {
	alphaNumString = alphaNumString + "";
	return ((alphaNumString.length > 0) &&
		isComposedOfChars("abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_", alphaNumString));
}
function indexOfFirstNotIn(okayChars, inString) {
	var i;
	for (i=0; i < inString.length; i++){
		var charm = inString.charAt(i);
		if (okayChars.indexOf(charm) == -1) {return i;}
	}
	return -1;
}
function indexOfLastNotIn(okayChars, inString) {
	var i;
	for (i = inString.length - 1; i >= 0; i--) {
		var charm=inString.charAt(i);
		if (okayChars.indexOf(charm) == -1) {return i;}
	}
	return -1;
}
function strip(string, sChar) {
	if ((string == null) || (string == "")) {return "";}
	var startIndex = indexOfFirstNotIn(sChar, string);
	var endIndex = indexOfLastNotIn(sChar, string);
	if (startIndex == -1) {return "";}
	return string.substring(startIndex, endIndex + 1);
}
function isComposedOfChars(validChars, inString) {
	return (indexOfFirstNotIn(validChars, inString) == -1);
}



function firstNotInLast(badChars, inString) {
	var i;
	// alert(badChars + " " + inString);
	for (i=0; i < inString.length; i++){
		var charm = inString.charAt(i);
	//	alert(charm);
		if (badChars.indexOf(charm) != -1) {return i;}
	}
	return -1;
}

function containsBadChars(invalidChars, inString) {
	return (firstNotInLast(invalidChars, inString) == -1);
}

function trimS(sString) {
	while (sString.substring(0,1) == ' ') {
		sString = sString.substring(1, sString.length);
	}
	while (sString.substring(sString.length-1, sString.length) == ' ') {
		sString = sString.substring(0,sString.length-1);
	}
	return sString;
}

function containsValidCharsID(accessIDString) {
	accessIDString = accessIDString + "";
	return ((accessIDString.length > 0) &&
		isComposedOfChars("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789~@#*()`;,.?/=+-_{}", accessIDString));
}

function containsValidCharsPW(passcodeString) {
	passcodeString = passcodeString + "";
	return ((passcodeString.length > 0) &&
		isComposedOfChars("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789~@#%*()`:;\",'.?/=+-_{}", passcodeString));
}

function convertExtendedASCII(l_id)  {

	var local_id = l_id;
	var return_id = "";
	
	var convCharArray = new Array(" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", "!", " ", " ", " ", " ", " ", " ", " ", " ", " ", "\"", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", "\"", " ", " ", " ", "?",
"A", "A", "A", "A", "A", "A", "A", "C", "E", "E", "E", "E", "I", "I", "I", "I", "D", "N", "O", "O", "O", "O", "O", " ", "O", "U", "U", "U", "U", "Y", "P", "B", "a", "a", "a", "a", "a", "a", "a", "c", "e", "e", "e", "e", "i", "i", "i", "i", "d", "n", "o", "o", "o", "o", "o", " ", "o", "u", "u", "u", "u", "y", "p", "y")

	//loop through ID
	for (var i = 0; i < local_id.length; i++) {
	
		var charCode = local_id.charCodeAt(i);
		var thisChar = local_id.charAt(i);     
		var arrayIndx;

		if((charCode > 127) && (charCode < 256)) {
			// If extended ASCII character, then convert to standard ASCII
			arrayIndx = charCode - 128;
			return_id = return_id + convCharArray[arrayIndx];
			
		} else {
			return_id = return_id + thisChar;
		}
	 
	}     
	
return return_id;
}


