if (typeof(tagVars)=="undefined")
	tagVars = "";
	
if (typeof(lpUASorderTotal)!="undefined" && lpUASorderTotal!=""){
	tagVars = tagVars + '&PAGEVAR!OrderTotal=' + escape(lpUASorderTotal);
	tagVars = tagVars + '&PAGEVAR!' + lpUASunit + '_OrderTotal=' + escape(lpUASorderTotal);
	tagVars = tagVars + '&SESSIONVAR!Conversion=1';
}



if (typeof(lpUASonlineidOrderTotal)!="undefined")
	tagVars = tagVars + '&PAGEVAR!OnlineidOrderTotal=' + escape(lpUASonlineidOrderTotal);

if (typeof(lpUASpasscodeOrderTotal)!="undefined")
	tagVars = tagVars + '&PAGEVAR!PasscodeOrderTotal=' + escape(lpUASpasscodeOrderTotal);

if (typeof(lpUASconversionStage)!="undefined")
	tagVars = tagVars + '&PAGEVAR!ConversionStage=' + escape(lpUASconversionStage);

if (typeof(lpUASoLBResetErrorCounter)!="undefined")
	tagVars = tagVars + '&PAGEVAR!OLBResetErrorCounter=' + escape(lpUASoLBResetErrorCounter);

if (typeof(lpUASpasscodeErrorCounter)!="undefined")
	tagVars = tagVars + '&PAGEVAR!PasscodeErrorCounter=' + escape(lpUASpasscodeErrorCounter);

if (typeof(lpUASsection)!="undefined")
	tagVars = tagVars + '&PAGEVAR!Section=' + escape(lpUASsection);

if (typeof(lpUASstate)!="undefined")
	tagVars = tagVars + '&PAGEVAR!State=' + escape(lpUASstate);

if (typeof(lpUASerrortype)!="undefined")
	tagVars = tagVars + '&SESSIONVAR!Errortype=' + escape(lpUASerrortype);

if (typeof(lpUASonlineID)!="undefined")
	tagVars = tagVars + '&SESSIONVAR!OnlineID=' + escape(lpUASonlineID);

if (typeof(lpUASunit_Type)!="undefined"){
	tagVars = tagVars + '&PAGEVAR!Unit_Type=' + escape(lpUASunit_Type);
	tagVars = tagVars + '&SESSIONVAR!S_Unit_Type=' + escape(lpUASunit_Type);
}

	





lpUAStrimTagvars();
