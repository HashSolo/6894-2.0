function lpUAStrimTagvars(){



	var INITIAL_MAX_SIZE = 300;
	var MAX_TAGVARSURL_SIZE = 1600;
	var INITIAL_STRING = document.location.toString() + document.title;
	var STRING_MAX_SIZE = INITIAL_STRING.length + INITIAL_MAX_SIZE;
      
  
	if ((typeof(tagVars) == "undefined") || (tagVars == null))	
	tagVars = "";
	while ((tagVars.length + STRING_MAX_SIZE > MAX_TAGVARSURL_SIZE) && (tagVars.length > 0)) {
	 var idx = tagVars.lastIndexOf("&");
	
	 if (idx > 0)
	  tagVars = tagVars.substring(0, idx);
	 else
	  tagVars = "";
	}
}


if (typeof(tagVars)=="undefined")
	tagVars = "";


if (typeof(lpUASrouting)!="undefined")
	tagVars = tagVars + '&PAGEVAR!Routing=' + escape(lpUASrouting);
	
if (typeof(lpUASorderTotal)!="undefined")
	tagVars = tagVars + '&PAGEVAR!OrderTotal=' + escape(lpUASorderTotal);

if (typeof(lpUASconversionAction)!="undefined")
	tagVars = tagVars + '&PAGEVAR!ConversionAction=' + escape(lpUASconversionAction);

if (typeof(lpUASconversionStage)!="undefined")
	tagVars = tagVars + '&PAGEVAR!ConversionStage=' + escape(lpUASconversionStage);

if (typeof(lpUASerrorCounter)!="undefined")
	tagVars = tagVars + '&PAGEVAR!ErrorCounter=' + escape(lpUASerrorCounter);

if (typeof(lpUASstate)!="undefined")
	tagVars = tagVars + '&PAGEVAR!State=' + escape(lpUASstate);

lpUAStrimTagvars();

