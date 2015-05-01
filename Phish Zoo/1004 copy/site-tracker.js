/*
   SiteTracker 5.1.10
   Copyright (C) 2007 Site Intelligence Ltd.
*/

var undefined; // For JS engines that don't support this keyword natively

if (window['siAutoTracer'] == undefined)
{
	window.siAutoTracer = true;  // Set to false if you don't want automatic page tracking
}

var siBaseDirectory   = "/hbos";
var siTaggingServer   = "www.site-analytics.co.uk"; // Send tags to same server as document comes from
var siCookieName      = "HBOSVISITOR";
var siTracerCookieName= "SITRACER";
var siLinkAttribute   = "id";  // This must either be "si:link" or "id"
var siCookieQPName    = "simigvis";
var siCookiePath      = "/";
var siCookieTimeout   = 315360000000;
var siTracerTimeout   = 1000; // ms to wait maximum for tracer to be sent
var siDomainList      = new Array(".co.uk",".com");
var siCentralCookie   = false;
var siCentralReqName  = "req";
var siCentralRefName  = "refer";
var siCentralURL      = "";
var siTrackerUrl      = siBaseDirectory + "/track.gif";


//////////////////////////////////
// Do not edit below this line! //
//////////////////////////////////

var siIsCookieNew	    = "N";
var siCookieValue     = null;
var siExtClickID      = "t"+(new Date()).getTime()+"h"+window.history.length;
var siClickedLinkID   = "";
var siRefClickID      = "";
var siTracerPath      = "";
var siTracerQuery     = "";
var siSendParams      = "";
var siTracerCookieIdx = 0;
var siImageSet        = new Array(1);
var siReferrer        = document.referrer;
var siRequest         = document.URL;
var siPageIDAttrName  = "si:pageID";
var siTagType         = "";
var siSentPageTag     = false;
var siSyncTracersOutstanding = 0;
var siSyncFinishedAction = null;


function SiInternalGetLinkClickingClosure(anchor)
{
   var targetLoc = null;
   var destination = anchor.href;
   var target = anchor.getAttribute("target");
   if (target && target != "_self")
   {
      if (target == "_top") targetLoc = "top";
      else if (target == "_parent") targetLoc = "parent";
      else
      {
         if (target == "_new")
         {
            return function() { window.open(destination); }
         }
         else
         {
            targetLoc = top.frames[target];
            if (!targetLoc)
            {
               return function() { window.open(destination, target); }
            }
         }
      }
   }

   if (targetLoc == null) {
      return function() { location.href = destination; }
   }
   else
   {
      return function() { targetLoc.location.href = destination; }
   }
}


function SiEncodeString(Input)
{
  if (!Input)
  {
    return "";
  }

  var TransChars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  var Output     = "";
  for (var i = 0; i < Input.length; i += 3)
  {
    var NumBytesLeft = Input.length - i;
    var Value        = 0;
    Value  = (Input.charCodeAt(i) << 16) & 0x00ff0000;
    Value |= (NumBytesLeft > 1)? (Input.charCodeAt(i + 1) << 8) & 0x0000ff00 : 0;
    Value |= (NumBytesLeft > 2)? Input.charCodeAt(i + 2) & 0x000000ff : 0;
    Output += TransChars.charAt((Value & 0x00fC0000) >> 18);
    Output += TransChars.charAt((Value & 0x0003f000) >> 12);
    Output += (NumBytesLeft > 1)? TransChars.charAt((Value & 0x00000fc0) >> 6) : '_';
    Output += (NumBytesLeft > 2)? TransChars.charAt((Value & 0x0000003f)) : '_';
  }
  return Output;
}

//---------------------------------------------------------------------------------------------
// Encode all the relevent detals into a string
//---------------------------------------------------------------------------------------------
function SiEncodeDetails(Format)
{
   var Output = "";
   for (var i = 0; i < Format.length; i++)
   {
      var Data;
      switch (Format.charAt(i))
      {
      case 'r':
         Data = siReferrer;
         break;
      case 'p':
         Data = siRequest;
         break;
      case 'd':
         Data = screen.availWidth+"x"+screen.availHeight+"x"+screen.colorDepth+"."+navigator.javaEnabled();
         if (navigator.plugins) Data += "."+navigator.plugins.length;
         break;
      case 'c':
         Data = siCookieValue;
         break;
      case 'u':
         Data = window.history.length+"."+(Math.random()*1000)+"."+(new Date()).getTime();
         break;
      case 't':
         Data = siExtClickID;
         break;
      case 'f':
         Data = siTracerPath;
         break;
      case 'q':
         Data = siTracerQuery;
         break;
      case 'g':
         Data = SiCollateTagData();
         break;
      case 'w':
         Data = siIsCookieNew;
         break;
      case 'y':
      	Data = siTagType;
      	break;
      }
      Output += SiEncodeString(Data)+"*";
   }
   return Output;
}

function SiCollateTagData()
{
	var TagData = "";

	TagData += "co=" + window.screen.colorDepth;
	TagData += "&sr=" + window.screen.width + 'x' + window.screen.height;
	today = new Date();
	if (null != today)
	{
		TagData += "&lt=" + SiFormatDate(today);
	}
	if (navigator.javaEnabled())
	{
		TagData += "&jv=1";
	}
	else
	{
		TagData += "&jv=0";
	}
	TagData += SiCollatePageRules();

	if (document.body)
	{
		var ThisPageID = document.body.getAttribute(siPageIDAttrName);
		if (ThisPageID)
		{
			TagData += "&req=" + ThisPageID;
		}
	}

	if (siClickedLinkID && siClickedLinkID.length > 0)
	{
		TagData += "&rl=" + siClickedLinkID;
	}

	if (siRefClickID.length > 0)
	{
		TagData += "&rcid=" + siRefClickID;
	}

	return TagData;
}

function SiCollatePageRules()
{
	var PageData = "";

	if (window.siIsFrameset)
	{
		PageData += "&page:fset";
	}
	if (window.siIsMenu)
	{
		PageData += "&page:menu";
	}
	if (window.siIsExtraFrame)
	{
		PageData += "&page:x-frame";
	}
	if (window.siIsRedirection)
	{
		PageData += "&page:redir";
	}
	if (window.siIsPopup)
	{
		PageData += "&page:popup";
	}

	return PageData;
}

function SiIsMenu()
{
	window.siIsMenu = true;
}

function SiIsFrameset()
{
	window.siIsFrameset = true;
}

function SiIsExtraFrame()
{
	window.siIsExtraFrame = true;
}

function SiIsRedirection()
{
	window.siIsRedirection = true;
}

function SiIsPopup()
{
	window.siIsPopup = true;
}

function SiSetPageID(id)
{
	document.body.setAttribute(siPageIDAttrName,id);
}

function SiFormatDate(date)
{
	var output = "";
	output += date.getFullYear()+"";
	output += "-";
	output += leadingZero(date.getMonth()+1);
	output += "-";
	output += leadingZero(date.getDate());
	output += "T";
	output += leadingZero(date.getHours());
	output += ":";
	output += leadingZero(date.getMinutes());
	output += ":";
	output += leadingZero(date.getSeconds());
	return output;
}

function leadingZero(n)
{
	return (n>9?"":"0") + n;
}

function SiGetDomain()
{
  var DomainValue = null;
  var firstDot;
  var secondDot;
  var lastDot;
  var useHostname=document.location.hostname;
  if (useHostname != null)
  {
     var arIndex;
     for (arIndex = 0; (arIndex < siDomainList.length) && (DomainValue == null); arIndex++)
     {
        var tldIndex = useHostname.lastIndexOf(siDomainList[arIndex]);
        if (tldIndex > 0)
        {
           var nextDot = useHostname.lastIndexOf('.',tldIndex-1);
           if (nextDot >= 0)
           {
              DomainValue = useHostname.substring(nextDot);
           }
           else
           {
              DomainValue = "." + useHostname;
           }
        }
     }
  }
  return DomainValue;
}

function SiTrackLink(idVar)
{
   siTagType="trace";

	if (/string/.test(typeof(idVar)))
	{
		siClickedLinkID = idVar;
	}
	else
	{
		// We have an anchor tag or equivalent
		siClickedLinkID = idVar.getAttribute(siLinkAttribute);
	}

	SiInternalTrackLink();
	return true;
}

function SiInternalTrackLink()
{
   SiInternalDoLinkTrack(SiDeferTracer);
}

function SiInternalTrackLinkImmediate()
{
   SiInternalDoLinkTrack(SiSyncTracer);
}

function SiInternalTrackLinkWait()
{
   SiInternalDoLinkTrack(SiSendTracerWithWait);
}

function SiInternalTrackLinkViaSynchronousSend(anchorTag, closure)
{
   if (!closure)
      closure = SiInternalGetLinkClickingClosure(anchorTag);

   SiInternalDoLinkTrack(SiSyncTracer, closure, siTracerTimeout);
}

function SiInternalDoLinkTrack(dispatcher, closure, timeout)
{
   siRefClickID = siExtClickID;
   siTracerPath = "http://" + document.location.hostname + siBaseDirectory + "/link";
   siTracerQuery = "";

   // Call the dispatcher with appropriate arguments
   if (closure) {
      if (timeout) {
         dispatcher("fctgy","fdtgy", closure, timeout);
      } else {
         dispatcher("fctgy","fdtgy", closure);
      }
   } else {
      dispatcher("fctgy","fdtgy");
   }

   siClickedLinkID = "";
   siRefClickID = "";
   siTracerPath = "";
}

function SiTrackEvent(eventName)
{
   siClickedLinkID = eventName;
   siTagType = "trace";
   SiInternalTrackLink();
   SiSendDeferredTracers();
}

function SiTrackExternalLink(idVar,closure)
{
   siTagType="trace";
   if (/string/.test(typeof(idVar)))
   {
      siClickedLinkID = idVar;
   }
   else
   {
      siClickedLinkID = idVar.getAttribute(siLinkAttribute);
   }
   SiInternalTrackLinkViaSynchronousSend(idVar,closure);
   return false;
}

function SiTrackLinkToUntagged(anchorTag)
{
	siClickedLinkID = anchorTag.getAttribute(siLinkAttribute);
   siTagType="trace";
   SiInternalTrackLinkImmediate();
   siTagType = "page";
   siTracerPath = anchorTag.href;
   if (siTracerPath.indexOf('://') == -1)
   {
      if(siTracerPath.substring(0,1) != '/')
      {
      	var lastSlash = document.location.pathname.lastIndexOf('/');
      	var currentDirectory = document.location.pathname.substring(0, lastSlash + 1);
      	siTracerPath = currentDirectory + siTracerPath;
      }
      if(siTracerPath.substring(0,2) != '//')
      {
      	siTracerPath = '//' + document.location.hostname + siTracerPath;
      }
      siTracerPath = document.location.protocol + siTracerPath;
   }
   var closure = SiInternalGetLinkClickingClosure(anchorTag);
   SiSyncTracer("fctgy","fdtgy",closure,siTracerTimeout);
   return false;
}

function SiTrackData(query)
{
   siTracerPath = "http://" + document.location.hostname + siBaseDirectory + "/data";
   siTracerQuery = query;
   siTagType = "extra";
   SiSendTracer(false,"fqcrtgy","fqdrtgy");
}

function SiAddTrackParam( name, value )
{
   if (siSendParams != "")
    	siSendParams += "&";
   siSendParams += escape(name) + "=" + escape(value);
   if (siSendParams.length >= 512)
   	SiSendTrackParams();
}

function SiSendTrackParams()
{
	if (siSendParams != "")
	{
		SiTrackData(siSendParams);
	   siSendParams = "";
	}
}

function SiSetCookie(Name,Value,Timeout)
{
  	var Expiry=new Date;
  	Expiry.setTime(Expiry.getTime()+Timeout);
 	var Domain=SiGetDomain();
  	var CookieDetails=Name+"="+Value+((siCookiePath) ? "; path=" + siCookiePath : "")+((Domain) ? "; domain="+Domain : "");
  	document.cookie=CookieDetails + "; expires="+Expiry.toGMTString();
  	if (SiGetCookie(Name) != Value)
  	{
  	   // Failed to set persistent cookie so try session cookie
		document.cookie=CookieDetails;
	}
}

function SiGetCookie(Name)
{
   return SiExtractPart(Name,document.cookie,";");
}

function SiExtractPart(Name,DataSource,DataEndChar)
{
  var Prefix = Name+"=";
  var Value  = null;
  var Begin  = DataSource.indexOf(Prefix);
  if ((Begin != -1) && (Name.length > 0))
  {
    var End = DataSource.indexOf(DataEndChar,Begin);
    if (End == -1) End = DataSource.length;
    Value = DataSource.substring(Begin+Prefix.length,End);
  }
  return Value;
}

function SiDeleteCookie(Name,Path,Domain)
{
	var Expiry = new Date();
	Expiry.setTime(Expiry.getTime() - 1);
	if (!Path) Path = siCookiePath;
	if (!Domain) Domain = SiGetDomain();
	var CookieDetails=Name+"=null"+((Path) ? "; path=" + Path : "")+((Domain) ? "; domain="+Domain : "");
	document.cookie = CookieDetails + "; expires="+Expiry.toGMTString();
}

function SiBuildCookie()
{
   var CookieValue=SiExtractPart(siCookieQPName,document.URL,"&");
   if (CookieValue != null)
   {
      SiSetCookie(siCookieName,CookieValue,siCookieTimeout);

      if (siCentralCookie)
      {
         // Extract original referrer
         var extractedURL = SiExtractPart(siCentralRefName,document.URL,"&");
         if (extractedURL != null)
         {
            siReferrer = unescape(extractedURL);
         }

         // Extract original request
         extractedURL = SiExtractPart(siCentralReqName,document.URL,"&");
         if (extractedURL != null)
         {
            siRequest = unescape(extractedURL);
         }
      }
   }
   else
   {
      CookieValue=SiGetCookie(siCookieName);
      if (CookieValue == null)
      {
         CookieValue=SiEncodeDetails("u");
         siIsCookieNew = 'Y';
         SiSetCookie(siCookieName,CookieValue,siCookieTimeout);

         if (siCentralCookie)
         {
            // Check cookies can be set
            CookieValue = SiGetCookie(siCookieName);
            if (CookieValue != null)
            {
               var centralURL = siCentralURL;
               if (siCentralURL.indexOf("?") == -1)
               {
                  centralURL += "?";
               }
               else
               {
                  centralURL += "&";
               }
               centralURL += siCentralReqName+"="+escape(document.URL)+"&"+siCentralRefName+"="+escape(document.referrer);

               // Stop any possibility of tag firing
               siAutoTracer = false;

               // Redirect to central cookie server
               document.location = centralURL;
            }
         }
      }
   }
   CookieValue = SiGetCookie(siCookieName);
   return CookieValue;
}

function SiMigrateCookie(link)
{
   var CookieValue=SiGetCookie(siCookieName);
   if (CookieValue != null) link.href=link.href+((link.href.indexOf('?') > 0) ? "&" : "?")+siCookieQPName+"="+CookieValue;
   return true;
}

function SiMigrateCookieForm(form)
{
   var CookieValue=SiGetCookie(siCookieName);
   if (CookieValue != null)
   {
      if (form.method.toUpperCase() == "GET")
      {
      	try
      	{
	      	try
	      	{
	      		var element = document.createElement('<input name="'+siCookieQPName+'" type="hidden" value="'+CookieValue+'" />');
	      	}
	      	catch (e)
	      	{
	      		var element = document.createElement("input");
				   element.setAttribute("name", siCookieQPName);
				   element.setAttribute("type", "hidden");
				   element.setAttribute("value", CookieValue);
	      	}
	      	form.appendChild(element);
	      }
	      catch (andIgnore) {}
      }
      else
      {
      	form.action=form.action+((form.action.indexOf('?') > 0) ? "&" : "?")+siCookieQPName+"="+CookieValue;
      }
   }
   return true;
}

function SiProcessTracer(cookieForm,noCookieForm,dispatcher)
{
	var tracer;
	if (siCookieValue == null)
	{
		tracer = siTaggingServer + siTrackerUrl + "?f=" + noCookieForm + "&d=" + SiEncodeDetails(noCookieForm);
	}
	else
	{
		tracer = siTaggingServer + siTrackerUrl + "?f=" + cookieForm + "&d=" + SiEncodeDetails(cookieForm);
	}

	siTagType="";
	dispatcher(tracer);
}

function SiSendTracer(isPage,cookieForm,noCookieForm)
{
	var sendNow = function(tracer) {
		var Tracker = new Image();
		Tracker.src = tracer;
		siImageSet[siImageSet.length] = Tracker;
	}

   SiProcessTracer(cookieForm,noCookieForm,sendNow);
}

function SiSyncTracer(cookieForm,noCookieForm,action,timeout)
{
   var send = function(tracer) {
      var Tracker = new Image();
      if (action) {
         siSyncFinishedAction = action;
         if (timeout) {
            var timeoutFn = function() {
               if (siSyncFinishedAction) {
                  var exec = siSyncFinishedAction;
                  siSyncFinishedAction = null;
                  exec();
               }
            }
            timeoutFn.toString = function() { return 'var exec = siSyncFinishedAction; siSyncFinishedAction = null; exec();' };
            var runningTimeout = window.setTimeout(timeoutFn, timeout);
         }
      }
      var checkDone = function() {
         if (Tracker.complete && --siSyncTracersOutstanding == 0 && siSyncFinishedAction)
         {
            if (runningTimeout) window.clearTimeout(runningTimeout);
            var exec = siSyncFinishedAction;
            siSyncFinishedAction = null;
            exec();
         }
      };
      try {
      Tracker.addEventListener ?
      Tracker.addEventListener("load", checkDone, false) :
      Tracker.attachEvent("onreadystatechange", checkDone);
      } catch (e){}
      ++siSyncTracersOutstanding;
      Tracker.src = tracer;
      siImageSet[siImageSet.length] = Tracker;
   }
   SiProcessTracer(cookieForm,noCookieForm,send)
}

function SiDeferTracer(cookieForm,noCookieForm)
{
	if (siTracerCookieIdx < 25)
	{
		var defer = function(tracer) {
	      if (siTracerCookieIdx > 0)
	      {
	         if (SiGetCookie(siTracerCookieName + (siTracerCookieIdx - 1)) == null)
	            siTracerCookieIdx = 0;
	      }

			var cookieName = siTracerCookieName + siTracerCookieIdx++;
		   SiSetCookie(cookieName,tracer,1800000); // 30 minute lifetime
		}

		SiProcessTracer(cookieForm,noCookieForm,defer);
	}
}

function SiTrackFormData(form,list)
{
	return SiTrackForm(null,form,list);
}

function SiTrackExternalFormData(form,list,closure)
{
   siTracerPath = "http://" + document.location.hostname + siBaseDirectory + "/form";
   SiInternalBuildFormTracerQuery(form,list);
   siTagType = "extra";
   if (!closure)
      closure = function() { form.submit() };
   SiSyncTracer("fqcrty", "fqdrty", closure, siTracerTimeout);
   return false;
}

function SiTrackForm(filePath,form,list)
{
   siTracerPath = "http://" + document.location.hostname + (filePath == null ? siBaseDirectory + "/form" : filePath);
   SiInternalBuildFormTracerQuery(form,list);
   siTagType = "extra";
   SiDeferTracer("fqcrty","fqdrty");
   return true;
}

function SiInternalBuildFormTracerQuery(form,list)
{
   siTracerQuery = "";
   if (form.elements && (form.elements.length > 0))
   {
      var doamp=false;
      for (var i=0; i<form.elements.length; i++)
      {
         var el=form.elements[i];
         if (el.name)
         {
            var capture=(list == null);
            if (list != null)
            {
               for (var j=0; !capture && (j<list.length); j++)
               {
                  if (el.name == list[j]) capture = true;
               }
            }
            if (capture && (el.type == "radio")) capture = el.checked;
            if (capture)
            {
               if (doamp) siTracerQuery += "&";
               siTracerQuery += escape(el.name)+"="+((el.type=="checkbox") ? ((el.checked) ? "true" : "false") : escape(el.value));
               doamp=true;
            }
         }
      }
   }
}

function SiTrackTracer(filePath,queryString)
{
   SiInternalTrackTracer(filePath,queryString,null);
}

function SiSendAdditionalTracer(filePath,queryString)
{
   SiInternalTrackTracer(filePath,queryString,'trace');
}

function SiInternalTrackTracer(filePath,queryString,tagType)
{
   // If a base relative path prepend it with document protocol and hostname
   if (filePath.substring(0,1) == '/')
   {
      filePath = document.location.protocol + "//" + document.location.hostname + filePath;
   }
   siTracerPath = filePath;
   siTracerQuery = queryString;
   siTagType = tagType;
   SiSendTracer(false,"fqcrtgy","fqdrtgy");
}

function SiTrackPage(pageID)
{
   // If a page ID is supplied use it (ensuring it starts with /) as the path
   // but keep any existing query string
   if (pageID != undefined)
	{
	   if (pageID.indexOf('://') == -1)
	   {
	   	if(pageID.substring(0,1) != '/')
	        	pageID = '/' + pageID;

	     	pageID = document.location.protocol + "//" + document.location.hostname + pageID;
	   }

	   siRequest = pageID.indexOf('?') == -1 ? pageID + window.location.search : pageID + "&" + window.location.search.substring(1);
	}

	siTagType = "page";
	SiSendTracer(true,"pcrtgy","pdrtgy");
}

function SiInitPage()
{
   // Check if tagging server is specified
	if (siTaggingServer == null)
	{
		siTaggingServer = document.location.protocol + "//" + document.location.hostname + ":" + document.location.port;
	}
	else if (siTaggingServer.substring(0,4) != "http")
	{
		// Assume tagging server specifies hostname but pick up protocol+port from document location
		siTaggingServer = document.location.protocol + "//" + siTaggingServer + ":" + document.location.port;
	}

	siCookieValue = SiBuildCookie();
	SiSendDeferredTracers();
}

function SiSendDeferredTracers()
{
	var i = 0;
	var cookieName = siTracerCookieName + i++;
	var tracer = SiGetCookie(cookieName);
	while (tracer)
	{
		var Tracker = new Image();
   	Tracker.src = tracer;
      siImageSet[siImageSet.length] = Tracker;

      SiDeleteCookie(cookieName);

      cookieName = siTracerCookieName + i++;
      tracer = SiGetCookie(cookieName);
	}
}


SiInitPage();

if (window.siAutoTracer)
{
   SiTrackPage();
}
