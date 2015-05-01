// The onload event of visible frame runs this function.
function initialiseResolver()
{
   // Add jsEnabled parameter to default form action
   var currentAction = document.getElementById("DefaultFormAction").value;
   document.getElementById("DefaultFormAction").value = currentAction + "&jsEnabled=true";

   // Set action of loginForm to ensure logic in executeResolver() runs when log on button is selected
   document.getElementById("SRRUNNING").value="FALSE";
}

var hiddenFrameLoaded;

// This function is run after ValidateForm is successful
function executeResolver(evt)
{
   // Get reference to login form on main frame
   var theForm = document.loginForm;
   var strAction = document.getElementById("SRRUNNING").value;
   
   if(strAction.indexOf("FALSE")>= 0)
   {
      // Disable Sign On button on main frame
      parent.main.document.getElementById("SignonButton").disabled = true;

      // Ensure submit event is blocked for Internet Explorer
      if (window.event) {
         window.event.returnValue = false;
      }
 
      // Ensure submit event is blocked for Netscape/Firefox/Safari
      if (evt) {
         evt.returnValue = false;
         evt.cancel = true;
         if(evt.preventDefault)
         evt.preventDefault();
      }
 
      // change action of form to ensure this logic does not run again
      document.getElementById("SRRUNNING").value="TRUE";

      // Extract CorporateSignonCorpId (CRN) from main frame
      var CorpID = parent.main.document.loginForm.CorporateSignonCorpId.value;
	  
      // Set source of data frame (will cause submit)
      var ResolverUrl = parent.main.document.loginForm.ServiceResolverURL.value;
      //ResolverUrl = ResolverUrl + "&jsEnabled=true"
      //ResolverUrl = ResolverUrl + "&CorporateSignonCorpId=" + CorpID; 
      // Clear src first incase already loaded (if window not closed from previous login. Netscape/Firefox won't reload data frame if CRN is same)
      // window.parent.document.getElementById("data").src = "hiddenframe.asp";
	  hiddenFrameLoaded = false;
	  parent.data.document.getElementById("EWF_ENTRY_CORPORATESIGNONCORPID").value = CorpID;
	  parent.data.document.HIDDENFORM.submit();

      //window.parent.document.getElementById("data").src = ResolverUrl;

      // Check if loaded after configured timout value
      window.checkServiceResolverLoaded = checkServiceResolverLoaded; // this gets around an IE issue with setTimeout
      var hiddenFrameTimeout = parent.main.document.loginForm.HiddenFrameTimeout.value;
      if (hiddenFrameTimeout > 0) {
         setTimeout("checkServiceResolverLoaded();", hiddenFrameTimeout);
      }
      else {
         // Use default timeout period if none was configured
         setTimeout("checkServiceResolverLoaded();", 10000);
      }
   }
}

// This function checks to see if the service resolver document has loaded into the hidden frame
function checkServiceResolverLoaded() 
{
   // If service resolver is not loaded
   if (hiddenFrameLoaded != true)
   // if (window.parent.data.loaded != true)
   {
      // Clear the hidden frame document (to prevent it loading after we've given up on it)
      window.parent.document.getElementById("data").src = "hiddenframe.asp"

      // Enable submit button
      parent.main.document.getElementById("SignonButton").disabled=false; 

      // Re-initialise the service resolver functionality
      initialiseResolver();

      // Display error message
      alert('A problem has occurred connecting to Internet Banking. Please try again.');
   } 
}

// useTraget() sets action of loginForm to target, enables log on button and fires button click event
function useTarget(target)
{
   // Set target for loginForm to submit to
   parent.main.document.loginForm.action = target;

   // Enable submit button
   parent.main.document.getElementById("SignonButton").disabled=false; 

   // Fire button submit event (assumes all pages have their own SubmitEBS function
   SubmitEBS('Modify');
}

// This method runs onload of the page returned by the hidden frame.
function resolveService()
{

	hiddenFrameLoaded = true;
   // Get status from element in jsp in data frame
   var status = String(parent.data.document.getElementById("ServiceResolverStatus").value);

   // Check status
   if (status.toUpperCase() == "FINACLE") 
   {
      // Get url from hidden field in main frame
      var url = parent.main.document.getElementById("DefaultFormAction").value; 
      // Submit to Url
      completeFormSubmission(url);
   }
   else if (status.toUpperCase() == "EDIFY")
   {
      // Copy login data to appropriate fields and submit
      parent.main.document.loginForm.USERID.value = parent.main.document.loginForm.CorporateSignonCorpId.value;
      parent.main.document.loginForm.PIN.value = parent.main.document.loginForm.CorporateSignonPassword.value;
      // Get url from element in jsp in hidden frame
      var url = parent.data.document.getElementById("ServiceResolverUrl").value; 
	  // Add box number to URL
	  url  = url + jBox;

	  // Submit to Url
	  if (window.parent.data.srerror == true)
      {	
	      parent.main.document.loginForm.EXTRA2.value = "SRERROR";
      }
      completeFormSubmission(url);
   } 
   else 
   {
      // An error has occurred and been returned by the business layer
      // Get url from element in jsp in hidden frame
      var url = parent.data.document.getElementById("ServiceResolverUrl").value; 
      // Submit to Url
      completeFormSubmission(url);
   }
} 

// This function finally submits the login form
function completeFormSubmission(target)
{
   useTarget(target);
   if(document.loginForm.target=='ANZLOGON_1')
   {
      window.setTimeout('resetPage();',1500);
   }
}

function resetPage() {
   // Clear the hidden frame document (to prevent it loading after we've given up on it)
   var hiddenFrame = window.parent.document.getElementById("data").src;
   window.parent.document.getElementById("data").src = hiddenFrame;

   // Reload the page to reset it to original state
   // window.parent.document.location.reload();
   initialiseResolver();
}