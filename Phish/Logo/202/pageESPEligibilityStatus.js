// Invoking a non-modal Dialog box object to show the Expanded Seller Protection eligibilty

  var myNonModalLightbox;
 
  YAHOO.util.Event.onAvailable("espEligibilty", function(){ 
	YAHOO.util.Dom.addClass("espEligibilty", "accessAid");
});
 
  YAHOO.util.Event.addListener("sellerprotection","click",function(e){ if(!myNonModalLightbox) myNonModalLightbox = new PAYPAL.util.Dialog("espEligibilty");
	  YAHOO.util.Event.stopPropagation(e);
	  YAHOO.util.Event.preventDefault(e); 
	  myNonModalLightbox.show();
	  }
	 );
  YAHOO.util.Event.addListener(document,"click",closeNonModalDialog);
  YAHOO.util.Event.addListener("closeNMD","click",closeNonModal);
  
  function closeNonModalDialog(event){
  
    event = event || window.event;
	var target = YAHOO.util.Event.getTarget(event);
	if(myNonModalLightbox)
		 if(myNonModalLightbox.panel && !YAHOO.util.Dom.isAncestor(myNonModalLightbox.panel.element, target)){ 
	 myNonModalLightbox.close();
	   }
}
  //This function is added to retain the cross image (at right top corner of the dailog box) functionality intact.
  function closeNonModal(event){if(myNonModalLightbox)myNonModalLightbox.close();}


  fnStopPageMove = function(e){
	//this will stop the movement when user clicks on balloon control link
	YAHOO.util.Event.preventDefault(e);
}
YAHOO.util.Event.addListener("bubbleText1", "click", fnStopPageMove);
YAHOO.util.Event.addListener("bubbleText2", "click", fnStopPageMove);
