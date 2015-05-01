
        var home_entity = "HOME";
	var business_entity = "BUSINESS";
	var corperate_entity = "CORPERATE";
        var general_entity = "GENERAL";

	var destination = "";
	var destinationURL = "";
	var logoffCommand = "";

	function invokePublicWarning(newEntity, entityURL) {
		destination = newEntity;
		destinationURL =entityURL;			
		leaveFusedSite(entityURL);
        }

	var loggedOn = false;

	function setLoggedOn() {
		loggedOn=true;
	}

	function setLoggedOff() {
		loggedOn=false;
	}

