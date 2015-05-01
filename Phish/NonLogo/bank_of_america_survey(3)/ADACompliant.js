// This script ensures ADA compliance by making the LivePerson invitation
// visible during page load. Original created by AL from LivePerson
//
// Change History
// --------------------------------------------------
// + 7.27.2007   
//  JA 
//  Added a timeout call to ensure the invitation is visible immediately
//  after the page loads, but resets the invitation to the hidden but correctly
//  sized state two seconds after page loads.
//  Replace calls to unsupported document.all with document.getElementById
// + 9.6.2007
//  JA
//  Much simpler method, just use transparent 1x1 images to start. 
// + 9.11.2007
//  JA
// Removed an unnecessary function call
// + 10.06.2007
// Added functionality to insert an invisible text element which 
// has the text of the invitation inside it
// + 10.12.2007
// Added insertion of spanish text when lpUASlanguage = spanish
var lpTinyImgSrc = "https://server.iad.liveperson.net/hcp/width/img1.gif";

document.getElementById("need_help").src = lpTinyImgSrc;
document.getElementById("need_close").src = lpTinyImgSrc;
document.getElementById("mylayer").style.visibility = "visible";

var lpInvitationTextDiv = document.createElement("div");
lpInvitationTextDiv.style.visibility = "hidden";
lpInvitationTextDiv.style.display = "none";

function buildSpanishInvitationText(assistanceSpecialtyText){
	return "Chat ahora con un specialista de banca en línea de Bank of America." +
	 "Hola, soy un especialista de Bank of America, que ayuda a los clientes con banca en linea." + 
	 "¿Hay algo que pueda ayudarle en el día de hoy a través de una sesión de chat en vivo seguro?";
}

// function to build the different text for the different
// types of instance to offer
function buildEnglishInvitationText(assistanceSpecialtyText){
	return "Title: Chat now with an Online Banking Professional" + 
    "Text:  Hello, I'm a Bank of America associate who helps customers " +
    assistanceSpecialtyText + " Online Banking." +
    "Is there anything I can help you with today, through a secure, live chat session?"
}

var lpInvitationTexts = {};
if(typeof(lpUASlanguage) != "undefined" && lpUASlanguage == "english"){
	lpInvitationTexts["olb-passcode"] = buildEnglishInvitationText("sign in to");
	lpInvitationTexts["olb-service-model"] = buildEnglishInvitationText("with");
	lpInvitationTexts["olb"] = buildEnglishInvitationText("enroll into");
	lpInvitationTexts["olb-billpay"] = buildEnglishInvitationText("pay bills with");
}else{
	lpInvitationTexts["olb-passcode"] = buildSpanishInvitationText("acceder a la");
	lpInvitationTexts["olb-service-model"] = buildSpanishInvitationText("con");
	lpInvitationTexts["olb"] = buildSpanishInvitationText("con");
	lpInvitationTexts["olb-billpay"] = buildSpanishInvitationText("con");
}

// inserts the invitation text in the anchor tag for the
// accept invitation link
if(document.getElementById("needRef") != null){
	document.getElementById("needRef").appendChild(lpInvitationTextDiv);
	
	var inviteText = lpInvitationTexts[lpUASunit];
	if(inviteText == null){
		inviteText = lpInvitationTexts["olb-service-model"];
	}
    lpInvitationTextDiv.innerHTML = inviteText;
}
