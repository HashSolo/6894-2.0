//<!--
//1@@m1

ebay.oDocument.oPage.onBeforeLoad=function()
{}
ebay.oDocument.oPage.onAfterLoad=function()
{var oP=window.parent;if(typeof(oP)!="object")
return;if(typeof(oP.document)=="unknown")
return;if(typeof(oP.ebay)=="undefined")
return;var oPD=oP.ebay.oDocument,oPC=oPD.getConfig("Security.Gif.Audio.Challenge"),oC,oD,oL;oD=this.parent;oC=oD.getConfig("Security.Gif.Bot.Helper");if(oPD&&oC)
oPD.getFormElem(oPC.sTokenStringId).value=oD.getFormElem(oC.sTokenStringId).value;oL=oPD._getControl(oPC.sBotListenLinkId);if(oL)
oL.enable(true);}
// b=12011051 -->