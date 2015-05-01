//<!--
//1@@m16

function EbayUserIdOnePageOpt(pParent,pName,pCfg)
{if(!this.objType)
this.objType="EbayUserIdOnePageOpt";this.base=EbayHTML;this.base(pParent,pName,pName,false,pCfg);this.bReceivedResponse=false;this.init=function()
{with(this)
{var oP=parent,t=this,evtOnClick;t.oFrame=new EbayHTMLFrame(t,cfg.sFrameId);t.oContinueBtn=new EbayHTMLButton(t,cfg.sContinueButtonId);if(typeof(cfg.aParams)!="undefined")
{t.oParams=new Array();for(pi=0,len=cfg.aParams.length;pi<len;pi++)
{if(typeof(cfg.aParams[pi])!="undefined"&&cfg.aParams[pi]!="")
t.oParams[pi]=new EbayHTMLText(t,cfg.aParams[pi]);}}
t.oSmlThrob=new EbayHTMLLayer(this,cfg.sThrobone);t.oHelpMsgDiv=new EbayHTMLLayer(this,cfg.sUserSecDiv);t.oErrorIcon=new EbayHTMLLayer(this,("img_"+cfg.sUserIdTextElemName));t.oTopErrDiv=null;if(cfg.sTopErrDiv)
t.oTopErrDiv=new EbayHTMLLayer(this,cfg.sTopErrDiv);t.oUserIdLookupBtn=new EbayHTMLButton(t,cfg.sUserIdLookupBtnId);t.oUserIdLookupBtn.subscribeEvents("onmouseout","onmouseover");t.oUserIdLookupBtn.bind();t.oTickImg=new EbayHTMLImage(this,cfg.sTickImg);if(cfg.sEmailIdTextElemName)
{t.oEmail=new EbayHTMLText(t,cfg.sEmailIdTextElemName);t.sEmail="";}
t.oUserId=new EbayHTMLText(this,cfg.sUserIdTextElemName);t.oUserId.bind();t.oUserId.onblur=function()
{t.triggerReq();}
t.oMsgDiv=new EbayHTMLLayer(this,cfg.sErrorLayerId);t.oUserIdLookupBtn.onmouseover=function()
{t.bHalt=true;}
t.oUserIdLookupBtn.onmouseout=function()
{t.bHalt=false;}
t.oUserIdLookupBtn.onclick=function()
{t.bHalt=false;t.triggerReq();}
t.iRefreshCount=0;t.bHalt=false;t.reqInProg=false;t.oPSuggestions=new EbayUserIdSuggestions(t,"Registration.OnePage.UserIdSugg",this.oDocument.getConfig("Registration.OnePage.UserIdSugg"));t.oPSuggestions.init();var oCfg=this.oDocument.getConfig("Registration.OnePage.overlay");var pos=findPos(oUserId.eElem);oCfg.xPos=pos[0];oCfg.yPos=pos[1];t.oOverlay=new EbayUserIdOverlay(t,"Registration.OnePage.overlay",oCfg);t.oOverlay.init();if(cfg.bInitSuggOnLoad)
t.oPSuggestions.displaySuggestions(cfg.aUserIds);}}
this.triggerReq=function()
{with(this){if(!bHalt&&!oUserId.eElem.readOnly){if(!reqInProg){initRequest();}}}}
this.findPos=function(obj)
{var xPos=obj.offsetLeft;var yPos=obj.offsetTop;var tmp=obj.offsetParent;while(tmp!=null)
{xPos+=tmp.offsetLeft;yPos+=tmp.offsetTop;tmp=tmp.offsetParent;}
return[xPos,yPos+40];}
this.initRequest=function()
{with(this)
{if(oUserId.getValue()!="")
{oContinueBtn.enable(false);oUserIdLookupBtn.show(false);oUserIdLookupBtn.eElem.blur();oSmlThrob.show(true);}
sendRequest();}}
this.sendRequest=function()
{with(this)
{if(typeof(oOverlay)!="undefined"&&oOverlay.bIsOpen)
{bReceivedResponse=false;startRequest();}
else
{if(oUserId.getValue()!="")
{if(typeof(oEmail)!="undefined")extractEmailid(this);bReceivedResponse=false;oErrorIcon.show(false);oMsgDiv.show(false);if(this.oPSuggestions.bIsVisible)
this.oPSuggestions.clearMe();startRequest();}
else
{var oQWConfig=ebay.oDocument.getConfig('Registration.ErrorListings');if(!oQWConfig){oHelpMsgDiv.show(false);oErrorIcon.show(true);oMsgDiv.show(true);oMsgDiv.setClass(cfg.sErrClass);oMsgDiv.setValue(cfg.sEmptyErrorText);}
oContinueBtn.enable(true);oSmlThrob.show(false);oUserIdLookupBtn.show(true);}}
return false;}}
this.startRequest=function()
{with(this)
{reqInProg=true;var s=appendParams(cfg.sPostUrl);var elem=document.createElement('script');elem.src=s;document.body.appendChild(elem);startTimer();}}
this.appendParams=function(pPostUrl)
{with(this)
{var s=pPostUrl+"&"+cfg.sUserIdParam+"="+encodeURIComponent(oUserId.getValue());s+="&"+cfg.sCountParam+"="+iRefreshCount;if(cfg.sGlobalReg!=null)
{s+="&"+"globalreg"+"="+cfg.sGlobalReg;}
if(typeof(oOverlay)!="undefined"&&oOverlay.bIsOpen)
{var arr=oOverlay.aTexts;for(var pi=0,len=arr.length;pi<len;pi++)
{if(arr[pi])
s+="&"+arr[pi].name+"="+encodeURIComponent(arr[pi].getValue());}
if(typeof(oOverlay.aOvlyParams)!="undefined")
{var addParams=oOverlay.aOvlyParams;for(var i=0,len=addParams.length;i<len;i++)
{if(addParams[i])
s+="&"+addParams[i].name+"="+encodeURIComponent(addParams[i].getValue());}}}
else if(typeof(oParams)!="undefined")
{for(pi=0,len=oParams.length;pi<len;pi++)
{if(oParams[pi]&&oParams[pi].getValue()!="")
s+="&"+cfg.aParams[pi]+"="+encodeURIComponent(oParams[pi].getValue());}
if(typeof(sEmail)!="undefined"&&sEmail!="")
s+="&email="+sEmail;}
return s;}}
this.endRequest=function()
{with(this)
{oFrame.setSource(oGlobals.oEnvironment.sPicsDir+"s.gif");oContinueBtn.enable(true);oSmlThrob.show(false);reqInProg=false;}}
this.checkResponse=function()
{with(this)
{if(!bReceivedResponse)
{oUserIdLookupBtn.show(true);endRequest();}}}
this.startTimer=function()
{with(this)
{var t=(typeof(cfg.iTimeout)=="undefined")?40000:cfg.iTimeout;this.iTimer=parent.parent.win.setTimeout("ebay.oDocument.oPage._getControl(\""+name+"\").checkResponse()",t);}}
this.endTimer=function()
{this.parent.parent.win.clearTimeout(this.iTimer);}
this.processResponse=function(pConfig)
{with(this)
{if(typeof(oOverlay)!="undefined"&&oOverlay.bIsOpen)
{this.oOverlay.displayOlySugg(pConfig);}
else
{if(pConfig.bIsUserIdAvail)
{displaySuccess();this.oPSuggestions.oSuggestDiv.show(false);this.oPSuggestions.bIsVisible=false;}
else
{displayError(pConfig.sErrorTextChild);if(pConfig.aUserIds.length>0)
{iRefreshCount++;this.oPSuggestions.displaySuggestions(pConfig.aUserIds);}}}
endTimer();bReceivedResponse=true;if(typeof(pConfig.bResetCount)!="undefined"&&pConfig.bResetCount)
iRefreshCount=0;endRequest();}}
this.displaySuccess=function()
{with(this)
{oUserIdLookupBtn.show(false);oUserIdLookupBtn.eElem.blur();oHelpMsgDiv.show(false);oErrorIcon.show(false);if(oTopErrDiv!=null)
oTopErrDiv.show(false);oTickImg.show(true);oMsgDiv.setClass(cfg.sClass);oMsgDiv.setValue(cfg.sSuccessText);oMsgDiv.show(true);}}
this.displayError=function(msg)
{with(this)
{oUserIdLookupBtn.show(true);oErrorIcon.show(true);oMsgDiv.show(true);oMsgDiv.setClass(cfg.sErrClass);oMsgDiv.setValue(msg);oHelpMsgDiv.setClass("hideLabel");oTickImg.show(false);}}
this.placeSugg=function(userId)
{this.oUserId.setValue(userId);this.displaySuccess();this.oPSuggestions.clearMe();}
this.displayOverlay=function()
{this.oOverlay.show();}
this.writeCheckUserIdButton=function(pConfig)
{this.cfg=this.oConfig=pConfig;with(this)
{var oP=parent,oD=oP.oDocument,oG=oGlobals,oC=oG.oClient;this.sSpacerHtml='<img  src="'+oG.oEnvironment.sPicsDir+'s.gif" width="10" height="1" alt="">';if((oC.bIE&&oC.iVer>=5&&oC.bWin)||oC.bFirefox||oC.bSafari||(oC.bNav&&oC.iVer==6&&oC.fMinorVer>=0.2)||(oC.bNav&&oC.iVer>6)||(oC.bOpera&&oC.iVer>=7))
{oD.doc.write(sSpacerHtml+'<input type="hidden" value="'+cfg.sUserIdLookupBtnText+'" name="'+cfg.sUserIdLookupBtnId+'" style="padding:0 .25em 0 .25em; width:auto; overflow:visible;" id="'+cfg.sUserIdLookupBtnId+'">');_registerListener(oD._getEvent("load"),EVENT_BEFORE,"init");}}}
this.extractEmailid=function(pt)
{var emailid="";var oEmail=pt.oEmail;var pEmail=oEmail.getValue();var end=pEmail.indexOf("@");if(end==-1)
{this.sEmail=pEmail;return;}
emailid=pEmail.substring(0,end);this.sEmail=emailid;}}
new EbayUserIdOnePageOpt(ebay.oDocument.oPage,"Registration.OnePage.UserIdOptimization");function EbayUserIdOverlay(pParent,pName,pCfg)
{if(!this.objType)
this.objType="EbayUserIdOverlay";this.base=EbayHTML;this.base(pParent,pName,pName,false,pCfg);this.init=function()
{with(this)
{var t=this;t.cfg=pCfg;var oLayerParentiFrameShim=t.oLayerParentiFrameShim=new EbayHTMLOverlayContent(t,"UserIDPreview",t.cfg);t.oOvrLy=new EbayHTMLLayer(t,pCfg.OvlyId);t.oOvrLy.bind();t.oOverLyWidth=t.oOvrLy.eElem.style.width;t.oOverLyWidth=parseInt(t.oOverLyWidth.replace(new RegExp("px",""),""));var c=this.parent.parent.oGlobals.oClient;t.oOvrLy.eElem.style.top=pCfg.yPos+"px";t.oOvrLy.eElem.style.left=pCfg.xPos+"px";t.oTrob=new EbayHTMLLayer(t,pCfg.sOvlthrobId);t.oTrob.bind();t.aTexts=new Array();for(var i=0,len=pCfg.aOvltxtId.length;i<len;i++)
{t.aTexts[i]=new EbayHTMLText(t,pCfg.aOvltxtId[i]);t.aTexts[i].bind();t.aTexts[i].onblur=function(){if(this.getValue()!="")
{var p=this.parent;p.oPSuggestions.clearMe();p.oPSuggestions.oSuggestDiv.show(false);p.oTrob.show(true);p.parent.sendRequest();}}}
if(typeof(pCfg.aOvlyParams)!="undefined")
{t.aOvlyParams=new Array();for(var i=0,len=pCfg.aOvlyParams.length;i<len;i++)
{t.aOvlyParams[i]=new EbayHTMLText(t,pCfg.aOvlyParams[i]);t.aOvlyParams[i].bind();}}
t.oClose=new EbayHTMLAnchor(t,pCfg.sOvlyCloseId);t.oClose.bind();t.oClose.onclick=function()
{var p=this.parent;p.close();p.oOvrLy.eElem.style.width=p.oOverLyWidth+"px";p.parent.oUserId.focus(true);}
t.oShadow=new EbayHTMLLayer(t,pCfg.sOvlyShadow);t.oShadow.bind();if(typeof(pCfg.sUseridOverlayboxId)!="undefined")
{t.oUseridOverlayboxId=new EbayHTMLLayer(t,pCfg.sUseridOverlayboxId);t.oUseridOverlayboxId.bind();}
t.oMask=new EbayOverlayMask(ebay.oDocument.oPage,'lyrGrayout_sec');t.initSugg();t.bIsOpen=false;}}
this.initSugg=function()
{if(typeof(this.oPSuggestions)=="undefined")
{this.oPSuggestions=new EbayUserIdSuggestions(this,"Registration.OnePage.oLyUserIdSugg",this.oDocument.getConfig("Registration.OnePage.oLyUserIdSugg"));this.oPSuggestions.init();}}
this.displayOlySugg=function(inncfg)
{with(this)
{oTrob.show(false);oPSuggestions.displaySuggestions(inncfg.aUserIds);}}
this.show=function()
{with(this)
{oMask.show(true);oOvrLy.show(true);if(typeof(pCfg.sUseridOverlayboxId)!="undefined")
{oOvrLy.eElem.style.width=oUseridOverlayboxId.eElem.offsetWidth+'px';oOvrLy.eElem.style.height=oUseridOverlayboxId.eElem.offsetHeight+'px';oLayerParentiFrameShim.setIframeShim();oShadow.eElem.style.height=oUseridOverlayboxId.eElem.offsetHeight-10+'px';}
aTexts[0].focus(true);bIsOpen=true;}}
this.close=function()
{with(this)
{oTrob.show(false);oMask.show(false);oOvrLy.show(false);if(typeof(oPSuggestions)!="undefined")
oPSuggestions.clearMe();for(var i=0,len=aTexts.length;i<len;i++)
aTexts[i].setValue("");bIsOpen=false;}}}
function EbayUserIdSuggestions(pParent,pName,pCfg)
{if(!this.objType)
this.objType="EbayUserIdSuggestions";this.base=EbayHTML;this.base(pParent,pName,pName,false,pCfg);this.init=function()
{with(this)
{var t=this;t.cfg=pCfg;t.iMaxUserIds=pCfg.iMaxUserIds;t.iMaxPerSet=pCfg.iMaxPerSet;t.iNumberofSets=pCfg.iMaxUserIds/pCfg.iMaxPerSet;t.iCurrUserSet=-1;t.bIsVisible=false;t.bisOverlay=pCfg.bisOverlay;t.oSuggestDiv=new EbayHTMLLayer(t,pCfg.sSuggestDiv);t.oSuggestDiv.bind();t.oSugFirstAnc=new EbayHTMLAnchor(t,pCfg.sSugFirstAnc);t.oSugFirstAnc.bind();t.oUserIdLinks=new Array();t.oUserIdImgs=new Array();t.oUserIdDivs=new Array();for(var i=1;i<=iMaxUserIds;i++)
{t.oUserIdLinks[i]=new EbayHTMLAnchor(t,pCfg.sUserLinkId+i);t.oUserIdLinks[i].bind();t.oUserIdLinks[i].ind=i;t.oUserIdLinks[i].subscribeEvents("onmouseover","onmouseout");t.oUserIdLinks[i].onmouseover=function(){this.parent.highlight(this);}
t.oUserIdLinks[i].onmouseout=function(){this.parent.greyOutLinks(this);}
t.oUserIdLinks[i].onclick=function(){this.parent.onUseridSelect(this);}
t.oUserIdImgs[i]=new EbayHTMLImage(t,pCfg.sUserLinkImg+i);t.oUserIdImgs[i].bind();t.oUserIdImgs[i].ind=i;t.oUserIdImgs[i].onmouseover=function(){this.parent.highlight(this);}
t.oUserIdImgs[i].onmouseout=function(){this.parent.greyOutLinks(this);}
t.oUserIdImgs[i].onclick=function(){this.parent.onUseridSelect(this);}
t.oUserIdDivs[i]=new EbayHTMLLayer(t,pCfg.sUserIdDiv+i);t.oUserIdDivs[i].bind();t.oUserIdDivs[i].ind=i;t.oUserIdImgs[i].show(true);t.oUserIdDivs[i].show(true);}
t.oUserIdSet=new EbayHTMLLayer(t,pCfg.sUserIdSet);}}
this.displaySuggestions=function(auserIds)
{with(this)
{this.aSuggList=auserIds;var iMaxAvailIds=aSuggList.length;var iEmptySug=iMaxUserIds-iMaxAvailIds;for(var i=0;i<iMaxAvailIds;i++)
{if(oUserIdLinks[i+1]){oUserIdLinks[i+1].setText(aSuggList[i]);oUserIdDivs[i+1].show(true);}}
for(var i=1;i<=iEmptySug;i++){oUserIdDivs[i+iMaxAvailIds].show(false);}
if(!oUserIdSet.eElem){oUserIdSet.bind();}
oUserIdSet.show(true);oSuggestDiv.show(true);}}
this.displaySet=function(obj,isPrev,isFocus)
{with(this)
{var ind=obj.ind+1;if(isPrev)
ind=obj.ind-1;if(iCurrUserSet!=-1)
{oUserIdSet[iCurrUserSet].show(false);}
oUserIdSet[ind].show(true);if(oUserIdSet[ind].eElem.firstChild.firstChild)
{var sugEle=oUserIdSet[ind].eElem.firstChild.firstChild.lastChild;if(isFocus&&sugEle&&sugEle.nodeName=='SPAN'){sugEle.firstChild.focus(true);}}
iCurrUserSet=ind;}}
this.disableNext=function(ind)
{with(this)
{}}
this.resetNext=function(ind)
{with(this)
{}}
this.highlight=function(obj)
{with(this)
{oUserIdLinks[obj.ind].eElem.style.textDecoration="underline";oUserIdImgs[obj.ind].source(cfg.sBlueImg);}}
this.greyOutLinks=function(obj)
{with(this)
{oUserIdLinks[obj.ind].eElem.style.textDecoration="";oUserIdImgs[obj.ind].source(cfg.sGreyImg);}}
this.onUseridSelect=function(obj)
{if(this.bisOverlay)
{this.parent.close();this.clearMe();this.parent.parent.placeSugg(this.aSuggList[obj.ind-1]);}
else
this.parent.placeSugg(this.aSuggList[obj.ind-1]);}
this.clearMe=function()
{with(this)
{for(var i=1;i<=iMaxUserIds;i++)
{oUserIdLinks[i].setText("");oUserIdDivs[i].show(false);}
for(var j=1;j<=iNumberofSets;j++)
resetNext(j);if(iCurrUserSet!=-1)
{oUserIdSet[iCurrUserSet].show(false);iCurrUserSet=-1;}
oSuggestDiv.show(false);bIsVisible=false;}}}
ebay.oDocument.oPage.handleResponse=function(bIsUserIdAvail,aUserIds,sErrorTextChild)
{var c=new Object();c.bIsUserIdAvail=bIsUserIdAvail;c.aUserIds=aUserIds;for(var i=0,len=c.aUserIds.length;i<len;i++)
c.aUserIds[i]=c.aUserIds[i].trim();c.sErrorTextChild=sErrorTextChild;this._getControlEx("Registration.OnePage.UserIdOptimization")._exec("processResponse",c);}

//2@@m5

function EbayOverlayMask(pParent,pName,pCfg)
{if(!this.objType)
this.objType="EbayOverlayMask";this.base=EbayHTMLLayer;this.base(pParent,pName,pCfg);this.bodyLoc=ebay.oDocument.doc.getElementsByTagName("body")[0];this._registerListener(ebay.oDocument._getEvent('resize'),this.EVENT_AFTER,'resize');this.aSelElems=this.oDocument.doc.getElementsByTagName('SELECT');this.baseShow=this.show;this.show=function(pShow)
{with(this)
{for(var index=0,len=aSelElems.length;index<len;index++)
{aSelElems[index].disabled=pShow;}
var st=(pShow)?'block':'',db=document.body,bsel=(pShow)?true:false,bShown=false;top(0);left(0);width(ebay.oDocument.doc.body.scrollWidth+'px');height(ebay.oDocument.doc.body.scrollHeight+'px');with(eElem.style)
{bShown=display!='none';display=st;opacity=0.35;filter='alpha(Opacity=35)';}
baseShow(pShow,true);}}
this.resize=function()
{if(this.eElem&&this.eElem.style.display=='block')
{this.width(ebay.oDocument.doc.body.scrollWidth+'px');this.height(ebay.oDocument.doc.body.scrollHeight+'px');}}
this._createElement=function()
{with(this)
{if(this.getElem())
return;var mask=oDocument.doc.createElement('DIV');mask.setAttribute('id',name);with(mask.style)
{position='absolute';display='none';top='0px';left='0px';zIndex=2;}
bodyLoc.appendChild(mask);bind();}}
this._createElement();}

//3@@m6

function EbayHTMLLayer(pParent,pName,pDisabled,pCfg)
{if(!this.objType)
this.objType="EbayHTMLLayer";this.base=EbayHTML;this.base(pParent,pName,pName,pDisabled,pCfg);this.aBindEvents=new Array;this.getElem=ebHTMLLayerGetElem;this.getValue=ebHTMLLayerGetValue;this.setValue=ebHTMLLayerSetValue;}
function ebHTMLLayerGetElem(pName)
{var s=pName,d=this.oDocument.doc;if(d.getElementById)
return d.getElementById(s);else if(d.all)
return d.all(s);this.throwWarning("Not supported","getElem");}
function ebHTMLLayerGetValue(pIsText)
{if(this.eElem)
{if(pIsText)
{if(this.oDocument.oGlobals.oClient.bFirefox)
return this.eElem.textContent;else
return this.eElem.innerText;}
else
return this.eElem.innerHTML;}
else
return"";}
function ebHTMLLayerSetValue(pVal,pIsText)
{if(this.eElem)
{if(pIsText)
{if(this.oDocument.oGlobals.oClient.bFirefox)
this.eElem.textContent=pVal;else
this.eElem.innerText=pVal;}
else
this.eElem.innerHTML=pVal;}}

//4@@m22

function EbayHTMLOverlay(pParent,pName,pCfg,pDisabled)
{if(!this.objType)
this.objType="EbayHTMLOverlay";this.base=EbayHTMLLayer;this.base(pParent,pName,pDisabled,pCfg);this.isSupported=ebIsBrowserSupported;if(!this.isSupported())
return;this.sPosStyle=pCfg.posStyle||'absolute';this.sTop=pCfg.top;this.iTopPadding=0;this.iLeftPadding=0;this.sLeft=pCfg.left;this.sWidth=pCfg.width?parseInt(pCfg.width):0;this.sHeight=pCfg.height?parseInt(pCfg.height):0;this.sLayerHTML=pCfg.layerHTML||"";this.sContentDiv=pCfg.contentDiv||"";this.bForceReposition=pCfg.bForceReposition||false;this.bNoSetContent=pCfg.bNoSetContent;this.bClearValueOnClose=typeof(pCfg.bClearValueOnClose)!='undefined'?pCfg.bClearValueOnClose:true;this.bCustomHTML=pCfg.customHTML||false;this.bTransparent=pCfg.transparent||false;this.setPosition=ebHTMLOverlaySetPosition;this.centerTop=ebHTMLOverlayCenterTop;this.centerLeft=ebHTMLOverlayCenterLeft;this.setContent=ebHTMLOverlaySetContent;this.closeOverlay=this.close=ebHTMLOverlayCloseOverlay;this.display=ebHTMLOverlayDisplay;}
function ebHTMLOverlayDisplay(pContent)
{with(this)
{if(!eElem)
bind();if(!bNoSetContent)
setContent(pContent);setPosition();show(true);if(this.oConfig.bOnFocus&&(!(oGlobals.oClient.bIE&&oGlobals.oClient.iVer<7)))
{var sFirstInsElem="sFirstInsElem";var oFirstEle=this.eElem.firstChild;if(oFirstEle.id!=sFirstInsElem)
{var anchorTag=document.createElement('a');anchorTag.href="javascript:void(0);";anchorTag.id=sFirstInsElem;var sLyrName=this.oConfig.sStartLyrName;if(sLyrName)
anchorTag.innerHTML="<b class='g-hdn'>"+sLyrName+"</b>";this.eElem.insertBefore(anchorTag,oFirstEle);anchorTag.focus();}
else
oFirstEle.focus(true);}}}
function ebHTMLOverlaySetPosition()
{with(this)
{if(sPosStyle.is('absolute'))
{if(!sTop||bForceReposition)
centerTop();if(!sLeft||bForceReposition)
centerLeft();top(sTop);left(sLeft);}}}
function ebHTMLOverlayCenterTop()
{with(this)
{var oD=oDocument,winHeight=oD.doc.body.clientHeight,cL=oGlobals.oClient;if(!cL.bIE)
winHeight=oD.win.innerHeight;else if(typeof(winHeight)=='undefined'&&cL.iVer>=6)
winHeight=oD.doc.documentElement.clientHeight;var s=oD.doc.body.scrollTop+(winHeight-sHeight)/2;if(document.documentElement)
s+=document.documentElement.scrollTop;sTop=(parseInt(s)+iTopPadding)+'px';return s;}}
function ebHTMLOverlayCenterLeft()
{with(this)
{var winWidth=document.body.clientWidth,cL=oGlobals.oClient;if(!cL.bIE)
winWidth=window.innerWidth;var s=winWidth/2-sWidth/2;sLeft=(parseInt(s)+iLeftPadding)+'px';return s;}}
function ebHTMLOverlaySetContent(pContent)
{with(this)
{if(sContentDiv!='')
{var oL=new EbayHTMLLayer(this,sContentDiv);oL.bind();oL.setValue(pContent);}
else
setValue(pContent);}}
function ebHTMLOverlayCloseOverlay()
{with(this)
{if(bClearValueOnClose)
{var cts=this.controls;if(sContentDiv!=''&&cts[sContentDiv])
{cts[sContentDiv].setValue('&nbsp;');}
else
{setValue('&nbsp;');if(this.oConfig.bOnFocus&&this.oSelectedElem)
{ele=this.controls[this.oSelectedElem.name];if(ele)
ele.focus(true);}}}
show();if(!this.oGlobals.oClient.bFirefox)
cleanupMemory();}}
function ebIsBrowserSupported()
{var cL=this.oGlobals.oClient,bNS4=document.layers;if(bNS4||(cL.bMac&&!cL.bMacppc&&!cL.bMactel))
return false;return true;}

//5@@m11

function EbayHTMLOverlayContent(pParent,pName,pCfg)
{if(!this.objType)
this.objType="EbayHTMLOverlayContent";this.base=EbayHTMLOverlay;this.base(pParent,pName,pCfg);if(!this.isSupported())
return null;this.bUseIfShim=pCfg.useIfShim||false;this.sContent=pCfg.contentHTML;this.sLayerUI=pCfg.layerHTML;this.iBorderWidth=pCfg.iBorderWidth||0;this.hide=this.closeOverlay;this.setIframeShim=ebHTMLOverlayContentSetIframeShim;this.displayBase=this.display;this.display=ebHTMLOverlayContentDisplay;this.closeBase=this.close;this.closeOverlay=this.close=ebHTMLOverlayContentClose;var cl=this.oGlobals.oClient;this.bUseIfShim=(pCfg.useIfShim&&(cl.bIE&&!cl.bMac));this.eIframeShim=null;this.setContentBase=this.setContent;this.setContent=ebHTMLOverlayContentSetContent;}
function ebHTMLOverlayContentDisplay(pContent)
{with(this)
{displayBase(sContent||pContent);if(bUseIfShim)
setIframeShim();}}
function ebHTMLOverlayContentSetIframeShim()
{with(this)
{if(eElem)
{var d=oDocument,isRel=sPosStyle=='relative',e=isRel?e.firstChild:eElem;var w=width(),h=height(),f=eIframeShim=d.createElement('IFRAME'),bw=iBorderWidth;with(f.style)
{position='absolute';top=isRel?'20px':(0-bw)+'px';left=isRel?'20px':(0-bw)+'px';zIndex='-1';width=(w+bw)+'px';height=(h+bw)+'px';}
f.frameBorder='0';f.title=' ';f.src=oGlobals.oEnvironment.sPicsDir+'s.gif';e.appendChild(f);}}}
function ebHTMLOverlayContentSetContent(pContent)
{this.sContent=pContent;this.setContentBase(pContent);}
function ebHTMLOverlayContentClose()
{with(this)
{if(bUseIfShim&&eIframeShim&&eElem)
eElem.removeChild(eIframeShim);closeBase();}}

//6@@m1

function EbayFieldValidation(pParent,pName,pCfg)
{if(!this.objType)
this.objType="EbayfieldValidation";this.base=EbayHTML;this.base(pParent,pName,pName,false,pCfg);var bBlur=false;var aFlds=pCfg.aFlds;var sErrPrf=pCfg.sErrPrf;var sErrImg=pCfg.sErrImg;this.oTxtVal=[];this.oEmptChk;this.oEmptImg;this.init=function()
{with(this)
{for(var json in aFlds)
{oTxtVal[json]=new EbayHTMLText(this,aFlds[json]);if(!oTxtVal[json].eElem)
oTxtVal[json].bind();oEmptChk=oTxtVal[json].oEmptChk=new EbayHTMLLayer(this,sErrPrf+aFlds[json]);if(!oEmptChk.eElem)
oEmptChk.bind();oEmptImg=oTxtVal[json].oEmptImg=new EbayHTMLImage(this,sErrImg+aFlds[json]);if(!oEmptImg.eElem)
oEmptImg.bind();oTxtVal[json]._registerEvent("onblur","parent.txtEmptyChk");}}}
this.init();this.txtEmptyChk=function()
{var sTxtVal=this.getValue();with(this)
{if(sTxtVal=="")
{oEmptChk.show(true);oEmptImg.show(true);}
else
{oEmptChk.show(false);oEmptImg.show(false);}}}}
// b=12011051 -->