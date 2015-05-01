//<!--
//1@@m18

function EbayToolbar(pParent,pName,pCfg)
{if(!this.objType)
this.objType="EbayToolbar";this.baseObject=EbayBaseControl;this.baseObject(pParent,pName);this.TBDaemonID="eBayToolbarCommLib.IWebEvent.1";this.sTBHelper="eBayToolbar.Helper";var g=this.oGlobals;var env=this.oEnv=g.oEnvironment;this.isInstalled=false;this.client=g.oClient;this.config=pCfg||(new EBayConfig(pName));new EbayHTMLForm(this,this.config.signinFormID);this.qualifyPageType=function(){with(this)
{var t=true;for(var i=0;i<aPageTypes.length;i++)
{if(nPageType==aPageTypes[i])
{t=false;}}}
return t;}
this.onSignIn=function()
{var oF,PageType,sAction,u,sID,oSF,oCfg=this.config,oNI;oF=this.controls[oCfg.signinFormID];PageType=0;this.nPageType=oF.getElementValue("pageType");this.aPageTypes=oCfg.pageTypes.split(",");oSF=document.forms[oCfg.signinFormID];if(this.nPageType!=""&&this.qualifyPageType())
{if(this.activeXSupported(this.TBDaemonID))
{if(oSF!=null)
{sAction=new String(oSF.action);if(this.oDocument.doc.location.protocol.has("https"))
sAction=sAction.replace("http:","https:");oSF.action=sAction;}
u=oF.getElementValue("userid");TBDaemon=new ActiveXObject(this.TBDaemonID);sID=TBDaemon.GetSIDForUser(u);if(sID!="")
{this.createElement(oSF,'runame',env.sCountry.hasAny("cn")?'Soda1-EBAYTOOLBAK7I2R-mwwbb':'EBAYTOOLBAR');if(typeof(oSF.sid)=="undefined")
{this.createElement(oSF,'SID',sID);}}}}}
this.createElement=function(oF,n,v)
{var o=document.createElement("input");o.type='hidden';o.name=n;o.value=v;oF.appendChild(o);}
this.activeXSupported=function(n)
{var oC=this.client;return(!(oC.bMac||oC.bMacppc)&&oC.bIE)&&this.client.activeXLibLoaded(n);}
this.refresh=function()
{with(this)
{var oTBH,oTB,done,oCfg=config;if(activeXSupported(sTBHelper))
{oTBH=new ActiveXObject(sTBHelper);done=oTBH.doSomething(oCfg.toolbarCommand+"?00="+oCfg.eBayUserId+"&05="+oCfg.eBayItemId);}
if(activeXSupported(TBDaemonID))
{oTB=new ActiveXObject(TBDaemonID);oTB.WatchListChanged();}}}
this.refreshListItem=function()
{with(this)
{var oTB;if(activeXSupported(TBDaemonID))
{oTB=new ActiveXObject(TBDaemonID);if(typeof(oTB.OnItemListed)=="undefined"||typeof(oTB.OnItemListed)==false||typeof(oTB.OnItemListed)=="unknown")return;oTB.OnItemListed();}}}}
function EBayToolbarConfig(name)
{if(!this.objType)
this.objType="EBayToolbarConfig";this.base=EbayConfig;this.base(name);}

//2@@m11

function EbayHTMLPopup(pParent,pName,pCfg)
{if(!this.objType)
this.objType="EbayHTMLPopup";this.base=EbayBaseControl;this.base(pParent,pName);this.oConfig=pCfg||null;if(!pCfg)
{this.sUrl="";this.iWidth=this.iHeight=this.iLeft=this.iTop=null;this.bToolbar=this.bLocation=this.bStatus=this.bScrollbars=this.bResizable=this.bMenubar=true;}
else
{var u="undefined";this.sUrl=typeof(pCfg.sUrl)!=u?pCfg.sUrl:"";this.iWidth=typeof(pCfg.iWidth)!=u?pCfg.iWidth:null;this.iHeight=typeof(pCfg.iHeight)!=u?pCfg.iHeight:null;this.iLeft=typeof(pCfg.iLeft)!=u?pCfg.iLeft:null;this.iTop=typeof(pCfg.iTop)!=u?pCfg.iTop:null;this.bToolbar=typeof(pCfg.bToolbar)!=u?pCfg.bToolbar:true;this.bLocation=typeof(pCfg.bLocation)!=u?pCfg.bLocation:true;this.bStatus=typeof(pCfg.bStatus)!=u?pCfg.bStatus:true;this.bScrollbars=typeof(pCfg.bScrollbars)!=u?pCfg.bScrollbars:true;this.bResizable=typeof(pCfg.bResizable)!=u?pCfg.bResizable:true;this.bMenubar=typeof(pCfg.bMenubar)!=u?pCfg.bMenubar:true;}
this.oWin=null;this.sProps=this.sCustomProps="";this.bModal=false;this.sSavedFocusFunction=null;this.iHBuffer=this.iWBuffer=0;this.show=ebHTMLPopupShow;this.getParamString=ebHTMLGetParamString;this.modality=ebHTMLModality
this.showEx=ebHTMLPopupShowEx;this.resizeParent=ebHTMLPopupResizeParent;this.close=ebHTMLPopupClose;this.focus=ebHTMLPopupFocus;this.sizeToContent=ebHTMLPopupSizeToContent;this.clearControls=ebHTMLPopupClearControls;}
function ebHTMLPopupShow(pIsPopUnder)
{var bPopUnder=(typeof(pIsPopUnder)!="undefined"&&pIsPopUnder)?true:false;with(this)
{if(sUrl.length==0)
return null;var sP=getParamString();var oD=oDocument,tWin=oD.win;oD.setGlobalParent(this);modality(tWin);if(sUrl.has("ej2child=true")&&!sUrl.has("ej2parent="))
sUrl+="&ej2parent="+name;var w=tWin.open(sUrl,name,sP);if(bPopUnder)
w.blur();if(w&&!bPopUnder)
w.focus();oWin=w;return w;}}
function ebHTMLGetParamString()
{with(this)
{sP=(iWidth!=null)?",width="+iWidth:"";sP+=(iHeight!=null)?",height="+iHeight:"";sP+=(iLeft!=null)?",screenX="+iLeft+",left="+iLeft:"";sP+=(iTop!=null)?",screenY="+iTop+",top="+iTop:"";sP+=",toolbar="+((bToolbar)?"1":"0");sP+=",location="+((bLocation)?"1":"0");sP+=",status="+((bStatus)?"1":"0");sP+=",scrollbars="+((bScrollbars)?"1":"0");sP+=",resizable="+((bResizable)?"1":"0");sP+=",menubar="+((bMenubar)?"1":"0");sP+=(sCustomProps.length>0)?","+sCustomProps:"";if(sP.length>0)
sP=sP.substring(1);sProps=sP;return sP;}}
function ebHTMLModality(pWin)
{if(pWin)
{with(this)
{if(bModal)
{pWin.g_ebPopupObject=this;sSavedFocusFunction=pWin.onfocus;pWin.onfocus=function()
{try
{g_ebPopupObject.focus();}
catch(e)
{}}}}}}
function ebHTMLPopupShowEx(pUrl,pWidth,pHeight,pToolbar,pLocation,pStatus,pScrollbars,pResizable,pMenubar,pLeft,pTop,pCustomsProps,pModal,pWBuffer,pHBuffer)
{with(this)
{if(pUrl)
sUrl=pUrl;iWidth=pWidth;iHeight=pHeight;iLeft=pLeft;iTop=pTop;bToolbar=pToolbar;bLocation=pLocation;bStatus=pStatus;bScrollbars=pScrollbars;bResizable=pResizable;bMenubar=pMenubar;if(pCustomsProps)
sCustomProps=pCustomsProps;bModal=pModal;iHBuffer=pHBuffer;iWBuffer=pWBuffer;return show();}}
function ebHTMLPopupResizeParent(pX,pY,pW,pH)
{var p=this.parent;if(p)
{if(!isNaN(pX)&&!isNaN(pY))
p.moveTo(pX,pY);if(!isNaN(pW)&&!isNaN(pH))
p.resizeTo(pW,pH);}}
function ebHTMLPopupClose()
{with(this)
{if(bModal)
oDocument.win.onfocus=sSavedFocusFunction;oDocument.closeWindow(oWin);}
this.clearControls();}
function ebHTMLPopupClearControls()
{this.controls=[];}
function ebHTMLPopupFocus()
{var w=this.oWin;if(w&&!w.closed)
w.focus();else
this.close();}
function ebHTMLPopupSizeToContent()
{with(this)
{var c=oGlobals.oClient;if(!(c.bNav&&(c.iVer<5)))
{var ims=oWin.document.images,len=ims.length;var bottom=0,right=0,cB,cR;for(var i=0;i<len;i++)
{cB=ims[i].offsetTop+ims[i].offsetHeight;cR=ims[i].offsetLeft+ims[i].offsetWidth;if(cB>bottom)bottom=cB;if(cR>right)right=cR;}
oWin.resizeTo(right+iWBuffer,bottom+iHBuffer);}}}

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

//4@@m6

function EbayHTMLButton(pParent,pElemName,pDisabled,pCfg)
{if(!this.objType)
this.objType="EbayHTMLButton";this.base=EbayHTMLFormElem;this.base(pParent,pElemName,pDisabled,pCfg);this.getValue=ebHTMLButtonGetValue;this.setValue=ebHTMLButtonSetValue;this.enableBase=this.enable
this.enable=ebHTMLButtonEnable;this.subscribeEvents("onclick");}
function ebHTMLButtonGetValue()
{return this.eElem.value;}
function ebHTMLButtonSetValue(pValue)
{var e=this.eElem;if(e)
e.value=pValue;}
function ebHTMLButtonEnable(pEnable,pYukonize)
{if(typeof(pYukonize)!=='undefined'&&pYukonize)
{var e=this.eElem;e.style.opacity=!pEnable?".5":"";e.style.filter=!pEnable?"alpha(opacity=50)":"";this.bBtnDisabled=!pEnable;}
else
this.enableBase(pEnable);}

//5@@m10

function EbayHTMLAnchor(pParent,pName,pDisabled,pCfg)
{if(!this.objType)
this.objType="EbayHTMLAnchor";this.base=EbayHTML;this.base(pParent,pName,pName,pDisabled,pCfg);this.getElem=ebHTMLAnchorGetElem;this.enableBase=this.enable;this.enable=ebHTMLAnchorEnable;this.subscribeEvents("onclick");}
function ebHTMLAnchorGetElem(pName)
{var d=this.oDocument.doc,l=null;l=d.links[pName];if(l)return l;if(d.getElementById)
l=d.getElementById(pName);if(l)return l;if(d.all)
l=d.all[pName];if(l)return l;if(d.layers)
{var lyrs=d.layers;var len=lyrs.length;for(var i=0;i<len;i++)
{l=this.getElem(lyrs[i].document,pName);if(l)
return l;}}
for(var j=0;j<d.links.length;j++)
{l=d.links[j];if(typeof(l.name)=="undefined")
{if(l.onclick)
{var oc=l.onclick.toString();if(oc.indexOf("{#"+pName+"#}")!=-1)
return l;}}
else
{if(l.name==pName)
return l;}
l=null;}
return l;}
function ebHTMLAnchorEnable(pEnable)
{var cur=(pEnable)?"hand":"default";var el=this.eElem;if(el&&el.style)
{el.style.cursor=cur;el.style.color=pEnable?"":"gray";}
this.enableBase(pEnable);}
function setEbayLink(pS)
{return true;}

//6@@m2

ebay.oDocument.oPage.initASQoverview=function()
{var c=this.parent.getConfig('Ask_Seller_Question_Overview_Config');if(c)
{this.oAllQuestionsLayer=new EbayHTMLLayer(this,c.sAllQuestionsLayer,false,c);this.oLinkLayer=new EbayHTMLLayer(this,c.sLinkLayer,false,c);this.oShowQuestionsLink=new EbayHTMLAnchor(this,c.sShowAllLink,false,c);this.oShowQuestionsLink.onclick=function()
{var oP=this.parent;oP.showQuestionsLayer(true);oP.showLinkLayer(false);}}}
ebay.oDocument.oPage.showLinkLayer=function(bBoolean)
{this.oLinkLayer.show(bBoolean);}
ebay.oDocument.oPage.showQuestionsLayer=function(bBoolean)
{this.oAllQuestionsLayer.show(bBoolean);}
ebay.oDocument.oPage.setText=function()
{if(this.oShowQuestionsLink&&this.oShowQuestionsLink.eElem)
this.oShowQuestionsLink.eElem.innerHTML=this.oShowQuestionsLink.oConfig.sShowAllLinkText;}
ebay.oDocument.oPage.initASQAfterLoad=function()
{this.setText();}

//7@@m7

ebay.oDocument.oPage.toolbarASQcall=function()
{var oC=this.oGlobals.oClient;if(oC.bMac||oC.bMacppc)
return false;var tb=new EbayToolbarDetect(this,"toolbar");if(tb.isEbayToolbarEnabled())
{var xToolbar=new ActiveXObject(tb.sActiveXLib_V2);if(typeof(xToolbar.onASQ)=="undefined"||typeof(xToolbar.onASQ)=="unknown"||xToolbar.onASQ==false)
return;xToolbar.onASQ();}}
ebay.oDocument.oPage.onBeforeLoad=function()
{this.initASQoverview();var c=this.parent.getConfig("simpleConfig");if(c)
{var oF=new EbayHTMLForm(this,c.formId);var oS=new EbayHTMLSelect(oF,c.selectId);oF.onsubmit=function()
{var oD=this.oDocument;var eMsg=oD.getFormElem('message');var eDTxt=oD.getFormElem('defaultText');if(eDTxt.value==eMsg.value)
eMsg.value='';var eSubmit=oD.getFormElem('Submit');if(typeof(eSubmit.disabled)!='undefined')
eSubmit.disabled=true;}
oS.onchange=function()
{var selValue=this.getSelectedValue();if(selValue==5)
{var oF=this.parent;oF.setAction(c.path);oF.submit();}}}
c=this.parent.getConfig("ebaytoolbar");if(c)
{if(c.sMethodToCall.indexOf("onASQ")!=-1)
this.toolbarASQcall();}
c=this.parent.getConfig("MyMessages.RespondToQuestion.ShowHide");if(c)
new EbayMessagesQuestionResponseShowHide(this,"EbayMessagesQuestionResponseShowHide",c);var oCfg=this.parent.getConfig("ASQ.SkypeTab");if(oCfg)
this.oASQTab=new EbayASQTab(this,"tab",oCfg);}
ebay.oDocument.oPage.onAfterLoad=function()
{var oSkypeCfg=this.parent.getConfig("ASQ.SkypeTab");if(oSkypeCfg)
{var oSkype=new EbaySkype(this,"skype");if(!oSkype.bInstalled)
{this.initNADivs=function(pDivs,pLen)
{for(var i=0;i<pLen;i++)
{var oCallNAdiv=new EbayHTMLLayer(this,pDivs[i]);oCallNAdiv.bind();oCallNAdiv.show(true);}}
var aCallNADivs=oSkypeCfg.aCallNADivs,iLen1=aCallNADivs?aCallNADivs.length:0;this.initNADivs(aCallNADivs,iLen1);var aChatNADivs=oSkypeCfg.aChatNADivs,iLen2=aChatNADivs?aChatNADivs.length:0;this.initNADivs(aChatNADivs,iLen2);}
if(oSkypeCfg.sSkypeName)
{if(oSkype.bInstalled)
{var sSkypeUrl="skype:"+oSkypeCfg.sSkypeName+"?",sTab,oTab,aTabs;sSkypeUrl+=(parseInt(oSkypeCfg.iSkypeAction)==1)?"chat":"call";this.oDocument.doc.location.href=sSkypeUrl;aTabs=oSkypeCfg.aSelectTabs;if(this.oASQTab&&aTabs.length>0)
{if((parseInt(oSkypeCfg.iSkypeAction)==1))
sTab=aTabs[0];else
sTab=aTabs[1];oTab=this.oASQTab.controls[sTab];if(oTab)
oTab.onclick();}}}}
var oSkypePromoCfg=this.parent.getConfig("ASQ.SkypePromotion");if(oSkypePromoCfg)
{var oSkype=new EbaySkype(this,"skype"),aPDs=oSkypePromoCfg.aSkypePromotionDivs,iLen=aPDs?aPDs.length:0;if(oSkype)
{for(var i=0;i<iLen;i++)
{var oDiv=new EbayHTMLLayer(this,aPDs[i]);oDiv.bind();oDiv.show(!oSkype.bInstalled);}}}
this.initASQAfterLoad();}

//8@@m10

function EbayHTMLText(pParent,pName,pDisabled,pCfg,bHidden)
{if(!this.objType)
this.objType="EbayHTMLText";this.base=EbayHTMLFormElem;this.base(pParent,pName,pDisabled,pCfg);this.value=ebHTMLTextValue;this.getValue=ebHTMLTextGetValue;this.setValue=ebHTMLTextSetValue;this.select=ebHTMLTextSelect;if(bHidden!=true)
this.subscribeEvents("onchange","onblur","onfocus","onkeydown","onkeyup");}
function ebHTMLTextValue(pVal)
{var e=this.eElem;if(e)
{if(typeof(pVal)=="undefined")
return e.value;else
e.value=pVal;}}
function ebHTMLTextGetValue()
{return this.value();}
function ebHTMLTextSetValue(pVal)
{return this.value(pVal);}
function ebHTMLTextSelect()
{var e=this.eElem;if(e)
e.select();}

//9@@m5

function EbayHTMLTextIndexed(pParent,pName,pDisabled,pCfg)
{if(!this.objType)
this.objType="EbayHTMLTextIndexed";this.base=EbayHTMLText;this.base(pParent,pName,pDisabled,pCfg);this.getElem=ebHTMLTextIndexedGetElem;this.iIndex=pCfg.iElemIndex?pCfg.iElemIndex:0;}
function ebHTMLTextIndexedGetElem(pName)
{var oD=this.oDocument,t=oD.getFormElem(pName,"text");t=t?t:oD.getFormElem(pName,"textarea");t=t?t:oD.getFormElem(pName);if(t)
{if(typeof(t.length)!="undefined")
{if(t.length>0)
t=t[this.iIndex];}}
return t;}

//10@@m17

function EbayCharCounterManager(pParent,pName,pCfg)
{if(!this.objType)
this.objType="EbayCharCounterManager";this.baseObject=EbayBaseControl;this.baseObject(pParent,pName);this.createControls=ebCharCounterCreateControls;this.createControls();}
function EbayCharCounter(pParent,pName,pCfg)
{if(!this.objType)
this.objType="EbayCharCounter";this.baseObject=EbayBaseControl;this.baseObject(pParent,pName);this.oConfig=pCfg?pCfg:null;this.oConfig.elemIndex=this.oConfig.elemIndex?this.oConfig.elemIndex-1:0;this.updateDiv=ebCharCounterUpdateDiv;this.dLength=ebDetermineLength;this.checkCounter=ebCheckCounter;this.autoClear=ebCharCounterOnAutoClear;this.alreadyCleared=false;this.clearCounter=ebCharCounterClearCounter;this.oCounter=null;this.init=function()
{var c=this.oConfig;var inp=this.oInput=new EbayHTMLTextIndexed(this,c.inputFieldId,false,c);this.oInput.iIndex=c.elemIndex;inp.subscribeEvents("onclick");if(this.oInput.length>0)
this.oInput=this.oInput[this.elemIndex];this.oLayer=new EbayHTMLLayer(this,c.divForTextId,false,c);if(c.autoClear)
{var autoclear=this.oInput._registerEvent("onclick","autoClear");this.oInput.autoClear=this.autoClear;}
this._registerListener(this.oDocument._getEvent("load"),this.EVENT_AFTER,"updateDiv");var e1=this.oInput._registerEvent("onblur","");var e2=this.oInput._registerEvent("onfocus","");this._registerListener(e1,this.EVENT_AFTER,"clearCounter");this._registerListener(e2,this.EVENT_AFTER,"checkCounter");}
this.init();}
function ebCharCounterCreateControls()
{var aConfigs=ebay.oDocument.aConfigs;var c;for(var oc in aConfigs)
{c=aConfigs[oc];if(c.objType=="EBayCharCounterConfig")
{if(c.inputFieldId&&c.divForTextId)
{new EbayCharCounter(this,oc,c);}}}}
function ebCharCounterOnAutoClear()
{if(this.eElem.defaultValue!=this.eElem.value)
{return;}
if(!this.alreadyCleared)
{this.value("");this.parent.updateDiv();this.alreadyCleared=true;}}
function ebCharCounteronChangeDiv()
{this.parent.updateDiv();}
function ebCheckCounter()
{var iDelay=((this.oConfig)?this.oConfig.iCounterDelay:false)||1000;if(this.oCounter)
window.clearInterval(this.oCounter);this.oCounter=window.setInterval("ebay._getControl(\""+this.name+"\").updateDiv()",iDelay);}
function ebCharCounterClearCounter()
{if(this.oCounter)
window.clearInterval(this.oCounter);}
function ebCharCounterUpdateDiv()
{var v=this.oInput.value(),c=this.oConfig;var l=ebDetermineLength(v);var chrRemain=c.maxChars-l;var max=c.maxChars;if(l>max)
{var i=parseInt(max)+ebCharCounterGetNoofCRs(v);var cl=this.oGlobals.oClient;if(v.substr(i)!="\r\n")
i++;if(cl.bFirefox)
var iPos=this.oInput.eElem.scrollTop
this.oInput.value(v.substr(0,max));if(cl.bFirefox)
this.oInput.eElem.scrollTop=iPos;}
var str='';if(chrRemain<=0)
str=c.noCharLeft;else if(chrRemain==1)
str=c.singleCharLeft;else
str=c.localizeText;this.oLayer.setValue(str.replaceTokensEx("##n##",chrRemain));}
function ebDetermineLength(v)
{if(typeof(v)=="string")
{var regexp=new RegExp("\r\n","g");var lines=v.match(regexp);v=v.replace(regexp,"");var l=0;if(lines!=null)
{var lineCount=lines.length;if(lineCount>=1)
{l=lineCount*2;}}
l=l+v.length;return l;}
else
{return 0;}}
function EBayCharCounterConfig(name)
{if(!this.objType)
this.objType="EBayCharCounterConfig";this.base=EbayConfig;this.base(name);this.elemIndex=0;this.formname="";this.inputFieldId="";this.divForTextId="";this.localizeText="";this.singleCharLeft="";this.noCharLeft="";this.maxChars=0;this.autoClear=false;this.anchorId="";}
function Localize(a)
{return a;}
function ebCharCounterGetNoofCRs(v)
{var regexp=new RegExp("\r\n","g");var lines=v.match(regexp);var l=0;if(lines!=null)
{var lineCount=lines.length;if(lineCount>=1)
{l=lineCount;}}
return l-1;}

//11@@m23

function EbayHTMLFrame(pParent,pName,pCfg)
{if(!this.objType)
this.objType="EbayHTMLFrame";this.base=EbayHTML;this.base(pParent,pName,pName,false,pCfg);this.eFrameElem=null;this.getElem=ebHTMLFrameGetElem;this.bindHTML=ebHTMLFrameBindHTML;this.bindEvents=this.enable=function(){};this.setSource=ebHTMLFrameSetSource;this.cleanupMemoryBase=this.cleanupMemory;this.cleanupMemory=ebHTMLFrameCleanupMemory;this.resize=ebHTMLFrameResize;this.onBeforeResize=this.onAfterResize=null;}
function ebHTMLFrameGetElem(pName)
{with(this)
{var f=null,oD=oDocument;var d=oD.doc,w=oD.win;if(w.frames)
f=eFrameElem=w.frames[pName];if(d.getElementById)
f=d.getElementById(pName);return f;}}
function ebHTMLFrameBindHTML()
{with(this)
{eElem=getElem(sElemName);if(eElem)
assignJSObject(eElem);}}
function ebHTMLFrameCleanupMemory()
{this.cleanupMemoryBase();this.eFrameElem=null;}
function ebHTMLFrameSetSource(pURL)
{if(pURL==null||pURL.trim()==''){return;}
with(this)
{oDocument.setGlobalParent(this);if(pURL.has("ej2child=true"))
pURL+="&ej2parent="+name;if(eFrameElem)
eFrameElem.location.replace(pURL);else if(eElem)
eElem.src=pURL;}}
function ebHTMLFrameResize(pMaxWidth)
{with(this)
{if(onBeforeResize)
onBeforeResize();var f=eFrameElem;if(!f||!(f.document||f.contentDocument))
f=getElem(sElemName);if(f&&typeof(f.document)!="unknown")
{var oDoc=f.document?f.document:f.contentDocument,db=oDoc.body,es=eElem.style,c=this.parent.oGlobals.oClient,w="100%",h=db.offsetHeight,oh;if(c.bSafari)
{oh=db.offsetHeight;w=oDoc.width;h=ebay.oDocument.doc.doctype!=null?oDoc.height+15:oDoc.height+1;}
else if(c.bFirefox)
{w=oDoc.width;h=oDoc.height}
else if(c.bWin||c.bOpera)
{w=db.scrollWidth;h=c.bNav&&ebay.oDocument.doc.doctype!=null?db.scrollHeight+30:db.scrollHeight;}
if(pMaxWidth&&c.bFirefox)
w="100%";if(this.oConfig)
{w=this.oConfig.iWidth||w;h=this.oConfig.iHeight||h;}
es.width=(w=="100%")?w:w+"px";es.height=h+"px";if(onAfterResize)
onAfterResize();}}}

//12@@m1

function EbaySkype(pParent,pName)
{if(!this.objType)
this.objType="EbaySkype";this.base=EbayBaseControl;this.base(pParent,pName);this.sMimeType="application/x-skype";this.sCtlName="Skype.Detection";this.bInstalled=false;this.init=function()
{this.bInstalled=this.isInstalled();}
this.isInstalled=function()
{var oc=ebay.oGlobals.oClient;var cantDetect=oc.bSafari||oc.bOpera;var supportActiveX=oc.bActiveXSupported;if(cantDetect)
return true;if(!supportActiveX)
{var mObj;if(oc.aMimeTypes&&oc.aMimeTypes.length)
{mObj=oc.aMimeTypes[this.sMimeType];}
return(typeof(mObj)=="object");}
return oc.activeXLibLoaded(this.sCtlName);}
this.init();}

//13@@m4

function EbayASQTab(pParent,pName,pCfg)
{if(!this.objType)
this.objType="EbayASQTab";this.oCfg=pCfg;this.base=EbayBaseControl;this.base(pParent,pName);this._registerListener(this.oDocument._getEvent("load"),this.EVENT_AFTER,"initOnAfterLoad");this.createObjects=function()
{var oCfg=this.oCfg,aTabs=oCfg.aTabs,iLen=aTabs.length;var aSkypeDivs=oCfg.aSkypeDivs,iCount=aSkypeDivs.length;this.oFrame=new EbayHTMLFrame(this,oCfg.sSkypeiFrame);this.oErrorDiv=new EbayHTMLLayer(this,oCfg.sErrorDiv);this.sDefaultTab=oCfg.sDefaultTab;for(var j=0;j<iCount;j++)
new EbayHTMLLayer(this,aSkypeDivs[j]);for(var i=0;i<iLen;i++)
{var oAnch=new EbayHTMLAnchor(this,aTabs[i][0]);var oLink=new EbayHTMLAnchor(oAnch,aTabs[i][1]);var oDiv1=new EbayHTMLLayer(this,aTabs[i][2]);var oDiv2=new EbayHTMLLayer(this,aTabs[i][3]);var oDiv3=new EbayHTMLLayer(this,aTabs[i][4]);oAnch.aChildren=[oDiv1,oDiv2,oDiv3];oAnch.sSkypeUrl=aTabs[i][5];if(this.sDefaultTab.is(aTabs[i][0]))
this.oLastSelected=oAnch;oAnch._registerEvent("onclick","parent.showHideTab");oLink._registerEvent("onclick","parent.parent.syncTab");this.oLink=oLink;}}
this.createObjects();this.initOnAfterLoad=function()
{var oCfg=this.oCfg,aLayers=oCfg.aInitDivs,iLen=aLayers?aLayers.length:0,aNSKDivs=oCfg.aNonSkypeDivs,iCount=aNSKDivs?aNSKDivs.length:0;var oLS=this.oLastSelected;if(oLS&&oLS.sSkypeUrl)
this.oFrame.setSource(oLS.sSkypeUrl);if(oCfg.bASQOption2)
{var oSkype=new EbaySkype(this,"skype");if(oSkype)
{if(iLen)
this.showHideDivs(aLayers,iLen,oSkype.bInstalled);if(iCount)
this.showHideDivs(aNSKDivs,iCount,!oSkype.bInstalled);}}
else
{if(iLen)
this.showHideDivs(aLayers,iLen,true);if(iCount)
this.showHideDivs(aNSKDivs,iCount,false);}}
this.showHideDivs=function(pDivs,pCount,pShow)
{for(var i=0;i<pCount;i++)
{var oDiv=new EbayHTMLLayer(this,pDivs[i]);oDiv.bind();oDiv.show(pShow);}}
this.syncTab=function()
{this.parent.onclick();}
this.createObjects();this.showHideTab=function()
{var oP=this.parent,oLastSelected=oP.oLastSelected,oPrevSelChildren=oLastSelected.aChildren;if(this.name==oLastSelected.name)
return;if(oP.oCfg.sErrorDiv)
oP.oErrorDiv.show((this.name.is(oP.sDefaultTab)));this.aChildren[0].show(true);this.aChildren[1].show(false);this.aChildren[2].show(true);oPrevSelChildren[0].show(false);oPrevSelChildren[1].show(true);oPrevSelChildren[2].show(false);this.parent.oLastSelected=this;if(this.sSkypeUrl)
oP.oFrame.setSource(this.sSkypeUrl);}}

//14@@m2

ebay.oDocument.oPage.createCharCounterMgr=function()
{with(this)
{var oD=parent;_registerListener(oD._getEvent("load"),EVENT_BEFORE,"createCCMgr");this.createCCMgr=function()
{new EbayCharCounterManager(oD.parent,"ccManager");}}}
ebay.oDocument.oPage.createCharCounterMgr();

//15@@m21

function EbayHTMLSelect(pParent,pName,pDisabled,pCfg)
{if(!this.objType)
this.objType="EbayHTMLSelect";this.base=EbayHTMLFormElem;this.base(pParent,pName,pDisabled,pCfg);this.iSelIndex=-1;this.createOption=ebHTMLSelectCreateOption;this.clearOptions=ebHTMLSelectClearOptions;this.getValueByIndex=ebHTMLSelectGetValueByIndex;this.getSelectedIndex=ebHTMLSelectGetSelectedIndex;this.getSelectedValue=ebHTMLSelectGetSelectedValue;this.getSelectedText=ebHTMLSelectGetSelectedText;this.getOptionsLength=ebHTMLSelectGetOptionsLength;this.setOption=ebHTMLSelectSetOption;this.insertOption=ebHTMLSelectInsertOption;this.deleteOption=ebHTMLSelectDeleteOption;this.selectByIndex=ebHTMLSelectSelectByIndex;this.selectByValue=ebHTMLSelectSelectByValue;this.selectByText=ebHTMLSelectSelectByText;this.doSelect=ebHTMLSelectDoSelect;this.getIndexByValue=ebHTMLSelectGetIndexByValue;this.getValue=this.getSelectedValue;this.setValue=this.selectByValue;this.subscribeEvents("onchange");}
function ebHTMLSelectClearOptions()
{var e=this.eElem;if(e)
{var opts=e.options;while(opts.length>0)
opts[opts.length-1]=null;}}
function ebHTMLSelectCreateOption(pName,pText)
{if(this.eElem)
{var nOpt=new Option(pText,pName,false,false),opts,lo,oC=ebay.oGlobals.oClient;opts=this.eElem.options;opts[opts.length]=nOpt;idx=opts.length-1;return idx;}}
function ebHTMLSelectGetValueByIndex(pIdx,pTextOnly)
{if(pIdx>-1)
{opt=this.eElem.options[pIdx];if(opt)
return pTextOnly?opt.text:opt.value;}
this.throwError("Invalid index","get");return"";}
function ebHTMLSelectGetSelectedIndex()
{return this.eElem.selectedIndex;}
function ebHTMLSelectGetSelectedValue()
{return this.getValueByIndex(this.eElem.selectedIndex);}
function ebHTMLSelectGetSelectedText()
{return this.getValueByIndex(this.eElem.selectedIndex,true);}
function ebHTMLSelectGetOptionsLength()
{return this.eElem.options.length;}
function ebHTMLSelectSelectByIndex(pIndex)
{this.eElem.selectedIndex=this.iSelIndex=pIndex;}
function ebHTMLSelectDoSelect(pVal,pIsText)
{if(this.eElem)
{var e=this.eElem,o,rv=false,opts=e.options,len=opts.length;for(var i=0;i<len&&!rv;i++)
{o=opts[i];if(((pIsText||(o.value==""))&&(pVal==o.text))||(!pIsText&&(o.value==pVal)))
{e.selectedIndex=this.iSelIndex=i;rv=true;}}}
else
this.throwWarning("HTML element '"+this.name+"' not found","selectByValue");return rv;}
function ebHTMLSelectSelectByValue(pVal)
{return this.doSelect(pVal);}
function ebHTMLSelectSelectByText(pVal)
{return this.doSelect(pVal,true);}
function ebHTMLSelectSetOption(pVal,pText,pInd)
{if(this.eElem)
{if((pInd!=null)&&(pInd>-1))
{var o=this.eElem.options[pInd];o.value=pVal;o.text=pText;}
else
this.createOption(pVal,pText);}
else
this.throwWarning("HTML element '"+this.name+"' not found","selectByValue");}
function ebHTMLSelectInsertOption(pVal,pText,pInd)
{with(this)
{var e=eElem,opts=e.options,len=opts.length;var inOpt=new Array(pText,pVal),tmpOpt=new Array(2);var sel=getSelectedValue();len++;if(pInd>=len)
return;for(i=pInd;i<len;i++)
{if(i<len-1)
tmpOpt=[e.options[i].text,e.options[i].value];opts[i]=new Option(inOpt[0],inOpt[1]);inOpt=tmpOpt;}
selectByValue(sel);}}
function ebHTMLSelectDeleteOption(pInd)
{if(typeof(pInd)!='undefined')
{var opts=this.eElem.options;if(opts[pInd])
opts[pInd]=null;}}
function ebHTMLSelectGetIndexByValue(pVal,pIsText)
{var opts=this.eElem.options,len=opts.length,i=0;for(;i<len;i++)
{o=opts[i];if((o.value==pVal)||(pIsText&&(o.text==pVal)))
return i;}
return-1;}

//16@@m9

ebay.oDocument.oPage.initASAQ=function()
{var cfg=this.parent.getConfig("simpleConfig");var ocfg1=this.parent.getConfig("M2MContactAAQForm");if(cfg!=null)
{var selectVar=new EbayHTMLSelect(this,cfg.selectId);var formVar=new EbayHTMLForm(this,cfg.formId);formVar._registerEvent("onsubmit","isDefaultMessage");selectVar._registerEvent("onchange","onChange");selectVar.onChange=function()
{var selValue=selectVar.getSelectedValue();if(selValue==5)
{formVar.setAction(cfg.path);formVar.submit();}}}
var cc=new EbayCharCounterManager(ebay,"ccManager");if(ocfg1!=null)
{oSubjectField=new EbayHTMLText(this,ocfg1.subjectField);oItemSelect=new EbayHTMLSelect(this,ocfg1.itemSelect);sDefaultSubject=ocfg1.defaultSubject;sItemSubject=ocfg1.itemSubject;sLocation=ocfg1.thisLoc;oItemSelect._registerEvent("onchange","onChangeItem");oItemSelect.onChangeItem=function()
{if(oItemSelect.getSelectedValue()!=-2)
{isItem=sLocation.indexOf("sitemid=");if(isItem==-1)
{window.location=sLocation+'&sitemid='+oItemSelect.getSelectedValue();}else
{sLocation=sLocation.substring(0,isItem+8);window.location=sLocation+oItemSelect.getSelectedValue();}}
else
{isItem=sLocation.indexOf("sitemid=");if(isItem==-1)
{window.location=sLocation+'&sitemid=-2';}else
{sLocation=sLocation.substring(0,isItem+8);window.location=sLocation+'-2';}
oSubjectField.value(sDefaultSubject);}}}
var c=this.parent.getConfig("MyMessages.RespondToQuestion.ShowHide");if(c)
new EbayMessagesQuestionResponseShowHide(this,"EbayMessagesQuestionResponseShowHide",c);}
ebay.oDocument.oPage.init=function()
{var oD=this.parent;this._registerListener(oD._getEvent("load"),oD.EVENT_BEFORE,"initASAQ");}
ebay.oDocument.oPage.init();function isDefaultMessage(){if(document.contactform.message.value==document.contactform.defaultText.value){document.contactform.message.value='';}
if(typeof(document.contactform.Submit.disabled)!='undefined')
document.contactform.Submit.disabled=true;}

//17@@m13

function EbayHTMLBaseCheckboxRadio(pParent,pName,pDisabled,pCfg)
{if(!this.objType)
this.objType="EbayHTMLBaseCheckboxRadio";this.base=EbayHTMLFormElem;this.base(pParent,pName,pDisabled,pCfg);this.bGroup=false;this.bindHTML=ebHTMLBaseCheckboxRadioBindHTML;this.bindEvents=ebHTMLBaseCheckboxRadioBindEvents;this.check=ebHTMLBaseCheckboxRadioCheck;this.selectByIndex=ebHTMLBaseCheckboxRadioSelectByIndex;this.selectByValue=ebHTMLBaseCheckboxRadioSelectByValue;this.isCheckedByValue=ebHTMLBaseCheckboxRadioIsCheckedByValue;this.getValueByIndex=ebHTMLBaseCheckboxRadioGetValueByIndex;this.getIndexByValue=ebHTMLBaseCheckboxRadioGetIndexByValue;this.getValue=null;this.enableBase=this.enable;this.enable=ebHTMLBaseCheckboxRadioEnable;this.setValue=this.selectByValue;this.onBeforeCheck=null;this.onAfterCheck=null;this.subscribeEvents("onclick");}
function ebHTMLBaseCheckboxRadioBindHTML()
{with(this)
{eElem=getElem(sElemName);if(eElem)
{if(eElem.length)
{bGroup=true;for(var i=0;i<eElem.length;i++)
assignJSObject(eElem[i]);cleanupMemory=ebHTMLBaseCheckboxRadioCleanupMemory;}
else
{bGroup=false;assignJSObject(eElem);}}
if(bDisabled)
enable(false);}}
function ebHTMLBaseCheckboxRadioCleanupMemory()
{var e=this.eElem;if(e)
{for(var j=0;j<e.length;j++)
{for(var i in e[j].jsObjs)
{e[j].jsObjs[i]=null;}
e[j].jsObjs=null;}
this.eElem=null;}}
function ebHTMLBaseCheckboxRadioBindEvents()
{with(this)
{if(!eElem)
return;var e=aBindEvents,len=e.length,fStr;for(var i in e)
{if(eElem.length&&eElem.length>0)
{for(var ii=0;ii<eElem.length;ii++)
eval("eElem[ii]."+e[i]+" = function(){"+this.bindEventString(e[i],ii)+"}");}
else
{eval("eElem."+e[i]+" = new Function(this.bindEventString(e[i],0))");}}}}
function ebHTMLBaseCheckboxRadioCheck(pChecked,pIndex)
{if(pIndex<0)
return;with(this)
{if(eElem)
{if(bGroup&&typeof(pIndex)=='undefined')
{var len=eElem.length;for(var i=0;i<len;i++)
eElem[i].checked=pChecked;}
else if(bGroup&&eElem[pIndex])
eElem[pIndex].checked=pChecked;else if(!bGroup)
eElem.checked=pChecked;}}}
function ebHTMLBaseCheckboxRadioSelectByIndex(pIdx,pCheck)
{var chx=typeof pCheck!='undefined'?pCheck:true;with(this)
{if(onBeforeCheck)
onBeforeCheck();var e=bGroup?eElem[pIdx]:eElem;if(e)
{e.checked=chx;if(onAfterCheck)
onAfterCheck();}}}
function ebHTMLBaseCheckboxRadioSelectByValue(pVal,pCheck)
{var chx=typeof pCheck!='undefined'?pCheck:true;with(this)
{if(onBeforeCheck)
onBeforeCheck();var e=eElem;if(!e)
return;if(bGroup)
{for(var i=0;i<e.length;i++)
{if(e[i].value==pVal)
{e[i].checked=chx;if(onAfterCheck)
onAfterCheck();return;}}}
else
{if(e.value==pVal)
{e.checked=chx;if(onAfterCheck)
onAfterCheck();}}}}
function ebHTMLBaseCheckboxRadioIsCheckedByValue(pValue)
{with(this)
{var e=eElem;if(e&&bGroup)
{var len=e.length;for(var i=0;i<len;i++)
{if(e[i].value==pValue)
return isChecked(i);}}}}
function ebHTMLBaseCheckboxRadioGetValueByIndex(pIndex)
{with(this)
{var e=eElem;if(e&&bGroup)
return e[pIndex].value;return null;}}
function ebHTMLBaseCheckboxRadioGetIndexByValue(pValue)
{with(this)
{var e=eElem;if(e&&bGroup)
{var len=e.length;for(var i=0;i<len;i++)
{if(e[i].value==pValue)
return i;}}
return-1;}}
function ebHTMLBaseCheckboxRadioEnable(pEnable)
{with(this)
{enableBase(pEnable);if(bGroup)
{var v=pEnable?"true":"false",e=eElem,len=e.length;if(e)
{for(var i=0;i<len;i++)
{e[i].onfocus=new Function("return "+v+";");e[i].disabled=!pEnable;}}}}}

//18@@m7

function EbayHTMLCheckbox(pParent,pName,pDisabled,pCfg)
{if(!this.objType)
this.objType="EbayHTMLCheckbox";this.base=EbayHTMLBaseCheckboxRadio;this.base(pParent,pName,pDisabled,pCfg);this.getElem=ebHTMLCheckboxGetElem;this.isChecked=ebHTMLCheckboxIsChecked;this.getValue=ebHTMLCheckboxGetValue;this.setValue=this.selectByValue;}
function ebHTMLCheckboxGetElem(pName)
{return this.oDocument.getFormElem(pName,"checkbox");}
function ebHTMLCheckboxIsChecked(pIndex)
{with(this)
{if(eElem)
{if(bGroup&&eElem[pIndex])
return eElem[pIndex].checked;else if(!bGroup)
return eElem.checked;}}
return false;}
function ebHTMLCheckboxGetValue(pUnCheckedValue)
{var e=this.eElem,rv="";if(this.bGroup)
{rv=[];for(var i=0;i<e.length;i++)
{if(e[i].checked)
rv[rv.length]=e[i].value;}
if(!rv.length)rv="";}
else
{if(pUnCheckedValue)
return e.value;if(e.checked)
return e.value;}
return rv;}

//19@@m2

ebay.oDocument.oPage.initProduceSearch=function()
{var oP=this.parent;var oC=oP.oDocument.getConfig("ViewItem.ProductSearch");if(oC)
{if(oC.sYear)
{oYear=new EbayHTMLSelect(this,oC.sYear);}
oMake=new EbayHTMLSelect(this,oC.sMake);oModel=new EbayHTMLSelect(this,oC.sModel);oModelLyr=new EbayHTMLLayer(this,oC.sModelDiv);oTradeChk=new EbayHTMLCheckbox(this,oC.sTradeCheckBox);oTradeValdiv=new EbayHTMLLayer(this,oC.sTradeValdiv);bIsViewItem=oC.bIsViewItem;sAnyField=oC.sAnyField;this.aVal=oC.aArray;if(bIsViewItem)
{oShadowLyr=new EbayHTMLLayer(this,oC.sShadowLyr);oShadowLyr.bind();}
oMake.bind();oModelLyr.bind();oTradeChk.bind();oTradeValdiv.bind();if(oTradeValdiv.eElem)
oTradeValdiv.show();if(oModelLyr.eElem)
oModelLyr.show(true);oMake._registerEvent("onchange","this.parent.enableModel");if(oTradeChk&&oTradeChk.eElem)
{oTradeChk._registerEvent("onclick","this.parent.onTradeCheck");}
this.enableModel=function()
{with(this)
{val=getSelectedValue();if(oModel.eElem)
oModel.bind();var aOpt=this.parent.aVal[val];if(aOpt)
{oModel.enable(true);oModel.clearOptions();for(var val in aOpt)
oModel.createOption(aOpt[val],val);}
else
{if(val==sAnyField)
{oModel.clearOptions();oModel.enable(false);oModel.createOption(val,val);}}}}
this.onTradeCheck=function()
{with(this)
{if(oTradeValdiv&&!oTradeValdiv.eElem)
{oTradeValdiv.bind();}
if(oYear)
{oYear.bind();oYear.enable(true);}
if(oMake)
{oMake.enable(true);}
if(isChecked())
{oTradeValdiv.show(true);if(bIsViewItem)oShadowLyr.eElem.style.height=oShadowLyr.height()+oTradeValdiv.height();}
else
{if(bIsViewItem)oShadowLyr.eElem.style.height=oShadowLyr.height()-oTradeValdiv.height();oTradeValdiv.show();}}}}}
ebay.oDocument.oPage.initEmailOverlay=function()
{var oP=this.parent;var pCfg=oP.oDocument.getConfig("Emailform.Overlay");if(pCfg)
{var oTimeslct=this.oTimeslct=new EbayHTMLSelect(this,pCfg.sTimeslct);if(!oTimeslct.eElem)
oTimeslct.bind();oTimeslct.show(false);var goLyr=this.goLyr=new EbaySYI3Grayout(this,'grayout_lyr');if(!goLyr.oLayer.eElem)
goLyr.oLayer.bind();goLyr.display(document.body.scrollWidth,document.body.scrollHeight);var oContentLyr=this.oContentLyr=new EbayHTMLOverlayContent(this,pCfg.sOverlay,pCfg);if(!oContentLyr.eElem)
oContentLyr.bind();oContentLyr.show(true);oContentLyr.setIframeShim();oShadowLyr=new EbayHTMLLayer(this,pCfg.sShadowLyr);oShadowLyr.bind();oBodyContent=new EbayHTMLLayer(this,pCfg.sContent);oBodyContent.bind();oShadowLyr.eElem.style.height=oShadowLyr.height()+oBodyContent.height();var oCloseLyr=new Array();for(var i=0;i<pCfg.aCloseLayer.length;i++)
{oCloseLyr[i]=new EbayHTMLAnchor(this,pCfg.aCloseLayer[i]);if(!oCloseLyr[i].eElem)
oCloseLyr[i].bind();oCloseLyr[i]._registerEvent("onclick","parent.closeLayer");}}
this.closeLayer=function()
{this.parent.oTimeslct.show(true);this.parent.oContentLyr.closeOverlay();this.parent.goLyr.hide();}}
ebay.oDocument.oPage.initAdditionalComments=function()
{var oP=this.parent;var pCfg=oP.oDocument.getConfig("Emailform.AdditionalComments");if(pCfg)
{oTlyr=new EbayHTMLText(this,pCfg.sTextArea);if(!oTlyr.eElem)
oTlyr.bind();oTlyr._registerEvent("onfocus","parent.clearText");oTlyr._registerEvent("onblur","parent.recallText");}
this.clearText=function()
{if(this.eElem.value==pCfg.sTextValue)
{this.eElem.value="";oTlyr.setClass(pCfg.sTextClass_1);}}
this.recallText=function()
{if(this.eElem.value=="")
{oTlyr.setClass(pCfg.sTextClass);this.eElem.value=pCfg.sTextValue;}
else
{var sMsg=oTlyr.eElem.value;this.eElem.value=sMsg.substring(0,pCfg.sMaxChar);}}}

//20@@m1

function EbayMessagesQuestionResponseShowHide(pParent,pName,pCfg)
{if(!this.objType)
this.objType="EbayMessagesQuestionResponseShowHide";this.base=EbayBaseControl;this.base(pParent,pName);var c=this.oConfig=pCfg;var sSA=c.sShowLinkName;var sHA=c.sHideLinkName;var sSL=c.sShowLayerName;var sHL=c.sHideLayerName;this.oShowLayer=new EbayHTMLLayer(this,sSL);this.oHideLayer=new EbayHTMLLayer(this,sHL);this.oShowAnchor=new EbayHTMLAnchor(this,sSA);this.oHideAnchor=new EbayHTMLAnchor(this,sHA);with(this)
{oShowAnchor.oSL=oHideLayer;oShowAnchor.oHL=oShowLayer;oHideAnchor.oSL=oShowLayer;oHideAnchor.oHL=oHideLayer;oShowAnchor._registerEvent("onclick","ebMessagesQuestionResponseShowHideOnClick");oHideAnchor._registerEvent("onclick","ebMessagesQuestionResponseShowHideOnClick");}}
function ebMessagesQuestionResponseShowHideOnClick()
{this.oSL.show(true);this.oHL.show(false);return false;}

//21@@m6

function EbayHTMLForm(pParent,pName,pCfg)
{if(!this.objType)
this.objType="EbayHTMLForm";this.base=EbayHTML;this.base(pParent,pName,pName,false,pCfg);this.getElem=function(){return this.getDocElem(arguments[0],'forms');};this.enable=function(){};this.getElementValue=ebHTMLFormGetElementValue;this.setElementValue=ebHTMLFormSetElementValue;this.getElements=ebHTMLFormGetElements;this.getElement=ebHTMLFormGetElement;this.setAction=ebHTMLFormSetAction;this.getAction=ebHTMLFormGetAction;this.setTarget=ebHTMLFormSetTarget;this.getTarget=ebHTMLFormGetTarget;this.submit=ebHTMLFormSubmit;this.clear=ebHTMLFormClear;this.subscribeEvents("onsubmit");this.onBeforeSubmit=null;this.onAfterSubmit=null;}
function ebHTMLFormGetElements()
{var e=this.eElem;return e?e.elements:new Array;}
function ebHTMLFormGetElement(pName)
{var elems=this.getElements();return elems[pName]?elems[pName]:null;}
function ebHTMLFormGetElementValue(pName)
{var elems=this.getElements();if(elems[pName]&&elems[pName].value)
return elems[pName].value;return"";}
function ebHTMLFormSetElementValue(pName,pValue)
{var elems=this.getElements(),elem=elems[pName];if(elem)
{if(elem.length)
{for(var i=0,len=elem.length;i<len;i++)
elem[i].value=pValue;}
else
elem.value=pValue;}}
function ebHTMLFormSetAction(pAction)
{var e=this.eElem;if(e)
e.action=pAction;}
function ebHTMLFormGetAction()
{var e=this.eElem;if(e)
return e.action;}
function ebHTMLFormSetTarget(pTarget)
{var e=this.eElem;if(e)
e.target=pTarget;}
function ebHTMLFormGetTarget()
{var e=this.eElem;if(e)
return e.target;}
function ebHTMLFormSubmit()
{if(this.onBeforeSubmit)
this.onBeforeSubmit();var e=this.eElem;if(e)
{e.submit();if(this.onAfterSubmit)
this.onAfterSubmit();}
else
this.throwError("Element '"+this.sElemName+"' does not exist on the page","submit");}
function ebHTMLFormClear()
{var elems=this.getElements();for(i=0;i<elems.length;i++)
{var elem=elems[i];var type=elem.type;switch(type)
{case"text":case"textarea":elem.value="";break;case"checkbox":elem.checked=false;break;case"select-one":elem.selectedIndex=0;}}}

//22@@m1

function editQuestion()
{if(document.getElementById)
{var oOQ=document.getElementById('origQuestion');if(typeof(oOQ)!="undefined")
{var eq=document.getElementById('editedquestion');if(eq)
eq.value=oOQ.innerText?oOQ.innerText:oOQ.innerHTML;document.getElementById('origQuesContainer').style.display="none";document.getElementById('editedQuesContainer').style.display="inline";}
else
{document.all('editedquestion').value=oOQ.innerHTML;document.all('origQuesContainer').style.display="none";document.all('editedQuesContainer').style.display="inline";}}}

//23@@m1

ebay.oDocument.oPage.initEmailSubject=function()
{var oP=this.parent;var oC=oP.oDocument.getConfig("EmailSubject");if(oC)
{oSelect=new EbayHTMLSelect(this,oC.sDropdown);oMsgDiv=new EbayHTMLLayer(this,oC.sMsgDiv);oHiddenFld=new EbayHTMLText(this,oC.sHiddenFld);this.sMsg=oC.sMsg;oSelect.bind();oMsgDiv.bind();oHiddenFld.bind();if(oSelect.eElem)
{oSelect._registerEvent("onchange","parent.paintEmailSubject");var val=this.sMsg.replaceTokens(oSelect.getSelectedText());oMsgDiv.setValue(val);if(oHiddenFld.eElem)
{oHiddenFld.setValue(val)}}}
this.paintEmailSubject=function()
{with(this)
{var val=parent.sMsg.replaceTokens(getSelectedText());if(oMsgDiv&&oMsgDiv.eElem)
{oMsgDiv.setValue(val);}
if(oHiddenFld&&oHiddenFld.eElem)
{oHiddenFld.setValue(val);}}}}

//24@@m2

function EbayHTMLAudioGifChallenge(pParent,pName,pDisabled,pCfg)
{if(!this.objType)
this.objType="EbayHTMLAudioGifChallenge";this.base=EbayHTML;this.base(pParent,pName,pName,pDisabled,pCfg);this.oLinkRefresh=this.oLinkListen=null;this.init=ebHTMLAudioGifChallengeInit;this.createControls=ebHTMLAudioGifChallengeCreateControls;with(this)
_registerListener(oDocument._getEvent("load"),EVENT_BEFORE,"createControls");this.init();}
function ebHTMLAudioGifChallengeCreateControls()
{with(this)
{var oP=this.parent,oD=oP.parent,c=oD.getConfig("Security.Gif.Audio.Challenge"),oR,oL,oF,oT;if(c)
{oF=new EbayHTMLFrame(this,c.sBotFrameId,c);oR=new EbayHTMLAnchor(this,c.sBotRefreshLinkId,false,c);oR.onclick=function()
{var oP=this.parent,oC=this.oConfig,oL=oP._getControl(oC.sBotListenLinkId);oL.enable(false);oT=oD.getFormElem(c.sTokenTextId);if(oT)
oT.value="";oL.eElem.href="#";oF.setSource(oC.sBotIframeUrl);var oPop=oP._getControl("audioPopup");if(oPop)
oPop.close();return false;}
oL=new EbayHTMLAnchor(this,c.sBotListenLinkId,false,c);oL.onclick=function()
{var c=this.oConfig,oPop,l,t,url=c.sAudioHelpPopupUrl,ts;l=parseInt(screen.availWidth-c.sAudioHelpPopupWidth-10);t=0;ts=this.parent.oDocument.getFormElem(c.sTokenStringId).value;url+=url.has("?")?"&":"?";url+=c.sTokenStringQueryParamName+"="+ts;oPop=new EbayHTMLPopup(this,"audioPopup",c);oPop.showEx(url,c.sAudioHelpPopupWidth,c.sAudioHelpPopupHeight,0,0,0,0,1,0,l,t);return false;}}}}
function ebHTMLAudioGifChallengeInit()
{var oD=this.parent.parent;oD._registerListener(oD._getEvent("load"),oD.EVENT_BEFORE,"createControls");}
new EbayHTMLAudioGifChallenge(ebay.oDocument.oPage,"AudioGifChallege");

//25@@m1

if(!ebay.oDocument.oPage.bNotDowngradeDomain){ebay.oDocument.downgradeDomain();}
// b=8247622 -->