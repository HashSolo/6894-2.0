//<!--
//1@@m28

function EbayDocument(pWin,pName)
{this.htmlWin=pWin||null;this.htmlDoc=pWin?pWin.document:null;this.name=pName||null;this.status=null;this.controls=new Array;this.events=new Array;this.bindHTML=EbayDocumentBindHTML;this.bindEvents=EbayDocumentBindEvents;this.addControl=EbayAddControl;this.getFormElem=EbayGetFormElem;this.onLoad=EbayDocumentOnLoad;this.onUnload=EbayDocumentOnUnLoad;}
window.EbayDocument=EbayDocument;function EbayGetFormElem(pName,pType)
{if(!this.htmlDoc)
return null;var frms=this.htmlDoc.forms;var ln=frms.length;for(var i=0;i<ln;i++)
{var elems=frms[i].elements;for(var j=0;j<elems.length;j++)
{if(elems[j].name==pName)
{if(pType)
{if(elems[j].type==pType)
return elems[pName];}
else
return elems[j];}}}}
function EBayPreviewDocument(pWin,pName)
{this.baseObject=EbayDocument;this.baseObject(pWin,pName);this.isPreviewMode=true;this.skipElems=new Array;this.skipLinks=new Array;this.onLoad=EbayDisableDoc;}
function EbayDisableDoc()
{if(!this.isPreviewMode)
return;var doc=this.htmlDoc;for(var j=0;j<doc.forms.length;j++)
{var frms=doc.forms;var iElems=frms[j].elements.length;frms[j].onsubmit=EbayDisableOnSubmit;for(var i=0;i<iElems;i++)
{var elem=frms[j].elements[i];if(elem.id!="")
document.getElementById(elem.id);var skEs=this.skipElems;var skip=false;if(skEs.length>0)
{for(var k=0;k<skEs.length;k++)
{if(elem.name==skEs[k])
{skip=true;break;}}}
if(!skip)
EbayDisableFormElement(elem);}}
var iLinks=doc.links.length;for(var i=0;i<iLinks;i++)
{var lnk=doc.links[i];var skLnks=this.skipLinks;var skip=false;if(lnk.href)
{for(var k=0;k<skLnks.length;++k)
{if(lnk.href.indexOf(skLnks[k])!=-1)
{skip=true;break;}}}
if(!skip)
{lnk.href="#";lnk.onclick=EbayDisableLink;}}}
function EbayDisableLink()
{return false;}
function EbayDisableFormElement(pElem)
{if(pElem.type)
{switch(pElem.type.toLowerCase())
{case"hidden":break;case"text":pElem.disabled=true;break;case"button","submit":pElem.onclick=null;break;default:pElem.onclick=null;break;}}}
function EbayDisableText()
{if(typeof(this.disabled)!="undefined")
this.htmlElement.disabled=this.disabled=true;}
function EbayEnableText()
{if(typeof(this.disabled)!="undefined")
this.htmlElement.disabled=this.disabled=false;}
function EbayDisableOnSubmit()
{return false;}
function EbayControl(pEbayDoc,pParent,pHTMLDoc,pElementName)
{this.htmlElement=null;this.htmlElementName=pElementName||null;this.name=pElementName||null;this.parent=pParent||null;this.ebayDoc=pEbayDoc||null;this.htmlDoc=pHTMLDoc||null;this.controls=new Array;this.listeners=new Array;this.bindHTML=null;this.bindEvents=null;this.registerListener=EbayControlRegisterListener;this.allowUpdateOnload=false;this.onBeforeLoad=null;this.onAfterLoad=null;this.onBeforeUnload=null;this.onAfterUnload=null;this.onUpdate=EbayControlOnUpdate;if(this.parent&&this.parent.addControl)this.parent.addControl(this);}
window.EbayControl=EbayControl;function EbayControlRegisterListener(pControl)
{this.listeners[this.listeners.length]=pControl;}
function EbayControlOnUpdate(pParent)
{var parent=pParent?pParent:this.ebayControl;for(var i=0;i<parent.listeners.length;i++)
{if(parent.listeners[i].onUpdate)parent.listeners[i].onUpdate(parent);}}
function EbayText(pebayDoc,pParent,pHTMLDoc,pElementName)
{if(!this.objType)
this.objType="ebayText";this.baseObject=EbayControl;this.baseObject(pebayDoc,pParent,pHTMLDoc,pElementName);this.disabled=false;this.bindHTML=EbayBindHTMLText;this.bindEvents=tbd;this.disable=EbayDisableText;this.enable=EbayEnableText;}
window.EbayText=EbayText;function EbayHyperLink(pEbayDoc,pParent,pHTMLDoc,pElementName,pLink)
{if(!this.objType)
this.objType="EbayHyperLink";this.baseObject=EbayControl;this.baseObject(pEbayDoc,pParent,pHTMLDoc,pElementName);this.link=pLink||null;this.mouseOverText="";this.urlPath=null;this.eventBound=false;this.bindHTML=EbayBindHTMLHyperLink;this.bindEvents=EbayBindEventsHyperLink;this.getLink=EbayGetLink;this.onClick=null;this.onMouseOver=tbd;this.onMouseOut=tbd;}
window.EbayHyperLink=EbayHyperLink;function EbayBindHTMLText()
{this.htmlElement=this.ebayDoc.getFormElem(this.htmlElementName,"text");if(this.htmlElement)
{this.htmlElement.ebayControl=this;if(this.disabled)
this.disable();else
this.enable();}}
function EbayMoveLayer(pLayer,x,y)
{if(document.getElementById)
{pLayer.style.left=x+'px';pLayer.style.top=y+'px';}
else if(document.all)
{pLayer.style.left=x;pLayer.style.top=y;}
else if(document.layers)
{pLayer.pageX=x;pLayer.pageY=y;}}
function EbayDocumentBindHTML()
{this.htmlDoc.ebayDoc=this;EbayBindAllControls(this,false);}
function EbayDocumentBindEvents()
{EbayBindAllControls(this,true);}
function EbayBindAllControls(pThis,pIsEvents)
{var cctrl;var ctrls=pThis.controls;for(var i=0;i<ctrls.length;i++)
{cctrl=ctrls[i];if(pIsEvents)
{if(cctrl&&cctrl.bindEvents&&cctrl.objType!="EbayHyperLink")
cctrl.bindEvents();}
else
{if(cctrl&&cctrl.bindHTML)
cctrl.bindHTML();}}}
function EbayAddControl(pControl)
{var isSet=false;if(pControl.htmlElementName)
{var ctrls=this.controls;for(var i=0;i<ctrls.length;i++)
{if(ctrls[i].htmlElementName==pControl.htmlElementName)
{this.controls[i]=pControl;return;}}}
this.controls[this.controls.length]=pControl;}
function EbayDocumentOnLoad()
{this.bindHTML();this.bindEvents();}
function EbayDocumentOnUnLoad()
{}
function EbayBindHTMLHyperLink()
{this.htmlElement=this.getLink(this.htmlDoc,this.htmlElementName);if(this.htmlElement)
{this.link=this.htmlElement.href;this.htmlElement.ebayControl=this;}
this.bindEvents();}
function EbayGetLink(pDoc,pLinkName)
{var lnk=null;if(pDoc&&pLinkName&&!this.ebayDoc.htmlWin.closed)
{if(pDoc.all)
lnk=pDoc.all[pLinkName];if(lnk)return lnk;if(pDoc.getElementById)
lnk=pDoc.getElementById(pLinkName);if(lnk)return lnk;for(var j=0;j<pDoc.links.length;j++)
{lnk=pDoc.links[j];if(typeof(lnk.name)!="undefined")
{if(lnk.name==pLinkName)
return lnk;}
else
{if(lnk.onclick)
{var oc=lnk.onclick.toString();if(oc.indexOf("{#"+pLinkName+"#}")!=-1)
return lnk;}}}
lnk=null;if(pDoc.layers)
{var lyrs=pDoc.layers;var len=lyrs.length;for(var i=0;i<len;i++)
{if(this.ebayDoc.htmlDoc==null)
return;else
{lnk=this.getLink(lyrs[i].document,pLinkName);if(lnk)
return lnk;}}}}
return lnk;}
function setEbayLink(pS)
{return true;}
function EbayBindEventsHyperLink()
{if(!this.htmlElement)
return;this.htmlElement.onclick=this.onClick;}
function EbayImage(pEbayDoc,pParent,pHTMLDoc,pElementName)
{if(!this.objType)
this.objType="EbayImage";this.baseObject=EbayControl;this.baseObject(pEbayDoc,pParent,pHTMLDoc,pElementName);this.image=pElementName||null;this.mouseOverText="";this.bindHTML=EbayBindHTMLImage;this.bindEvents=tbd;this.getImage=EbayGetImage;}
window.EbayImage=EbayImage;function EbayBindHTMLImage()
{this.htmlElement=this.getImage(this.htmlDoc,this.htmlElementName);if(this.htmlElement)
this.htmlElement.ebayControl=this;}
function EbayGetImage(pDoc,pImageName)
{var image=null;if(pDoc&&pImageName&&!this.ebayDoc.htmlWin.closed)
{var len=pDoc.images.length;for(var i=0;i<len;i++)
{if(!this.ebayDoc.htmlDoc)
return;else if(pDoc.images[i].name==pImageName)
return pDoc.images[i];}
if(pDoc.layers)
{var lyrs=pDoc.layers;var len=lyrs.length;for(var i=0;i<len;i++)
{if(this.ebayDoc.htmlDoc==null)
return;else
{image=this.getImage(lyrs[i].document,pImageName);if(image)
return image;}}}}
return image;}
function EbaySelect(pEbayDoc,pParent,pHTMLDoc,pElementName)
{if(!this.objType)
this.objType="EbaySelect";this.baseObject=EbayControl;this.baseObject(pEbayDoc,pParent,pHTMLDoc,pElementName);this.bindHTML=EbayBindHTMLSelect;this.bindEvents=EbayBindEventsSelect;this.clearOptions=EbaySelectClearOptions;this.getSelect=EbayGetSelect;this.onAfterLoad=null;this.onChange=tbd;}
window.EbaySelect=EbaySelect;function EbayBindHTMLSelect()
{this.htmlElement=this.getSelect(this.htmlDoc,this.htmlElementName);if(this.htmlElement)
this.htmlElement.ebayControl=this;if(this.allowUpdateOnload&&this.update)
this.update();}
function EbayBindEventsSelect()
{if(this.htmlElement)
{this.htmlElement.onchange=this.onChange;if(this.onAfterLoad)this.onAfterLoad();}}
function EbaySelectClearOptions()
{if(this.htmlElement)
{var len=this.htmlElement.options.length;for(var i=0;i<len;i++)
{this.htmlElement.options[0]=null;}}}
function EbayGetLayer(pDoc,pLayerName)
{var layer=null;if(pDoc&&pLayerName&&!this.ebayDoc.htmlWin.closed)
{if(pDoc.getElementById)
{layer=pDoc.getElementById(pLayerName);}
else if(pDoc.all)
{layer=pDoc.all[pLayerName];}
else if(pDoc.layers)
{var lyrs=pDoc.layers;var len=lyrs.length;layer=lyrs[pLayerName];if(!layer){for(var i=0;i<len;i++)
{if(lyrs[i].name==pLayerName||lyrs[i].id==pLayerName)
{layer=lyrs[i];}
else
{layer=this.getLayer(lyrs[i].document,pLayerName);}
if(layer)break;}}}}
return layer;}
function EbayGetSelect(pHtmlDoc,pName)
{if(!pHtmlDoc)
return null;var elem;var frms=pHtmlDoc.forms;var ln=frms.length;for(var i=0;i<ln;i++)
{var elems=frms[i].elements;elem=elems[pName];if(elem)
return elem;}
return null;}
function EbayGetRadio(pHtmlDoc,pName)
{if(!pHtmlDoc)
return null;var elem;var frms=pHtmlDoc.forms;var ln=frms.length;for(var i=0;i<ln;i++)
{var elems=frms[i].elements;elem=elems[pName];if(elem)
return elem;}
return null;}
function EbayRadio(pEbayDoc,pParent,pHTMLDoc,pElementName)
{if(!this.objType)
this.objType="EbayRadio";this.baseObject=EbayControl;this.baseObject(pEbayDoc,pParent,pHTMLDoc,pElementName);this.bindHTML=EbayBindHTMLRadio;this.bindEvents=EbayBindEventsRadio;this.getRadio=EbayGetRadio;this.onClick=null;this.onAfterLoad=null;}
window.EbayRadio=EbayRadio;function EbayBindHTMLRadio()
{this.htmlElement=this.getRadio(this.htmlDoc,this.htmlElementName);if(this.htmlElement)
{if(this.htmlElement.length&&this.htmlElement.length>0)
{for(var i=0;i<this.htmlElement.length;i++)
{if(this.htmlElement[i])
this.htmlElement[i].ebayControl=this;}}
else
this.htmlElement.ebayControl=this;}}
function EbayBindEventsRadio()
{if(this.htmlElement&&this.onClick)
{if(this.htmlElement.length&&this.htmlElement.length>0)
{for(var i=0;i<this.htmlElement.length;i++)
this.htmlElement[i].onclick=this.onClick;}
else
this.htmlElement.onclick=this.onClick;}
if(this.onAfterLoad)this.onAfterLoad();}
function EbayGetCheckBox(pHtmlDoc,pName)
{return this.ebayDoc.getFormElem(pName);}
function EbayCheckBox(pEbayDoc,pParent,pHTMLDoc,pElementName)
{if(!this.objType)
this.objType="EbayCheckBox";this.baseObject=EbayControl;this.baseObject(pEbayDoc,pParent,pHTMLDoc,pElementName);this.bindHTML=EbayBindHTMLCheckBox;this.bindEvents=EbayBindEventsCheckBox;this.getCheckBox=EbayGetCheckBox;this.onClick=null;this.onAfterLoad=null;}
window.EbayCheckBox=EbayCheckBox;function EbayBindHTMLCheckBox()
{this.htmlElement=this.getCheckBox(this.htmlDoc,this.htmlElementName);if(this.htmlElement)this.htmlElement.ebayControl=this;}
function EbayBindEventsCheckBox()
{if(this.onClick&&typeof(this.htmlElement)!='undefined')
this.htmlElement.onclick=this.onClick;if(this.onAfterLoad)this.onAfterLoad();}
function EbayGetInput(pHtmlDoc,pName)
{return this.ebayDoc.getFormElem(pName);}
function EbayInput(pEbayDoc,pParent,pHTMLDoc,pElementName)
{if(!this.objType)
this.objType="EbayInput";this.baseObject=EbayControl;this.baseObject(pEbayDoc,pParent,pHTMLDoc,pElementName);this.bindHTML=EbayBindHTMLInput;this.bindEvents=EbayBindEventsInput;this.getInput=EbayGetInput;this.onClick=null;this.onAfterLoad=null;}
window.EbayInput=EbayInput;function EbayBindHTMLInput()
{this.htmlElement=this.getInput(this.htmlDoc,this.htmlElementName);if(this.htmlElement)this.htmlElement.ebayControl=this;}
function EbayBindEventsInput()
{if(this.onClick)this.htmlElement.onclick=this.onClick;if(this.onAfterLoad)this.onAfterLoad();}
function EbayGetObject(pDoc,pObjName)
{return pDoc.getElementById(pObjName);}
function EBayConfig(pName)
{if(!this.objType)
this.objType="EBayConfig";this.name=pName;this.set=ebConfigSetVar;this.get=ebConfigGetString;this.copy=ebConfigObjectCopy;}
window.EBayConfig=EBayConfig;function ebConfigSetVar(pName,pVal)
{eval("this."+pName+" = '"+pVal+"';");}
function ebConfigGetString(pName,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10)
{var s=eval("this."+pName);var r="";if(!s)
return r;var len=s.length;for(var i=0;i<len;i++)
{if(s.substring(i,i+2)=="##")
{r+=new String(eval('p'+s.charAt(i+2)));i+=2;}
else
r+=s.charAt(i);}
return r;}
function ebConfigObjectCopy(pObj)
{if(pObj)
{for(var i in pObj)
{var prop=eval("pObj."+i),ti="this."+i;if(prop&&(typeof(eval("prop.copy"))=="function")&&prop.objType&&(i!="parent"))
{eval(ti+"=new "+prop.objType+"();");eval(ti).copy(prop);}
if(prop&&typeof prop=="function")continue;if(prop&&typeof prop=="object"&&prop.toString().indexOf("function")==0)continue;eval(ti+"=prop;");}}}
function EbayGetForm(pDoc,pFormName)
{if(!pDoc)
return null;var frms=pDoc.forms;var ln=frms.length;for(var i=0;i<ln;i++)
{if(frms[i].name==pFormName)
return frms[i];}}
function EbayLabel(pEbayDoc,pParent,pHTMLDoc,pElementName,pLabel,pCssClass,pCssInline)
{if(!this.objType)
this.objType="EbayLabel";this.baseObject=EbayControl;this.baseObject(pEbayDoc,pParent,pHTMLDoc,pElementName);this.label=pLabel;this.getLayer=EbayGetLayer;this.bindHTML=EbayLabelBindHTML;this.onUpdate=null;this.setLabel=EbaySetLabel;this.getLabel=EbayGetLabel;this.cssInline=pCssInline;this.cssClass=pCssClass;this.hide=EbayHideLabel;this.show=EbayShowLabel;this.moveLyr=EbayMoveLayer;this.moveTo=EbayLabelMoveTo;}
function EbayLabelBindHTML()
{if(this.ebayDoc.htmlDoc&&this.htmlElementName)
{this.htmlElement=this.getLayer(this.ebayDoc.htmlDoc,this.htmlElementName);this.htmlElement.ebayControl=this;if(document.layers)
{var lyr=this.htmlElement;var peer=null;if(!lyr.peerLayer)
{peer=lyr.peerLayer=new Layer(lyr.clip.width,lyr.parentLayer);peer.clip.height=lyr.clip.height;peer.moveTo(lyr.pageX,lyr.pageY);lyr.visibility='hide';peer.visibility='show';}}
if(this.label)
{this.setLabel(this.label);}}}
function EbaySetLabel(text)
{var str='<span ';if(this.cssClass)
{str+='class="'+this.cssClass+'" ';}
if(this.cssInline)
{str+='style="'+this.cssInline+'" ';}
str+='>'+text+'</span>';if(document.all||document.getElementById)
{this.htmlElement.innerHTML=str;}
else if(document.layers)
{with(this.htmlElement.peerLayer.document)
{open();write(str);close();}}
this.label=text;}
function EbayGetLabel()
{return this.label;}
function EbayHideLabel()
{if(document.getElementById||document.all)
{this.htmlElement.style.visibility='hidden';}
else if(document.layers)
{this.htmlElement.visibility='hide';this.setLabel('');}}
function EbayShowLabel()
{if(document.getElementById||document.all)
{this.htmlElement.style.visibility='visible';}
else if(document.layers)
{this.htmlElement.visibility='show';}}
function EbayLabelMoveTo(x,y)
{if(document.getElementById||document.all)
{this.moveLyr(this.htmlElement,x,y);}
else if(document.layers)
{this.moveLyr(this.htmlElement.peerLayer,x,y);}}
function tbd()
{}

//2@@m99

function Localize(pStr){return pStr;}
var onLoadFunctions=new Array();function addOnLoadFunction(f){onLoadFunctions[onLoadFunctions.length]=f;}
function OnLoad(){if(anchorOnLoad&&document.getElementById){var anchorData=document.getElementById("onLoadAnchor");if(anchorData!=null&&typeof(window.scrollTo)!="undefined"){var jumpTo=anchorData.href.substring(anchorData.href.lastIndexOf("#")+1)+"Anchor";var jump=document.getElementById(jumpTo);if(jump!=null&&typeof(window.scrollTo)!="undefined"){window.scrollTo(0,findPosY(jump));}}}
for(var i=0;i<onLoadFunctions.length;i++){var current=onLoadFunctions[i];var parenIndex=current.indexOf("(");if(parenIndex!=-1){var command=current.substring(0,parenIndex);eval('if(typeof('+command+') == "function") {eval(onLoadFunctions[i]);}');}}}
function findPosY(obj){var curtop=0;if(obj.offsetParent)
{while(obj.offsetParent)
{curtop+=obj.offsetTop
obj=obj.offsetParent;}}
else if(obj.y)
curtop+=obj.y;return curtop;}
function getElement(id){if(document.all){return document.all(id);}
if(document.getElementsByTagName)
return document.getElementsByName(id);}
function periodFilterChanged(selectElement,pPeriodElem)
{var aElems=["SubmitAction.ChangePeriod","SubmitAction.ChangeFormat","SubmitAction.BulkCopyToFolder","SubmitAction.BulkMoveToFolder","SubmitAction.MessageCopyTo","SubmitAction.MessageMoveTo","SubmitAction.BulkMarkAs"];if(pPeriodElem)
{for(var i=0;i<aElems.length;i++)
{var eL=selectElement.form.elements[aElems[i]];if(eL)
{if(aElems[i]==pPeriodElem)
eL.value="Go";else
eL.name="X"+aElems[i];}}
selectElement.form.submit();}
else
{var selectedText=selectElement.options[selectElement.selectedIndex].text;setGoButtonState(selectElement.form,shouldEnableGoButton(selectedText));}}
function shouldEnableGoButton(text){return text.indexOf("coming soon")==-1;}
function setGoButtonState(aForm,enabled){var cntrl=findGoButton(aForm);var u="undefined";if(cntrl)
if(typeof(cntrl.disabled)!=u)
cntrl.disabled=!enabled;}
function findGoButton(aForm){var formElements=aForm.elements;for(var i=0;i<formElements.length;i++){var anElement=formElements[i];if(anElement.name=="SubmitAction.ChangePeriod")
return anElement;}}
function checkAll(field){var value=field.checked;var form=field.form;var items=form.elements['LineID'];if(!items)
items=form.elements['acceptItem'];if(items.length){var i;for(i=0;i<items.length;i++)
if(!items[i].disabled)
items[i].checked=value;}else{if(!items.disabled)
items.checked=value;}}
function checkboxClicked(form){var selectAllBox=form.elements['LineIDALL'];var items=form.elements['LineID'];if(!items)
items=form.elements['acceptItem'];selectAllBox.checked=allCheckboxesAreChecked(items);}
function allCheckboxesAreChecked(checkboxes){if(checkboxes.length){var allChecked=true;for(var i=0;i<checkboxes.length;i++){if(!checkboxes[i].checked)
return false;}
return true;}
return checkboxes.checked;}
function verifyRemoveContainer(msg){var yes=confirm(msg);if(yes)
return true;else
return false;}
function statsParam(ctr,act){var a=window["ssPrefix"];var b=window["ssCtr"+ctr];var c=window["ssWgt"+act];if(typeof(a)=="undefined"||typeof(b)=="undefined"||typeof(c)=="undefined"){return"";}
return"&"+a+b+c;}
function cobrandURL(url){if(typeof(ebayCBFCobrandURL)!="undefined"){return ebayCBFCobrandURL(url);}
else{return url;}}
function verifyDeleteSalesPromo(strConfirm){if(confirm(strConfirm))
{return true;}
return false;}
function actionViewSimilar(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,oid,fis,leafCat)
{var sQryStr="";if(typeof(fis)!="undefined")
sQryStr="&itemstate="+fis;if(typeof(leafCat)!="undefined")
sQryStr+="&sibeleafcat="+leafCat;var s="<a href='"+cobrandURL(viewSimilarURL)+enctitle+"&itemid="+sid+sQryStr+statsParam(ctr,"MEVSI");if(ctr=="Lost")
s+="&sellername="+slr+"&Search_VSI_SameSeller=1";return s+"'>"+msg+"</a>";}
function actionView(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(viewItemURL)+sid+statsParam(ctr,"VI")+"'>"+msg+"</a>";}
function actionBid(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(viewItemURL)+sid+statsParam(ctr,"BID")+"'><img src='"+btnbBidNow+"' border='0'/></a>";}
function actionAddDesc(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(addDescURL)+sid+statsParam(ctr,"AID")+"'>"+msg+"</a>";}
function actionBIN(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(viewItemURL)+sid+statsParam(ctr,"BIN")+"'><img src='"+btnBIN+"' border='0'/></a>";}
function actionViewOther(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(viewOtherURL)+slr+statsParam(ctr,"MESOI")+"'>"+msg+"</a>";}
function actionViewRelated(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(viewRelatedURL)+cat+"/index.html?"+statsParam(ctr,"MEFOI")+"'>"+msg+"</a>";}
function actionPayNow(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(payNowURL)+sid+"&transactionid="+tid+"&quantity="+qty+statsParam(ctr,"PN")+"'><img src='"+btnPayNow+"' border='0'/></a>";}
function actionPayStatus(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(viewPayURL)+sid+"&transId="+tid+statsParam(ctr,"VPS")+"'>"+msg+"</a>";}
function actionLeaveSellerFeedback(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(transFeedbackURL)+sid+"&transactID="+tid+"&useridto="+slr+statsParam(ctr,"LF")+"'>"+msg+"</a>";}
function actionLeaveBuyerFeedback(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(transFeedbackURL)+sid+"&transactID="+tid+"&useridto="+byr+statsParam(ctr,"LF")+"'>"+msg+"</a>";}
function actionMarkNotShipped(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(lineActionURL)+rid+"&View="+ctr+"&SubmitAction.MarkNotShipped=x'>"+msg+"</a>";}
function actionMarkPaid(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(lineActionURL)+rid+"&View="+ctr+"&SubmitAction.MarkPaid=x"+statsParam(ctr,"MPD")+"'>"+msg+"</a>";}
function actionMarkNotPaid(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(lineActionURL)+rid+"&View="+ctr+"&SubmitAction.MarkNotPaid=x'>"+msg+"</a>";}
function actionSendInvoice(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(invoiceURL)+sid+"&transId="+tid+statsParam(ctr,"INV")+"'>"+msg+"</a>";}
function actionSendRevisedInvoice(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(invoiceURL)+sid+"&transId="+tid+statsParam(ctr,"RINV")+"'>"+msg+"</a>";}
function actionSellSimilar(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(sellSimilarURL)+sid+statsParam(ctr,"SSI")+"'>"+msg+"</a>";}
function actionSellLikeItem(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(sellLikeItemURL)+sid+statsParam(ctr,"LILT")+"'>"+msg+"</a>";}
function actionRYI(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(ryiURL)+sid+statsParam(ctr,"RVS")+"'>"+msg+"</a>";}
function actionEndItem(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(endItem)+sid+statsParam(ctr,"ENDI")+"'>"+msg+"</a>";}
function actionEndGTCItem(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(endGTCItem)+sid+statsParam(ctr,"ENDI")+"'>"+msg+"</a>";}
function actionRelist(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(relistURL)+sid+statsParam(ctr,"RL")+"'>"+msg+"</a>";}
function actionRelistFlagged(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(relistURL)+sid+statsParam(ctr,"RL")+"'>"+msg+"</a> <img src='http://pics.ebaystatic.com/aw/pics/myebay/blueCheck.gif'>";}
function actionTCR(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(tcrURL)+sid+statsParam(ctr,"TCR")+"'>"+msg+"</a>";}
function actionSecondChance(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(secondChanceURL)+sid+statsParam(ctr,"SCOF")+"'>"+msg+"</a>";}
function actionEditPromo(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(editPromoURL)+sid+statsParam(ctr,"EPRO")+"'>"+msg+"</a>";}
function actionReviewItemDescription(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(ReviewItemDescriptionURL)+sid+statsParam(ctr,"RID")+"'>"+msg+"</a>";}
function actionViewShippingStatus(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(viewShippingURL)+sid+"&transId="+tid+statsParam(ctr,"VSHP")+"'>"+msg+"</a>";}
function actionPrintPostage(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,oid){if(typeof(oid)=="undefined"||oid=='')
return"<a href='"+cobrandURL(printPostageURL)+sid+"&transactionid="+tid+statsParam(ctr,"PSHP")+"'>"+msg+"</a>";else
return"<a href='"+cobrandURL(orderPrintPostageURL)+"&orderId="+oid+statsParam(ctr,"PSHP")+"'>"+msg+"</a>";}
function actionRefineSearch(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(refineSearchUrl)+qry+"'>"+msg+"</a>";}
function actionPostToWIN(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(PostToWINURL)+"&itemId="+sid+"'>"+msg+"</a>";}
function actionWINPost(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){if(ctr=="Lost")
return"<a href='"+cobrandURL(WINPostURL)+"&itemId="+sid+"'>"+msg+"</a>";else
return"<a href='"+cobrandURL(WINPostURL)+qry+"'>"+msg+"</a>";}
function actionEditPreferences(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){var lName='editMax_'+rid;if(ebay&&ebay.oDocument.getConfig('SavedSearch.EmailOption'))
return"<a name='"+lName+"' onclick='genSearchFavoriteJS(\""+lName+"\");return false;' href='"+cobrandURL(editPreferencesURL)+rid+"'>"+msg+"</a>";else
return"<a href='"+cobrandURL(editPreferencesURL)+rid+"'>"+msg+"</a>";}
function actionEditPreferencesSaveSearch(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(editPreferencesSaveSearchURL)+rid+"'>"+msg+"</a>";}
function actionEditPreferencesSaveProduct(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(editPreferencesSaveProductURL)+rid+"'>"+msg+"</a>";}
function actionSignUpForEmails(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){var lName='signupMax_'+rid;if(ebay&&ebay.oDocument.getConfig('SavedSearch.EmailOption'))
return"<a name='"+lName+"' onclick='genSearchFavoriteJS(\""+lName+"\");return false;' href='"+cobrandURL(signUpForEmailsURL)+rid+"'>"+msg+"</a>";else
return"<a href='"+cobrandURL(signUpForEmailsURL)+rid+"'>"+msg+"</a>";}
function actionSignUpForEmailsSaveSearch(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(signUpForEmailsSaveSearchURL)+rid+"'>"+msg+"</a>";}
function actionSignUpForEmailsSaveProduct(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(signUpForEmailsSaveProductURL)+rid+"'>"+msg+"</a>";}
function actionGetAReminder(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(getAReminderURL)+rid+"'>"+msg+"</a>";}
function actionViewRelistedItem(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,oid){if(typeof(oid)!="undefined"&oid!='')
return"<a href='"+cobrandURL(viewRelistedItemURL)+oid+statsParam(ctr,"VRI")+"'>"+msg+"</a>";else
return"<a href='"+cobrandURL(viewRelistedItemURL)+sid+statsParam(ctr,"VRI")+"'>"+msg+"</a>";}
function actionSendPaymentReminder(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(sendPaymentReminderURL)+sid+statsParam(ctr,"PMTR")+"'>"+msg+"</a>";}
function msgPaymentReminderSent(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return msg;}
function actionRefineStoreSearch(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(refineStoreSearchURL)+qry+"'>"+msg+"</a>";}
function actionContactBuyer(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(contactInfo)+sid+"&transactid="+tid+"&useridto="+byr+"&"+statsParam(ctr,"CBYR")+"'>"+msg+"</a>";}
function actionContactSeller(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(contactInfo)+sid+"&transactid="+tid+"&useridto="+byr+"&"+statsParam(ctr,"CSLR")+"'>"+msg+"</a>";}
function actionContactSellerClassifieds(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(contactInfo)+sid+statsParam(ctr,"CSLR")+"'><img src='"+btnContactSeller+"' border='0'/></a>";}
function actionRespondToQuestion(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(RespondToQuestionURL)+sid+statsParam(ctr,"RTQ")+"'>"+msg+"</a>";}
function actionMarkAsAnsweredMyebay(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(MarkAsAnsweredMyebayURL)+sid+statsParam(ctr,"MAA")+"'>"+msg+"</a>";}
function actionMarkAsAnsweredMyebayWithQuestions(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(MarkAsAnsweredMyebayWithQuestionsURL)+sid+statsParam(ctr,"MAA")+"'>"+msg+"</a>";}
function actionMarkAsAnsweredSmpro(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(MarkAsAnsweredSmproURL)+sid+statsParam(ctr,"MAA")+"'>"+msg+"</a>";}
function actionMarkAsAnsweredSmproWithQuestions(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(MarkAsAnsweredSmproWithQuestionsURL)+sid+statsParam(ctr,"MAA")+"'>"+msg+"</a>";}
function actionOnline(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(onlineURL)+sid+statsParam(ctr,"OL")+"'>"+msg+"</a>";}
function actionFixedPrice(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(fixedPriceURL)+sid+statsParam(ctr,"FP")+"'>"+msg+"</a>";}
function actionRelistAsStore(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(relistAsStoreURL)+sid+statsParam(ctr,"RAS")+"'>"+msg+"</a>";}
function actionReportUpi(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty){return"<a href='"+cobrandURL(reportUpiURL)+rid+statsParam(ctr,"UPI")+"'>"+msg+"</a>";}
function actionReportInr(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty){return"<a href='"+cobrandURL(reportInrURL)+rid+statsParam(ctr,"INR")+"'>"+msg+"</a>";}
function actionResolveAProblemUpi(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty){return"<a href='"+cobrandURL(resolveAProblemUpiURL)+sid+"&TransactionId="+tid+"&DisputeType=1'>"+msg+"</a>";}
function actionResolveAProblemInr(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty){return"<a href='"+cobrandURL(resolveAProblemInrURL)+sid+"&TransactionId="+tid+"&DisputeType=3'>"+msg+"</a>";}
function actionViewInrCaseDetails(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty){return"<a href='"+cobrandURL(viewCaseDetailsURL)+sid+"&TransactionId="+tid+"&DisputeType=3'>"+msg+"</a>";}
function actionViewUpiCaseDetails(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty){return"<a href='"+cobrandURL(viewCaseDetailsURL)+sid+"&TransactionId="+tid+"&DisputeType=1'>"+msg+"</a>";}
function actionBuyerSelectEscrow(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(escrowPayEbayURL)+sid+"&transactionId="+tid+"&orderId=0"+"'><img src='"+btnPayNow+"' border='0'/></a>";}
function actionEscrowShipItem(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(escrowShipItemURL)+eid+statsParam(ctr,"ESHP")+"'>"+msg+"</a>";}
function actionAcceptEscrowPaymentRelease(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(escrowAcceptItemURL)+eid+statsParam(ctr,"PAPP")+"'>"+msg+"</a>";}
function actionRequestEscrowRefundReq(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(escrowRequestRefundURL)+eid+statsParam(ctr,"PREF")+"'>"+msg+"</a>";}
function actionUpdateWalletInfo(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(escrowUpdateWalletURL)+eid+statsParam(ctr,"PWAL")+"'>"+msg+"</a>";}
function actionRespondEscrowRefundReq(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(escrowRespondRefundURL)+eid+statsParam(ctr,"EREF")+"'>"+msg+"</a>";}
function actionViewSellerOtherItems(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(viewSellersOtherItemsURL)+slr+statsParam(ctr,"MESOI")+"'>"+msg+"</a>";}
function actionEditSellerEmailFrequencyPreferences(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(editSellerEmailPreferencesURL)+slr+statsParam(ctr,"EP")+"'>"+msg+"</a>";}
function actionAddFavoriteSeller(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(addFavoriteSellerURL)+slr+"&View="+ctr+statsParam(ctr,"ADDFS")+"'>"+msg+"</a>";}
function actionAddCurrentSellerToFavoriteSellers(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(addFavoriteSellerURL)+slr+"&View="+ctr+statsParam(ctr,"ADDFS")+"'>"+msg+"</a>";}
function actionViewSellerStore(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(viewSellersStoreURL)+rid+statsParam(ctr,"MESST")+"'>"+msg+"</a></nobr>";}
function actionContactSellerMember(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return'<a href="'+cobrandURL(ContactSellerURL)+slr+'&iid='+sid+'&redirect=0'+statsParam(ctr,"SCI")+'">'+msg+'</a>';}
function actionAddTracking(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return'<a href="'+cobrandURL(addTrackingURL)+'&LineId='+rid+'">'+msg+'</a>';}
function actionContactBuyerMember(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return'<a href="'+cobrandURL(ContactBuyerURL)+byr+'&iid='+sid+'&redirect=0'+statsParam(ctr,"BCI")+'">'+msg+'</a>';}
function actionRequestTotal(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return'<a href="'+cobrandURL(RequestTotalURL)+sid+'&transactionid='+tid+'&quantity=1&editAddress=0&level=1&paymentMethod=-2'+statsParam(ctr,"RT")+'">'+msg+'</a>';}
function actionRequestTotalButton(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return'<a href="'+cobrandURL(RequestTotalURL)+sid+'&transactionid='+tid+'&quantity=1&editAddress=0&level=1&paymentMethod=-2">'+'<img src="'+btnRequestTotal+'" border="0"/>'+'</a>';}
function actionGetMembersContactInfo(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return'<a href="'+cobrandURL(GetMembersContactInfoURL)+sid+'&transactid='+tid+'">'+msg+'</a>';}
function actionViewPayPalPaymentBuyer(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return'<a href="'+cobrandURL(ViewPayPalPaymentURL)+sid+'&transid='+tid+'&buyerorseller=0'+statsParam(ctr,"VPP")+'">'+msg+'</a>';}
function actionViewPayPalPaymentSeller(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return'<a href="'+cobrandURL(ViewPayPalPaymentURL)+sid+'&transid='+tid+'&buyerorseller=1'+statsParam(ctr,"VPP")+'">'+msg+'</a>';}
function actionSecondChanceOffer(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return'<a href="'+cobrandURL(SecondChanceOfferURL)+sid+'">'+msg+'</a>';}
function actionReviewLeads(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return'<a href="'+cobrandURL(reviewLeadsURL)+sid+'">'+msg+'</a>';}
function actionSellingReviewLeads(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return'<a href="'+cobrandURL(sellingreviewLeadsURL)+sid+'">'+msg+'</a>';}
function actionUnsoldReviewLeads(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return'<a href="'+cobrandURL(unsoldreviewLeadsURL)+sid+'">'+msg+'</a>';}
function actionUncombineInvoice(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,oid){var pagename="UNCB";return'<a href="'+cobrandURL(UncombineInvoiceURL)+oid+'&mode=0&'+statsParam(ctr,pagename)+'returnURL='+encodeURIComponent(MyeBayCurrentPageURL+replacePagename(window.location.search?("&"+window.location.search.split("&")[2]):"",pagename))+'">'+msg+'</a>';}
function actionArrangeFreightShipping(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return'<a href="'+arrangeFreightShippingURL+'">'+msg+'</a>';}
function actionMarkShipped(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,oid){if(typeof(MarkAsShippedURL)!='undefined')
{var pagename="SHP";if(typeof(oid)=="undefined"||oid=='')
return'<a href="'+cobrandURL(MarkAsShippedURL)+'itemid='+sid+'&transid='+tid+statsParam(ctr,pagename)+'&ru='+encodeURIComponent(MyeBayCurrentPageURL+replacePagename(window.location.search?("&"+window.location.search.split("&")[2]):"",pagename)+'#'+rid+'Anchor')+'">'+msg+'</a>';else
return'<a href="'+cobrandURL(MarkAsShippedURL)+'orderid='+oid+statsParam(ctr,pagename)+'&ru='+encodeURIComponent(MyeBayCurrentPageURL+replacePagename(window.location.search?("&"+window.location.search.split("&")[2]):"",pagename)+'#'+rid+'Anchor')+'">'+msg+'</a>';}
else
return"<a href='"+cobrandURL(lineActionURL)+rid+"&View="+ctr+"&SubmitAction.MarkShipped=x"+statsParam(ctr,"SHP")+"'>"+msg+"</a>";}
function actionMarkShippedByCashOnDelivery(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,oid){var pagename="SHP";if(typeof(MarkShippedByCashOnDeliveryURL)!='undefined')
{if(typeof(oid)=="undefined"||oid=='')
return'<a href="'+cobrandURL(MarkShippedByCashOnDeliveryURL)+'itemid='+sid+'&transid='+tid+statsParam(ctr,pagename)+'&ru='+escape(MyeBayCurrentPageURL+replacePagename(window.location.search?("&"+window.location.search.split("&")[2]):"",pagename)+'#'+rid+'Anchor')+'">'+msg+'</a>';else
return'<a href="'+cobrandURL(MarkShippedByCashOnDeliveryURL)+'orderid='+oid+statsParam(ctr,pagename)+'&ru='+escape(MyeBayCurrentPageURL+replacePagename(window.location.search?("&"+window.location.search.split("&")[2]):"",pagename)+'#'+rid+'Anchor')+'">'+msg+'</a>';}
else
return"<a href='"+cobrandURL(lineActionURL)+rid+"&View="+ctr+"&SubmitAction.MarkShippedByCashOnDelivery=x"+statsParam(ctr,"SHP")+"'>"+msg+"</a>";}
function actionMarkPaymentReceived(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,oid){var pagename="MPR";if(typeof(oid)=="undefined"||oid=='')
return'<a href="'+cobrandURL(MarkPaymentReceivedURL)+'itemid='+sid+'&transid='+tid+statsParam(ctr,pagename)+'&ru='+encodeURIComponent(MyeBayCurrentPageURL+replacePagename(window.location.search?("&"+window.location.search.split("&")[2]):"",pagename)+'#'+rid+'Anchor')+'">'+msg+'</a>';else
return'<a href="'+cobrandURL(MarkPaymentReceivedURL)+'orderid='+oid+statsParam(ctr,pagename)+'&ru='+encodeURIComponent(MyeBayCurrentPageURL+replacePagename(window.location.search?("&"+window.location.search.split("&")[2]):"",pagename)+'#'+rid+'Anchor')+'">'+msg+'</a>';}
function actionMarkPaymentSent(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,oid){var pagename="MPS";if(typeof(oid)=="undefined"||oid=='')
return'<a href="'+cobrandURL(MarkPaymentSentURL)+'itemid='+sid+'&transid='+tid+statsParam(ctr,pagename)+'&amp;ru='+encodeURIComponent(MyeBayCurrentPageURL+replacePagename(window.location.search?("&"+window.location.search.split("&")[2]):"",pagename)+'#'+rid+'Anchor')+'">'+msg+'</a>';else
return'<a href="'+cobrandURL(MarkPaymentSentURL)+'orderid='+oid+statsParam(ctr,pagename)+'&ru='+encodeURIComponent(MyeBayCurrentPageURL+replacePagename(window.location.search?("&"+window.location.search.split("&")[2]):"",pagename)+'#'+rid+'Anchor')+'">'+msg+'</a>';}
function actionUnmarkShipped(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,oid){var pagename="USHP";if(typeof(oid)=="undefined"||oid=='')
return'<a href="'+cobrandURL(UnmarkAsShippedURL)+'itemid='+sid+'&transid='+tid+statsParam(ctr,pagename)+'&ru='+encodeURIComponent(MyeBayCurrentPageURL+replacePagename(window.location.search?("&"+window.location.search.split("&")[2]):"",pagename)+'#'+rid+'Anchor')+'">'+msg+'</a>';else
return'<a href="'+cobrandURL(UnmarkAsShippedURL)+'orderid='+oid+statsParam(ctr,pagename)+'&ru='+encodeURIComponent(MyeBayCurrentPageURL+replacePagename(window.location.search?("&"+window.location.search.split("&")[2]):"",pagename)+'#'+rid+'Anchor')+'">'+msg+'</a>';}
function actionMarkNotShippedByCashOnDelivery(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,oid){var pagename="USHP";if(typeof(oid)=="undefined"||oid=='')
return'<a href="'+cobrandURL(MarkNotShippedByCashOnDeliveryURL)+'itemid='+sid+'&transid='+tid+statsParam(ctr,pagename)+'&ru='+escape(MyeBayCurrentPageURL+replacePagename(window.location.search?("&"+window.location.search.split("&")[2]):"",pagename)+'#'+rid+'Anchor')+'">'+msg+'</a>';else
return'<a href="'+cobrandURL(MarkNotShippedByCashOnDeliveryURL)+'orderid='+oid+statsParam(ctr,pagename)+'&ru='+escape(MyeBayCurrentPageURL+replacePagename(window.location.search?("&"+window.location.search.split("&")[2]):"",pagename)+'#'+rid+'Anchor')+'">'+msg+'</a>';}
function actionUnmarkPaymentReceived(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,oid){var pagename="UMPR";if(typeof(oid)=="undefined"||oid=='')
return'<a href="'+cobrandURL(UnmarkPaymentReceivedURL)+'itemid='+sid+'&transid='+tid+statsParam(ctr,pagename)+'&ru='+encodeURIComponent(MyeBayCurrentPageURL+replacePagename(window.location.search?("&"+window.location.search.split("&")[2]):"",pagename)+'#'+rid+'Anchor')+'">'+msg+'</a>';else
return'<a href="'+cobrandURL(UnmarkPaymentReceivedURL)+'orderid='+oid+statsParam(ctr,pagename)+'&ru='+encodeURIComponent(MyeBayCurrentPageURL+replacePagename(window.location.search?("&"+window.location.search.split("&")[2]):"",pagename)+'#'+rid+'Anchor')+'">'+msg+'</a>';}
function actionUnmarkPaymentSent(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,oid){var pagename="UMPS";if(typeof(oid)=="undefined"||oid=='')
return'<a href="'+cobrandURL(UnmarkPaymentSentURL)+'itemid='+sid+'&transid='+tid+statsParam(ctr,pagename)+'&ru='+EncodedReturnURL+encodeURIComponent(CurrentURLQueryString+'//'+window.location.hostname+window.location.pathname+replacePagename(window.location.search,pagename)+'#'+rid+'Anchor')+'">'+msg+'</a>';else
return'<a href="'+cobrandURL(UnmarkPaymentSentURL)+'orderid='+oid+statsParam(ctr,pagename)+'&ru='+EncodedReturnURL+encodeURIComponent(CurrentURLQueryString+'//'+window.location.hostname+window.location.pathname+replacePagename(window.location.search,pagename)+'#'+rid+'Anchor')+'">'+msg+'</a>';}
function actionWriteReview(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return'<a href="'+cobrandURL(WriteReviewURL)+sid+'">'+msg+'</a>';}
function actionWriteReviewSite(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return'<a href="'+cobrandURL(eid)+'">'+msg+'</a>';}
function actionEditSalesPromo(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(EditSalesPromoURL)+sid+"&mode=2'>"+msg+"</a>";}
function actionViewSalesPromo(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(ViewSalesPromoURL)+sid+"'>"+msg+"</a>";}
function actionDeleteSalesPromo(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(DeleteSalesPromoURL)+sid+"&mode=8' onclick='return verifyDeleteSalesPromo(strConfirm);'>"+msg+"</a>";}
function smGenActionURL(msg,sid,tid,oid,surl,tidparam){if(typeof(oid)=="undefined"||oid=='')
return'<a href="'+cobrandURL(surl)+'itemid='+sid+'&'+tidparam+'='+tid+'">'+msg+'</a>';else
return'<a href="'+cobrandURL(surl)+'orderid='+oid+'">'+msg+'</a>';}
function smGenSoldListingActionURL(msg,sid,tid,oid,sact){if(typeof(oid)=="undefined"||oid=='')
return'<a href="'+cobrandURL(SMSoldListingsActionURL)+sid+'%2B'+tid+'&'+sact+'">'+msg+'</a>';else
return'<a href="'+cobrandURL(SMSoldListingsActionURL)+oid+'&'+sact+'">'+msg+'</a>';}
function actionSMEmailBuyer(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,oid){return smGenActionURL(msg,sid,tid,oid,SMEmailURL,'transactionid');}
function actionSMLeaveFeedback(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,oid){return smGenActionURL(msg,sid,tid,oid,SMFeedbackURL,'transid');}
function actionSMPrint(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,oid){return smGenActionURL(msg,sid,tid,oid,SMPrintURL,'transactionid');}
function actionSMEditSalesRecord(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,oid){return smGenActionURL(msg,sid,tid,oid,SMEditSalesRecordURL,'transid');}
function actionSMMarkPaid(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,oid){return smGenSoldListingActionURL(msg,sid,tid,oid,'markselectedtop=MarkPaidStatus&marktop=Select');}
function actionSMMarkShipped(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,oid){return smGenSoldListingActionURL(msg,sid,tid,oid,'markselectedtop=MarkShippedStatus&marktop=Select');}
function actionSMMarkShippedByCashOnDelivery(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,oid){return smGenSoldListingActionURL(msg,sid,tid,oid,'markselectedtop=MarkShippedStatus&marktop=Select');}
function actionSMMarkPaidShipped(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,oid){return smGenSoldListingActionURL(msg,sid,tid,oid,'markselectedtop=MarkPaidAndShippedStatus&marktop=Select');}
function actionSMArchive(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,oid){return smGenSoldListingActionURL(msg,sid,tid,oid,'BulkArchive=1');}
function actionSMRelist(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,oid){return smGenSoldListingActionURL(msg,sid,tid,oid,'BulkRelist=1');}
function actionSMSellSimilar(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,oid){return smGenSoldListingActionURL(msg,sid,tid,oid,'BulkSellSimilar=1');}
function actionSMFileUPI(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,oid){if(typeof(oid)=="undefined"||oid=='')
return'<a href="'+cobrandURL(SMFileUPIUrl)+sid+'%2B'+tid+'%2B0">'+msg+'</a>';else
return'<a href="'+cobrandURL(SMFileUPIUrl)+sid+'%2B'+tid+'%2B'+oid+'">'+msg+'</a>';}
function actionSMFileFVF(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,oid){if(typeof(oid)=="undefined"||oid=='')
return'<a href="'+cobrandURL(SMFileFVFUrl)+sid+'%2B'+tid+'%2B0">'+msg+'</a>';else
return'<a href="'+cobrandURL(SMFileFVFUrl)+sid+'%2B'+tid+'%2B'+oid+'">'+msg+'</a>';}
function showActionPopup(refObj,item,trans,seller,category,buyer,lineID,view,queryString,params,quantity,escrowId,enctitle,oid,fis,leafCat){var i;var content='<table border="0" cellpadding="0" cellspacing="0">'+'<tr><td align="center" valign="middle"><table>';var action;var jsAction;for(i=0;i<params.length;i++)
{action=params[i];if(typeof(JSActions[action])!='undefined')
{jsAction=JSActions[action];content+='<tr><td class="menuitem"';if(i>0)
content+=' colspan="2">';else
content+='>';if(typeof(oid)=="undefined")
content+=jsAction[0](jsAction[1],item,trans,seller,category,buyer,lineID,view,queryString,quantity,escrowId,enctitle);else
{if(typeof(fis)!="undefined")
content+=jsAction[0](jsAction[1],item,trans,seller,category,buyer,lineID,view,queryString,quantity,escrowId,enctitle,oid,fis,leafCat);else
content+=jsAction[0](jsAction[1],item,trans,seller,category,buyer,lineID,view,queryString,quantity,escrowId,enctitle,oid);}
content+='</td>';if(i==0)
content+='<td><img src="'+btnMnuCls+'" align="absmiddle" border="0" onclick="hideDialogBox()"/></td>';content+='</tr>';}}
content+='</table></td></tr></table>';showDialogBox(refObj,content);}
var DialogBox=null;function showDialogBox(refObj,content){if(DialogBox==null){if(document.all){DialogBox=document.all("DialogBoxWrapper");}
else if(document.getElementById){DialogBox=document.getElementById("DialogBoxWrapper");}}
if(document.all){var left=event.clientX+document.body.scrollLeft+10;var top=event.clientY+document.body.scrollTop-10;}
else if(document.getElementById){var left=refObj.offsetLeft+refObj.offsetWidth;var top=refObj.offsetTop;while(refObj.offsetParent){refObj=refObj.offsetParent;left+=refObj.offsetLeft;top+=refObj.offsetTop;}}
DialogBox.innerHTML=content;DialogBox.style.left=left-150;DialogBox.style.top=top;DialogBox.style.display="block";var oF=document.createElement("IFRAME");oF.frameBorder=0;oF.src=picsDir+"s.gif";with(oF.style)
{position='absolute';zIndex='-1';top='0px'
left='0px';width=DialogBox.offsetWidth-2+'px';height=DialogBox.offsetHeight-2+'px';}
DialogBox.appendChild(oF);timeoutDialogBox();}
function hideDialogBox(){if(DialogBox)
DialogBox.style.display="none";}
if(document.all||document.getElementById){document.onmousemove=captureMousePosition;}
xMousePos=0;yMousePos=0;xMousePosMax=0;yMousePosMax=0;function timeoutDialogBox(){var boxX=parseFloat(DialogBox.offsetLeft);var boxY=parseFloat(DialogBox.offsetTop);var boxW=parseFloat(DialogBox.offsetWidth);var boxH=parseFloat(DialogBox.offsetHeight);if(xMousePos>boxX&&xMousePos<(boxX+boxW)&&yMousePos>boxY&&yMousePos<(boxY+boxH)){setTimeout("timeoutDialogBox(DialogBox)",1000);}
else if(actionClaimYourPayment){setTimeout("timeoutDialogBox(DialogBox)",1000);}
else{hideDialogBox();}}
function captureMousePosition(e){if(document.layers){xMousePos=e.pageX;yMousePos=e.pageY;xMousePosMax=window.innerWidth+window.pageXOffset;yMousePosMax=window.innerHeight+window.pageYOffset;}else if(document.all){xMousePos=window.event.clientX+document.body.scrollLeft;yMousePos=window.event.clientY+document.body.scrollTop;xMousePosMax=document.body.clientWidth+document.body.scrollLeft;yMousePosMax=document.body.clientHeight+document.body.scrollTop;}else if(document.getElementById){xMousePos=e.pageX;yMousePos=e.pageY;xMousePosMax=window.innerWidth+window.pageXOffset;yMousePosMax=window.innerHeight+window.pageYOffset;}}
function openEmailLogWindow(path){if(navigator.appName.indexOf("WebTV")==-1)
{hw=window.open(path,"EmailLog","height=520,width=450,status=no,toolbar=no,menubar=no,location=no,resizable=yes,titlebar=no");hw.focus();return false;}
else
{document.location.href=path;}}
function checkIt(pFormElement){if(pFormElement)
pFormElement.checked=true;}
function actionViewSellersResponse(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid){return"<a href='"+cobrandURL(viewSellersResponseURL)+sid+statsParam(ctr,"VSR")+"'>"+msg+"</a>";}
function actionRespondToOffers(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid){return"<a href='"+cobrandURL(respondToOffersURL)+sid+statsParam(ctr,"RTO")+"'>"+msg+"</a>";}
function actionMakeAnotherBestOffer(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid){return"<a href='"+cobrandURL(makeAnotherBestOfferURL)+sid+statsParam(ctr,"MABO")+"'>"+msg+"</a>";}
function actionEmailFavoriteSellerToFriend(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return getActionLink(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,"emailFavoriteSellerToFriendURL","ETF");}
function actionEmailFavoriteSearchToFriend(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return getActionLink(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,"emailFavoriteSearchToFriendURL","ETF");}
function actionEmailFavoriteCategoryToFriend(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return getActionLink(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,"emailFavoriteCategoryToFriendURL","ETF");}
function getActionLink(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,urlVarName,actionSuffix)
{var u="undefined";var v=eval(urlVarName);var url="<a href='";if(typeof(v)!=u)
url+=v;url+=rid;if(qry.length>0)
url+="&"+qry;if(cat.length>0)
url+="&id="+cat;url+=statsParam(ctr,actionSuffix);url+="'>"+msg+"</a>";return url;}
function replacePagename(pUrl,pNewPagename)
{var iStartIndex=-1;var iEndIndex=-1;var sUrl=pUrl.toLowerCase();iStartIndex=sUrl.indexOf("&sspagename");if(iStartIndex>=0)
{iEndIndex=sUrl.indexOf("&",iStartIndex+1);if(iEndIndex>0)
sUrl=pUrl.substr(0,iStartIndex)+pUrl.substr(iEndIndex);else
sUrl=pUrl.substr(0,iStartIndex);}
sUrl=sUrl+"&ssPageName="+pNewPagename;return sUrl}
function actionViewSimilarWidget(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a target='_parent' href='"+cobrandURL(viewSimilarURL)+enctitle+"'>"+msg+"</a>";}
function actionAddFavoriteSellerWidget(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a target='_parent' href='"+cobrandURL(addFavoriteSellerURL)+slr+"'>"+msg+"</a>";}
function actionViewSellerOtherItemsWidget(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a target='_parent' href='"+cobrandURL(viewSellersOtherItemsURL)+slr+"'>"+msg+"</a>";}
function actionBINWidget(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a target='_parent' href='"+cobrandURL(viewItemURL)+sid+"'><img src='"+btnBIN+"' border='0'/></a>";}
function actionBidWidget(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a target='_parent' href='"+cobrandURL(viewItemURL)+sid+"'><img src='"+btnbBidNow+"' border='0'/></a>";}
function actionMarkAsPaid(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(lineActionURL)+rid+"&View="+"SellerSpecificView"+"&SubmitAction.MarkPaid=x"+"&SellerLoginName="+slr+statsParam(ctr,"MPD")+"'>"+msg+"</a>";}
function actionMarkAsNotPaid(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(lineActionURL)+rid+"&View="+"SellerSpecificView"+"&SellerLoginName="+slr+"&SubmitAction.MarkNotPaid=x'>"+msg+"</a>";}
function actionConfigureWirelessAlerts(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(ConfigureWirelessAlertURL)+sid+statsParam(ctr,"CWA")+"'>"+msg+"</a>";}
function actionEditTemplate(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return'<a href="'+cobrandURL(editTemplateURL)+sid+'&BulkURLStack='+tid+'&CurrentPage='+qty+'&URLStack='+slr+'">'+msg+'</a>';}
function actionDelete(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return'<a href="'+cobrandURL(deleteURL)+sid+'&BulkURLstack='+tid+'&CurrentPage='+qty+'&URLStack='+slr+'">'+msg+'</a>';}
function actionEditStoreTemplate(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return'<a href="'+cobrandURL(editStoreTemplateURL)+sid+'">'+msg+'<a/>';}
function actionMLEditStoreTemplate(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return'<a href="'+cobrandURL(editMailingListURL)+sid+'">'+msg+'<a/>';}
function actionDeleteStoreTemplate(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return'<a href="'+cobrandURL(deleteStoreTemplateURL)+sid+'">'+msg+'<a/>';}
function actionSEDeleteStoreTemplate(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return'<a href="'+cobrandURL(deleteSEStoreTemplateURL)+sid+'">'+msg+'<a/>';}
function actionDuplicateStoreTemplate(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return'<a href="'+cobrandURL(duplicateStoreTemplateURL)+sid+'">'+msg+'<a/>';}
function actionViewStoreTemplate(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return'<a href="'+cobrandURL(viewStoreTemplateURL)+sid+'">'+msg+'<a/>';}
function actionIncreaseMaxBid(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return'<a href="'+cobrandURL(increaseMaxBidURL)+sid+"&mode=show&frompage=myebay&action.editbid=&groupid="+byr+'">'+msg+'<a/>';}
function actionEditMaxBid(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return'<a href="'+cobrandURL(editMaxBidURL)+"&groupid="+byr+"&itemids="+sid+"&frompage=myebay"+'">'+msg+'<a/>';}
function actionDeleteBidManager(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return'<a href="'+cobrandURL(deleteBidManagerURL)+sid+"&View="+ctr+"&bidGroupId="+byr+"&SubmitAction.BulkDelete=Delete"+'">'+msg+'<a/>';}
function actionManageGroup(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return'<a href="'+cobrandURL(manageGroupURL)+byr+'">'+msg+'<a/>';}
function actionAssignRules(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,oid){return'<a href="'+cobrandURL(assignRulesURL)+sid+'">'+msg+'<a/>';}
function actionRemoveRules(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,oid){return'<a href="'+cobrandURL(removeRulesURL)+sid+'">'+msg+'<a/>';}
function actionEditProductTemplate(msg,sid,url,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,oid){return'<a href="'+byr+'">'+msg+'<a/>';}
function actionEditInvTemplate(msg,sid,url,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,oid){return'<a href="'+url+'">'+msg+'<a/>';}
function actionAssignSco(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,oid){return'<a href="'+cobrandURL(automateScoURL)+sid+'">'+msg+'<a/>';}
function actionRemoveSco(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,oid){return'<a href="'+cobrandURL(removeScoURL)+sid+'">'+msg+'<a/>';}
function actionMarkAsResolved(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,oid){return'<a href="'+cobrandURL(MarkAsResolvedURL)+sid+"&ctime="+tid+"&ruletype="+slr+'">'+msg+'<a/>';}
function actionRelistManually(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,oid){return'<a href="'+cobrandURL(relistURL)+sid+'">'+msg+'<a/>';}
function actionSendSco(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,oid){return'<a href="'+cobrandURL(sendSco)+sid+'">'+msg+'<a/>';}
function actionUpdateQuantity(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,oid){return'<a href="'+cobrandURL(UpdateQuantityURL)+cat+'">'+msg+'<a/>';}
function actionSendEscrowPaymentOnline(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(SendEscrowPaymentOnlineURL)+sid+"&transactionid="+tid+"'>"+msg+"</a>";}
function actionViewOfferDetails(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid){return"<a href='"+cobrandURL(viewSellersResponseURL)+sid+statsParam(ctr,sMEVOD)+"'>"+msg+"</a>";}
function actionMakeBestOffer(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid){return"<a href='"+cobrandURL(makeAnotherBestOfferURL)+sid+statsParam(ctr,sMEBO)+"'>"+msg+"</a>";}
function actionReviewAllOffers(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid){return"<a href='"+cobrandURL(viewSellersResponseURL)+sid+statsParam(ctr,sRAO)+"'>"+msg+"</a>";}
function actionPrintCustomsForm(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){if(typeof(oid)=="undefined")
return"<a href='"+cobrandURL(printCustomsFormURL)+sid+"&transactionid="+tid+"'>"+msg+"</a>";else
return"<a href='"+cobrandURL(orderPrintCustomsFormURL)+"&orderId="+oid+"'>"+msg+"</a>";}
function actionDigitalDeliveryAvailable(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(DigitalDeliveryAvailableURL)+sid+"&transId="+tid+statsParam(ctr,"DD")+"'>"+msg+"</a>";}
function actionEditPost(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle)
{return"<a href='"+cobrandURL(winEditURL)+sid+statsParam(ctr,"WIN_EDIT")+" '>"+msg+"</a>";}
function actionDeletePost(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle)
{return"<a href='"+cobrandURL(winDeleteURL)+sid+statsParam(ctr,"WIN_DELETE")+" '>"+msg+"</a>";}
function actionRepost(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle)
{return"<a href='"+cobrandURL(winRepostURL)+sid+statsParam(ctr,"WIN_REPOST")+"'>"+msg+"</a>";}
function actionSearchInWantItNow(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle)
{return"<a href='"+cobrandURL(winSearchURL)+enctitle+"&sacategory="+cat+statsParam(ctr,"WIN_SEARCH")+"'>"+msg+"</a>";}
function writeActionButton(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13)
{if((document.all||document.getElementById)&&navigator.userAgent.toLowerCase().indexOf("webtv")==-1)
{var button="<img";button+=" src='"+a13+"'";button+=" id='trg"+a1+"'";button+=" onclick='showActionPopup(this, ";button+="\""+a1+"\",";button+="\""+a2+"\",";button+="\""+a3+"\",";button+="\""+a4+"\",";button+="\""+a5+"\",";button+="\""+a6+"\",";button+="\""+a7+"\",";button+="\""+a8+"\",";button+="["+a9+"],";button+="\""+a10+"\",";button+="\""+a11+"\",";button+="\""+a12+"\"";button+=")'>";document.write(button);}}
function actionGetSMSAlerts(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(getSMSAlertsURL)+sid+"&currentPage="+pageName+statsParam(ctr,"GSA")+"'>"+msg+"</a>";}
function actionGetIMAlerts(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return"<a href='"+cobrandURL(getIMAlertsURL)+sid+"&currentPage="+pageName+statsParam(ctr,"GIA")+"'>"+msg+"</a>";}
function actionFindPartsAndAccessories(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return'<a href="'+cobrandURL(qry)+'QQssPageName=MYV:MYE:PSRCH'+'">'+msg+'</a>';}
function actionFindVehicles(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return'<a href="'+cobrandURL(ctr)+'QQssPageName=MYV:MYE:VSRCH'+'">'+msg+'</a>';}
function actionSellThisVehicle(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return'<a href="'+cobrandURL(sellThisVehicleURL)+'">'+msg+'</a>';}
function actionFindAllRelatedItems(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return'<a href="'+cobrandURL(cat)+'QQssPageName=MYV:MYE:ESRCH'+'">'+msg+'</a>';}
function actionSeeCompletedListings(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return'<a href="'+cobrandURL(SeeCompletedListingsURL)+enctitle+'">'+msg+'</a>';}
function actionAddAnotherVehicle(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return'<a href="'+cobrandURL(AddAnotherVehicleURL)+'">'+msg+'</a>';}
function actionEditVehicle(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return'<a href="'+cobrandURL(EditVehicleURL)+sid+'&favsearchid='+rid+'">'+msg+'</a>';}
function actionEditSearch(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle){return'<a href="'+cobrandURL(editSearchURL)+rid+'">'+msg+'</a>';}
function actionClaimYourPayment(msg,sid,tid,slr,cat,byr,rid,ctr,qry,qty,eid,enctitle,oid){return'<a href="'+cobrandURL(claimYourPaymentURL)+sid+"&transactionid="+tid+'">'+msg+'</a>';}
function genSearchFavoriteJS(pLinkName)
{if(!ebay)return;var cfg=ebay.oDocument.getConfig('SavedSearch.EmailOption');if(cfg)
{ebay.clearControl(pLinkName);var cll=new EbayOverLayComboRequest(ebay.oDocument.oPage,pLinkName);cll.bHandleIn=cll.bHandleOut=false;cll.bUseLock=false;cll.bSimpleMode=true;cll.init();cll.createElements();cll.bind();cll.sendRequest();}}

//3@@m3

function getAdditionalStyles(baseURI){getAdditionalStylesheet(mainSheet,"ns","-ns");getAdditionalStylesheet(mainSheet,"gecko","-ns");}
function getAdditionalStylesheet(baseURI,targetClient,modifier){var client=getClientName();if(client==targetClient){var marker=baseURI.lastIndexOf(".");var prefix=baseURI.substring(0,marker);var suffix=baseURI.substring(marker);var path=prefix+modifier+suffix;document.write("<link rel=\"stylesheet\" type=\"text/css\" href=\""+path+"\"/>");}}
function getClientName(){var agent=navigator.userAgent.toLowerCase();var name=navigator.appName.toLowerCase();var isOpera=(agent.indexOf("opera")!=-1);var isIE=(!isOpera&&agent.indexOf("msie")!=-1);var isGecko=(!isOpera&&agent.indexOf("gecko")!=-1);var isNS=(!isOpera&&!isIE&&!isGecko&&name.indexOf("netscape")!=-1);if(isOpera){return"opera";}
if(isIE){return"ie";}
if(isGecko){return"gecko";}
if(isNS){return"ns";}
return"unknown";}

//4@@m21
<!--
function SendShippingParams(path,getValues){var originZipCode='';var bMachinable='';var packageSize='';var higherUnitWeight='';var lowerUnitWeight='';var packHandlingFee='';var price='';if(getValues)
{originZipCode=encodeURIComponent(document.PaymentShipping.originatingzip.value);if(document.PaymentShipping.oddshapedpackage.checked){document.PaymentShipping.oddshapedpackage.value="TRUE";}else{document.PaymentShipping.oddshapedpackage.value="FALSE";}
bMachinable=document.PaymentShipping.oddshapedpackage.value;for(i=0;i<document.PaymentShipping.packagesize.length;i++){if(document.PaymentShipping.packagesize[i].selected&&document.PaymentShipping.packagesize[i].value!="")
{packageSize=document.PaymentShipping.packagesize[i].value;}}
higherUnitWeight=encodeURIComponent(document.PaymentShipping.higherunitweight.value);lowerUnitWeight=encodeURIComponent(document.PaymentShipping.lowerunitweight.value);packHandlingFee=encodeURIComponent(document.PaymentShipping.packaginghandlingfee.value);price=encodeURIComponent(document.PaymentShipping.price.value);}
path=path+"&originZipCode="+originZipCode+"&bMachinable="+bMachinable+"&packageSize="+packageSize+"&higherUnitWeight="+higherUnitWeight+"&lowerUnitWeight="+lowerUnitWeight+"&packHandlingFee="+packHandlingFee+"&price="+price;hw=window.open(path,"shipcalc","height=620,width=570,status=no,toolbar=no,menubar=no,location=no,resizable=yes,scrollbars=yes,titlebar=no");hw.focus();return false;}
function openShippingCalculator(path,form){if(form>0){itemId=encodeURIComponent(document.shippingcalcbox.itemId.value);destinationZipCode=encodeURIComponent(document.shippingcalcbox.destinationZipCode.value);var tid='';if(document.shippingcalcbox.TransactionId)
tid=encodeURIComponent(document.shippingcalcbox.TransactionId.value);path=document.shippingcalcbox.action+"?EmitBuyerShippingCalculator&itemId="+itemId+"&destinationZipCode="+destinationZipCode+"&TransactionId="+tid;}
win=window.open(path,"","height=620,width=800,status=no,toolbar=no,menubar=no,location=no,scrollbars=yes,resizable=yes,titlebar=no");win.focus();return false;}
function openShippingCalculatorNoForm(path,itemId,zipCode){itemId=encodeURIComponent(itemId);path=path+"?EmitBuyerShippingCalculator&itemId="+itemId;if(zipCode){path=path+"&destinationZipCode="+zipCode;}
win=window.open(path,"","height=450,width=630,status=no,toolbar=no,menubar=no,location=no,scrollbars=yes,resizable=yes,titlebar=no");win.focus();return false;}

//5@@m2

function EBayObject(pParent,pName)
{if(!this.objType)
this.objType="EBayObject";this.parent=pParent;this.name=pName;this.document=document;this.window=window;this.REGISTER=ebObjectRegister;this.COPY=ebObjectCopy;}
function ebObjectRegister(pName,pVal)
{eval("this."+pName+"=pVal;");}
function ebObjectCopy(pObj)
{for(var i in pObj)
{var prop=eval("pObj."+i),ti="this."+i;if(prop&&(typeof(eval("prop.COPY"))=="function")&&prop.objType&&(i!="parent"))
{eval(ti+"=new "+prop.objType+"();");eval(ti).COPY(prop);}
else if((i!="window")&&(i!="document"))
eval(ti+"=prop;");}}
function EBayConfigObject(pName)
{if(!this.objType)
this.objType="EBayConfigObject";this.base=EBayObject;this.base(null,pName);this.GET=ebConfigObjectGetString;}
function ebConfigObjectGetString(pStrName)
{var s=eval("this."+pStrName);var re,token,args=arguments,len=args.length;for(var i=1;i<len;i++)
{token="<#"+i+"#>";re=new RegExp(token,"gi");if(s.indexOf(token)!=-1)
s=s.replace(re,args[i])}
return s;}
// b=8170495 -->