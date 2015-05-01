//<!--
//\include\js\lib\_toolbox\_base\ebayBasic.js@@\main\26

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
{var elem=frms[j].elements[i];var skEs=this.skipElems;var skip=false;if(skEs.length>0)
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
{case"hidden":break;case"text":break;case"button","submit":pElem.onclick=null;break;default:pElem.onclick=null;break;}}}
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
this.objType="ebayText";this.baseObject=EbayControl;this.baseObject(pebayDoc,pParent,pHTMLDoc,pElementName);this.disabled=false;this.bindHTML=EbayBindHTMLText;this.bindEvents=EbayBindEventsText;this.disable=EbayDisableText;this.enable=EbayEnableText;}
window.EbayText=EbayText;function EbayHyperLink(pEbayDoc,pParent,pHTMLDoc,pElementName,pLink)
{if(!this.objType)
this.objType="EbayHyperLink";this.baseObject=EbayControl;this.baseObject(pEbayDoc,pParent,pHTMLDoc,pElementName);this.link=pLink||null;this.mouseOverText="";this.urlPath=null;this.eventBound=false;this.bindHTML=EbayBindHTMLHyperLink;this.bindEvents=EbayBindEventsHyperLink;this.getLink=EbayGetLink;this.onClick=null;this.onMouseOver=EbayOnMouseOverHyperLink;this.onMouseOut=EbayOnMouseOutHyperLink;}
window.EbayHyperLink=EbayHyperLink;function EbayOnMouseOverHyperLink()
{}
function EbayOnMouseOutHyperLink()
{}
function EbayBindHTMLText()
{this.htmlElement=this.ebayDoc.getFormElem(this.htmlElementName,"text");if(this.htmlElement)
{this.htmlElement.ebayControl=this;if(this.disabled)
this.disable();else
this.enable();}}
function EbayBindEventsText()
{}
function EbayWriteLayer(pText)
{}
function EbayShowLayer(bShow)
{}
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
this.objType="EbayImage";this.baseObject=EbayControl;this.baseObject(pEbayDoc,pParent,pHTMLDoc,pElementName);this.image=pElementName||null;this.mouseOverText="";this.bindHTML=EbayBindHTMLImage;this.bindEvents=EbayBindEventsImage;this.getImage=EbayGetImage;}
window.EbayImage=EbayImage;function EbayBindHTMLImage()
{this.htmlElement=this.getImage(this.htmlDoc,this.htmlElementName);if(this.htmlElement)
this.htmlElement.ebayControl=this;}
function EbayBindEventsImage()
{}
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
this.objType="EbaySelect";this.baseObject=EbayControl;this.baseObject(pEbayDoc,pParent,pHTMLDoc,pElementName);this.bindHTML=EbayBindHTMLSelect;this.bindEvents=EbayBindEventsSelect;this.clearOptions=EbaySelectClearOptions;this.getSelect=EbayGetSelect;this.onAfterLoad=null;this.onChange=EbaySelectOnChange;}
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
function EbaySelectOnChange()
{}
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
// -->
