//<!--
//1@@m11

function EbayHTMLAnchor(pParent,pName,pDisabled,pCfg)
{if(!this.objType)
this.objType="EbayHTMLAnchor";this.base=EbayHTML;this.base(pParent,pName,pName,pDisabled,pCfg);this.getElem=ebHTMLAnchorGetElem;this.enableBase=this.enable;this.enable=ebHTMLAnchorEnable;this.subscribeEvents("onclick");}
function ebHTMLAnchorGetElem(pName)
{var d=this.oDocument.doc,l=null,len=null;l=d.links[pName];if(l)return l;if(d.getElementById)
l=d.getElementById(pName);if(l)return l;if(d.all)
l=d.all[pName];if(l)return l;if(d.layers)
{var lyrs=d.layers;len=lyrs.length;for(var i=0;i<len;i++)
{l=this.getElem(lyrs[i].document,pName);if(l)
return l;}}
len=d.links.length;for(var j=0;j<len;j++)
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

//2@@m7

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
{var elems=this.getElements(),len=elems.length;for(i=0;i<len;i++)
{var elem=elems[i];var type=elem.type;switch(type)
{case"text":case"textarea":elem.value="";break;case"checkbox":elem.checked=false;break;case"select-one":elem.selectedIndex=0;}}}

//3@@m11

function EbayHTMLText(pParent,pName,pDisabled,pCfg,bHidden)
{if(!this.objType)
this.objType="EbayHTMLText";this.base=EbayHTMLFormElem;this.base(pParent,pName,pDisabled,pCfg);this.value=ebHTMLTextValue;this.getValue=ebHTMLTextGetValue;this.setValue=ebHTMLTextSetValue;this.select=ebHTMLTextSelect;this.enableBase=this.enable;this.enable=ebHTMLTextEnable;if(bHidden!=true)
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
function ebHTMLTextEnable(pEnable)
{this.enableBase(pEnable);this.setStyle('backgroundColor',!pEnable?'#ccc':'#fff');}

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

//5@@m8

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
{rv=[];var len=e.length;for(var i=0;i<len;i++)
{if(e[i].checked)
rv[rv.length]=e[i].value;}
if(!rv.length)rv="";}
else
{if(pUnCheckedValue)
return e.value;if(e.checked)
return e.value;}
return rv;}

//6@@m16

function EbayHTMLBaseCheckboxRadio(pParent,pName,pDisabled,pCfg)
{if(!this.objType)
this.objType="EbayHTMLBaseCheckboxRadio";this.base=EbayHTMLFormElem;this.base(pParent,pName,pDisabled,pCfg);this.bGroup=false;this.bindHTML=ebHTMLBaseCheckboxRadioBindHTML;this.bindEvents=ebHTMLBaseCheckboxRadioBindEvents;this.check=ebHTMLBaseCheckboxRadioCheck;this.selectByIndex=ebHTMLBaseCheckboxRadioSelectByIndex;this.selectByValue=ebHTMLBaseCheckboxRadioSelectByValue;this.isCheckedByValue=ebHTMLBaseCheckboxRadioIsCheckedByValue;this.getValueByIndex=ebHTMLBaseCheckboxRadioGetValueByIndex;this.getIndexByValue=ebHTMLBaseCheckboxRadioGetIndexByValue;this.getValue=null;this.enableBase=this.enable;this.enable=ebHTMLBaseCheckboxRadioEnable;this.setValue=this.selectByValue;this.onBeforeCheck=null;this.onAfterCheck=null;this.subscribeEvents("onclick");}
function ebHTMLBaseCheckboxRadioBindHTML()
{with(this)
{eElem=getElem(sElemName);if(eElem)
{if(eElem.length)
{bGroup=true;var len=eElem.length;for(var i=0;i<len;i++)
assignJSObject(eElem[i]);cleanupMemory=ebHTMLBaseCheckboxRadioCleanupMemory;}
else
{bGroup=false;assignJSObject(eElem);}}
if(bDisabled)
enable(false);}}
function ebHTMLBaseCheckboxRadioCleanupMemory()
{var e=this.eElem;if(e)
{var len=e.length;for(var j=0;j<len;j++)
{for(var i in e[j].jsObjs)
{e[j].jsObjs[i]=null;}
e[j].jsObjs=null;}
this.eElem=null;}}
function ebHTMLBaseCheckboxRadioBindEvents()
{with(this)
{if(!eElem)
return;var e=aBindEvents,len=e.length,fStr;for(var i in e)
{var len2=eElem.length;if(len2&&len2>0)
{for(var ii=0;ii<len2;ii++)
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
{var len=e.length;for(var i=0;i<len;i++)
{if(e[i].value==pVal)
{e[i].checked=chx;if(onAfterCheck)
onAfterCheck();}}}
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

//7@@m8

function EbayHTMLRadio(pParent,pName,pDisabled,pCfg)
{if(!this.objType)
this.objType="EbayHTMLRadio";this.base=EbayHTMLBaseCheckboxRadio;this.base(pParent,pName,pDisabled,pCfg);this.getElem=ebHTMLRadioGetElem;this.getValue=ebHTMLRadioGetValue;this.getSelectedIndex=ebHTMLRadioGetSelectedIndex;}
function ebHTMLRadioGetElem(pName)
{return this.oDocument.getFormElem(pName,"radio");}
function ebHTMLRadioGetValue()
{var e=this.eElem;if(!e){return"";}
if(this.bGroup)
{var len=e.length;for(var i=0;i<len;i++)
{if(e[i].checked)
return e[i].value;}}
else
{if(e.checked)
return e.value;}
return"";}
function ebHTMLRadioGetSelectedIndex()
{var e=this.eElem;if(!this.bGroup)
return 0;else
{var len=e.length;for(var i=0;i<len;i++)
{if(e[i].checked)
return i;}}
return-1;}

//8@@e1

function EbayHTMLSelect(pParent,pName,pDisabled,pCfg)
{if(!this.objType)
this.objType="EbayHTMLSelect";this.base=EbayHTMLFormElem;this.base(pParent,pName,pDisabled,pCfg);this.iSelIndex=-1;this.createOption=ebHTMLSelectCreateOption;this.clearOptions=ebHTMLSelectClearOptions;this.getValueByIndex=ebHTMLSelectGetValueByIndex;this.getSelectedIndex=ebHTMLSelectGetSelectedIndex;this.getSelectedValue=ebHTMLSelectGetSelectedValue;this.getSelectedText=ebHTMLSelectGetSelectedText;this.getOptionsLength=ebHTMLSelectGetOptionsLength;this.setOption=ebHTMLSelectSetOption;this.insertOption=ebHTMLSelectInsertOption;this.deleteOption=ebHTMLSelectDeleteOption;this.selectByIndex=ebHTMLSelectSelectByIndex;this.selectByValue=ebHTMLSelectSelectByValue;this.selectByText=ebHTMLSelectSelectByText;this.doSelect=ebHTMLSelectDoSelect;this.getIndexByValue=ebHTMLSelectGetIndexByValue;this.getValue=this.getSelectedValue;this.setValue=this.selectByValue;this.subscribeEvents("onchange");}
function ebHTMLSelectClearOptions()
{var e=this.eElem;if(e)
{var opts=e.options;while(opts.length>0)
opts[opts.length-1]=null;}}
function ebHTMLSelectCreateOption(pName,pText,pClass,pDisabled)
{if(this.eElem)
{var nOpt=new Option(pText,pName,false,false),opts,lo,oC=ebay.oGlobals.oClient;if(typeof(pClass)!="undefined")
nOpt.className=pClass;if(typeof(pDisabled)!="undefined")
nOpt.disabled=pDisabled;opts=this.eElem.options;opts[opts.length]=nOpt;idx=opts.length-1;return idx;}}
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
function ebHTMLSelectSetOption(pVal,pText,pInd,pClass,pDisabled)
{if(this.eElem)
{if((pInd!=null)&&(pInd>-1))
{var o=this.eElem.options[pInd];o.value=pVal;o.text=pText;}
else
this.createOption(pVal,pText,pClass,pDisabled);}
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

//9@@m6

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

//10@@m7

function EbayHTMLImage(pParent,pName,pDisabled,pSource,pDisabledSource,pCfg)
{if(!this.objType)
this.objType="EbayHTMLImage";this.base=EbayHTML;this.base(pParent,pName,pName,pDisabled,pCfg);this.sEnabledSource=this.sDisabledSource=pSource;if(pDisabledSource)
this.sDisabledSource=pDisabledSource;this.getElem=ebHTMLImageGetElem;this.source=ebHTMLImageSource;this.enableBase=this.enable;this.enable=ebHTMLImageEnable;this.subscribeEvents("onclick","onmouseover","onmouseout");}
function ebHTMLImageGetElem(pName)
{return this.getDocElem(pName,'images');}
function ebHTMLImageSource(pSrc,pText)
{var im=this.eElem;if(typeof(im)=='undefined')
return;if(typeof(pSrc)=="undefined")
return(im)?im.src:"";else
{im.src=pSrc;if(pText!=null)
im.alt=pText;}}
function ebHTMLImageEnable(pEnable)
{with(this)
{enableBase(pEnable);if(sDisabledSource&&eElem)
eElem.src=(pEnable)?sEnabledSource:sDisabledSource;}}

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

//12@@m4

function EbayHTMLTextRestricted(pParent,pName,pCfg)
{if(!this.objType)
this.objType="EbayHTMLTextRestricted";this.base=EbayHTMLText;this.base(pParent,pName);this.oConfig=pCfg||null;this.subscribeEvents("oncopy","onpaste","oncut","ondragstart");this.oncopy=ebHTMLTextRestricted;this.onpaste=ebHTMLTextRestricted;this.oncut=ebHTMLTextRestricted;this.ondragstart=ebHTMLTextRestricted;}
function ebHTMLTextRestricted()
{return false;}

//13@@m3

function EbayParentChildSel(pParent,pName,pCfg)
{if(!this.objType)
this.objType="EbayParentChildSel";this.baseObject=EbayBaseControl;this.baseObject(pParent,pName);this.oConfig=pCfg?pCfg:null;this.parentCntrl=null;this.childCntrl=null;this.init=function()
{var c=this.oConfig;this.parentCntrl=new EbayHTMLSelect(this,c.parentCntrlId,false,c);this.parentCntrl._registerEvent("onchange","ebParentChildSelChangeChild");this.childCntrl=new EbayHTMLSelect(this,c.childCntrlId,false,c);}
this.init();}
function ebParentChildSelChangeChild()
{var a=this.oConfig.relationSets.parseSets();var c;var s=this.getSelectedValue();for(var i in a)
{c=a[i];for(var j=0,len=c.length;j<len;j++)
{if(s==c[j])
{var cntrl=this.parent.childCntrl;if(cntrl.getSelectedValue()!=i)
{cntrl.selectByValue(i);cntrl.eElem.onchange();}}}}}
function EbayParentChildSelConfig(name)
{if(!this.objType)
this.objType="EbayParentChildSelConfig";this.base=EbayConfig;this.base(name);this.parentCntrlId="";this.childCntrlId="";this.relationSets="";}

//14@@

function EbayPassword(pEbayDoc,pParent,pHTMLDoc,pElementName,pAction,pUpdatePassword)
{if(!this.objType)
this.objType="EbayPassword";this.baseObject=EbayControl;this.baseObject(pEbayDoc,pParent,pHTMLDoc,pElementName);this.checkStrings=new Array();this.inputs=new Array();this.master=null;this.updated=false;this.seclevel=0;this.oldvalue="fasdklasdklasdklasdklasdklasdklasdkla";this.updatePassword=pUpdatePassword||false;var type=this.ebayDoc.getFormElem(this.name).type;if(type=="hidden")
this.updatePassword=false;else if(type=="password")
this.updatePassword=true;this.bindHTML=EbayBindHTMLPassword;this.bindEvents=EbayBindEventsPassword;this.onchange=pAction;if(type=="password")this.onkeyup=pAction;}
function EbayBindHTMLPassword()
{this.htmlElement=this.ebayDoc.getFormElem(this.htmlElementName);if(this.htmlElement)this.htmlElement.ebayControl=this;}
function EbayBindEventsPassword()
{if(this.onkeyup)
this.htmlElement.onkeyup=this.onkeyup;if(this.onchange)
this.htmlElement.onchange=this.onchange;if(this.onmouseout)
this.htmlElement.onmouseout=this.onmouseout;}

//15@@

var psvPasswordElem="";var psvUserIdElem="";var psvFirstElem="";var psvLastElem="";var psvAddrElem="";var psvZipElem="";var psvCityElem="";var psvEmailElem="";var psvPhoneElem1="";var psvPhoneElem2="";var psvExtElem1="";var psvExtElem2="";var psvDayElem="";var psvMonElem="";var psvYearElem="";var psvMeterImage="";var security_level=new Array;var INVALID;var ebayDoc=null;var password=null;var userid=null;var firstname=null;var lastname=null;var addr=null;var city=null;var zip=null;var email=null;var birthyear=null;var birthmonth=null;var birthday=null;var phone1=null;var phone2=null;var ext1=null;var ext2=null;var psvOn;function psvLoadImages()
{iDir=ebay.oGlobals.oEnvironment.sPicsDir;if(typeof(country)!="undefined")
{if(country!="us")
iDir+=country+"/";}
var imgDirName="pswdSecVal";if(iDir.indexOf(imgDirName)==-1)
{iDir+=imgDirName+"/";}
security_level[0]=new Image;security_level[1]=new Image;security_level[2]=new Image;security_level[3]=new Image;security_level[0].src=iDir+"empty_148x11.gif";security_level[1].src=iDir+"state1_148x11.gif";security_level[2].src=iDir+"state2_148x11.gif";security_level[3].src=iDir+"state3_148x11.gif";INVALID=security_level.length;security_level[INVALID]=new Image;security_level[INVALID].src=iDir+"invalid_148x11.gif";}
function psvOnLoad(){psvLoadImages()
if(!psvOn)
return;ebayDoc=new EbayDocument(window,"RegistrationPassword");firstname=SetUpControl(psvFirstElem,UpdateName);lastname=SetUpControl(psvLastElem,UpdateName);city=SetUpControl(psvCityElem,UpdateName);zip=SetUpControl(psvZipElem,UpdateSimple);ext1=SetUpControl(psvExtElem1,UpdateNumeric);ext2=SetUpControl(psvExtElem2,UpdateNumeric);addr=SetUpControl(psvAddrElem,UpdateAddr);userid=SetUpControl(psvUserIdElem,UpdateUserId);email=SetUpControl(psvEmailElem,UpdateEmail);phone1=SetUpPhone(psvPhoneElem1,UpdatePhone);phone2=SetUpPhone(psvPhoneElem2,UpdatePhone);SetUpBirthday();password=SetUpControl(psvPasswordElem,UpdatePassword);updateMeter(0);}
function SetUpControl(fieldName,action){var control=null;if(fieldName!=""&&ebayDoc.getFormElem(fieldName))
{control=new EbayPassword(ebayDoc,ebayDoc,ebayDoc.htmlDoc,fieldName,action,true);control.bindHTML();control.bindEvents();control.onchange();}
return control;}
function SetUpPhone(fieldName,action)
{var control=null;if(fieldName!="")
{var phones=new Array();var p=fieldName.split(/\|/);for(var i=0,len=p.length;i<len;i++)
if(ebayDoc.getFormElem(p[i]))
phones[phones.length]=p[i];var l=phones.length-1;if(l!=-1)
{p=new Array();for(var i=0;i<=l;i++)
{var j=l-i;p[j]=new EbayPassword(ebayDoc,ebayDoc,ebayDoc.htmlDoc,phones[i],action,true);p[j].bindHTML();p[j].bindEvents();p[l].inputs[j]=p[j];if(j!=l)
p[j].master=p[l];}
control=p[l];control.bindHTML();control.bindEvents();control.onchange();}}
return control;}
function SetUpBirthday()
{if((psvMonElem!=""&&ebayDoc.getFormElem(psvMonElem))&&(psvDayElem!=""&&ebayDoc.getFormElem(psvDayElem))&&(psvYearElem!=""&&ebayDoc.getFormElem(psvYearElem)))
{birthday=new EbayPassword(ebayDoc,ebayDoc,ebayDoc.htmlDoc,psvDayElem,UpdateBirthday,true);birthmonth=new EbayPassword(ebayDoc,ebayDoc,ebayDoc.htmlDoc,psvMonElem,UpdateBirthday,true);birthyear=new EbayPassword(ebayDoc,ebayDoc,ebayDoc.htmlDoc,psvYearElem,UpdateBirthday,true);birthday.bindHTML();birthday.bindEvents();birthmonth.bindHTML();birthmonth.bindEvents();birthyear.bindHTML();birthyear.bindEvents();birthday.onchange();}}
function updateMeter(pSecLevel){var img=document.images[psvMeterImage];if(typeof(img)!="undefined")
img.src=security_level[pSecLevel].src;}
function UpdateSimple()
{var c=this.ebayControl||this;var v=c.htmlElement.value;c.checkStrings=new Array;if(v!="")
{c.checkStrings[0]=v;c.checkStrings[1]=v.replace(/\s+/g,"");}
if(window.password!=null&&c.updatePassword)
password.onchange();}
function UpdateName()
{var c=this.ebayControl||this;var v=c.htmlElement.value;c.checkStrings=new Array;if(v!="")
{c.checkStrings[0]=v;c.checkStrings[1]=v.replace(/\s+/g,"");var a=v.split(/[\s-\.\,]+/);for(var i=0,len=a.length;i<len;i++)
{var s=a[i];if(s.length>=4)c.checkStrings[c.checkStrings.length]=s;for(var j=i-1;j>=0;j--)
{s+=a[j];if(s.length>=4)c.checkStrings[c.checkStrings.length]=s;}}}
if(window.password!=null&&c.updatePassword)
password.onchange();}
function UpdateNumeric()
{var c=this.ebayControl||this;var v=c.htmlElement.value;if(v=="")
c.checkStrings=new Array;else
{v=v.replace(/[^\d]/g,"");c.checkStrings[0]=v;}
if(window.password!=null&&c.updatePassword)
password.onchange();}
function UpdateUserId()
{var c=this.ebayControl||this;var v=c.htmlElement.value;if(v=="")
c.checkStrings=new Array;else
{c.checkStrings[0]=v;c.checkStrings[1]=Reverse(v);if(v.match(/\s+/))
{v=v.replace(/\s+/g,"");c.checkStrings[2]=v;c.checkStrings[3]=Reverse(v);}}
if(window.password!=null&&c.updatePassword)
password.onchange();}
function UpdateEmail()
{var c=this.ebayControl||this;var v=c.htmlElement.value;if(v=="")
c.checkStrings=new Array;else
{v=v.replace(/@.*$/g,"");c.checkStrings[0]=v;c.checkStrings[1]=Reverse(v);}
if(window.password!=null&&c.updatePassword)
password.onchange();}
function UpdateBirthday()
{var c=birthday;var m=birthmonth.htmlElement.value;if(m.match(/^\d$/))m="0"+m;var d=birthday.htmlElement.value;if(d.match(/^\d$/))d="0"+d;var y=birthyear.htmlElement.value;y=y.replace(/[^\d]/g,"");if(m.match(/\d\d/)&&d.match(/\d\d/))
c.checkStrings=new Array(m+d,d+m);else
c.checkStrings=new Array();if(y.match(/\d\d\d\d/))
c.checkStrings[c.checkStrings.length]=y;if(window.password!=null&&c.updatePassword)
password.onchange();}
function UpdatePhone()
{var c=null;if(this.ebayControl)
c=this.ebayControl.master||this.ebayControl;else
c=this.master||this;var a=new Array();c.checkStrings=new Array();for(var i=0,len=c.inputs.length;i<len;i++)
{var v=c.inputs[i].htmlElement.value;if(v!="")
a=a.concat(v.split(/[^\d]+/).reverse());}
for(var i=0,len=a.length;i<len;i++)
{var s=a[i];if(s.length>=4)c.checkStrings[c.checkStrings.length]=s;for(var j=i-1;j>=0;j--)
{s+=a[j];if(s.length>=4)c.checkStrings[c.checkStrings.length]=s;}}
if(window.password!=null&&c.updatePassword)
password.onchange();}
function UpdateAddr()
{var c=this.ebayControl||this;var v=c.htmlElement.value;c.checkStrings=new Array();if(v!="")
{c.checkStrings[c.checkStrings.length]=v;v=v.replace(/[\.,]/g,"");c.checkStrings[c.checkStrings.length]=v;c.checkStrings[c.checkStrings.length]=v.replace(/\s*\b\d+[a-z]?\b\s*/i,"");var num=v.replace(/[^\d]*/g,"");if(num.length>=4)c.checkStrings[c.checkStrings.length]=num;var a=v.split(/\s+/);var l=a.length;for(var i=0;i<l;i++)
{var s=a[i];if(s.length>=4)c.checkStrings[c.checkStrings.length]=s;for(var j=i+1;j<l;j++)
{s+=a[j];if(s.length>=4)c.checkStrings[c.checkStrings.length]=s;}}}
if(window.password!=null&&c.updatePassword)
password.onchange();}
function Reverse(s){var t="";for(var i=s.length-1;i>=0;i--)
t+=s.charAt(i);return t;}
function indexOfNoCase(s1,s2)
{return s1.toUpperCase().indexOf(s2.toUpperCase());}
function AllConsecutiveOrSame(s)
{var safe=s.replace(/([\*\\\/\.\?\{\}\:\$\^\[\]\(\)\,\+])/g,"\\$1");var re=new RegExp(safe,"i");var letters="abcdefghijklmnopqrstuvwxyz";var numbers="01234567890";if(letters.match(re)||numbers.match(re))
return true;var first=s.charAt(0);for(var i=1,len=s.length;i<len;i++)
if(s.charAt(i)!=first)return false;return true;}
function UpdatePassword()
{var c=this.ebayControl||this;var v=c.htmlElement.value;var seclevel=0;if(v!="")
{var invalidStrings=new Array("ebay","yabe");invalidStrings=invalidStrings.concat(email.checkStrings,userid.checkStrings);for(var i=0,len=invalidStrings.length;i<len;i++)
{if(indexOfNoCase(v,invalidStrings[i])!=-1)
{seclevel=INVALID;break;}}
if(seclevel!=INVALID)
{if(v.length<6)
seclevel=0;else
{seclevel=1;if(!AllConsecutiveOrSame(v))
{if(v.match(/[A-Z].*[A-Z]/i)&&v.match(/[^a-zA-Z\s].*[^a-zA-Z\s]/))
seclevel++;if(v.length>=8)
seclevel++;var infoStrings=new Array();if(ext1)
infoStrings=infoStrings.concat(ext1.checkStrings);if(ext2)
infoStrings=infoStrings.concat(ext2.checkStrings);if(birthday)
infoStrings=infoStrings.concat(birthday.checkStrings);if(phone1)
infoStrings=infoStrings.concat(phone1.checkStrings);if(phone2)
infoStrings=infoStrings.concat(phone2.checkStrings);if(firstname)
infoStrings=infoStrings.concat(firstname.checkStrings);if(lastname)
infoStrings=infoStrings.concat(lastname.checkStrings);if(addr)
infoStrings=infoStrings.concat(addr.checkStrings);if(city)
infoStrings=infoStrings.concat(city.checkStrings);if(zip)
infoStrings=infoStrings.concat(zip.checkStrings);for(var i=0,len=infoStrings.length;i<len;i++)
{if(indexOfNoCase(v,infoStrings[i])!=-1)
{seclevel--;break;}}}}}}
if(c.seclevel!=seclevel)
updateMeter(seclevel);c.seclevel=seclevel;}
function defaultRadio()
{if(ebay)
{var oD=ebay.oDocument;var oPc=oD.oPage.oConfig;var hUser=oD.getFormElem("user");var rdo=oD.getFormElem(oPc.sUserIdRadioElemName,"radio");if(hUser&&rdo)
{rdo[0].checked=true;if(hUser)
hUser.value=rdo[0].value;if(userid)
userid.onchange();}}}

//16@@m30

function EbayDocument(pWin,pName)
{this.htmlWin=pWin||null;this.htmlDoc=pWin?pWin.document:null;this.name=pName||null;this.status=null;this.controls=new Array;this.events=new Array;this.bindHTML=EbayDocumentBindHTML;this.bindEvents=EbayDocumentBindEvents;this.addControl=EbayAddControl;this.getFormElem=EbayGetFormElem;this.onLoad=EbayDocumentOnLoad;this.onUnload=EbayDocumentOnUnLoad;}
window.EbayDocument=EbayDocument;function EbayGetFormElem(pName,pType)
{if(!this.htmlDoc)
return null;var frms=this.htmlDoc.forms;var ln=frms.length;for(var i=0;i<ln;i++)
{var elems=frms[i].elements,len=elems.length;for(var j=0;j<len;j++)
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
return;var doc=this.htmlDoc,len=doc.forms.length;for(var j=0;j<len;j++)
{var frms=doc.forms;var iElems=frms[j].elements.length;frms[j].onsubmit=EbayDisableOnSubmit;for(var i=0;i<iElems;i++)
{var elem=frms[j].elements[i];if(elem.id!="")
document.getElementById(elem.id);var skEs=this.skipElems;var skip=false,len2=skEs.length;if(len2>0)
{for(var k=0;k<len2;k++)
{if(elem.name==skEs[k])
{skip=true;break;}}}
if(!skip)
EbayDisableFormElement(elem);}}
var iLinks=doc.links.length;for(var i=0;i<iLinks;i++)
{var lnk=doc.links[i];var skLnks=this.skipLinks;var skip=false;if(lnk.href)
{var len3=skLnks.length;for(var k=0;k<len3;++k)
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
{var parent=pParent?pParent:this.ebayControl,len=parent.listeners.length;for(var i=0;i<len;i++)
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
{var cctrl,ctrls=pThis.controls,len=ctrls.length;for(var i=0;i<len;i++)
{cctrl=ctrls[i];if(pIsEvents)
{if(cctrl&&cctrl.bindEvents&&cctrl.objType!="EbayHyperLink")
cctrl.bindEvents();}
else
{if(cctrl&&cctrl.bindHTML)
cctrl.bindHTML();}}}
function EbayAddControl(pControl)
{var isSet=false;if(pControl.htmlElementName)
{var ctrls=this.controls,len=ctrls.length;for(var i=0;i<len;i++)
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
lnk=pDoc.getElementById(pLinkName);if(lnk)return lnk;var len=pDoc.links.length;for(var j=0;j<len;j++)
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
{var len=this.htmlElement.length;if(len&&len>0)
{for(var i=0;i<len;i++)
{if(this.htmlElement[i])
this.htmlElement[i].ebayControl=this;}}
else
this.htmlElement.ebayControl=this;}}
function EbayBindEventsRadio()
{if(this.htmlElement&&this.onClick)
{var len=this.htmlElement.length;if(len&&len>0)
{for(var i=0;i<len;i++)
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

//17@@m29

var bPopUpWindow=true,winPopped=false;var platform=navigator.platform.toLowerCase();var surveyCookie="surveySays";var surveyValue="shown";var surveyTempPopupFile="popout/temp_survey_popup.html";function eBayIsIE4()
{var agent=navigator.userAgent.toLowerCase();var is_ie=(agent.indexOf("msie")!=-1);var minor=is_ie?parseFloat(agent.substring(agent.indexOf('msie')+5)):parseFloat(navigator.appVersion);var isie4=parseInt(minor)==4;return isie4;}
function setPopOutSwitch(bOnOrOff,myfunction)
{bPopUpWindow=bOnOrOff;}
function popupWindow(mypage,target,w,h,scroll,no,pLeft,pTop,params){if(target=="_self")
return false;LeftPosition=(screen.width)?(screen.width-w)/2:0;TopPosition=(screen.height)?(screen.height-h)/2:0;if(pLeft)
LeftPosition=pLeft;if(pTop)
TopPosition=pTop;settings='height='+h+',width='+w+',top='+TopPosition+',left='+LeftPosition+',scrollbars='+scroll+',resizable='+no+'';top.win=window.open(mypage,target,settings+","+params);winPopped=true;return false;}
function checkHome(){var valueOfCheckHome=false;var agt=navigator.userAgent.toLowerCase();var myopera=(agt.indexOf("opera")!=-1);if(eBayIsIE4())
{return false;}
if(!myopera&&platform.indexOf('mac')==-1&&typeof(oHomePage)!='undefined')
{valueOfCheckHome=eval('oHomePage.isHomePage("http://"+document.location.host+"/")');}
else
{valueOfCheckHome=false;}
return valueOfCheckHome;}
function findForms()
{var formsArray=document.forms,len=formsArray.length;var i;var strFormSubmit="";for(i=0;i<len;i++)
{strFormSubmit+='<SC'+'RIPT FOR=\"'+formsArray[i].name+'\" EVENT=\"onsubmit\" LANGUAGE=\"JavaScript\">';strFormSubmit+='setPopOutSwitch(false)\;';strFormSubmit+='<\/SC'+'RIPT>';}
document.write(strFormSubmit);}
function startCounterCookie(){var now="";var countscookie="";var countcookie="";now=new Date();Year=now.getUTCFullYear();Month=now.getMonth();Day=now.getDate();now=new Date(Year,Month,Day);now.setHours(now.getHours()+24);countcookie=readCookieletEx("dp1","count");if(!countcookie){countcookie=1;}else{countcookie=parseInt(countcookie)+1;}
writeCookieletEx("dp1","count",countcookie,"","/",now);testIt();}
function popOutWindow(pURL,width,height,pSurvey,pSampleRate)
{if(typeof(width)=='undefined')
width='275';if(typeof(height)=='undefined')
width='375';var valueOfCounts=0;var valueOfCheckHome=false;var valueOfKeepMeSignIn=null;var valueOfEbaySignIn=null;if(!pSurvey)
{valueOfKeepMeSignIn=readCookieEx("keepmesignin");valueOfEbaySignIn=readCookieEx("ebaysignin");valueOfCheckHome=checkHome();testIt();valueOfCounts=readCookieletEx("dp1","count");}
testIt();if(bPopUpWindow)
{if((valueOfCheckHome!=true)&&(valueOfCounts<2))
{if((valueOfEbaySignIn!="in")&(valueOfKeepMeSignIn!="in"))
{var notMac=((navigator.userAgent.toLowerCase()).indexOf("mac")==-1);if(pSurvey&&!winPopped&&notMac&&(readCookieEx(surveyCookie)!=surveyValue))
return showSurveyWindow(pURL,width,height,pSampleRate);else if(!pSurvey)
return popupWindow(pURL,'SureYouWantToLeave',width,height,'no','yes');}}}}
function showExitPollSurveyWindow(pURL,pWidth,pHeight)
{var chn=document.location.hostname;var host="http://pages"+chn.substr(chn.indexOf("."));host+="/";var sInd=host.indexOf(".stores.");if(sInd!=-1)
host=host.substring(0,sInd+1)+host.substring(sInd+8,host.length);pURL=host+surveyTempPopupFile+"?"+pURL;pURL+="&ep=&ts="+new String(new Date())+"&width=600&height=400";var pWin=popupWindow(pURL,'ebSurvey',600,400,'yes','yes','5000','5000');return pWin;}
function showSurveyWindow(pURL,pWidth,pHeight,pSampleRate)
{var rnd=Math.floor(Math.random()*10000);if((rnd%3)!=0)
return false;var reg=readCookieEx("reg");var etfc=readCookieEx("etfc");var uType="A";if(reg=="1")
{if(etfc=="1")
uType="C";else if(etfc=="2")
uType="D";else
uType="B";}
pURL+=uType;var chn=document.location.hostname;var host="http://pages"+chn.substr(chn.indexOf("."));host+="/";var sInd=host.indexOf(".stores.");if(sInd!=-1)
host=host.substring(0,sInd+1)+host.substring(sInd+8,host.length);pURL=host+surveyTempPopupFile+"?"+pURL;pURL+="&ts="+new String(new Date())+"&width=600&height=400";var pWin=popupWindow(pURL,'ebSurvey',600,400,'yes','yes','5000','5000');return pWin;}
function writeClickListener()
{var agt=navigator.userAgent.toLowerCase();var mymac=(agt.indexOf("mac")!=-1);var myopera=(agt.indexOf("opera")!=-1);if(platform.indexOf('mac')==-1&&!myopera)
{var string='';string+='<SCR'+'IPT FOR="document" EVENT="onmousedown()" LANGUAGE="JavaScript">';string+='var eventTag = window.event.srcElement.tagName.toLowerCase();';string+='if(eventTag == "a" || eventTag == "img" || eventTag == "area" || eventTag == "input")';string+='{';string+='setPopOutSwitch(false)\;';string+='}';string+='<\/SC'+'RIPT>';string+='<SCR'+'IPT FOR="document" EVENT="onkeypress()" LANGUAGE="JavaScript">';string+='var eventTag = window.event.srcElement.tagName.toLowerCase();';string+='if(eventTag == "input")';string+='{';string+='setPopOutSwitch(false)\;';string+='}';string+='<\/SC'+'RIPT>';document.write(string);}}
function eBayIsFramed()
{if(top==self)
{return false;}
else
{return true;}}
function eBayToolbarPluginDetect()
{var plugin=false;var mimetype="application/x-eBay-Toolbar-Plugin-Functions";var nsname="eBay Plugin Functions";if(typeof(navigator.mimeTypes)!='undefined'&&navigator.mimeTypes[mimetype]&&navigator.mimeTypes[mimetype].enabledPlugin)
{if(navigator.plugins)
plugin=true;}
else if(navigator.userAgent&&navigator.userAgent.indexOf("MSIE")>=0&&(navigator.userAgent.indexOf("Windows")>=0)){if(eBayIsIE4())
{return false;}
else if(typeof(IECheckClient)=="unknown")
{plugin=IECheckClient();}}
return plugin;}
if(eBayToolbarPluginDetect()){setPopOutSwitch(false,"toolbar");}
if(eBayIsFramed()){setPopOutSwitch(false,"framed");}
function testIt()
{}
var popWinLoaded=true;

//18@@m9

function ebHelpContextualRebrand(pGuide,pFeature,pNoEscape)
{var hPath="/help/",hIndexPath=hPath+"index.html",hInd;var dl=document.links,url,pre,post,oTxt;for(var i=dl.length-1;i>-1;i--)
{url=dl[i].href;hInd=url.indexOf(hIndexPath);if(hInd!=-1)
{pre=url.substring(0,url.lastIndexOf("/")+1);post=url.substring(url.lastIndexOf("/")+1);dl[i].href=url=pre+pGuide+"/"+post;}
if(pFeature&&url.contains(hPath))
{hInd=url.indexOf(".html")+5;var preUrl=url.substring(0,hInd),postUrl=url.substr(hInd);post="fromFeature=";if(pNoEscape)
post+=pFeature;else
post+=encodeURIComponent(pFeature);post="?"+post;if(postUrl.charAt(0)=='?')
postUrl="&"+postUrl.substr(1);oTxt=dl[i].innerHTML;dl[i].href=preUrl+post+postUrl;if(oTxt)
dl[i].innerHTML=oTxt;}}}

//19@@m8

function EBayClient()
{var agt=navigator.userAgent.toLowerCase();this.major=parseInt(navigator.appVersion);this.webTV=this.opera=this.nav=this.ie=this.safari=false;var vInd=0;if(agt.indexOf("webtv")!=-1)
{this.webTV=true;vInd=agt.indexOf("webtv/")+6;}
else if(agt.indexOf("safari")!=-1)
{this.safari=true;vInd=agt.lastIndexOf("safari")+7;}
else if(agt.indexOf("opera")!=-1)
{this.opera=true;vInd=agt.lastIndexOf("opera")+6;}
else if(agt.indexOf("firefox")!=-1)
{this.firefox=true;vInd=agt.indexOf("firefox")+7;}
else if(navigator.appName.toLowerCase()=="netscape")
{this.nav=true;vInd=agt.lastIndexOf("/")+1;var tmp=parseInt(agt.substring(vInd));if(isNaN(tmp))
vInd=agt.lastIndexOf("netscape")+9;}
else if(agt.indexOf("msie")!=-1)
{this.ie=true;vInd=agt.indexOf("msie")+4;}
this.version=parseInt(agt.substring(vInd));this.win=(agt.indexOf("win")!=-1);this.mac=(agt.indexOf("mac")!=-1);this.macppc=(this.mac&&((agt.indexOf("ppc")!=-1)||(agt.indexOf("powerpc")!=-1)));this.isAXLoaded=EbayDetectActX;this.isAXSupported=EbaySupportActX;this.msnTv=(agt.indexOf("msntv")!=-1);this.xpSp2=(agt.indexOf("sv1")!=-1&&!this.msnTv);this.xp=(this.win&&agt.indexOf("windows nt 5.1")!=-1&&!this.msnTv);}
if(typeof(client)=="undefined")
var client=new EBayClient();if(typeof(is)=="undefined")
var is=new EBayClient();function EbaySupportActX()
{return this.isAXLoaded("Scripting.Dictionary");}
function EbayDetectActX(pActXName)
{if(!client.ie)
return false;var h='<scr'+'ipt language="JavaScript" type="text/JavaScript">';h+='var aX;';h+='</scr'+'ipt>';document.writeln(h+'<scr'+'ipt language="vbscript" type="text/vbscript">');document.writeln('on error resume next');document.writeln('aX = IsObject(CreateObject("'+pActXName+'"))');document.writeln('</scr'+'ipt>');if(typeof(aX)=="undefined"||!aX)
return false;else
return true;}

//20@@m4

function EbayPopupWindow(pEbayDoc,pParent,pName)
{if(!this.objType)
this.objType="EbayPopupWindow";this.baseObject=EbayControl;this.baseObject(pEbayDoc,pParent,pEbayDoc.htmlDoc,pName);this.url="";this.width=null;this.height=null;this.left=null;this.top=null;this.toolbar=true;this.location=true;this.status=true;this.scrollbars=true;this.resizable=true;this.menubar=true;this.props="";this.customProps="";this.winObj=null;this.show=ebayShowPopup;this.showEx=ebayShowPopupEx;this.resizeParent=EbayPopupResizeParent;}
window.EbayPopupWindow=EbayPopupWindow;function ebayShowPopup()
{if(this.url.length==0)
return null;var sP="";if(this.width!=null)sP+=",width="+this.width;if(this.height!=null)sP+=",height="+this.height;if(this.left!=null)sP+=",screenX="+this.left+",left="+this.left;if(this.top!=null)sP+=",screenY="+this.top+",top="+this.top;sP+=",toolbar="+((this.toolbar)?"1":"0");sP+=",location="+((this.location)?"1":"0");sP+=",status="+((this.status)?"1":"0");sP+=",scrollbars="+((this.scrollbars)?"1":"0");sP+=",resizable="+((this.resizable)?"1":"0");sP+=",menubar="+((this.menubar)?"1":"0");if(this.customProps.length>0)
sP+=","+this.customProps;if(sP.length>0)sP=sP.substring(1);this.props=sP;var w=window.open(this.url,this.name,sP);if(w)
w.focus();this.winObj=w;return w;}
function ebayShowPopupEx(width,height,left,top,toolbar,location,status,scrollbars,resizable,menubar,customprops,url)
{if(url)
this.url=url;this.width=width;this.height=height;this.left=left;this.top=top;this.toolbar=toolbar;this.location=location;this.status=status;this.scrollbars=scrollbars;this.resizable=resizable;this.menubar=menubar;if(customprops)
this.customProps=customProps;return this.show();}
function EbayPopupResizeParent(pX,pY,pW,pH)
{var p=this.parent;if(p)
{if(!isNaN(pX)&&!isNaN(pY))
p.moveTo(pX,pY);if(!isNaN(pW)&&!isNaN(pH))
p.resizeTo(pW,pH);}}

//21@@m5

function toolboxOnLoad()
{var u="undefined";framesCheck();if(typeof(ITRegisterEnterInfoOnLoad)!=u)
{ITRegisterEnterInfoOnLoad();if(typeof(ebayDoc)!=u&&ebayDoc!=null)
{ebayDoc.onLoad();ebayDoc=null;}}
if(typeof(initFocus)!=u)
initFocus();if(typeof(cnRegisterSellerVerificationOnload)!=u)
cnRegisterSellerVerificationOnload();if(typeof(psvOnLoad)!=u)
psvOnLoad();if(typeof(ebaySurveyMangerOnLoad)!=u)
ebaySurveyMangerOnLoad();if(typeof(ebayDoc)!=u&&ebayDoc!=null)
ebayDoc.onLoad();}
function toolboxOnUnload()
{if(typeof(ebaySurveyManagerOnUnload)!="undefined")
ebaySurveyManagerOnUnload();}
function framesCheck()
{var u="undefined";var p=pageName;if(typeof(p)==u)
return false;if(p.toLowerCase()!="pageregistersuccess")
return false;else
window.bustFrames();return true;}

//22@@m5

function Pop(path,ficenter,price,isMotorcycle)
{var popWinHt="";if((isMotorcycle!="")&&(ficenter!=""))
{path=path+"?isMotorcycle="+isMotorcycle+"&ficenter=1";popWinHt="400";}
if((isMotorcycle!="")&&(ficenter==""))
{path=path+"?isMotorcycle="+isMotorcycle+"&ficenter=";popWinHt="400";}
if((isMotorcycle=="")&&(ficenter!=""))
{path=path+"?isMotorcycle=&ficenter=1";popWinHt="400";}
if((isMotorcycle=="")&&(ficenter==""))
{path=path+"?isMotorcycle=&ficenter=";popWinHt="290";}
if(price!=""){path=path+"&price="+price;}else{path=path+"&price=0";}
hw=window.open(path,"MonthlyPaymentCalc","height="+popWinHt+",width=375,status=no,toolbar=no,menubar=no,location=no,resizable=yes,scrollbars=no,titlebar=no");hw.focus();return false;}
function roundAmount(n){var s=""+Math.round(n*100)/100
var i=s.indexOf('.')
if(i<0)return s+".00"
var t=s.substring(0,i+1)+s.substring(i+1,i+3)
if(i+2==s.length)t+="0"
return t;}
function stripComma(target){if(null!=target)
{splitstring=target.split(",");if(null!=splitstring)
{target=splitstring[0];for(j=1,len=splitstring.length;j<len;j++)
{target+=splitstring[j];}}}
return target;}
function stripNonNumeric(pPrice)
{var tmp="",chr;for(var i=0,len=pPrice.length;i<len;i++)
{chr=pPrice.charAt(i);if((chr=='.')||!isNaN(parseInt(chr)))
tmp+=chr;}
return tmp;}
function GetYears(months){return months/12;}
function MonthlyPaymentCalc(){price=document.MonthlyPayment.price.value;price=stripComma(price);APR=stripNonNumeric(document.MonthlyPayment.APR.value);payment=document.MonthlyPayment.payment.value;payment=stripComma(payment);for(var i=0,len=document.MonthlyPayment.months.length;i<len;i++){if(document.MonthlyPayment.months[i].selected==true){months=document.MonthlyPayment.months[i].value;}}
APR=APR/100;temp1=(1+(APR/12));temp2=GetYears(months);temp3=Math.pow((temp1),-(temp2*12));if(price.indexOf(".")!=-1){_alert("Decimals should not be entered in the total price box");return false;}else if(payment.indexOf(".")!=-1){_alert("Decimals should not be entered in the down payment box");return false;}else{RemainingPayment=price-payment;if(RemainingPayment>0){EstMonthlyPayment=(RemainingPayment*APR)/(12*(1-temp3));EstMonthlyPayment=roundAmount(EstMonthlyPayment);if(isNaN(EstMonthlyPayment))
EstMonthlyPayment="0.00";}else{EstMonthlyPayment="0.00";}
document.MonthlyPayment.EstMonthlyPayment.value=EstMonthlyPayment;return false;}}
function _alert(str)
{if(typeof(ebay)!="undefined")
ebay.oDocument.messageBox(str);else
alert(str);}

//23@@m5

function ebDowngradeDomainTo()
{var dd=document.domain,i=dd.indexOf(".ebay."),qs;if(i!=-1)
{dd=dd.substr(i+1);qs=decodeURI(document.location.search);if((i=qs.indexOf("downgradeDomainTo="))>-1)
dd=qs.substring(i+18,qs.indexOf(dd)+dd.length);if(document.domain!=dd||!document.all)
document.domain=new String(dd);}}
ebDowngradeDomainTo();

//24@@m3

function EbayCookieEncoder(pParent)
{if(!this.objType)
this.objType="EbayCookieEncoder";this.baseObject=EbayBaseControl;this.baseObject(pParent,this.objType);this.init=function()
{this.generateClientId();}
this.generateClientId=function()
{var cJ=ebay.oDocument.oCookieJar,cId=cJ.readCookie("cid");if(cId)
{if(cId.length==8)
{cId+=this.generateRandomId();cId+="#"+this.generateHash(cId);cJ.writeCookie("cid",cId,"","",cJ.getDate(1));}}}
this.generateRandomId=function()
{var cV='',ch="0123456789ABCDEFGHIJKLMNOPQRSTUVWXTZabcdefghijklmnopqrstuvwxyz",len=8,i,inx;for(i=0;i<len;i++)
{inx=Math.floor(Math.random()*ch.length);cV+=ch.substring(inx,inx+1);}
return cV;}
this.generateHash=function(str)
{var i=0,h=0;for(var i=0,len=str.length;i<len;i++)
{h=str.charCodeAt(i)+(h<<6)+(h<<16)-h;}
h=(h&0x7FFFFFFF);return h;}
this.init();}
new EbayCookieEncoder(ebay.oDocument.oPage);

//25@@m4

function EbaySurveyManager(pEbayDoc,pParent,pHTMLDoc,pElementName,pSurveyManagerConfig,pSurveyDataConfig)
{if(!this.objType)
this.objType="EbaySurveyManager";this.baseObject=EbayControl;this.baseObject(pEbayDoc,pParent,pHTMLDoc,pElementName);this.config=pSurveyManagerConfig;this.data=pSurveyDataConfig||new Array(0);this.bindHTML=EbayBindSurvey;this.bindEvents=EbayBindEventsSurvey;this.onLoad=EbaySurveyManagerOnLoad;this.onUnload=EbaySurveyManagerOnUnload;this.popup=null;this.showPopup=true;this.determine=EbayDetermineSurvey;this.partialExceptionUrls=new Array(".sandbox.");this.partialReferrerUrls=new Array("google");this.pageSpecific=true;this.tokenValues=new Array();this.popupHtml="temp_survey_popup.html";this.isShown=EbayIsShownSurvey;this.replaceSurveyTokens=EbayReplaceTokensSurvey;this.isExceptionalUrl=EbayIsExceptionalUrlSurvey;this.isFromPartnerUrl=EbayIsFromPartnerUrlSurvey;this.isInSample=EbayIsInSampleSurvey;this.addSurveyDataConfig=EbayAddSurveyDataConfig;this.addSurveyDataConfigEx=EbayAddSurveyDataConfigEx;}
window.EbaySurveyManager=EbaySurveyManager;function EbayDetermineSurvey()
{var hiddenHtml;var temp;var configToShow=null;var today=new Date();temp=this.config.disable;if(temp)
{this.showPopup=false;return false;}
if(this.isFromPartnerUrl())
{this.showPopup=false;return false;}
if(this.isShown())
{this.showPopup=false;return false;}
temp=this.config.hiddenPopupFolderUrl;if(temp.length>0)
{if(this.isExceptionalUrl(temp))
{this.showPopup=false;return false;}}
hiddenHtml=temp+this.popupHtml;var t=this.data.length;if(this.pageSpecific)
{for(var i=0;i<t;i++)
{configToShow=this.data[i];temp=configToShow.pageName.toLowerCase();if(temp==pageName.toLowerCase())
break;else
configToShow=null;}}else
{if(t>0)
configToShow=this.data[0];}
if(!configToShow)
{this.showPopup=false;return false;}
temp=parseInt(configToShow.samplingRate,10);if(temp>0)
{if(!this.isInSample(temp))
{this.showPopup=false;return false;}}
temp=configToShow.surveyType;if(temp.length>0)
{if(temp!="POPUP")
{this.showPopup=false;return false;}}
temp=configToShow.startDate;if(temp.length>0)
{if(temp-today>0)
{this.showPopup=false;return false;}}
temp=configToShow.expDate;if(temp.length>0)
{if(temp-today<0)
{this.showPopup=false;return false;}}
this.popup=new EbayPopupWindow(this.ebayDoc,this.ebayDoc,"popup");with(this.popup)
{left=5000;top=5000;temp=configToShow.width;width=((temp)?temp:600);temp=configToShow.height;height=((temp)?temp:480);temp=configToShow.toolbar;toolbar=((temp)?temp:false);temp=configToShow.location;location=((temp)?temp:false);temp=configToShow.status;status=((temp)?temp:false);temp=configToShow.scrollbars;scrollbars=((temp)?temp:true);temp=configToShow.resizable;resizable=((temp)?temp:true);temp=configToShow.menubar;menubar=((temp)?temp:false);url=hiddenHtml+"?"+configToShow.url;this.replaceSurveyTokens();}
return true;}
function EbayBindSurvey()
{}
function EbayBindEventsSurvey()
{}
function EbaySurveyManagerOnLoad()
{this.determine();}
function EbaySurveyManagerOnUnload()
{if(this.showPopup)
{with(this.popup)
{url=url+"&ep=&ts="+new String(new Date())+"&width="+width+"&height="+height;show();}}}
function EbayIsInSampleSurvey(pSamplingRate)
{var max=pSamplingRate;var x=Math.round(Math.random()*(max-1));var y=x==(max-1);return y;}
function EbayIsExceptionalUrlSurvey(pUrl)
{var u="undefined";if(typeof(this.partialExceptionUrls.length)==u)
return false;for(var i=0,len=this.partialExceptionUrls.length;i<len;i++)
{if(pUrl.toLowerCase().indexOf(this.partialExceptionUrls[i].toLowerCase())!=-1)
return true;}
return false;}
function EbayIsFromPartnerUrlSurvey()
{var ref=window.document.referrer;var u="undefined";if(typeof(this.partialReferrerUrls.length)==u)
return false;for(var i=0,len=this.partialReferrerUrls.length;i<len;i++)
{if(ref.toLowerCase().indexOf(this.partialReferrerUrls[i].toLowerCase())!=-1)
return true;}
return false;}
function EbayReplaceTokensSurvey()
{var f="this.popup.url.toString().replaceTokens(";var c=this.tokenValues.length;for(var j=0;j<c;j++)
{f+="'"+this.tokenValues[j]+"'"
if(j!=c-1)
f+=","}
f+=");"
this.popup.url=eval(f);}
function EbayIsShownSurvey()
{var surveyShown=readCookieletEx("dp1","xSrvy");if(surveyShown=="y")
return true
return false;}
function EbayAddSurveyDataConfig(pName,pUrl,pPageName)
{this.data[this.data.length]=eBaySurveyDataConfig(pName,pUrl,pPageName);}
function EbayAddSurveyDataConfigEx(pName,pUrl,pPageName,pSamplingRate,pSurveyType,pStartDate,pExpDate)
{this.data[this.data.length]=eBaySurveyDataConfigEx(pName,pUrl,pPageName,pSamplingRate,pSurveyType,pStartDate,pExpDate);}

//26@@m7

var ebayDoc;var surveyManager;function ebaySurveyMangerOnLoad()
{var u="undefined";if(typeof(surveyConfig)==u)
return false;if(surveyConfig.disable)
return false;if(typeof(ebayDoc)==u||ebayDoc==null)
ebayDoc=new EbayDocument(window,"SurveyMangerDoc");var c=new EBayClient();if(c.mac||c.nav||c.xpSp2)
return false;surveyManager=new EbaySurveyManager(ebayDoc,ebayDoc,document,"surveyManager",surveyConfig,gSurveys);surveyManager.tokenValues=new Array(surveyConfig.entryPoint,surveyConfig.partnerID);surveyManager.onLoad();}
function ebaySurveyManagerOnUnload()
{var u="undefined";if(typeof(surveyConfig)==u)
return false;if(surveyConfig.disable)
return false;if(surveyManager)
surveyManager.onUnload();}

//27@@m27

ebay.oDocument.oPage.onBeforeLoad=function()
{var oCfg=this.parent.getConfig("uservalidation");if(oCfg)
new EbayFieldValidation(ebay.oDocument.oPage,"uservalidation",oCfg);var pCfg=this.parent.getConfig("eBayPasswordPolicy");if(pCfg)
new EbayPasswordopt(ebay.oDocument.oPage,"eBayPasswordPolicy",pCfg);var oConfig=this.parent.getConfig('Registration.ErrorListings');if(oConfig)
new EbayQuickWins(ebay.oDocument.oPage,"Registration.ErrorListings",oConfig);if(typeof(toolboxOnLoad)!="undefined")
toolboxOnLoad();var oD=ebay.oDocument,oP=oD.oPage,oPc=oP.oConfig,elChangeSite=null,cfg;cfg=oD.getConfig("retypeEmailAddress");var oRetypeEmail=new EbayHTMLText(oP,cfg.sRetypeEmailAddressID,false,cfg);oRetypeEmail.iKeyups=0;oRetypeEmail.bFieldRecentlyEmpty=true;var evt=oRetypeEmail._registerEvent("onfocus","checkFieldLength");oRetypeEmail.checkFieldLength=function(){if(this.getValue().length>0){this.bFieldRecentlyEmpty=false;}else{this.bFieldRecentlyEmpty=true;}}
var evt=oRetypeEmail._registerEvent("onkeyup","onUserTypeInEmailField");oRetypeEmail.onUserTypeInEmailField=function(){oRetypeEmail.iKeyups++;if(this.getValue().length==0)
{this.bFieldRecentlyEmpty=true;oRetypeEmail.iKeyups=0;}}
var evt=oRetypeEmail._registerEvent("onblur","onUserFinishRetypeEmailField");oRetypeEmail.onUserFinishRetypeEmailField=function(){if(oRetypeEmail.iKeyups<4&&this.bFieldRecentlyEmpty){oRetypeEmail.setValue("");}
oRetypeEmail.iKeyups=0;}
var eFrm=new EbayHTMLForm(oP,"RegisterEnterInfo");eFrm.setAction=oPc.sTargetUrl;if(oPc.sContinueElemName&&oPc.sContinueElemName.length>0)
{var eButton=new EbayHTMLButton(eFrm,oPc.sContinueElemName);var oAcceptErrorDiv=new EbayHTMLLayer(eFrm,oPc.sDivname);oAcceptErrorDiv.bind();if(oPc.sRegisterError)
{var oRegErrorDiv=new EbayHTMLLayer(eFrm,oPc.sRegisterError);oRegErrorDiv.bind();}
var oAcceptCB=new EbayHTMLCheckbox(eFrm,oPc.sCheckId);oAcceptCB.bind();var oAcceptCB1=new EbayHTMLCheckbox(eFrm,oPc.sCheckId1);oAcceptCB1.bind();var oAcceptCB2=new EbayHTMLCheckbox(eFrm,oPc.sCheckId2);oAcceptCB2.bind();oAcceptCB.onclick=function()
{this.parent.parent.userAcceptCheck();}
if(oAcceptCB1.eElem){oAcceptCB1.onclick=function()
{this.parent.parent.userAcceptCheck();}}
if(oAcceptCB2.eElem){oAcceptCB2.onclick=function()
{this.parent.parent.userAcceptCheck();}}
this.userAcceptCheck=function(){var bCheck=oAcceptCB.isChecked();var bCheck1=true;var bCheck2=true;if(oAcceptCB1.eElem)
bCheck1=oAcceptCB1.isChecked();if(oAcceptCB2.eElem)
bCheck2=oAcceptCB2.isChecked();bCheck=bCheck&&bCheck1&&bCheck2;oAcceptErrorDiv.show(!bCheck);}
eButton.onclick=function()
{var oConfig=this.parent.parent.parent.getConfig('Registration.ErrorListings');if(oConfig)
var oFieldCheck=new EbayFieldCheck(ebay.oDocument.oPage,"Registration.FieldCheck",oConfig);var oDiv0=oConfig.sDivPrefix+"email";var d=new EbayHTMLLayer(this,oDiv0);if(!d.eElem)
d.bind();var oDiv2=oConfig.sDivPrefix1+"retype_email";var d2=new EbayHTMLLayer(this,oDiv2);if(!d2.eElem)
d2.bind();var oImgdiv0=oConfig.sImgDivPrefix+"email";var i=new EbayHTMLLayer(this,oImgdiv0);if(!i.eElem)
i.bind();var oImgdiv1=oConfig.sImgDivPrefix+"retype_email";var i1=new EbayHTMLLayer(this,oImgdiv1);if(!i1.eElem)
i1.bind();var eDiv=oConfig.sExactDiv[0];var eDiv1=eDiv[1];var eDiv2=new EbayHTMLLayer(this,eDiv1);if(!eDiv2.eElem)
eDiv2.bind();var sVal0=eFrm.getElement("email").value;var sVal1=eFrm.getElement("retype_email").value;var G=(sVal0==sVal1)?true:false;if(!G){d.show(true);d.setValue(oConfig.sMthEmailEr);eDiv2.show(false);i1.show(true);}
else if((sVal0=='')&&(sVal1=='')){d.show(true);eDiv2.show(false);i.show(true);i1.show(true);}
else{d.show(false);eDiv2.show(true);i1.show(false);}
eFrm.setAction=oPc.sTargetUrl;var oCmd=eFrm.getElement("MfcISAPICommand");if(oCmd)
oCmd.value=oPc.sTargetType;var oMode=eFrm.getElement("mode");if(oMode)
{oMode.value="0";if(oPc.sTargetMode)
oMode.value=oPc.sTargetMode;}
if(oAcceptCB.eElem)
{var bCheck=oAcceptCB.isChecked();var bCheck1=true;var bCheck2=true;if(oAcceptCB1.eElem)
bCheck1=oAcceptCB1.isChecked();if(oAcceptCB2.eElem)
bCheck2=oAcceptCB2.isChecked();bCheck=bCheck&&bCheck1&&bCheck2;if(oConfig)
{var oUserCfg=ebay.oDocument.getConfig("Registration.OnePage.UserIdLookupCntrl");if(typeof(oUserCfg)=="undefined")
oUserCfg=ebay.oDocument.getConfig("Registration.OnePage.Registration.OnePage.UserIdOptimization");if(oUserCfg)
{if(oUserCfg.sUserIdErrorLayerId)
{var oUserErrorLayer=new EbayHTMLLayer(this,oUserCfg.sUserIdErrorLayerId);if(!oUserErrorLayer.eElem)
oUserErrorLayer.bind();}
if(oUserCfg.sUserSecDiv&&oUserErrorLayer.eElem.style.visibility=="visible")
{var oUserSecDiv=new EbayHTMLLayer(this,oUserCfg.sUserSecDiv);if(!oUserSecDiv.eElem)
oUserSecDiv.bind();if(oUserCfg.sErrorClass&&oUserErrorLayer.eElem.innerHTML.has(oUserCfg.sErrorClass))
oUserSecDiv.show(false);}}
if(!bCheck||!oFieldCheck.bFlag)
{oAcceptErrorDiv.setValue(oPc.sChkErrorMsg);oAcceptErrorDiv.show(!bCheck);if(oELPgEr)
oELPgEr.focus(true);if(oRegErrorDiv)
oRegErrorDiv.show(true);return false;}
else
{oAcceptErrorDiv.show(!bCheck);if(oRegErrorDiv)
oRegErrorDiv.show(false);eFrm.submit();}
return false;}
else
eFrm.submit();return false;}else if(oPc.sCheckId&&!oAcceptCB.eElem&&oRegErrorDiv&&oConfig&&oFieldCheck&&!oFieldCheck.bFlag){oRegErrorDiv.show(true);return false;}}}
function ctry_NoCall(){this.onblur=null;}
function ctry_blurEv(){this.onblur=elCountry_onchange;}
function elCountry_onchange()
{eFrm.getElement("MfcISAPICommand").value=oPc.sCountryTargetType;var oMode=eFrm.getElement("mode");var oCountrySel=eFrm.getElement(oPc.sCountrySelectName);var vAllCountries="1000",countryVal="";if(typeof(oPc.sAllCountriesVal)!='undefined')
vAllCountries=oPc.sAllCountriesVal;if(oCountrySel)
{countryVal=oCountrySel.options[oCountrySel.selectedIndex].value;if(oMode)
{if(countryVal==vAllCountries)
oMode.value="0";else
oMode.value="1";}}
else
{if(oMode)
oMode.value="0";}
var oSurvey=ebay.oDocument.oPage._getControl("KeyFlow.Exit.Survey");if(oSurvey)
oSurvey.disable();eFrm.submit();}
if(oPc.sCountryLinkName&&oPc.sCountryLinkName.length>0)
{var elCountry=new EbayHTMLAnchor(eFrm,oPc.sCountryLinkName);elCountry.onclick=elCountry_onchange;}
else if(oPc.sCountrySelectName)
{var elCountry=new EbayHTMLSelect(eFrm,oPc.sCountrySelectName);elCountry.onchange=elCountry_onchange;}
if(oPc.sAnchirChangeSiteName&&oPc.sAnchirChangeSiteName.length>0)
{var elChangeSite=new EbayHTMLAnchor(eFrm,oPc.sAnchirChangeSiteName);elChangeSite.onclick=function()
{eFrm.getElement("MfcISAPICommand").value=oPc.sSiteChangeTargetType;var oMode=eFrm.getElement("mode");if(oMode)
oMode.value="0";eFrm.submit();return false;}}
new EbayHTMLTextRestricted(eFrm,"email");new EbayHTMLTextRestricted(eFrm,"retype_email");var c=ebay.oDocument.aConfigs['statecountryrel'];if(c)
{var sc=new EbayParentChildSel(eFrm,"statecountry",c);if(oPc.sCountryLinkName&&oPc.sCountryLinkName.length<=0)
{sc.childCntrl.onchange=elCountry_onchange;}}
var oD=ebay.oDocument;var oP=oD.oPage;var oPc=oP.oConfig;oPc.MasterFormName="RegisterEnterInfo";oPc.UserIdHiddenName="user";var hUser=oD.getFormElem(oPc.UserIdHiddenName);var oForm=new EbayHTMLForm(this,oPc.formName);var eRadio=new EbayHTMLRadio(oP,oPc.sUserIdRadioElemName);eRadio.onclick=function(pSourceElem)
{var v=pSourceElem.value;if(v=="on"||v=="")
v=eText.value();if(hUser)
hUser.value=v;if(userid)
{userid.updatePassword=true;userid.onchange();}}
eRadio.update=function(pEvent)
{typeof(oPc.iSelectedCustomRadioIndx)=="undefined"?oPc.iSelectedCustomRadioIndx=0:null;this.selectByIndex(oPc.iSelectedCustomRadioIndx);}
var eText=new EbayHTMLText(oP,oPc.sTextUserIdElemName);var e=eText._registerEvent("onfocus","focus")
eRadio._registerListener(e,eRadio.EVENT_AFTER,"update");eText.focus=function()
{eRadio.update();}
eText.onblur=function()
{if(hUser)
hUser.value=this.value();if(userid)
userid.onchange();}
var oC=ebay.oDocument.getConfig("Registration.UserIdLookupCntrl");if(oC)
new EbayAutoSuggest(this,"Registration.UserIdLookupCntrl",oC)
var oCfg=oD.getConfig("Registration.OnePage.UserAgreement");if(oCfg)
{var oEnAnchor=new EbayHTMLAnchor(this,oCfg.sEnAnchor);var oFrAnchor=new EbayHTMLAnchor(this,oCfg.sFrAnchor);var oEnContent=new EbayHTMLLayer(this,oCfg.sEnContent);var oFrContent=new EbayHTMLLayer(this,oCfg.sFrContent);var oEnVersion=new EbayHTMLLayer(this,oCfg.sEnVersion);var oFrVersion=new EbayHTMLLayer(this,oCfg.sFrVersion);oEnVersion._registerListener(oD._getEvent("load"),oD.EVENT_AFTER,"show(true)");oEnContent._registerListener(oD._getEvent("load"),oD.EVENT_AFTER,"show(true)");oEnAnchor._registerEvent("onclick","ShowEnglishVersion");oFrAnchor._registerEvent("onclick","ShowFrenchVersion");oEnAnchor.ShowEnglishVersion=function()
{oFrVersion.show(false);oFrContent.show(false);oEnContent.show(true);oEnVersion.show(true);}
oFrAnchor.ShowFrenchVersion=function()
{oFrContent.show(true);oEnVersion.show(false);oFrVersion.show(true);oEnContent.show(false);}}
var oD=this.parent,oC=oD.getConfig("Registration.ChinaCBT.BubbleHelp"),oB,e,aE,l,i;if(oC)
{aE=oC.aElemNames,l=aE.length;oB=new EbayBubbleHelp(this,oC.name,oC);for(i=0;i<l;i++)
{e=new EbayHTML(oB,aE[i],aE[i]);oB.subscribeElemEvents(e,oC.sBubbleText);}}
var oShowSecConfig=this.parent.getConfig("Registration.ShowmySecretQn");if(oShowSecConfig){var oPickQn=new EbayHTMLSelect(this,oShowSecConfig.sPickQn);oPickQn.bind();oPickQn.show(true);oPickQn._registerEvent("onchange","parent.showOwnQn");this.showOwnQn=function(){var oOwnQn=new EbayHTMLLayer(this,oShowSecConfig.sOwnQn);oOwnQn.bind();var b=(oPickQn.getSelectedValue()==oShowSecConfig.iVal)?true:false;oOwnQn.show(b);}}}
ebay.oDocument.oPage.onAfterLoad=function()
{var oPass=this._getControl("pass");if(psvOn&&oPass&&oPass.eElem&&oPass.eElem.onkeyup!='UpdatePassword'){oPass.eElem.onkeyup=UpdatePassword;}
var oD=ebay.oDocument,oP=oD.oPage,oPc=oP.oConfig;var eFrm=oP.controls["RegisterEnterInfo"];var elems=eFrm.getElements();for(var i=0,len=elems.length;i<len;i++)
{if(elems[i].type=="text")
{elems[i].focus();break;}}
var oCmd=eFrm.getElement("MfcISAPICommand");if(oCmd)
oCmd.value=oPc.sTargetType;var elCountry=this.controls["RegisterEnterInfo"].controls[oPc.sCountrySelectName];if(elCountry)
{var vAllCountries="1000",countryVal="";if(typeof(oPc.sAllCountriesVal)!='undefined')
vAllCountries=oPc.sAllCountriesVal;if(elCountry.eElem)
countryVal=elCountry.getSelectedValue();if(countryVal==vAllCountries)
elCountry.selectByIndex(0);}
var reicfg=oD.getConfig("reitracking");if(typeof(reicfg)!="undefined")
{with(reicfg)
{var im=new EbayHTMLImage(this,imageID,false);im.bind();var e,reielem;var sTracked="",sFunc="";for(var i=0,len=events.length;i<len;i++)
{e=events[i].split("=");reielem=eFrm.getElement(e[0]);if(reielem!=null&&(reielem.value==""||(reielem.type.has("select")&&reielem.value.is("default"))))
{sFunc='reielem.onchange = function ()';sFunc+='{';sFunc+=' if (!this.value.is("") && sTracked.indexOf(":'+e[1]+':")==-1)';sFunc+=' {';sFunc+='  var ord = (new Date()).getTime();';sFunc+='  im.source("'+command+'&'+eventID+'='+e[1]+'&ord=" + ord);';sFunc+='  sTracked += ":'+e[1]+':"';sFunc+=' }';sFunc+='};';eval(sFunc);}}}}
var oD=ebay.oDocument;var oP=oD.oPage;var oPc=oP.oConfig;var c=this.controls[oPc.sUserIdRadioElemName];if(c.eElem)
{if(!oPc.errorCondition&&typeof(c.eElem[0])!='undefined')
c.eElem[0].checked=true;var e=ebay.oDocument.getFormElem("user");if(e&&typeof(c.eElem[0])!='undefined'&&c.eElem[0].checked)
e.value=c.eElem[0].value;}
if(this.AO_ws)
this.AO_ws();}
function sendBusinessRegistration()
{for(var i=0,j=0,len=document.RegisterEnterInfo.elements.length;i<len;i++,j++)
{if((document.RegisterEnterInfo.elements[i].type!="radio")&&(document.RegisterEnterInfo.elements[i].type!="checkbox"))
{if(document.RegisterEnterInfo.elements[i].name!="bizflow")
{document.BusinessRegistration.elements[j].value=document.RegisterEnterInfo.elements[i].value;}
else
{j=j-1;}}
else if((document.RegisterEnterInfo.elements[i].type=="checkbox"))
{document.BusinessRegistration.elements[j].checked=document.RegisterEnterInfo.elements[i].checked;document.BusinessRegistration.elements[j].value=document.RegisterEnterInfo.elements[i].checked?1:0;}
else
{j=j-1;if(document.RegisterEnterInfo.elements[i].name!="BizFlow")
{if(document.RegisterEnterInfo.elements[i].checked==true)
{var el=document.createElement("input");el.setAttribute("type","hidden");el.setAttribute("name",document.RegisterEnterInfo.elements[i].name);el.setAttribute("value",document.RegisterEnterInfo.elements[i].value);document.forms.BusinessRegistration.appendChild(el);}}}
if(document.RegisterEnterInfo.elements[i].name=="MfcISAPICommand")
document.BusinessRegistration.elements[i].value="RegisterEnterInfo";if(document.RegisterEnterInfo.elements[i].name=="mode")
document.BusinessRegistration.elements[i].value="0";if(document.RegisterEnterInfo.elements[i].name=="accountType")
if(document.RegisterEnterInfo.elements[i].value=="1")
document.BusinessRegistration.elements[i].value="2";else
document.BusinessRegistration.elements[i].value="1";}
var bLog=document.BusinessRegistration.elements["buyerLogging"];if(bLog)bLog.value=(bLog.value=="0")?"1":"0";document.BusinessRegistration.submit();}
function eRadio_onAfterCheck()
{}
ebay.oDocument.oPage.initbirthdate3=function()
{var oP=this.parent;var pCfg=oP.oDocument.getConfig("registration.birthdate3");if(pCfg)
{oTlyr=new EbayHTMLText(this,pCfg.sTextBox);if(!oTlyr.eElem)
oTlyr.bind();oImg=new EbayHTMLImage(this,pCfg.sYearImg);if(!oImg.eElem)
oImg.bind();oTxt=new EbayHTMLLayer(this,pCfg.sYearText);if(!oTxt.eElem)
oTxt.bind();oGry=new EbayHTMLLayer(this,pCfg.sGreyTxt);if(!oGry.eElem)
oGry.bind();oTlyr._registerEvent("onfocus","parent.clearText");oTlyr._registerEvent("onblur","parent.recallText");oTlyr._registerEvent("onkeyup","parent.validTxt");}
this.clearText=function()
{if(this.eElem.value==pCfg.sTextValue)
{this.eElem.value="";oTlyr.setClass(pCfg.sTextClass_1);}}
this.recallText=function()
{if(this.eElem.value=="")
{oTlyr.setClass(pCfg.sTextClass);this.eElem.value=pCfg.sTextValue;}}
this.validTxt=function()
{var ValidChars="0123456789",IsNumber=true,Char,val=this.eElem.value;for(i=0,len=val.length;i<len&&IsNumber==true;i++)
{var m=document.getElementById("div1_birthdate2");Char=val.charAt(i);if(ValidChars.indexOf(Char)==-1)
{IsNumber=false;if(m&&((m.style.visibility=='hidden')||(m.style.display=='none')))
{oImg.show(true);oTxt.show(true);oGry.show(false);}else
{oGry.show(false);}}
else
{if(m&&((m.style.visibility=='visible')||(m.style.display=='')))
{oGry.show(false);}else
{oImg.show(false);oTxt.show(false);oGry.show(true);}}}
return IsNumber;}}

//28@@m8

function EbayFieldCheck(pParent,pName,pCfg)
{if(!this.objType)
this.objType="EbayFieldCheck";this.base=EbayHTML;this.base(pParent,pName,pName,false,pCfg);this.aGD=pCfg.aGrpDetails;this.oPrevObj;this.oPrevtxt;this.bNull;this.bCheck;this.bFlag=true;this.bAjaxErrorCheck=false;this.bBirthdate1=false;this.bBirthdate2=false;this.bBirthdate3=false;this.bDayphone1=false;this.bDayphone2=false;this.bDayphone3=false;this.bBizno1=false;this.bBizno2=false;this.init=function()
{with(this)
{for(x in aGD)
{var oParDiv=new EbayHTMLText(this,x);for(var i=0,iLen=pCfg.aGrpDetails[x].length;i<iLen;i++)
{oElem=this.parent._getControl("Registration.ErrorListings")._getControl(aGD[x][i]);if(oElem.eElem)
{switch(oElem.eElem.type)
{case"text":case"password":{pType="Text";break;}
case"radio":{pType="Radio";break;}
case"checkbox":{pType="Checkbox";break;}
case"select-one":{pType="Select";break;}}}
eval("var oE=new EbayHTML"+pType+"(oParDiv,aGD[x][i]);");if(!oE.eElem)
oE.bind();validateText(oE);}}}}
this.validateText=function(oObject)
{if((oObject.eElem)&&(oObject.eElem.type=="text"||oObject.eElem.type=="password"||oObject.name=="gender"||oObject.eElem.type=="select-one"))
{if(psvOn&&oObject.name=="pass"&&typeof(UpdatePassword)!='undefined'){oObject.eElem.onkeyup=UpdatePassword;}
var oP=this.parent._getControl("Registration.ErrorListings");var oDiv=oP._getControl(pCfg.sDivPrefix+oObject.name);var oDiv1=oP._getControl(pCfg.sDivPrefix1+oObject.name);var oImgdiv=oP._getControl(pCfg.sImgDivPrefix+oObject.name);var oPgErr=oP._getControl(pCfg.sPgErrPrefix+oObject.name);var oPgTop=oP._getControl(pCfg.sPgTopErr);var sShCss=pCfg.sPgErrClass;for(var k=0,len=pCfg.aOptionaltxtfld.length;k<len;k++)
{if(pCfg.aOptionaltxtfld[k]==oObject.name)
oP.bNull=true;}
if(!oP.bNull&&oObject.name!="countryId")
{if(oObject.eElem.type=="text"||oObject.eElem.type=="password")
var oObjVal=oObject.getValue();if(oObject.eElem.type=="select-one")
var oObjVal=oObject.getSelectedIndex();if(oObject.name=="gender")
var oObjVal=oObject.getSelectedIndex();oDiv.show();var b=oObjVal?false:true;if(oObject.name=="birthdate1"){bBirthdate1=b;}else if(oObject.name=="birthdate2"){bBirthdate2=b;}else if(oObject.name=="birthdate3"){bBirthdate3=b;}
if(oObject.name=="birthdate2"&&(bBirthdate1||bBirthdate2||bBirthdate3)){b=true;}
if(oObject.name=="dayphone1"){this.bDayphone1=b;}else if(oObject.name=="dayphone2"){this.bDayphone2=b;}else if(oObject.name=="dayphone3"){this.bDayphone3=b;}
if(oObject.name=="dayphone1"&&(this.bDayphone1||this.bDayphone2||this.bDayphone3)){b=true;if(b==true){var oP1=this.parent._getControl("Registration.ErrorListings");var oYearImg=oP1.oConfig.sYearImg;if(oYearImg!=undefined)
{var oYearImg1=new EbayHTMLImage(this,oYearImg);if(!oYearImg1.eElem)
oYearImg1.bind();oYearImg1.show(false);}
var oYearTxt=oP1.oConfig.sYearTxt;if(oYearTxt!=undefined)
{var oYearTxt1=new EbayHTMLLayer(this,oYearTxt);if(!oYearTxt1.eElem)
oYearTxt1.bind();oYearTxt1.show(false);}}}
if(oObject.name=="bizNumber1"){bBizno1=b;}else if(oObject.name=="bizNumber2"){bBizno2=b;}
if(oObject.name=="bizNumber1"&&(bBizno1||bBizno2)){b=true;}
if(oObject.name=="myquestion"){var oShowSecQnConfig=ebay.oDocument.getConfig("Registration.ShowmySecretQn");if(oShowSecQnConfig){var oOwnSecQn=new EbayHTMLLayer(this,oShowSecQnConfig.sOwnQn);oOwnSecQn.bind();if(oOwnSecQn.eElem.style.display=="none"&&b)
b=false;}}
if(oObject.name=="gender"){if(oObjVal==-1){b=true;}else if(oObjVal==0){b=false;}}
if(b&&oObject.name==oP.parent.oConfig.sTextUserIdElemName)
{if(oP.parent.oConfig.sUserIdRadioElemName)
{var oUserRadio=new EbayHTMLRadio(this,oP.parent.oConfig.sUserIdRadioElemName);if(!oUserRadio.eElem)
oUserRadio.bind();var sRadioValue=oUserRadio.getValue();b=(sRadioValue=="on"||sRadioValue=="")?true:false;}}
oDiv1.show(b);oImgdiv.show(b);if(b&&oPgErr.eElem&&oPgTop.eElem)
{oPgTop.eElem.className=sShCss;oPgErr.eElem.className=sShCss;errAnc=oPgErr.eElem.firstChild.id;var oErrAnc=new EbayHTMLAnchor(this,errAnc);if(!oErrAnc.eElem)
oErrAnc.bind();if(oErrAnc.eElem)
{oErrAnc.oObj=oObject;oErrAnc.subscribeEvents("onclick");oErrAnc._registerEvent("onclick","parent.clickErrList");this.clickErrList=function()
{this.oObj.eElem.focus();}}}
for(iIndex in pCfg.sSecErrorDivs)
{if(oObject.name==pCfg.sSecErrorDivs[iIndex][0])
{var sParam=pCfg.sDivPrefix1+oObject.name;var oQuickErrorDiv=new EbayHTMLLayer(this,sParam);var oSecDiv=new EbayHTMLLayer(this,pCfg.sSecErrorDivs[iIndex][1]);if(!oQuickErrorDiv.eElem)
oQuickErrorDiv.bind();if(!oSecDiv.eElem)
oSecDiv.bind();if(oQuickErrorDiv.eElem)
oSecDiv.show(!b);}}
if(b)
this.bFlag=false;}
else
oP.bNull=false;}}
this.init();}

//29@@m4

function EbayHTMLAudioGifChallenge(pParent,pName,pDisabled,pCfg)
{if(!this.objType)
this.objType="EbayHTMLAudioGifChallenge";this.base=EbayHTML;this.base(pParent,pName,pName,pDisabled,pCfg);this.oLinkRefresh=this.oLinkListen=null;this.init=ebHTMLAudioGifChallengeInit;this.createControls=ebHTMLAudioGifChallengeCreateControls;with(this)
_registerListener(oDocument._getEvent("load"),EVENT_BEFORE,"createControls");this.init();}
function ebHTMLAudioGifChallengeCreateControls()
{with(this)
{var oP=this.parent,oD=oP.parent,c=oD.getConfig("Security.Gif.Audio.Challenge"),oR,oL,oF,oT;if(c)
{oF=new EbayHTMLFrame(this,c.sBotFrameId,c);oR=new EbayHTMLAnchor(this,c.sBotRefreshLinkId,false,c);oR.onclick=function()
{var oP=this.parent,oC=this.oConfig,oL=oP._getControl(oC.sBotListenLinkId);if(oL.eElem)
oL.enable(false);oT=oD.getFormElem(c.sTokenTextId);if(oT&&c.sTokenDefTxt&&oT.value!=c.sTokenDefTxt)
oT.value="";if(oL.eElem)
oL.eElem.href="#";oF.setSource(oC.sBotIframeUrl);var oPop=oP._getControl("audioPopup");if(oPop)
oPop.close();return false;}
oL=new EbayHTMLAnchor(this,c.sBotListenLinkId,false,c);oL.onclick=function()
{var c=this.oConfig,oPop,l,t,url=c.sAudioHelpPopupUrl,ts;l=parseInt(screen.availWidth-c.sAudioHelpPopupWidth-10);t=0;ts=this.parent.oDocument.getFormElem(c.sTokenStringId).value;url+=url.has("?")?"&":"?";url+=c.sTokenStringQueryParamName+"="+ts;oPop=new EbayHTMLPopup(this,"audioPopup",c);oPop.showEx(url,c.sAudioHelpPopupWidth,c.sAudioHelpPopupHeight,0,0,0,1,1,0,l,t);return false;}}}}
function ebHTMLAudioGifChallengeInit()
{var oD=this.parent.parent;oD._registerListener(oD._getEvent("load"),oD.EVENT_BEFORE,"createControls");}
new EbayHTMLAudioGifChallenge(ebay.oDocument.oPage,"AudioGifChallege");

//30@@m4

function EbayAutoSuggest(pParent,pName,pCfg)
{if(!this.objType)
this.objType="EbayAutoSuggest";this.base=EbayHTML;this.base(pParent,pName,pName,false,pCfg);this.cfg=pCfg;var oObj=this;this.oDoc=this.oDocument.doc;this.init=function()
{with(this)
{var oP=parent,evtOnClick;this.oFrame=new EbayHTMLFrame(this,cfg.sFrameId);ebay.oDocument.oPage.oLookupResult=this.oLookupResult=new EbayHTMLLayer(this,cfg.sLookupResultLayerId);this.oUserIdTextElemName=new EbayHTMLText(this,cfg.sUserIdTextElemName);this.oHdnTrackVar=new EbayHTMLText(this,cfg.sHdnTrackVar);this.oFirstName=new EbayHTMLText(this,cfg.sFirstName);this.oLastName=new EbayHTMLText(this,cfg.sLastName);this.oZip=new EbayHTMLText(this,cfg.sZip);this.oDayPhone1=new EbayHTMLText(this,cfg.sDayPhone1);this.oDayPhone2=new EbayHTMLText(this,cfg.sDayPhone2);this.oDayPhone3=new EbayHTMLText(this,cfg.sDayPhone3);this.oBirthDate1=new EbayHTMLSelect(this,cfg.sBirthDate1);this.oBirthDate2=new EbayHTMLSelect(this,cfg.sBirthDate2);this.oBirthDate3=new EbayHTMLText(this,cfg.sBirthDate3);this.oUserIdTextElemName._registerEvent("onkeyup","parent.autoSuggest");this.oRadio=oP._getControl(cfg.sUserIdRadioElemName);if(this.oRadio)
evtOnClick=this.oRadio.onclick;this.oRadio=new EbayHTMLRadio(this,cfg.sUserIdRadioElemName);this.oRadio.onclick=function(pSource)
{if(evtOnClick)
evtOnClick(pSource);var v=pSource.value;if(v!="on"&&v!="")
this.parent.oUserIdTextElemName.setValue(v);}}}
var bKeyCode=false;this.autoSuggest=function()
{with(this.parent)
{if(oUserIdTextElemName.getValue()!=""&&oUserIdTextElemName.getValue().length>=cfg.iMinChar){if(!bKeyCode){txtVal=oUserIdTextElemName.getValue();if(cfg.bIncludeTimer)
startTimer();else
checkResponse();}
bKeyCode=false;}
else
oLookupResult.show(false);}}
this.checkResponse=function()
{with(this)
{var s=cfg.sPostUrl;s+="&"+oUserIdTextElemName.name+"="+oUserIdTextElemName.getValue();if(typeof(oFirstName.getValue())!='undefined'&&oFirstName.getValue()!='')
s+="&"+oFirstName.name+"="+oFirstName.getValue();if(typeof(oLastName.getValue())!='undefined'&&oLastName.getValue()!='')
s+="&"+oLastName.name+"="+oLastName.getValue();if(typeof(oZip.getValue())!='undefined'&&oZip.getValue()!='')
s+="&"+oZip.name+"="+oZip.getValue();if(typeof(oDayPhone1.getValue())!='undefined'&&oDayPhone1.getValue()!='')
s+="&"+oDayPhone1.name+"="+oDayPhone1.getValue();if(typeof(oDayPhone2.getValue())!='undefined'&&oDayPhone2.getValue()!='')
s+="&"+oDayPhone2.name+"="+oDayPhone2.getValue();if(typeof(oDayPhone3.getValue())!='undefined'&&oDayPhone3.getValue()!='')
s+="&"+oDayPhone3.name+"="+oDayPhone3.getValue();if(oBirthDate1.eElem&&typeof(oBirthDate1.eElem)!='undefined'&&typeof(oBirthDate1.getValue())!='undefined'&&oBirthDate1.getValue()!='')
s+="&"+oBirthDate1.name+"="+oBirthDate1.getValue();if(oBirthDate2.eElem&&typeof(oBirthDate2.eElem)!='undefined'&&typeof(oBirthDate2.getValue())!='undefined'&&oBirthDate2.getValue()!='')
s+="&"+oBirthDate2.name+"="+oBirthDate2.getValue();if(typeof(oBirthDate3.getValue())!='undefined'&&oBirthDate3.getValue()!='')
s+="&"+oBirthDate3.name+"="+oBirthDate3.getValue();oFrame.setSource(s);}}
this.processResponse=function(pCfg)
{with(this)
{this.oDocument.win.clearTimeout(this.iTimer);this.oLookupResult.show(false);if(this.oUserIdTextElemName.getValue()!=""&&this.oUserIdTextElemName.getValue().length>=this.cfg.iMinChar&&pCfg.bIsResult==true){this.oLookupResult.show(true);sHtml=pCfg.sResultContent;iLen=pCfg.iResLength;sHgltClass=pCfg.sHgltClass;sOrdyClass=pCfg.sOrdyClass;sHtml=sHtml?sHtml:this.sSpacerHtml;this.oLookupResult.setValue(sHtml);var aResult=this.oDocument.oPage.aResult=new Array();for(var i=0;i<iLen;i++)
{aResult[i]=new EbayHTMLLayer(this,pCfg.sDivName+i);aResult[i].subscribeEvents("onclick","onmouseover","onmouseout");aResult[i]._registerEvent("onmouseover","parent.hgltOption");aResult[i]._registerEvent("onmouseout","parent.deHgltOption");aResult[i]._registerEvent("onclick","parent.showValue");aResult[i].bind();aResult[i].eElem.style.width=cfg.iWidth;}}}}
var aResult=ebay.oDocument.oPage.aResult;this.hgltOption=function(){if(this.parent.oUserIdTextElemName.getValue()!=""){var aResult=ebay.oDocument.oPage.aResult;for(var h=0,len=aResult.length;h<len;h++)
aResult[h].setClass(sOrdyClass);this.setClass(sHgltClass);}}
this.deHgltOption=function(){var aResult=ebay.oDocument.oPage.aResult;for(var d=0,len=aResult.length;d<len;d++)
{if(aResult[d].getClass()==sHgltClass)
aResult[d].setClass(sOrdyClass);}}
this.showValue=function(){with(this.parent)
{oUserIdTextElemName.setValue(this.eElem.innerHTML);oLookupResult.show(false);oHdnTrackVar.setValue("1");}}
this.startTimer=function()
{with(this)
{var t=(typeof(cfg.iTimeout)=="undefined")?30000:cfg.iTimeout;clearTimeout(this.iTimer)
this.iTimer=oDocument.win.setTimeout("ebay.oDocument.oPage._getControl(\""+name+"\").checkResponse()",t);}}
var bNxtFlag=false,bPrevFlag=false,pValue;this.previousSuggestion=function()
{for(var i=1,len=aResult.length;i<len;i++)
{if(aResult[i].getClass()==sHgltClass)
{aResult[i].setClass(sOrdyClass);aResult[i-1].setClass(sHgltClass);pValue=aResult[i-1].getValue();bPrevFlag=false;break;}}
if(!bPrevFlag&&pValue==txtVal)
{aResult[aResult.length-1].setClass(sHgltClass);pValue=aResult[aResult.length-1].getValue();bPrevFlag=true;}
if(!bPrevFlag&&i==aResult.length)
{aResult[0].setClass(sOrdyClass);pValue=txtVal;bPrevFlag=false;}
this.oUserIdTextElemName.setValue(pValue);return true;}
this.nextSuggestion=function()
{for(var i=0,len=aResult.length;i<(len-1);i++){if(aResult[i].getClass()==sHgltClass){aResult[i].setClass(sOrdyClass);aResult[i+1].setClass(sHgltClass);pValue=aResult[i+1].getValue();bNxtFlag=true;break;}}
if(!bNxtFlag)
{aResult[0].setClass(sHgltClass);pValue=aResult[0].getValue();i=0;}
if(i==(aResult.length-1))
{aResult[i].setClass(sOrdyClass);bNxtFlag=false;pValue=txtVal;}
this.oUserIdTextElemName.setValue(pValue);return true;}
this.hideSuggestions=function()
{for(var c=0,len=aResult.length;c<len;c++)
{var cls=aResult[c].getClass();if(cls==sHgltClass)
{pVal=aResult[c].getValue();this.oUserIdTextElemName.setValue(pVal);this.oLookupResult.show(false);this.oHdnTrackVar.setValue("1");}}}
this.oDoc.onclick=function()
{oLookupResult=ebay.oDocument.oPage.oLookupResult;if(typeof(oLookupResult)!="undefined")
oLookupResult.show(false);}
this.oDoc.onkeypress=function(evt)
{var evt=(evt)?evt:((event)?event:null);var node=(evt.target)?evt.target:((evt.srcElement)?evt.srcElement:null);if((evt.keyCode==13)&&(node.type=="text"))
{bKeyCode=true;oObj.hideSuggestions();return false;}}
this.oDoc.onkeydown=function(pEvent){var evt=pEvent||event;if(evt.keyCode==38||evt.keyCode==40){bKeyCode=true;aResult=ebay.oDocument.oPage.aResult;oLookupResult=ebay.oDocument.oPage.oLookupResult;if(oLookupResult.eElem.style.visibility=="visible"){switch(evt.keyCode){case 38:oObj.previousSuggestion();break;case 40:oObj.nextSuggestion();break;}}}}
this.init();}

//31@@m2

ebay.oDocument.oPage.AO_ws=function()
{with(this)
{var oDoc=parent;var oCfg=oDoc.getConfig("TrackingPageNames");if(oCfg)
{var pUrl=oCfg.sPathName;if(!oDoc.createElement||!pUrl)
return;var oScript=oDoc.createElement("script");oScript.type='text/javascript';oScript.src=pUrl;var oFrag=document.getElementsByTagName("head")||document.getElementsByTagName("body");oFrag[0].appendChild(oScript);}}}

//32@@m13

function EbayQuickWins(pParent,pName,pCfg)
{if(!this.objType)
this.objType="EbayQuickwins";this.base=EbayHTML;this.base(pParent,pName,pName,false,pCfg);this.aApoStts=[];this.aGD=pCfg.aGrpDetails;this.oPrevObj;this.oPrevtxt;this.bNull;this.bCheck;this.eFrmm=new EbayHTMLForm(this,pCfg.sTargetType);this.chgctry=pCfg.sCountryTargetType;this.oStateDD=new EbayHTMLSelect(this,pCfg.sStateSelectName);if(!this.oStateDD.eElem)
this.oStateDD.bind();this.oCtryDD=new EbayHTMLSelect(this,pCfg.sCountryselectName);if(!this.oCtryDD.eElem)
this.oCtryDD.bind();this.init=function()
{with(this)
{if(oStateDD!=null&&oStateDD!='')
{oStateDD._registerEvent("onchange","parent.onChange");}
for(x in pCfg.aApoState)
{aApoStts[x]=pCfg.aApoState[x];}
this.onChange=function()
{for(x in this.parent.aApoStts)
{if(this.getSelectedValue()==pCfg.aApoState[x])
{eFrmm.getElement("MfcISAPICommand").value=this.parent.chgctry;if(this.parent.oCtryDD.eElem.value!=225)
{this.parent.oCtryDD.eElem.value=225;eFrmm.submit();}}}}
for(x in aGD)
{var oParDiv=new EbayHTMLText(this,x);for(var i=0,iLen=pCfg.aGrpDetails[x].length;i<iLen;i++)
{var oElem=new EbayHTMLText(this,aGD[x][i]);oElem.bind();if(oElem.eElem)
{switch(oElem.eElem.type)
{case"text":case"password":{createObj(oParDiv,x,i,"Text");break;}
case"radio":{createObj(oParDiv,x,i,"Radio");break;}
case"checkbox":{createObj(oParDiv,x,i,"Checkbox");break;}
case"select-one":{createObj(oParDiv,x,i,"Select");break;}}}}}}}
this.createObj=function(oParObj,temp,iInd,pType){with(this){eval("var oE=new EbayHTML"+pType+"(oParObj,aGD[temp][iInd]);");oE._registerEvent("onfocus","parent.parent.getPrevObj");oE._registerEvent((pType=="Radio")?"onclick":"onblur","parent.parent.validateText");if(aGD[temp][iInd]=='ProvinceId'){var u="undefined";framesCheck();if(typeof(ITRegisterEnterInfoOnLoad)!=u){oE._registerEvent("onchange","parent.parent.provinceOnChangeEvent");oE._registerEvent("onAfterLoad","parent.parent.provinceOnAfterLoadEvent");}}
oEL=new EbayHTMLLayer(this,pCfg.sDivPrefix+aGD[temp][iInd]);oEL.bind();oEL1=new EbayHTMLLayer(this,pCfg.sDivPrefix1+aGD[temp][iInd]);oEL1.bind();oELImg=new EbayHTMLLayer(this,pCfg.sImgDivPrefix+aGD[temp][iInd]);oELImg.bind();oELError=new EbayHTMLLayer(this,pCfg.sPgErrPrefix+aGD[temp][iInd]);oELError.bind();oELPgEr=new EbayHTMLAnchor(this,pCfg.sPgTopErr);oELPgEr.bind();}}
this.provinceOnChangeEvent=function(){provinceOnChange();}
this.provinceOnAfterLoadEvent=function(){provinceOnAfterLoad();}
this.getPrevObj=function()
{var oFrame=new EbayHTMLFrame(this,pCfg.sFrameId);oFrame.bind();var oP=this.parent.parent;if(oP.oPrevObj&&oP.oPrevObj!=this.parent)
{var sParams=oP.oPrevObj.name;if(pCfg.sGlobalReg)
{sParams+="&"+"globalreg"+"="+pCfg.sGlobalReg;}
var sGrpname=oP.oConfig.aGrpDetails[oP.oPrevObj.name];this.bFrame=false;for(var m=0,len=sGrpname.length;m<len;m++)
{var oGrpname=oP.oPrevObj.controls[sGrpname[m]];if(oGrpname)
{str=oP.URLencode(oP.oPrevObj.controls[sGrpname[m]].getValue());sParams+="&"+sGrpname[m]+"="+str;if(oGrpname!="countryId"){var oObjVal=this.parent.parent.getObjValue(oGrpname);if(oObjVal)
this.bFrame=true;if((oGrpname.name=="gender")&&(oObjVal==0))
this.bFrame=true;}
var oAdditionalParam=oP.oConfig.sAdditionalParam;var bFlag=((m==sGrpname.length-1)&&(oAdditionalParam!=undefined)&&(oGrpname.name=='personalId'));if(bFlag)
{for(var k=0,len=oAdditionalParam.length;k<len;k++)
{if(oAdditionalParam[k]=='gender'){var oUserRadio=new EbayHTMLRadio(this,oAdditionalParam[k]);if(!oUserRadio.eElem)
oUserRadio.bind();var sRadioValue=oUserRadio.getValue();if(sRadioValue!="")
sParams+="&"+oAdditionalParam[k]+"="+sRadioValue;}else{var oDivSec=new EbayHTMLText(this,oAdditionalParam[k]);oDivSec.bind();if(oDivSec!=undefined&&oDivSec.eElem!=undefined){sParams+="&"+oAdditionalParam[k]+"="+oDivSec.eElem.value;}else{sParams+="&"+oAdditionalParam[k]+"="+"";}}}}}}
if(oP.oPrevObj.name=="PHONE_GROUP"){var oPhoneAdditionalParam=oP.oConfig.sPhoneAdditionalParam;if(oPhoneAdditionalParam!=undefined){var oPhoneDivSec=new EbayHTMLText(this,oPhoneAdditionalParam);oPhoneDivSec.bind();if(oPhoneDivSec!=undefined&&oPhoneDivSec.eElem!=undefined){sParams+="&"+oPhoneAdditionalParam+"="+oPhoneDivSec.eElem.value;}}}
for(x in pCfg.aMatchingfld)
{if(oP.oPrevObj&&oP.oPrevObj.name==x)
{var sVal=oP.oPrevObj.controls[pCfg.aMatchingfld[x][0]].getValue();var sVal2=oP.oPrevObj.controls[pCfg.aMatchingfld[x][1]].getValue();var oDiv=oP._getControl(pCfg.sDivPrefix+pCfg.aMatchingfld[x][1]);var oDiv1=oP._getControl(pCfg.sDivPrefix1+pCfg.aMatchingfld[x][1]);var oImgdiv=oP._getControl(pCfg.sImgDivPrefix+pCfg.aMatchingfld[x][1]);oDiv1.show();var b=(sVal==sVal2)?true:false;if(!b){var mEr=(x=='EMAIL_GROUP')?pCfg.sMthEmailEr:pCfg.sMthPassEr;oDiv.setValue(mEr);}
oImgdiv.show(!b);for(i=0,len=pCfg.sExactDiv.length;i<len;i++)
{if(oP.oPrevObj.name==pCfg.sExactDiv[i][0])
{var oDivSec=new EbayHTMLLayer(this,pCfg.sExactDiv[i][1]);if(!oDivSec.eElem)
oDivSec.bind();oDivSec.show(b);}}
var bNew=(b&&this.bFrame)?true:false;if(bNew)
{oFrame.setSource(pCfg.sUrl+sParams);}
else{b?oDiv1.show(!b):oDiv.show(!b);oImgdiv.show(!b);for(i=0,len=pCfg.sExactDiv.length;i<len;i++)
{if(oP.oPrevObj.name==pCfg.sExactDiv[i][0])
{var oDivSec=new EbayHTMLLayer(this,pCfg.sExactDiv[i][1]);if(!oDivSec.eElem)
oDivSec.bind();oDivSec.show(b);}}}
oP.oPrevtxt=this.name;oP.oPrevObj=this.parent;return false;}}
for(x in pCfg.aMatchingfld)
{if(oP.oPrevObj&&oP.oPrevObj.name!=x)
{if(this.bFrame){if(pCfg.aOptnlIframechk)
{for(var y=0,len=pCfg.aOptnlIframechk.length;y<len;y++)
{if(oP.oPrevObj&&oP.oPrevObj.name==pCfg.aOptnlIframechk[y])
{var bSet=true;break;}}
if(!bSet){oFrame.setSource(pCfg.sUrl+sParams);}
else
bSet=false;}
else{oFrame.setSource(pCfg.sUrl+sParams);}
break;}
else
{for(temp in oP.oPrevObj.controls)
{var oTempObj=oP.controls[pCfg.sDivPrefix1+oP.oPrevObj._getControl(temp).name];var oTempImg=oP.controls[pCfg.sImgDivPrefix+oP.oPrevObj._getControl(temp).name];if(ebay.oDocument.oPage._getControl("Registration.FieldCheck")&&!ebay.oDocument.oPage._getControl("Registration.FieldCheck").bFlag)
{oTempImg.show(true);oTempObj.show(true);}}}}}}
oP.oPrevtxt=this.name;oP.oPrevObj=this.parent;}
this.validateText=function()
{if(this.eElem.type=="text"||this.eElem.type=="password"||this.eElem.type=="select-one"||this.name=="gender")
{var oP=this.parent.parent;var oDiv=oP._getControl(pCfg.sDivPrefix+this.name);var oDiv1=oP._getControl(pCfg.sDivPrefix1+this.name);var oImgdiv=oP._getControl(pCfg.sImgDivPrefix+this.name);if(oP.oPrevObj==this.parent)
{for(var k=0,len=pCfg.aOptionaltxtfld.length;k<len;k++)
{if(pCfg.aOptionaltxtfld[k]==this.name)
oP.bNull=true;}}
if(!oP.bNull)
{if(this.eElem.type=="text"||this.eElem.type=="password")
var oObjVal=this.getValue();if(this.eElem.type=="select-one")
var oObjVal=this.getSelectedIndex();if(this.name=="gender")
var oObjVal=this.getSelectedIndex();oDiv.show();var b=oObjVal?false:true;if(this.name=="birthdate1"||this.name=="birthdate2"||this.name=="birthdate3"){oDiv=oP._getControl(pCfg.sDivPrefix+"birthdate2");oDiv1=oP._getControl(pCfg.sDivPrefix1+"birthdate2");oImgdiv=oP._getControl(pCfg.sImgDivPrefix+"birthdate2");var oBirthdate1=new EbayHTMLSelect(this,"birthdate1");if(!oBirthdate1.eElem)
oBirthdate1.bind();var oBirthdate1ObjVal=oBirthdate1.getSelectedIndex();var bBirthdate1Flag=oBirthdate1ObjVal?false:true;var oBirthdate2=new EbayHTMLSelect(this,"birthdate2");if(!oBirthdate2.eElem)
oBirthdate2.bind();var oBirthdate2ObjVal=oBirthdate2.getSelectedIndex();var bBirthdate2Flag=oBirthdate2ObjVal?false:true;var oBirthdate3=new EbayHTMLText(this,"birthdate3");if(!oBirthdate3.eElem)
oBirthdate3.bind();var oBirthdate3ObjVal=oBirthdate3.getValue();var bBirthdate3Flag=oBirthdate3ObjVal?false:true;if(bBirthdate1Flag||bBirthdate2Flag||bBirthdate3Flag){b=true;}else{b=false;}}else if(this.name=="dayphone1"||this.name=="dayphone2"||this.name=="dayphone3"){oDiv=oP._getControl(pCfg.sDivPrefix+"dayphone1");oDiv1=oP._getControl(pCfg.sDivPrefix1+"dayphone1");oImgdiv=oP._getControl(pCfg.sImgDivPrefix+"dayphone1");var oDayphone1=new EbayHTMLText(this,"dayphone1");if(!oDayphone1.eElem)
oDayphone1.bind();var oDayphone1ObjVal=oDayphone1.getValue();var bDayphone1Flag=oDayphone1ObjVal?false:true;var oDayphone2=new EbayHTMLText(this,"dayphone2");var bDayphone2Flag=false;if(!oDayphone2.eElem)
oDayphone2.bind();if(oDayphone2.eElem){var oDayphone2ObjVal=oDayphone2.getValue();bDayphone2Flag=oDayphone2ObjVal?false:true;}
var oDayphone3=new EbayHTMLText(this,"dayphone3");var bDayphone3Flag=false;if(!oDayphone3.eElem)
oDayphone3.bind();if(oDayphone3.eElem){var oDayphone3ObjVal=oDayphone3.getValue();bDayphone3Flag=oDayphone3ObjVal?false:true;}
if(bDayphone1Flag||bDayphone2Flag||bDayphone3Flag){b=true;}else{b=false;}}else if(this.name=="bizNumber1"||this.name=="bizNumber2"){oDiv=oP._getControl(pCfg.sDivPrefix+"bizNumber1");oDiv1=oP._getControl(pCfg.sDivPrefix1+"bizNumber1");oImgdiv=oP._getControl(pCfg.sImgDivPrefix+"bizNumber1");var oBizNumber1=new EbayHTMLText(this,"bizNumber1");if(!oBizNumber1.eElem)
oBizNumber1.bind();var oBizNumber1ObjVal=oBizNumber1.getValue();var bBizNumber1Flag=oBizNumber1ObjVal?false:true;var oBizNumber2=new EbayHTMLText(this,"bizNumber2");if(!oBizNumber2.eElem)
oBizNumber2.bind();var oBizNumber2ObjVal=oBizNumber2.getValue();var bBizNumber2Flag=oBizNumber2ObjVal?false:true;if(bBizNumber1Flag||bBizNumber2Flag){b=true;}else{b=false;}}else if(this.name=="gender"){if(oObjVal==-1){b=true;}else if(oObjVal==0){b=false;}}
if(ebay.oDocument.oPage._getControl("Registration.FieldCheck")&&!ebay.oDocument.oPage._getControl("Registration.FieldCheck").bFlag)
{oDiv1.show(b);oImgdiv.show(b);for(iIndex in pCfg.sSecErrorDivs)
{if(this.eElem.name==pCfg.sSecErrorDivs[iIndex][0])
{var sParam=pCfg.sDivPrefix1+this.eElem.name;var oQuickErrorDiv=new EbayHTMLLayer(this,sParam);var oSecDiv=new EbayHTMLLayer(this,pCfg.sSecErrorDivs[iIndex][1]);if(!oQuickErrorDiv.eElem)
oQuickErrorDiv.bind();if(!oSecDiv.eElem)
oSecDiv.bind();if(oQuickErrorDiv.eElem)
oSecDiv.show(!b);}}}}
else
oP.bNull=false;}}
this.getObjValue=function(pObj){if(pObj.eElem.type=="select-one"){return pObj.getSelectedIndex();}else if(pObj.eElem.type=="text"||pObj.eElem.type=="password"){return pObj.getValue();}else if(pObj.name=="gender"){return pObj.getSelectedIndex();}}
this.URLencode=function(sStr)
{return escape(sStr).replace(new RegExp("\'","g"),'%2B').replace(new RegExp("\'","g"),'%22').replace(new RegExp("\'","g"),'%27');}
this.processResponse=function(pConfig)
{for(var j=0,len=pConfig.aErrorResp.length;j<len;j++)
{var oDiv=this._getControl(pCfg.sDivPrefix+pConfig.aTxtbxname[j]);oDiv1=this._getControl(pCfg.sDivPrefix1+pConfig.aTxtbxname[j]);if(oDiv1)
oDiv1.show();if(oDiv)
{var oImgdiv=this._getControl(pCfg.sImgDivPrefix+pConfig.aTxtbxname[j]);var bError=pConfig.bIsError&&(pConfig.aErrorResp[j]!=" "&&pConfig.aErrorResp[j]!="");oDiv.show(bError);oImgdiv.show(bError);oDiv.setValue(pConfig.aErrorResp[j]);if(pCfg.sSecErrorDivs)
{for(iIndex in pCfg.sSecErrorDivs)
{if(pConfig.aTxtbxname[j]==pCfg.sSecErrorDivs[iIndex][0])
{var sParam=pCfg.sDivPrefix1+pConfig.aTxtbxname[j];var oQuickErrorDiv=new EbayHTMLLayer(this,sParam);var oSecDiv=new EbayHTMLLayer(this,pCfg.sSecErrorDivs[iIndex][1]);if(!oQuickErrorDiv.eElem)
oQuickErrorDiv.bind();if(!oSecDiv.eElem)
oSecDiv.bind();if(oQuickErrorDiv.eElem)
oSecDiv.show(!bError);}}}}}}
if(pCfg.sFormName)
{var eFrm=new EbayHTMLForm(this,pCfg.sFormName);var eButton=new EbayHTMLButton(eFrm,pCfg.sButtonId);var oAcceptErrorDiv=new EbayHTMLLayer(eFrm,pCfg.sDivname);oAcceptErrorDiv.bind();var oAcceptCB=new EbayHTMLCheckbox(eFrm,pCfg.sCheckId);oAcceptCB.bind();eButton.onclick=function()
{var bCheck=oAcceptCB.isChecked();if(!bCheck)
{oAcceptErrorDiv.setValue(pCfg.sChkErrorMsg);oAcceptErrorDiv.show(true);return false;}
else
{eFrm.submit();}
return false;}}
this.init();}

//33@@m1

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

//34@@m22

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

//35@@m11

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

//36@@m10

function EbayPasswordopt(pParent,pName,pCfg)
{if(!this.objType)
this.objType="EbayPasswordopt";this.base=EbayHTML;this.base(pParent,pName,pName,false,pCfg);var bLocalShowLayer=false;var bBlur=false;var bBrowserCheck=this.oGlobals.oClient;this.init=function()
{with(this)
{var oLayerParentiFrameShim=this.oLayerParentiFrameShim=new EbayHTMLOverlayContent(this,"passwordlayer",pCfg);var oOverlayboxId=this.oOverlayboxId=new EbayHTMLLayer(this,pCfg.sOverlayboxId);if(!oOverlayboxId.eElem)
oOverlayboxId.bind();var oShadowlayerId=this.oShadowlayerId=new EbayHTMLLayer(this,pCfg.sShadowlayerId);if(!oShadowlayerId.eElem)
oShadowlayerId.bind();var oRadioName=this.oRadioName=new EbayHTMLRadio(this,pCfg.sRadioName);if(!oRadioName.eElem)
oRadioName.bind();var oReTextBox=this.oReTextBox=new EbayHTMLText(this,pCfg.sRetypePassText);if(!oReTextBox.eElem)
oReTextBox.bind();var oRetypePassErrDiv=this.oRetypePassErrDiv=new EbayHTMLLayer(this,pCfg.sRetypePassErrDiv);if(!oRetypePassErrDiv.eElem)
oRetypePassErrDiv.bind();var bIsQuickWinOff=this.bIsQuickWinOff=pCfg.bIsQuickWinOff||false;var oErrorDivId=this.oErrorDivId=new EbayHTMLLayer(this,pCfg.sErrorResDivId);if(!oErrorDivId.eElem)
oErrorDivId.bind();var oErrorImgId=this.oErrorImgId=new EbayHTMLImage(this,pCfg.sErrorResImgId);if(!oErrorImgId.eElem)
oErrorImgId.bind();var oLayerParent=this.oLayerParent=new EbayHTMLLayer(this,pCfg.sPasslayerDiv);if(!oLayerParent.eElem)
oLayerParent.bind();var oPassHelpTxt=this.oPassHelpTxt=new EbayHTMLLayer(this,pCfg.sPassHelpTxt);if(!oPassHelpTxt.eElem)
oPassHelpTxt.bind();var oPassHpAnc=this.oPassHpAnc=new EbayHTMLAnchor(this,pCfg.sPassHpAnc);if(!oPassHpAnc.eElem)
oPassHpAnc.bind();var oInlinePassErImg=this.oInlinePassErImg=new EbayHTMLImage(this,pCfg.sInlinePassErImg);if(!oInlinePassErImg.eElem)
oInlinePassErImg.bind();var oRegisterBtn=this.oRegisterBtn=new EbayHTMLButton(this,pCfg.sRegisterBtnId);if(!oRegisterBtn.eElem)
oRegisterBtn.bind();var oInlineMsg=this.oInlineMsg=new EbayHTMLLayer(this,pCfg.sInlineErmsg);if(!oInlineMsg.eElem)
oInlineMsg.bind();var oFrame=this.oFrame=new EbayHTMLFrame(this,pCfg.sFrame);if(!oFrame.eElem)
oFrame.bind();var oText=this.oText=new EbayHTMLText(this,pCfg.sPassword);oText.subscribeEvents("onfocus");if(!oText.eElem)
oText.bind();var oUserTextField=this.oUserTextField=new EbayHTMLText(this,pCfg.sUserField);if(!oUserTextField.eElem)
oUserTextField.bind();var oEmailTextField=this.oEmailTextField=new EbayHTMLText(this,pCfg.sEmailField);if(!oEmailTextField.eElem)
oEmailTextField.bind();var bIframeShim=this.bIframeShim=typeof(pCfg.bIframeShim)!='undefined'?pCfg.bIframeShim:true;oText._registerEvent("onfocus","parent.onFocus");oText._registerEvent("onblur","parent.onBlur");oText._registerEvent("onkeyup","parent.onKeyUp");oRegisterBtn._registerEvent("onclick","parent.onClickRegBtn");oReTextBox._registerEvent("onblur","parent.onBlurReTextBox");}}
this.init();this.onClickRegBtn=function()
{var p=this.parent;p.oInlinePassErImg.show(false);p.oInlineMsg.setValue("");p.oInlineMsg.show(false);}
this.onBlurReTextBox=function()
{var p=this.parent;oText=p.oText;oReTextBox=p.oReTextBox;oRetypePassErrDiv=p.oRetypePassErrDiv;if(oText.getValue()==oReTextBox.getValue())
{if(p.bIsQuickWinOff)
oRetypePassErrDiv.show(false);if((oText.getValue().length<6)&&(oText.getValue().length!=0))
{p.oInlineMsg.setValue("");p.oInlineMsg.show(false);p.oInlinePassErImg.show(false);}}}
this.onKeyUp=function()
{var p=this.parent;var iLenText=p.oText.getValue().length;if(iLenText>=6)
{var evt=(window.event)?event.keyCode:arguments[2].keyCode,oText=p.oText,oUserTextField=p.oUserTextField,oEmailTextField=p.oEmailTextField,sUrl=pCfg.sUrl,sTextValue=oText.getValue(),sEncodesUrl=encodeURIComponent(sTextValue),sText1stSixChar=oText.getValue().substr(0,6);p.oInlineMsg.setValue("");p.oInlineMsg.show(false);p.oInlinePassErImg.show(false);if(sTextValue.length==6)
p.oLayerParent.sStoreTextValue=sTextValue;if((evt==46)||(evt==8)||(evt==127)||(p.oLayerParent.sStoreTextValue!=sText1stSixChar))
{bLocalShowLayer=false;}
if(oUserTextField.name!=undefined)
{if(pCfg.sRadioName)
{sUserIdName=(p.oRadioName.getValue()=="on")?oUserTextField.name:p.oRadioName.name;sUserIdValue=(p.oRadioName.getValue()=="on")?oUserTextField.getValue():encodeURIComponent(p.oRadioName.getValue());}
else
{sUserIdName=oUserTextField.name;sUserIdValue=encodeURIComponent(oUserTextField.getValue());}}
if(!bLocalShowLayer&&!(evt>=35&&evt<=40))
{if((oUserTextField.name!=undefined)&&(oEmailTextField.name!=undefined))
{sUrl+="&"+sUserIdName+"="+sUserIdValue+"&"+oEmailTextField.name+"="+encodeURIComponent(oEmailTextField.getValue())+"&"+oText.name+"="+sEncodesUrl;}
else if(oUserTextField.name!=undefined)
{sUrl+="&"+sUserIdName+"="+sUserIdValue+"&"+oText.name+"="+sEncodesUrl;}
else if(oEmailTextField.name!=undefined)
{sUrl+="&"+oEmailTextField.name+"="+encodeURIComponent(oEmailTextField.getValue())+"&"+oText.name+"="+sEncodesUrl;}
else
{sUrl+="&"+oText.name+"="+sEncodesUrl;}
p.oFrame.eElem.title=pCfg.sFrmTitle;p.oFrame.setSource(sUrl);}}
else
{oResImageIds=this.oResImageIds=[];for(i=0,len=pCfg.aResImageIds.length;i<len;i++)
{oResImageIds[i]=this.oResImageIds[i]=new EbayHTMLImage(this,pCfg.aResImageIds[i]);if(!oResImageIds[i].eElem)
oResImageIds[i].bind();oResImageIds[i].show(false);}
bLocalShowLayer=false;}}
this.onFocus=function()
{var p=this.parent;bBlur=false;if(bBrowserCheck.bFirefox)
{var sLeft=p.oText.width()+60;if(p.oInlineMsg.eElem.style.display!="none")
{sLeft+=15;}
p.oLayerParent.eElem.style.left=sLeft+'px';}
p.oLayerParent.show(true);p.oLayerParent.eElem.style.height=p.oOverlayboxId.eElem.offsetHeight+5+'px';p.oShadowlayerId.eElem.style.height=p.oOverlayboxId.eElem.offsetHeight-10+'px';if(p.bIframeShim)
{p.oLayerParentiFrameShim.setIframeShim();p.oLayerParentiFrameShim.eIframeShim.tabIndex='-1';}}
this.onBlur=function()
{var p=this.parent;if(p.bIsQuickWinOff)
{p.oErrorDivId.show(false);p.oErrorImgId.show(false);}
if(p.oText.getValue().length>=6)
p.oLayerParent.show(!bLocalShowLayer);else if(p.oText.getValue().length==0)
{p.oInlineMsg.setValue("");p.oInlineMsg.show(false);p.oInlinePassErImg.show(false);p.oLayerParent.show(false);}
else
{p.oLayerParent.show(false);p.oInlineMsg.show(true);p.oInlineMsg.setValue(pCfg.sInPassErTxt);p.oInlinePassErImg.show(true);if(p.oPassHpAnc.eElem)
p.oPassHpAnc.eElem.removeAttribute("href");p.oPassHelpTxt.setClass("hideLabel");}
bBlur=true;}
this.Response=function(cfg)
{var oImageAll;for(i=0,len=cfg.aImageIds.length;i<len;i++)
{oImageAll=this.oImageAll=new EbayHTMLImage(this,cfg.aImageIds[i]);if(!oImageAll.eElem)
oImageAll.bind();oImageAll.show(false);}
var oImage,FCnt,oFCnt,oPassHd;for(i=0,len=cfg.aImageTorF.length;i<len;i++)
{oImage=this.oImage=new EbayHTMLImage(this,cfg.aImageTorF[i]);if(!oImage.eElem)
oImage.bind();oImage.show(true);if(oImage.name.match('fail')&&pCfg)
{oPassHd=this.oPassHd=new EbayHTMLLayer(this,pCfg.sPassHd);if(!oPassHd.eElem)
oPassHd.bind();FCnt=cfg.aImageTorF[i]+pCfg.sPassHdCnt;oFCnt=this.oFCnt=new EbayHTMLLayer(this,FCnt);if(!oFCnt.eElem)
oFCnt.bind();FCnt1=cfg.aImageTorF[i]+'-cnt1';oFCnt1=this.oFCnt1=new EbayHTMLLayer(this,FCnt1);if(!oFCnt1.eElem)
oFCnt1.bind();oPassHd.setText(pCfg.sPassHdTxt);oFCnt.setText(oFCnt1.getText());}}
bLocalShowLayer=cfg.bShowLayer;if(bBlur)
{cfg.bShowLayer?this.oLayerParent.show(false):this.oLayerParent.show(true);}}}
// b=12011051 -->