
(function(){var _1=1.5,_bOldSupport=typeof(vjo)=="object";if(_bOldSupport&&vjo._v==_1){return;}
var _2={};var _3=(typeof window!="undefined")?window:this,wv=_3.vjo;if(!wv||!wv._v||wv._v<_1){_3.vjo=_2;}
var _4={};var _5=this;var _6=[];_extend(_2,{_v:1.5,loader:{async:true,load:function(){}},isArray:function(_7){if(!_7){return false;}
return(_7.constructor==Array||(typeof _7=="object"&&_7.join&&_7.splice));},getType:function(_8,_9){if(!_8){return;}
_8=_getTypeName(_8);var _a=_8.lastIndexOf("$")+1,dot=_8.lastIndexOf(".")+1,idx=_a||dot,cn=idx?_8.substring(idx):_8,_8=_a?(_8.substring(0,_a-1)+"."+cn):_8,obj=_4[_8];if(!_9&&!obj){_2.loader.load(_8);obj=_4[_8];}
if(obj){return obj.pkg[cn];}
return;},mixin:function(_b,_c){var _d=this.getType(_b);if(_d){if(_d.vj$&&_d.vj$._vjType==="mtype"){if(_d._props){throw"cannot mixin static props to an instance";}
_extend(_c,_d._protos);}}},hitch:function(_e,fn,_10){if(!_isFn(fn)){var _11=fn;fn=_e;_e=_11;}
return function(){return fn.apply((_10&&this!=window)?this:_e,arguments);};},curry:function(fn){var _13=Array.prototype.slice;var _14=_13.call(arguments,1);return function(){return fn.apply(this,_14.concat(_13.apply(arguments)));};},bind:function(ctx,fn){if(typeof fn=="string"){fn=ctx[fn];}
var _17=Array.prototype.slice;var _18=_17.call(arguments,2);return function(){return fn.apply(ctx,_18.concat(_17.apply(arguments)));};},create:function(_19,_1a){var _1b=document.createElement(_19);_1b.tagName=="TEXTAREA"?_1b.value=_1a:_1b.innerHTML=_1a;return _1b;},make:function(_1c,clz){var a=arguments,len=a.length,tp=(_isFn(clz)&&clz.vj$)?clz:this.getType(clz),_vjo={};if(len<2||!tp){throw"context and valid type are required";}
_forEach(_1c.vj$,function(val,key){if(_isFn(val)&&val.vj$&&val.vj$._vjType){this[key]=val;}},_vjo);var _21;var _22=Array.prototype.slice.call(a,2,len);return{protos:function(obj){_21=obj;return this;},endType:function(){var t=_2.ctype(),clztype=tp.vj$._vjType,rv;if(clztype==="itype"){t.satisfies(tp);}else{if(clztype==="ctype"||clztype==="atype"){var _25=_TypeMeta.get(tp);if(_25&&!_25._completed){_25.complete();}
t.inherits(_2.getType(tp.vj$._class));}else{throw"incompatible anonomyous type";}}
t.protos(_21);t.complete();t=t.endType();rv=_Type.createNoConstruct(t);rv.vj$=t.vj$;_extend(rv.vj$,_vjo);(function(){if(this.base){this.base.apply(this,arguments);}}).apply(rv,_22);if(rv.base&&rv._getBase){var fn=function(){};fn.prototype=rv._getBase();rv.base=new fn;rv.base._parent=rv;}
rv.vj$.parent=tp._outer||_1c;rv.vj$.outer=tp._outer;if(_21&&_21.constructs){_21.constructs.apply(rv);}
return rv;}};},needs:function(clz,_28){if(!clz){return;}
var _29=_createPkg(clz),cls=_29.className,tp=_29.pkg[cls];this.getType(clz);},createArray:function(val,_2b){var arr=[];if(arguments.length>1){for(var ii=0;ii<_2b;ii++){if(arguments.length>2){var tmp=[val];for(var k=2;k<arguments.length;k++){tmp[tmp.length]=arguments[k];}
arr[ii]=_2.createArray.apply(this,tmp);}else{arr[ii]=val;}}}
return arr;},isInstanceOf:function(_30,_31){if(_30===null){return false;}
var _32=_isVjoType(_31);if(!_32||(_32&&_31.vj$._vjType!="itype")){return _30 instanceof _31;}
if(!_30.vj$){return false;}
if(_isInstanceForInterface(_30.getClass(),_31)){return true;}
var id=_30.getClass().getName(),meta;for(;;){meta=_TypeMeta.getById(id);if(meta){var ihs=meta._inherits;if(ihs.length==1&&_2.getType(ihs[0])){id=ihs[0];if(_isInstanceForInterface(_2.getType(ihs[0]).clazz,_31)){return true;}}else{break;}}else{break;}}
return false;},meta:{_list:{},load:function(_35,_36){this._list[_35+"__rtti"]=_36;},get:function(_37){return this._list[_37+"__rtti"];},has:function(_38){return(this._list[_38+"__rtti"]?true:false);}}});_2.NEEDS_IMPL=function(){throw"needs implementation";};_2.Object=function(){this.vj$={_class:"vjo.Object",_vjType:"ctype",Object:_2.Object,_meta:{}};};_2.Object.prototype={_hashCode:-1,constructs:function(){},getClass:_getClazz,hashCode:function(){if(this._hashCode==-1){this._hashCode=++_2.Object._hashCounter;}
return this._hashCode;},equals:function(o){return(this===o);},toString:function(){return this.getClass().getName()+"@"+this.hashCode().toString(16);}};_extend(_2.Object,{vj$:{_class:"vjo.Object",_vjType:"ctype",Object:_2.Object,_meta:{}},_hashCounter:0,isInstance:function(o){return _2.isInstanceOf(o,this);}});_2.Class=function(clz,typ){this._name=clz,this._type=typ||"ctype",this._satisfied=[];this.vj$={_class:"vjo.Class",_vjType:"ctype",Class:_2.Class,_meta:{}};};_2.Class.prototype=new _2.Object();_extend(_2.Class.prototype,{getName:function(){var n=this._name;if(n){return n;}
return null;},getSimpleName:function(){var n=this.getName();var idx=n.lastIndexOf(".");if(idx!=-1){n=n.substring(idx+1);if(n){idx=n.lastIndexOf("$");if(idx>0){n=n.substring(idx+1);}}}
return n;},getPackageName:function(){var n=this.getName();if(n!=null){var i=n.lastIndexOf(".");if(i>=0){return n.substring(0,i);}}
return"";},isInterface:function(){return(this._type==="itype");},isInstance:function(obj){var tp=_2.getType(this.getName());if(tp){return _2.isInstanceOf(obj,tp);}
return false;},toString:function(){return(this.isInterface()?"interface ":"class ")+this.getName();},getClass:_getClazz,_rtti:null,_metaloaded:false,getMeta:function(){if(this._rtti==null){var err="Metadata Not Available Exception";if(this._metaloaded){throw err;}else{var _45=_2.meta.get(this._name);if(_45){this._rtti=new _2.TypeMetadata(_45());this._metaloaded=true;}else{throw err;}}}
return this._rtti;},getConstructors:function(){return this.getMeta().getConstructors();},getFields:function(){return this.getMeta().getFields();},getDeclaredFields:function(){return this.getMeta().getDeclaredFields();},getMethods:function(){return this.getMeta().getMethods();},getDeclaredMethods:function(){return this.getMeta().getDeclaredMethods();},getModifiers:function(){return this.getMeta().getModifiers();},getAnnotations:function(){return this.getMeta().getAnnotations();},getInterfaces:function(){return this.getMeta().getInterfaces();},getVjoType:function(){return _2.getType(this.getName());}});_extend(_2.Class,{create:function(nm,typ){if(!nm){return new _2.Class(nm,typ?typ:"ctype");}
if(!_6[nm]){_6[nm]=new _2.Class(nm,typ?typ:"ctype");}
return _6[nm];}});_extend(_2.Class,{forName:function(clz){try{var o=eval(clz);if(o&&o.clazz){return o.clazz;}}
catch(e){}
throw"Type not found for '"+clz+"'";},isInstance:function(o){return _2.isInstanceOf(o,_2.Class);},clazz:_2.Class.create("vjo.Class")});_2.Object.clazz=_2.Class.create("vjo.Object");_2.obj=function(_4b,_4c){if(!_4b){throw"Invalid type name";}
if(!_4c){return null;}
var _4d,err="Invalid type '"+_4b+"'.";try{_4d=eval(_4b);}
catch(e){throw err;}
if(!_4d){return err;}
var o=new _4d();for(var i in _4c){o[i]=_4c[i];}
return o;};_2.findType=function(_51,_52){var _53=_51;for(;;){if(_53.clazz.meta!=null&&_53.clazz.meta[_52]!=null){return _53.clazz.meta[_52];}
if(_53==_51.prototype.constructor){return;}else{_53=_51.prototype.constructor;}}};var _54=(function(){var _55={};var _56={};var _57={};var _58=[];var _59={};var _5a={};var _5b=[];function _addType(clz){if(!_55[clz]){_55[clz]=[];}}
function _execInners(clz){var ins=_59[clz];if(ins){var len=ins.length;for(var i=0;i<len;i++){var _61=ins[i];if(_61){_61.complete();}}}
_59[clz]=null;}
function _processNeeds(clz){var n=_57[clz];if(n){while(n.length>0){n.pop()();}
_57[clz]=null;}}
function _pushDep(clz,_65,_66){var arr=_55[clz];_65.push(clz);_66[clz]=true;if(!arr||arr.length===0){return;}
var len=arr.length,i=0;for(;i<len;i++){var key=arr[i];if(!_66[key]){_pushDep(key,_65,_66);}}
return;}
function _isCompleted(clz){var _6b=_TypeMeta.getById(clz);if(_6b&&_6b._completed){return true;}
return false;}
function _canCompleteInternal(clz,_6d,_6e){if(_isCompleted(_6d)){return true;}
var arr=_55[_6d];if(!arr||arr.length===0){return true;}
var len=arr.length,i=0;for(;i<len;i++){var key=arr[i];if(!_2.loader.async){if(key===clz){return false;}}else{if(!_2.getType(key)){_addIncomplete(clz,key);return false;}}
if(!_6e[key]){_6e[key]=true;if(!_canCompleteInternal(clz,key,_6e)){return false;}}}
return true;}
function _addIncomplete(clz,dep){if(!_5b[dep]){_5b[dep]=[];}
_5b[dep][_5b[dep].length]=clz;}
function _canComplete(clz){var aD=_55[clz];if(aD){var len=aD.length;if(_2.loader.async){return _canCompleteInternal(clz,clz,{});}else{var stk=_58,len2=stk.length;for(var i=0;i<len;i++){var dep=aD[i];for(var j=0;j<len2;j++){if(stk[j]===dep){return false;}}}
if(len2>0){return _canCompleteInternal(clz,clz,{});}}}
return true;}
function _complete(_7b){_processNeeds(_7b);if(_56[_7b]){_56[_7b].complete();_56[_7b]=null;}
_execInners(_7b);_5a[_7b]=true;}
return{addDep:function(clz,dep){if(!clz){return;}
_addType(clz);var aD=_55[clz];aD[aD.length]=dep;if(!_2.loader.async){var stk=_58;if(stk.length==0){stk.push(clz);}
if(stk[stk.length-1]===clz){stk.push(dep);}}},popDep:function(clz,dep){if(_2.loader.async){return;}
var stk=_58;if(stk[stk.length-1]===dep){stk.pop();}},addInner:function(clz,fn){if(!_59[clz]){_59[clz]=[];}
var ins=_59[clz];ins.push(fn);},deferNeed:function(clz,fn){var n=_57;if(!n[clz]){n[clz]=[];}
n[clz].push(fn);},register:function(clz,fn){_56[clz]=fn;},load:function(clz){if(!clz||_5a[clz]){return false;}
var _8c=_55[clz];if(!_2.loader.async){var stk=_58,len=stk.length;if(len>0&&stk[len-1]===clz){stk.pop();}}
var isC=false;if(_8c&&_canComplete(clz)){var stk=[];_pushDep(clz,stk,{});while(stk.length>0){var _8f=stk.pop();if(_isCompleted(_8f)){continue;}
_complete(_8f);}
isC=true;}
if(!_8c||_8c.length==0){_complete(clz);}
var _90=_5b[clz];if(!isC&&_90){for(var i=0,len=_90.length;i<len;i++){var itm=_90[i];if(itm==null){continue;}
if(_canComplete(itm)){_complete(itm);_90[i]=null;}}}
return true;}};})();var _93={create:function(clz,isI){isI=isI||false;if(!_isValidClz(clz)){throw"Invalid type name '"+clz+"'";}
var _96=function(){var _97=_TypeMeta.getById(_96.vj$._meta._metaId),bConstruct=!_96.__donotconstruct;if(_93.shouldAutoConstruct(this,_96,_97)){var b=_93.createNoConstruct(_96),rv=_93.construct(b,arguments);return rv||b;}
this.constructor=_96;if(_97&&!_97._completed){_97.complete();_forEach(_96.prototype,function(val,key,_9b){this[key]=val;},this);}
this.vj$=_96.vj$;var t=this.vj$._vjType;if(bConstruct&&(t=="itype"||t=="atype"||t=="mtype")){throw t+" "+this.vj$._class+" cannot be instantiated";}
_93.processInners(this,_96.vj$,_96.clazz);if(bConstruct){var val=_93.construct(this,arguments);if(val){return val;}}
return null;};_96._name="base";_96.vj$={_class:clz,_meta:{}};_96.isInstance=function(o){return _2.isInstanceOf(o,this);};return _96;},shouldAutoConstruct:function(ctx,_a0,_a1){return(!(ctx instanceof _a0)&&!_a0.__donotconstruct&&_a1&&_a1._options.autoConstruct);},createNoConstruct:function(_a2){_a2.__donotconstruct=true;var rv=new _a2();delete _a2.__donotconstruct;return rv;},needs:function(clz,_a5){if(!clz||this.vj$._meta._isInner){return this;}
var _a6=[],useAlias=false;if(typeof clz=="string"){_a6=[clz];useAlias=(_a5)?true:false;}else{if(_2.isArray(clz)){_a6=clz;}else{return this;}}
_forEach(_a6,function(val,key,obj){var cl=val,pObj=_4[val],idx=cl.lastIndexOf("."),cn=(idx>-1)?cl.substring(idx+1):cl,tp=(pObj)?pObj.pkg[cn]:null;_54.addDep(this.vj$._class,cl);if(!tp){tp=_2.getType(cl);}
_54.popDep(this.vj$._class,cl);_93.addToNameSpace(this,tp,cl,cn,_a5,useAlias);},this);return this;},addToNameSpace:function(ctx,_ac,_ad,_ae,_af,_b0){if(_2.isInstanceOf(_ac,_MType)||_af===""){return;}
if(_ac){var nm=(_b0)?_af:_ae,err=false;if(ctx.vj$[nm]&&ctx.vj$[nm]!==_ac){if(_bOldSupport){err=true;}else{throw"Name collision with type '"+nm+"' in need list.";}}
if(!err){ctx.vj$[nm]=_ac;}}else{_54.deferNeed(ctx.vj$._class,_2.curry(function(_b2,_b3,ctx){var tp=_2.getType(_b3);if(!tp||_2.isInstanceOf(tp,_MType)){return;}
if(ctx.vj$[_b2]&&ctx.vj$[_b2]!==tp){throw"Name collision with "+nm+"in need list.";}
ctx.vj$[_b2]=tp;},(_b0)?_af:_ae,_ad,ctx));}},props:function(obj,_b7){_forEach(obj,function(val,key,obj){if(!_isValidProp(key)){return;}
var o=this[key]=val;if(_93.addInner(this,o,"s_inners",key)){if(this.vj$[key]){throw"'"+key+"' in type '"+this.vj$._class+"' conflicts with needed type name";}
var _v=_createVjNS(this.vj$,key,o);o.vj$=_v;if(!this.vj$._meta._isInner){_54.addInner(this.vj$._class,_TypeMeta.get(o));var rt=this.vj$._class;_93.updateInners(rt,rt+"."+key,o,true);}}else{if(_isFn(o)&&!o._name&&!o.vj$){o._name=key;}}},this);if(obj.toString!=Object.prototype.toString){this.toString=obj.toString;}
return this;},protosHandler:function(fn,_bf){return function(){var _c0=this.base,error=false,rv,out=this.vj$.outer;_93.setBase(this,_bf);try{rv=_93.execRealFn(fn,arguments,this,_bf,out);}
catch(e){error=e;}
this.base=_c0;if(error){throw error;}
return rv;};},protos:function(obj,_c2){if(!obj){return;}
_forEach(obj,function(val,key,obj){if(key==="base"){return;}
var _c6=this.prototype[key],isType=_isVjoType(val);if(_93.isValidProto(key,val,_c6)&&!isType){this.prototype[key]=_93.protosHandler(val,this);}else{if(isType&&!this.vj$._meta._isInner){_54.addInner(this.vj$._class,_TypeMeta.get(val));var rt=this.vj$._class;_93.updateInners(rt,rt+"."+key,val,false);}
this.prototype[key]=val;}
if(!_93.addInner(this,val,"_inners",key)){if(_isFn(val)&&!val._name&&!isType){val._name=key;}}else{if(this.vj$[key]){throw"'"+key+"' in type '"+this.vj$._class+"' conflicts with needed type name";}}},this);if(obj.toString!=Object.prototype.toString){this.prototype.toString=obj.toString;}
return this;},isValidProto:function(key,val,_ca){return(_isFn(_ca)&&key.indexOf("constructs")!=0&&(key.indexOf("_ovld")===-1||key.indexOf("_ovld")!=(key.length-5))&&!_isVjoType(_ca)&&_isFn(val));},satisfies:function(_cb,_cc){var _cd=[];if(_2.isArray(_cb)){_cd=_cb;}else{_cd=[_cb];}
_forEach(_cd,function(val,key,obj){var cl=_getTypeName(val),_cb;if(_isVjoType(cl)){_cb=cl;var clz=_cb.vj$._class||"",idx=clz.lastIndexOf("."),cn=(idx>-1)?clz.substring(idx+1):clz;if(cn){this.vj$[cn]=_cb;}}
var _d3=(_cb)?_cb:_2.getType(cl);if(_d3){if(!_cc){this.clazz._satisfied.push(_d3);}
for(var i in _d3){var val=_d3[i];if(_isValidProp(i)&&!this[i]){this[i]=val;}}}},this);return this;},inherits:function(_d5,isB){_d5=_getTypeName(_d5);if(!isB&&!_isValidInh(_d5)){throw"Cannot inherit from '"+_d5+"'";}
var _d7=("vjo.Object"===_d5);if(!isB&&_d7){return this;}
var _d8;if(_isVjoType(_d5)){_d8=_d5;var clz=_d8.vj$._class||"",idx=clz.lastIndexOf("."),cn=(idx>-1)?clz.substring(idx+1):clz;if(cn){this.vj$[cn]=_d8;}}else{_d8=(_d7)?_2.Object:this.vj$[_d5]||_2.getType(_d5);}
if(_isVjoType(_d8)){var _da=_TypeMeta.get(_d8);if(_da&&!_da._completed){_da.complete();}
_93.createInherits(this,_d8);}
return this;},setBase:function(ctx,_dc){ctx.base=(_dc.prototype._getBase)?_dc.prototype._getBase():null;if(ctx.base){ctx.base._parent=ctx;}},createInherits:function(_dd,_de){var _df=_de.prototype,cls=_93.createNoConstruct(_de);cls.constructs=null;cls.constructor=_dd;var _e0={};if(!(_de.vj$._class=="vjo.Object")){_e0.vj$=_de.vj$;}
if(_df._getBase&&_df._getBase()._constructs){_e0._constructs=true;}
cls.base=function(){var _e1=this.base,_df=_de.prototype,gb=_df._getBase,c=_df.constructs;if(_df.base){this.base=_df.base;}
var _e2=(c)?c.toString():"",b=(_e2.indexOf("this.base(")===-1&&_e2.indexOf("this.constructs")===-1);if(gb&&gb()._constructs&&b){this.base();}
if(_df.constructs){var _e3=this.vj$._meta._isInner;try{_93.execRealFn(_df.constructs,arguments,this,_de,_e3);}
catch(e){throw e;}}
this.base=_e1;};cls._getBase=function(){return _e0;};function createBaseRef(_e4,fn,der){return function(){var scp=(this._parent)?this._parent:this,rv,error=false,cbase=scp.base;_93.setBase(scp,_e4);try{rv=_93.execRealFn(fn,arguments,scp,_e4);}
catch(e){error=e;}
scp.base=cbase;if(error){throw error;}
return rv;};}
function hasBaseCall(fn,key){var _ea=fn.toString();return(_ea.indexOf("this.base."+key+"(")!=-1);}
var vO=_2.Object,vOP=vO.prototype,bIsO=(vOP===_df);for(var i in _df){if(i==="toString"){continue;}
var pt=_df[i],bFn=_isFn(pt);if(i==="constructs"&&bFn){if(_de!==vO&&pt.length===0){_e0._constructs=true;}}else{if(_df==vOP||_isValidInst(i)){if(bFn&&!(_2.isInstanceOf(pt,RegExp))&&!pt.vj$&&(bIsO||vOP[i]!==pt)){_e0[i]=createBaseRef(_de,pt,_dd);if(!pt.__isChained){var _ee=pt.toString();cls[i]=hasBaseCall(pt,i)?(function(fn){return function(){return fn.apply(this,arguments);};})(_e0[i]):(function(_f0,fn){return function(){try{if(this.vj$._vjType==="etype"){return _93.execRealFn(fn,arguments,this,_f0,true);}else{return _93.execRealFn(fn,arguments,this,_f0);}}
catch(e){throw e;}};})(_de,pt);cls[i].__isChained=true;}}else{if(_isVjoType(pt)&&pt.vj$._meta._isInner){var _f2=_dd.vj$._meta;if(!_f2._inners){_f2._inners={};}
_f2._inners[i]=pt;}
cls[i]=pt;}}}}
if(_e0.toString!=_2.Object.prototype.toString){_e0.toString=createBaseRef(_de,_df.toString,_dd);}
_dd.prototype=cls;},updateInners:function(_f3,_f4,_f5,_f6){if(_f5&&_f5.vj$){var vj=_f5.vj$,clz=_f5.clazz;vj._class=_f4;var idx=_f4.lastIndexOf("."),snm=_f4.substring(idx+1);vj[snm]=_f5;if(clz&&_f3){if(_f4.indexOf(_f3)==0){var tmp=_f4.replace(_f3,"");while(tmp.indexOf(".")>-1){tmp=tmp.replace(".","$");}
clz._name=_f3+tmp;}else{clz._name=_f3+"$"+snm;}}
_createPkg(_f4,true).pkg[snm]=_f5;var ins=(_f6)?vj._meta.s_inners:vj._meta._inners;if(ins){_forEach(ins,function(val,key){_forEach(vj,function(val,key){if(!this[key]&&val&&val.vj$){this[key]=val;}},val.vj$);var m=_TypeMeta.getById(val.vj$._meta._metaId);if(val.vj$&&m){_54.addInner(_f3,m);}
_93.updateInners(_f3,_f4+"."+key,val,_f6);});}}},addInner:function(clz,_101,_102,key){if(!clz||!_101||!key){return false;}
var vj=_101.vj$;if(!vj){return false;}
if(_isVjoType(_101)&&vj._meta._isInner){var cvj=clz.vj$;if(!vj._class&&cvj._class){var cn=vj._class=cvj._class+"."+key;if(_101.clazz){_101.clazz._name=cn;_6[cn]=_101.clazz;}
_createPkg(cn,true).pkg[key]=_101;}
if(_102){if(!cvj._meta[_102]){cvj._meta[_102]={};}
cvj._meta[_102][key]=_101;}
return true;}
return false;},execRealFn:function(fn,args,ctx,base,_10b){var _10c=false,rv,t={vj$:ctx.vj$};if(!_10b){_fixScope((ctx.base&&ctx.base.vj$&&base.vj$._class==ctx.base.vj$._class)?ctx.base:base,ctx);}
try{rv=fn.apply(ctx,args);}
catch(e){_10c=e;}
if(!_10b){_fixScope(t,ctx);}
if(_10c){throw _10c;}
return rv;},createClazz:function(typ){var old=typ.clazz,nm=typ.vj$._class;if(typ.clazz&&typ.clazz._name){nm=typ.clazz._name;}
typ.clazz=_2.Class.create(nm,typ.vj$._vjType);if(old){typ.clazz._satisfied=old._satisfied;}
if(typ.prototype){typ.prototype.getClass=_getClazz;}else{typ.getClass=_getClazz;}
typ.vj$.type=typ;},canCallBase:function(obj){var str=obj.constructs.toString();return obj.base&&str.indexOf("this.base(")===-1&&str.indexOf("this.constructs")===-1;},tmpFunc:function(){},construct:function(ctx,args){var c=ctx.constructs;var fn,rv,_116=false;if(ctx.base&&ctx._getBase){fn=_93.tmpFunc;fn.prototype=ctx._getBase();_116=fn.prototype._constructs||false;}
if(_116&&(!c||_93.canCallBase(ctx))){ctx.base();}
if(c){rv=c.apply(ctx,args);}
if(fn){ctx.base=new fn;ctx.base._parent=ctx;}
if(rv){return rv;}},processInners:function(_117,_118,_119){var _11a=(_118)?_118._meta._inners:null;if(!_118||!_11a||_11a.length==0||!_117){return;}
for(var k in _11a){_117[k]=_2.curry(function(t,type,nm){var cn=t.vj$._class+"."+nm;var m=_TypeMeta.get(type.vj$._meta._metaId);if(m){m.complete();}
var tp=_93.createNoConstruct(type);var _v=_createVjNS(_118,nm,type);tp.vj$=_v;tp.vj$.outer=t;_93.processInners(tp,_v,type.clazz);type.vj$._class=tp.vj$._class=cn;var s=type.prototype;if(type.clazz&&!type.clazz._name){type.clazz._name=tp.vj$._class;_6[cn]=type.clazz;}else{_93.createClazz(type);}
_93.construct(tp,Array.prototype.slice.call(arguments,3));if(tp.vj$.outer&&tp.base.vj$){var _v=_createVjNS(tp.vj$);_extend(_v,tp.base.vj$);tp.base.vj$=_v;}
return tp;},_117,_11a[k],k);_117[k]._outer=_117;_117[k].vj$=_11a[k].vj$;}}};var _124=(typeof console!="undefined");_extend(_2,{sysout:{print:function(){if(_124){console.info.apply(this,arguments);}},println:function(){if(_124){console.info.apply(this,arguments);}},printStackTrace:function(){}},syserr:{print:function(){if(_124){console.warn.apply(this,arguments);}},println:function(){if(_124){console.warn.apply(this,arguments);}},printStackTrace:function(){}},jsunit:{assertEquals:function(){},assertTrue:function(){},assertFalse:function(){},assertNotNull:function(){}}});function _TypeMeta(name,kind,cfg){this._needs=[];this._props=null;this._protos=null;this._satisfies=[];this._mixins=[];this._inherits=[];this._inits=null;this._expects=[];this._completed=(kind==="type")?true:false;this._isInner=(name)?false:true;this._name=name;this._kind=kind;this._options={autoConstruct:true};this.init(cfg);this.setup();}
_TypeMeta.prototype={init:function(cfg){this._cfg={satisfiesFn:_93.satisfies,inheritsFn:_93.inherits,protosFn:_93.protos,postDefFn:function(){},typeDef:null,baseType:"vjo.Object"};if(cfg){_extend(this._cfg,cfg);}},setup:function(){var t=this._type=this._cfg.typeDef||_93.create(this._name,"itype"===this._kind);t.vj$._vjType=this._kind||"ctype";var id=this._name;if(this._isInner){id=_TypeMeta.id();t.vj$._meta._isInner=true;}else{var pObj=_createPkg(id);if(!pObj.pkg[pObj.className]){pObj.pkg[pObj.className]=this._type;}
t.vj$[pObj.className]=t;_54.register(id,this);}
this._isDup=(!this._isInner&&_TypeMeta.getById(id)!=null);if(!this._isDup){_TypeMeta.put(this,id);}
t._inherits=null;_93.createClazz(t);t.vj$._meta._metaId=id;},needs:function(need,_12d){_93.needs.apply(this._type,arguments);return this;},singleton:function(){return this;},options:function(opts){if(opts){for(var k in opts){this._options[k]=opts[k];}}
return this;},makeFinal:function(){return this;},satisfies:function(type){var clzs=[];if(_2.isArray(type)){clzs=type;}else{clzs=[type];}
_forEach(clzs,function(val,key,obj){var _135=_getTypeName(val);this.needs(_135);_93.needs.call(this._type,_135);this._satisfies.push(val);},this);return this;},props:function(_136){if(this._props){throw"multiple props blocks are not allowed";}
this._props=_136;_93.props.apply(this._type,arguments);return this;},protos:function(_137){if(this._protos){throw"multiple protos blocks are not allowed";}
this._protos=_137;return this;},inherits:function(type){var clzs=[];if(_2.isArray(type)){clzs=type;}else{clzs=[type];}
_forEach(clzs,function(val,key,obj){var _13d=_getTypeName(val);this.needs(_13d);this._inherits.push(val);},this);return this;},mixin:function(type){var clzs=[];if(_2.isArray(type)){clzs=type;}else{clzs=[type];}
_forEach(clzs,function(val,key,obj){var _143=_getTypeName(val);this.needs(_143);this._mixins.push(val);},this);return this;},inits:function(fn){this._inits=fn;return this;},validateAndMerge:function(_145,_146){var p=(_146)?_145[_146]:_145;return function(val,key,obj){if(p[key]){throw"collision when mixing in '"+key+"' into "+this._name;}else{p[key]=val;}};},mergeMixins:function(_14b){var mxns=this._mixins;for(var i=0;i<mxns.length;i++){var m=_2.getType(mxns[i]);if(!m||!m.vj$||m.vj$._vjType!="mtype"){throw mxns[i]+"is not a valid mtype.";}
var exp=m._expects;sats=m._satisfiers;if(!this._protos){this._protos={};}
if(!this._props){this._props={};}
_copyNS(m.vj$,this._type.vj$);var nm=m.vj$._class,idx=nm.lastIndexOf(".");var clz=(idx!=-1)?nm.substring(idx+1):nm;if(!this._type.vj$[clz]){this._type.vj$[clz]=this._type;}else{throw clz+" is already defined in the current namespace";}
_forEach(m._protos,this.validateAndMerge(this,"_protos"),this);_forEach(m._props,this.validateAndMerge(_14b,"_props"),this);for(var j=0;j<sats.length;j++){this._satisfies.push(sats[i]);}
if(exp){this._expects.push(exp);}}},completeSatisfies:function(){if(this._satisfies.length>0){this._cfg.satisfiesFn.call(this._type,this._satisfies);}
if(this._expects.length>0){this._cfg.satisfiesFn.call(this._type,this._expects,true);}},completeInherits:function(){var ilen=this._inherits.length;if(ilen>0){if(this._kind!="itype"&&ilen>1){throw"type can only inherit from one type";}
for(var i=0;i<ilen;i++){this._cfg.inheritsFn.call(this._type,this._inherits[i]);}}else{if(this._kind!="itype"){this._cfg.inheritsFn.call(this._type,this._cfg.baseType,true);}}
if(ilen==0){var _155=this._protos||{};_forEach(["hashCode","equals","getClass"],function(val){if(!_155[val]){this[val]=_2.Object.prototype[val];}},this._type.prototype);}},completeDef:function(){if(this._protos){this._cfg.protosFn.call(this._type,this._protos);}
if(!this._type.prototype.constructs){this._type.prototype.constructs=function(){};}
this._cfg.postDefFn.call(this);if(this._inits&&!this._isDup){this._inits.call(this._type);}},complete:function(){if(this._completed){return this;}
this._completed=true;_updateInnerEtypes(this._type.vj$);var p={_props:{}};this.mergeMixins(p);_93.props.call(this._type,p._props);this.completeSatisfies();this.completeInherits();this.completeDef();return this;},typesAvail:function(list){for(var i=0;i<list.length;i++){if(!_isVjoType(list[i])){return false;}}
return true;},canComplete:function(){var b=(this._inits==null)&&this.typesAvail(this._inherits);if(b){b=this.typesAvail(this._satisfies);}
if(b){b=this.typesAvail(this._mixins);}
return b;},endType:function(){if(!this._isInner){_54.load(this._name);if(_2.validateType){_2.validateType(this._type);}}else{if(this.canComplete()){this.complete();}}
return this._type;}};_extend(_TypeMeta,{_count:0,_pre:"tmp",_reg:{},id:function(){return this._pre+this._count++;},put:function(meta,id){var nm=(id)?id:this._pre+this._count++;this._reg[nm]=meta;},get:function(type){var _15f;if(_isVjoType(type)){return this._reg[type.vj$._meta._metaId];}else{if(_15f=_2.getType(type)){return this._reg[_15f.vj$._meta._metaId];}else{return this._reg[type];}}},getById:function(id){return this._reg[id];}});_2.ctype=function(clz){clz=_getTypeName(clz);var t=new _TypeMeta(clz);return t;};_2.type=function(clz){clz=_getTypeName(clz);var t=new _TypeMeta(clz,"type");t.inits=function(fn){if(fn&&!this._isDup){fn.call(this._type);}
return this;};t.props=function(_166){_93.props.apply(this._type,arguments);return this;};t.protos=function(_167){_93.protos.apply(this._type,arguments);return this;};t.inherits=function(clz){_93.inherits.apply(this._type,arguments);return this;};t.satisfies=function(clz){_93.satisfies.apply(this._type,arguments);return this;};return t;};_2.itype=function(clz){clz=_getTypeName(clz);var t=new _TypeMeta(clz,"itype",{inheritsFn:function(_16c){var type=(this.vj$[_16c])?this.vj$[_16c]:(this.vj$.b&&this.vj$.b[_16c])?this.vj$.b[_16c]:_2.getType(_16c);if(type){for(var i in type){var val=type[i];if(_isValidProp(i)&&!this[i]){this[i]=val;}}}
return this;}});t._type.isInstance=function(obj){return _2.isInstanceOf(obj,this);};return t;};_2.atype=_2.ctype;function _MType(clz){function _addMixinMethods(to,_173,ns){if(!_173||typeof _173!="object"){return;}
var b=true;for(var i in _173){b=false;if(!reservedMProp[i]){to[i]=_173[i];}}
return b;}
var t=this;t.vj$={_vjType:"mtype",_class:clz,_meta:{}};t._props=null;t._protos={};t._expects="";t._satisfiers=[];t.needs=function(){return _93.needs.apply(this,arguments);};t.props=function(_178){var p=this._props||{};if(!_addMixinMethods(p,_178,this.vj$)){if(!this._props){this._props=p;}}
return this;};t.protos=function(_17a){if(_17a&&_17a["constructs"]){throw"mtype cannot have constructs block";}
_addMixinMethods(this._protos,_17a,this.vj$);return this;};t.expects=function(clz){this._expects=_2.getType(clz);return this;};t.satisfies=function(clz){var clzs=[];if(_2.isArray(clz)){clzs=clz;}else{clzs=[clz];}
_forEach(clzs,function(val,key,obj){var _181=_getTypeName(val);this.needs(_181);this._satisfiers.push(val);},this);return this;};t.endType=function(){if(this.vj$._class){_54.load(this.vj$._class);}
return this;};return t;}
_2.mtype=function(clz){clz=_getTypeName(clz);var base=new _MType(clz);if(!clz||_isInnerClass(clz)){base.vj$._meta._isInner=true;}
if(!clz){return base;}
var pObj=_createPkg(clz);return(pObj.pkg[pObj.className])?base:(pObj.pkg[pObj.className]=base);};_2.etype=function(clz){clz=_getTypeName(clz);_createEnum();var eDef=function(args){this.vj$=eDef.vj$;if(args!=false){if(!this.constructs){throw"'"+this.vj$._class+"' cannot be instantiated";}
var rv=this.constructs.apply(this,args);}};eDef.vj$={_class:clz,_vjType:"etype",_meta:{}};eDef.isInstance=function(o){return _2.isInstanceOf(o,this);};var t=new _TypeMeta(clz,"etype",{typeDef:eDef,baseType:"vjo.Enum",postDefFn:function(){this._type.prototype.toString=_2.Enum.prototype.name;if(this._eVals){this._type.values.call(this._type,this._eVals);}}});_extend(t,{inherits:function(){throw"Invalid type definition. etype cannot be inheritted from another type";},values:function(vals){this._eVals=vals;return this;}});var en=t._type;en._enums=[];en.from=_2.Enum.from;en._nativeValueOf=en.valueOf;en.valueOf=_2.Enum.from;en.values=function(vals){if(arguments.length==0){if(this._enums.slice){return this._enums.slice();}else{var a=[];for(var i=0;i<this._enums.length;i++){if(this._enums[i]){a[i]=this._enums[i];}}
return a;}}else{var ord=0;if(typeof vals=="string"&&vals.length>0){while(vals.indexOf(" ")>-1){vals=vals.replace(" ","");}
if(vals.indexOf(",")>0){var a=vals.split(","),t;if(a[0]&&a[0].indexOf(":")>0){throw"Invalid labels for etype values: "+a[0];}
for(var i=0,l=a.length;i<l;i++){var eV=a[i];if(i==0&&t&&t.length>0){eV=t[0];}else{if(a[i].indexOf(":")>-1){eV=a[i].split(":")[0];}}
this._enums[this._enums.length]=new en(false);this._enums[this._enums.length-1]._name=eV;}}else{this._enums[0]=new en(false);this._enums[0]._name=vals;}}else{for(var itm in vals){this._enums[this._enums.length]=new en(vals[itm]);this._enums[this._enums.length-1]._name=itm;}}
for(var i=0,l=this._enums.length;i<l;i++){if(this._enums[i]){var nm=this._enums[i]._name;if(this[nm]){throw"Invalid prop member. Cannot use etype value as prop member.";}
this[nm]=this._enums[i];this[nm]._ord=ord++;}}}
en.prototype.constructs=null;return this;};return t;};_2.otype=function(clz){return{defs:function(defs){return this;},endType:function(){}};};var _196={},reservedProto={},reservedMProp={},reservedClz={},reservedInh={};_forEach("props protos inherits prototype inits satisfies mixin _inherits _satisfiers singleton isInstance vj$".split(" "),function(val,key,obj){this[val]=true;},_196);_forEach("constructs getClass _getBase base vj$".split(" "),function(val,key,obj){this[val]=true;},reservedProto);_forEach("props protos _props _protos vj$ _expects expects _satisfiers satisfies endType".split(" "),function(val,key,obj){this[val]=true;},reservedMProp);_forEach("vjo.Class vjo.Object".split(" "),function(val,key,obj){this[val]=true;},reservedClz);function _isVjoType(clz,_1a4){if(clz&&clz.vj$&&clz.vj$._vjType){if(_isFn(clz)){return true;}else{if(clz.vj$._vjType==="mtype"&&!_1a4){return true;}}}
return false;}
function _isInstanceForInterface(_1a5,_1a6){var clz=_1a5,arr=clz._satisfied;for(var i=0;i<arr.length;i++){if(_isInterfaceInstanceOf(_1a6,arr[i])){return true;}}
return false;}
function _isInterfaceInstanceOf(src,_1aa){if(src===_1aa){return true;}
var meta=_TypeMeta.get(_1aa.vj$._meta._metaId),inhs;if(meta&&(inhs=meta._inherits)){for(var i=0;i<inhs.length;i++){var _1ad=_getTypeName(inhs[i]);if(src===_2.getType(_1ad)){return true;}}}
return false;}
function _createPkg(_1ae,_1af){if(!_1ae){return null;}
if(_4[_1ae]){return _4[_1ae];}
var _1b0=_1ae.split("."),len=_1b0.length;if(_1af){_4[_1ae]={pkg:{className:_1b0[len-1]}};}else{var pkg=(_1b0[0]=="vjo")?_5:this;for(var i=0;i<len-1&&pkg&&_1b0[i];i++){pkg=(pkg[_1b0[i]])?pkg[_1b0[i]]:pkg[_1b0[i]]={};}
_4[_1ae]={pkg:pkg,className:(len>0)?_1b0[len-1]:""};}
return _4[_1ae];}
function _createEnum(){if(_isFn(_2.Enum)){return;}
var nm="vjo.Enum";var _1b4=_2.ctype(nm).props({from:function(){if(arguments.length==0||arguments.length>2){return this._nativeValueOf.apply(this,arguments);}
var s=arguments[0];if(arguments.length==2){s=arguments[1];if(s){var clz=arguments[0];try{var n=clz.getName();while(n.indexOf("$")>0){n=n.replace("$",".");}
var o=eval(n);if(o[s]){return o[s];}}
catch(a){}}
throw"No enum const "+arguments[0].getName()+"."+s;}else{if(this[s]){return this[s];}}
throw"No enum const "+s;}}).protos({_name:null,_ord:-1,constructs:function(){throw"cannot instantiate 'vjo.Enum'";},name:function(){return this._name;},ordinal:function(){return this._ord;},compareTo:function(o){if(o==null){throw"compare to Etype value cannot be null";}
return(this.ordinal()-o.ordinal());},equals:function(o){return(this===o);},getDeclaringClass:function(){var _1bb=this.getClass();return _1bb;}}).endType();_1b4._nativeValueOf=_1b4.valueOf;_1b4.valueOf=_2.Enum.from;reservedClz[nm]=true;reservedInh[nm]=true;}
function _getClazz(){if(this.vj$._vjType==="ctype"&&!this.vj$._meta._isInner&&this.constructor.clazz){var n=clz=this.constructor.clazz._name,idx=n.lastIndexOf(".");if(idx!=-1){clz=n.substring(idx+1);}
if(this.constructor.vj$[clz]){return this.constructor.vj$[clz].clazz;}}else{var n=clz=this.vj$._class,idx=n.lastIndexOf(".");if(idx!=-1){clz=n.substring(idx+1);}
if(this.vj$[clz]){return this.vj$[clz].clazz;}}
return null;}
function _updateInnerEtypes(_1bd){if(!_1bd._class){return;}
var _1be=_1bd._meta;if(_1be.s_inners){for(var k in _1be.s_inners){if(_1be.s_inners[k].vj$._vjType=="etype"){for(var i=0;i<_1be.s_inners[k]._enums.length;i++){_1be.s_inners[k]._enums[i].vj$=_1be.s_inners[k].vj$;_updateInnerEtypes(_1be.s_inners[k]._enums[i].vj$);}}
_updateInnerEtypes(_1be.s_inners[k].vj$);}}
if(_1be._inners){for(var k in _1be._inners){if(_1be._inners[k].vj$._vjType=="etype"){if(!_1be._inners[k].vj$._class){_1be._inners[k].clazz._name=_1be._inners[k].vj$._class=_1be._class+"."+k;}
for(var i=0;i<_1be._inners[k]._enums.length;i++){_1be._inners[k]._enums[i].vj$=_1be._inners[k].vj$;_updateInnerEtypes(_1be._inners[k]._enums[i].vj$);}}
_updateInnerEtypes(_1be._inners[k].vj$);}}}
function _hasCollisionWithMixin(type,name,_1c3){var mxns=type.vj$._meta.mixins;if(!mxns||mxns.length==0){return false;}
for(var i=0;i<mxns.length;i++){var mxn=mxns[i];if(_1c3){if(mxn._props&&mxn._props[name]){return true;}}else{if(mxn._protos[name]){return true;}}}
return false;}
function _isValidInst(_1c7){return!(reservedProto[_1c7]===true);}
function _isValidClz(_1c8){return!(reservedClz[_1c8]===true);}
function _isValidInh(_1c9){return!(reservedInh[_1c9]===true);}
function _extend(_1ca,_1cb){for(var name in _1cb){var copy=_1cb[name];if(copy!==undefined){_1ca[name]=copy;}}
if(_1cb.toString!=Object.prototype.toString){_1ca.toString=_1cb.toString;}}
function _forEach(_1ce,_1cf,_1d0){if(!_1ce){return;}
var name,i=0,len=_1ce.length;if(!_2.isArray(_1ce)){for(name in _1ce){if(_1cf.call(_1d0,_1ce[name],name,_1ce)===false){break;}}}else{for(;i<len;i++){if(_1cf.call(_1d0,_1ce[i],i,_1ce)===false){break;}}}
return _1ce;}
function _createVjNS(ns,name,type){var rv={};_extend(rv,ns);if(name&&type){delete rv._meta;_extend(rv,type.vj$);rv[name]=type;}
return rv;}
function _copyNS(from,to){_forEach(from,function(val,key){if(key!="type"&&_isVjoType(val,true)){if(this[key]&&this[key]!==val){throw key+" is already defined in the current namespace";}
this[key]=val;}},to);}
function _isInnerClass(clz){if(!clz){return true;}else{if(clz.indexOf(".")==-1){return false;}}
var tp=clz;while((i=tp.lastIndexOf("."))>0){tp=tp.substring(0,i);if(_4[tp]){return true;}}
return false;}
function _isFn(fn){return typeof fn=="function";}
function _isValidProp(pVal){return!(_196[pVal]===true);}
function _fixScope(from,to){to.vj$=from.vj$;}
function _getTypeName(name){if(typeof name!="string"){return name;}
var clz=name;if(name){var idx=name.indexOf(" "),idx2=name.indexOf("<");if(idx>0){if(idx2>0){idx=Math.min(idx,idx2);}}else{idx=idx2;}
if(idx>0){clz=name.substring(0,idx);}}
return clz;}}).apply(this);
vjo.ctype("vjo.dsf.error.Error").protos({constructs:function(_1,_2,_3){this.message=_1;this.url=_2;this.lineNumber=_3;this.userAgent=navigator.userAgent;}}).endType();
vjo.ctype("vjo.dsf.error.ErrorHandlerManager").needs("vjo.dsf.error.Error").props({dsfErrors:[],handlers:[],add:function(_1,_2,_3){this.dsfErrors[this.dsfErrors.length]=new vjo.dsf.error.Error(_1,_2,_3);return true;},convertErrorTo:function(_4){var _5=_4?"<br>":"\n";var _6=_4?"<html><title>":"";var _7=_4?"</title><body>":_5;var _8=_4?"</body></html>":"";var _9=_4?"<h2>":"";var _a=_4?"</h2>"+_5:_5;var _b=_4?"<b>":"";var _c=_4?"</b>":"";var _d=_6+"V4 JS Errors"+_7;_d+=_9+"Total number of errors = "+this.dsfErrors.length+_a;for(var i=0;i<this.dsfErrors.length;i++){_d+=_b+"Message: "+_c+this.dsfErrors[i].message;_d+=_5;_d+=_b+"URL :"+_c+this.dsfErrors[i].url;_d+=_5;_d+=_b+"Line Number :"+_c+this.dsfErrors[i].lineNumber;_d+=_5;_d+=_b+"User Agent :"+_c+this.dsfErrors[i].userAgent;_d+=_5;_d+=_5;}
_d+=_8;return _d;},toHTML:function(){return this.convertErrorTo(true);},toText:function(){return this.convertErrorTo(false);},show:function(){var _f=window.open("","V4JSErrors","height=500,width=500,menubar=no,status=no,scrollbars=yes");if(window.focus){_f.focus();}
_f.document.write(this.toHTML());_f.document.close();},register:function(_10){if(!_10){return this;}
if(typeof _10!="object"){return this;}
this.handlers[this.handlers.length]=_10;return _10;},clear:function(){this.handlers.length=0;},process:function(_11,_12,_13){for(var i=0;i<this.handlers.length;i++){this.handlers[i].handle(_11,_12,_13);}},enableOnError:function(){window.onerror=function(_15,url,_17){vjo.dsf.error.ErrorHandlerManager.process(_15,url,_17);return true;};}}).endType();
vjo.ctype("vjo.dsf.error.DefaultErrorHandler").needs("vjo.dsf.error.ErrorHandlerManager").protos({constructs:function(){},handle:function(_1,_2,_3){vjo.dsf.error.ErrorHandlerManager.add(_1,_2,_3);}}).endType();
vjo.ctype("vjo.dsf.Event").protos({src:null,eventType:null,nativeEvent:null,constructs:function(_1,_2,_3){this.src=_1;this.eventType=_2;this.nativeEvent=_3;}}).endType();
vjo.ctype("vjo.dsf.Element").props({get:function(_1){var d=document,e=null;if(typeof(d.getElementById)!="undefined"){e=d.getElementById(_1);}else{if(!e&&d.all){e=d.all[_1];}}
return e;},toggleHideShow:function(_3,_4){var e=this.get(_3),s,d,u="undefined";if(e){s=e.style;d=s.display;if(typeof(_4)===u){_4=(d===""||d==="block")?false:true;}
e.bIsShown=_4;s.display=(_4)?"block":"none";}},promoteToBody:function(_6){var e=this.get(_6),b=document.body;if(e&&b&&e.parentNode&&(e.parentNode!==b)){e.parentNode.removeChild(e);b.appendChild(e);}},toggleVisibility:function(_8,_9){var e=this.get(_8),v,s,u="undefined";if(e){s=e.style;v=s.visibility;if(typeof(_9)===u){_9=(v==="")?false:true;}
e.bIsVisible=_9;s.visibility=(_9)?"":"hidden";}},enable:function(_b,_c){var e=this.get(_b);if(e){e.disabled=!_c;}},left:function(_e,_f){return this.setLTWH(_e,_f,"Left");},top:function(_10,_11){return this.setLTWH(_10,_11,"Top");},width:function(_12,_13){return this.setLTWH(_12,_13,"Width");},height:function(_14,_15){return this.setLTWH(_14,_15,"Height");},setLTWH:function(_16,_17,_18){var e=this.get(_16);if(e){if((_17!=null)&&!isNaN(parseInt(_17))){e.style[_18.toLowerCase()]=_17;}
return e["offset"+_18];}},createElement:function(_1a){return document.standardCreateElement?document.standardCreateElement(_1a):document.createElement(_1a);},containsElement:function(_1b,_1c){while((_1c!=null)&&(_1c!=_1b)&&(_1c.parentNode!=null)){_1c=_1c.parentNode;}
return(_1c==_1b);},getElementByTagClass:function(_1d,tag,_1f){var _20=_1d.getElementsByTagName(tag);for(var ndx=0;((ndx<_20.length)&&(_20[ndx].className.match(_1f)==null));ndx++){}
return(ndx<_20.length)?_20[ndx]:null;},getElementsByTagClass:function(_22,tag,_24){var _25=new Array();var _26=_22.getElementsByTagName(tag);for(var ndx=0;(ndx<_26.length);ndx++){if(_26[ndx].className.match(_24)){_25.push(_26[ndx]);}}
return _25;},toggleHideShowRow:function(_28,_29){var e=this.get(_28),ua=navigator.userAgent.toLowerCase(),s,d,u="undefined",p=(ua.indexOf("firefox")!=-1)?"table-row":"block";if(e){s=e.style;d=s.display;if(typeof(_29)===u){_29=(d===""||d===p)?false:true;}
e.bIsShown=_29;s.display=(_29)?p:"none";}}}).endType();
vjo.ctype("vjo.dsf.EventDispatcher").needs(["vjo.dsf.Event","vjo.dsf.Element"]).props({eventHandlers:{},nativeEventHandlers:{},unboundElems:[],fCustomLoad:{},process:function(_1,_2){var _3=this.eventHandlers[_1];if(!_3){return true;}
var _4=_3[_2.eventType];if(!_4){return true;}
var _5,_6=_4.length;for(var i=0;i<_6;i++){var _8=_4[i].handle(_2);if(_8&&_8.objType=="dsf_Message"){if(vjo.dsf.ServiceEngine){vjo.dsf.ServiceEngine.handleRequest(_8);}
if(_8.returnData===false){_5=false;}}else{if(_5!=false&&typeof _8!="undefined"){_5=_8;}}}
return _5;},register:function(id,_a,_b,_c){if(!id||!_a||!_b){return this;}else{if(typeof _b.handle!="function"){if(typeof _b=="function"){var _d=_b,scp=_c||this;var _e={handle:function(){return _d.apply(scp,arguments);}};_b=_e;}else{return this;}}}
var _f=this.eventHandlers[id];if(!_f){_f=this.eventHandlers[id]={};}
if(!_f[_a]){_f[_a]=[];}
var len=_f[_a].length;_f[_a][len]=_b;return _b;},unregister:function(id,_12){if(!this.eventHandlers[id]){return;}
this.eventHandlers[id][_12]=[];},registerNative:function(_13,_14,_15){var id=(_13==window)?"body":_13.id;var _17=this.nativeEventHandlers[id];if(!_17){_17=this.nativeEventHandlers[id]={};}
if(!_17[_14]){_17[_14]=[];}
var len=_17[_14].length;_17[_14][len]=_15;},add:function(id,_1a,_1b,_1c){if(!id||!_1a||!_1b){return this;}
var b=this.isBound(id,_1a),rv=this.register(id,_1a,_1b,_1c);if(!b){b=this.bind(id,_1a);if(b===null){this.unboundElems[this.unboundElems.length]=id;}}
return rv;},addEventListener:function(_1e,_1f,_20,_21,_22){var scp=_21||vjo.global;if(typeof _1e=="string"){_1e=this.vj$.Element.get(_1e);}
if(!_1e){return false;}
var _24=function(_25){var ev=_25||window.event;var rv=_20.call(scp,ev);if(rv===false){vjo.dsf.EventDispatcher.stopEvent(ev);}
if(typeof rv!="undefined"){return rv;}};if(window.addEventListener){_1e.addEventListener(_1f,_24,_22||false);this.registerNative(_1e,_1f,_24);return _24;}else{if(window.attachEvent){_1e.attachEvent("on"+_1f,_24);this.registerNative(_1e,_1f,_24);return _24;}}
_1e["on"+_1f]=_24;return false;},bind:function(id,_29){var _2a=this.vj$.Element.get(id);if(id=="body"||_2a==document.body){_2a=document.body;if(_29=="load"||_29=="unload"){var rv=this.addEventListener(window,_29,function(_2c){var oED=vjo.dsf.EventDispatcher;if(typeof oED.fCustomLoad[_29]=="function"){oED.fCustomLoad[_29]();}
oED.run(document.body,_2c||window.event,_29);oED.unregister("body",_29);oED.fCustomLoad={};});if(rv===false){if(_2a.vjLoadSet){return this;}else{_2a.vjLoadSet=true;var _2e=window["on"+_29]||"";if(_2e){this.fCustomLoad[_29]=_2e;}}}
return this;}}
if(_2a){this.addEventListener(_2a,_29,this.notifier,_2a);return this;}
return null;},notifier:function(_2f,_30){return vjo.dsf.EventDispatcher.run(this,_2f||window.event,_30);},reBind:function(){var eH=this.eventHandlers,uE=this.unboundElems,len=uE.length,tmp=[];for(var i=0;i<len;i++){var id=uE[i],hdls=eH[id];if(hdls){for(var _34 in hdls){if(!this.hasBinding(id,_34)){var rv=this.bind(id,_34);if(rv===null){tmp[tmp.length]=id;}}}}}
this.unboundElems=tmp;},isBound:function(id,_37){var _38=this.eventHandlers[id];return(_38&&_38[_37]&&_38[_37].length>0);},hasBinding:function(id,_3a){var nEH=this.nativeEventHandlers;if(nEH[id]&&nEH[id][_3a]){var aH=nEH[id][_3a],len=aH.length,rv=false;for(var i=0;i<len;i++){var str=aH[i].toString();if(str&&str.indexOf("vjo.dsf.EventDispatcher")!=-1){return true;}}}
return false;},removeEventListener:function(_3f,_40,_41){if(!_3f||!_40){return;}else{if(typeof _3f=="string"){_3f=this.vj$.Element.get(_3f);}}
if(window.addEventListener&&_41){_3f.removeEventListener(_40,_41,false);}else{if(window.attachEvent&&_41){_3f.detachEvent("on"+_40,_41);}else{_3f["on"+_40]=null;}}},detachNativeHandlers:function(_42,_43){var id=(_42==window)?"body":_42.id;var _45=this.nativeEventHandlers[id];if(_45&&_45[_43]){var h=_45[_43],len=h.length;for(var i=0;i<len;i++){this.removeEventListener(_42,_43,_45[_43][i]);}
_45[_43]=[];}},detachHandler:function(id,_49,_4a){var _4b=this.eventHandlers[id];if(!_4b||!_4b[_49]){return;}
var h=[],len=_4b[_49].length;for(var i=0;i<len;i++){if(_4a!=_4b[_49][i]){h[h.length]=_4b[_49][i];}}
this.eventHandlers[id][_49]=h;},detachHandlers:function(id,_4f){this.unregister(id,_4f);var _50=this.vj$.Element.get(id);if(id=="body"){_50=window;}
if(_50){this.detachNativeHandlers(_50,_4f);}},stopEvent:function(e){this.stopPropagation(e);this.preventDefault(e);},stopPropagation:function(e){if(e.stopPropagation){e.stopPropagation();}else{e.cancelBubble=true;}},preventDefault:function(e){if(e.preventDefault){e.preventDefault();}else{e.returnValue=false;}},target:function(_54){return this.resolveTextNode((_54.target)?_54.target:_54.srcElement);},currentTarget:function(_55){return this.resolveTextNode((_55.currentTarget)?_55.currentTarget:_55.srcElement);},relatedTarget:function(_56){if(_56.relatedTarget){return this.resolveTextNode(_56.relatedTarget);}else{if((_56.type==="mouseover")&&_56.fromElement){return this.resolveTextNode(_56.fromElement);}else{if((_56.type==="mouseout")&&_56.toElement){return this.resolveTextNode(_56.toElement);}else{return null;}}}},resolveTextNode:function(_57){return(_57&&(_57.nodeType==3))?_57.parentNode:_57;},cleanUp:function(){var _58=this.nativeEventHandlers;for(var id in _58){for(var ev in _58[id]){if(ev!="unload"){this.detachHandlers(id,ev,true);}}}},getId:function(src,id){var _5d=id;if(_5d===null||!_5d){_5d=src.id;}
return _5d;},getBodyId:function(src,id){var _60=this.getId(src,id);if(!_60||src==document.body){_60="body";}
return _60;},unload:function(src,_62){return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"unload",_62));},change:function(src,_64){return this.process(this.getId(src),new vjo.dsf.Event(src,"change",_64));},submit:function(src,_66){return this.process(this.getId(src),new vjo.dsf.Event(src,"submit",_66));},reset:function(src,_68){return this.process(this.getId(src),new vjo.dsf.Event(src,"reset",_68));},select:function(src,_6a){return this.process(this.getId(src),new vjo.dsf.Event(src,"select",_6a));},blur:function(src,_6c){return this.process(this.getId(src),new vjo.dsf.Event(src,"blur",_6c));},focus:function(src,_6e){return this.process(this.getId(src),new vjo.dsf.Event(src,"focus",_6e));},keydown:function(src,_70){return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"keydown",_70));},keypress:function(src,_72){return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"keypress",_72));},keyup:function(src,_74){return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"keyup",_74));},click:function(src,_76){return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"click",_76));},dblclick:function(src,_78){return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"dblclick",_78));},mousedown:function(src,_7a){return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"mousedown",_7a));},mousemove:function(src,_7c){return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"mousemove",_7c));},mouseout:function(src,_7e){return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"mouseout",_7e));},mouseover:function(src,_80){return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"mouseover",_80));},mouseup:function(src,_82){return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"mouseup",_82));},contextmenu:function(src,_84){return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"contextmenu",_84));},load:function(src,_86){return this.run(src,_86,"load");},run:function(src,_88,_89){var et=_89||_88.type;var id=this.getBodyId(src);var _8c=new vjo.dsf.Event(src,et,_88);var rv=this.process(id,_8c);if(et==="load"&&id==="body"){this.unregister("body","load");}
return rv;}}).inits(function(){vjo.dsf.EventDispatcher.addEventListener(window,"load",function(){vjo.dsf.EventDispatcher.addEventListener(window,"unload",function(){vjo.dsf.EventDispatcher.cleanUp();});});}).endType();
vjo.itype("vjo.dsf.ITransport").protos({handleRequest:function(_1,_2){}}).endType();
vjo.ctype("vjo.dsf.ServiceResponse").protos({objType:undefined,errors:undefined,data:undefined,constructs:function(){this.errors=[];},addError:function(_1){if(this.errors){this.errors[this.errors.length]=_1;}}}).endType();
vjo.ctype("vjo.dsf.Message").protos({objType:"dsf_Message",trspType:"",svcId:"",request:null,response:null,status:null,svcConfig:null,stok:"",pId:"",constructs:function(){if(arguments.length===1){this.svcId=arguments[0];}else{if(arguments.length===2){this.svcId=arguments[0]+"."+arguments[1];this.svcName=arguments[0];this.opName=arguments[1];}}
this.rawRequest="";this.clientContext={};this.trspType="InProc";this.returnData=true;this.trace="";this.v="0";this.headers=null;}}).endType();
vjo.ctype("vjo.dsf.InProcReqtHdl").satisfies("vjo.dsf.ITransport").needs("vjo.dsf.ServiceResponse").needs("vjo.dsf.Message").protos({constructs:function(){this.svcHdls={};},registerSvcHdl:function(_1,_2){this.svcHdls[_1]=_2;},getSvcHdl:function(_3){return this.svcHdls[_3];},handleRequest:function(_4,_5){if(!_4.response){_4.response=new vjo.dsf.ServiceResponse();}
var _6=this.svcHdls[_4.svcId];if(_6){_4.response.data=_6.invoke(_4);}
if(typeof _4.status=="undefined"||_4.status==null){_4.status=1;}}}).endType();
vjo.ctype("vjo.dsf.Error").protos({id:undefined,message:undefined,constructs:function(_1,_2){this.id=_1;this.message=_2;}}).endType();
vjo.ctype("vjo.dsf.SvcErr").needs("vjo.dsf.Error").needs("vjo.dsf.Message").props({InvRsp:"InvalidRsp",InvRqBnd:"RqInvalidBnd",InvRspBnd:"RspInvalidBnd",InvRq:"InvalidRequest",TO:"TimedOut",RqUnk:"Client.Request.UnknownError",SvcRq:"SYS.SVC_REQUEST_ERROR",SvcRsp:"SYS.SVC_RESPONSE_ERROR",GlbRq:"SYS.GLOBAL_REQUEST_ERROR",GlbRsp:"SYS.GLOB_RESPONSE_ERROR",TrnRq:"SYS.TRANS_REQUEST_ERROR",TrnRsp:"SYS.TRANS_RESPONSE_ERROR",Prs:"SYS.JSON_PARSE_ERROR",SvcPrc:"SYS.DARWIN_SERVICE_PROTOCOL_ERROR",err:function(_1,_2,_3){_1.response.addError(new vjo.dsf.Error(_2,_3));}}).endType();
vjo.ctype("vjo.dsf.ServiceRegistry").needs("vjo.dsf.ITransport").props({t:[],b:[],getBinding:function(_1){return this.b[_1];},getTransport:function(_2){return this.t[_2];},registerBinding:function(_3,_4){var _5=false;if(_3&&_4){this.b[_3]=_4;_5=true;}
return _5;},registerTransport:function(_6,_7){var _8=false;if(_6&&_7){if(vjo.isInstanceOf(_7,vjo.dsf.ITransport)){this.t[_6]=_7;_8=true;}}
return _8;}}).endType();
vjo.ctype("vjo.dsf.RemoteReqtHdl").satisfies("vjo.dsf.ITransport").needs("vjo.dsf.SvcErr").needs("vjo.dsf.ServiceRegistry").needs("vjo.dsf.ServiceResponse").needs("vjo.dsf.Message").protos({handleRequest:function(_1,_2){_1.response=new vjo.dsf.ServiceResponse();if(!_1.svcConfig){vjo.dsf.SvcErr.err(_1,vjo.dsf.SvcErr.InvRq,"message.svcConfig is undefined");return;}
var _3=vjo.dsf.ServiceRegistry.getBinding(_1.svcConfig.reqtMarshalling);if(_3){_3.serialize(_1);}else{vjo.dsf.SvcErr.err(_1,vjo.dsf.SvcErr.InvRqBnd,"reqtMarshalling="+_1.svcConfig.reqtMarshalling);}
this.invoke(_1);},invoke:function(_4){var _5="XHR";if(_4.svcConfig.respMarshalling=="JSCALLBACK"){_5="JSONP";}
var _6=vjo.dsf.ServiceRegistry.getTransport(_5);if(_6){_6.handleRequest(_4,this.callback);}},callback:function(_7){try{var _8=vjo.dsf.ServiceRegistry.getBinding(_7.svcConfig.respMarshalling);if(_8){_8.deserialize(_7);}else{vjo.dsf.SvcErr.err(_7,vjo.dsf.SvcErr.InvRspBnd,"marshalling="+_7.svcConfig.respMarshalling+":responseText="+_7.response);}}
catch(e){vjo.dsf.SvcErr.err(_7,vjo.dsf.SvcErr.InvRsp,"marshalling="+_7.svcConfig.respMarshalling+":responseText="+_7.response);}
vjo.dsf.ServiceEngine.handleResponse(_7);}}).endType();
vjo.ctype("vjo.dsf.SvcConfig").protos({objType:null,url:null,method:null,reqtMarshalling:null,respMarshalling:null,async:true,timeout:0,constructs:function(_1,_2){this.objType="dsf_SvcConfig";this.svcType=null;this.url=_2;this.method=_1;this.reqtMarshalling="raw";this.respMarshalling="raw";this.async=true;this.timeout=0;}}).endType();
vjo.ctype("vjo.dsf.XDomainRequest").satisfies("vjo.dsf.ITransport").needs("vjo.dsf.EventDispatcher").props({callbacks:[],sCallbackName:"callback",sPreId:"xdr_",sPreExtId:"xdr_ext_",iCount:0,bUseIframe:(navigator.userAgent.indexOf("Firefox")>0),onLoad:function(){this.bodyLoaded=true;},getReqDiv:function(){return document.getElementsByTagName(this.bodyLoaded?"body":"head")[0];},send:function(_1){if(!document.createElement||!_1){return;}
var _2="",eid="",rdm="_vrdm="+(new Date()).getTime();if(typeof _1=="string"){_2=_1;eid=this.sPreExtId+this.iCount++;}else{if(_1.objType=="dsf_Message"&&_1.svcConfig){var cb=this.createCallback(_1);eid=this.sPreId+this.callbacks[this.callbacks.length-1];_2=_1.svcConfig.url+"&callback="+cb;}}
var _4=null,doc;doc=document;var _5=this.createElement("script");_5.id=eid;_5.type="text/javascript";var _6=true;if(arguments.length>1){_6=arguments[1];}
if(_6){_2=_2+((_2.indexOf("?")==-1)?"?":"&")+rdm;}
_5.src=_2;this.getReqDiv().appendChild(_5);return eid;},createCallback:function(_7){var _8=this.callbacks.length,name=this.sCallbackName+_8,eid=this.sPreId+name;this.callbacks[_8]=name;this[name]=function(_9){vjo.dsf.XDomainRequest.loaded(eid);_7.response=_9;vjo.dsf.ServiceEngine.handleResponse(_7);};var _a="",rv=((_a)?_a+".":"")+"vjo.dsf.XDomainRequest."+name;return rv;},loaded:function(_b){var e=document.getElementById(_b);if(e!==null){e.parentNode.removeChild(e);}},createElement:function(_d){return(typeof(createElementV4)!="undefined")?createElementV4(_d):document.createElement(_d);}}).protos({handleRequest:function(_e,_f){vjo.dsf.XDomainRequest.send(_e);}}).inits(function(){vjo.dsf.EventDispatcher.addEventListener(window,"load",this.onLoad,this);}).endType();
vjo.ctype("vjo.dsf.ServiceEngine").needs(["vjo.dsf.InProcReqtHdl","vjo.dsf.RemoteReqtHdl","vjo.dsf.ServiceResponse","vjo.dsf.SvcConfig","vjo.dsf.Message","vjo.dsf.XDomainRequest","vjo.dsf.ServiceRegistry","vjo.dsf.SvcErr"]).props({STATUS:{ABORT:-1,JUMP:1},init:function(){this.svcReqtHdls=[];this.svcRespHdls=[];this.glbReqtHdls=[];this.glbRespHdls=[];this.trspReqtHdls=[];this.trspRespHdls=[];this.svcHdls=[];this.inProcHdl=new this.vj$.InProcReqtHdl();this.remoteHdl=new this.vj$.RemoteReqtHdl();this.E=this.vj$.SvcErr;vjo.dsf.ServiceRegistry.registerTransport("JSONP",new vjo.dsf.XDomainRequest());},handleRequest:function(_1){var rt,ab=this.STATUS.ABORT,m=_1;if((m.status!=ab)&&typeof rt=="undefined"){rt=this.processServiceRequestHandlers(_1);}
if((m.status!=ab)&&typeof rt=="undefined"){rt=this.processGlobalRequestHandlers(_1);}
if((m.status!=ab)&&typeof rt=="undefined"){this.processTransportHandlers(_1);}
if((m.status!=ab)&&("Remote"!=_1.trspType||typeof rt!="undefined")){this.handleResponse(_1,rt);}
return _1.returnData;},processServiceRequestHandlers:function(_4){var rt;var _6=this.svcReqtHdls[_4.svcId];if(_6){try{var _7=_6.length;for(var i=0;i<_7;i++){_4.trace=_4.trace+"-->svcReqtHdl_"+i;_6[i].handleRequest(_4);if(_4.status==this.STATUS.JUMP){rt="SVC";this.err(_4,this.E.SvcRq,this.E.SvcRq);break;}}}
catch(e){rt="SVC";this.err(_4,this.E.SvcRq,this.E.SvcRq);}}
return rt;},processGlobalRequestHandlers:function(_9){var rt;if(_9.status!=this.STATUS.JUMP){try{var _b=this.glbReqtHdls.length;for(var i=0;i<_b;i++){_9.trace=_9.trace+"-->glbReqtHdl_"+i;this.glbReqtHdls[i].handleRequest(_9);if(_9.status==this.STATUS.JUMP){rt="GLB";this.err(_9,this.E.GlbRq,this.E.GlbRq);break;}}}
catch(e){rt="GLB";this.err(_9,this.E.GlbRq,this.E.GlbRq);}}
return rt;},processTransportHandlers:function(_d){if(_d.status!=this.STATUS.JUMP&&_d.trspType){var _e=vjo.dsf.ServiceRegistry.getTransport(_d.trspType);if(_e){_e.handleRequest(_d,this.handleResponse);}else{var _f=this.trspReqtHdls[_d.trspType];if(_f){try{var len=_f.length;for(var i=0;i<len;i++){_d.trace=_d.trace+"-->trspReqtHdl_"+i;_f[i].handleRequest(_d);if(_d.status==this.STATUS.JUMP){this.err(_d,this.E.TrnRq,this.E.TrnRq);break;}}}
catch(e){this.err(_d,this.E.TrnRq,this.E.TrnRq);}}
if(_d.status!=this.STATUS.JUMP&&_d.status!=vjo.dsf.ServiceEngine.STATUS.ABORT){if(_d.trspType=="undefined"){this.err(_d,this.E.TrnRq,this.E.TrnRq);}else{if("InProc"===_d.trspType){this.inProcHdl.handleRequest(_d);}else{if("Remote"===_d.trspType){this.remoteHdl.handleRequest(_d);}}}}}}},handleResponse:function(msg,_13){if(msg.trspType!="undefined"&&msg.trspType&&typeof _13=="undefined"){this.processTransResponseHandlers(msg);}
if(_13!="SVC"){this.processGlobalResponseHandlers(msg);}
this.processServiceResponseHandlers(msg);},processTransResponseHandlers:function(msg){var _15=this.trspRespHdls[msg.trspType];try{if(_15){for(var i=_15.length-1;i>=0;i--){msg.trace=msg.trace+"-->trspRespHdl_"+i;_15[i].handleResponse(msg);}}}
catch(e){this.err(msg,this.E.TrnRsp,this.E.TrnRsp);}},processGlobalResponseHandlers:function(msg){try{for(var i=this.glbRespHdls.length-1;i>=0;i--){msg.trace=msg.trace+"-->glbRespHdl_"+i;this.glbRespHdls[i].handleResponse(msg);}}
catch(e){this.err(msg,this.E.GlbRsp,this.E.GlbRsp);}},processServiceResponseHandlers:function(msg){var _1a;if(msg.clientContext){_1a=msg.clientContext.svcApplier;}
try{if(_1a){if(typeof _1a.onResponse=="function"){_1a.onResponse(msg);}else{if(typeof _1a=="function"){_1a(msg);}}}}
catch(e){this.err(msg,this.E.SvcRsp,this.E.SvcRsp);}
var _1b=this.svcRespHdls[msg.svcId];if(_1b){try{for(var i=_1b.length-1;i>=0;i--){msg.trace=msg.trace+"-->svcRespHdl_"+i;_1b[i].handleResponse(msg);}}
catch(e){this.err(msg,this.E.SvcRsp,this.E.SvcRsp);}}},createHandler:function(_1d,_1e){if(typeof _1d[_1e]!="function"){if(typeof _1d=="function"){var _1f=_1d,obj={},self=this;obj[_1e]=function(){return _1f.apply(self,arguments);};_1d=obj;}}
return _1d;},registerSvcHdl:function(_20,_21){if(!_20||!_21){return;}
_21=this.createHandler(_21,"invoke");this.inProcHdl.registerSvcHdl(_20,_21);},getSvcHdl:function(_22){return this.inProcHdl.getSvcHdl(_22);},registerSvcReqtHdl:function(_23,_24){if(!_23||!_24){return;}
if(!this.svcReqtHdls){this.init();}
if(typeof this.svcReqtHdls[_23]=="undefined"){this.svcReqtHdls[_23]=[];}
var _25=this.svcReqtHdls[_23];_25[_25.length]=this.createHandler(_24,"handleRequest");},registerSvcRespHdl:function(_26,_27){if(!_26||!_27){return;}
if(typeof this.svcRespHdls[_26]=="undefined"){this.svcRespHdls[_26]=[];}
var _28=this.svcRespHdls[_26];_28[_28.length]=this.createHandler(_27,"handleResponse");},registerGlbReqtHdl:function(_29){if(!_29){return;}
this.glbReqtHdls[this.glbReqtHdls.length]=this.createHandler(_29,"handleRequest");},registerGlbRespHdl:function(_2a){if(!_2a){return;}
this.glbRespHdls[this.glbRespHdls.length]=this.createHandler(_2a,"handleResponse");},registerTrspReqtHdl:function(_2b,_2c){if(!_2b||!_2c){return;}
if(typeof this.trspReqtHdls[_2b]=="undefined"){this.trspReqtHdls[_2b]=[];}
if(!vjo.dsf.ServiceRegistry.registerTransport(_2b,_2c)){var _2d=this.trspReqtHdls[_2b];_2d[_2d.length]=this.createHandler(_2c,"handleRequest");}},registerTrspRespHdl:function(_2e,_2f){if(!_2e||!_2f){return;}
if(typeof this.trspRespHdls[_2e]=="undefined"){this.trspRespHdls[_2e]=[];}
var _30=this.trspRespHdls[_2e];_30[_30.length]=this.createHandler(_2f,"handleResponse");},err:function(msg,_32,_33){if(!msg.response){msg.response=new vjo.dsf.ServiceResponse();}
if(msg.response.addError){msg.response.addError(new vjo.dsf.Error(_32,_33));}else{msg.response.addError=vjo.dsf.ServiceResponse.prototype["addError"];if(msg.response.addError){msg.response.addError(new vjo.dsf.Error(_32,_33));}}},register:function(_34,_35,_36){var _s=vjo.dsf.ServiceEngine;switch(_34){case 0:_s.registerSvcHdl(_35,_36);break;case 1:_s.registerSvcReqtHdl(_35,_36);break;case 2:_s.registerGlbReqtHdl(_35,_36);break;case 3:_s.registerTrspReqtHdl(_35,_36);break;case 4:_s.registerSvcRespHdl(_35,_36);break;case 5:_s.registerGlbRespHdl(_35,_36);break;case 6:_s.registerTrspRespHdl(_35,_36);break;}}}).inits(function(){this.init();}).endType();
vjo.ctype("vjo.dsf.assembly.VjClientAssemblerRequest").protos({constructs:function(_1,_2,_3,_4,_5){this.sUrl=_1;this.fCallback=_2;this.oScope=_3||window;this.sCallbackParam=_4;var b=_5;if(typeof(b)=="undefined"){b=true;}
this.bSendResponseOnLoad=b;}}).endType();
vjo.ctype("vjo.dsf.assembly.VjClientAssembler").needs(["vjo.dsf.EventDispatcher","vjo.dsf.ServiceEngine","vjo.dsf.assembly.VjClientAssemblerRequest"]).props({load:function(_1){var m=_1,url=m.sUrl||"",name=this.generateCallback(m),ver="",cb=((ver)?ver+".":"")+"vjo.dsf.assembly.VjClientAssembler."+name;if(m&&m.objType=="dsf_Message"){url=m.svcConfig.url+"?";url+=vjo.dsf.Service.generateReqParams(m)+"&callback="+cb;}else{if(m.sCallbackParam){url=url+"&"+m.sCallbackParam+"="+cb;}}
return vjo.dsf.XDomainRequest.send(url);},generateCallback:function(_3){var m=_3;var _5=this.aCallbacks.length;var _6=this.sPreCallback+_5;this.aCallbacks[_5]=_6;this.aModels[_6]=m;this[_6]=function(){if(this.loaded[_6]===true){return;}
this.loaded[_6]=true;if(this.bBodyLoaded||!m.bSendResponseOnLoad){if(m.objType=="dsf_Message"){m.response=arguments[0];vjo.dsf.ServiceEngine.handleResponse(m);}else{m.fCallback.apply(m.oScope,arguments);}}else{if(m.objType=="dsf_Message"){this.aResponses[_6]=arguments[0];m.response=this.aResponses[_6];}else{this.aResponses[_6]=arguments;}
if(this.bBodyLoaded){this.assemble();}}};return _6;},assemble:function(){this.bBodyLoaded=true;if(this.bLock){setTimeout("vjo.dsf.assembly.VjClientAssembler.assemble()",1000);return;}
this.bLock=true;try{for(var _7 in this.aResponses){this.loaded[_7]=true;var m=this.aModels[_7];if(this.aResponses[_7]!=null){if(m.objType=="dsf_Message"){vjo.dsf.ServiceEngine.handleResponse(m);}else{m.fCallback.apply(m.oScope,this.aResponses[_7]);}}
this.aResponses[_7]=null;}}
finally{this.bLock=false;}}}).inits(function(){this.aCallbacks=[];this.aResponses={};this.aModels={};this.sPreCallback="_callback";this.bBodyLoaded=false;this.bLock=false;this.loaded={};this.vj$.EventDispatcher.addEventListener(window,"load",function(){vjo.dsf.assembly.VjClientAssembler.assemble();});}).endType();
vjo.ctype("vjo.dsf.typeextensions.string.Comparison").endType();String.prototype.has=function(_1){return(this.indexOf(_1)!=-1);};String.prototype.hasArg=function(_2){var a=_2,rv=false;if(typeof(a)=="string"){rv=this.has(a);}else{var aL=a.length;for(var j=0;j<aL&&!rv;j++){rv=this.has(a[j]);}}
return rv;};String.prototype.hasAny=function(){var a=arguments,l=a.length,rv=false;for(var i=0;i<l&&!rv;i++){rv=this.hasArg(a[i]);}
return rv;};String.prototype.hasAll=function(){var a=arguments,l=a.length;for(var i=0;i<l;i++){if(!this.hasArg(a[i])){return false;}}
return true;};String.prototype.is=function(s){return(this==s);};String.prototype.isAny=function(){var a=arguments,l=a.length,rv=false,aL;for(var i=0;i<l&&!rv;i++){if(typeof(a[i])=="string"){rv=(this==a[i]);}else{aL=a[i].length;for(var j=0;j<aL&&!rv;j++){rv=(this==a[i][j]);}}}
return rv;};
vjo.ctype("vjo.dsf.typeextensions.string.HexToDecimal").endType();String.prototype.hex2Dec=function(){return parseInt(this,16);};
vjo.ctype("vjo.dsf.typeextensions.number.DecimalToHex").endType();Number.prototype.dec2Hex=function(){return parseInt(this,10).toString(16);};
vjo.ctype("vjo.dsf.cookie.VjCookieJar").needs(["vjo.dsf.typeextensions.string.Comparison","vjo.dsf.typeextensions.string.HexToDecimal","vjo.dsf.typeextensions.number.DecimalToHex"]).props({Default_Cookie_Format:{"COOKIELET_DELIMITER":"^","NAME_VALUE_DELIMITER":"/","escapedValue":true},DP_Cookie_Format:{"COOKIELET_DELIMITER":"^","NAME_VALUE_DELIMITER":"/","bUseExp":true,"startDelim":"b"},Session_Cookie_Format:{"COOKIELET_DELIMITER":"^","NAME_VALUE_DELIMITER":"=","escapedValue":true,"startDelim":"^"},DS_Cookie_Format:{"COOKIELET_DELIMITER":"^","NAME_VALUE_DELIMITER":"/"},sPath:"/",aConversionMap:{"reg":["dp1","reg"],"recent_vi":["ebay","lvmn"],"ebaysignin":["ebay","sin"],"p":["dp1","p"],"etfc":["dp1","etfc"],"keepmesignin":["dp1","kms"],"ItemList":["ebay","wl"],"BackToList":["s","BIBO_BACK_TO_LIST"]},aFormatMap:{},sCOMPAT:"10",sCONVER:"01",sSTRICT:"00",sModesCookie:"ebay",sModesCookielet:"cv",readCookie:function(_1,_2){var rv=this.readCookieObj(_1,_2).value;return(rv)?decodeURIComponent(rv):"";},createDefaultCookieBean:function(_4,_5){var _6={};_6.name=_4;_6.cookieletname=_5;_6.value="";_6.maxage=0;_6.rawcookievalue="";_6.mode="";return _6;},readCookieObj:function(_7,_8){var _9=this.createDefaultCookieBean(_7,_8);this.update();this.checkConversionMap(_9);_9.rawcookievalue=this.aCookies[_9.name];if(!_9.name||!_9.rawcookievalue){_9.value="";}else{if(!_9.cookieletname){this.readCookieInternal(_9);}else{this.readCookieletInternal(_9);}}
return(typeof(_9)!="undefined")?_9:"";},checkConversionMap:function(_a){var _b=this.aConversionMap[_a.name];if(_b){_a.mode=this.getMode(_a.name);_a.name=_b[0];_a.cookieletname=_b[1];}},readCookieInternal:function(_c){_c.value=_c.rawcookievalue;return _c;},readCookieletInternal:function(_d){var _e=this.getCookielet(_d.name,_d.cookieletname,_d.rawcookievalue);var _f=this.getFormat(_d.name);if(_e&&_f.bUseExp){var _10=_e;_e=_e.substring(0,_e.length-8);if(_10.length>8){_d.maxage=_10.substring(_10.length-8);}}
_d.value=_e;if(_d.mode==this.sCOMPAT){_d.value=_d.rawcookievalue;}
return _d;},readMultiLineCookie:function(_11,_12){if(!_11||!_12){return"";}
var val,r="";var _15=this.aConversionMap[_11];if(_15){val=this.readCookieObj(_15[0],_15[1]).value||"";}
if(val){r=this.getCookielet(_11,_12,val)||"";}
return(typeof(r)!="undefined")?r:"";},writeCookie:function(_16,_17,_18){var _19=this.aConversionMap[_16];if(_19){this.writeCookielet(_19[0],_19[1],_17,_18);return;}
var _1a=this.getFormat(_16);if(_17&&_1a.escapedValue){_17=encodeURIComponent(_17);}
this.writeRawCookie(_16,_17,_18);},writeRawCookie:function(_1b,_1c,_1d){if(_1b&&(_1c!==undefined)){if((isNaN(_1c)&&_1c.length<4000)||(_1c+"").length<4000){if(typeof _1d=="number"){_1d=this.getExpDate(_1d);}
var _1e=_1d?new Date(_1d):new Date(this.getExpDate(730));var _1f=this.getFormat(_1b);var _20=this.sCookieDomain;var dd=document.domain;if(!dd.has(_20)){var _22=dd.indexOf(".ebay.");if(_22>0){this.sCookieDomain=dd.substring(_22);}}
if(document.cookie){document.cookie=_1b+"="+(_1c||"")+((_1d||_1f.bUseExp)?"; expires="+_1e.toGMTString():"")+"; domain="+this.sCookieDomain+"; path="+this.sPath;}}}},writeCookieEx:function(_23,_24,_25){this.writeCookie(_23,_24,this.getExpDate(_25));},writeCookielet:function(_26,_27,_28,_29,_2a){if(_26&&_27){this.update();var _2b=this.getFormat(_26);if(_2b.bUseExp&&_28){if(typeof _29=="number"){_29=this.getExpDate(_29);}
var _2c=_29?new Date(_29):new Date(this.getExpDate(730));var _2d=Date.UTC(_2c.getUTCFullYear(),_2c.getUTCMonth(),_2c.getUTCDate(),_2c.getUTCHours(),_2c.getUTCMinutes(),_2c.getUTCSeconds());_2d=Math.floor(_2d/1000);_28+=_2d.dec2Hex();}
var val=this.createCookieValue(_26,_27,_28);this.writeRawCookie(_26,val,_2a);}},writeMultiLineCookie:function(_2f,_30,_31,_32,_33){this.update();var val=this.createCookieValue(_2f,_30,_31);if(val){var _35=this.aConversionMap[_2f];if(_35){this.writeCookielet(_35[0],_35[1],val,_32,_33);}}},getBitFlagOldVersion:function(_36,_37){var dec=parseInt(_36,10);var b=dec.toString(2),r=dec?b.charAt(b.length-_37-1):"";return(r=="1")?1:0;},setBitFlagOldVersion:function(_3a,_3b,_3c){var b="",p,i,e,l;_3a=parseInt(_3a+"",10);if(_3a){b=_3a.toString(2);}
l=b.length;if(l<_3b){e=_3b-l;for(i=0;i<=e;i++){b="0"+b;}}
p=b.length-_3b-1;return parseInt(b.substring(0,p)+_3c+b.substring(p+1),2);},getBitFlag:function(_3e,_3f){if(_3e!=null&&_3e.length>0&&_3e.charAt(0)=="#"){var _40=_3e.length;var q=_3f%4;var _42=Math.floor(_3f/4)+1;var _43=_40-_42;var _44=parseInt(_3e.substring(_43,_43+1),16);var _45=1<<q;return((_44&_45)==_45)?1:0;}else{return this.getBitFlagOldVersion(_3e,_3f);}},setBitFlag:function(_46,_47,_48){if(_46!=null&&_46.length>0&&_46.charAt(0)=="#"){var _49=_46.length;var q=_47%4;var _4b=Math.floor(_47/4)+1;if(_49<=_4b){if(_48!=1){return _46;}
var _4c=_4b-_49+1;var _4d=_46.substring(1,_49);while(_4c>0){_4d="0"+_4d;_4c--;}
_46="#"+_4d;_49=_46.length;}
var _4e=_49-_4b;var _4f=parseInt(_46.substring(_4e,_4e+1),16);var _50=1<<q;if(_48==1){_4f|=_50;}else{_4f&=~_50;}
_46=_46.substring(0,_4e)+_4f.toString(16)+_46.substring(_4e+1,_49);return _46;}else{if(_47>31){return _46;}
return this.setBitFlagOldVersion(_46,_47,_48);}},createCookieValue:function(_51,_52,_53){var _54=this.aConversionMap[_51],format=this.getFormat(_51),mode=this.getMode(_51),val;if(_54&&(mode==this.sSTRICT||mode==this.sCONVER)){val=this.readCookieObj(_54[0],_54[1]).value||"";}else{val=this.aCookies[_51]||"";}
if(format){var _55=this.getCookieletArray(val,format);_55[_52]=_53;var str="";for(var i in _55){if(_55[i]){str+=i+format.NAME_VALUE_DELIMITER+_55[i]+format.COOKIELET_DELIMITER;}}
if(str&&format.startDelim){str=format.startDelim+str;}
val=str;if(format.escapedValue){val=encodeURIComponent(val);}}
return val;},update:function(){var aC=document.cookie.split("; ");this.aCookies={};for(var i=0;i<aC.length;i++){var sC=aC[i].split("=");var _5b=this.getFormat(sC[0]),cv=sC[1],sd=_5b.startDelim;if(sd&&cv&&cv.indexOf(sd)===0){sC[1]=cv.substring(sd.length,cv.length);}
this.aCookies[sC[0]]=sC[1];}},getCookielet:function(_5c,_5d,_5e){var _5f=this.getFormat(_5c);var _60=this.getCookieletArray(_5e,_5f);return _60[_5d]||"";},getFormat:function(_61){return this.aFormatMap[_61]||vjo.dsf.cookie.VjCookieJar.Default_Cookie_Format;},getCookieletArray:function(_62,_63){var rv=[],val=_62||"";if(_63.escapedValue){val=decodeURIComponent(val);}
var a=val.split(_63.COOKIELET_DELIMITER);for(var i=0;i<a.length;i++){var idx=a[i].indexOf(_63.NAME_VALUE_DELIMITER);if(idx>0){rv[a[i].substring(0,idx)]=a[i].substring(idx+1);}}
return rv;},getExpDate:function(_68){var _69;if(typeof _68=="number"&&_68>=0){var d=new Date();d.setTime(d.getTime()+(_68*24*60*60*1000));_69=d.toGMTString();}
return _69;},getMode:function(_6b){var h=this.readCookieObj(this.sModesCookie,this.sModesCookielet).value,b;if(!(_6b in this.aConversionMap)){return null;}
if(!h){return"";}
if(h===0){return this.sSTRICT;}
if(h&&h!="0"){if(h.has(".")){var a=h.split(".");for(var i=0;i<a.length;i++){b=a[i].hex2Dec().toString(2)+b;}}else{b=h.hex2Dec().toString(2);}
i=0;var l=b.length,j;for(o in this.aConversionMap){j=l-(2*(i+1));var f=b.substring(j,j+2).toString(10);f=(!f)?this.sSTRICT:f;if(_6b==o){return(f.length==1)?"0"+f:f;}
i++;}
return null;}}}).inits(function(){var vCJ=vjo.dsf.cookie.VjCookieJar;vCJ.aFormatMap={"r":vCJ.Default_Cookie_Format,"dp1":vCJ.DP_Cookie_Format,"npii":vCJ.DP_Cookie_Format,"ebay":vCJ.Session_Cookie_Format,"reg":vCJ.Session_Cookie_Format,"apcCookies":this.Session_Cookie_Format,"ds2":vCJ.DS_Cookie_Format};}).endType();
vjo.ctype("vjo.dsf.utils.URL").props({addArg:function(_1,_2,_3){if(_1==null||_1==undefined){return null;}
if(_1.indexOf("?")<0){_1+="?"+_2+"="+_3;return _1;}
var _4=this.getArgPairIfExists(_1,_2);if(_4!==null){_1=_1.replace(_4,_2+"="+_3);}else{_1+="&"+_2+"="+_3;}
return _1;},getArg:function(_5,_6){if(_5==null||_5==undefined){return null;}
if(_5.indexOf("?")<0){return null;}
var _7=this.getArgPairIfExists(_5,_6);if(_7!==null){return _7.substring(_7.indexOf("=")+1);}
return null;},getArgPairIfExists:function(_8,_9){var _a=_8.indexOf("?");if(_a<0){return null;}
var _b=_8;var _c,_d;while(_a>=0){_b=_b.substring(_a+1);_c=_b;_a=_b.indexOf("&");if(_a>=0){_c=_b.substring(0,_a);}
_d=_c.substring(0,_c.indexOf("="));if(_d==_9){return _c;}}
return null;}}).endType();
vjo.ctype("vjo.darwin.core.sitespeed.SiteSpeed").needs(["vjo.dsf.EventDispatcher","vjo.dsf.utils.URL","vjo.dsf.cookie.VjCookieJar"]).props({params:{},gauge:function(){if(typeof(oGaugeInfo)!="undefined"){var _1=(new Date()).getTime()-oGaugeInfo.iST;this.addParam("ct21",_1);var _2=document.referrer;var _3=vjo.dsf.utils.URL.getArg(document.location.href,"ForceSiteSpeedGauge");if(_3=="true"||(_2!=null&&_2.indexOf("ebay.")>0)){if(oGaugeInfo.uldTime!=null&&oGaugeInfo.uldTime!=""){_1=(new Date()).getTime()-oGaugeInfo.uldTime;if(_1<300000){this.addParam("ex3",_1);}}}
this.gaugeInternal();}},gaugeUnload:function(){var _4=(new Date()).getTime();vjo.dsf.cookie.VjCookieJar.writeCookielet("ssg","uld",_4);if(typeof(oGaugeInfo)!="undefined"){var _5=(new Date()).getTime()-oGaugeInfo.iST;this.addParam("ex2",_5);this.gaugeInternal();}},gaugeBodyLoad:function(){if(typeof(oGaugeInfo)!="undefined"){var _6=(new Date()).getTime()-oGaugeInfo.iST;this.addParam("ctb",_6);var _7=vjo.dsf.utils.URL.getArg(document.location.href,"ForceSiteSpeedGauge");if(_7=="true"){var _8=new Image(1,1);if(_8){_8.src="http://p.ebaystatic.com/aw/pics/sitespeed/past/speedp.gif";}}}},addParam:function(_9,_a){if(_9&&_a){this.params[_9]=_a;}},getParams:function(){var rv="";for(var k in this.params){rv+="&"+k+"="+this.params[k];}
return rv;},gaugeInternal:function(){var _d=vjo.dsf.cookie.VjCookieJar,sbf=_d.readCookie("ebay","sbf");_d.writeCookielet("ebay","sbf",_d.setBitFlag(sbf,20,1));var _e=this.getParams();if(typeof(oGaugeInfo)!="undefined"&&oGaugeInfo.bFlag!=1){if(oGaugeInfo.sent===true){return;}
oGaugeInfo.sent=true;var _f=new Image(1,1);if(_f){_f.src=oGaugeInfo.sUrl.replace(/&amp;/g,"&")+_e;}}}}).inits(function(){var oCJ=vjo.dsf.cookie.VjCookieJar;sbf=oCJ.readCookie("ebay","sbf");b=(sbf)?oCJ.getBitFlag(sbf,20):0;if(typeof(oGaugeInfo)!="undefined"){oGaugeInfo.bFlag=b;oGaugeInfo.sent=false;oGaugeInfo.uldTime=oCJ.readCookie("ssg","uld");}
var scp=vjo.darwin.core.sitespeed.SiteSpeed;vjo.dsf.EventDispatcher.addEventListener(window,"load",scp.gauge,scp);vjo.dsf.EventDispatcher.addEventListener(window,"beforeunload",scp.gaugeUnload,scp);}).endType();
vjo.ctype("vjo.Registry").props({controls:[],put:function(_1,_2){this.controls[_1]=_2;if(this.isKeyValid(_1)){this["_"+_1]=this.controls[_1];}
return this.controls[_1];},get:function(_3){return this.controls[_3];},dump:function(){var _4=this.controls;var _5="controls on page:\n";for(var i in _4){_5+="key = "+i;_5+="controlName = "+_4[i].objtype;_5+="\n";}
return _5;},isKeyValid:function(_7){if(typeof _7!="string"){return false;}
return/^([a-zA-Z0-9_$]+)$/.test(_7);}}).endType();
vjo.ctype("vjo.dsf.client.Browser").needs("vjo.dsf.typeextensions.string.Comparison").props({init:function(){this.bFirefox=this.bWebTV=this.bOpera=this.bNav=this.bIE=this.bSafari=this.bWin=this.bMac=this.bMacppc=this.bMactel=this.bActiveXSupported=this.bWinXp=this.bXpSp2=this.bAOL=this.bVista=false;this.iVer=this.fVer=-1;this.fMinorVer=0;this.aMimeTypes=null;var nv=navigator,agt=nv.userAgent.toLowerCase(),i=0,ver;with(this){if(agt.has("webtv")){bWebTV=true;i=agt.indexOf("webtv/")+6;}else{if(agt.has("firefox")){bFirefox=true;i=agt.lastIndexOf("firefox")+8;}else{if(agt.has("safari")){bSafari=true;i=agt.lastIndexOf("safari")+7;}else{if(typeof(window.opera)!="undefined"){bOpera=true;i=agt.lastIndexOf("opera")+6;}else{if(nv.appName.is("Netscape")){bNav=true;i=agt.lastIndexOf("/")+1;}else{if(agt.has("msie")){bIE=true;i=agt.indexOf("msie")+4;if(agt.has("aol")||agt.has("america online")){bAOL=true;}}}}}}}
ver=bOpera?window.opera.version():agt.substring(i);iVer=parseInt(ver);fVer=parseFloat(ver);fMinorVer=fVer-iVer;bWin=agt.has("win");bWinXp=(bWin&&agt.has("windows nt 5.1"));bVista=(bWin&&agt.has("windows nt 6.0"));bXpSp2=(bWinXp&&agt.has("sv1"));bMac=agt.has("mac");bMacppc=(bMac&&agt.hasAny("ppc","powerpc"));bMactel=(bMac&&agt.has("intel"));aMimeTypes=nv.mimeTypes;bActiveXSupported=(!(bMac||bMacppc)&&(typeof(ActiveXObject)=="function"));}}}).inits(function(){vjo.dsf.client.Browser.init();}).endType();
vjo.ctype("vjo.dsf.document.Element").needs("vjo.dsf.Element","E").props({toggleHideShowRow:function(_1,_2){},toggleHideShow:function(_3,_4){},toggleVisibility:function(_5,_6){}}).inits(function(){vjo.dsf.document.Element=this.vj$.E;}).endType();
vjo.ctype("vjo.dsf.utils.Handlers").needs(["vjo.dsf.EventDispatcher","vjo.dsf.Message","vjo.dsf.ServiceEngine"]).props({ED:vjo.dsf.EventDispatcher,SE:vjo.dsf.ServiceEngine,attachEvt:function(_1,_2,_3,_4){return this.ED.addEventListener(_1,_2,_3,_4);},detachEvt:function(_5,_6,_7){this.ED.removeEventListener(_5,_6,_7);return true;},newMsg:function(_8){return new vjo.dsf.Message(_8);},handle:function(_9){this.SE.handleRequest(_9);},createHdl:function(_a,_b,_c){var _d=_a,hdl={};hdl[_c]=function(){_b.apply(_d,arguments);};return hdl;},attachSvc:function(_e,_f,_10){var t=this,hdl=t.createHdl(_10,_f,"invoke");if(t.SE&&hdl){t.SE.registerSvcHdl(_e,hdl);}},attachSvcReqt:function(_12,_13,_14){var t=this,hdl=t.createHdl(_14,_13,"handleRequest");if(t.SE&&hdl){t.SE.registerSvcReqtHdl(_12,hdl);}},attachSvcResp:function(_16,_17,_18){var t=this,hdl=t.createHdl(_18,_17,"handleResponse");if(t.SE&&hdl){t.SE.registerSvcRespHdl(_16,hdl);}},resetSvc:function(_1a){this.SE.inProcHdl.svcHdls[_1a]=[];},resetSvcReqt:function(_1b){this.SE.svcReqtHdls[_1b]=[];},resetSvcResp:function(_1c){this.SE.svcRespHdls[_1c]=[];}}).endType();
vjo.ctype("vjo.dsf.utils.Object").props({hitch:function(_1,_2){var _3;if(typeof _2=="string"){_3=_1[_2];}else{_3=_2;}
return function(){return _3.apply(_1,arguments);};},extend:function(_4,_5){function inheritance(){}
inheritance.prototype=_5.prototype;_4.prototype=new inheritance();_4.prototype.constructor=_4;_4.baseConstructor=_5;_4.superClass=_5.prototype;}}).endType();
vjo.ctype("vjo.dsf.utils.JsLoader").props({queue:[],pending:null,load:function(_1,_2,_3){var _4={url:_1,callback:_2,scope:_3},head,stag;var t=this;if(t.pending){t.queue.push(_4);return;}
this.pending=_4;head=document.getElementsByTagName("head")[0];stag=document.createElement("script");stag.onload=stag.onreadystatechange=function(){if(!this.readyState||this.readyState=="loaded"||this.readyState=="complete"){t.oncomplete();stag.onload=stag.onreadystatechange=null;head.removeChild(stag);}};stag.type="text/javascript";stag.src=_1;head.appendChild(stag);},oncomplete:function(){var t=this,o=t.pending;if(o.callback){o.callback.call(o.scope||window);}
t.pending=null;if(t.queue.length>0){var _7=this.queue.shift();t.load(_7.url,_7.callback,_7.scope);}}}).endType();
vjo.ctype("vjo.dsf.utils.CssLoader").props({load:function(_1){if(document.createStyleSheet){document.createStyleSheet(_1);}else{var _2=document.getElementsByTagName("head")[0],style=document.createElement("link");style.rel="stylesheet";style.type="text/css";style.href=_1;_2.appendChild(style);}}}).endType();
vjo.ctype("vjo.darwin.core.rtm.RTMHelper").props({oRtm:null,setRtm:function(_1){if(_1){this.oRtm=_1;}},render:function(_2){var t=this;if(t.oRtm){t.oRtm.renderPromos(_2);}},onMessage:function(_4){this.oRtm.onMessage(_4);}}).endType();
vjo.itype("vjo.dsf.common.IJsServiceHandler").protos({invoke:function(_1){}}).endType();
vjo.ctype("vjo.darwin.core.rtm.RTMInit").needs(["vjo.dsf.utils.URL","vjo.darwin.core.rtm.RTMHelper","vjo.dsf.assembly.VjClientAssembler"]).satisfies("vjo.dsf.common.IJsServiceHandler").protos({constructs:function(_1){this.oRTM=new vjo.darwin.core.rtm.RTM(_1);vjo.darwin.core.rtm.RTMHelper.setRtm(this.oRTM);},invoke:function(_2){var _3=this.oRTM.processGlobalNavPids(true);var _4=this.oRTM.processGlobalNavPids(false);if(_3!=""){var U=vjo.dsf.utils.URL;var _6=this.oRTM.oJSBean.url;_6=U.addArg(_6,"e","USC:"+this.oRTM.getSegment(false));var p=U.getArg(_6,"p");var ph=U.getArg(_6,"ph");if((p!=null)&&!p.has(_3)){this.oRTM.oJSBean.url=_6.replace("p="+p,"p="+p+":"+_3);if(ph!=null){this.oRTM.oJSBean.url=this.oRTM.oJSBean.url.replace("ph="+ph,"ph="+ph+":"+_4);}}
if(typeof(_oGlobalNavRTMInfo)!="undefined"){_oGlobalNavRTMInfo.aRTMPlacementData=[];}}
this.oRTM.appendUrl("&z="+this.oRTM.getFlashVersion());this.oRTM.appendUrl("&bw="+this.oRTM.getBrowserWidth());if(this.oRTM.getCIDCookie()!=null&&this.oRTM.getCIDCookie()!=""&&this.oRTM.getCIDCookie()!="undefined"){this.oRTM.appendUrl("&cg="+this.oRTM.getCIDCookie());}
if(this.oRTM.getEncodingType()!=null){this.oRTM.appendUrl("&enc="+this.oRTM.getEncodingType());}
this.oRTM.appendUrl("&v=4");this.oRTM.registerGlobalNavPlacements();if(!this.oRTM.oJSBean.onload){this.onSend();}else{vjo.dsf.EventDispatcher.addEventListener(window,"load",this.onLoad,this);}
vjo.dsf.EventDispatcher.addEventListener(window,"load",this.oRTM.setTimeOut,this.oRTM);_2.returnData=false;return _2;},onLoad:function(){var _9=this;window.setTimeout(function(){_9.onSend();},0);},onSend:function(){var _a=this.oRTM,opt=_a.bOptimize,func=opt?_a.storeResponse:_a.setInlineContent;var _b=new vjo.dsf.assembly.VjClientAssemblerRequest(_a.oJSBean.url,func,_a,"cb",!opt);vjo.dsf.assembly.VjClientAssembler.load(_b);}}).endType();
vjo.ctype("vjo.dsf.client.ActiveX").needs("vjo.dsf.client.Browser").props({init:function(){var oC=vjo.dsf.client.Browser;if(oC.bIE){var d=document,dw;dw=function(s){d.writeln(s);};dw("<scr"+"ipt language=\"vbscript\" type=\"text/vbscript\">");dw("\tFunction vbCheckActiveXControl (pActXName)");dw("\t\taX = false");dw("\t\ton error resume next");dw("\t\taX = IsObject(CreateObject(pActXName))");dw("\t\tvbCheckActiveXControl = aX");dw("End Function");dw("</scr"+"ipt>");}},isLibLoaded:function(_4){var oC=vjo.dsf.client.Browser;return oC.bActiveXSupported&&vbCheckActiveXControl(_4);}}).inits(function(){vjo.dsf.client.ActiveX.init();}).endType();
vjo.ctype("vjo.dsf.utils.Bit").needs("vjo.dsf.cookie.VjCookieJar").props({CJ:vjo.dsf.cookie.VjCookieJar,getMulti:function(_1,_2,_3){var r="",i,CJ=this.CJ;for(i=0;i<_3;i++){r=CJ.getBitFlag(_1,_2+i)+r;}
return parseInt(r,2);},setMulti:function(_5,_6,_7,_8){var i=0,CJ=this.CJ,v,l,e;v=_8.toString(2).substring(0,_7);l=v.length;if(l<_7){e=_7-l;for(var j=0;j<e;j++){v="0"+v;}
l=l+e;}
for(i=0;i<l;i++){_5=CJ.setBitFlag(_5,_6+i,v.substring(l-i-1,l-i));}
return _5;}}).endType();
vjo.ctype("vjo.dsf.flash.Version").needs(["vjo.dsf.client.Browser","vjo.dsf.client.ActiveX","vjo.dsf.utils.Bit","vjo.dsf.cookie.VjCookieJar"]).props({versions:[10,9],get:function(){var t=this,B=t.vj$.Browser,v=0,vs=t.versions,i,A=t.vj$.ActiveX,cv;cv=t.rw(false);if(cv){return(cv==1)?0:cv;}
if(B.bIE&&B.bWin&&!B.bOpera){for(i=0;i<vs.length;i++){if(A.isLibLoaded("ShockwaveFlash.ShockwaveFlash."+vs[i])){v=vs[i];break;}}}else{var n=navigator,pd,id,swf="Shockwave Flash";if(n.plugins[swf]){pd=n.plugins[swf].description;id=pd.indexOf("Flash")+5;v=parseInt(pd.substr(id,pd.length));}
if(B.bWebTV){v=3;}}
t.rw(true,v);return v;},rw:function(_3,_4){var t=this,n=t.vj$,C=n.VjCookieJar,B=n.Bit;cl=C.readCookie("ebay","sbf");if(!_3){return B.getMulti(cl,40,5);}else{if(_3){_4=(_4==0)?1:_4;C.writeCookielet("ebay","sbf",B.setMulti(cl,40,5,_4));}}}}).endType();
vjo.ctype("vjo.darwin.core.rtm.RTM").needs(["vjo.dsf.typeextensions.string.Comparison","vjo.dsf.cookie.VjCookieJar","vjo.dsf.client.Browser","vjo.dsf.client.ActiveX","vjo.dsf.EventDispatcher","vjo.dsf.flash.Version","vjo.darwin.core.rtm.RTMInit","vjo.dsf.utils.JsLoader","vjo.dsf.utils.CssLoader","vjo.Registry"]).protos({oJSBean:null,aContent:[],iContentLen:0,bTimedOut:false,oTimeoutId:null,iTIMEOUT:300,oClient:null,bResponseReturned:false,iOrd:0,aGlobalNavPlacements:null,promos:null,globalAdOverlay:null,response:null,renderedStatus:null,bBodyLoaded:false,renderCSS:false,bOptimize:false,debug:false,bPromoScript:false,bPromoStyle:false,iBodyLoadedTime:0,constructs:function(_1){this.oJSBean=_1||{};this.oJSBean.quickPids=this.oJSBean.quickPids||[];this.aContent=[];this.iContentLen=0;this.bTimedOut=false;this.oTimeoutId=null;this.iTIMEOUT=3000;this.oClient=vjo.dsf.client.Browser;this.bResponseReturned=false;this.iOrd=(new Date()).getTime();this.aGlobalNavPlacements=null;this.promos=new Object();this.globalAdOverlay=null;vjo.Registry.put("GlobalRtmInstance",this);this.response=null;this.renderedStatus={};this.bBodyLoaded=false;this.renderCSS=false;this.bOptimize=true;this.debug=false;if(this.bOptimize){vjo.dsf.EventDispatcher.add("body","load",this.render,this);}},getFlashVersion:function(){return this.vj$.Version.get();},getEncodingType:function(){return"UTF-8";},setInlineContent:function(_2){if(this.TimedOut()){return;}
this.hidePromos(this.promos);this.aContent=_2;this.iContentLen=this.aContent.length;var _3=false;this.bPromoScript=false;this.bPromoStyle=false;if(!this.renderCSS){for(var j=0;j<this.iContentLen;j++){var _5=this.aContent[j];if(_5.CSSMetaData){this.renderCSS=true;this.processCSS(_5.CSSMetaData);}}}
for(var i=0;i<this.iContentLen;i++){with(this){var _5=aContent[i];if(_5.JSMetaData){this.processJS(_5.JSMetaData);}else{if(!_5.CSSMetaData){this.processPromo(_5);}}}}
var _7=new vjo.dsf.Message("RTM_COMPLETE");_7.status=1;_7.vjRTMObject=this;vjo.dsf.ServiceEngine.handleRequest(_7);if(_3){var _7=new vjo.dsf.Message("PROMO_COMPLETE");_7.vjPromoClientObject=this;vjo.dsf.ServiceEngine.handleRequest(_7);}
return;},TimedOut:function(){this.bResponseReturned=true;if(this.bTimedOut){return true;}
if(this.oTimeoutId){window.clearTimeout(this.oTimeoutId);}
return false;},processJS:function(_8){if(_8.JSURLs){var _9=_8.JSURLs;for(var i=0;i<_9.length;i++){vjo.dsf.utils.JsLoader.load(_9[i]);}}
if(_8.EventHandlers){var _b=_8.EventHandlers;for(i=0;i<_b.length;i++){var _c=document.createElement("script");_c.text=_b[i];document.getElementsByTagName("head")[0].appendChild(_c);}}},processCSS:function(_d){if(_d.CSSURLs){var _e=_d.CSSURLs;for(var i=0;i<_e.length;i++){vjo.dsf.utils.CssLoader.load(_e[i]);}}},processPromoResponse:function(_10){var map=_10.content.data.map;var _12=0;var id="-1";var _14=false;for(var _15 in map){var _16=_15.match(/ME([0-9]+)(.*)/);if(_16){_12++;id=_10.id;}}
var _17=map.VisualPreview;if(_17){this.loadVisualPreview(map.VisualPreview);}
var _18=map.Style;if(_18&&!this.bPromoStyle){this.loadStyle(_18);this.bPromoStyle=true;}
for(var _15 in map){var _16=_15.match(/ME([0-9]+)(.*)/);if(_16){var _19=this.loadPromo(_10.id,_16[2],map[_15],_12>1);if(_19){_14=true;}}}
var _1a=map.Script;if(_1a&&!this.bPromoScript){this.bPromoScript=this.loadScript(_1a);}
if(_14){var _1b=this.getIndex(id);var _1c=this.getUIElement(this.oJSBean.htmlIds[_1b]);if(_1c&&!_1c.length){_1c.style.display="block";}}else{this.processNoneAd(_10);}
return;},loadVisualPreview:function(_1d){var _1e=this.getUIElement("VisualPreviewContent");if(_1e==null){_1e=document.createElement("div");_1e.name="VisualPreviewContent";document.body.appendChild(_1e);}
_1e.innerHTML=_1d;},loadStyle:function(_1f){var _20=document.body.appendChild(document.createElement("style"));_20.setAttribute("type","text/css");if(_20.styleSheet){_20.styleSheet.cssText=_1f;}else{_20.appendChild(document.createTextNode(_1f));}},loadPromo:function(id,_22,_23,_24){var _25=this.getIndex(id);var _26;if(_24){_26=this.oJSBean.merchPrefix+id+_22;}else{_26=this.oJSBean.htmlIds[_25];}
var _27=this.getUIElement(_26);if(this.getStatus(_26)){return true;}
if(_27&&!_27.length){this.addStatus(_26,true);var _28=document.createElement("div");_28.innerHTML=_23;_27.appendChild(_28);return true;}else{this.addStatus(_26,false);}
return false;},hidePromos:function(_29){for(var _2a in _29){var _2b=document.getElementById(_2a);if(_2b!=null){_2b.style.display="none";}}},loadScript:function(_2c){with(window){try{eval(_2c);return true;}
catch(except){}}
return false;},processPopUnderAd:function(pAd){var _2e="height="+pAd.height;_2e+=",width="+pAd.width;_2e+=",menubars=no,scrollbars=no'";var id="p_u_"+pAd.id;var _30=window.open("",id,_2e);if(_30){_30.blur();_30.document.open();_30.document.write(pAd.content);_30.document.close();}
return;},processDoubleClickAd:function(pAd){if(!pAd){return;}
var _32=this.oJSBean;var _33=this.getIndex(pAd.id);var id=_32.htmlIds[_33];var _35=this.getUIElement(id);var url=_32.dblclkUrls[_33];if(!url){return;}
if(this.getStatus(id)){return;}
if(_35&&!_35.length){this.addStatus(id,true);if(pAd.content!=""){url+=pAd.content+";";}
url+="ord="+this.iOrd;_35.innerHTML=this.createIframe(id,url,_32.heights[_33],_32.widths[_33]);_35.style.display="block";}else{this.addStatus(id,false);}
return;},processNoneAd:function(pAd){var _38=this.oJSBean;var _39=this.getIndex(pAd.id);var id=_38.htmlIds[_39];var _3b=this.getUIElement(id);var _3c=_38.defaultUrls[_39];if(this.getStatus(id)){return;}
if(_3b&&!_3b.length){this.addStatus(id,true);if(!_3c||_3c=="collapse"||_3c==""){_3b.style.height="0px";_3b.style.height="0px";_3b.style.display="none";}else{_3b.innerHTML=this.createIframe(id,_3c,_38.heights[_39],_38.widths[_39]);}}else{this.addStatus(id,false);}
return;},processHTMLAd:function(pAd){if(pAd.height=="-1"||pAd.height=="9999"){pAd.height="auto";}
if(pAd.width=="-1"||pAd.width=="9999"){pAd.width="auto";}
var _3e=this.getIndex(pAd.id);var _3f=this.getUIElement(this.oJSBean.htmlIds[_3e]);if(this.getStatus(this.oJSBean.htmlIds[_3e])){return;}
if(_3f&&!_3f.length){this.addStatus(this.oJSBean.htmlIds[_3e],true);var _40=_3f.style,h,w;_40.height=h=(pAd.height.has("auto"))?pAd.height:pAd.height+"px";_40.width=w=(pAd.width.has("auto"))?pAd.width:pAd.width+"px";if((w!="auto")&&(h!="auto")){if(pAd.expand){_40.textAlign="left";}else{_40.overflow="hidden";}}
_3f.innerHTML=pAd.content;_3f.style.display="block";var id=pAd.id;if(id==184||id==188||id==218||id==569||id==570){this.executeScript(id);}}else{this.addStatus(this.oJSBean.htmlIds[_3e],false);}
return;},executeScript:function(id){var _43="rtm_html_"+id;var _44=document.getElementById(_43).getElementsByTagName("script")[0];if(_44!=null){var _45=document.createElement("script");_45.type="text/javascript";if(_44.text){_45.text=_44.text;}else{_45.textContent=_44.textContent;}
document.getElementById(_43).appendChild(_45);}},processHTMLFormAd:function(pAd){if(pAd.height=="-1"||pAd.height=="9999"){pAd.height="auto";}
if(pAd.width=="-1"||pAd.width=="9999"){pAd.width="auto";}
var _47=this.getIndex(pAd.id);var _48=this.getUIElement(this.oJSBean.htmlIds[_47]);if(this.getStatus(this.oJSBean.htmlIds[_47])){return;}
if(_48&&!_48.length){this.addStatus(this.oJSBean.htmlIds[_47],true);var _49=_48.style,h,w;_49.height=h=(pAd.height.has("auto"))?pAd.height:pAd.height+"px";_49.width=w=(pAd.width.has("auto"))?pAd.width:pAd.width+"px";if(h!="auto"&&w!="auto"){_49.overflow="hidden";}
var _4a=document.createElement("iframe");_4a.setAttribute("hspace",0);_4a.setAttribute("vspace",0);_4a.setAttribute("width","100%");_4a.setAttribute("frameBorder",0);_4a.setAttribute("scrolling","no");_4a.setAttribute("marginWidth",0);_4a.setAttribute("marginHeight",0);_48.appendChild(_4a);_4a.doc=null;if(_4a.contentDocument){_4a.doc=_4a.contentDocument;}else{if(_4a.contentWindow){_4a.doc=_4a.contentWindow.document;}else{if(_4a.document){_4a.doc=_4a.document;}}}
if(_4a.doc==null){throw"Document not found, append the parent element to the DOM before creating the IFrame";}
_4a.doc.open();try{_4a.doc.write(pAd.content);}
finally{_4a.doc.close();}}else{this.addStatus(this.oJSBean.htmlIds[_47],false);}
return;},getUIElement:function(_4b){var s=_4b,d=window.document;if(d.getElementById){return d.getElementById(s);}else{if(d.all){return d.all(s);}}
return null;},getIndex:function(pId){for(var i=0;i<this.iContentLen;i++){if(this.aContent[i].id==pId){return i;}}
return;},createIframe:function(pId,_50,_51,_52){var _53="ifrm_"+pId;var f="<iframe frameborder=\"no\" border=\"0\" marginwidth=\"0\" marginheight=\"0\" scrolling=\"no\""+" id=\""+_53+"\""+" name=\""+_53+"\""+" src=\""+_50+"\""+" width=\""+_52+"\" height=\""+_51+"\"></iframe>";return f;},appendUrl:function(_55){this.oJSBean.url+=_55;},processGlobalNavPids:function(_56){var _57="";var _58;if(typeof(_oGlobalNavRTMInfo)!="undefined"){_58=_oGlobalNavRTMInfo;this.aGlobalNavPlacements=_58.aRTMPlacementData;}
if(_58&&this.aGlobalNavPlacements&&this.aGlobalNavPlacements.length>0){var _59=_oGlobalNavRTMInfo.aRTMPlacementData,data;for(i=0;i<_59.length;i++){if(_56){data=_59[i];}
if(_57){_57+=":";}
_57+=_56?data.pid:"0";}}
return _57;},registerGlobalNavPlacements:function(){if(!this.aGlobalNavPlacements||this.aGlobalNavPlacements.length==0){return;}
var _5a=this.oJSBean;var len=this.aGlobalNavPlacements.length;for(var i=0;i<len;i++){var _5d=_5a.htmlIds.length;var _5e=this.aGlobalNavPlacements[i];_5a.htmlIds[_5d]=(_5e.htmlId)?_5e.htmlId:"glbl_nav_no_html_id";_5a.pids[_5d]=(_5e.pid)?_5e.pid:"glbl_nav_no_pid";_5a.heights[_5d]=(_5e.maxHeight)?_5e.maxHeight:"glbl_nav_no_height";_5a.widths[_5d]=(_5e.maxWidth)?_5e.maxWidth:"glbl_nav_no_width";_5a.dblclkUrls[_5d]=(_5e.dblclkUrl)?_5e.dblclkUrl:"glbl_nav_no_dblclk";_5a.defaultUrls[_5d]=(_5e.defaultUrl)?_5e.defaultUrl:"collapse";if(_5e.renderBeforeOnload&&_5a.quickPids){_5a.quickPids[_5a.quickPids.length]=_5e.pid;}}},setTimeOut:function(){if(!this.bResponseReturned){var _5f="vjo.darwin.core.rtm.RTM.processTimeOut()";var _60=this;var _61=function(){_60.processTimeOut();};this.oTimeoutId=window.setTimeout(_61,this.iTIMEOUT);}},processTimeOut:function(){this.bTimedOut=true;var _62=this.oJSBean;var _63=_62.defaultUrls.length;for(var i=0;i<_63;i++){var id=_62.htmlIds[i];var _66=this.getUIElement(id);var _67=_62.defaultUrls[i];if(this.getStatus(id)){continue;}
if(_66&&!_66.length){this.addStatus(id,true);if(!_67||_67=="collapse"||_67==""){_66.style.height="0px";_66.style.height="0px";_66.style.display="none";}else{_66.innerHTML=this.createIframe(id,_67,_62.heights[i],_62.widths[i]);}}else{this.addStatus(id,false);}}
var _68=Math.random();if(_68<0.05){var _69=_62.url;var _6a=_69.indexOf("&");_69=_69.substring(0,_6a);_69=_69+"&p="+_62.pids.join(":")+"&ite=2&to="+this.iTIMEOUT;_69=_69.replace("RtmCmd","RtmIt");var _6b=document.createElement("script");_6b.type="text/javascript";_6b.src=_69;document.getElementsByTagName("head")[0].appendChild(_6b);}
window.clearTimeout(this.oTimeoutId);var _6c=new vjo.dsf.Message("RTM_COMPLETE");_6c.status=0;_6c.vjRTMObject=this;vjo.dsf.ServiceEngine.handleRequest(_6c);return;},getBrowserWidth:function(){var b=this.oClient;var _6e=document.body.clientWidth;if(!b.bIE){_6e=window.innerWidth;}
return _6e;},popUp:function(_6f,_70,_71,_72,_73,_74,_75,_76,_77,_78,_79,_7a){var sP="";sP+=(_71!=null)?",width="+_71:"";sP+=(_72!=null)?",height="+_72:"";sP+=(_76!=null)?",screenX="+_76+",left="+_76:"";sP+=(_77!=null)?",screenY="+_77+",top="+_77:"";sP+=",toolbar="+((_74)?"1":"0");sP+=",location="+((_78)?"1":"0");sP+=",status="+((_73)?"1":"0");sP+=",scrollbars="+((_75)?"1":"0");sP+=",resizable="+((_79)?"1":"0");sP+=",menubar="+((_7a)?"1":"0");if(sP.length>0){sP=sP.substring(1);}
window.open(_70,_6f,sP);return false;},getSegment:function(_7c){var oCJ=vjo.dsf.cookie.VjCookieJar,e=oCJ.readCookie("etfc"),r=oCJ.readCookie("reg"),s=oCJ.readCookie("ebay","sin"),c,n;if(e=="0"){n="3";c="E";}else{if(e=="1"){n="4";c="C";}else{if(e=="2"){n="5";c="D";}else{if((e==""&&(r!=""&&r!=";"))||s=="in"||e=="5"){n="2";c="B";}else{n="1";c="A";}}}}
return _7c?c:n;},getCIDCookie:function(){var oCJ=vjo.dsf.cookie.VjCookieJar;var cid=oCJ.readCookie("npii","cguid");if(cid!="undefined"&&cid!=""){return cid;}},openReportAd:function(_80,_81,_82,_83,_84,_85,_86,_87,_88,_89,_8a){if(this.globalAdOverlay==null){this.globalAdOverlay=new vjo.darwin.core.rtm.ReportAd(_82,_83,_84,_85,_88,_89,_81,_8a);}
if(this.globalAdOverlay!=null){this.globalAdOverlay.init(_86,_87);this.globalAdOverlay.open(_80);}},closeReportAd:function(){if(this.globalAdOverlay!=null){if(this.globalAdOverlay){this.globalAdOverlay.close();}}},submitReportAd:function(){if(this.globalAdOverlay!=null){if(this.globalAdOverlay){this.globalAdOverlay.submitReport();}}},yahooAdBckGrnd:function(_8b){var _8c=_8b;var oFr=document.getElementById(_8c);oFr.style.visibility="visible";oFr.parentNode.style.backgroundColor="transparent";},addStatus:function(_8e,_8f){this.renderedStatus[_8e]=_8f;},getStatus:function(_90){return this.renderedStatus[_90];},storeResponse:function(_91){var t=this,c=_91,qp;t.response=c;t.aContent=c;t.iContentLen=c.length;if(t.bBodyLoaded){t.setInlineContent(c);}else{if(!this.TimedOut()){qp=t.oJSBean.quickPids;if(qp&&qp.length>0){t.renderPromos(qp);}}}
if(_91!=null){for(var i=0;i<_91.length;i++){if(_91[i]!=null&&_91[i].content!=null&&_91[i].content.indexOf("openReportAd")>0){this.addReportAdScript();break;}}
if(!t.renderCSS){for(var j=0;j<_91.length;j++){var _95=_91[j];if(_95.CSSMetaData){t.renderCSS=true;t.processCSS(_95.CSSMetaData);}}}}
if(_91!=null&&_91.indexOf("openReportAd")>0){this.addReportAdScript();}},addReportAdScript:function(){try{if(this.oJSBean.reportAdJsUrl){var _96=document.createElement("script");_96.type="text/javascript";_96.src=this.oJSBean.reportAdJsUrl;document.getElementsByTagName("head")[0].appendChild(_96);}}
catch(er){}},render:function(){var t=this,r=t.response;if(r){t.setInlineContent(r);}
t.bBodyLoaded=true;t.iBodyLoadedTime=(new Date()).getTime();},processPromo:function(_98){with(this){if(_98.type=="doubleclick"){processDoubleClickAd(_98);}else{if(_98.type=="html"){processHTMLAd(_98);}else{if(_98.type=="popUnder"){processPopUnderAd(_98);}else{if(_98.type=="promo"){bHasPromo=true;processPromoResponse(_98);}else{if(_98.type=="htmlform"){processHTMLFormAd(_98);}else{processNoneAd(_98);}}}}}}},renderPromos:function(_99){var r=this.response,i,j;if(r){for(j=0;j<r.length;j++){for(i=0;i<_99.length;i++){if(r[j].id==_99[i]+""){this.processPromo(r[j]);}}}}},decodeMsg:function(msg){msg=unescape(msg);if(msg.charAt(0)=="#"){msg=msg.substring(1);}
msg="|"+msg;return msg;},getMsgParam:function(msg,_9d,_9e){var v=null;var s=msg.indexOf("|"+_9d+":");if(s>-1){s=s+_9d.length+2;var e=msg.indexOf("|",s);if(msg.charAt(s)=="{"){e=msg.indexOf("}",s)+1;}
if(e<0){e=msg.length;}
v=msg.substring(s,e);}
if(_9e||(typeof(_9e)=="number")){if(!v||(v=="")){v=_9e;}}
return v;},getIDFromMsg:function(msg){var id=this.getMsgParam(msg,"id",null);var i=id.lastIndexOf("_");if(i>-1){id=id.substring(i+1);}
return id;},setParentOverflowByClass:function(_a5,_a6,_a7){var e=_a5;while(e.parentNode){if(e.className==_a6){e.style.overflow=_a7;break;}
e=e.parentNode;}},throwError:function(_a9){throw new Error("eBayAdsRuntime - "+_a9);},processMsg:function(_aa,pid,_ac,ad,msg){var _af=(_aa=="ad.frame.expand");if(_af&&!ad.allowExpandOnLoad){var _b0=this.iBodyLoadedTime;var end=(new Date()).getTime();var _b2=3000;if(_b0&&((end-_b0)<_b2)){this.throwError("can't expand right after page onload");}}
if(_af||(_aa=="ad.frame.collapse")){var _b3="rtm_iframe_"+pid;var _b4="rtm_html_"+pid;var ifr=document.getElementById(_b3);var div=document.getElementById(_b4);if(!ifr){_b3="ifrm_"+pid;ifr=document.getElementById(_b3);}
if(!div){_b4="rtm_div_"+pid;div=document.getElementById(_b4);if(!div){_b4="single_rtm_"+pid;div=document.getElementById(_b4);}}
var dir="";var _b8=ad.width;var _b9=ad.height;if(_af){dir=this.getMsgParam(msg,"direction",null);if(!dir){this.throwError("no direction");}
dir=dir.toLowerCase();var _ba=ad.expand.toLowerCase();if(!ad.expand||(_ba.indexOf(dir)<0)){this.throwError("wrong direction: "+dir+", should be: "+_ba);}
var _b8=this.getMsgParam(msg,"width",null);if(_b8>ad.maxWidth){this.throwError("width exceeds max: "+_b8+", should be < "+ad.maxWidth);}
var _b9=this.getMsgParam(msg,"height",null);if(_b9>ad.maxHeight){this.throwError("height exceeds max: "+_b9+", should be < "+ad.maxHeight);}}
if(!ifr||!div){this.throwError("no div or iframe");}
var i=dir.indexOf("-");if(i>-1){if(dir.indexOf("-",i+1)>-1){this.throwError("invalid direction: "+dir);}else{this.processMsgDir(dir.substring(0,i),_af,ifr,div,_b8,_b9);this.processMsgDir(dir.substring(i+1),_af,ifr,div,_b8,_b9);}}else{this.processMsgDir(dir,_af,ifr,div,_b8,_b9);}
var url=this.oJSBean.url;var i=url.indexOf("?");if(i>-1){url=url.substring(0,i);}
var _bd=7;if(_af){_bd=6;}
var sb=[];sb.push(url+"?RtmIt");sb.push("&ite="+_bd);sb.push("&m="+_ac);sb.push("&ord="+(new Date()).getTime());var i=new Image();i.src=sb.join("");}},processMsgDir:function(dir,_c0,ifr,div,_c3,_c4){var ie=(typeof(window.addEventListener)=="undefined");var ie6=(navigator.userAgent.indexOf("MSIE 6.")>-1);var ff=(navigator.userAgent.indexOf("Firefox")>-1);if(_c0){ifr.style.zIndex=9000;if(ie6){div.style.overflow="hidden";}else{if(ie){div.style.overflow="visible";}}
if(dir=="up"){ifr.style.position="relative";var ot=_c4-ifr.height;ifr.height=_c4;ifr.style.top=-ot+"px";}else{if(dir=="down"){ifr.height=_c4;}else{if(dir=="left"){ifr.style.position="relative";var ol=_c3-ifr.width;ifr.width=_c3;ifr.style.left=-ol+"px";}else{if(dir=="right"){ifr.width=_c3;this.setParentOverflowByClass(ifr,"ff-left","visible");}else{this.throwError("invalid direction: "+dir);}}}}}else{ifr.style.zIndex=0;if(ie){div.style.overflow="hidden";}
var _ca=((ifr.width==_c3)&&(Math.abs(_c4-ifr.height)<30));ifr.style.position="absolute";ifr.style.top=null;ifr.style.left=null;ifr.width=_c3;ifr.height=_c4;this.setParentOverflowByClass(ifr,"ff-left","hidden");if(_ca){this.throwError("already collapsed");}}},onMessage:function(msg){try{msg=this.decodeMsg(msg);var pid=this.getIDFromMsg(msg);var st=this.getMsgParam(msg,"st",null);var _ce=this.getMsgParam(msg,"m",null);if(!pid||!st||!_ce){this.throwError("no id, st, or m");}
var i=this.getIndex(pid);if(typeof(i)!="number"){this.throwError("no placement "+pid);}
var ad=this.aContent[i];if(!ad){this.throwError("no ad "+pid);}
if(st!=ad.st){this.throwError("token mismatch");}
var _d1=this.getMsgParam(msg,"topic",null);this.processMsg(_d1,pid,_ce,ad,msg);}
catch(e){if(this.debug){throw e;}}}}).props({siteCatalyst:function(_d2,_d3,PID,_d5){var esc=vjo.darwin.tracking.sitecatalyst;if(esc){var _d7="Search",loc="North";if(PID==188||PID==570){_d7="Browse";}
if(PID==569||PID==570){loc="Sky";}
var s=esc.EbaySiteCatalyst.s;s.linkTrackVars="prop27";s.linkTrackEvents="None";s.prop27=_d3+":"+_d7+":"+PID+":"+_d5;if(_d2){s.linkTrackVars="prop28";s.prop28=s.prop27;s.prop27=null;}
s.tl(true,"o","RTM "+loc+" Ad");}}}).endType();
vjo.ctype("vjo.dsf.Json").endType();if(!this.JSON){JSON=function(){function f(n){return n<10?'0'+n:n;}
Date.prototype.toJSON=function(key){return this.getUTCFullYear()+'-'+
f(this.getUTCMonth()+1)+'-'+
f(this.getUTCDate())+'T'+
f(this.getUTCHours())+':'+
f(this.getUTCMinutes())+':'+
f(this.getUTCSeconds())+'Z';};var cx=/[\u0000\u00ad\u0600-\u0604\u070f\u17b4\u17b5\u200c-\u200f\u2028-\u202f\u2060-\u206f\ufeff\ufff0-\uffff]/g,escapeable=/[\\\"\x00-\x1f\x7f-\x9f\u00ad\u0600-\u0604\u070f\u17b4\u17b5\u200c-\u200f\u2028-\u202f\u2060-\u206f\ufeff\ufff0-\uffff]/g,gap,indent,meta={'\b':'\\b','\t':'\\t','\n':'\\n','\f':'\\f','\r':'\\r','"':'\\"','\\':'\\\\'},rep;function quote(string){escapeable.lastIndex=0;return escapeable.test(string)?'"'+string.replace(escapeable,function(a){var c=meta[a];if(typeof c==='string'){return c;}
return'\\u'+('0000'+
(+(a.charCodeAt(0))).toString(16)).slice(-4);})+'"':'"'+string+'"';}
function str(key,holder){var i,k,v,length,mind=gap,partial,value=holder[key];if(value&&typeof value==='object'&&typeof value.toJSON==='function'){value=value.toJSON(key);}
if(typeof rep==='function'){value=rep.call(holder,key,value);}
switch(typeof value){case'string':return quote(value);case'number':return isFinite(value)?String(value):'null';case'boolean':case'null':return String(value);case'object':if(!value){return'null';}
gap+=indent;partial=[];if(typeof value.length==='number'&&!(value.propertyIsEnumerable('length'))){length=value.length;for(i=0;i<length;i+=1){partial[i]=str(i,value)||'null';}
v=partial.length===0?'[]':gap?'[\n'+gap+
partial.join(',\n'+gap)+'\n'+
mind+']':'['+partial.join(',')+']';gap=mind;return v;}
if(rep&&typeof rep==='object'){length=rep.length;for(i=0;i<length;i+=1){k=rep[i];if(typeof k==='string'){v=str(k,value,rep);if(v){partial.push(quote(k)+(gap?': ':':')+v);}}}}else{for(k in value){if(Object.hasOwnProperty.call(value,k)){v=str(k,value,rep);if(v){partial.push(quote(k)+(gap?': ':':')+v);}}}}
v=partial.length===0?'{}':gap?'{\n'+gap+
partial.join(',\n'+gap)+'\n'+
mind+'}':'{'+partial.join(',')+'}';gap=mind;return v;}}
return{stringify:function(value,replacer,space){var i;gap='';indent='';if(typeof space==='number'){for(i=0;i<space;i+=1){indent+=' ';}}else if(typeof space==='string'){indent=space;}
rep=replacer;if(replacer&&typeof replacer!=='function'&&(typeof replacer!=='object'||typeof replacer.length!=='number')){throw new Error('JSON.stringify');}
return str('',{'':value});},parse:function(text,reviver){var j;function walk(holder,key){var k,v,value=holder[key];if(value&&typeof value==='object'){for(k in value){if(Object.hasOwnProperty.call(value,k)){v=walk(value,k);if(v!==undefined){value[k]=v;}else{delete value[k];}}}}
return reviver.call(holder,key,value);}
cx.lastIndex=0;if(cx.test(text)){text=text.replace(cx,function(a){return'\\u'+('0000'+
(+(a.charCodeAt(0))).toString(16)).slice(-4);});}
if(/^[\],:{}\s]*$/.test(text.replace(/\\["\\\/bfnrtu]/g,'@').replace(/"[^"\\\n\r]*"|true|false|null|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?/g,']').replace(/(?:^|:|,)(?:\s*\[)+/g,''))){j=eval('('+text+')');return typeof reviver==='function'?walk({'':j},''):j;}
throw new SyntaxError('JSON.parse');}};}();}
vjo.ctype("vjo.util.List").protos({constructs:function(_1){this.javaClass=_1||"java.util.ArrayList";this.list=[];},get:function(_2){if(this.size()>_2){return this.list[_2];}
return null;},add:function(_3){return this.list[this.size()]=_3;},remove:function(_4){var _5=this.size(),tmp=this.list,nlist=[],rv=false;for(var i=0;i<_5;i++){if(!rv&&tmp[i]===_4){rv=true;}else{nlist[nlist.length]=tmp[i];}}
this.list=nlist;return rv;},size:function(){return this.list.length;}}).endType();
vjo.ctype("vjo.util.Map").protos({constructs:function(_1){this.javaClass=_1||"java.util.HashMap";this.map={};},get:function(_2){return this.map[_2];},put:function(_3,_4){return this.map[_3]=_4;},remove:function(_5){var _6=this.map[_5];delete this.map[_5];return _6;},size:function(){var _7=0;for(var _8 in this.map){_7++;}
return _7;}}).endType();
vjo.ctype("vjo.dsf.Enc").props({unescape:vjo.NEEDS_IMPL,decodeURI:vjo.NEEDS_IMPL,decodeURIComponent:vjo.NEEDS_IMPL,encodeURIComponent:vjo.NEEDS_IMPL,encodeURI:vjo.NEEDS_IMPL}).inits(function(){vjo.dsf.Enc.unescape=window.unescape;vjo.dsf.Enc.decodeURI=window.decodeURI;vjo.dsf.Enc.decodeURIComponent=window.decodeURIComponent;vjo.dsf.Enc.encodeURIComponent=window.encodeURIComponent;vjo.dsf.Enc.encodeURI=window.encodeURI;}).endType();
vjo.needs("vjo.dsf.Json");vjo.ctype("vjo.dsf.Service").needs(["vjo.dsf.ServiceResponse","vjo.dsf.Enc","vjo.dsf.SvcErr","vjo.dsf.Message","vjo.dsf.ServiceRegistry"]).props({callback:function(_1,_2){if(_1.readyState!=4){return;}
_2.response=new vjo.dsf.ServiceResponse();try{if(_1.status>=200&&_1.status<300){_2.response.data=_1.responseText;var _3=vjo.dsf.ServiceRegistry.getBinding(_2.svcConfig.respMarshalling);if(_3){_3.deserialize(_2);_2.status=1;}else{this.E.err(_2,this.E.InvRspBnd,"respMarshalling="+_2.svcConfig.respMarshalling);}}else{var _4=this.E.RqUnk;if(404===_1.status){_4=this.E.SvcPrc;}
this.E.err(_2,_4,"status="+_1.status+":readyState="+_1.readyState);}}
catch(e){this.E.err(_2,this.E.RqUnk,"status="+_1.status+":readyState="+_1.readyState);}
vjo.dsf.ServiceEngine.handleResponse(_2);delete _1.onreadystatechange;_1=null;},getXmlHttpReq:function(){var _5=false;try{_5=new ActiveXObject("Msxml2.XMLHTTP");}
catch(e){try{_5=new ActiveXObject("Microsoft.XMLHTTP");}
catch(e){_5=false;}}
if(!_5&&typeof XMLHttpRequest!="undefined"){_5=new XMLHttpRequest();}
return _5;},getClientInfo:function(){if(this.clientInfo){return this.clientInfo;}
var nv=navigator,agt=nv.userAgent.toLowerCase(),i=0,ver=0,b="";if(agt.indexOf("firefox")!=-1){b="Firefox";i=agt.lastIndexOf("firefox")+8;}else{if((/webkit|khtml/).test(agt)){b="Safari";i=agt.lastIndexOf("safari")+7;}else{if(typeof(window.opera)!="undefined"){b="Opera";i=agt.lastIndexOf("opera")+6;}else{if(nv.appName=="Netscape"){b="Netscape";i=agt.lastIndexOf("/")+1;}else{if(agt.indexOf("msie")!=-1){b="IE";i=agt.indexOf("msie")+4;}}}}}
if(b){ver=parseInt((b=="Opera")?window.opera.version():agt.substring(i));}
this.clientInfo=b+":"+ver+":";return this.clientInfo;},generateReqParams:function(_7){var _8="svcid="+vjo.dsf.Enc.encodeURIComponent(_7.svcId);if(_7.stok){_8+="&stok="+_7.stok;}
if(_7.pId){_8+="&pId="+_7.pId;}
if(_7.v){_8+="&v="+_7.v;}
_8=_8+"&reqttype="+_7.svcConfig.reqtMarshalling;_8=_8+"&resptype="+_7.svcConfig.respMarshalling;_8=_8+"&clientType="+this.getClientInfo();_8+="&request=";var _9=_7.request,reqtmarsh=_7.svcConfig.reqtMarshalling;if("JSCALLBACK"==reqtmarsh){_8+=vjo.dsf.Enc.encodeURIComponent(JSON.stringify(_9));}
return _8;}}).inits(function(){this.E=this.vj$.SvcErr;}).endType();
// en_US/e655/SYS-D9_vjo_e65510597155_1_en_US
// b=10597155