
(function(){var _1=1.5,_bOldSupport=typeof(vjo)=="object";if(_bOldSupport&&vjo._v==_1){return;}
var _2={};var _3=(typeof window!="undefined")?window:this,wv=_3.vjo;if(!wv||!wv._v||wv._v<_1){_3.vjo=_2;}
_extend(_2,{loader:null,_v:1.5,_global:this,_bScope:null,_typeMap:{},_bSubClass:false,_callStack:[],isArray:function(_4){if(!_4){return false;}
return(_4.constructor==Array||(typeof _4=="object"&&_4.join&&_4.splice));},getType:function(_5){if(!_5){return;}
var _6=_2._typeMap[_5],idx=_5.lastIndexOf("."),cn=(idx>-1)?_5.substring(idx+1):_5;if(_6){return _6.pkg[cn];}
return;},mixin:function(_7,_8){var _9=this.getType(_7);if(_9){if(_9.vj$&&_9.vj$._type==="mtype"){if(_9._props){throw"cannot mixin static props to an instance";}
this.extend(_8,_9._protos);}}},hitch:function(_a,fn,_c){if("function"!=(typeof fn)){var _d=fn;fn=_a;_a=_d;}
return function(){return fn.apply((_c&&this!=window)?this:_a,arguments);};},curry:function(fn){var _f=Array.prototype.slice;var _10=_f.apply(arguments);_10.shift();return function(){return fn.apply(this,_10.concat(_f.apply(arguments)));};},bind:function(ctx,fn){var _13=Array.prototype.slice;var _14=_13.apply(arguments);_14.shift();_14.shift();return function(){return fn.apply(ctx,_14.concat(_13.apply(arguments)));};},create:function(_15,_16){var _17=document.createElement(_15);_17.innerHTML=_16;return _17;},make:function(_18,clz){var a=arguments,len=a.length,tp=(typeof clz=="function"&&clz.vj$)?clz:this.getType(clz),_vjo={};if(len<2||!tp){throw"context and valid type are required";}
_forEach(_18.vj$,function(val,key){if(typeof val=="function"&&val.vj$&&val.vj$._type){this[key]=val;}},_vjo);var _1d=Array.prototype.slice.call(a,2,len);return{_protos:null,protos:function(obj){this._protos=obj;return this;},endType:function(){var t=_2.ctype(),clztype=tp.vj$._type,rv;if(clztype==="itype"){t.satisfies(tp);}else{if(clztype==="ctype"||clztype==="atype"){var _20=TypeMeta.get(tp.vj$._meta._metaId);if(_20&&!_20._completed){_20.complete();}
t.inherits(_2.getType(tp.vj$._class));}else{throw"incompatible anonomyous type";}}
t.protos(this._protos);t.complete();t=t.endType();t.__donotconstruct=true;rv=new t;delete t.__donotconstruct;rv.vj$=t.vj$;_2.extend(rv.vj$,_vjo);(function(){if(this.base){this.base.apply(this,arguments);}}).apply(rv,_1d);if(rv.base&&rv._getBase){var fn=function(){};fn.prototype=rv._getBase();rv.base=new fn;rv.base._parent=rv;}
rv.vj$.parent=(tp._outer)?tp._outer:_18;rv.vj$.outer=tp._outer;if(this._protos&&this._protos.constructs){this._protos.constructs.apply(rv);}
return rv;}};},needs:function(clz,_23){if(!clz){return;}
if(!this._bScope){this._bScope={};}
var _24=_createPkg(clz),cls=_24.className,tp=_24.pkg[cls];if(!tp){if(this.loader!=null){var _25=this._bScope;this._bScope=null;this.loader.load(clz);this._bScope=_25;tp=_24.pkg[cls];}}
if(tp){if(typeof _23=="string"&&_23!=""){cls=_23;}
this._bScope[cls]=tp;}},getVersion:function(){return _2._global._scope_name||"";},versionJsText:function(txt){var scp=this.getVersion();if(scp){var str="(function(vjo){"+txt;str+="}).apply("+scp+",["+scp+".vjo]);";return str;}
return txt;},forEach:_forEach,extend:_extend,needsLib:function(){},createArray:function(val,_2a){var arr=[];if(arguments.length>1){for(var ii=0;ii<_2a;ii++){if(arguments.length>2){var tmp=[val];for(var k=2;k<arguments.length;k++){tmp[tmp.length]=arguments[k];}
arr[ii]=_2.createArray.apply(this,tmp);}else{arr[ii]=val;}}}
return arr;},isInstanceOf:function(_2f,_30){if(_2f===null){return false;}
var _31=_isVjoType(_30);if(!_31||(_31&&_30.vj$._type!="itype")){return _2f instanceof _30;}
if(!_2f.vj$){return false;}
if(_isInstanceForInterface(_2f.getClass(),_30)){return true;}
var id=_2f.vj$._meta._metaId,meta;for(;;){meta=TypeMeta.get(id);if(meta){var ihs=meta._inherits;if(ihs.length==1&&_2.getType(ihs[0])){id=ihs[0];if(_isInstanceForInterface(_2.getType(ihs[0]).clazz,_30)){return true;}}else{break;}}else{break;}}
return false;}});_2.NEEDS_IMPL=function(){throw"needs implementation";};_2.Object=function(){this.vj$={_class:"vjo.Object",_type:"ctype",Object:_2.Object,_meta:{}};};_2.Object.prototype={_hashCode:-1,constructs:function(){},getClass:_getClazz,hashCode:function(){if(this._hashCode==-1){this._hashCode=++_2.Object._hashCounter;}
return this._hashCode;},equals:function(o){return(this===o);},toString:function(){return this.getClass().getName()+"@"+this.hashCode().toString(16);}};_2.extend(_2.Object,{vj$:{_class:"vjo.Object",_type:"ctype",Object:_2.Object,_meta:{}},_hashCounter:0,isInstance:function(o){return _2.isInstanceOf(o,this);}});_2.Class=function(clz,typ){this._name=clz,this._type=typ||"ctype",this._satisfied=[];this.vj$={_class:"vjo.Class",_type:"ctype",Class:_2.Class,_meta:{}};};_2.Class.prototype=new _2.Object();_2.extend(_2.Class.prototype,{getName:function(){var n=this._name;if(n){return n;}
return null;},getSimpleName:function(){var n=this.getName();var idx=n.lastIndexOf(".");if(idx!=-1){n=n.substring(idx+1);if(n){idx=n.lastIndexOf("$");if(idx>0){n=n.substring(idx+1);}}}
return n;},getPackageName:function(){var n=this.getName();if(n!=null){var i=n.lastIndexOf(".");if(i>=0){return n.substring(0,i);}}
return"";},isInterface:function(){return(this._type==="itype");},isInstance:function(obj){var tp=_2.getType(this.getName());if(tp){return _2.isInstanceOf(obj,tp);}
return false;},toString:function(){return(this.isInterface()?"interface ":"class ")+this.getName();},getClass:_getClazz});_2.extend(_2.Class,{forName:function(clz){var err="Type not found for '"+clz+"'";try{var o=eval(clz);if(o&&o.clazz){return o.clazz;}}
catch(e){}
throw err;},isInstance:function(o){return _2.isInstanceOf(o,_2.Class);},clazz:new _2.Class("vjo.Class","ctype")});_2.Object.clazz=new _2.Class("vjo.Object","ctype");_2.obj=function(_43,_44){if(typeof(_43)=="undefined"||_43==null||_43===""){throw"Invalid type name";}
if(!_44){return null;}
var _45,err="Invalid type '"+_43+"'.";try{_45=eval(_43);}
catch(e){throw err;}
if(!_45){return err;}
var o=new _45();for(var i in _44){o[i]=_44[i];}
return o;};var _48={map:{},inits:{},needs:{},stack:[],inners:{},loaded:{},addType:function(clz){if(!this.map[clz]){this.map[clz]=[];}},addDep:function(clz,dep){if(!clz){return;}
this.addType(clz);var aD=this.map[clz];aD[aD.length]=dep;if(_2.loader){var stk=this.stack;if(stk.length==0){stk.push(clz);}
if(stk[stk.length-1]===clz){stk.push(dep);}}},addInner:function(clz,fn){if(!this.inners[clz]){this.inners[clz]=[];}
var ins=this.inners[clz];ins.push(fn);},execInners:function(clz){var ins=this.inners[clz];if(ins){var len=ins.length;for(var i=0;i<len;i++){var _55=ins[i];if(_55){_55.complete();}}}
this.inners[clz]=null;},removeDep:function(clz,dep){if(!_2.loader){return;}
var stk=this.stack;if(stk[stk.length-1]===dep){stk.pop();}},addNeeds:function(clz){var n=this.needs[clz];if(n){while(n.length>0){n.pop()();}
this.needs[clz]=null;}},deferNeed:function(clz,fn){var n=this.needs;if(!n[clz]){n[clz]=[];}
n[clz].push(fn);},register:function(clz,fn){this.inits[clz]=fn;return true;},load:function(clz){if(!clz||this.loaded[clz]){return false;}
var _61=this.map[clz],ins=this.inits;if(_2.loader){var stk=this.stack,len=stk.length;if(len>0&&stk[len-1]===clz){stk.pop();}}
if(_61&&!this.hasCirDep(clz)){var stk=[];this._pushDep(clz,stk,{});while(stk.length>0){var dep=stk.pop(),fn=ins[dep];this.addNeeds(dep);if(fn){fn.complete();}
ins[dep]=null;this.execInners(dep);this.loaded[dep]=true;}}
if((!_61||_61.length==0)){this.addNeeds(clz);if(ins[clz]){ins[clz].complete();ins[clz]=null;}
this.execInners(clz);this.loaded[clz]=true;}
return true;},_pushDep:function(clz,_65,_66){var arr=this.map[clz];_65.push(clz);_66[clz]=true;if(!arr||arr.length===0){return;}
var len=arr.length,i=0;for(;i<len;i++){var key=arr[i];if(!_66[key]){this._pushDep(key,_65,_66);}}
return;},_hasCirDep:function(clz,_6b,_6c){var arr=this.map[_6b];if(!arr||arr.length===0){return false;}
var len=arr.length,i=0;for(;i<len;i++){var key=arr[i];if(_2.loader){if(key===clz){return true;}}else{if(!_2.getType(key)){return true;}}
if(!_6c[key]){_6c[key]=true;if(this._hasCirDep(clz,key,_6c)){return true;}}}
return false;},hasCirDep:function(clz){var aD=this.map[clz],bInline=(!_2.loader);if(aD){var len=aD.length;if(bInline){return this._hasCirDep(clz,clz,{});}else{var stk=this.stack,len2=stk.length;for(var i=0;i<len;i++){var dep=aD[i];for(var j=0;j<len2;j++){if(stk[j]===dep){return true;}}}
if(len2>0){return this._hasCirDep(clz,clz,{});}}}
return false;}};var _77={needs:function(clz,_79){if(!clz||this.vj$._meta._isInner){return this;}
var _7a=[],useAlias=false;if(typeof clz=="string"){_7a=[clz];useAlias=(_79)?true:false;}else{if(_2.isArray(clz)){_7a=clz;}else{return this;}}
_forEach(_7a,function(val,key,obj){var cl=val,pObj=_2._typeMap[val],idx=cl.lastIndexOf("."),cn=(idx>-1)?cl.substring(idx+1):cl,tp=(pObj)?pObj.pkg[cn]:null;_48.addDep(this.vj$._class,cl);if(!tp&&_2.loader){var _7f=_2._bScope;_2._bScope=null;_2.loader.load(cl);_2._bScope=_7f;pObj=_2._typeMap[cl];tp=(pObj)?pObj.pkg[cn]:null;}
_48.removeDep(this.vj$._class,cl);if(_79!==""){if(tp&&!(_2.isInstanceOf(tp,_MType))){var nm=(useAlias)?_79:cn,err=false;if(this.vj$[nm]&&this.vj$[nm]!==tp){if(_bOldSupport){err=true;}else{throw"Name collision with type '"+nm+"' in need list.";}}
if(!err){this.vj$[nm]=tp;}}else{_48.deferNeed(this.vj$._class,(function(_81,_82,ctx){return function(){var tp=_2.getType(_82);if(tp&&tp.vj$&&!(_2.isInstanceOf(tp,_MType))){if(ctx.vj$[_81]&&ctx.vj$[_81]!==tp){throw"Name collision with "+nm+"in need list.";}
ctx.vj$[_81]=_2.getType(_82);}};})((useAlias)?_79:cn,cl,this));}}},this);return this;},props:function(obj,_86){_forEach(obj,function(val,key,obj){if(_isValidProp(key)){var o=this[key]=val;if(_addInner(this,o,"s_inners",key)){if(this.vj$[key]){throw"'"+key+"' in type '"+this.vj$._class+"' conflicts with needed type name";}
var _v={};_2.extend(_v,this.vj$);delete _v._meta;_2.extend(_v,o.vj$);_v[key]=o;o.vj$=_v;if(!this.vj$._meta._isInner){_48.addInner(this.vj$._class,TypeMeta.get(o.vj$._meta._metaId));var rt=this.vj$._class;_updateInners(rt,rt+"."+key,o,true);}}else{if(typeof o=="function"&&!o._name&&!o.vj$){o._name=key;}}}},this);return this;},protos:function(obj,_8e){if(!obj){return;}
_forEach(obj,function(val,key,obj){if(key!="base"&&key!="b"){var _92=this.prototype[key],isType=_isVjoType(val);if(_92&&typeof _92=="function"&&key.indexOf("constructs")!=0&&(key.indexOf("_ovld")===-1||key.indexOf("_ovld")!=(key.length-5))&&!_isVjoType(_92)&&typeof val=="function"&&!isType){this.prototype[key]=(function(_93,_94){return function(){var _95=this.base,t={},error=false,rv,out=this.vj$.outer,fix=true;if(fix){this.base=(_94.prototype._getBase)?_94.prototype._getBase():null;if(this.base){this.base._parent=this;}
t.vj$=this.vj$;if(!out){_fixScope(_94,this);}}
try{rv=_93.apply(this,arguments);}
catch(e){error=e;}
if(fix){if(!out){_fixScope(t,this);}
this.base=_95;}
if(error){throw error;}
return rv;};})(val,this);}else{if(isType&&!this.vj$._meta._isInner){_48.addInner(this.vj$._class,TypeMeta.get(val.vj$._meta._metaId));var rt=this.vj$._class;_updateInners(rt,rt+"."+key,val,false);}
this.prototype[key]=val;}
if(!_addInner(this,val,"_inners",key)){if(typeof val=="function"&&!val._name&&!isType){val._name=key;}}else{if(this.vj$[key]){throw"'"+key+"' in type '"+this.vj$._class+"' conflicts with needed type name";}}}
if(obj.toString!=Object.prototype.toString){this.prototype.toString=obj.toString;}},this);return this;},isInstance:function(obj){return _2.isInstanceOf(obj,this);},endType:function(){_updateInnerEtypes(this.vj$);_48.load(this.vj$._class);if(_2.validateType){_2.validateType(this);}
return this;}};var _98={satisfies:function(_99,_9a){var _9b=[];if(_2.isArray(_99)){_9b=_99;}else{_9b=[_99];}
_forEach(_9b,function(val,key,obj){var cl=_getTypeName(val),_99;if(_isVjoType(cl)){_99=cl;var clz=_99.vj$._class||"",idx=clz.lastIndexOf("."),cn=(idx>-1)?clz.substring(idx+1):clz;if(cn){this.vj$[cn]=_99;}}
var _a1=(_99)?_99:_2.getType(cl);if(_a1){if(!_9a){this.clazz._satisfied.push(_a1);}
for(var i in _a1){var val=_a1[i];if(_isValidProp(i)&&!this[i]){this[i]=val;}}}},this);return this;},inherits:function(_a3,isB){_a3=_getTypeName(_a3);if(!isB&&!_isValidInh(_a3)){throw"Cannot inherit from '"+_a3+"'";}
var _a5=("vjo.Object"===_a3);if(!isB&&_a5){return this;}
var _a6;if(_isVjoType(_a3)){_a6=_a3;var clz=_a6.vj$._class||"",idx=clz.lastIndexOf("."),cn=(idx>-1)?clz.substring(idx+1):clz;if(cn){this.vj$[cn]=_a6;}}else{if(_a5){_a6=_2.Object;}else{_a6=(this.vj$[_a3])?this.vj$[_a3]:_2.getType(_a3);}}
if(_isVjoType(_a6)){var _a8=TypeMeta.get(_a6.vj$._meta._metaId);if(_a8&&!_a8._completed){_a8.complete();}
_createInherits(this,_a6);}
return this;}};_2.extend({sysout:{print:function(){if(typeof console!="undefined"){console.info.apply(this,arguments);}},println:function(){if(typeof console!="undefined"){console.info.apply(this,arguments);}},printStackTrace:function(){}},syserr:{print:function(){if(typeof console!="undefined"){console.warn.apply(this,arguments);}},println:function(){if(typeof console!="undefined"){console.warn.apply(this,arguments);}},printStackTrace:function(){}},jsunit:{assertEquals:function(){},assertTrue:function(){},assertFalse:function(){},assertNotNull:function(){}}});function TypeMeta(_a9,_aa,cfg){this._needs=[];this._props=null;this._protos=null;this._satisfies=[];this._mixins=[];this._inherits=[];this._inits=null;this._completed=(_aa==="type")?true:false;this._isInner=(_a9)?false:true;this._name=_a9;this._kind=_aa;this._options={autoConstruct:true};this.init(cfg);this.setup();}
TypeMeta.prototype={init:function(cfg){this._cfg={satisfiesFn:_98.satisfies,inheritsFn:_98.inherits,protosFn:_77.protos,postDefFn:function(){},typeDef:null,baseType:"vjo.Object"};if(cfg){_2.extend(this._cfg,cfg);}},setup:function(){var t=this._type=this._cfg.typeDef||_createType(this._name,"itype"===this._kind);t.vj$._type=this._kind||"ctype";var id=this._name;if(this._isInner){id=TypeMeta.id();t.vj$._meta._isInner=true;}else{var _af=_createPkg(id);if(!_af.pkg[_af.className]){_af.pkg[_af.className]=this._type;}
t.vj$[_af.className]=t;_48.register(id,this);}
this._isDup=(!this._isInner&&TypeMeta.get(id)!=null);if(!this._isDup){TypeMeta.put(this,id);}
t._inherits=null;_createClazz(t);t.vj$._meta._metaId=id;},needs:function(_b0,_b1){_77.needs.apply(this._type,arguments);return this;},singleton:function(){return this;},options:function(_b2){if(_b2){for(var k in _b2){this._options[k]=_b2[k];}}
return this;},makeFinal:function(){return this;},satisfies:function(_b4){var _b5=[];if(_2.isArray(_b4)){_b5=_b4;}else{_b5=[_b4];}
_forEach(_b5,function(val,key,obj){var _b9=_getTypeName(val);this.needs(_b9);_77.needs.call(this._type,_b9);this._satisfies.push(val);},this);return this;},props:function(_ba){if(this._props){throw"multiple props blocks are not allowed";}
this._props=_ba;_77.props.apply(this._type,arguments);return this;},protos:function(_bb){if(this._protos){throw"multiple protos blocks are not allowed";}
this._protos=_bb;return this;},inherits:function(_bc){var _bd=[];if(_2.isArray(_bc)){_bd=_bc;}else{_bd=[_bc];}
_forEach(_bd,function(val,key,obj){var _c1=_getTypeName(val);this.needs(_c1);this._inherits.push(val);},this);return this;},mixin:function(_c2){var _c3=[];if(_2.isArray(_c2)){_c3=_c2;}else{_c3=[_c2];}
_forEach(_c3,function(val,key,obj){var _c7=_getTypeName(val);this.needs(_c7);this._mixins.push(val);},this);return this;},inits:function(fn){this._inits=fn;return this;},validateAndMerge:function(_c9,_ca){var p=(_ca)?_c9[_ca]:_c9;return function(val,key,obj){if(p[key]){throw"collision when mixing in '"+key+"' into "+this._name;}else{p[key]=val;}};},complete:function(){if(this._completed){return this;}
this._completed=true;_updateInnerEtypes(this._type.vj$);var _cf=this._mixins;var p={_props:{}};for(var i=0;i<_cf.length;i++){var m=_2.getType(_cf[i]);if(!m||!m.vj$||m.vj$._type!="mtype"){throw _cf[i]+"is not a valid mtype.";}
var exp=m._expects;sats=m._satisfiers;if(!this._protos){this._protos={};}
if(!this._props){this._props={};}
_copyNS(m.vj$,this._type.vj$);var nm=m.vj$._class,idx=nm.lastIndexOf(".");var clz=(idx!=-1)?nm.substring(idx+1):nm;if(!this._type.vj$[clz]){this._type.vj$[clz]=this._type;}else{throw clz+" is already defined in the current namespace";}
_forEach(m._protos,this.validateAndMerge(this,"_protos"),this);_forEach(m._props,this.validateAndMerge(p,"_props"),this);for(var j=0;j<sats.length;j++){this._satisfies.push(sats[i]);}}
_77.props.call(this._type,p._props);if(this._satisfies.length>0){this._cfg.satisfiesFn.call(this._type,this._satisfies);}
if(exp){this._cfg.satisfiesFn.call(this._type,[exp],true);}
var _d7=this._inherits.length;if(_d7>0){if(this._kind!="itype"&&_d7>1){throw"type can only inherit from one type";}
for(var i=0;i<_d7;i++){this._cfg.inheritsFn.call(this._type,this._inherits[i]);}}else{if(this._kind!="itype"){this._cfg.inheritsFn.call(this._type,this._cfg.baseType,true);}}
if(_d7==0){var _d8=this._protos||{};_forEach(["hashCode","equals","getClass"],function(val){if(!_d8[val]){this[val]=_2.Object.prototype[val];}},this._type.prototype);}
if(this._protos){this._cfg.protosFn.call(this._type,this._protos);}
if(!this._type.prototype.constructs){this._type.prototype.constructs=function(){};}
this._cfg.postDefFn.call(this);if(this._inits&&!this._isDup){this._inits.call(this._type);}
return this;},canComplete:function(){var b=(this._inits==null);for(var i=0;i<this._inherits.length;i++){var s=this._inherits[i];if(!_isVjoType(s)){b=false;break;}}
for(var i=0;b&&i<this._satisfies.length;i++){var s=this._satisfies[i];if(!_isVjoType(s)){b=false;break;}}
for(var i=0;b&&i<this._mixins.length;i++){var m=this._mixins[i];if(!_isVjoType(m)){b=false;break;}}
return b;},endType:function(){if(!this._isInner){_48.load(this._name);if(_2.validateType){_2.validateType(this._type);}}else{if(this.canComplete()){this.complete();}}
return this._type;}};_2.extend(TypeMeta,{_count:0,_pre:"tmp",_reg:{},id:function(){return this._pre+this._count++;},put:function(_de,id){var nm=(id)?id:this._pre+this._count++;this._reg[nm]=_de;},get:function(id){return this._reg[id];}});_2.ctype=function(clz){clz=_getTypeName(clz);var t=new TypeMeta(clz);return t;};_2.type=function(clz){clz=_getTypeName(clz);var t=new TypeMeta(clz,"type");t.inits=function(fn){if(fn&&!this._isDup){fn.call(this._type);}
return this;};t.props=function(_e7){_77.props.apply(this._type,arguments);return this;};t.protos=function(_e8){_77.protos.apply(this._type,arguments);return this;};t.inherits=function(clz){_98.inherits.apply(this._type,arguments);return this;};t.satisfies=function(clz){_98.satisfies.apply(this._type,arguments);return this;};return t;};_2.itype=function(clz){clz=_getTypeName(clz);var t=new TypeMeta(clz,"itype",{inheritsFn:function(_ed){var _ee=(this.vj$[_ed])?this.vj$[_ed]:(this.vj$.b&&this.vj$.b[_ed])?this.vj$.b[_ed]:_2.getType(_ed);if(_ee){for(var i in _ee){var val=_ee[i];if(_isValidProp(i)&&!this[i]){this[i]=val;}}}
return this;}});t._type.isInstance=function(obj){return _2.isInstanceOf(obj,this);};return t;};_2.atype=_2.ctype;function _addMixinMethods(to,_f3,ns){if(!_f3||typeof _f3!="object"){return;}
var b=true;for(var i in _f3){b=false;if(!reservedMProp[i]){to[i]=_f3[i];}}
return b;}
function _MType(clz){var t=this;t.vj$={_type:"mtype",_class:clz,_meta:{}};t._props=null;t._protos={};t._expects="";t._satisfiers=[];t.needs=function(){return _77.needs.apply(this,arguments);};t.props=function(_f9){var p;if(!this._props){p={};}else{p=this._props;}
if(!_addMixinMethods(p,_f9,this.vj$)){if(!this._props){this._props=p;}}
return this;};t.protos=function(_fb){if(_fb&&_fb["constructs"]){throw"mtype cannot have constructs block";}
_addMixinMethods(this._protos,_fb,this.vj$);return this;};t.expects=function(clz){this._expects=_2.getType(clz);return this;};t.satisfies=function(clz){var _fe=[];if(_2.isArray(clz)){_fe=clz;}else{_fe=[clz];}
_forEach(_fe,function(val,key,obj){var _102=_getTypeName(val);this.needs(_102);this._satisfiers.push(val);},this);return this;};t.endType=function(){if(this.vj$._class){_48.load(this.vj$._class);}
return this;};return t;}
_2.mtype=function(clz){clz=_getTypeName(clz);var base=new _MType(clz);if(!clz||_isInnerClass(clz)){base.vj$._meta._isInner=true;}
if(!clz){return base;}
var pObj=_createPkg(clz);return(pObj.pkg[pObj.className])?base:(pObj.pkg[pObj.className]=base);};_2.etype=function(clz){clz=_getTypeName(clz);_createEnum();var eDef=function(args){this.vj$=eDef.vj$;if(args!=false){if(!this.constructs){throw"'"+this.vj$._class+"' cannot be instantiated";}
var rv=this.constructs.apply(this,args);}};eDef.vj$={_class:clz,_type:"etype",_meta:{}};eDef.isInstance=function(o){return _2.isInstanceOf(o,this);};var t=new TypeMeta(clz,"etype",{typeDef:eDef,baseType:"vjo.Enum",postDefFn:function(){this._type.prototype.toString=_2.Enum.prototype.name;if(this._eVals){this._type.values.call(this._type,this._eVals);}}});_2.extend(t,{inherits:function(){throw"Invalid type definition. etype cannot be inheritted from another type";},values:function(vals){this._eVals=vals;return this;}});var en=t._type;en._enums=[];en.from=_2.Enum.from;en.values=function(vals){if(arguments.length==0){if(typeof this._enums.slice!="undefined"){return this._enums.slice();}else{var a=[];for(var i=0;i<this._enums.length;i++){if(typeof this._enums[i]!="undefined"){a[i]=this._enums[i];}}
return a;}}else{var ord=0;if(typeof vals=="string"&&vals.length>0){while(vals.indexOf(" ")>-1){vals=vals.replace(" ","");}
if(vals.indexOf(",")>0){var a=vals.split(","),t;if(a[0]&&a[0].indexOf(":")>0){throw"Invalid labels for etype values: "+a[0];}
for(var i=0,l=a.length;i<l;i++){var eV=a[i];if(i==0&&t&&t.length>0){eV=t[0];}else{if(a[i].indexOf(":")>-1){eV=a[i].split(":")[0];}}
this._enums[this._enums.length]=new en(false);this._enums[this._enums.length-1]._name=eV;}}else{this._enums[0]=new en(false);this._enums[0]._name=vals;}}else{for(var itm in vals){this._enums[this._enums.length]=new en(vals[itm]);this._enums[this._enums.length-1]._name=itm;}}
for(var i=0,l=this._enums.length;i<l;i++){if(typeof this._enums[i]!="undefined"){var nm=this._enums[i]._name;if(this[nm]){throw"Invalid prop member. Cannot use etype value as prop member.";}
this[nm]=this._enums[i];this[nm]._ord=ord++;}}}
en.prototype.constructs=null;return this;};return t;};_2.otype=function(clz){return{defs:function(defs){return this;},endType:function(){}};};var _117={},reservedProto={},reservedMProp={},reservedClz={},reservedInh={};_forEach("props protos inherits prototype inits satisfies mixin _inherits _satisfiers singleton isInstance vj$".split(" "),function(val,key,obj){this[val]=true;},_117);_forEach("constructs getClass _getBase base vj$".split(" "),function(val,key,obj){this[val]=true;},reservedProto);_forEach("props protos _props _protos vj$ _expects expects _satisfiers satisfies endType".split(" "),function(val,key,obj){this[val]=true;},reservedMProp);_forEach("vjo.Class vjo.Object".split(" "),function(val,key,obj){this[val]=true;},reservedClz);function _constructType(ctx,args){var bC=true;if(!ctx.constructs){ctx.constructs=function(){};bC=false;}
var fn,rv,c=ctx.constructs,dconstruct=false;if(ctx.base&&ctx._getBase){fn=function(){};fn.prototype=ctx._getBase();dconstruct=fn.prototype._constructs||false;}
if(dconstruct&&ctx.base){var cstr=c.toString();if(cstr.indexOf("this.base(")===-1&&cstr.indexOf("this.constructs")===-1){ctx.base();}}
if(bC){rv=c.apply(ctx,args);}
if(fn){ctx.base=new fn;ctx.base._parent=ctx;}
if(rv){return rv;}}
function _createType(clz,isI){isI=isI||false;if(!_isValidClz(clz)){throw"Invalid type name '"+clz+"'";}
var base=function(){var _12c=TypeMeta.get(base.vj$._meta._metaId),bConstruct=!base.__donotconstruct;if(!(this instanceof base)&&bConstruct&&_12c&&_12c._options.autoConstruct){base.__donotconstruct=true;var b=new base();delete base.__donotconstruct;var rv=_constructType(b,arguments);if(rv){return rv;}
return b;}
this.constructor=base;if(_12c&&!_12c._completed){_12c.complete();_forEach(base.prototype,function(val,key,_131){this[key]=val;},this);}
this.vj$=base.vj$;var t=this.vj$._type;if(bConstruct&&(t=="itype"||t=="atype"||t=="mtype")){throw t+" "+this.vj$._class+" cannot be instantiated";}
_processInners(this,base.vj$,base.clazz);if(bConstruct){var val=_constructType(this,arguments);if(val){return val;}}
return null;};base._name="base";base.vj$={_class:clz,_meta:{}};base.isInstance=function(o){return _2.isInstanceOf(o,this);};return base;}
function _createClazz(typ){var old=typ.clazz,nm=typ.vj$._class;if(typ.clazz&&typ.clazz._name){nm=typ.clazz._name;}
typ.clazz=new _2.Class(nm,typ.vj$._type);if(old){typ.clazz._satisfied=old._satisfied;}
if(typ.prototype){typ.prototype.getClass=_getClazz;}else{typ.getClass=_getClazz;}}
function _createInherits(_137,type){var _139=type.prototype;type.__donotconstruct=true;var cls=new type();delete type.__donotconstruct;cls.constructs=null;cls.constructor=_137;var _13b={};if(_139._getBase&&_139._getBase()._constructs){_13b._constructs=true;}
cls.base=function(){var _13c=this.base,_139=type.prototype,gb=_139._getBase,c=_139.constructs;if(_139.base){this.base=_139.base;}
var cstr=(c)?c.toString():"",b=(cstr.indexOf("this.base(")===-1&&cstr.indexOf("this.constructs")===-1);if(gb&&gb()._constructs&&b){this.base();}
if(_139.constructs){var isIn=this.vj$._meta._isInner,error=false;if(!isIn){_fixScope(type,this);}
try{_139.constructs.apply(this,arguments);}
catch(e){error=e;}
if(!isIn){_fixScope(_137,this);}
if(error){throw error;}}
this.base=_13c;};cls._getBase=function(){return _13b;};function createBaseRef(type,func,der){return function(){var scp=(this._parent)?this._parent:this,rv,error=false,cbase=scp.base;scp.base=(type.prototype._getBase)?type.prototype._getBase():null;if(scp.base){scp.base._parent=scp;}
_fixScope(type,scp);try{rv=func.apply(scp,arguments);}
catch(e){error=e;}
_fixScope(der,scp);scp.base=cbase;if(error){throw error;}
return rv;};}
var vO=_2.Object,vOP=vO.prototype,bIsO=(vOP===_139);for(var i in _139){var pt=_139[i];if(i==="toString"){continue;}
if(i==="constructs"&&typeof pt==="function"){if(type!==vO&&pt.length===0){_13b._constructs=true;}}else{if(_139==vOP||_isValidInst(i)){if(typeof pt==="function"&&!(_2.isInstanceOf(pt,RegExp))&&!pt.vj$&&(bIsO||vOP[i]!==pt)){_13b[i]=createBaseRef(type,pt,_137);if(!pt.__isChained){var _146=pt.toString();if(_146.indexOf("this.base."+i+"(")!=-1){cls[i]=(function(fn){var fn=function(){return fn.apply(this,arguments);};fn.__isChained=true;return fn;})(_13b[i]);}else{cls[i]=(function(type,func,der){var fn=function(){var _14c=false,rv;_fixScope(type,this);try{rv=func.apply(this,arguments);}
catch(e){_14c=e;}
_fixScope(der,this);if(_14c){throw _14c;}
return rv;};fn.__isChained=true;return fn;})(type,pt,_137);}}}else{if(pt&&pt.vj$&&pt.vj$._type&&pt.vj$._meta._isInner){if(!_137.vj$._meta._inners){_137.vj$._meta._inners={};}
_137.vj$._meta._inners[i]=pt;}
cls[i]=pt;}}}}
if(_13b.toString!=_2.Object.prototype.toString){_13b.toString=createBaseRef(type,_139.toString,_137);}
_137.prototype=cls;}
function _isVjoType(clz,_14e){if(clz&&clz.vj$&&clz.vj$._type){if(typeof clz==="function"){return true;}else{if(clz.vj$._type==="mtype"&&!_14e){return true;}}}
return false;}
function _isInstanceForInterface(_14f,_150){var clz=_14f,arr=clz._satisfied;for(var i=0;i<arr.length;i++){if(_isInterfaceInstanceOf(_150,arr[i])){return true;}}
return false;}
function _isInterfaceInstanceOf(src,_154){if(src===_154){return true;}
var meta=TypeMeta.get(_154.vj$._meta._metaId),inhs;if(meta&&(inhs=meta._inherits)){for(var i=0;i<inhs.length;i++){var _157=_getTypeName(inhs[i]);if(src===_2.getType(_157)){return true;}}}
return false;}
function _createPkg(clz,_159){if(!clz){return null;}
if(_2._typeMap[clz]){return _2._typeMap[clz];}
var _15a=clz.split("."),len=_15a.length;if(_159){_2._typeMap[clz]={pkg:{className:_15a[len-1]}};}else{var pkg=(_15a[0]=="vjo")?_2._global:this;for(var i=0;i<len-1&&pkg&&_15a[i];i++){pkg=(pkg[_15a[i]])?pkg[_15a[i]]:pkg[_15a[i]]={};}
_2._typeMap[clz]={pkg:pkg,className:(len>0)?_15a[len-1]:""};}
return _2._typeMap[clz];}
function _createEnum(){if(typeof _2.Enum=="function"){return;}
var nm="vjo.Enum";var _15e=_2.ctype(nm).props({from:function(){if(!arguments[0]){throw"Invalid argument value: "+arguments[0];}
var s=arguments[0];if(arguments.length==2){s=arguments[1];if(s){var clz=arguments[0];try{var n=clz.getName();while(n.indexOf("$")>0){n=n.replace("$",".");}
var o=eval(n);if(o[s]){return o[s];}}
catch(a){}}
throw"No enum const "+arguments[0].getName()+"."+s;}else{if(this[s]){return this[s];}}
throw"No enum const "+s;}}).protos({_name:null,_ord:-1,constructs:function(){throw"cannot instantiate 'vjo.Enum'";},name:function(){return this._name;},ordinal:function(){return this._ord;},compareTo:function(o){if(o==null){throw"compare to Etype value cannot be null";}
return(this.ordinal()-o.ordinal());},equals:function(o){return(this===o);}}).endType();reservedClz[nm]=true;reservedInh[nm]=true;}
function _getClazz(){var n=clz=this.vj$._class;var idx=n.lastIndexOf(".");if(idx!=-1){clz=n.substring(idx+1);}
if(this.vj$[clz]){return this.vj$[clz].clazz;}
return null;}
function _updateInners(_167,_168,_169,_16a){if(_169&&_169.vj$){_169.vj$._class=_168;var idx=_168.lastIndexOf("."),snm=_168.substring(idx+1);_169.vj$[snm]=_169;if(_169.clazz&&_167){if(_168.indexOf(_167)==0){var tmp=_168.replace(_167,"");while(tmp.indexOf(".")>-1){tmp=tmp.replace(".","$");}
_169.clazz._name=_167+tmp;}else{_169.clazz._name=_167+"$"+snm;}}
_createPkg(_168,true).pkg[snm]=_169;var ins=(_16a)?_169.vj$._meta.s_inners:_169.vj$._meta._inners;if(ins){_forEach(ins,function(val,key){_forEach(_169.vj$,function(val,key){if(!this[key]&&val&&val.vj$){this[key]=val;}},val.vj$);var m=TypeMeta.get(val.vj$._meta._metaId);if(val.vj$&&m){_48.addInner(_167,m);}
_updateInners(_167,_168+"."+key,val,_16a);});}}}
function _updateInnerEtypes(_173){if(!_173._class){return;}
var _174=_173._meta;if(_174.s_inners){for(var k in _174.s_inners){if(_174.s_inners[k].vj$._type=="etype"){for(var i=0;i<_174.s_inners[k]._enums.length;i++){_174.s_inners[k]._enums[i].vj$=_174.s_inners[k].vj$;_updateInnerEtypes(_174.s_inners[k]._enums[i].vj$);}}
_updateInnerEtypes(_174.s_inners[k].vj$);}}
if(_174._inners){for(var k in _174._inners){if(_174._inners[k].vj$._type=="etype"){if(!_174._inners[k].vj$._class){_174._inners[k].clazz._name=_174._inners[k].vj$._class=_174._class+"."+k;}
for(var i=0;i<_174._inners[k]._enums.length;i++){_174._inners[k]._enums[i].vj$=_174._inners[k].vj$;_updateInnerEtypes(_174._inners[k]._enums[i].vj$);}}
_updateInnerEtypes(_174._inners[k].vj$);}}}
function _addInner(clz,_178,_179,key){if(!clz||!_178||!key){return false;}
if(_178.vj$&&_178.vj$._type&&_178.vj$._meta._isInner){if(!_178.vj$._class&&clz.vj$._class){var cn=_178.vj$._class=clz.vj$._class+"."+key;if(_178.clazz){_178.clazz._name=cn;}
_createPkg(cn,true).pkg[key]=_178;}
if(_179){if(!clz.vj$._meta[_179]){clz.vj$._meta[_179]={};}
clz.vj$._meta[_179][key]=_178;}
return true;}
return false;}
function _hasCollisionWithMixin(type,name,_17e){var mxns=type.vj$._meta.mixins;if(!mxns||mxns.length==0){return false;}
for(var i=0;i<mxns.length;i++){var mxn=mxns[i];if(_17e){if(mxn._props&&mxn._props[name]){return true;}}else{if(mxn._protos[name]){return true;}}}
return false;}
function _isValidInst(pVal){return!(reservedProto[pVal]===true);}
function _isValidClz(pVal){return!(reservedClz[pVal]===true);}
function _isValidInh(pVal){return!(reservedInh[pVal]===true);}
function _extend(){var a=arguments,len=a.length,target,source;if(len==1){target=this;source=a[0];}else{target=a[0];source=a[1];}
for(var name in source){var copy=source[name];if(copy!==undefined){target[name]=copy;}}
if(source.toString!=Object.prototype.toString){target.toString=source.toString;}}
function _forEach(_188,_189,_18a){if(!_188){return;}
var name,i=0,len=_188.length;if(!_2.isArray(_188)){for(name in _188){if(_189.call(_18a,_188[name],name,_188)===false){break;}}}else{for(;i<len;i++){if(_189.call(_18a,_188[i],i,_188)===false){break;}}}
return _188;}
function _processInners(_18c,_18d,_18e){var _18f=(_18d)?_18d._meta._inners:null;if(!_18d||!_18f||_18f.length==0||!_18c){return;}
for(var k in _18f){_18c[k]=(function(t,type,nm){var fn=function(){var cn=t.vj$._class+"."+nm;var m=TypeMeta.get(type.vj$._meta._metaId);if(m){m.complete();}
type.__donotconstruct=true;var tp=new type();delete type.__donotconstruct;var _v={};_2.extend(_v,_18d);delete _v._meta;_2.extend(_v,type.vj$);_v[nm]=type;tp.vj$=_v;tp.vj$.outer=t;_processInners(tp,_v,type.clazz);type.vj$._class=tp.vj$._class=cn;var s=type.prototype;if(type.clazz&&!type.clazz._name){type.clazz._name=tp.vj$._class;}else{_createClazz(type);}
if(s.constructs){s.constructs.apply(tp,arguments);}
return tp;};fn._outer=t;return fn;})(_18c,_18f[k],k);_18c[k].vj$=_18f[k].vj$;}}
function _copyNS(from,to){_forEach(from,function(val,key){if(_isVjoType(val,true)){if(this[key]&&this[key]!==val){throw key+" is already defined in the current namespace";}
this[key]=val;}},to);}
function _isInnerClass(clz){if(!clz){return true;}else{if(clz.indexOf(".")==-1){return false;}}
var tp=clz;while((i=tp.lastIndexOf("."))>0){tp=tp.substring(0,i);if(_2._typeMap[tp]){return true;}}
return false;}
function _isValidProp(pVal){return!(_117[pVal]===true);}
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
var _5,len=_4.length;for(var i=0;i<len;i++){var _7=_4[i].handle(_2);if(_7&&_7.objType=="dsf_Message"){if(vjo.dsf.ServiceEngine){vjo.dsf.ServiceEngine.handleRequest(_7);}
if(_7.returnData===false){_5=false;}}else{if(_5!=false&&typeof _7!="undefined"){_5=_7;}}}
return _5;},register:function(id,_9,_a,_b){if(!id||!_9||!_a){return this;}else{if(typeof _a.handle!="function"){if(typeof _a=="function"){var _c=_a,scp=_b||this;var _d={handle:function(){return _c.apply(scp,arguments);}};_a=_d;}else{return this;}}}
var _e=this.eventHandlers[id];if(!_e){_e=this.eventHandlers[id]={};}
if(!_e[_9]){_e[_9]=[];}
var _f=_e[_9].length;_e[_9][_f]=_a;return _a;},unregister:function(id,_11){if(!this.eventHandlers[id]){return;}
this.eventHandlers[id][_11]=[];},registerNative:function(_12,_13,_14){var id=(_12==window)?"body":_12.id;var _16=this.nativeEventHandlers[id];if(!_16){_16=this.nativeEventHandlers[id]={};}
if(!_16[_13]){_16[_13]=[];}
var len=_16[_13].length;_16[_13][len]=_14;},add:function(id,_19,_1a,_1b){if(!id||!_19||!_1a){return this;}
var b=this.isBound(id,_19),rv=this.register(id,_19,_1a,_1b);if(!b){b=this.bind(id,_19);if(b===null){this.unboundElems[this.unboundElems.length]=id;}}
return rv;},addEventListener:function(_1d,_1e,_1f,_20,_21){var scp=_20||vjo.global;if(typeof _1d=="string"){_1d=this.vj$.Element.get(_1d);}
if(!_1d){return false;}
var _23=function(_24){var ev=_24||window.event;var rv=_1f.call(scp,ev);if(rv===false){vjo.dsf.EventDispatcher.stopEvent(ev);}
if(typeof rv!="undefined"){return rv;}};if(window.addEventListener){_1d.addEventListener(_1e,_23,_21||false);this.registerNative(_1d,_1e,_23);return _23;}else{if(window.attachEvent){_1d.attachEvent("on"+_1e,_23);this.registerNative(_1d,_1e,_23);return _23;}}
_1d["on"+_1e]=_23;return false;},bind:function(id,_28){var _29=this.vj$.Element.get(id);if(id=="body"||_29==document.body){_29=document.body;if(_28=="load"||_28=="unload"){var rv=this.addEventListener(window,_28,function(_2b){var oED=vjo.dsf.EventDispatcher;if(typeof oED.fCustomLoad[_28]=="function"){oED.fCustomLoad[_28]();}
oED.run(document.body,_2b||window.event,_28);oED.unregister("body",_28);oED.fCustomLoad={};});if(rv===false){if(_29.vjLoadSet){return this;}else{_29.vjLoadSet=true;var _2d=window["on"+_28]||"";if(_2d){this.fCustomLoad[_28]=_2d;}}}
return this;}}
if(_29){this.addEventListener(_29,_28,this.notifier,_29);return this;}
return null;},notifier:function(_2e,_2f){return vjo.dsf.EventDispatcher.run(this,_2e||window.event,_2f);},reBind:function(){var eH=this.eventHandlers,uE=this.unboundElems,len=uE.length,tmp=[];for(var i=0;i<len;i++){var id=uE[i],hdls=eH[id];if(hdls){for(var _33 in hdls){if(!this.hasBinding(id,_33)){var rv=this.bind(id,_33);if(rv===null){tmp[tmp.length]=id;}}}}}
this.unboundElems=tmp;},isBound:function(id,_36){var _37=this.eventHandlers[id];return(_37&&_37[_36]&&_37[_36].length>0);},hasBinding:function(id,_39){var nEH=this.nativeEventHandlers;if(nEH[id]&&nEH[id][_39]){var aH=nEH[id][_39],len=aH.length,rv=false;for(var i=0;i<len;i++){var str=aH[i].toString();if(str&&str.indexOf("vjo.dsf.EventDispatcher")!=-1){return true;}}}
return false;},removeEventListener:function(_3e,_3f,_40){if(!_3e||!_3f){return;}else{if(typeof _3e=="string"){_3e=this.vj$.Element.get(_3e);}}
if(window.addEventListener&&_40){_3e.removeEventListener(_3f,_40,false);}else{if(window.attachEvent&&_40){_3e.detachEvent("on"+_3f,_40);}else{_3e["on"+_3f]=null;}}},detachNativeHandlers:function(_41,_42){var id=(_41==window)?"body":_41.id;var _44=this.nativeEventHandlers[id];if(_44&&_44[_42]){var h=_44[_42],len=h.length;for(var i=0;i<len;i++){this.removeEventListener(_41,_42,_44[_42][i]);}
_44[_42]=[];}},detachHandler:function(id,_48,_49){var _4a=this.eventHandlers[id];if(!_4a||!_4a[_48]){return;}
var h=[],len=_4a[_48].length;for(var i=0;i<len;i++){if(_49!=_4a[_48][i]){h[h.length]=_4a[_48][i];}}
this.eventHandlers[id][_48]=h;},detachHandlers:function(id,_4e){this.unregister(id,_4e);var _4f=this.vj$.Element.get(id);if(id=="body"){_4f=window;}
if(_4f){this.detachNativeHandlers(_4f,_4e);}},stopEvent:function(e){this.stopPropagation(e);this.preventDefault(e);},stopPropagation:function(e){if(e.stopPropagation){e.stopPropagation();}else{e.cancelBubble=true;}},preventDefault:function(e){if(e.preventDefault){e.preventDefault();}else{e.returnValue=false;}},target:function(_53){return this.resolveTextNode((_53.target)?_53.target:_53.srcElement);},currentTarget:function(_54){return this.resolveTextNode((_54.currentTarget)?_54.currentTarget:_54.srcElement);},relatedTarget:function(_55){if(_55.relatedTarget){return this.resolveTextNode(_55.relatedTarget);}else{if((_55.type==="mouseover")&&_55.fromElement){return this.resolveTextNode(_55.fromElement);}else{if((_55.type==="mouseout")&&_55.toElement){return this.resolveTextNode(_55.toElement);}else{return null;}}}},resolveTextNode:function(_56){return(_56&&(_56.nodeType==3))?_56.parentNode:_56;},cleanUp:function(){var _57=this.nativeEventHandlers;for(var id in _57){for(var ev in _57[id]){if(ev!="unload"){this.detachHandlers(id,ev,true);}}}},getId:function(src,id){var _5c=id;if(_5c===null||!_5c){_5c=src.id;}
return _5c;},getBodyId:function(src,id){var _5f=this.getId(src,id);if(!_5f||src==document.body){_5f="body";}
return _5f;},unload:function(src,_61){return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"unload",_61));},change:function(src,_63){return this.process(this.getId(src),new vjo.dsf.Event(src,"change",_63));},submit:function(src,_65){return this.process(this.getId(src),new vjo.dsf.Event(src,"submit",_65));},reset:function(src,_67){return this.process(this.getId(src),new vjo.dsf.Event(src,"reset",_67));},select:function(src,_69){return this.process(this.getId(src),new vjo.dsf.Event(src,"select",_69));},blur:function(src,_6b){return this.process(this.getId(src),new vjo.dsf.Event(src,"blur",_6b));},focus:function(src,_6d){return this.process(this.getId(src),new vjo.dsf.Event(src,"focus",_6d));},keydown:function(src,_6f){return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"keydown",_6f));},keypress:function(src,_71){return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"keypress",_71));},keyup:function(src,_73){return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"keyup",_73));},click:function(src,_75){return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"click",_75));},dblclick:function(src,_77){return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"dblclick",_77));},mousedown:function(src,_79){return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"mousedown",_79));},mousemove:function(src,_7b){return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"mousemove",_7b));},mouseout:function(src,_7d){return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"mouseout",_7d));},mouseover:function(src,_7f){return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"mouseover",_7f));},mouseup:function(src,_81){return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"mouseup",_81));},load:function(src,_83){return this.run(src,_83,"load");},run:function(src,_85,_86){var et=_86||_85.type;var id=this.getBodyId(src);var _89=new vjo.dsf.Event(src,et,_85);var rv=this.process(id,_89);if(et==="load"&&id==="body"){this.unregister("body","load");}
return rv;}}).inits(function(){vjo.dsf.EventDispatcher.addEventListener(window,"load",function(){vjo.dsf.EventDispatcher.addEventListener(window,"unload",function(){vjo.dsf.EventDispatcher.cleanUp();});});}).endType();

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

vjo.ctype("vjo.dsf.XDomainRequest").needs("vjo.dsf.EventDispatcher").props({callbacks:[],sCallbackName:"callback",sPreId:"xdr_",sPreExtId:"xdr_ext_",iCount:0,bUseIframe:(navigator.userAgent.indexOf("Firefox")>0),onLoad:function(){this.bodyLoaded=true;},getReqDiv:function(){return document.getElementsByTagName(this.bodyLoaded?"body":"head")[0];},send:function(_1){if(!document.createElement||!_1){return;}
var _2="",eid="",rdm="_vrdm="+(new Date()).getTime();if(typeof _1=="string"){_2=_1;eid=this.sPreExtId+this.iCount++;}else{if(_1.objType=="dsf_Message"&&_1.svcConfig){var cb=this.createCallback(_1);eid=this.sPreId+this.callbacks[this.callbacks.length-1];_2=_1.svcConfig.url+"&callback="+cb;}}
var _4=null,doc;doc=document;var _5=this.createElement("script");_5.id=eid;_5.type="text/javascript";var _6=true;if(arguments.length>1){_6=arguments[1];}
if(_6){_2=_2+((_2.indexOf("?")==-1)?"?":"&")+rdm;}
_5.src=_2;this.getReqDiv().appendChild(_5);return eid;},createCallback:function(_7){var _8=this.callbacks.length,name=this.sCallbackName+_8,eid=this.sPreId+name;this.callbacks[_8]=name;this[name]=function(_9){var _a;try{_a=_9;}
catch(e){_a=new vjo.dsf.ServiceResponse();var _b=new vjo.dsf.Error();_b.id="SYS.JSON_PARSE_ERROR";_b.message="SYS.JSON_PARSE_ERROR";_a.errors=[_b];}
this.loaded(eid);_7.response=_a;vjo.dsf.ServiceEngine.handleResponse(_7);};var _c="",rv=((_c)?_c+".":"")+"vjo.dsf.XDomainRequest."+name;return rv;},loaded:function(_d){var e=document.getElementById(_d);if(e!=null){e.parentNode.removeChild(e);}},createElement:function(_f){return(typeof(createElementV4)!="undefined")?createElementV4(_f):document.createElement(_f);}}).inits(function(){vjo.dsf.EventDispatcher.addEventListener(window,"load",this.onLoad,this);}).endType();

vjo.ctype("vjo.dsf.RemoteReqtHdl").needs("vjo.dsf.XDomainRequest").protos({constructs:function(){this.reqTimers={};this.timerCount=0;this.processed={};},handleRequest:function(_1){_1.trace=_1.trace+"-->RemoteHdl_"+_1.svcId;if(_1.svcConfig){this.invoke(_1);}},invoke:function(_2){var _3=vjo.dsf.Service,xmlHttpReq=_3.getXmlHttpReq(),requestUrl=this.appendPort80ForSafari(document.location.href,_2.svcConfig.url),cfg=_2.svcConfig;_2.status=-1;if(cfg.respMarshalling=="JSCALLBACK"){vjo.dsf.XDomainRequest.send(_2);return;}
try{var _4=(cfg.async===false)?false:true;xmlHttpReq.open(cfg.method,requestUrl,_4);}
catch(e){var _5=new vjo.dsf.ServiceResponse();var _6=new vjo.dsf.Error();_6.id="SYS.DARWIN_SERVICE_PROTOCOL_ERROR";_6.message="SYS.PROTOCOL_ERROR: Cannot open URL '"+requestUrl+"'";_5.errors=[_6];_2.response=_5;vjo.dsf.ServiceEngine.handleResponse(_2);return;}
var _7=this.timerCount++;if(_4){this.setupReadyState(xmlHttpReq,_2,_7);}
if(cfg.method=="POST"){xmlHttpReq.setRequestHeader("Content-Type","application/x-www-form-urlencoded");xmlHttpReq.setRequestHeader("Content-Length",_2.rawRequest.length);xmlHttpReq.send(_2.rawRequest);}else{xmlHttpReq.send(null);}
if(!_4&&!this.checkAndSetProcessed(_7)){vjo.dsf.Service.callback(xmlHttpReq,_2);}else{if(cfg.timeout){var _8=this;this.reqTimers[_7]=window.setTimeout(function(){_8.timeout(xmlHttpReq,_2,_7);},cfg.timeout);}}},setupReadyState:function(_9,_a,_b){var _c=this;_9.onreadystatechange=function(){if(_9.readyState!=4){return;}
if(_c.checkAndSetProcessed(_b)){return;}
var _d=_c.reqTimers[_b];if(_d){window.clearTimeout(_d);delete _c.reqTimers[_b];}
vjo.dsf.Service.callback(_9,_a);};},timeout:function(_e,_f,idx){if(this.checkAndSetProcessed(idx)){return;}
delete _e.onreadystatechange;_e.abort();delete this.reqTimers[idx];var _11=new vjo.dsf.ServiceResponse();var _12=new vjo.dsf.Error();_12.id="SYS.DARWIN_SERVICE_PROTOCOL_ERROR";_12.message="SYS.PROTOCOL_ERROR: Service timed out.";_11.errors=[_12];_f.response=_11;vjo.dsf.ServiceEngine.handleResponse(_f);},checkAndSetProcessed:function(idx){if(this.processed[idx]){return true;}
this.processed[idx]=true;return false;},appendPort80ForSafari:function(_14,_15){try{if(navigator.userAgent.toLowerCase().indexOf("safari")>=0){var _16="(([^:]*)://([^:/?]*)(:([0-9]+))?)?([^?#]*)([?]([^#]*))?(#(.*))?",ajaxUrl=_15,safariIssuePort="80";var _17=_14.match(_16);if(_17&&_17.length>=5&&_17[5]&&_17[5]==safariIssuePort){var _18=ajaxUrl.match(_16);if(_18&&_18[2]&&_18[2].length>0){if(_18[5]&&_18[5].length>0){}else{var _19="";if(_18[2]){_19+=_18[2]+"://";}
if(_18[3]){_19+=_18[3]+":"+safariIssuePort;}
if(_18[6]){_19+=_18[6];}
if(_18[8]){_19+="?"+_18[8];}
if(_18[10]){_19+="#"+_18[10];}
_15=_19;}}}}}
catch(e){}
return _15;}}).endType();

vjo.ctype("vjo.dsf.ServiceResponse").protos({constructs:function(){this.objType="dsf_ServiceResponse";this.errors=[];this.data=null;}}).endType();

vjo.ctype("vjo.dsf.InProcReqtHdl").needs("vjo.dsf.ServiceResponse").protos({constructs:function(){this.svcHdls={};},registerSvcHdl:function(_1,_2){this.svcHdls[_1]=_2;},getSvcHdl:function(_3){return this.svcHdls[_3];},handleRequest:function(_4){var _5=this.svcHdls[_4.svcId];if(_5){var _6=new vjo.dsf.ServiceResponse();_6.data=_5.invoke(_4);_4.trace=_4.trace+"-->SvcHdl_"+_4.svcId;if(_6){_4.response=_6;}}
if(typeof _4.status=="undefined"||_4.status==null){_4.status=1;}}}).endType();

vjo.ctype("vjo.dsf.Error").protos({constructs:function(_1,_2){this.id=_1;this.message=_2;}}).endType();

vjo.ctype("vjo.dsf.Enc").inits(function(){if(typeof(vjo.dsf.Enc.unescape)!="undefined"){return;}
vjo.dsf.Enc.unescape=window.unescape;vjo.dsf.Enc.decodeURI=window.decodeURI;vjo.dsf.Enc.decodeURIComponent=window.decodeURIComponent;vjo.dsf.Enc.encodeURIComponent=window.encodeURIComponent;vjo.dsf.Enc.encodeURI=window.encodeURI;}).endType();

vjo.ctype("vjo.dsf.Service").needs(["vjo.dsf.Error","vjo.dsf.ServiceResponse","vjo.dsf.Enc"]).props({callback:function(_1,_2){try{if(_1.readyState!=4){return;}
if(_1.status>=200&&_1.status<300){var _3=_1.responseText;if(_2.svcConfig.respMarshalling=="JSON"){try{_3=eval("("+_3+")");}
catch(e){_3=new vjo.dsf.ServiceResponse();var _4=new vjo.dsf.Error();_4.id="SYS.JSON_PARSE_ERROR";_4.message="SYS.JSON_PARSE_ERROR";_3.errors=[_4];}}else{if(_2.svcConfig.respMarshalling=="XML"){_3=_1.responseXML;}}
_2.response=_3;_2.status=1;}else{var _5=new vjo.dsf.ServiceResponse();var _4=new vjo.dsf.Error();_4.id="SYS.DARWIN_SERVICE_PROTOCOL_ERROR";_4.message="SYS.PROTOCOL_ERROR: status = "+_1.status;_5.errors=[_4];_2.response=_5;}}
catch(e){var _5=new vjo.dsf.ServiceResponse();var _4=new vjo.dsf.Error();_4.id="SYS.DARWIN_SERVICE_PROTOCOL_ERROR";_4.message="SYS.PROTOCOL_ERROR: unknown ";_5.errors=[_4];_2.response=_5;}
vjo.dsf.ServiceEngine.handleResponse(_2);delete _1.onreadystatechange;_1=null;},getXmlHttpReq:function(){var _6=false;try{_6=new ActiveXObject("Msxml2.XMLHTTP");}
catch(e){try{_6=new ActiveXObject("Microsoft.XMLHTTP");}
catch(e){_6=false;}}
if(!_6&&typeof XMLHttpRequest!="undefined"){_6=new XMLHttpRequest();}
return _6;},getClientInfo:function(){if(this.clientInfo){return this.clientInfo;}
var nv=navigator,agt=nv.userAgent.toLowerCase(),i=0,ver=b="";if(agt.indexOf("firefox")!=-1){b="Firefox";i=agt.lastIndexOf("firefox")+8;}else{if((/webkit|khtml/).test(agt)){b="Safari";i=agt.lastIndexOf("safari")+7;}else{if(typeof(window.opera)!="undefined"){b="Opera";i=agt.lastIndexOf("opera")+6;}else{if(nv.appName=="Netscape"){b="Netscape";i=agt.lastIndexOf("/")+1;}else{if(agt.indexOf("msie")!=-1){b="IE";i=agt.indexOf("msie")+4;}}}}}
if(b){ver=parseInt((b=="Opera")?window.opera.version():agt.substring(i));}
this.clientInfo=b+":"+ver+":";return this.clientInfo;},generateReqParams:function(_8){var _9="svcid="+vjo.dsf.Enc.encodeURIComponent(_8.svcId);if(_8.stok){_9+="&stok="+_8.stok;}
if(_8.pId){_9+="&pId="+_8.pId;}
if(_8.v){_9+="&v="+_8.v;}
_9=_9+"&reqttype="+_8.svcConfig.reqtMarshalling;_9=_9+"&resptype="+_8.svcConfig.respMarshalling;_9=_9+"&clientType="+this.getClientInfo();_9+="&request=";var _a=_8.request,reqtmarsh=_8.svcConfig.reqtMarshalling;if(reqtmarsh=="JSON"){_9+=vjo.dsf.Enc.encodeURIComponent(JSON.stringify(_a));}else{if(reqtmarsh=="JSCALLBACK"){_9+=vjo.dsf.Enc.encodeURIComponent(JSON.stringify(_a));}else{if(reqtmarsh=="XML"){_9+=vjo.dsf.Enc.encodeURIComponent(dsf_xmlize(_a,"Request"));}else{_9+=vjo.dsf.Enc.encodeURIComponent(_a);}}}
return _9;},xmlize:function(_b,_c,_d){_d=_d?_d:"";var s=_d+"<"+_c+">";if(!(_b instanceof Object)||_b instanceof Number||_b instanceof String||_b instanceof Boolean||_b instanceof Date){s+=dsf_escape(""+_b);}else{s+="\n";var _f="";var _10=_b instanceof Array;for(var _11 in _b){if(_10&&_11=="______array"){continue;}
s+=this.xmlize(_b[_11],(_10?"array-item key=\""+_11+"\"":_11),_d+"   ");}
s+=_d;}
return s+=(_c.indexOf(" ")!=-1?"</array-item>\n":"</"+_c+">\n");},escape:function(_12){return _12.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&apos;");}}).endType();

vjo.ctype("vjo.dsf.SvcConfig").protos({constructs:function(_1,_2){this.objType="dsf_SvcConfig";this.url=_2;this.method=_1;this.reqtMarshalling="raw";this.respMarshalling="raw";this.async=true;this.timeout=0;}}).endType();

vjo.ctype("vjo.dsf.Message").protos({trspType:"",svcId:"",constructs:function(_1){this.objType="dsf_Message";this.svcId=_1;this.request;this.response;this.rawRequest="";this.clientContext={};this.trspType="InProc";this.status;this.svcConfig;this.returnData=true;this.trace="";this.v="0";}}).endType();

vjo.ctype("vjo.util.Map").protos({constructs:function(_1){this.javaClass=_1||"java.util.HashMap";this.map={};},get:function(_2){return this.map[_2];},put:function(_3,_4){return this.map[_3]=_4;},remove:function(_5){var _6=this.map[_5];delete this.map[_5];return _6;},size:function(){var _7=0;for(var _8 in this.map){_7++;}
return _7;}}).endType();

vjo.ctype("vjo.util.List").protos({constructs:function(_1){this.javaClass=_1||"java.util.ArrayList";this.list=[];},get:function(_2){if(this.size()>_2){return this.list[_2];}
return null;},add:function(_3){return this.list[this.size()]=_3;},remove:function(_4){var _5=this.size(),tmp=this.list,nlist=[],rv=false;for(var i=0;i<_5;i++){if(!rv&&tmp[i]===_4){rv=true;}else{nlist[nlist.length]=tmp[i];}}
this.list=nlist;return rv;},size:function(){return this.list.length;}}).endType();

vjo.needs("vjo.dsf.Json");vjo.ctype("vjo.dsf.ServiceEngine").needs(["vjo.dsf.RemoteReqtHdl","vjo.dsf.InProcReqtHdl","vjo.dsf.Service","vjo.dsf.ServiceResponse","vjo.dsf.SvcConfig","vjo.dsf.Message","vjo.util.Map","vjo.util.List"]).props({STATUS:{ABORT:-1,JUMP:1},init:function(){this.svcReqtHdls={};this.svcRespHdls={};this.glbReqtHdls=[];this.glbRespHdls=[];this.trspReqtHdls={};this.trspRespHdls={};this.svcHdls={};this.inProcHdl=new this.vj$.InProcReqtHdl();this.remoteHdl=new this.vj$.RemoteReqtHdl();this.registerTrspReqtHdl("Remote",vjo.bind(this,this.remoteTrspHdl));this.registerTrspRespHdl("Remote",vjo.bind(this,this.remoteRespTrspHdl));},handleRequest:function(_1){var rt,ab=this.STATUS.ABORT,m=_1;if((m.status!=ab)&&typeof rt=="undefined"){rt=this.processServiceRequestHandlers(_1);}
if((m.status!=ab)&&typeof rt=="undefined"){rt=this.processGlobalRequestHandlers(_1);}
if((m.status!=ab)&&typeof rt=="undefined"){rt=this.processTransportHandlers(_1);}
if((m.status!=ab)&&(_1.trspType!="Remote"||typeof rt!="undefined")){this.handleResponse(_1,rt);}
return _1.returnData;},processServiceRequestHandlers:function(_3){var rt;var _5=this.svcReqtHdls[_3.svcId];if(_5){try{var _6=_5.length;for(var i=0;i<_6;i++){_3.trace=_3.trace+"-->svcReqtHdl_"+i;_5[i].handleRequest(_3);if(_3.status==this.STATUS.JUMP){rt="SVC";this.genResponseError(_3,"SYS.SVC_REQUEST_ERROR","SYS.SVC_REQUEST_ERROR");break;}}}
catch(e){rt="SVC";this.genResponseError(_3,"SYS.SVC_REQUEST_ERROR","SYS.SVC_REQUEST_ERROR");}}
return rt;},processGlobalRequestHandlers:function(_8){var rt;if(_8.status!=this.STATUS.JUMP){try{var _a=this.glbReqtHdls.length;for(var i=0;i<_a;i++){_8.trace=_8.trace+"-->glbReqtHdl_"+i;this.glbReqtHdls[i].handleRequest(_8);if(_8.status==this.STATUS.JUMP){rt="GLB";this.genResponseError(_8,"SYS.GLOBAL_REQUEST_ERROR","SYS.GLOBAL_REQUEST_ERROR");break;}}}
catch(e){rt="GLB";this.genResponseError(_8,"SYS.GLOBAL_REQUEST_ERROR","SYS.GLOBAL_REQUEST_ERROR");}}
return rt;},processTransportHandlers:function(_c){if(_c.status!=this.STATUS.JUMP&&_c.trspType){var _d=this.trspReqtHdls[_c.trspType];if(_d){try{var _e=_d.length;for(var i=0;i<_e;i++){_c.trace=_c.trace+"-->trspReqtHdl_"+i;_d[i].handleRequest(_c);if(_c.status==this.STATUS.JUMP){this.genResponseError(_c,"SYS.TRANS_REQUEST_ERROR","SYS.TRANS_REQUEST_ERROR");break;}}}
catch(e){this.genResponseError(_c,"SYS.TRANS_REQUEST_ERROR","SYS.TRANS_REQUEST_ERROR");}}
if(_c.status!=this.STATUS.JUMP&&_c.status!=this.STATUS.ABORT){if(_c.trspType==="Remote"){this.remoteHdl.handleRequest(_c);}else{this.inProcHdl.handleRequest(_c);}}}
return;},handleResponse:function(_10,_11){if(_10.trspType&&typeof _11=="undefined"){this.processTransResponseHandlers(_10);}
if(_11!="SVC"){this.processGlobalResponseHandlers(_10);}
this.processServiceResponseHandlers(_10);},processTransResponseHandlers:function(msg){var _13=this.trspRespHdls[msg.trspType];try{if(_13){for(var i=_13.length-1;i>=0;i--){msg.trace=msg.trace+"-->trspRespHdl_"+i;_13[i].handleResponse(msg);}}}
catch(e){this.genResponseError(msg,"SYS.TRANS_RESPONSE_ERROR","SYS.TRANS_RESPONSE_ERROR");}},processGlobalResponseHandlers:function(msg){try{for(var i=this.glbRespHdls.length-1;i>=0;i--){msg.trace=msg.trace+"-->glbRespHdl_"+i;this.glbRespHdls[i].handleResponse(msg);}}
catch(e){this.genResponseError(msg,"SYS.GLOB_RESPONSE_ERROR","SYS.GLOB_RESPONSE_ERROR");}},processServiceResponseHandlers:function(msg){var _18;if(msg.clientContext){_18=msg.clientContext.svcApplier;}
try{if(_18){if(typeof _18.onResponse=="function"){_18.onResponse(msg);}else{if(typeof _18=="function"){_18(msg);}}}}
catch(e){this.genResponseError(msg,"SYS.SVC_RESPONSE_ERROR","SYS.SVC_RESPONSE_ERROR");}
var _19=this.svcRespHdls[msg.svcId];if(_19){try{for(var i=_19.length-1;i>=0;i--){msg.trace=msg.trace+"-->svcRespHdl_"+i;_19[i].handleResponse(msg);}}
catch(e){this.genResponseError(msg,"SYS.SVC_RESPONSE_ERROR","SYS.SVC_RESPONSE_ERROR");}}},createHandler:function(_1b,_1c){if(typeof _1b[_1c]!="function"){if(typeof _1b=="function"){var _1d=_1b,obj={},self=this;obj[_1c]=function(){return _1d.apply(self,arguments);};_1b=obj;}}
return _1b;},registerSvcHdl:function(_1e,_1f){if(!_1e||!_1f){return;}
_1f=this.createHandler(_1f,"invoke");this.inProcHdl.registerSvcHdl(_1e,_1f);},getSvcHdl:function(_20){return this.inProcHdl.getSvcHdl(_20);},registerSvcReqtHdl:function(_21,_22){if(!_21||!_22){return;}
if(typeof this.svcReqtHdls[_21]=="undefined"){this.svcReqtHdls[_21]=[];}
var _23=this.svcReqtHdls[_21];_23[_23.length]=this.createHandler(_22,"handleRequest");},registerSvcRespHdl:function(_24,_25){if(!_24||!_25){return;}
if(typeof this.svcRespHdls[_24]=="undefined"){this.svcRespHdls[_24]=[];}
var _26=this.svcRespHdls[_24];_26[_26.length]=this.createHandler(_25,"handleResponse");},registerGlbReqtHdl:function(_27){if(!_27){return;}
this.glbReqtHdls[this.glbReqtHdls.length]=this.createHandler(_27,"handleRequest");},registerGlbRespHdl:function(_28){if(!_28){return;}
this.glbRespHdls[this.glbRespHdls.length]=this.createHandler(_28,"handleResponse");},registerTrspReqtHdl:function(_29,_2a){if(!_29||!_2a){return;}
if(typeof this.trspReqtHdls[_29]=="undefined"){this.trspReqtHdls[_29]=[];}
var _2b=this.trspReqtHdls[_29];_2b[_2b.length]=this.createHandler(_2a,"handleRequest");},registerTrspRespHdl:function(_2c,_2d){if(!_2c||!_2d){return;}
if(typeof this.trspRespHdls[_2c]=="undefined"){this.trspRespHdls[_2c]=[];}
var _2e=this.trspRespHdls[_2c];_2e[_2e.length]=this.createHandler(_2d,"handleResponse");},remoteTrspHdl:function(_2f){var cfg=_2f.svcConfig;if(!cfg||cfg.objType!="dsf_SvcConfig"){return;}else{if(cfg.respMarshalling=="JSCALLBACK"){if(typeof vjo.dsf.assembly!="undefined"&&typeof vjo.dsf.assembly.VjClientAssembler!="undefined"&&!vjo.dsf.assembly.VjClientAssembler.bBodyLoaded){vjo.dsf.assembly.VjClientAssembler.load(_2f);_2f.status=-1;return;}}}
if(_2f.request&&_2f.request.javaClass){delete _2f.request.b;}
var svc=vjo.dsf.Service,requestParams=svc.generateReqParams(_2f),requestUrl=cfg.url;if(_2f.svcConfig.method=="GET"){requestUrl=requestUrl+"?"+requestParams;}else{_2f.rawRequest=requestParams;}
_2f.svcConfig.url=requestUrl;},remoteRespTrspHdl:function(_32){var _33=_32.response;if(_33!=null&&_33.data!=null){this.processData(_33.data);}},processData:function(_34){this.processObj(_34);for(var _35 in _34){var o=_34[_35];if(o!=null&&typeof o=="object"){this.processData(o);}}},processObj:function(obj){var _38=obj.javaClass;if(_38&&_38.length>0){if(/java.util.([^\s])*List/.test(_38)){this.addMethods(obj,vjo.util.List.prototype);}else{if(/java.util.([^\s])*Map/.test(_38)){this.addMethods(obj,vjo.util.Map.prototype);}}}},addMethods:function(obj,_3a){for(var key in _3a){obj[key]=_3a[key];}},genResponseError:function(msg,_3d,_3e){if(typeof msg.response=="undefined"){var _3f=new vjo.dsf.ServiceResponse();msg.response=_3f;}
var _40=new vjo.dsf.Error(_3d,_3e);msg.response.errors[msg.response.errors.length]=_40;},register:function(_41,_42,_43){var _s=vjo.dsf.ServiceEngine;switch(_41){case 0:_s.registerSvcHdl(_42,_43);break;case 1:_s.registerSvcReqtHdl(_42,_43);break;case 2:_s.registerGlbReqtHdl(_42,_43);break;case 3:_s.registerTrspReqtHdl(_42,_43);break;case 4:_s.registerSvcRespHdl(_42,_43);break;case 5:_s.registerGlbRespHdl(_42,_43);break;case 6:_s.registerTrspRespHdl(_42,_43);break;}}}).inits(function(){this.init();}).endType();

vjo.ctype("vjo.dsf.assembly.VjClientAssemblerRequest").protos({constructs:function(_1,_2,_3,_4,_5){this.sUrl=_1;this.fCallback=_2;this.oScope=_3||window;this.sCallbackParam=_4;var b=_5;if(typeof(b)=="undefined"){b=true;}
this.bSendResponseOnLoad=b;}}).endType();

vjo.ctype("vjo.dsf.assembly.VjClientAssembler").needs(["vjo.dsf.EventDispatcher","vjo.dsf.ServiceEngine","vjo.dsf.assembly.VjClientAssemblerRequest"]).singleton().protos({constructs:function(){this.aCallbacks=[];this.aResponses={};this.aModels={};this.sPreCallback="_callback";this.bBodyLoaded=false;this.bLock=false;this.loaded={};},load:function(_1){var m=_1,url=m.sUrl||"",name=this.generateCallback(m),ver="",cb=((ver)?ver+".":"")+"vjo.dsf.assembly.VjClientAssembler."+name;if(m&&m.objType=="dsf_Message"){url=m.svcConfig.url+"?";url+=vjo.dsf.Service.generateReqParams(m)+"&callback="+cb;}else{if(m.sCallbackParam){url=url+"&"+m.sCallbackParam+"="+cb;}}
return vjo.dsf.XDomainRequest.send(url);},generateCallback:function(_3){var m=_3;var _5=this.aCallbacks.length;var _6=this.sPreCallback+_5;this.aCallbacks[_5]=_6;this.aModels[_6]=m;this[_6]=function(){if(this.loaded[_6]===true){return;}
this.loaded[_6]=true;if(this.bBodyLoaded||!m.bSendResponseOnLoad){if(m.objType=="dsf_Message"){m.response=this.getResponse(arguments[0]);vjo.dsf.ServiceEngine.handleResponse(m);}else{m.fCallback.apply(m.oScope,arguments);}}else{if(m.objType=="dsf_Message"){this.aResponses[_6]=this.getResponse(arguments[0]);m.response=this.aResponses[_6];}else{this.aResponses[_6]=arguments;}
if(this.bBodyLoaded){this.assemble();}}};return _6;},assemble:function(){this.bBodyLoaded=true;if(this.bLock){setTimeout("vjo.dsf.assembly.VjClientAssembler.assemble()",1000);return;}
this.bLock=true;try{for(var _7 in this.aResponses){this.loaded[_7]=true;var m=this.aModels[_7];if(this.aResponses[_7]!=null){if(m.objType=="dsf_Message"){vjo.dsf.ServiceEngine.handleResponse(m);}else{m.fCallback.apply(m.oScope,this.aResponses[_7]);}}
this.aResponses[_7]=null;}}
finally{this.bLock=false;}},getResponse:function(_9){var _a;try{_a=_9;}
catch(e){_a=new vjo.dsf.ServiceResponse();var _b=new vjo.dsf.Error();_b.id="SYS.JSON_PARSE_ERROR";_b.message="SYS.JSON_PARSE_ERROR";_a.errors=[_b];}
return _a;}}).inits(function(){vjo.dsf.assembly.VjClientAssembler=new vjo.dsf.assembly.VjClientAssembler();vjo.dsf.EventDispatcher.addEventListener(window,"load",function(){vjo.dsf.assembly.VjClientAssembler.assemble();});}).endType();

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
var val,r="";var _14=this.aConversionMap[_11];if(_14){val=this.readCookieObj(_14[0],_14[1]).value||"";}
if(val){r=this.getCookielet(_11,_12,val)||"";}
return(typeof(r)!="undefined")?r:"";},writeCookie:function(_15,_16,_17){var _18=this.aConversionMap[_15];if(_18){this.writeCookielet(_18[0],_18[1],_16,_17);return;}
var _19=this.getFormat(_15);if(_16&&_19.escapedValue){_16=encodeURIComponent(_16);}
this.writeRawCookie(_15,_16,_17);},writeRawCookie:function(_1a,_1b,_1c){if(_1a&&(_1b!==undefined)){if((isNaN(_1b)&&_1b.length<4000)||(_1b+"").length<4000){if(typeof _1c=="number"){_1c=this.getExpDate(_1c);}
var _1d=_1c?new Date(_1c):new Date(this.getExpDate(730));var _1e=this.getFormat(_1a);var _1f=this.sCookieDomain;var dd=document.domain;if(!dd.has(_1f)){var _21=dd.indexOf(".ebay.");if(_21>0){this.sCookieDomain=dd.substring(_21);}}
if(document.cookie){document.cookie=_1a+"="+(_1b||"")+((_1c||_1e.bUseExp)?"; expires="+_1d.toGMTString():"")+"; domain="+this.sCookieDomain+"; path="+this.sPath;}}}},writeCookieEx:function(_22,_23,_24){this.writeCookie(_22,_23,this.getExpDate(_24));},writeCookielet:function(_25,_26,_27,_28,_29){if(_25&&_26){this.update();var _2a=this.getFormat(_25);if(_2a.bUseExp&&_27){if(typeof _28=="number"){_28=this.getExpDate(_28);}
var _2b=_28?new Date(_28):new Date(this.getExpDate(730));var _2c=Date.UTC(_2b.getUTCFullYear(),_2b.getUTCMonth(),_2b.getUTCDate(),_2b.getUTCHours(),_2b.getUTCMinutes(),_2b.getUTCSeconds());_2c=Math.floor(_2c/1000);_27+=_2c.dec2Hex();}
var val=this.createCookieValue(_25,_26,_27);this.writeRawCookie(_25,val,_29);}},writeMultiLineCookie:function(_2e,_2f,_30,_31,_32){this.update();var val=this.createCookieValue(_2e,_2f,_30);if(val){var _34=this.aConversionMap[_2e];if(_34){this.writeCookielet(_34[0],_34[1],val,_31,_32);}}},getBitFlagOldVersion:function(_35,_36){_35=parseInt(_35,10);var b=_35.toString(2),r=_35?b.charAt(b.length-_36-1):"";return(r=="1")?1:0;},setBitFlagOldVersion:function(_38,_39,_3a){var b="",p,i,e,l;_38=parseInt(_38,10);if(_38){b=_38.toString(2);}
l=b.length;if(l<_39){e=_39-l;for(i=0;i<=e;i++){b="0"+b;}}
p=b.length-_39-1;return parseInt(b.substring(0,p)+_3a+b.substring(p+1),2);},getBitFlag:function(_3c,_3d){if(_3c!=null&&_3c.length>0&&_3c.charAt(0)=="#"){var _3e=_3c.length;var q=_3d%4;var _40=Math.floor(_3d/4)+1;var _41=_3e-_40;var _42=parseInt(_3c.substring(_41,_41+1),16);var _43=1<<q;return((_42&_43)==_43)?1:0;}else{return this.getBitFlagOldVersion(_3c,_3d);}},setBitFlag:function(_44,_45,_46){if(_44!=null&&_44.length>0&&_44.charAt(0)=="#"){var _47=_44.length;var q=_45%4;var _49=Math.floor(_45/4)+1;if(_47<=_49){if(_46!=1){return _44;}
var _4a=_49-_47+1;var _4b=_44.substring(1,_47);while(_4a>0){_4b="0"+_4b;_4a--;}
_44="#"+_4b;_47=_44.length;}
var _4c=_47-_49;var _4d=parseInt(_44.substring(_4c,_4c+1),16);var _4e=1<<q;if(_46==1){_4d|=_4e;}else{_4d&=~_4e;}
_44=_44.substring(0,_4c)+_4d.toString(16)+_44.substring(_4c+1,_47);return _44;}else{if(_45>31){return _44;}
return this.setBitFlagOldVersion(_44,_45,_46);}},createCookieValue:function(_4f,_50,_51){var _52=this.aConversionMap[_4f],format=this.getFormat(_4f),mode=this.getMode(_4f),val;if(_52&&(mode==this.sSTRICT||mode==this.sCONVER)){val=this.readCookieObj(_52[0],_52[1]).value||"";}else{val=this.aCookies[_4f]||"";}
if(format){var _53=this.getCookieletArray(val,format);_53[_50]=_51;var str="";for(var i in _53){if(_53[i]){str+=i+format.NAME_VALUE_DELIMITER+_53[i]+format.COOKIELET_DELIMITER;}}
if(str&&format.startDelim){str=format.startDelim+str;}
val=str;if(format.escapedValue){val=encodeURIComponent(val);}}
return val;},update:function(){var aC=document.cookie.split("; ");this.aCookies={};for(var i=0;i<aC.length;i++){var sC=aC[i].split("=");var _59=this.getFormat(sC[0]),cv=sC[1],sd=_59.startDelim;if(sd&&cv&&cv.indexOf(sd)===0){sC[1]=cv.substring(sd.length,cv.length);}
this.aCookies[sC[0]]=sC[1];}},getCookielet:function(_5a,_5b,_5c){var _5d=this.getFormat(_5a);var _5e=this.getCookieletArray(_5c,_5d);return _5e[_5b]||"";},getFormat:function(_5f){return this.aFormatMap[_5f]||vjo.dsf.cookie.VjCookieJar.Default_Cookie_Format;},getCookieletArray:function(_60,_61){var rv=[],val=_60||"";if(_61.escapedValue){val=decodeURIComponent(val);}
var a=val.split(_61.COOKIELET_DELIMITER);for(var i=0;i<a.length;i++){var idx=a[i].indexOf(_61.NAME_VALUE_DELIMITER);if(idx>0){rv[a[i].substring(0,idx)]=a[i].substring(idx+1);}}
return rv;},getExpDate:function(_66){var _67;if(typeof _66=="number"&&_66>=0){var d=new Date();d.setTime(d.getTime()+(_66*24*60*60*1000));_67=d.toGMTString();}
return _67;},getMode:function(_69){var h=this.readCookieObj(this.sModesCookie,this.sModesCookielet).value,b;if(!(_69 in this.aConversionMap)){return null;}
if(!h){return"";}
if(h===0){return this.sSTRICT;}
if(h&&h!="0"){if(h.has(".")){var a=h.split(".");for(i=0;i<a.length;i++){b=a[i].hex2Dec().toString(2)+b;}}else{b=h.hex2Dec().toString(2);}
i=0;var l=b.length,j;for(o in this.aConversionMap){j=l-(2*(i+1));f=b.substring(j,j+2).toString(10);f=(!f)?this.sSTRICT:f;if(_69==o){return(f.length==1)?"0"+f:f;}
i++;}
return null;}}}).inits(function(){var vCJ=vjo.dsf.cookie.VjCookieJar;vCJ.aFormatMap={"r":vCJ.Default_Cookie_Format,"dp1":vCJ.DP_Cookie_Format,"npii":vCJ.DP_Cookie_Format,"ebay":vCJ.Session_Cookie_Format,"reg":vCJ.Session_Cookie_Format,"apcCookies":this.Session_Cookie_Format,"ds2":vCJ.DS_Cookie_Format};}).endType();

vjo.ctype("vjo.darwin.core.sitespeed.SiteSpeed").needs(["vjo.dsf.EventDispatcher","vjo.dsf.cookie.VjCookieJar"]).props({params:{},gauge:function(){this.gaugeInternal();},gaugeUnload:function(){this.addParam("ex2","1");this.gaugeInternal();},gaugeBodyLoad:function(){if(typeof(oGaugeInfo)!="undefined"){var _1=(new Date()).getTime()-oGaugeInfo.iST;this.addParam("ctb",_1);}},addParam:function(_2,_3){if(_2&&_3){this.params[_2]=_3;}},getParams:function(){var rv="";for(var k in this.params){rv+="&"+k+"="+this.params[k];}
return rv;},gaugeInternal:function(){var _6=vjo.dsf.cookie.VjCookieJar,sbf=_6.readCookie("ebay","sbf");_6.writeCookielet("ebay","sbf",_6.setBitFlag(sbf,20,1));var _7=this.getParams();if(typeof(oGaugeInfo)!="undefined"&&oGaugeInfo.bFlag!=1){if(oGaugeInfo.sent===true){return;}
oGaugeInfo.sent=true;var _8=new Image(1,1),delta=(new Date()).getTime()-oGaugeInfo.iST;if(_8){_8.src=oGaugeInfo.sUrl.replace(/&amp;/g,"&")+delta+_7;}}}}).inits(function(){var _9=vjo.dsf.cookie.VjCookieJar;sbf=_9.readCookie("ebay","sbf");b=(sbf)?_9.getBitFlag(sbf,20):0;if(typeof(oGaugeInfo)!="undefined"){oGaugeInfo.bFlag=b;oGaugeInfo.sent=false;}
var _a=vjo.darwin.core.sitespeed.SiteSpeed;vjo.dsf.EventDispatcher.addEventListener(window,"load",_a.gauge,_a);vjo.dsf.EventDispatcher.addEventListener(window,"beforeunload",_a.gaugeUnload,_a);}).endType();

vjo.ctype("vjo.Registry").props({controls:[],put:function(_1,_2){this.controls[_1]=_2;if(this.isKeyValid(_1)){this["_"+_1]=this.controls[_1];}
return this.controls[_1];},get:function(_3){return this.controls[_3];},dump:function(){var _4=this.controls;var _5="controls on page:\n";for(var i in _4){_5+="key = "+i;_5+="controlName = "+_4[i].objtype;_5+="\n";}
return _5;},isKeyValid:function(_7){if(typeof _7!="string"){return false;}
return/^([a-zA-Z0-9_$]+)$/.test(_7);}}).endType();

vjo.ctype("vjo.dsf.client.Browser").needs("vjo.dsf.typeextensions.string.Comparison").props({init:function(){this.bFirefox=this.bWebTV=this.bOpera=this.bNav=this.bIE=this.bSafari=this.bWin=this.bMac=this.bMacppc=this.bMactel=this.bActiveXSupported=this.bWinXp=this.bXpSp2=this.bAOL=this.bVista=false;this.iVer=this.fVer=-1;this.fMinorVer=0;this.aMimeTypes=null;var nv=navigator,agt=nv.userAgent.toLowerCase(),i=0,ver;with(this){if(agt.has("webtv")){bWebTV=true;i=agt.indexOf("webtv/")+6;}else{if(agt.has("firefox")){bFirefox=true;i=agt.lastIndexOf("firefox")+8;}else{if(agt.has("safari")){bSafari=true;i=agt.lastIndexOf("safari")+7;}else{if(typeof(window.opera)!="undefined"){bOpera=true;i=agt.lastIndexOf("opera")+6;}else{if(nv.appName.is("Netscape")){bNav=true;i=agt.lastIndexOf("/")+1;}else{if(agt.has("msie")){bIE=true;i=agt.indexOf("msie")+4;if(agt.has("aol")||agt.has("america online")){bAOL=true;}}}}}}}
ver=bOpera?window.opera.version():agt.substring(i);iVer=parseInt(ver);fVer=parseFloat(ver);fMinorVer=fVer-iVer;bWin=agt.has("win");bWinXp=(bWin&&agt.has("windows nt 5.1"));bVista=(bWin&&agt.has("windows nt 6.0"));bXpSp2=(bWinXp&&agt.has("sv1"));bMac=agt.has("mac");bMacppc=(bMac&&agt.hasAny("ppc","powerpc"));bMactel=(bMac&&agt.has("intel"));aMimeTypes=nv.mimeTypes;bActiveXSupported=(!(bMac||bMacppc)&&(typeof(ActiveXObject)=="function"));}}}).inits(function(){vjo.dsf.client.Browser.init();}).endType();

vjo.ctype("vjo.dsf.document.Element").needs("vjo.dsf.Element","E").props({toggleHideShowRow:function(_1,_2){},toggleHideShow:function(_3,_4){},toggleVisibility:function(_5,_6){}}).inits(function(){vjo.dsf.document.Element=this.vj$.E;}).endType();

vjo.ctype("vjo.dsf.utils.Handlers").needs(["vjo.dsf.EventDispatcher","vjo.dsf.Message","vjo.dsf.ServiceEngine"]).props({ED:vjo.dsf.EventDispatcher,SE:vjo.dsf.ServiceEngine,attachEvt:function(_1,_2,_3,_4){return this.ED.addEventListener(_1,_2,_3,_4);},detachEvt:function(_5,_6,_7){return this.ED.removeEventListener(_5,_6,_7);},newMsg:function(_8){return new vjo.dsf.Message(_8);},handle:function(_9){this.SE.handleRequest(_9);},createHdl:function(_a,_b,_c){var _d=_a,hdl={};hdl[_c]=function(){_b.apply(_d,arguments);};return hdl;},attachSvc:function(_e,_f,_10){var t=this,hdl=t.createHdl(_10,_f,"invoke");if(t.SE&&hdl){t.SE.registerSvcHdl(_e,hdl);}},attachSvcReqt:function(_12,_13,_14){var t=this,hdl=t.createHdl(_14,_13,"handleRequest");if(t.SE&&hdl){t.SE.registerSvcReqtHdl(_12,hdl);}},attachSvcResp:function(_16,_17,_18){var t=this,hdl=t.createHdl(_18,_17,"handleResponse");if(t.SE&&hdl){t.SE.registerSvcRespHdl(_16,hdl);}},resetSvc:function(_1a){this.SE.inProcHdl.svcHdls[_1a]=[];},resetSvcReqt:function(_1b){this.SE.svcReqtHdls[_1b]=[];},resetSvcResp:function(_1c){this.SE.svcRespHdls[_1c]=[];}}).endType();

vjo.ctype("vjo.dsf.utils.Object").props({hitch:function(_1,_2){var _3;if(typeof _2=="string"){_3=_1[_2];}else{_3=_2;}
return function(){return _3.apply(_1,arguments);};},extend:function(_4,_5){function inheritance(){}
inheritance.prototype=_5.prototype;_4.prototype=new inheritance();_4.prototype.constructor=_4;_4.baseConstructor=_5;_4.superClass=_5.prototype;}}).endType();

vjo.ctype("vjo.dsf.document.Form").props({get:function(_1){var f=document.forms[_1];return f?f:null;},getElem:function(_3,_4){var f=this.get(_3),e;if(f){e=f.elements[_4];}
return e?e:null;},submit:function(_6){var f=this.get(_6);if(f){f.submit();}},setAction:function(_8,_9){this.setAttr(_8,"action",_9);},getAction:function(_a){return this.getAttr(_a,"action");},setTarget:function(_b,_c){this.setAttr(_b,"target",_c);},getTarget:function(_d){return this.getAttr(_d,"target");},setAttr:function(_e,_f,_10){var f=this.get(_e);if(f){eval("f."+_f.toLowerCase()+"=psAttrValue;");}},getAttr:function(_12,_13){var f=this.get(_12),v=null;if(f){v=eval("f."+_13.toLowerCase());}
return v;}}).endType();

vjo.ctype("vjo.dsf.window.utils.VjWindow").props({open:function(_1,_2,_3,_4,_5,_6,_7){if(_5){var _8=(window.screen.width-_6)/2;var _9=(window.screen.height-_7)/2;_3+=",left="+_8+",top="+_9;}
return window.open(_1,_2,_3,_4);},location:function(_a){document.location.href=_a;},alert:function(_b){window.alert(_b);},confirm:function(_c){return window.confirm(_c);}}).endType();

vjo.ctype("vjo.darwin.core.button.Button").needs("vjo.dsf.Element","E").needs("vjo.dsf.Message","M").needs("vjo.dsf.ServiceEngine","S").needs("vjo.dsf.client.Browser").needs("vjo.dsf.EventDispatcher","ED").protos({constructs:function(m){var t=this;t.m=m;t.state="";var Sc=m.scope;t.dCls=Sc+"-b "+"moz "+Sc+"-b";t.Cls=[{"mouseup":[0,""],"mousedown":[0,"p"],"mouseout":[0,""],"mouseover":[0,"o"],"focus":[1,"o"],"blur":[1,""],"entrue":"d","enfalse":""}];t.O=[];t.init();},init:function(){var t=this,D=t.vj$.ED,O=t.get();D.addEventListener(window,"load",function(){t.enDis(t.m.dis);},window);for(var i in t.Cls[0]){var V=O[t.Cls[0][i][0]];if(V){D.addEventListener(V,i,t.handler(i),t);}}},handler:function(eT){var t=this;return function(){t.changeState({"eventType":eT});};},get:function(){var t=this,E=t.vj$.E,m=t.m;if(t.O.length==0){t.O.push(E.get(m.SId),E.get(m.BId),E.get(m.LId));}
return t.O;},changeState:function(e){var t=this,O=t.get(),eT=e.eventType;if((O[1].disabled||O[0].className.indexOf("-bd")!=-1)||(t.state=="mousedown"&&eT=="focus")){return;}
t.state=eT;O[0].className=t.dCls+t.Cls[0][eT][1]+" "+t.m.BT;},enable:function(){this.enDis(false);},disable:function(){this.enDis(true);},isDisabled:function(){return this.m.dis;},enDis:function(_c){var t=this,N="none",E=t.vj$.E,m=t.m,O=t.get(),b=O[1],bS=b.style,tS=(O[2])?O[2].style:"",B=vjo.dsf.client.Browser.bIE;O[0].className=t.dCls+t.Cls[0]["en"+_c]+" "+t.m.BT;t.m.dis=_c;if(m.tp!==2){b.disabled=_c;}
if(!B&&m.tp!=2){return;}
if(_c){tS.display="inline-block";if(!B){tS.display="-moz-inline-box";b.style.setProperty("display","none","important");}
bS.display=N;}else{tS.display=N;bS.display="";}}}).props({enable:function(_e){var o=vjo.Registry.get(_e);if(o){o.enable();}},disable:function(ins){var o=vjo.Registry.get(ins);if(o){o.disable();}},onSubmit:function(_12,evt,_14){this.sendMessage(_12,evt);return(!_14&&_14==false)?false:true;},sendMessage:function(_15,evt){var o=this.vj$,msg=new o.M(_15);msg.event=evt;o.S.handleRequest(msg);}}).endType();

vjo.ctype("vjo.darwin.core.memberbadge.MemberBadge").needs(["vjo.dsf.document.Element","vjo.dsf.window.utils.VjWindow"]).props({openLink:function(_1){if(_1){var _2=_1;var _3=window.screen.width;vjo.dsf.window.utils.VjWindow.open(_1,"","width=440,height=500,resizable=yes,top=0,left="+(_3-450)+",location=no,menubar=no,scrollbars=no,status=no",false,false);return false;}}}).endType();

vjo.ctype("vjo.dsf.utils.Timer").needs("vjo.dsf.utils.Object").protos({timer:null,isRunning:false,interval:null,onTick:function(){},onStart:null,onStop:null,constructs:function(_1){this.interval=_1;},setInterval:function(ms){var t=this;if(t.isRunning){window.clearInterval(t.timer);}
t.interval=ms;if(t.isRunning){t.setInt();}},start:function(){var t=this;if(typeof t.onStart=="function"){t.onStart();}
t.isRunning=true;t.setInt();},stop:function(){var t=this;if(typeof t.onStop=="function"){t.onStop();}
t.isRunning=false;window.clearInterval(t.timer);},setInt:function(){var t=this;t.timer=window.setInterval(vjo.dsf.utils.Object.hitch(t,"onTick"),t.interval);}}).endType();

vjo.ctype("vjo.dsf.document.Shim").needs("vjo.dsf.client.Browser").props({add:function(_1,_2,_3){var f,p="px",w,h,s;if(this.check()){w=_1.offsetWidth;h=_1.offsetHeight;w+=_2?_2:0;h+=_3?_3:0;f=document.createElement("IFRAME");s=f.style;s.width=w+p;s.height=h+p;s.filter="chroma(color='white')";f.frameBorder=0;s.position="absolute";s.left="0"+p;s.top="0"+p;s.zIndex="-1";s.filter="Alpha(Opacity=\"0\")";if(document.location.protocol=="https:"){f.src="https://securepics.ebaystatic.com/aw/pics/s.gif";}
_1.appendChild(f);return f;}
return null;},remove:function(_5,_6){if(this.check()){if(_6&&_6.parentNode){_6.parentNode.removeChild(_6);}}},check:function(){var B=vjo.dsf.client.Browser;return(B.bIE||B.bFirefox);}}).endType();

vjo.ctype("vjo.dsf.document.Positioning").props({getScrollLeftTop:function(){var d=document,rv=[0,0],db=d.body,de=d.documentElement;if(db){rv[0]+=db.scrollLeft;rv[1]+=db.scrollTop;}
if(de){rv[0]+=de.scrollLeft;rv[1]+=de.scrollTop;}
return rv;},getOffsetLeft:function(_2){var e=_2,l=0;while(e){l+=e.offsetLeft;e=e.offsetParent;}
return l;},getOffsetTop:function(_4){var e=_4,t=0;while(e){t+=e.offsetTop;e=e.offsetParent;}
return t;},getClientWidth:function(){var s=self,d=document,de=d.documentElement,w;if(s.innerWidth){w=s.innerWidth;}else{if(de&&de.clientWidth){w=de.clientWidth;}else{w=d.body.clientWidth;}}
return w;},getClientHeight:function(){var s=self,d=document,de=d.documentElement,h;if(s.innerHeight){h=s.innerHeight;}else{if(de&&de.clientHeight){h=de.clientHeight;}else{h=d.body.clientHeight;}}
return h;},getEventLeftTop:function(_8){var u="undefined",evt=window.event||_8,xOff=(typeof(screenLeft)!=u)?screenLeft:screenX,yOff=(typeof(screenTop)!=u)?screenTop:(screenY+(outerHeight-innerHeight)-25);return[evt.screenX-xOff,evt.screenY-yOff];}}).endType();

vjo.ctype("vjo.darwin.core.utils.ServiceUtils").needs("vjo.dsf.ServiceEngine","SE").needs("vjo.dsf.Message","M").props({rgSv:function(_1,_2){this.vj$.SE.registerSvcHdl(_1,_2);},rgSvRsp:function(_3,_4){this.vj$.SE.registerSvcRespHdl(_3,_4);},sndM:function(_5){this.vj$.SE.handleRequest(typeof(_5)=="object"?_5:this.gM(_5));},gM:function(_6){return new this.vj$.M(_6);}}).endType();

vjo.ctype("vjo.darwin.core.mask1.Mask").needs("vjo.dsf.EventDispatcher","EV").needs("vjo.dsf.Element","E").needs("vjo.darwin.core.utils.ServiceUtils","SE").needs("vjo.dsf.client.Browser").protos({constructs:function(id,_2,_3,_4,_5,_6){var t=this,b=vjo.dsf.client.Browser;t.IE=(b.bIE&&b.iVer<=6);var o;if(!o){t.o=o=t.vj$.E.createElement("DIV");o.id=id;}
t.minTries=0;t.maxTries=10;t.promote2bodyFunc=function(){var _9=document.documentElement.doScroll;try{if(t.minTries>=t.maxTries){return;}
if(_9){_9("left");}
var b=document.body;if(b){b.appendChild(t.o);}
t.isPromoted=true;}
catch(err){t.minTries++;setTimeout(t.promote2bodyFunc,200);}};var _b=(t.IE)?_3:_2;if(o){var s=t.st=o.style;s.position="absolute";s.display="none";s.top="0";s.left="0";s.zIndex=_6?_6:5000;s.background="transparent url("+_b+") repeat left top";}
t.ar=[];var f1=function(m){t.open(m);};var f2=function(m){t.close(m);};var j=t.vj$.SE;j.rgSv(_4+id,f1);j.rgSv(_5+id,f2);},open:function(_12){var t=this,s=t.st;if(!t.isPromoted){t.promote2bodyFunc();}
s.display="block";t.setWH();if(t.IE){t.dEs(true,_12.containerId);}
var rF=function(){t.setWH();};t.vj$.EV.addEventListener(window,"resize",rF,window);},close:function(){var t=this;t.st.display="none";if(t.IE){t.dEs(false);}},mkAr:function(_16){var t=this,sels=document.body.getElementsByTagName("select"),k=0,l=sels.length,id,cntr=t.vj$.E.get(_16);for(id=0;id<l;id++){var s=sels[id],jk=t.vj$.E.containsElement(cntr,s);if(!jk){this.ar[k++]=s;}}},dEs:function(_19,_1a){var t=this.ar,i;if(_19){this.mkAr(_1a);}
var l=t.length;for(i=0;i<l;i++){t[i].disabled=_19;}},setWH:function(){var t=this.st,u="px",de=document.documentElement;t.height=de.scrollHeight+u;t.width=de.scrollWidth+u;}}).props({sndMsg:function(_1e,id){var o=vjo.darwin.core.utils.ServiceUtils,m=o.gM(_1e);m.containerId=id;o.sndM(m);}}).endType();

vjo.ctype("vjo.dsf.utils.URL").props({addArg:function(_1,_2,_3){if(_1==null||_1==undefined){return null;}
if(_1.indexOf("?")<0){_1+="?"+_2+"="+_3;return _1;}
var _4=this.getArgPairIfExists(_1,_2);if(_4!==null){_1=_1.replace(_4,_2+"="+_3);}else{_1+="&"+_2+"="+_3;}
return _1;},getArg:function(_5,_6){if(_5==null||_5==undefined){return null;}
if(_5.indexOf("?")<0){return null;}
var _7=this.getArgPairIfExists(_5,_6);if(_7!==null){return _7.substring(_7.indexOf("=")+1);}
return null;},getArgPairIfExists:function(_8,_9){var _a=_8.indexOf("?");if(_a<0){return null;}
var _b=_8;var _c,argName;while(_a>=0){_b=_b.substring(_a+1);_c=_b;_a=_b.indexOf("&");if(_a>=0){_c=_b.substring(0,_a);}
argName=_c.substring(0,_c.indexOf("="));if(argName==_9){return _c;}}
return null;}}).endType();

vjo.ctype("vjo.dsf.client.ActiveX").needs("vjo.dsf.client.Browser").props({init:function(){var oC=vjo.dsf.client.Browser;if(oC.bIE){var d=document,dw;dw=function(s){d.writeln(s);};dw("<scr"+"ipt language=\"vbscript\" type=\"text/vbscript\">");dw("\tFunction vbCheckActiveXControl (pActXName)");dw("\t\taX = false");dw("\t\ton error resume next");dw("\t\taX = IsObject(CreateObject(pActXName))");dw("\t\tvbCheckActiveXControl = aX");dw("End Function");dw("</scr"+"ipt>");}},isLibLoaded:function(_4){var oC=vjo.dsf.client.Browser;return oC.bActiveXSupported&&vbCheckActiveXControl(_4);}}).inits(function(){vjo.dsf.client.ActiveX.init();}).endType();

vjo.ctype("vjo.darwin.core.rtm.RTMHelper").props({setRtm:function(_1){if(_1){this.oRtm=_1;}},render:function(_2){var t=this;if(t.oRtm){t.oRtm.renderPromos(_2);}},onMessage:function(_4){this.oRtm.onMessage(_4);}}).endType();

vjo.itype("vjo.dsf.common.IJsServiceHandler").protos({invoke:function(_1){}}).endType();

vjo.ctype("vjo.darwin.core.rtm.RTMInit").needs(["vjo.dsf.utils.URL","vjo.darwin.core.rtm.RTMHelper"]).satisfies("vjo.dsf.common.IJsServiceHandler").protos({constructs:function(_1){this.oRTM=new vjo.darwin.core.rtm.RTM(_1);vjo.darwin.core.rtm.RTMHelper.setRtm(this.oRTM);},invoke:function(_2){var _3=this.oRTM.processGlobalNavPids(true);var _4=this.oRTM.processGlobalNavPids(false);if(_3!=""){var U=vjo.dsf.utils.URL;var _6=this.oRTM.oJSBean.url;_6=U.addArg(_6,"e","USC:"+this.oRTM.getSegment(false));var p=U.getArg(_6,"p");var ph=U.getArg(_6,"ph");if((p!=null)&&!p.has(_3)){this.oRTM.oJSBean.url=_6.replace("p="+p,"p="+p+":"+_3);if(ph!=null){this.oRTM.oJSBean.url=this.oRTM.oJSBean.url.replace("ph="+ph,"ph="+ph+":"+_4);}}
if(typeof(_oGlobalNavRTMInfo)!="undefined"){_oGlobalNavRTMInfo.aRTMPlacementData=[];}}
this.oRTM.appendUrl("&z="+this.oRTM.getFlashVersion());this.oRTM.appendUrl("&bw="+this.oRTM.getBrowserWidth());if(this.oRTM.getCIDCookie()!=null&&this.oRTM.getCIDCookie()!=""&&this.oRTM.getCIDCookie()!="undefined"){this.oRTM.appendUrl("&cg="+this.oRTM.getCIDCookie());}
if(this.oRTM.getEncodingType()!=null){this.oRTM.appendUrl("&enc="+this.oRTM.getEncodingType());}
this.oRTM.appendUrl("&v=4");this.oRTM.registerGlobalNavPlacements();if(!this.oRTM.oJSBean.onload){this.onSend();}else{vjo.dsf.EventDispatcher.addEventListener(window,"load",this.onLoad,this);}
vjo.dsf.EventDispatcher.addEventListener(window,"load",this.oRTM.setTimeOut,this.oRTM);_2.returnData=false;return _2;},onLoad:function(){var _9=this;window.setTimeout(function(){_9.onSend();},0);},onSend:function(){var _a=this.oRTM,opt=_a.bOptimize,func=opt?_a.storeResponse:_a.setInlineContent;var _b=new vjo.dsf.assembly.VjClientAssemblerRequest(_a.oJSBean.url,func,_a,"cb",!opt);vjo.dsf.assembly.VjClientAssembler.load(_b);}}).endType();

vjo.ctype("vjo.dsf.utils.Bit").needs("vjo.dsf.cookie.VjCookieJar").props({CJ:vjo.dsf.cookie.VjCookieJar,getMulti:function(_1,_2,_3){var r="",i,CJ=this.CJ;for(i=0;i<_3;i++){r=CJ.getBitFlag(_1,_2+i)+r;}
return parseInt(r,2);},setMulti:function(_5,_6,_7,_8){var i=0,CJ=this.CJ,v,l,e;v=_8.toString(2).substring(0,_7);l=v.length;if(l<_7){e=_7-l;for(j=0;j<e;j++){v="0"+v;}
l=l+e;}
for(i=0;i<l;i++){_5=CJ.setBitFlag(_5,_6+i,v.substring(l-i-1,l-i));}
return _5;}}).endType();

vjo.ctype("vjo.dsf.flash.Version").needs(["vjo.dsf.client.Browser","vjo.dsf.client.ActiveX","vjo.dsf.utils.Bit","vjo.dsf.cookie.VjCookieJar"]).props({versions:[10,9],get:function(){var t=this,B=t.vj$.Browser,v=0,vs=t.versions,i,A=t.vj$.ActiveX,cv;cv=t.rw(false);if(cv){return(cv==1)?0:cv;}
if(B.bIE&&B.bWin&&!B.bOpera){for(i=0;i<vs.length;i++){if(A.isLibLoaded("ShockwaveFlash.ShockwaveFlash."+vs[i])){v=vs[i];break;}}}else{var n=navigator,pd,id,swf="Shockwave Flash";if(n.plugins[swf]){pd=n.plugins[swf].description;id=pd.indexOf("Flash")+5;v=parseInt(pd.substr(id,pd.length));}
if(B.bWebTV){v=3;}}
t.rw(true,v);return v;},rw:function(_3,_4){var t=this,n=t.vj$,C=n.VjCookieJar,B=n.Bit;cl=C.readCookie("ebay","sbf");if(!_3){return B.getMulti(cl,40,5);}else{if(_3){_4=(_4==0)?1:_4;C.writeCookielet("ebay","sbf",B.setMulti(cl,40,5,_4));}}}}).endType();

vjo.ctype("vjo.dsf.utils.JsLoader").props({queue:[],pending:null,load:function(_1,_2,_3){var _4={url:_1,callback:_2,scope:_3},head,stag;var t=this;if(t.pending){t.queue.push(_4);return;}
this.pending=_4;head=document.getElementsByTagName("head")[0];stag=document.createElement("script");stag.onload=stag.onreadystatechange=function(){if(!this.readyState||this.readyState=="loaded"||this.readyState=="complete"){t.oncomplete();stag.onload=stag.onreadystatechange=null;head.removeChild(stag);}};stag.type="text/javascript";stag.src=_1;head.appendChild(stag);},oncomplete:function(){var t=this,o=t.pending;if(o.callback){o.callback.call(o.scope||window);}
t.pending=null;if(t.queue.length>0){var _7=this.queue.shift();t.load(_7.url,_7.callback,_7.scope);}}}).endType();

vjo.ctype("vjo.dsf.utils.CssLoader").props({load:function(_1){if(document.createStyleSheet){document.createStyleSheet(_1);}else{var _2=document.getElementsByTagName("head")[0],style=document.createElement("link");style.rel="stylesheet";style.type="text/css";style.href=_1;_2.appendChild(style);}}}).endType();

vjo.ctype("vjo.darwin.core.rtm.RTM").needs(["vjo.dsf.typeextensions.string.Comparison","vjo.dsf.cookie.VjCookieJar","vjo.dsf.client.Browser","vjo.dsf.client.ActiveX","vjo.dsf.EventDispatcher","vjo.dsf.flash.Version","vjo.darwin.core.rtm.RTMInit","vjo.dsf.utils.JsLoader","vjo.dsf.utils.CssLoader"]).protos({constructs:function(_1){this.oJSBean=_1||{};this.oJSBean.quickPids=this.oJSBean.quickPids||[];this.aContent=[];this.iContentLen=0;this.bTimedOut=false;this.oTimeoutId=null;this.iTIMEOUT=3000;this.oClient=vjo.dsf.client.Browser;this.bResponseReturned=false;this.iOrd=(new Date()).getTime();this.aGlobalNavPlacements=null;this.promos=new Object();this.globalAdOverlay=null;vjo.Registry.put("GlobalRtmInstance",this);this.response=null;this.renderedStatus={};this.bBodyLoaded=false;this.renderCSS=false;this.bOptimize=true;this.debug=false;if(this.bOptimize){vjo.dsf.EventDispatcher.add("body","load",this.render,this);}},getFlashVersion:function(){return this.vj$.Version.get();},getEncodingType:function(){return"UTF-8";},setInlineContent:function(_2){if(this.TimedOut()){return;}
this.hidePromos(this.promos);this.aContent=_2;this.iContentLen=this.aContent.length;var _3=false;this.bPromoScript=false;this.bPromoStyle=false;if(!this.renderCSS){for(var j=0;j<this.iContentLen;j++){var _5=this.aContent[j];if(_5.CSSMetaData){this.renderCSS=true;this.processCSS(_5.CSSMetaData);}}}
for(var i=0;i<this.iContentLen;i++){with(this){var _5=aContent[i];if(_5.JSMetaData){this.processJS(_5.JSMetaData);}else{if(!_5.CSSMetaData){this.processPromo(_5);}}}}
var _7=new vjo.dsf.Message("RTM_COMPLETE");_7.status=1;_7.vjRTMObject=this;vjo.dsf.ServiceEngine.handleRequest(_7);if(_3){var _7=new vjo.dsf.Message("PROMO_COMPLETE");_7.vjPromoClientObject=this;vjo.dsf.ServiceEngine.handleRequest(_7);}
return;},TimedOut:function(){this.bResponseReturned=true;if(this.bTimedOut){return true;}
if(this.oTimeoutId){window.clearTimeout(this.oTimeoutId);}
return false;},processJS:function(_8){if(_8.JSURLs){var _9=_8.JSURLs;for(var i=0;i<_9.length;i++){vjo.dsf.utils.JsLoader.load(_9[i]);}}
if(_8.EventHandlers){var _b=_8.EventHandlers;for(i=0;i<_b.length;i++){var _c=document.createElement("script");_c.text=_b[i];document.getElementsByTagName("head")[0].appendChild(_c);}}},processCSS:function(_d){if(_d.CSSURLs){var _e=_d.CSSURLs;for(var i=0;i<_e.length;i++){vjo.dsf.utils.CssLoader.load(_e[i]);}}},processPromoResponse:function(_10){var map=_10.content.data.map;var _12=0;var id="-1";var _14=false;for(var _15 in map){var _16=_15.match(/ME([0-9]+)(.*)/);if(_16){_12++;id=_10.id;}}
var _17=map.VisualPreview;if(_17){this.loadVisualPreview(map.VisualPreview);}
var _18=map.Style;if(_18&&!this.bPromoStyle){this.loadStyle(_18);this.bPromoStyle=true;}
for(var _15 in map){var _16=_15.match(/ME([0-9]+)(.*)/);if(_16){promoLoaded=this.loadPromo(_10.id,_16[2],map[_15],_12>1);if(promoLoaded){_14=true;}}}
var _19=map.Script;if(_19&&!this.bPromoScript){this.bPromoScript=this.loadScript(_19);}
if(_14){var _1a=this.getIndex(id);var _1b=this.getUIElement(this.oJSBean.htmlIds[_1a]);if(_1b&&!_1b.length){_1b.style.display="block";}}else{this.processNoneAd(_10);}
return;},loadVisualPreview:function(_1c){var _1d=this.getUIElement("VisualPreviewContent");if(_1d==null){_1d=document.createElement("div");_1d.name="VisualPreviewContent";document.body.appendChild(_1d);}
_1d.innerHTML=_1c;},loadStyle:function(_1e){var _1f=document.body.appendChild(document.createElement("style"));_1f.setAttribute("type","text/css");if(_1f.styleSheet){_1f.styleSheet.cssText=_1e;}else{_1f.appendChild(document.createTextNode(_1e));}},loadPromo:function(id,_21,_22,_23){var _24=this.getIndex(id);var _25;if(_23){_25=this.oJSBean.merchPrefix+id+_21;}else{_25=this.oJSBean.htmlIds[_24];}
var _26=this.getUIElement(_25);if(this.getStatus(_25)){return true;}
if(_26&&!_26.length){this.addStatus(_25,true);var _27=document.createElement("div");_27.innerHTML=_22;_26.appendChild(_27);return true;}else{this.addStatus(_25,false);}
return false;},hidePromos:function(_28){for(var _29 in _28){var _2a=document.getElementById(_29);if(_2a!=null){_2a.style.display="none";}}},loadScript:function(_2b){with(window){try{eval(_2b);return true;}
catch(except){}}
return false;},processPopUnderAd:function(pAd){var _2d="height="+pAd.height;_2d+=",width="+pAd.width;_2d+=",menubars=no,scrollbars=no'";var id="p_u_"+pAd.id;var _2f=window.open("",id,_2d);if(_2f){_2f.blur();_2f.document.open();_2f.document.write(pAd.content);_2f.document.close();}
return;},processDoubleClickAd:function(pAd){if(!pAd){return;}
var _31=this.oJSBean;var _32=this.getIndex(pAd.id);var id=_31.htmlIds[_32];var _34=this.getUIElement(id);var url=_31.dblclkUrls[_32];if(!url){return;}
if(this.getStatus(id)){return;}
if(_34&&!_34.length){this.addStatus(id,true);if(pAd.content!=""){url+=pAd.content+";";}
url+="ord="+this.iOrd;_34.innerHTML=this.createIframe(id,url,_31.heights[_32],_31.widths[_32]);_34.style.display="block";}else{this.addStatus(id,false);}
return;},processNoneAd:function(pAd){var _37=this.oJSBean;var _38=this.getIndex(pAd.id);var id=_37.htmlIds[_38];var _3a=this.getUIElement(id);var _3b=_37.defaultUrls[_38];if(this.getStatus(id)){return;}
if(_3a&&!_3a.length){this.addStatus(id,true);if(!_3b||_3b=="collapse"||_3b==""){_3a.style.height="0px";_3a.style.height="0px";_3a.style.display="none";}else{_3a.innerHTML=this.createIframe(id,_3b,_37.heights[_38],_37.widths[_38]);}}else{this.addStatus(id,false);}
return;},processHTMLAd:function(pAd){if(pAd.height=="-1"||pAd.height=="9999"){pAd.height="auto";}
if(pAd.width=="-1"||pAd.width=="9999"){pAd.width="auto";}
var _3d=this.getIndex(pAd.id);var _3e=this.getUIElement(this.oJSBean.htmlIds[_3d]);if(this.getStatus(this.oJSBean.htmlIds[_3d])){return;}
if(_3e&&!_3e.length){this.addStatus(this.oJSBean.htmlIds[_3d],true);var _3f=_3e.style,h,w;_3f.height=h=(pAd.height.has("auto"))?pAd.height:pAd.height+"px";_3f.width=w=(pAd.width.has("auto"))?pAd.width:pAd.width+"px";if((w!="auto")&&(h!="auto")){if(pAd.expand){_3f.textAlign="left";}else{_3f.overflow="hidden";}}
_3e.innerHTML=pAd.content;_3e.style.display="block";var id=pAd.id;if(id==184||id==188||id==218||id==569||id==570){this.executeScript(id);}}else{this.addStatus(this.oJSBean.htmlIds[_3d],false);}
return;},executeScript:function(id){var _42="rtm_html_"+id;var _43=document.getElementById(_42).getElementsByTagName("script")[0];if(_43!=null){var _44=document.createElement("script");_44.type="text/javascript";if(_43.text){_44.text=_43.text;}else{_44.textContent=_43.textContent;}
document.getElementById(_42).appendChild(_44);}},processHTMLFormAd:function(pAd){if(pAd.height=="-1"||pAd.height=="9999"){pAd.height="auto";}
if(pAd.width=="-1"||pAd.width=="9999"){pAd.width="auto";}
var _46=this.getIndex(pAd.id);var _47=this.getUIElement(this.oJSBean.htmlIds[_46]);if(this.getStatus(this.oJSBean.htmlIds[_46])){return;}
if(_47&&!_47.length){this.addStatus(this.oJSBean.htmlIds[_46],true);var _48=_47.style,h,w;_48.height=h=(pAd.height.has("auto"))?pAd.height:pAd.height+"px";_48.width=w=(pAd.width.has("auto"))?pAd.width:pAd.width+"px";if(h!="auto"&&w!="auto"){_48.overflow="hidden";}
var _49=document.createElement("iframe");_49.setAttribute("hspace",0);_49.setAttribute("vspace",0);_49.setAttribute("width","100%");_49.setAttribute("frameBorder",0);_49.setAttribute("scrolling","no");_49.setAttribute("marginWidth",0);_49.setAttribute("marginHeight",0);_47.appendChild(_49);_49.doc=null;if(_49.contentDocument){_49.doc=_49.contentDocument;}else{if(_49.contentWindow){_49.doc=_49.contentWindow.document;}else{if(_49.document){_49.doc=_49.document;}}}
if(_49.doc==null){throw"Document not found, append the parent element to the DOM before creating the IFrame";}
_49.doc.open();try{_49.doc.write(pAd.content);}
finally{_49.doc.close();}}else{this.addStatus(this.oJSBean.htmlIds[_46],false);}
return;},getUIElement:function(_4a){var s=_4a,d=window.document;if(d.getElementById){return d.getElementById(s);}else{if(d.all){return d.all(s);}}
return null;},getIndex:function(pId){for(var i=0;i<this.iContentLen;i++){if(this.aContent[i].id==pId){return i;}}
return;},createIframe:function(pId,_4f,_50,_51){var _52="ifrm_"+pId;var f="<iframe frameborder=\"no\" border=\"0\" marginwidth=\"0\" marginheight=\"0\" scrolling=\"no\""+" id=\""+_52+"\""+" name=\""+_52+"\""+" src=\""+_4f+"\""+" width=\""+_51+"\" height=\""+_50+"\"></iframe>";return f;},appendUrl:function(_54){this.oJSBean.url+=_54;},processGlobalNavPids:function(_55){var _56="";var _57;if(typeof(_oGlobalNavRTMInfo)!="undefined"){_57=_oGlobalNavRTMInfo;this.aGlobalNavPlacements=_57.aRTMPlacementData;}
if(_57&&this.aGlobalNavPlacements&&this.aGlobalNavPlacements.length>0){var _58=_oGlobalNavRTMInfo.aRTMPlacementData,data;for(i=0;i<_58.length;i++){if(_55){data=_58[i];}
if(_56){_56+=":";}
_56+=_55?data.pid:"0";}}
return _56;},registerGlobalNavPlacements:function(){if(!this.aGlobalNavPlacements||this.aGlobalNavPlacements.length==0){return;}
var _59=this.oJSBean;var len=this.aGlobalNavPlacements.length;for(var i=0;i<len;i++){var _5c=_59.htmlIds.length;var _5d=this.aGlobalNavPlacements[i];_59.htmlIds[_5c]=(_5d.htmlId)?_5d.htmlId:"glbl_nav_no_html_id";_59.pids[_5c]=(_5d.pid)?_5d.pid:"glbl_nav_no_pid";_59.heights[_5c]=(_5d.maxHeight)?_5d.maxHeight:"glbl_nav_no_height";_59.widths[_5c]=(_5d.maxWidth)?_5d.maxWidth:"glbl_nav_no_width";_59.dblclkUrls[_5c]=(_5d.dblclkUrl)?_5d.dblclkUrl:"glbl_nav_no_dblclk";_59.defaultUrls[_5c]=(_5d.defaultUrl)?_5d.defaultUrl:"collapse";if(_5d.renderBeforeOnload&&_59.quickPids){_59.quickPids[_59.quickPids.length]=_5d.pid;}}},setTimeOut:function(){if(!this.bResponseReturned){var _5e="vjo.darwin.core.rtm.RTM.processTimeOut()";var _5f=this;var _60=function(){_5f.processTimeOut();};this.oTimeoutId=window.setTimeout(_60,this.iTIMEOUT);}},processTimeOut:function(){this.bTimedOut=true;var _61=this.oJSBean;var _62=_61.defaultUrls.length;for(var i=0;i<_62;i++){var id=_61.htmlIds[i];var _65=this.getUIElement(id);var _66=_61.defaultUrls[i];if(this.getStatus(id)){continue;}
if(_65&&!_65.length){this.addStatus(id,true);if(!_66||_66=="collapse"||_66==""){_65.style.height="0px";_65.style.height="0px";_65.style.display="none";}else{_65.innerHTML=this.createIframe(id,_66,_61.heights[i],_61.widths[i]);}}else{this.addStatus(id,false);}}
var _67=Math.random();if(_67<0.05){var _68=_61.url;var _69=_68.indexOf("&");_68=_68.substring(0,_69);_68=_68+"&p="+_61.pids.join(":")+"&ite=2&to="+this.iTIMEOUT;_68=_68.replace("RtmCmd","RtmIt");var _6a=document.createElement("script");_6a.type="text/javascript";_6a.src=_68;document.getElementsByTagName("head")[0].appendChild(_6a);}
window.clearTimeout(this.oTimeoutId);var _6b=new vjo.dsf.Message("RTM_COMPLETE");_6b.status=0;_6b.vjRTMObject=this;vjo.dsf.ServiceEngine.handleRequest(_6b);return;},getBrowserWidth:function(){var b=this.oClient;var _6d=document.body.clientWidth;if(!b.bIE){_6d=window.innerWidth;}
return _6d;},popUp:function(_6e,_6f,_70,_71,_72,_73,_74,_75,_76,_77,_78,_79){var sP="";sP+=(_70!=null)?",width="+_70:"";sP+=(_71!=null)?",height="+_71:"";sP+=(_75!=null)?",screenX="+_75+",left="+_75:"";sP+=(_76!=null)?",screenY="+_76+",top="+_76:"";sP+=",toolbar="+((_73)?"1":"0");sP+=",location="+((_77)?"1":"0");sP+=",status="+((_72)?"1":"0");sP+=",scrollbars="+((_74)?"1":"0");sP+=",resizable="+((_78)?"1":"0");sP+=",menubar="+((_79)?"1":"0");if(sP.length>0){sP=sP.substring(1);}
window.open(_6f,_6e,sP);return false;},getSegment:function(_7b){var oCJ=vjo.dsf.cookie.VjCookieJar,e=oCJ.readCookie("etfc"),r=oCJ.readCookie("reg"),s=oCJ.readCookie("ebay","sin"),c,n;if(e=="0"){n="3";c="E";}else{if(e=="1"){n="4";c="C";}else{if(e=="2"){n="5";c="D";}else{if((e==""&&(r!=""&&r!=";"))||s=="in"||e=="5"){n="2";c="B";}else{n="1";c="A";}}}}
return _7b?c:n;},getCIDCookie:function(){var oCJ=vjo.dsf.cookie.VjCookieJar;var cid=oCJ.readCookie("npii","cguid");if(cid!="undefined"&&cid!=""){return cid;}},openReportAd:function(_7f,_80,_81,_82,_83,_84,_85,_86,_87,_88){if(this.globalAdOverlay==null){this.globalAdOverlay=new vjo.darwin.core.rtm.ReportAd(_81,_82,_83,_84,_87,_88,_80);}
if(this.globalAdOverlay!=null){this.globalAdOverlay.init(_85,_86);this.globalAdOverlay.open(_7f);}},closeReportAd:function(){if(this.globalAdOverlay!=null){if(this.globalAdOverlay){this.globalAdOverlay.close();}}},submitReportAd:function(){if(this.globalAdOverlay!=null){if(this.globalAdOverlay){this.globalAdOverlay.submitReport();}}},yahooAdBckGrnd:function(_89){var _8a=_89;var oFr=document.getElementById(_8a);oFr.style.visibility="visible";oFr.parentNode.style.backgroundColor="transparent";},addStatus:function(_8c,_8d){this.renderedStatus[_8c]=_8d;},getStatus:function(_8e){return this.renderedStatus[_8e];},storeResponse:function(_8f){var t=this,c=_8f,qp;t.response=c;t.aContent=c;t.iContentLen=c.length;if(t.bBodyLoaded){t.setInlineContent(c);}else{if(!this.TimedOut()){qp=t.oJSBean.quickPids;if(qp&&qp.length>0){t.renderPromos(qp);}}}
if(_8f!=null){for(var i=0;i<_8f.length;i++){if(_8f[i]!=null&&_8f[i].content!=null&&_8f[i].content.indexOf("openReportAd")>0){this.addReportAdScript();break;}}
if(!t.renderCSS){for(var j=0;j<_8f.length;j++){var _93=_8f[j];if(_93.CSSMetaData){t.renderCSS=true;t.processCSS(_93.CSSMetaData);}}}}
if(_8f!=null&&_8f.indexOf("openReportAd")>0){this.addReportAdScript();}},addReportAdScript:function(){try{if(this.oJSBean.reportAdJsUrl){var _94=document.createElement("script");_94.type="text/javascript";_94.src=this.oJSBean.reportAdJsUrl;document.getElementsByTagName("head")[0].appendChild(_94);}}
catch(er){}},render:function(){var t=this,r=t.response;if(r){t.setInlineContent(r);}
t.bBodyLoaded=true;t.iBodyLoadedTime=(new Date()).getTime();},processPromo:function(_96){with(this){if(_96.type=="doubleclick"){processDoubleClickAd(_96);}else{if(_96.type=="html"){processHTMLAd(_96);}else{if(_96.type=="popUnder"){processPopUnderAd(_96);}else{if(_96.type=="promo"){bHasPromo=true;processPromoResponse(_96);}else{if(_96.type=="htmlform"){processHTMLFormAd(_96);}else{processNoneAd(_96);}}}}}}},renderPromos:function(_97){var r=this.response,i,j;if(r){for(j=0;j<r.length;j++){for(i=0;i<_97.length;i++){if(r[j].id==_97[i]+""){this.processPromo(r[j]);}}}}},decodeMsg:function(msg){msg=unescape(msg);if(msg.charAt(0)=="#"){msg=msg.substring(1);}
msg="|"+msg;return msg;},getMsgParam:function(msg,_9b,_9c){var v=null;var s=msg.indexOf("|"+_9b+":");if(s>-1){s=s+_9b.length+2;var e=msg.indexOf("|",s);if(msg.charAt(s)=="{"){e=msg.indexOf("}",s)+1;}
if(e<0){e=msg.length;}
v=msg.substring(s,e);}
if(_9c||(typeof(_9c)=="number")){if(!v||(v=="")){v=_9c;}}
return v;},getIDFromMsg:function(msg){var id=this.getMsgParam(msg,"id");var i=id.lastIndexOf("_");if(i>-1){id=id.substring(i+1);}
return id;},setParentOverflowByClass:function(_a3,_a4,_a5){var e=_a3;while(e.parentNode){if(e.className==_a4){e.style.overflow=_a5;break;}
e=e.parentNode;}},throwError:function(_a7){throw new Error("eBayAdsRuntime - "+_a7);},processMsg:function(_a8,pid,_aa,ad,msg){var _ad=(_a8=="ad.frame.expand");if(_ad&&!ad.allowExpandOnLoad){var _ae=this.iBodyLoadedTime;var end=(new Date()).getTime();var _b0=3000;if(_ae&&((end-_ae)<_b0)){this.throwError("can't expand right after page onload");}}
if(_ad||(_a8=="ad.frame.collapse")){var _b1="rtm_iframe_"+pid;var _b2="rtm_html_"+pid;var ifr=document.getElementById(_b1);var div=document.getElementById(_b2);if(!ifr){_b1="ifrm_"+pid;ifr=document.getElementById(_b1);}
if(!div){_b2="rtm_div_"+pid;div=document.getElementById(_b2);if(!div){_b2="single_rtm_"+pid;div=document.getElementById(_b2);}}
var dir="";var _b6=ad.width;var _b7=ad.height;if(_ad){dir=this.getMsgParam(msg,"direction");if(!dir){this.throwError("no direction");}
dir=dir.toLowerCase();var _b8=ad.expand.toLowerCase();if(!ad.expand||(_b8.indexOf(dir)<0)){this.throwError("wrong direction: "+dir+", should be: "+_b8);}
var _b6=this.getMsgParam(msg,"width");if(_b6>ad.maxWidth){this.throwError("width exceeds max: "+_b6+", should be < "+ad.maxWidth);}
var _b7=this.getMsgParam(msg,"height");if(_b7>ad.maxHeight){this.throwError("height exceeds max: "+_b7+", should be < "+ad.maxHeight);}}
if(!ifr||!div){this.throwError("no div or iframe");}
var i=dir.indexOf("-");if(i>-1){if(dir.indexOf("-",i+1)>-1){this.throwError("invalid direction: "+dir);}else{this.processMsgDir(dir.substring(0,i),_ad,ifr,div,_b6,_b7);this.processMsgDir(dir.substring(i+1),_ad,ifr,div,_b6,_b7);}}else{this.processMsgDir(dir,_ad,ifr,div,_b6,_b7);}
var url=this.oJSBean.url;var i=url.indexOf("?");if(i>-1){url=url.substring(0,i);}
var _bb=7;if(_ad){_bb=6;}
var sb=[];sb.push(url+"?RtmIt");sb.push("&ite="+_bb);sb.push("&m="+_aa);sb.push("&ord="+(new Date()).getTime());var i=new Image();i.src=sb.join("");}},processMsgDir:function(dir,_be,ifr,div,_c1,_c2){var ie=(typeof(window.addEventListener)=="undefined");var ie6=(navigator.userAgent.indexOf("MSIE 6.")>-1);var ff=(navigator.userAgent.indexOf("Firefox")>-1);if(_be){ifr.style.zIndex=9000;if(ie6){div.style.overflow="hidden";}else{if(ie){div.style.overflow="visible";}}
if(dir=="up"){ifr.style.position="relative";var ot=_c2-ifr.height;ifr.height=_c2;ifr.style.top=-ot+"px";}else{if(dir=="down"){ifr.height=_c2;}else{if(dir=="left"){ifr.style.position="relative";var ol=_c1-ifr.width;ifr.width=_c1;ifr.style.left=-ol+"px";}else{if(dir=="right"){ifr.width=_c1;this.setParentOverflowByClass(ifr,"ff-left","visible");}else{this.throwError("invalid direction: "+dir);}}}}}else{ifr.style.zIndex=0;if(ie){div.style.overflow="hidden";}
var _c8=((ifr.width==_c1)&&(Math.abs(_c2-ifr.height)<30));ifr.style.position="absolute";ifr.style.top=null;ifr.style.left=null;ifr.width=_c1;ifr.height=_c2;this.setParentOverflowByClass(ifr,"ff-left","hidden");if(_c8){this.throwError("already collapsed");}}},onMessage:function(msg){try{msg=this.decodeMsg(msg);var pid=this.getIDFromMsg(msg);var st=this.getMsgParam(msg,"st");var _cc=this.getMsgParam(msg,"m");if(!pid||!st||!_cc){this.throwError("no id, st, or m");}
var i=this.getIndex(pid);if(typeof(i)!="number"){this.throwError("no placement "+pid);}
var ad=this.aContent[i];if(!ad){this.throwError("no ad "+pid);}
if(st!=ad.st){this.throwError("token mismatch");}
var _cf=this.getMsgParam(msg,"topic");this.processMsg(_cf,pid,_cc,ad,msg);}
catch(e){if(this.debug){throw e;}}}}).props({siteCatalyst:function(_d0,_d1,PID,_d3){var esc=vjo.darwin.tracking.sitecatalyst;if(esc){var _d5="Search",loc="North";if(PID==188||PID==570){_d5="Browse";}
if(PID==569||PID==570){loc="Sky";}
var s=esc.EbaySiteCatalyst.s;s.linkTrackVars="prop27";s.linkTrackEvents="None";s.prop27=_d1+":"+_d5+":"+PID+":"+_d3;if(_d0){s.linkTrackVars="prop28";s.prop28=s.prop27;s.prop27=null;}
s.tl(true,"o","RTM "+loc+" Ad");}}}).endType();

vjo.ctype("vjo.darwin.core.moduletab.ModuleTab").protos({constructs:function(_1){var m=_1;var t=this;t.sHtmlName=m.htmlName;t.sactiveRgt=m.activeRgt;t.sactiveLft=m.activeLft;t.shighlightLft=m.highlightLft;t.shighlightRgt=m.highlightRgt;t.sinactiveLft=m.inactiveLft;t.sinactiveRgt=m.inactiveRgt;t.shoverLft=m.hoverLft;t.shoverRgt=m.hoverRgt;t.stabcontentOuterOff=m.tabcontentOuterOff;t.stabcontentOuterOn=m.tabcontentOuterOn;t.sContentElementId=null;t.sActiveBkgColor=m.activeBkgColor;t.aTabs=[];t.iCurrTab=m.activeTabId;}}).endType();

vjo.ctype("vjo.darwin.core.moduletab.TabProperties").needs("vjo.Registry").needs("vjo.darwin.core.utils.ServiceUtils","SU").needs("vjo.dsf.Message","M").protos({_elm:null,_rg:null,constructs:function(_1){_elm=vjo.dsf.Element;_rg=vjo.Registry;m=_1;var t=this;t.hN=m.htmlName;t.iId=m.id;t.sUrl=m.url;t.bIsActive=m.active;t.bIsHighlighted=m.highlighted;t.bIsAjaxEnabled=m.ajaxEnabled;t.clkSvcId=m.clkSvcId;var _n=_rg.get(t.hN);if(_n!==null){var _4=_n.aTabs;_4[_4.length]=t;}
t.bindEventsInJS();t.attachEventListner(t.hN,t.clkSvcId);},bindEventsInJS:function(){var t=this;vjEd=vjo.dsf.EventDispatcher,_tProp=vjo.darwin.core.moduletab.TabProperties;var _6=t.hN+"_tab_rgt_"+t.iId,tbL=t.hN+"_tab_lft_"+t.iId,tb=t.hN+"_"+t.iId;if(t.sUrl===null){t.sUrl="";}
var _f=function(_8){return _tProp.switchToTab(t.hN,t.iId,_8,t.clkSvcId);};vjEd.add(tbL,"click",_f);vjEd.add(_6,"click",_f);vjEd.add(_6,"mouseover",function(_9){return _tProp.hoverChange(t.hN,t.iId,true);});vjEd.add(_6,"mouseout",function(_a){return _tProp.hoverChange(t.hN,t.iId,false);});},attachEventListner:function(_b,_c){var t=vjo.darwin.core.moduletab.TabProperties,fn=function(){t.initializer(_b,_c);};vjo.dsf.EventDispatcher.addEventListener(window,"load",fn);}}).props({tLft:"_tab_lft_",tRgt:"_tab_rgt_",tCnt:"_content_",tCurrId:"CurrId",switchToTab:function(_e,_f,_10,_11){var t=this,ob=t.vj$,pTab=_rg.get(_e),pArrTab=pTab.aTabs,hid=_elm.get(_e+t.tCurrId),currId=(hid)?hid.value:1,from,to;for(i=0;i<pArrTab.length;i++){if(currId==pArrTab[i].iId){from=i;}
if(_f==pArrTab[i].iId){to=i;}}
var tp=vjo.darwin.core.moduletab.TabProperties;var _14=_e+t.tRgt+pArrTab[from].iId,fTbL=_e+t.tLft+pArrTab[from].iId;var _15=_e+t.tRgt+pArrTab[to].iId,tTbL=_e+t.tLft+pArrTab[to].iId;if(!pArrTab){return;}
if(pArrTab[to].bIsActive===true||pArrTab[to].iId!=_f||pArrTab[to].dsbl==true){return;}
var rt=_elm.get(_15),url=pArrTab[to].sUrl;if(_10&&(!pArrTab[to].bIsAjaxEnabled)&&(url&&url.length>1&&url.toLowerCase().indexOf("javascript")<0)){var src=_10.nativeEvent.target?_10.nativeEvent.target:_10.nativeEvent.srcElement;if(src.tagName.toLowerCase()==="a"){src.blur();return;}
document.location.href=pArrTab[to].sUrl;return;}
if(hid){hid.value=_f;}
tp.setTabs(_f,pArrTab[from],_e,pTab,_11);tp.setTabs(_f,pArrTab[to],_e,pTab,_11);},sendMsg:function(_18,tb,rTb,lTb,url,cnt){var t=this,SU=t.vj$.SU,msg=SU.gM(_18);msg.tb=tb;msg.rTb=rTb;msg.lTb=lTb;msg.url=url;msg.cnt=cnt;SU.sndM(msg);},setClz:function(_1f,_20){var el;el=_elm.get(_1f);if(el){el.className=_20;}},setBkgColor:function(_22,_23){var el;el=_elm.get(_22);if(el){el.style.backgroundColor=_23;}},initializer:function(_25,_26){var t=this,pTab=_rg.get(_25),pArrTab=pTab.aTabs,tp=vjo.darwin.core.moduletab.TabProperties,hid=_elm.get(_25+tp.tCurrId),currId=(hid)?hid.value:1,to;for(i=0;i<pArrTab.length;i++){if(currId==pArrTab[i].iId){to=i;break;}}
if(pArrTab){if(pArrTab[to].bIsActive===true||pArrTab[to].dsbl==true){return;}
for(var i=0;i<pArrTab.length;i++){tp.setTabs(currId,pArrTab[i],_25,pTab,_26);}}},setTabs:function(_29,_2a,_2b,_2c,_2d){var t=vjo.darwin.core.moduletab.TabProperties;var _2f=_2b+t.tRgt+_2a.iId,fTbL=_2b+t.tLft+_2a.iId,tTcnt=_2b+t.tCnt+_2a.iId,url=_2a.sUrl;if(_29==_2a.iId){_2a.bIsActive=true;if(_2d){t.sendMsg(_2d,_29,_2f,fTbL,url,tTcnt);}
if(_2c.sActiveBkgColor!==null){t.setBkgColor(fTbL,_2c.sActiveBkgColor);t.setBkgColor(_2f,_2c.sActiveBkgColor);}
t.setClz(fTbL,_2c.sactiveLft);t.setClz(_2f,_2c.sactiveRgt);t.setClz(_2b+t.tCnt+_2a.iId,_2c.stabcontentOuterOn);}else{_2a.bIsActive=false;t.setBkgColor(fTbL,"");t.setBkgColor(_2f,"");if(_2a.bIsHighlighted===true){t.setClz(fTbL,_2c.shighlightLft);t.setClz(_2f,_2c.shighlightRgt);}else{t.setClz(fTbL,_2c.sinactiveLft);t.setClz(_2f,_2c.sinactiveRgt);}
t.setClz(_2b+t.tCnt+_2a.iId,_2c.stabcontentOuterOff);}},hoverChange:function(_30,_31,_32){var _33=_rg.get(_30),tp=vjo.darwin.core.moduletab.TabProperties,pArrTab=_33.aTabs;for(var i=0;i<pArrTab.length;i++){if(pArrTab[i].iId==_31){to=i;break;}}
var _35=_30+tp.tRgt+pArrTab[to].iId,tTbL=_30+tp.tLft+pArrTab[to].iId;if(pArrTab){if(pArrTab[to].bIsActive===true||pArrTab[to].dsbl==true){return;}
if(pArrTab[to].bIsHighlighted===true){if(_32===true){tp.setClz(tTbL,_33.shighlightLft+" "+_33.shoverLft);tp.setClz(_35,_33.shighlightRgt+" "+_33.shoverRgt);}else{tp.setClz(tTbL,_33.shighlightLft);tp.setClz(_35,_33.shighlightRgt);}}else{if(_32===true){tp.setClz(tTbL,_33.sinactiveLft+" "+_33.shoverLft);tp.setClz(_35,_33.sinactiveRgt+" "+_33.shoverRgt);}else{tp.setClz(tTbL,_33.sinactiveLft);tp.setClz(_35,_33.sinactiveRgt);}}}}}).endType();

vjo.ctype("vjo.darwin.core.overlaypanel3.OverlayPanel").needs("vjo.dsf.client.Browser","B").needs("vjo.dsf.document.Element","E").needs("vjo.dsf.document.Shim","S").needs("vjo.dsf.EventDispatcher","EV").needs("vjo.darwin.core.utils.ServiceUtils","SE").protos({constructs:function(_1){var t=this,J=t.J=t.vj$,B,M=t.m=_1,E=J.E,I=t.sId=M.CId;t.Open=false;t.sJsId=M.CJId;t.cW=M.cW;t.minTries=0;t.maxTries=10;t.isPromoted=false;t.promote2bodyFunc=function(){if(t.minTries>=t.maxTries){return;}
var _3=document.documentElement.doScroll;try{if(_3){_3("left");}
t.J.E.promoteToBody(t.sId);t.isPromoted=true;}
catch(err){t.minTries++;setTimeout(t.promote2bodyFunc,200);}};t.rgEH("load",t.promote2bodyFunc);t.ct=null;t.st=null;t.mst=-1;t.refE=null;t.ifm=null;B=J.B;t.IE=B.bIE&&B.iVer<7;M.sOn=!t.IE;t.fCall=false;var f1=function(m){t.open(m);},f2=function(m){t.close(m);};t.cusPos=null;J.SE.rgSv(M.OSN+I,f1);J.SE.rgSv(M.CSN+I,f2);J.SE.rgSvRsp(M.CSN+"ALL",f2);J.SE.rgSv("ALL"+M.CSN,f2);},getCtr:function(t){if(!t.ctr){t.ctr=t.J.E.get(t.sId+"_olp_pad");}
return t.ctr;},getOlp:function(t){if(!t.oOlp){t.oOlp=t.J.E.get(t.sId);}
return t.oOlp;},setMdl:function(_9,_a){var t=this,m=t.m;m.CD=(_9["2"])?+_9["2"]:0;m.OD=(_9["3"])?+_9["3"]:0;m.ScrlH=(_9["7"])?+_9["7"]:0;m.HOF=(_9["9"])?+_9["9"]:0;m.VOF=(_9["10"])?+_9["10"]:0;var td=t.J.E.get(t.sId+"_olp_cnt");if(td){td.parentNode.height=(_9["20"])?+_9["20"]:"auto";}
var _d=t.J.E.get(t.sId);if(_d){_d.firstChild.width=(_9["21"])?+_9["21"]:0;}
var _e=parseInt(_9["22"]),th=vjo.getType("vjo.darwin.core.overlaypanel3.OverlayPanel");m.scrbl=(th.i2B(_e,0));var ce=t.J.E.get(t.sId+"_cbtn");if(ce){ce.style.display=(th.i2B(_e,1))?"inline":"none";}
m.ICOMO=(th.i2B(_e,2));m.COB=(th.i2B(_e,3));m.DRG=(th.i2B(_e,5));m.SIC=(th.i2B(_e,6));m.HM=(th.i2B(_e,7));var _10=parseInt(_9["23"]);m.HTSC=(th.i2B(_10,3));if(th.i2B(_e,4)||m.DRG||th.i2B(_10,0)){t.dU=vjo.getType("vjo.darwin.core.utils.DragUtils");if(th.i2B(_e,4)){t.dU=new t.dU(t.sId+"_ttl",t.sId);}else{if(th.i2B(_10,0)){t.dU=new t.dU(t.sId+"tid",t.sId);}else{t.dU=new t.dU(t.sId,t.sId);}}
t.dU.regDrag();}else{if(t.dU){t.dU.detachEvts();}}
var _11=(t.J.E.get(_a+"cnh"));if(_11){var ob=new Object();ob.cnt=_11;t.setCon(ob);}
var ttl=(t.J.E.get(t.sId+"_ttl"));var ttn=(t.J.E.get(_a+"_ttl"));if(ttl.firstChild){t.J.E.get(t.m.CHI).appendChild(ttl.firstChild);}
if(ttn&&ttl){ttl.appendChild(ttn);}
m.HJId=m.HJId.replace("njs","");if(!th.i2B(_10,1)){m.HJId=m.HJId+"njs";t.J.E.get(t.sId+"arid").style.display="none";}},setCon:function(msg){var _16=msg.cnt;if(!_16){return;}
var t=this,cnt=t.getCtr(t),cntC=cnt.childNodes,l=cntC.length,isStr=(typeof _16=="string");if(isStr){cnt.innerHTML=_16;}else{for(var i=0;i<l;i++){var s=cntC[i];if(s.innerHTML!=undefined){if(s.getAttribute("ex")!=null){continue;}
s.parentNode.removeChild(s);}}
cnt.appendChild(_16);}
t.posPanel(t.oOlp,t.refE);},onResize:function(t){if(!t){t=this;}
if(t.Open&&(t.refE||t.SIC)){t.posPanel(t.oOlp,t.refE);}},regEvt:function(){var t=this,rF=function(){t.onResize(t);};if(!t.m.STK){t.rgEH("resize",rF);t.rgEH("scroll",rF);}
var f1=function(e){if(!t.Open){return;}
var el=e.nativeEvent.srcElement||e.nativeEvent.target;if(!t.m.COB){return;}
while(el){if(el.id==t.sId||(t.refE&&t.refE.id==el.id)){return;}
el=el.parentNode;}
t.close();};t.J.EV.add("body","click",f1,t);t.fCall=true;},rgEH:function(evt,hnd){this.J.EV.addEventListener(window,evt,hnd,window);},setPositionHandler:function(_21){this.cusPos=_21;},open:function(_22){var t=this,cnt=t.getCtr(t),rid,mm;if(!t.isPromoted){t.promote2bodyFunc();}
t.getOlp(t);if(_22){rid=_22.sAnchorId,cnt=t.ctr;t.refE=t.J.E.get(rid),mm=_22.model;t.setCon(_22);(mm)?t.setVals(mm):"";if(_22.dspModel){t.setMdl(_22.dspModel,_22.cmpId);}
if(mm){if(mm.posHandler){t.cusPos=mm.posHandler;}
if(mm.VOF){t.m.VOF=mm.VOF;}
if(mm.HOF){t.m.HOF=mm.HOF;}}
if(_22.ho!=null){t.m.ICOMO=_22.ho;}}
(!t.fCall)?t.regEvt():"";if(t.ctr){t.ctr.scrollTop=0;}
var f=function(){if(t.m.ICOMO){clearTimeout(t.ct);}
if(!t.ifm&&t.oOlp&&t.IE){t.ifm=t.J.S.add(t.oOlp);}
var _25=t.J.E.get(rid);if(_25||t.m.SIC){t.posPanel(t.oOlp,_25);}else{t.oOlp.style.visibility="hidden";return;}
t.oOlp.style.visibility="visible";t.Open=true;t.J.SE.sndM(t.m.POSN+t.sId);var _26=t.J.SE.gM(t.m.MCO);_26.containerId=t.sId;if(t.m.HM){t.J.SE.sndM(_26);}};clearTimeout(t.st);t.st=setTimeout(f,t.m.OD);},setVals:function(m){var t=this,md=t.m;(m[4])?(md.scrbl=(m[4]=="true")?true:false):"";(m[5])?md.scrlH=parseInt(m[5]):0;(m[1])?md.HA=m[1]:"";(m[7])?md.VA=m[7]:"";var ce=t.J.E.get(t.sId+"_cbtn");if(ce){ce.style.display=(m[9]=="true")?"inline":"none";}
md.ICOMO=(m[12]=="true")?true:false;md.HOF=(m[10]!=null)?parseInt(m[10]):0;md.VOF=(m[11]!=null)?parseInt(m[11]):0;},posPanel:function(olp,_2b){var u="px",t=this,m=t.m,h=t.cusPos?t.cusPos:(vjo.Registry.get(m.HJId));if(!h){return;}
var p=h.position(olp,_2b,m.HA,m.VA,m);if(t.ifm){var _2e=t.ifm.style;_2e.top="0"+u;_2e.left="0"+u;_2e.width=p[2]+u;_2e.height=p[3]+u;}},setMState:function(_2f){this.mst=_2f;},onMout:function(){var t=this;var f=function(){if(t.mst===0&&t.m.ICOMO){t.mst=-1;t.close();}};setTimeout(f,50);},onMin:function(){if(this.m.ICOMO){clearTimeout(this.ct);}},close:function(_32){var t=this,f=function(){if(!t.oOlp){return;}
var s=t.oOlp.style;s.visibility="hidden";s.left="-1000px";t.Open=false;var _35=(_32&&_32.isRetire);if(!_35){t.J.SE.sndM(t.m.PCSN+t.sId);if(t.m.HM){t.J.SE.sndM(t.m.MCI);}}};clearTimeout(t.ct);t.ct=setTimeout(f,(_32&&_32.ignDelay)?0:t.m.CD);}}).props({olpMsg:function(_36,_37,cnt,_39,ho,_3b){var o=this.vj$.SE,m=o.gM(_36);m.sAnchorId=_37;if(cnt){m.cnt=cnt;}
if(_39){m.model=_39;}
if(ho!=null){m.ho=ho;}
if(_3b!=null){m.ignDelay=_3b;}
o.sndM(m);},sendMessage:function(_3d,_3e){this.olpMsg(_3d,_3e.anchorId,_3e.cnt,_3e);},registerEvent:function(_3f,_40,_41,_42,_43){var t=this,handler=vjo.dsf.EventDispatcher,f1=function(){t.sendMessage(_40,_3f);};f2=function(){t.sendMessage(_41,_3f);};handler.add(_3f.anchorId,_42,f1,t);handler.add(_3f.anchorId,_43,f2,t);},i2B:function(num,idx){var val=num.toString(2),l=val.length-1,ds=(idx-l)<0?-(idx-l):(idx-l);if(idx>=val.length){return false;}
return(val.charAt(ds)==1);}}).endType();

vjo.ctype("vjo.darwin.core.utils.WindowDimension").props({D:undefined,getBrowserDimension:function(){var s=self,d=document.documentElement||document.body;if(s.innerHeight){return[s.innerWidth,s.innerHeight];}else{if(d){return[d.clientWidth,d.clientHeight];}}},getScrollXY:function(){var _2=0,scrOfY=0,scrOfH=0,scrOfW=0,d=document.documentElement||document.body;if(typeof(window.pageYOffset)=="number"){return[window.pageXOffset,window.pageYOffset,document.height,document.width];}else{if(d){return[d.scrollLeft,d.scrollTop,d.scrollHeight,d.scrollWidth];}}
return[_2,scrOfY,scrOfH,scrOfW];},getOffsetPosition:function(_3){var e=_3,l=0,t=0,z=0,tz;while(e){l+=e.offsetLeft;t+=e.offsetTop;if(e.style){tz=parseInt(e.style.zIndex,10);z=!isNaN(tz)&&tz>z?tz:z;}
e=e.offsetParent;}
return[l,t,z];}}).endType();

vjo.ctype("vjo.darwin.core.overlaypanel3.ZIndexUtil").props({getNewZIndex:function(_1,_2,_3,_4){if(!_3&&!_4){return 0;}
var z=((_1==false)?_2:_3);z=(_4&&_4>=z)?_4:z;return(z+10);}}).endType();

vjo.ctype("vjo.darwin.core.overlaypanel3.PositionElement").needs("vjo.darwin.core.overlaypanel3.ZIndexUtil","ZU").needs("vjo.darwin.core.utils.WindowDimension").protos({constructs:function(_1){this.cBT=_1;},W:vjo.darwin.core.utils.WindowDimension,position:function(_2,_3,_4,_5,_6){var t=this,tp="top",bt="bottom",rt="right",lt="left",u="px",oc=vjo.dsf.Element.get(_2.id+"_olp_pad"),ocS=oc.style,id=_6.overlayCompId,scrbl=_6.scrbl,scrlH=_6.scrlH,md=_6,z=0,olpS=_2.style;ocS.height=(scrlH>0)?(scrlH+u):"auto";ocS.overflowY=(scrlH>0)?"auto":"";ocS.overflowX="";olpS.left="-1000px";olpS.top="-1000px";var _8=25,P=t.P,hof=_6.HOF,vof=_6.VOF,iW=_2.offsetWidth,iH=_2.offsetHeight;var W=t.W,wD=W.getBrowserDimension(),aS=W.getScrollXY(),finalZ=0;var _a;var _b;if(_6.SIC){var _c=_6.CID,con=vjo.dsf.Element.get(_c),cW=_c?con.offsetWidth:wD[0],cH=_c?con.offsetHeight:wD[1],cd=_c?t.W.getOffsetPosition(con):aS;_a=(cW-iW)/2+cd[0];_b=(cH-iH)/2+cd[1];if(iH>(wD[1]+_8)){_b=aS[1]+_8;}
finalZ=cd[2]?cd[2]:finalZ;}else{if(_3){var rH=_3.offsetHeight,rW=_3.offsetWidth,oP=W.getOffsetPosition(_3),rX=oP[0],rY=oP[1],x=rX-aS[0],y=rY-aS[1],spB=wD[1]-y-rH-_8,spT=y-_8,ovh=0,vAl=bt,hAl;if(((iH+vof)<spB&&_5=="auto")||(_5==bt)){_b=(rY+rH+vof);}else{if(((iH+vof)<spT&&_5=="auto")||(_5==tp)){_b=rY-iH-vof;vAl=tp;}else{if(t.cBT){_b=aS[1];}else{if(spB>spT){_b=rY+rH+vof;ovh=spB;}else{ovh=spT;_b=(scrbl&&(ovh>55))?(rY-spT):(rY-iH-vof);vAl=tp;}
if(scrbl&&(ovh>55)){ocS.overflowY="auto";ocS.overflowX="hidden";ocS.height=(scrlH>0)?(scrlH+u):(ovh-_8+u);}}}}
iW=_2.offsetWidth;rW=_3.offsetWidth;var _e=W.getOffsetPosition(_3);rX=_e[0];if((((iW+_8+hof)<(wD[0]-x))&&_4=="auto")||(_4==lt)){_a=rX+hof+hof;hAl=lt;}else{_a=((rX+rW-hof)-iW+8);hAl=rt;}
finalZ=oP[2]?oP[2]:finalZ;}}
olpS.left=_a+u;olpS.top=_b+u;z=t.vj$.ZU.getNewZIndex(md.HM,md.zid,md.mzid,finalZ);olpS.zIndex=z;return[_a,_b,iW,iH,hAl,vAl,z];}}).endType();

vjo.ctype("vjo.darwin.core.utils.WindowUtils").props({eventTop:function(ev){var t=this,de=document.documentElement,db=document.body;if(ev.pageY!=null){return ev.pageY;}
if(de){return ev.clientY+Math.max(de.scrollTop,db.scrollTop);}else{return ev.clientY+db.scrollTop;}},eventLeft:function(ev){var t=this,de=document.documentElement,db=document.body;if(ev.pageX!=null){return ev.pageX;}
if(de){return ev.clientX+Math.max(de.scrollLeft,db.scrollLeft);}else{return ev.clientX+db.scrollLeft;}},offsetTop:function(e){var d=document.documentElement,c=(d&&d.clientTop)?d.clientTop:0;for(var o=0;(e!=null);e=e.offsetParent){o+=e.offsetTop;}
return o+c;}}).endType();

vjo.ctype("vjo.darwin.core.utils.DragUtils").needs("vjo.dsf.Element","E").needs("vjo.dsf.EventDispatcher","ED").needs("vjo.darwin.core.utils.WindowUtils","W").needs("vjo.darwin.core.utils.WindowDimension","WD").protos({constructs:function(_1,_2){var t=this;t.stDrag=false;t.xPix="";t.yPix="";t.xOs="";t.yOs="";t.p="";t.regId=_1;t.cId=_2;},regDrag:function(){var t=this;var f3=function(e){var _7=e.srcElement||e.target;var _8=["SELECT","INPUT","TEXTAREA"];if(_8.toString().indexOf(_7.tagName.toUpperCase())>=0){return;}
t.startDrag(e,t);};var f4=function(e){t.drag(e,t);};var f5=function(e){t.stopDrag(t);};var f6=function(e){t.vj$.E.get(t.regId).style.cursor="move";};t.aE("down",f3,t);t.aE("move",f4,t);t.aE("up",f5,t);t.aE("over",f6,t);},detachEvts:function(){var t=this;var D=t.vj$.ED;D.detachHandlers(t.regId,"mousedown");D.detachHandlers(t.regId,"mousemove");D.detachHandlers(t.regId,"mouseup");D.detachHandlers(t.regId,"mouseover");t.vj$.E.get(t.regId).style.cursor="";},aE:function(_11,f,t){var J=t.vj$,bId=document.getElementsByTagName("body")[0];if(_11==="up"||_11==="move"){J.ED.addEventListener(bId,"mouse"+_11,f,t);}
J.ED.addEventListener(t.regId,"mouse"+_11,f,t);},stopDrag:function(t){t.stDrag=false;},startDrag:function(_16,t){var J=t.vj$;t.stDrag=true;t.xPx=J.W.eventLeft(_16);t.yPx=J.W.eventTop(_16);t.p=J.WD.getOffsetPosition(J.E.get(t.cId));},drag:function(_19,t){var J=t.vj$;if(typeof _19!="undefined"){if(t.cId!==null&&t.stDrag===true){t.xOs=J.W.eventLeft(_19);t.yOs=J.W.eventTop(_19);var oS=(J.E.get(t.cId)).style,lt=t.xOs-t.xPx+t.p[0],tp=t.yOs-t.yPx+t.p[1];if(lt<0){lt=0;}
if(tp<0){tp=0;}
oS.left=lt+"px";oS.top=tp+"px";}}}}).endType();

vjo.ctype("vjo.darwin.core.overlaypanel3.OverlayPanelModel").needs("vjo.darwin.core.utils.ServiceUtils","SE").protos({constructs:function(_1,_2){t=this;t.anchorId=_1;t.cnt=_2;t.scroll=null;t.overlayH=null;t.scrollH=null;t.HAlign=null;t.ODelay=null;t.overlayW=null;t.HOF=0;t.VOF=0;}}).endType();

vjo.ctype("vjo.darwin.core.ebayheader.rebate.Rebate").props({Refresh:function(_1,_2,_3){var D=vjo.dsf.document;var E=D.Element;var _6=E.get(_1);if(_6){if(this.IsShowMagicRebate(_6,_2)){E.toggleHideShow(_1,true);}else{if(this.IsShowVibrantCoupon()){var _7=_6.getElementsByTagName("span")[0];if(_7){_7.innerHTML=_3;E.toggleHideShow(_1,true);}}}}},IsShowVibrantCoupon:function(){var _8=vjo.dsf.cookie.VjCookieJar.readCookie("ebay","sbf");if(_8){return vjo.dsf.cookie.VjCookieJar.getBitFlag(_8,29)==1;}},IsShowMagicRebate:function(_9,_a){var _b=_9.getElementsByTagName("img")[0],rate=vjo.dsf.cookie.VjCookieJar.readCookie("npii","mri"),perc;if(this.IsExpired()){return false;}
if(!rate){return false;}
perc=this.GetRate(rate);if(perc){perc=this.IsValidRate(perc);}
if(perc!==null&&_b){_b.src=_b.src.replaceToken(_b.src,_a,perc);return true;}else{return false;}},IsExpired:function(){var _c=new Date().getTime();var _d=this.getClientOffset(_c);var _e=vjo.dsf.cookie.VjCookieJar.readCookieObj("npii","mri");if(_e!==null){var _f=parseInt(_e.maxage,16)*1000;if(_f>0){var _10=_f-_c+_d;if(_10<0){return true;}}}
return false;},getClientOffset:function(_11){var _12;var _13=vjo.dsf.cookie.VjCookieJar.readCookie("ebay","cos");if(_13!==null&&_13.length>0){_12=parseInt(_13,16)*1000;}else{_12=3600000;}
return _12;},IsValidRate:function(_14){var R=parseInt(_14,10);return(R>0&&R<100)?R:null;},GetRate:function(cv){var ar=cv.split(":");return ar.length>3?ar[2]:null;}}).endType();

vjo.ctype("vjo.darwin.core.dynamicmenu.ReplaceHandler").props({replace:function(_1,_2,_3){var dm=vjo.Registry.get(_1);if(null!=dm){dm.replaceJSONDataHandler(_2,_3);}}}).endType();

vjo.ctype("vjo.darwin.tracking.rover.Rover").needs("vjo.dsf.cookie.VjCookieJar").props({roverTrack:function(){var _1=new Date().getTime();var _2=vjo.darwin.tracking.rover.Rover.getClientOffset(_1);var _3=vjo.dsf.cookie.VjCookieJar.readCookieObj("npii","tpim");if(_3==null||_3.value==""){return;}
var _4=parseInt(_3.maxage,16)*1000;if(_4>0){var _5=_4-_1+_2;var _6=15552000000;if((_5>_6||_5<0)&&typeof(RoverSyncDropped)=="undefined"&&typeof(RoverNsCapable)=="undefined"){vjo.darwin.tracking.rover.Rover.dropRoverSyncImage();}}},dropRoverSyncImage:function(){if(typeof(RoverDomainBaseUrl)!=="undefined"&&RoverDomainBaseUrl.length>0){var im=document.createElement("img");im.width="1";im.height="1";im.src=RoverDomainBaseUrl+"/roversync/?rtpim=1&mpt="+new Date().getTime();document.body.appendChild(im);}},getClientOffset:function(_8){var _9;var _a=vjo.dsf.cookie.VjCookieJar.readCookie("ebay","cos");if(_a!==null&&_a.length>0){_9=parseInt(_a,16)*1000;}else{if(typeof(svrGMT)!=="undefined"){_9=_8-svrGMT;var _b=Math.round(_9/1000);if(!isNaN(_b)){vjo.dsf.cookie.VjCookieJar.writeCookielet("ebay","cos",_b.toString(16));}}}
if(isNaN(_9)){_9=1800000;}
return _9;}}).endType();

vjo.ctype("vjo.darwin.core.datatable.DataTable").needs("vjo.dsf.document.Element").protos({constructs:function(_1,_2,_3){}}).props({hghLightSortColumn:function(_4,_5,_6,_7,_8,_9,_a,_b,_c){var _d=_4+"_srtHCol_"+_b;var hg=vjo.dsf.Element.get(_d);var _f;if(_6=="mouseOver"){if(_7){if(_8=="down"){hg.className=_5+" "+_9[2];}else{if(_8=="up"){hg.className=_5+" "+_9[4];}else{hg.className=_5+" "+_9[0];}}}else{hg.className=_5+" "+" "+_9[0];}}else{if(_6=="mouseOut"){if(_7){if(_8=="up"){hg.className=_5+" "+_9[3];}else{if(_8=="down"){hg.className=_5+" "+_9[1];}}}else{hg.className=_5;}}}}}).endType();

vjo.ctype("vjo.darwin.core.datatable.DataTableJsEventBinder").needs("vjo.dsf.document.Element").needs("vjo.darwin.core.datatable.DataTable").props({bindEventsInJS:function(_1,_2,_3,_4,_5,_6,_7,_8){var _9=_1+"_srtHCol_"+_7;var _a=vjo.dsf.document.Element.get(_9);vjo.dsf.EventDispatcher.addEventListener(_a,"mouseover",function(_b){return vjo.darwin.core.datatable.DataTable.hghLightSortColumn(_1,_2,"mouseOver",_3,_4,_5,_6,_7,_8);});vjo.dsf.EventDispatcher.addEventListener(_a,"mouseout",function(_c){return vjo.darwin.core.datatable.DataTable.hghLightSortColumn(_1,_2,"mouseOut",_3,_4,_5,_6,_7,_8);});}}).endType();

vjo.ctype("vjo.darwin.core.pageleveltab.TabProperties").needs("vjo.Registry").needs("vjo.darwin.core.utils.ServiceUtils","SU").protos({_elm:null,_rg:null,mName:null,constructs:function(_1){_elm=vjo.dsf.Element;_rg=vjo.Registry;m=_1;var t=this;t.hN=m.htmlName;mName=m.htmlName;t.iId=m.id;t.sUrl=m.url;t.bIsActive=m.active;t.bIsHighlighted=m.highlighted;t.bIsAjaxEnabled=m.ajaxEnabled;t.clkSvcId=m.clkSvcId;t.dsbl=m.dsbl;var _n=_rg.get(t.hN);if(_n!==null){var _4=_n.aTabs;_4[_4.length]=t;}
t.bindEventsInJS(_n.hon);t.attachEventListner(t.hN,t.clkSvcId);},bindEventsInJS:function(_5){var t=this,ED=vjo.dsf.EventDispatcher,_tProp=vjo.darwin.core.pageleveltab.TabProperties;var _7=t.hN+"_tab_rgt_"+t.iId,tbL=t.hN+"_tab_lft_"+t.iId,tb=t.hN+"_"+t.iId;if(t.sUrl===null){t.sUrl="";}
var _f=function(_9){return _tProp.switchToTab(t.hN,t.iId,_9,t.clkSvcId);},ar=[_7,tbL,"mouseover","mouseout"],j=0;ED.add(tbL,"click",_f);ED.add(_7,"click",_f);if(_5){for(j=0;j<2;j++){ED.add(ar[j],ar[2],function(_a){return _tProp.hoverChange(t.hN,t.iId,true);});ED.add(ar[j],ar[3],function(_b){return _tProp.hoverChange(t.hN,t.iId,false);});}}},attachEventListner:function(_c,_d){var t=vjo.darwin.core.pageleveltab.TabProperties,fn=function(){t.initializer(_c,_d);};vjo.dsf.EventDispatcher.addEventListener(window,"load",fn);}}).props({tLft:"_tab_lft_",tRgt:"_tab_rgt_",tCnt:"_content_",tCurrId:"CurrId",switchToTab:function(_f,_10,_11,_12){var t=this,pTab=_rg.get(_f),pArrTab=pTab.aTabs,hid=_elm.get(_f+t.tCurrId),currId=(hid)?hid.value:1,from,to;for(i=0;i<pArrTab.length;i++){if(currId==pArrTab[i].iId){from=i;}
if(_10==pArrTab[i].iId){to=i;}}
var tp=vjo.darwin.core.pageleveltab.TabProperties,fTbR=_f+t.tRgt+pArrTab[from].iId,fTbL=_f+t.tLft+pArrTab[from].iId,tTbR=_f+t.tRgt+pArrTab[to].iId,tTbL=_f+t.tLft+pArrTab[to].iId;if(pArrTab){if(pArrTab[to].bIsActive===true||pArrTab[to].dsbl==true){return;}
if(pArrTab[to].iId==_10){var rt=_elm.get(tTbR),url=pArrTab[to].sUrl;if(_11&&(!pArrTab[to].bIsAjaxEnabled)&&(url&&url.length>1&&url.toLowerCase().indexOf("javascript")<0)){var src=_11.nativeEvent.target?_11.nativeEvent.target:_11.nativeEvent.srcElement;if(src.tagName.toLowerCase()==="a"){src.blur();return;}
document.location.href=pArrTab[to].sUrl;return;}
if(hid){hid.value=_10;}
tp.setTabs(_10,pArrTab[from],_f,pTab,_12);tp.setTabs(_10,pArrTab[to],_f,pTab,_12);}}},sendMsg:function(_17,tb,rTb,lTb,url){var t=this,SU=t.vj$.SU,msg=SU.gM(_17);msg.tb=tb;msg.rTb=rTb;msg.lTb=lTb;msg.tbUrl=url;SU.sndM(msg);},setClz:function(_1d,_1e){var el;el=_elm.get(_1d);if(el){el.className=_1e;}},setBkgColor:function(_20,_21){var el;el=_elm.get(_20);if(el){el.style.backgroundColor=_21;}},initializer:function(_23,_24){var t=this,to,pTab=_rg.get(_23),pArrTab=pTab.aTabs,tp=vjo.darwin.core.pageleveltab.TabProperties,hid=_elm.get(_23+tp.tCurrId),currId=(hid)?hid.value:1;for(i=0;i<pArrTab.length;i++){if(currId==pArrTab[i].iId){to=i;break;}}
if(pArrTab){if(pArrTab[to].bIsActive===true||pArrTab[to].dsbl==true){return;}
for(var i=0;i<pArrTab.length;i++){tp.setTabs(currId,pArrTab[i],_23,pTab,_24);}}},setTabs:function(_27,_28,_29,_2a,_2b){var t=vjo.darwin.core.pageleveltab.TabProperties;var _2d=_29+t.tRgt+_28.iId,fTbL=_29+t.tLft+_28.iId;if(_27==_28.iId){_28.bIsActive=true;if(_2b){t.sendMsg(_2b,_27,_2d,fTbL);}
if(_2a.sActiveBkgColor!==null){t.setBkgColor(fTbL,_2a.sActiveBkgColor);t.setBkgColor(_2d,_2a.sActiveBkgColor);}
t.setClz(fTbL,_2a.sactiveLft);t.setClz(_2d,_2a.sactiveRgt);t.setClz(_29+t.tCnt+_28.iId,_2a.stabcontentOuterOn);}else{_28.bIsActive=false;t.setBkgColor(fTbL,"");t.setBkgColor(_2d,"");if(_28.bIsHighlighted===true){t.setClz(fTbL,_2a.shighlightLft);t.setClz(_2d,_2a.shighlightRgt);}else{t.setClz(fTbL,_2a.sinactiveLft);t.setClz(_2d,_2a.sinactiveRgt);}
t.setClz(_29+t.tCnt+_28.iId,_2a.stabcontentOuterOff);}},hoverChange:function(_2e,_2f,_30){var _31=_rg.get(_2e),tp=vjo.darwin.core.pageleveltab.TabProperties,pArrTab=_31.aTabs;for(var i=0;i<pArrTab.length;i++){if(pArrTab[i].iId==_2f){to=i;break;}}
var _33=_2e+tp.tRgt+pArrTab[to].iId,tTbL=_2e+tp.tLft+pArrTab[to].iId;if(pArrTab){if(pArrTab[to].bIsActive===true||pArrTab[to].dsbl==true){return;}
if(pArrTab[to].bIsHighlighted===true){if(_30===true){tp.setClz(tTbL,_31.shighlightLft+" "+_31.shoverLft);tp.setClz(_33,_31.shighlightRgt+" "+_31.shoverRgt);}else{tp.setClz(tTbL,_31.shighlightLft);tp.setClz(_33,_31.shighlightRgt);}}else{if(_30===true){tp.setClz(tTbL,_31.sinactiveLft+" "+_31.shoverLft);tp.setClz(_33,_31.sinactiveRgt+" "+_31.shoverRgt);}else{tp.setClz(tTbL,_31.sinactiveLft);tp.setClz(_33,_31.sinactiveRgt);}}}}}).endType();

vjo.ctype("vjo.darwin.core.pageleveltab.PageLevelTab").needs("vjo.Registry").protos({constructs:function(_1){var t=_1,o=this;o.sHtmlName=t.htmlName;o.sactiveRgt=t.activeRgt;o.sactiveLft=t.activeLft;o.shighlightLft=t.highlightLft;o.shighlightRgt=t.highlightRgt;o.sinactiveLft=t.inactiveLft;o.sinactiveRgt=t.inactiveRgt;o.shoverLft=t.hoverLft;o.shoverRgt=t.hoverRgt;o.stabcontentOuterOff=t.tabcontentOuterOff;o.stabcontentOuterOn=t.tabcontentOuterOn;o.sContentElementId=null;o.sActiveBkgColor=t.activeBkgColor;o.aTabs=[];o.iCurrTab=t.activeTabId;o.hon=t.hoverOn;}}).endType();

vjo.needs("vjo.dsf.typeextensions.string.Comparison");vjo.ctype("vjo.dsf.typeextensions.string.Trim").endType();String.prototype.trim=function(){var s=this;while(s.substring(0,1).isAny(" ","\n","\r")){s=s.substring(1,s.length);}
while(s.substring(s.length-1,s.length).isAny(" ","\n","\r")){s=s.substring(0,s.length-1);}
return s;};

vjo.ctype("vjo.darwin.core.ebayheader.searchbox.SearchBox").needs(["vjo.dsf.client.Browser","vjo.dsf.typeextensions.string.Trim"]).props({Focus:function(_1){var _2=vjo.dsf.document.Element.get(_1),B=vjo.dsf.client.Browser.bIE;if(typeof(_2)!="undefined"&&_2){if(B.bIE&&B.iVer==6){setTimeout(function(){_2.focus();},0);}else{_2.focus();}}},IeOptionDisabler:function(_3){if(vjo.dsf.client.Browser.bIE){var sl=vjo.dsf.document.Element.get(_3),idx;if(sl){sl.onchange=function(){idx=this.selectedIndex=(this.options[this.selectedIndex].disabled)?idx:this.selectedIndex;};sl.onfocus=function(){idx=this.selectedIndex;};this.greydisabledoption(sl);}}},greydisabledoption:function(e){var i,op;for(i=0;i<e.options.length;i++){op=e.options[i];if(op.disabled){op.style.color="graytext";}}}}).endType();

vjo.needs("vjo.dsf.typeextensions.string.Comparison");vjo.ctype("vjo.dsf.typeextensions.string.TokenReplacement").endType();String.prototype.replaceToken=function(_1,_2,_3){var rv=_1;while(rv.has(_2)){rv=rv.replace(_2,_3);}
return rv;};String.prototype.replaceTokensEx=function(_5){var rv=this,re,tkn,a=arguments,l=a.length;for(var i=1;i<l+1;i++){rv=this.replaceToken(rv,_5.replace("n",(i)),a[i]);}
return rv;};String.prototype.replaceTokens=function(){var rv=this,re,tkn,a=arguments,l=a.length;for(var i=0;i<l;i++){rv=this.replaceToken(rv,"<#"+(i+1)+"#>",a[i]);}
return rv;};

vjo.ctype("vjo.darwin.core.throbber.Throbber").needs("vjo.dsf.document.Element").needs("vjo.dsf.utils.Timer").protos({constructs:function(_1){var t=this,jsM=_1;t.sThrobberId=jsM.throbberId;t.sTimeOutId=jsM.timeOutId;t.E=vjo.dsf.document.Element;t.oThrobber="undefined";t.oTimeOut="undefined";t.bStarted=jsM.started||false;t.iTimeOut=jsM.timeOut?(jsM.timeOut*1000):jsM.timeOut;t.sTimeOutText=jsM.timeOutTextMessage;t.oTimer="undefined";t.bOpening=false;},start:function(_3){var t=this;if(t.oThrobber=="undefined"){t.oThrobber=t.E.get(t.sThrobberId);}
if(t.oThrobber&&!t.bStarted){t.oThrobber.style.display="block";t.bStarted=true;}
if(t.oTimeOut=="undefined"){t.oTimeOut=t.E.get(t.sTimeOutId);}
if(t.iTimeOut>0&&t.oTimeOut){t.oTimeOut.style.display="none";}
if(t.iTimeOut>0&&t.sTimeOutText){if(t.oTimer=="undefined"){t.oTimer=new vjo.dsf.utils.Timer(t.iTimeOut);t.oTimer.onTick=function(){t.timeOut();t.oTimer.stop();};}
t.oTimer.start();}},timeOut:function(_5){var t=this;t.stop();if(t.oTimeOut=="undefined"){t.oTimeOut=t.E.get(t.sTimeOutId);}
t.oTimeOut.style.display="block";},stop:function(_7){var t=this;if(t.oThrobber=="undefined"){t.oThrobber=t.E.get(t.sThrobberId);}
if(t.oThrobber&&t.bStarted){t.oThrobber.style.display="none";if(t.oTimer!="undefined"){t.oTimer.stop();}
t.bStarted=false;}}}).endType();

vjo.ctype("vjo.darwin.core.ebaytoolbar.VjEbayToolbarDetect").needs("vjo.dsf.client.ActiveX").props({isEnabled:function(){var V1="eBayToolbar.Helper",V2="eBayToolbarCommLib.IWebEvent.1";with(this){var _2=vj$.ActiveX;return(_2.isLibLoaded(V1)||_2.isLibLoaded(V2));}}}).endType();

vjo.ctype("vjo.darwin.core.bta.BuyerTransactionAlert").needs(["vjo.dsf.cookie.VjCookieJar","vjo.dsf.typeextensions.string.Comparison","vjo.dsf.client.Browser","vjo.dsf.document.Element","vjo.dsf.utils.Object","vjo.darwin.core.ebaytoolbar.VjEbayToolbarDetect"]).protos({constructs:function(_1,_2,_3,_4,_5,_6,_7,_8,_9,_a,_b){this.sId=_1;this.iPollingInterval=_2;this.iMaxHits=_3;this.iHitTimeout=_4;this.iServerHits=0;this.sLastCookieletValue="";this.sServerUrl=_5;if(document.location.protocol.has("https")){_6=_6.replace("http://pics.","https://securepics.");}
this.sImgServer=_6;this.sViewItemUrl=_7;this.aAlertInfo=[["h:h:alt:2",_9,"icon/iconOutbid_16x16.gif"],["h:h:alt:3",_8,"icon/iconWatchB_16x16.gif"],["h:h:alt:4",_9,"icon/iconOutbid_16x16.gif"],["h:h:alt:5",_a,"icon/iconchanceBlu_16x16.gif"],["h:h:alt:tcr",_b,"icon/iconMailBlue_16x16.gif"]];var c,oC=vjo.dsf.client.Browser,oCJ=vjo.dsf.cookie.VjCookieJar;if((oC.bNav&&oC.iVer<7)||(oC.bOpera&&(oC.iVer+oC.fMinorVer)<0.5)||(oC.bIE&&oC.iVer<5)){return;}
c=oCJ.readCookie("ebaysignin");if(!c||!c.is("in")){return;}
c=oCJ.readCookie("dp1","a1p");if(c&&c.length>0&&parseInt(c)>0){return;}
if(vjo.darwin.core.ebaytoolbar.VjEbayToolbarDetect.isEnabled()){return;}},setValue:function(_d,_e){var oL=this.oL;if(oL){if(_d.is("")&&!oL.ctrld){return;}
if(_e){if(vjo.dsf.client.Browser.bFirefox){oL.textContent=_d;}else{oL.innerText=_d;}}else{oL.innerHTML=_d;}
oL.ctrld=1;}},onRefresh:function(){var E=vjo.dsf.document.Element;if(!this.oL){this.oL=E.get(this.sId);}
if(!this.oL){return;}
var c=vjo.dsf.cookie.VjCookieJar.readCookie("npii","mri");if(c){return;}
c=vjo.dsf.cookie.VjCookieJar.readCookie("ebay","a2p");if(!c){this.onCookieExpire();return;}
var at=parseInt(c.charAt(8));if(isNaN(at)){return;}
if(at===0){this.setValue("");return;}
var nrt=parseInt(c.substring(0,8),16)*1000;if(isNaN(nrt)){return;}
var ct=new Date();ct=ct.getTime();if(at==6||at==9){if(!c.is(this.sLastCookieletValue)){this.iServerHits=0;}
this.setValue("");this.sLastCookieletValue=c;var t=(nrt>ct)?parseInt((nrt-ct)/1000):this.iPollingInterval;window.setTimeout(vjo.dsf.utils.Object.hitch(this,this.onCookieExpire),t*1000);return;}
if(ct>=nrt){this.onCookieExpire();return;}
this.iServerHits=0;var cfg=this.aAlertInfo;if(at<0&&at>=cfg.length){return;}
var ii=c.substring(9,c.lastIndexOf("."));if(!c.is(this.sLastCookieletValue)){var _18=cfg[at-1],imgSrv=this.sImgServer;var _19=imgSrv+"s.gif";var _1a="<img src=\""+_19+"\" width=\"10\" height=\"16\" style=\"vertical-align:middle\">|<img src=\""+_19+"\" width=\"10\" height=\"16\" style=\"vertical-align:middle\">";_1a+="<img src=\""+imgSrv+_18[2]+"?t\" style=\"vertical-align:middle\"><img src=\""+_19+"\" width=\"5\" height=\"16\" style=\"vertical-align:middle\">";var url=this.sViewItemUrl+"&item="+ii;_1a+="<a href=\""+url+"&ssPageName="+_18[0]+"\">"+_18[1]+"</a>";this.setValue(_1a);this.sLastCookieletValue=c;}
this.fireRefreshEvent();},fireRefreshEvent:function(_1c){if(!_1c){_1c=this.iPollingInterval;}
window.setTimeout(vjo.dsf.utils.Object.hitch(this,this.onRefresh),_1c*1000);},onCookieExpire:function(){var oCJ=vjo.dsf.cookie.VjCookieJar,signin=oCJ.readCookie("ebaysignin");if(!signin.has("in")){return;}
if(document.location.href.has("https:")){return;}
if(this.iServerHits<this.iMaxHits){this.iServerHits++;var ct=new Date();ct=ct.getTime();this.setValue("<img height=\"1\" width=\"1\" src=\""+this.sServerUrl+"&clientTime="+ct+"\" style=\"visibility:hidden;vertical-align:middle\">");this.fireRefreshEvent(this.iHitTimeout);}else{this.setValue("");oCJ.writeCookielet("ebay","a2p","1111111101111111111.");}}}).endType();

vjo.ctype("vjo.darwin.core.greetings.Sandbox").needs(["vjo.dsf.document.Element","vjo.dsf.EventDispatcher"]).props({init:function(){var l=vjo.dsf.document.Element.get("registerLink"),s="https://scgi.sandbox.ebay.com/ws/eBayISAPI.dll?RegisterEnterInfo",t="https://developer.ebay.com/DevZone/sandboxuser/Default.aspx";if(l){if(l.href==s){l.href=t;}}},adBanner:function(){var l=window.location.href,s="sandbox.",c;if(l.indexOf(s)!=-1){c="<sty"+"le type=\"text/css\">#gnheader {background: url(http://pics.ebaystatic.com/aw/pics/devprogram/watermark.gif)}</sty"+"le>";document.write(c);}}}).inits(function(){vjo.darwin.core.greetings.Sandbox.adBanner();vjo.dsf.EventDispatcher.addEventListener(window,"load",function(){vjo.darwin.core.greetings.Sandbox.init();});}).endType();

vjo.ctype("vjo.darwin.core.greetings.VjGreetingsServer").needs(["vjo.dsf.utils.URL","vjo.darwin.core.greetings.Sandbox"]).props({handleClick:function(_1,_2){var _3=_2.srcElement||_2.target;if(_3){_3.href=vjo.dsf.utils.URL.addArg(_1,"ru",encodeURIComponent(document.location.href));}}}).endType();

vjo.ctype("vjo.darwin.tracking.sojourner.SojData").singleton().protos({constructs:function(){this.sojD;}}).inits(function(){vjo.darwin.tracking.sojourner.SojData=new vjo.darwin.tracking.sojourner.SojData();}).endType();

vjo.ctype("vjo.darwin.tracking.sojourner.CalData").singleton().protos({constructs:function(){this.cal={};},setData:function(_1,_2){if(!_1||!_2){return;}
this.cal[_1]=_2;},getData:function(_3){if(_3){return this.cal[_3];}}}).inits(function(){vjo.darwin.tracking.sojourner.CalData=new vjo.darwin.tracking.sojourner.CalData();}).endType();

vjo.ctype("vjo.darwin.tracking.sojourner.TrackingRespHdl").needs(["vjo.darwin.tracking.sojourner.SojData","vjo.darwin.tracking.sojourner.CalData"]).props({handleResponse:function(_1){if(_1.errors&&_1.errors.length>0){vjo.darwin.tracking.sojourner.SojData.sojD="";}
if(_1.response&&_1.response.dataMap&&_1.response.dataMap.SOJDATA){vjo.darwin.tracking.sojourner.SojData.sojD=_1.response.dataMap.SOJDATA;}
if(_1.response&&_1.response.dataMap&&_1.response.dataMap.TDATA){vjo.darwin.tracking.sojourner.CalData.setData(_1.svcId,_1.response.dataMap.TDATA);}}}).endType();

vjo.ctype("vjo.dsf.window.utils.VjWindowUtils").props({getBrowserWindowHeight:function(){var s=self;var d=document;var de=d.documentElement;if(s.innerHeight){return s.innerHeight;}else{if(de&&de.clientHeight){return de.clientHeight;}}
return d.body.clientHeight;},getBrowserWindowWidth:function(){var s=self;var d=document;var de=d.documentElement;if(s.innerWidth){return s.innerWidth;}else{if(de&&de.clientWidth){return de.clientWidth;}}
return d.body.clientWidth;},getScrollXY:function(){var _7=0,scrOfY=0;if(typeof(window.pageYOffset)=="number"){scrOfY=window.pageYOffset;_7=window.pageXOffset;}else{if(document.body&&(document.body.scrollLeft||document.body.scrollTop)){scrOfY=document.body.scrollTop;_7=document.body.scrollLeft;}else{if(document.documentElement&&(document.documentElement.scrollLeft||document.documentElement.scrollTop)){scrOfY=document.documentElement.scrollTop;_7=document.documentElement.scrollLeft;}}}
return[_7,scrOfY];},toPixels:function(_8){return _8+"px";},scrollTop:function(){if(window.pageYOffset!=null){return window.pageYOffset;}
if(document.documentElement){return Math.max(document.documentElement.scrollTop,document.body.scrollTop);}else{return document.body.scrollTop;}},scrollLeft:function(){if(window.pageXOffset!=null){return window.pageXOffset;}
if(document.documentElement){return Math.max(document.documentElement.scrollLeft,document.body.scrollLeft);}else{return document.body.scrollLeft;}},scrollWidth:function(){if(document.documentElement){return document.documentElement.scrollWidth;}else{return Math.max(document.body.scrollWidth,document.body.offsetWidth);}},scrollHeight:function(){if(document.documentElement){return document.documentElement.scrollHeight;}else{return Math.max(document.body.scrollHeight,document.body.offsetHeight);}},clientTop:function(){if(document.documentElement){return document.documentElement.clientTop;}else{return document.body.clientTop;}},clientLeft:function(){if(document.documentElement){return document.documentElement.clientLeft;}else{return document.body.clientLeft;}},clientWidth:function(){var _9=document.documentElement;if(_9&&window.innerWidth){return Math.min(_9.clientWidth,window.innerWidth);}else{if(_9&&_9.clientWidth){return _9.clientWidth;}else{if(window.innerWidth){return window.innerWidth;}else{if(document.body.clientWidth){return document.body.clientWidth;}else{return document.body.offsetWidth;}}}}},clientHeight:function(){var _a=document.documentElement;if(_a&&window.innerHeight){return Math.min(_a.clientHeight,window.innerHeight);}else{if(_a&&_a.clientHeight){return _a.clientHeight;}else{if(window.innerHeight){return window.innerHeight;}else{if(document.body.clientHeight){return document.body.clientHeight;}else{return document.body.offsetHeight;}}}}},browserTop:function(){return(window.innerHeight)?window.screenY+(window.outerHeight-window.innerHeight):window.screenTop;},browserLeft:function(){return(window.innerWidth)?window.screenX+(window.outerWidth-window.innerWidth):window.screenLeft;},eventTop:function(_b){if(_b.pageY!=null){return _b.pageY;}
if(document.documentElement){return _b.clientY+Math.max(document.documentElement.scrollTop,document.body.scrollTop);}else{return _b.clientY+document.body.scrollTop;}},eventLeft:function(_c){if(_c.pageX!=null){return _c.pageX;}
if(document.documentElement){return _c.clientX+Math.max(document.documentElement.scrollLeft,document.body.scrollLeft);}else{return _c.clientX+document.body.scrollLeft;}},offsetTop:function(_d){var _e=(document.documentElement&&document.documentElement.clientTop)?document.documentElement.clientTop:0;for(var _f=0;(_d!=null);_d=_d.offsetParent){_f+=_d.offsetTop;}
return _f+_e;},offsetLeft:function(_10){var _11=(document.documentElement&&document.documentElement.clientLeft)?document.documentElement.clientLeft:0;for(var _12=0;(_10!=null);_10=_10.offsetParent){_12+=_10.offsetLeft;}
return _12+_11;},openWindow:function(url,_14,_15){var _16=new Array();var _17=vjo.dsf.window.utils.VjWindowUtils;_15.top=_17.browserTop()+Math.round((_17.clientHeight()-_15.height)/2)+25;_15.left=_17.browserLeft()+Math.round((_17.clientWidth()-_15.width)/2);for(var key in _15){_16.push(key.concat("=",_15[key]));}
return window.open(url,_14,_16.join(","),true);}}).endType();

vjo.ctype("vjo.darwin.core.ebayheader.autocomplete.AutoComplete").needs(["vjo.dsf.Message","vjo.dsf.Element","vjo.dsf.EventDispatcher","vjo.dsf.utils.Handlers","vjo.dsf.ServiceEngine","vjo.dsf.window.utils.VjWindowUtils"]).protos({H:vjo.dsf.utils.Handlers,constructs:function(_1,_2){var t=this,E=vjo.dsf.Element,ED=vjo.dsf.EventDispatcher;t.sFormId=_2;t.sAcDivId=_1+"_acdiv";t.acdivWidth=0;t.sShowImg=false;t.activated=true;t.H.attachSvcReqt("activateAutoComplete",t.activateAutoComplete,t);t.input=E.get(_1);t.input.setAttribute("AUTOCOMPLETE","OFF");t.acdiv=E.get(t.sAcDivId);var _4=document.forms;ED.add(_1,"keyup",function(e){if(!t.activated){return;}
var _6=e.nativeEvent.keyCode;var _7=new vjo.dsf.Message("SVC_GH_OUT");_7.clientContext={type:"kw_keyup",src:e.src,value:t.input.value,keyCode:_6};return _7;});ED.add(_1,"keydown",function(e){if(!t.activated){return;}
var _9=e.nativeEvent.keyCode;var _a=new vjo.dsf.Message("SVC_GH_OUT");_a.clientContext={type:"kw_keydown",src:e.src,value:t.input.value,keyCode:_9};return _a;});ED.add(_1,"mouseover",function(e){if(!t.activated){return;}
var _c=e.nativeEvent.keyCode;var _d=new vjo.dsf.Message("SVC_GH_OUT");_d.clientContext={type:"kw_mouseover",src:e.src};return _d;});ED.add(_1,"blur",function(e){if(!t.activated){return;}
var _f=e.nativeEvent.keyCode;var _10=new vjo.dsf.Message("SVC_GH_OUT");_10.clientContext={type:"kw_blur",src:e.src};return _10;});ED.add(this.sAcDivId,"click",function(e){if(!t.activated){return;}
var _12=new vjo.dsf.Message("SVC_GH_OUT");_12.clientContext={type:"show_click"};return _12;});vjo.dsf.ServiceEngine.registerSvcHdl("SVC_GH_IN",function(_13){if(!t.activated){return;}
var _14=_13.clientContext.type;if(_14=="kw_updvalue"){t.input.value=_13.clientContext.value;}else{if(_14=="kw_autocomplete"){if(_13.clientContext.bOn&&!_13.clientContext.bSkipFocus){t.input.blur();}
t.input.setAttribute("AUTOCOMPLETE",_13.clientContext.bOn?"ON":"OFF");if(_13.clientContext.bOn&&!_13.clientContext.bSkipFocus){t.input.blur();t.input.focus();}}else{if(_14=="search_updtrk"){for(i=0;i<_4.length;i++){if(_4[i].name==_2){for(var j=0;j<_4[i].length;j++){if(_4[i].elements[j].name=="_trksid"){var _16=_4[i].elements[j].value;var _17="";var _18=["p","m","l"];for(var _19 in _18){var _1a=new RegExp(_18[_19]+"[0-9]+(?=.|$)");var _1b=_16.match(_1a);var _1c=_13.clientContext.lnkStr.match(_1a);var _1d=_1c?_1c[0]:(_1b?_1b[0]:null);var _1e=_17.length>0?".":"";if(_1d){_17+=_1e+_1d;}}
_4[i].elements[j].value=_17;return;}}}}}else{if(_14=="search_submit"){for(i=0;i<_4.length;i++){if(_4[i].name==_2){_4[i].submit();return;}}}else{if(_14=="sug_icon_show"){if(_13.clientContext.bShow){t.showImage();}else{t.hideImage();}}else{if(_14=="kw_focus"){t.input.focus();t.input.value=t.input.value+"";}}}}}}});},activateAutoComplete:function(msg){var t=this;if(msg.activated){t.activated=true;}else{t.hideImage();t.activated=false;}},showImage:function(){var t=this;if(t.sShowImg){return;}
if(t.acdivWidth===0){t.acdiv.style.display="inline";t.acdivWidth=t.acdiv.offsetWidth;}
var _22=t.acdivWidth+3;if("BackCompat"==document.compatMode){_22=_22-5;}
t.input.style.width=(t.input.clientWidth-_22)+"px";t.input.style.borderRightWidth="0px";t.acdiv.style.display="inline";t.sShowImg=true;},hideImage:function(){var t=this;if(!t.sShowImg){return;}
var _24=t.acdiv.offsetWidth-5;if("BackCompat"==document.compatMode){_24=_24+6;}
t.input.style.width=(t.input.clientWidth+_24)+"px";t.input.style.borderRightWidth="1px";t.acdiv.style.display="none";t.sShowImg=false;}}).props({init:function(_25,_26){new vjo.darwin.core.ebayheader.autocomplete.AutoComplete(_25,_26);}}).endType();

vjo.ctype("vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteEncoder").protos({aCharList:[[new RegExp("[%]","g"),"_"],[new RegExp("[.]","g"),"_2e"],[new RegExp("[+]","g"),"_2b"],[new RegExp("[']","g"),"_27"]],pseudoDiv:null,constructs:function(){this.pseudoDiv=document.createElement("div");},encode:function(_1){var _2=encodeURIComponent(_1),t=this;for(var j=0;j<t.aCharList.length;j++){var _4=t.aCharList[j];_2=_2.replace(_4[0],_4[1]);}
return _2;},decodeCookie:function(_5){var _6=_5||"";_6=_6.replace(new RegExp("[+]","g")," ");_6=decodeURIComponent(_6);return _6;},encodeHTML:function(_7){var e=this.pseudoDiv;if(typeof(e.textContent)!="undefined"){e.textContent=_7;}else{e.innerText=_7;}
return e.innerHTML;}}).endType();

vjo.ctype("vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteCache").needs("vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteEncoder").protos({oCache:{},oIdx:{},oRef:{},oLeaf:{},oEncoder:null,UN:"undefined",TFU:"function",TNU:"number",constructs:function(){this.oEncoder=new vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteEncoder();},add:function(_1){try{var t=this,cacheItem=t.oCache[kw],kw,kwL,laL,prdL;var _3=false;if(_1[0].version){kw=_1[1];kwL=_1[2].sug?_1[2].sug:_1[2];laL=_1[3];prdL=_1[2].prd;_3=true;}else{kw=_1[0];kwL=_1[1];laL=_1[2];}
if(_3){if(_1.length>=4){t.addItem(kw,kw,"k",kwL,prdL);}else{laL=_1[2];}}else{if(_1.length>=3){t.addItem(kw,kw,"k",kwL,null);}else{try{if(typeof(kwL[0])=="string"){laL=null;t.addItem(kw,kw,"k",kwL,null);}else{if(typeof(kwL[0])=="object"&&(kwL[0]instanceof Array)){laL=_1[1];}}}
catch(err){laL=null;}}}
if(laL!==null){for(var i=0;i<laL.length;i++){var _5=laL[i];var _6=kw+_5[0];var _7=_5[1];var _8;var _9;if(_3){if(_5[2].sug){_8=_5[2].sug;}else{_8=_5[2];}
_9=_5[2].prd;}else{_8=_5[2];}
if(typeof(_8)==t.UN||_8===null){_7="fd";_8=_5[0];}
t.addItem(_6,kw,_7,_8,_9);}}else{t.addItem(kw,kw,"null",null);}}
catch(e){}},addItem:function(_a,_b,_c,_d,_e){var t=this;_a=_a.toLowerCase();_b=_b.toLowerCase();if(_c=="k"){t.oCache[_a]=t.buildItem("k",(typeof(_d)==t.TNU)?(""+_d):_d,_e,_b);return t.oCache[_a];}else{if(_c=="f"){t.oIdx[_a]=t.buildItem("f",(typeof(_d)==t.TNU)?(""+_d):_d,_e,_b);return t.oIdx[_a];}else{if(_c=="fd"){t.oRef[_a]=t.buildItem("fd",_d,_e,_b);return t.oRef[_a];}else{if(_c=="null"){t.oLeaf[_a]=t.buildItem("null",null,_e,_b);return t.oLeaf[_a];}}}}},buildItem:function(_10,_11,_12,_13){var _14={"type":_10,"shortPrefix":_13};if(_11!==null){_14.keyword=_11;}
if(_12!==null){_14.prd=_12;}
return _14;},get:function(_15,_16){_15=_15.toLowerCase();_16=_16.toLowerCase();var t=this,cacheItem=t.oCache[_15],indexItem=t.oIdx[_15],referenceItem=t.oRef[_15];if(typeof(cacheItem)!=t.UN&&typeof(cacheItem)!=t.TFU){return cacheItem;}
if(typeof(indexItem)!=t.UN&&typeof(indexItem)!=t.TFU){return indexItem;}
if(typeof(referenceItem)!=t.UN&&typeof(referenceItem)!=t.TFU){return referenceItem;}
if(_16){var _18=t.oLeaf[_16];if(typeof(_18)!=t.UN&&typeof(_18)!=t.TFU){return _18;}
var _19=[];for(var _1a in t.oIdx){var _1b=t.oIdx[_1a];if(_1b.shortPrefix==_16){_19.push(_1a);}}
_19.sort();if(_19.length===0){return null;}
var _1c=_19.length-1;for(i=0;i<_19.length;i++){if(_15<_19[i]){_1c=i-1;break;}}
if(_1c<0){_1c=0;}
return t.oIdx[_19[_1c]];}
return null;}}).endType();

vjo.ctype("vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteConfig").needs("vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteEncoder").protos({oModel:{"rootDir":"autofill","listSize":10,"delayTime":100,"dirDepth":3,"noSugShowTime":1500,"svcIn":"SVC_GH_IN","svcOut":"SVC_GH_OUT"},defaultAlgo:"1",algoMap:{"1":"f","2":"a"},widthDef:{"unit":9,"min":{"px":233,"char":26},"max":{"px":400,"char":40}},oEncoder:null,constructs:function(_1){this.oEncoder=new vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteEncoder();this.loadConfig(_1);},loadConfig:function(_2){var t=this,tM=t.oModel;if(_2.version===null||_2.algorithm===null){tM.version=_2.algoVerMap[t.defaultAlgo];tM.algorithm=t.algoMap[t.defaultAlgo];}else{tM.version=_2.version;tM.algorithm=t.algoMap[_2.algorithm];}
tM.trkSuggest=_2.trkSuggest;tM.trkProduct=_2.trkProduct;tM.trkRS=_2.trkRS;tM.trkInput=_2.trkInput;tM.trkShow=_2.trkShow;tM.trkHide=_2.trkHide;tM.containerId=_2.containerId;tM.idList=_2.idList;tM.sugDivId=_2.sugDivId;tM.noSugDivId=_2.noSugDivId;tM.hideLnk=_2.hideLnk;tM.baseURL=_2.baseURL;tM.prdURL=_2.prdBaseURL;tM.siteId=_2.siteId;tM.lastSearch=t.oEncoder.decodeCookie(_2.lastSearch);},updateWidthDef:function(_4){var t=this,inputWidth=parseInt(_4.offsetWidth,10);t.widthDef.max={"px":inputWidth,"char":Math.floor(inputWidth/t.widthDef.unit)-1};},getHideLnk:function(){return this.oModel.hideLnk;},getWidthUnit:function(){return this.widthDef.unit;},getWidthMin:function(){return this.widthDef.min;},getWidthMax:function(){return this.widthDef.max;},getRootDir:function(){return this.oModel.rootDir;},getListSize:function(){return this.oModel.listSize;},getDelayTime:function(){return this.oModel.delayTime;},getDirDepth:function(){return this.oModel.dirDepth;},getAlgorithm:function(){return this.oModel.algorithm;},getVersion:function(){return this.oModel.version;},getTrkSuggest:function(){return this.oModel.trkSuggest;},getTrkProduct:function(){return this.oModel.trkProduct;},getTrkRS:function(){return this.oModel.trkRS;},getTrkInput:function(){return this.oModel.trkInput;},getTrkShow:function(){return this.oModel.trkShow;},getTrkHide:function(){return this.oModel.trkHide;},getContainerId:function(){return this.oModel.containerId;},getIdList:function(){return this.oModel.idList;},getKeyUpSvc:function(){return this.oModel.keyUpSvc;},getKeyDownSvc:function(){return this.oModel.keyDownSvc;},getInputOverSvc:function(){return this.oModel.inputOverSvc;},getUpdInputSvc:function(){return this.oModel.updInputSvc;},getFillContentSvc:function(){return this.oModel.fillContentSvc;},getInSvc:function(){return this.oModel.svcIn;},getOutSvc:function(){return this.oModel.svcOut;},getBaseURL:function(){return this.oModel.baseURL;},getPrdURL:function(){return this.oModel.prdURL;},getSiteId:function(){return this.oModel.siteId;},getSugDivId:function(){return this.oModel.sugDivId;},getNoSugDivId:function(){return this.oModel.noSugDivId;},getNoSugShowTime:function(){return this.oModel.noSugShowTime;},getLastSearch:function(){return this.oModel.lastSearch;}}).endType();

vjo.ctype("vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteRequest").needs(["vjo.dsf.XDomainRequest","vjo.dsf.utils.Timer","vjo.dsf.document.Element","vjo.dsf.ServiceEngine","vjo.dsf.Message","vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteCache","vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteEncoder"]).protos({aReqList:[],sRespSvc:"autofill_response",inProcess:false,bInit:false,vX:vjo.dsf.XDomainRequest,uN:"undefined",constructs:function(){},init:function(_1){var t=this;if(t.bInit){return;}
t.oConfig={baseURL:_1.baseURL,dirDepth:_1.dirDepth,rootDir:_1.rootDir,algorithm:_1.algorithm,version:_1.version,siteId:_1.siteId};t.oCache=new vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteCache();t.oEncoder=new vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteEncoder();t.oProcessTimer=new vjo.dsf.utils.Timer();t.oProcessTimer.setInterval(20);t.oProcessTimer.onTick=function(){if(t.inProcess){return;}
t.inProcess=true;try{t.processQue();t.cleanQue();}
catch(e){}
t.inProcess=false;};t.oProcessTimer.start();t.bInit=true;},processQue:function(){var t=this;if(t.aReqList.length===0){return;}
var _4=t.aReqList[0];if(_4&&(_4.state=="wait")){t.aReqList[0].state="process";var _5=t.vX.bUseIframe;try{t.vX.bUseIframe=false;t.aReqList[0].scriptId=t.vX.send(_4.url);}
catch(e){}
t.vX.bUseIframe=_5;}},cleanQue:function(){var t=this;var _7=[],i;for(i=0;i<t.aReqList.length;i++){var _8=t.aReqList[i];if(_8.state=="wait"){_7.push(_8);}else{if(_8.state=="process"){var _9=new Date(),time=_9.getTime()-_8.timestamp;if(time>=1000){t.removeScriptTag(_8.scriptId);t.sendRespService(true,_8.requester,_8.prefix,null,null,_8.shortPrefix);}else{_7.push(_8);}}}}
t.aReqList=_7;},removeScriptTag:function(_a){try{this.vX.getReqDiv().removeChild(vjo.dsf.document.Element.get(_a));}
catch(e){}},getRespSvc:function(){return this.sRespSvc;},getShortPrefix:function(_b){return _b.substr(0,this.oConfig.dirDepth+1);},buildPath:function(_c,_d){var t=this,pd=_d?10000:t.oConfig.dirDepth;var _f=_c.length>pd?_c.substr(0,pd):_c.substr(0,_c.length-1);var _10=_c.length>pd?_c.substr(pd,1):_c.substr(_c.length-1,1);var _11="";var _12=_f.toLowerCase();var _13=_10.toLowerCase();for(var i=0;i<_12.length;i++){_11+=t.oEncoder.encode(_12.charAt(i))+"/";}
return[_11,t.oEncoder.encode(_13),_f+_10];},buildURL:function(){var t=this,tO=t.oConfig,url=tO.baseURL,href=document.location+"";if(href.search(/^https/g)!=-1){url=url.replace(/http:\/\/include/g,"https://secureinclude");url=url.replace(/com:80/g,"com");}
if(typeof(tO.version)==t.uN||tO.version===null){return null;}
if(url.lastIndexOf("/")<url.length-1){url+="/";}
url+=tO.rootDir+"/";url+=tO.algorithm+"/";url+=tO.siteId+"/";url+=tO.version+"/";return url;},addRequest:function(_16,_17,_18,_19){var t=this;var url=t.buildURL();var _1c=t.buildPath(_17,(_18=="fd"));if(url===null){return;}
if(url.lastIndexOf("/")<url.length-1){url+="/";}
url+=_1c[0]+_1c[1];if(_18=="f"&&typeof(_19)!=t.uN&&_19.length>0){url+=_19;}
url+=".js";var _1d=new Date();t.aReqList.push({"requester":_16,"prefix":_17,"shortPrefix":t.getShortPrefix(_17),"url":url,"state":"wait","timestamp":_1d.getTime(),"type":_18});},send:function(_1e,_1f){var t=this;if(!t.bInit){return;}
var _21=t.getShortPrefix(_1f);var _22=t.oCache.get(_1f,_21);if(_22===null){t.addRequest(_1e,_1f,"k");}else{if(_22.type=="f"||_22.type=="fd"){t.addRequest(_1e,_1f,_22.type,_22.keyword);}else{if(_22.type=="null"){t.sendRespService(true,_1e,_1f,[],[],_21,[]);}else{t.sendRespService(false,_1e,_1f,_22.keyword,[],_21,_22.prd);}}}},sendRespService:function(_23,_24,_25,_26,_27,_28,_29){var m=new vjo.dsf.Message(this.sRespSvc);if(_23){m.clientContext={"timeout":true,"prefix":_25,"shortPrefix":_28,"requestId":_24};}else{m.clientContext={"timeout":false,"prefix":_25,"shortPrefix":_28,"requestId":_24,"kwList":_26,"laList":_27,"prdList":_29};}
vjo.dsf.ServiceEngine.handleRequest(m);},handleResponse:function(_2b){var t=this,i,kw,pKwList,pLaList,pPrdList;if(_2b[0].version){kw=_2b[1];pKwList=_2b[2].sug;pLaList=_2b[3];pPrdList=_2b[2].prd;}else{kw=_2b[0];pKwList=_2b[1];pLaList=_2b[2];}
if(!t.bInit){return;}
t.oCache.add(_2b);var _2d=[];for(i=0;i<t.aReqList.length;i++){var _2e=t.aReqList[i];if(_2e.state!="process"){continue;}
if(_2e.prefix.toLowerCase()==_2e.shortPrefix.toLowerCase()){if(_2e.prefix.toLowerCase()==kw.toLowerCase()){_2e.state="done";t.removeScriptTag(_2e.scriptId);t.sendRespService(false,_2e.requester,kw,pKwList,pLaList,_2e.shortPrefix,pPrdList);}}else{if(_2e.shortPrefix.toLowerCase()==kw.toLowerCase()){var _2f=t.oCache.get(_2e.prefix,_2e.shortPrefix);_2e.state="done";t.removeScriptTag(_2e.scriptId);if(_2e.type=="f"&&_2f.type!="k"){t.sendRespService(true,_2e.requester,_2e.prefix,null,null,_2e.shortPrefix,pPrdList);}else{_2d.push(_2e);}}}}
for(i=0;i<_2d.length;i++){var _30=_2d[i];t.send(_30.requester,_30.prefix);}}}).inits(function(){vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteRequest=new vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteRequest();}).endType();

vjo.ctype("vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteLayer").needs(["vjo.dsf.document.Element","vjo.dsf.utils.Object","vjo.dsf.document.Shim","vjo.dsf.EventDispatcher","vjo.dsf.Message","vjo.dsf.ServiceEngine","vjo.dsf.cookie.VjCookieJar","vjo.dsf.window.utils.VjWindowUtils","vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteRequest","vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteEncoder"]).props({_do:function(_1){vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteRequest.handleResponse(_1);}}).protos({cKw:"",lKw:"",cSel:-1,cKwL:[],bInSugDiv:false,oRequest:vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteRequest,blQryEpt:false,oEncoder:new vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteEncoder(),VE:vjo.dsf.document.Element,VED:vjo.dsf.EventDispatcher,VO:vjo.dsf.utils.Object,VS:vjo.dsf.ServiceEngine,constructs:function(_2,_3){var t=this;var _5=t.VED,vO=t.VO,vS=t.VS;t.reqId=_3;t.oConfig=_2;vS.registerSvcHdl(t.oRequest.getRespSvc(),vO.hitch(t,"handleResp"));_5.add(_2.getContainerId(),"mouseover",function(){t.bInSugDiv=true;});_5.add(_2.getContainerId(),"mouseout",function(){t.bInSugDiv=false;});_5.addEventListener(window,"resize",t.onWindowResize,t);var _6=t.VE.get("_nkw_id")?"_nkw_id":"_nkw";_5.addEventListener(_6,"keydown",t.onEntSbmt,t);try{t.VE.promoteToBody(_2.getContainerId());}
catch(e){}
_5.add(_2.getHideLnk(),"click",function(_7){t.setHideSuggestion(true);t.selectSug(null);t.showSugDiv(false);t.setAutoComplete(true);t.showIcon(true);var _8=function(){t.createTrackingImg(t.oConfig.getTrkHide());};window.setTimeout(_8,500);return false;});vS.registerSvcHdl(_2.getOutSvc(),function(_9){var _a=_9.clientContext;t.setInput(_a.src);switch(_a.type){case"kw_keyup":t.kw_keyup(_a);break;case"kw_blur":t.kw_blur(_a);break;case"kw_keydown":t.kw_keydown(_a);break;case"kw_mouseover":t.kw_mouseover(_a);break;case"show_click":t.show_click(_a);break;}});t.prdDiv=t.VE.get("prdDivWrp");t.prdItms=[];var _b=t.prdDiv.getElementsByTagName("tr");for(var i=0;i<_b.length;i++){var _d={};_d.tr=_b[i];var _e=_b[i].getElementsByTagName("td");_d.imgTd=_e[0];_d.titleTd=_e[1];t.prdItms.push(_d);}
t.updTrk(_2.getTrkInput());t.setAutoComplete(t.isHideSuggestion(),true);},handleResp:function(_f){var t=this,cfg=t.oConfig,ctx=_f.clientContext,i;var _11=cfg.getIdList(),id;for(i=0;i<_11.length;i++){id=_11[i];t.VED.unregister(id,"mouseover");}
var _12=ctx.kwList||[];var _13=ctx.laList||[];var _14=ctx.prdList||[];if(t.reqId!=ctx.requestId||ctx.prefix.toLowerCase()!=t.getInputValue().toLowerCase()){return;}
var _15=t.getRecentSearch();if(_15&&_15.length>0){var _16=_15.toLowerCase();var _17=_16.indexOf(t.cKw.toLowerCase());var _18=false;var _19=_16;var _1a=0;while(_17>=0){if(t.isWordStart(_16,_17+_1a)){_18=true;break;}
_1a=_17+1;_19=_19.substr(_17+1);_17=_19.indexOf(t.cKw.toLowerCase());}
if(_18){var _1b=[];_1b.push(_15);for(i=0;i<_12.length;i++){if(_12[i].toLowerCase()!=_16){_1b.push(_12[i]);}}
if(_1b.length>cfg.getListSize()){_1b.pop();}
_12=_1b;}}
if(ctx.timeout&&(ctx.prefix==t.cKw)&&_12.length<1){if(!t.blQryEpt){t.showNoSugMessage(true);t.showSugDiv(true,cfg.getWidthMin().px);}else{t.showSugDiv(false);}
t.blQryEpt=true;return;}
t.blQryEpt=false;t.showNoSugMessage(false);t.cSel=-1;t.cKwL=[];var _11=cfg.getIdList();var _1c=_11.length;var _1d=_14.length;if(_1c>cfg.getListSize()){_1c=cfg.getListSize();}
if(t.oInput){cfg.updateWidthDef(t.oInput);}
var _1e=t.getMaxKwLength(_12,_1c)*cfg.getWidthUnit();if(_1e<cfg.getWidthMin().px){_1e=cfg.getWidthMin().px;}else{if(_1e>cfg.getWidthMax().px){_1e=cfg.getWidthMax().px;}}
for(i=0;i<_1c;i++){var en=_11[i],e=t.VE.get(en);if(e===null){continue;}
e.className="unsel";if(i<_12.length){e.innerHTML=t.genKwHTML(_12[i],t.cKw,cfg.getWidthMax()["char"]);t.VE.toggleHideShow(en,true);var trk=(_12[i]==_15)?cfg.getTrkRS():cfg.getTrkSuggest();t.cKwL[i]={"divId":en,"sugKw":_12[i],"trk":trk,"type":"kw"};}else{t.VE.toggleHideShow(en,false);}}
if(_1d>0&&(""+document.location).search(/https:/g)==-1){t.prdDiv.style.display="block";for(i=0;i<t.prdItms.length;i++){var _21=t.prdItms[i];if(_14[i]){_21.tr.style.display="block";_21.tr.className="unsel";_21.imgTd.innerHTML="<img width='32px' height='32px' src='"+_14[i][2]+"' />";_21.titleTd.innerHTML=t.genPrdHtml(_14[i][1],t.cKw,cfg.getWidthMax()["char"]);t.cKwL.push({"divId":_21.tr.id,"sugKw":t.getInputValue(),"trk":cfg.getTrkProduct(),"type":"prd","pid":_14[i][0]});}else{_21.tr.style.display="none";}}}else{t.prdDiv.style.display="none";}
t.showSugDiv(true,_1e);},getInputValue:function(){var t=this;return t.oInput?t.oInput.value:"";},isHideSuggestion:function(){var vC=vjo.dsf.cookie.VjCookieJar,pbf=vC.readCookie("dp1","pbf"),bit=vC.getBitFlag(pbf,29);return bit==1;},setHideSuggestion:function(_24){var vC=vjo.dsf.cookie.VjCookieJar,pbf=vC.readCookie("dp1","pbf");vC.writeCookielet("dp1","pbf",vC.setBitFlag(pbf,29,_24?1:0));},getRecentSearch:function(){var lss=this.oConfig.getLastSearch();if(lss!==null&&lss.length>0){lss=lss.substring(lss.indexOf(".")+1);return lss;}
return"";},getMaxKwLength:function(_27,_28){if(!_27){return 0;}
var max=0;var _2a=_27.length;if(_2a>_28){_2a=_28;}
for(var i=0;i<_2a;i++){var len=_27[i].length;if(len>max){max=len;}}
return max;},isWordStart:function(pKw,_2e){if(_2e<=0||_2e>pKw.length-1){return true;}
var _2f=new RegExp("[\\s \\.,]");return pKw.substr(_2e-1,1).search(_2f)>=0;},genKwHTML:function(pKw,_31,_32){var _33=_31.length;var out=pKw,t=this;var _35=t.oEncoder,part;var _36=-1;for(var i=0;i<pKw.length;i++){part=pKw.substr(i,_33);if(part.length!=_33){break;}
if(part.toLowerCase()==_31.toLowerCase()&&t.isWordStart(pKw,i)){_36=i;break;}}
var _38=pKw.substring(0,_36);var _39=pKw.substr(_36+_33);var _3a;if(_36>=0){if(pKw.length>_32){if(_38.length>_32){out=_35.encodeHTML(out.substr(0,_32));out+="...";}else{if((_38.length+part.length)>_32){_3a=_32-_38.length;part="<span class='hl'>"+_35.encodeHTML(part.substr(0,_3a))+"...</span>";out=_35.encodeHTML(_38)+part;}else{_3a=_32-_38.length-part.length;out=_35.encodeHTML(_38)+"<span class='hl'>"+_35.encodeHTML(part)+"</span>"+_35.encodeHTML(_39.substr(0,_3a))+"...";}}}else{out=_35.encodeHTML(_38)+"<span class='hl'>"+_35.encodeHTML(part)+"</span>"+_35.encodeHTML(_39);}}else{if(pKw.length>_32){out=_35.encodeHTML(out.substr(0,_32));out+="...";}}
return out;},genPrdHtml:function(_3b,_3c,_3d){var kws=_3c.split(/\s/);var out=_3b.length>75?_3b.substr(0,75)+"...":_3b;for(var i=0;i<kws.length;i++){if(kws[i]!=""){var reg=new RegExp(kws[i],"i");out=out.replace(reg,"<-"+_3b.substr(_3b.search(reg),kws[i].length)+"->");}}
out=out.replace(/<-/g,"<span class='hl'>");out=out.replace(/->/g,"</span>");return out;},startKeyTimer:function(_42){var t=this;t.stopKeyTimer();var _44=function(){var _45=t.getInputValue();if(_42!=_45||_45.length<1){return;}
t.oRequest.send(t.reqId,_42);};t.oKeyTimer=window.setTimeout(_44,t.oConfig.getDelayTime());},stopKeyTimer:function(){var t=this;if(t.oKeyTimer){window.clearTimeout(t.oKeyTimer);t.oKeyTimer=null;}},getKwSelect:function(){var t=this;if(t.cSel<0){return t.cKw;}
return t.cKwL[t.cSel].sugKw;},selectSug:function(_48,_49,_4a){var t=this,kw=t.cKw;if(t.cSel>=0){t.unselectSug(t.cKwL[t.cSel].divId);}
if(_48!==null){var e=t.VE.get(_48);for(var i=0;i<t.cKwL.length;i++){var _4e=t.cKwL[i];if(_4e.divId==_48){t.cSel=i;kw=_4e.sugKw;break;}}
if(e){e.className="sel";}}else{t.cSel=-1;}
if(!_4a){t.updateInput(kw);}
if(!_49){t.focusInput();}},unselectSug:function(_4f){var t=this,e=t.VE.get(_4f);if(e){e.className="unsel";}},createTrackingImg:function(_51){var _52=new Date(),r=_52.getTime();var _53=_51;if(_53.indexOf("?")>0){_53+="&"+r;}else{_53+="?"+r;}
var img=new Image();document.getElementsByTagName("body")[0].appendChild(img);img.setAttribute("src",_53);img.setAttribute("width","1");img.setAttribute("height","1");img.setAttribute("border","0");},sendInSvc:function(_55){var m=new vjo.dsf.Message(this.oConfig.getInSvc());m.clientContext=_55;this.VS.handleRequest(m);},isCtrlKey:function(_57){var _58=[38,39,40,27];for(var i=0;i<_58.length;i++){if(_58[i]==_57){return true;}}
return false;},isIgnorKey:function(_5a){var _5b=[16,17,18];for(var i=0;i<_5b.length;i++){if(_5a==_5b[i]){return true;}}
return false;},isSugShown:function(){var t=this;var _5e=t.VE.get(t.oConfig.getContainerId());disp=_5e.currentStyle?_5e.currentStyle.display:window.getComputedStyle(_5e,null).getPropertyValue("display");return(disp!="none");},isNoSugMsgShow:function(){var t=this;var _60=t.VE.get(t.oConfig.getNoSugDivId());disp=_60.currentStyle?_60.currentStyle.display:window.getComputedStyle(_60,null).getPropertyValue("display");return(disp!="none");},showNoSugMessage:function(_61){var t=this;t.VE.toggleHideShow(t.oConfig.getSugDivId(),!_61);t.VE.toggleHideShow("prdDivWrp",!_61);t.VE.toggleHideShow(t.oConfig.getNoSugDivId(),_61);if(_61){var _63=function(){if(t.isNoSugMsgShow()){t.showSugDiv(false);}};window.setTimeout(_63,t.oConfig.getNoSugShowTime());}},showSugDiv:function(_64,_65){var t=this;if(t.isHideSuggestion()&&_64){t.showIcon(true);return;}
var vS=vjo.dsf.document.Shim;var _68=t.oConfig.getContainerId();var _69=t.VE.get(_68);if(_64){t.posLayer(_65);t.VE.toggleHideShow(_68,true);if(t.oIframeShim){vS.remove(_69,t.oIframeShim);}
t.oIframeShim=vS.add(_69);var _6a=function(){t.regOverEvent(true);};window.setTimeout(_6a,100);}else{t.VE.toggleHideShow(_68,false);if(t.oIframeShim){vS.remove(_69,t.oIframeShim);t.oIframeShim=null;}
t.regOverEvent(false);}},onEntSbmt:function(e){var t=this,idx=t.cSel;if(e.keyCode==13&&idx>-1){var _6d=t.cKwL[idx].type,trk=t.cKwL[idx].trk,pid=t.cKwL[idx].pid;if(pid){var _6e=trk?"&_trksid="+trk:"";t.VED.stopEvent(e);document.location=t.oConfig.getPrdURL()+"/?_pid="+pid+_6e;return false;}}},regOverEvent:function(_6f){var t=this;var _71=t.oConfig.getIdList(),i,id;for(i=0;i<_71.length;i++){id=_71[i];t.VED.unregister(id,"mouseover");t.VED.unregister(id,"click");if(_6f){t.VED.add(id,"mouseover",function(_72){t.bInSugDiv=true;t.selectSug(_72.src.id,false,false);});t.VED.add(id,"click",function(_73){var _74=_73.src.attributes,type,trk,pid;for(i=0;i<t.cKwL.length;i++){if(t.cKwL[i].divId==_73.src.id){type=t.cKwL[i].type;trk=t.cKwL[i].trk;pid=t.cKwL[i].pid;break;}}
if(type=="prd"){if(pid){var _75=trk?"&_trksid="+trk:"";document.location=t.oConfig.getPrdURL()+"/?_pid="+pid+_75;}}else{t.selectSug(_73.src.id);t.submitForm();t.showSugDiv(false);}});}}},posLayer:function(_76){var t=this,gap;var _78=t.getAbsPos(t.oInput);if(_78===null){return false;}
var e=t.VE.get(t.oConfig.getContainerId());if(e===null){return false;}
gap=vjo.dsf.client.Browser.bIE?2:0;e.style.left=_78.left-gap+"px";e.style.top=_78.top+_78.height-gap+"px";if((typeof(_76)!="undefined")&&(_76!==null)){e.style.width=parseInt(_76,10)+"px";}
return true;},getAbsPos:function(_7a){var vW=vjo.dsf.window.utils.VjWindowUtils,t=this;if(_7a===null){return null;}
return{"left":parseInt(vW.offsetLeft(_7a),10),"top":parseInt(vW.offsetTop(_7a),10),"height":parseInt(_7a.offsetHeight,10),"width":parseInt(_7a.offsetWidth,10)};},updateInput:function(pKw){var t=this;if(t.isSugShown()){t.lKw=pKw;t.sendInSvc({"type":"kw_updvalue","value":pKw});}},updTrk:function(_7e){this.sendInSvc({"type":"search_updtrk","lnkStr":_7e});},submitForm:function(){var t=this;if(t.cSel>=0){var trk=t.cKwL[t.cSel].trk;t.updTrk(trk);}
this.sendInSvc({"type":"search_submit"});},showIcon:function(_81){this.sendInSvc({"type":"sug_icon_show","bShow":_81});},focusInput:function(){this.sendInSvc({"type":"kw_focus"});},setAutoComplete:function(bOn,_83){this.sendInSvc({"type":"kw_autocomplete","bOn":bOn,"bSkipFocus":_83});},setInput:function(_84){if(!_84){return;}
var t=this;if(!t.oInput){t.oInput=_84;}
t.oConfig.updateWidthDef(_84);},show_click:function(_86){var t=this;t.showIcon(false);t.setHideSuggestion(false);t.blQryEpt=false;t.oRequest.send(t.reqId,t.getInputValue());var _88=function(){t.createTrackingImg(t.oConfig.getTrkShow());};window.setTimeout(_88,500);t.setAutoComplete(false);},kw_blur:function(_89){var t=this;if(t.isSugShown()&&!t.isNoSugMsgShow()){if(!t.bInSugDiv){var _8b=t.cSel>=0?t.cKwL[t.cSel].divId:null;t.showSugDiv(false);t.unselectSug(_8b);}else{t.focusInput();}}},kw_mouseover:function(_8c){var t=this;if(t.isSugShown()&&!t.isNoSugMsgShow()){t.selectSug(null);}},kw_keydown:function(_8e){var t=this,kc=_8e.keyCode;if((kc==13||kc==9)&&t.isSugShown()&&!t.isNoSugMsgShow()){if(t.cSel>=0){t.updTrk(t.cKwL[t.cSel].trk);}
t.showSugDiv(false);}},kw_keyup:function(_90){var t=this,kc=_90.keyCode,cv=_90.value;if(t.isIgnorKey(kc)){return;}
if(t.isSugShown()&&!t.isNoSugMsgShow()){if(t.isCtrlKey(kc)){var e,nextId,divId;switch(kc){case 38:nextId=t.cSel-1;if(nextId<-1){nextId=t.cKwL.length-1;}
divId=nextId>=0?t.cKwL[nextId].divId:null;t.selectSug(divId);t.cSel=nextId;break;case 40:nextId=t.cSel+1;if(nextId>=t.cKwL.length){nextId=-1;}
divId=nextId>=0?t.cKwL[nextId].divId:null;t.selectSug(divId);break;case 39:if(cv.length!==0){t.cKw=cv;t.startKeyTimer(cv);t.lKw=cv;}
break;case 27:t.selectSug(null);t.showSugDiv(false);break;}}else{if(cv.length!==0){if(t.lKw!=cv){t.cKw=cv;t.startKeyTimer(cv);t.lKw=cv;}}else{t.showSugDiv(false);t.showIcon(false);t.cKw=cv;t.lKw=cv;t.blQryEpt=false;}}}else{if(cv.length!==0){if(t.lKw!=cv){t.cKw=cv;t.startKeyTimer(cv);t.lKw=cv;}}else{t.cKw=cv;if(t.isHideSuggestion()){t.showIcon(false);}
t.lKw=cv;t.blQryEpt=false;}}},onWindowResize:function(_93){if(this.isSugShown()){this.showSugDiv(true);}}}).endType();

vjo.ctype("vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteLayerInit").needs(["vjo.dsf.ServiceEngine","vjo.dsf.EventDispatcher","vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteConfig","vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteLayer","vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteRequest"]).protos({constructs:function(_1){this.oModel=_1;var t=this;var _3=function(){var _4=new vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteConfig(t.oModel);var _5={baseURL:_4.getBaseURL(),dirDepth:_4.getDirDepth(),rootDir:_4.getRootDir(),algorithm:_4.getAlgorithm(),version:_4.getVersion(),siteId:_4.getSiteId()};vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteRequest.init(_5);new vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteLayer(_4,t.oModel.requestId);};vjo.dsf.EventDispatcher.addEventListener(window,"load",_3,window);vjo.ctype("vjo.darwin.domain.finding.autofill.AutoFill").props({_do:function(_6){vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteLayer._do(_6);}});}}).props({init:function(_7){new vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteLayerInit(_7);}}).endType();

vjo.ctype("vjo.darwin.tracking.enabler.TrackingEnablerUtil").needs("vjo.dsf.EventDispatcher").needs("vjo.dsf.utils.URL").needs("vjo.dsf.cookie.VjCookieJar").props({seekParent:function(_1,_2){if(!_1||!_1.tagName){return"";}
if(_1.tagName.toLowerCase()=="a"||_1.tagName.toLowerCase()=="area"){return _1;}
if(_1.tagName.toUpperCase()=="INPUT"&&_1.getAttribute("type")&&_1.getAttribute("type").toUpperCase()=="SUBMIT"){return _1;}
if(_2>0){return this.seekParent(_1.parentNode,_2-1);}else{return"";}},splitParm:function(_3){var v=[-1,-1,-1,-1];var f=_3.split(".");for(var i=0;i<f.length;i++){var s=f[i].substr(0,1);if(s=="p"){v[0]=f[i].substr(1);}
if(s=="c"){v[1]=f[i].substr(1);}
if(s=="m"){v[2]=f[i].substr(1);}
if(s=="l"){v[3]=f[i].substr(1);}}
return v;},enc:function(i){var A=["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"];var P="z";var _b="";var B=A.length;var _d;if(i==-1){return P;}
while(i>=B){_d=i%B;_b=A[_d]+_b;i=i/B|0;}
_b=A[i]+_b;return _b;},checkSiteDomain:function(_e){var dd=document.domain,i=dd.indexOf(".ebay.");if(i!=-1){dd=dd.substr(i+1);}
if(_e&&_e.length>0){if(_e.charAt(0)=="/"){return true;}else{if(_e.indexOf(dd)==-1){return false;}}
return true;}
return dd;}}).endType();

vjo.ctype("vjo.darwin.tracking.enabler.TrackingEnabler").needs("vjo.dsf.utils.URL").needs("vjo.dsf.cookie.VjCookieJar").needs("vjo.darwin.tracking.enabler.TrackingEnablerUtil").props({rewriteURLs:function(_1,_2,_3,_4,_5){if(_1.nativeEvent===null||_1.nativeEvent===undefined){return;}
var _6=_1.nativeEvent.srcElement||_1.nativeEvent.target;if(_6===null||_6===undefined){return;}
if(_6.tagName.toLowerCase()=="img"||_6.tagName.toLowerCase()=="span"){_6=_6.parentNode;}
var _7=_6.getAttribute(_4);if(_7===null){return;}
_7=_7.split(_5);if(_7[0]){var _8=_6.href;if(_8&&vjo.darwin.tracking.enabler.TrackingEnablerUtil.checkSiteDomain(_8)){_8=vjo.dsf.utils.URL.addArg(_8,_2,_7[0]);if(_7[1]){_8=vjo.dsf.utils.URL.addArg(_8,_3,_7[1]);}
_6.href=" "+_8;}}},copySIDToCookie:function(_9,_a,_b,_c){var _d=";";var vj=vjo.dsf.cookie.VjCookieJar;var u=vjo.darwin.tracking.enabler.TrackingEnablerUtil;var x="undefined";if(typeof(_GlobalNavHeaderCookieTracking)==x||!_GlobalNavHeaderCookieTracking){return this.rewriteURLs(_9,_a,_c,_b,_d);}
if(typeof(_GlobalNavHeaderStatic)!=x&&_GlobalNavHeaderStatic){vj.writeCookielet("ds2","sotr");return;}
var pid=_GlobalNavHeaderSrcPageId;var V="a";var P="zzzz";var _14=_9.nativeEvent.srcElement||_9.nativeEvent.target;if(!_14){return;}
_14=u.seekParent(_14,3);if(!_14){return;}
var url=_14.href;var isF=false;var _17=_14.getAttribute(_b);if(_14.tagName=="INPUT"&&_14.getAttribute("type").toUpperCase()=="SUBMIT"){var _18=document.getElementsByName(_a);url=_14.form.action;for(var i=0;i<_18.length;i++){if(_18[i].tagName=="INPUT"&&_18[i].getAttribute("type")&&_18[i].getAttribute("type").toUpperCase()=="HIDDEN"&&_18[i].form==_14.form){isF=true;_17=_18[i].value;}}}
if(!u.checkSiteDomain(url)){return;}
var v=[-1,-1,-1,-1];var _1b=[-1,-1,-1,-1];var oc=vj.readCookie("ds2","sotr");if(oc&&oc.length==12&&oc.substr(0,1)=="a"){var _1d=oc.substr(1,4);if(_1d!="zzzz"){_1b[0]=_1d;}
_1d=oc.substr(5,1);if(_1d!="z"){_1b[1]=_1d;}
_1d=oc.substr(6,3);if(_1d!="zzz"){_1b[2]=_1d;}
_1d=oc.substr(9,3);if(_1d!="zzz"){_1b[3]=_1d;}}
var b=false;if(_17){_17=_17.split(_d);var _1f;if(url&&_17[0]){_1f=_17[0];if(_17[1]){try{url=vjo.dsf.utils.URL.addArg(url,_c,_17[1]);_14.href=" "+url;}
catch(e){}}
v=u.splitParm(_1f);b=true;}}
if(!b){var _20=_a+"=";if(url&&url.indexOf(_20)!=-1){try{var p=(url.substr(url.indexOf(_20)+_20.length)).split("&");_1f=p[0];v=u.splitParm(_1f);b=true;}
catch(e){}}}
if(!b&&v[0]==-1){if(!pid||!_9.nativeEvent){return;}else{v[0]=pid;b=true;}}
if(isF&&v[0]===0&&pid){v[0]=pid;}
if(b){var cv=V;if(v[0]==-1&&_1b[0]!=-1){cv+=_1b[0];}else{cv+=(u.enc(v[0])+P).substr(0,4);}
if(v[1]==-1&&_1b[1]!=-1){cv+=_1b[1];}else{cv+=u.enc(v[1]);}
if(v[2]==-1&&_1b[2]!=-1){cv+=_1b[2];}else{cv+=(u.enc(v[2])+P).substr(0,3);}
if(v[3]==-1&&_1b[3]!=-1){cv+=_1b[3];}else{cv+=(u.enc(v[3])+P).substr(0,3);}
vj.writeCookielet("ds2","sotr",cv);}}}).endType();

vjo.itype("vjo.dsf.common.IJsHandler").protos({handle:function(_1){}}).endType();

vjo.ctype("vjo.darwin.tracking.enabler.TrackingModuleEnabler").needs("vjo.dsf.utils.URL").needs("vjo.dsf.typeextensions.string.Comparison").needs("vjo.dsf.cookie.VjCookieJar").needs("vjo.darwin.tracking.enabler.TrackingEnablerUtil").satisfies("vjo.dsf.common.IJsHandler").protos({constructs:function(_1,_2,_3,_4){this.sCid=_1;this.sParms=_2;this.sCidParms=_3;this.sDelim=_4;this.oCJ=vjo.dsf.cookie.VjCookieJar;this.oU=vjo.darwin.tracking.enabler.TrackingEnablerUtil;},logModuleId:function(_5){var V="a";var P="zzzz";if(!_5.nativeEvent||_5.nativeEvent===null||_5.nativeEvent===undefined){return;}
var sc=false;if(typeof(_GlobalNavHeaderCookieTracking)!="undefined"&&_GlobalNavHeaderCookieTracking){sc=true;}
var _9=_5.nativeEvent.srcElement||_5.nativeEvent.target;if(!_9||_9===null||_9===undefined){return;}
_9=this.oU.seekParent(_9,3);if(!_9){return;}
var _a=this.sCidParms.split(this.sDelim);if(_a[0]&&_9&&_9.href&&!_9.href.has("javascript:")){if(sc){var v=[-1,-1,-1,-1];v=this.oU.splitParm(_a[0]);var _c=V;_c+=(this.oU.enc(v[0])+P).substr(0,4);_c+=this.oU.enc(v[1]);_c+=(this.oU.enc(v[2])+P).substr(0,3);_c+=(this.oU.enc(v[3])+P).substr(0,3);this.oCJ.writeCookielet("ds2","sotr",_c);}else{var _d=vjo.dsf.utils.URL.addArg(_9.href,this.sCid,_a[0]);if(_a[1]){_d=vjo.dsf.utils.URL.addArg(_d,this.sParms,_a[1]);}
_9.href=" "+_d;}}},getAnchor:function(_e){var e=_e;if(e&&e.tagName){if(!e.tagName.is("A")&&!e.tagName.is("INPUT")&&(e.tagName.is("INPUT")&&e.getAttribute("type")!="SUBMIT")){e=this.getAnchor(e.parentNode);}
return e;}},handle:function(_10){this.logModuleId(_10);}}).endType();

vjo.ctype("vjo.darwin.core.ebayheader.searchbox.MiniSearchBox").needs("vjo.dsf.EventDispatcher").protos({constructs:function(_1){this.table=document.getElementById(_1.tableID);var D=vjo.dsf.EventDispatcher,is=this.table.getElementsByTagName("input");this.input=is[0];this.submit=is[1];this.defaultText=_1.defaultText;this.dispDftText();this.showDftText=true;D.addEventListener(this.input,"click",this.clearDftText,this);D.addEventListener(this.submit,"click",this.onSubmit,this);},clearDftText:function(){if(this.showDftText){this.input.value="";this.input.className="";this.showDftText=false;}},dispDftText:function(){this.input.value=this.defaultText;},onSubmit:function(){if(this.input.value==this.defaultText){this.input.value="";}
return true;}}).endType();

vjo.ctype("vjo.darwin.core.ebayheader.rover.FooterRover").needs("vjo.dsf.cookie.VjCookieJar").props({roverService:function(_1){if(document.location.protocol.has("https:")){return;}
this.command=_1;if(!_1){return;}
if(!this.isCookieValid()){return;}
vjo.dsf.EventDispatcher.addEventListener(window,"load",this.sendRequest,this);},sendRequest:function(){var _2=new vjo.dsf.assembly.VjClientAssemblerRequest(this.command,this.handleResponse,this,"cb",false);vjo.dsf.assembly.VjClientAssembler.load(_2);},isCookieValid:function(){var _3=vjo.dsf.cookie.VjCookieJar.readCookie("dp1","idm");if(!_3){return true;}else{return false;}},handleResponse:function(_4){if(_4&&_4.length>1){var _5=_4.length-1;for(i=0;i<_5;i++){this.createImage(_4[i]);}
this.setCookieExpiration(_4[_5]);}},createImage:function(_6){if(_6&&_6.length>1){var _7=document.createElement("IMG");_7.width="1";_7.height="1";_7.src=_6;document.body.appendChild(_7);}},setCookieExpiration:function(_8){if(typeof _8=="number"&&_8>0){vjo.dsf.cookie.VjCookieJar.writeCookielet("dp1","idm","1",_8/86400,"");}}}).endType();

// it_IT/e639i/SYS-LIGER_vjo_e639i10231695_1_it_IT
// b=10231695