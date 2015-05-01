
(function(){var _1=typeof(vjo)=="object";if(_1&&vjo._v==1.4){return;}
var _2=this;vjo={loader:null,_v:1.4,global:this,_bScope:null,_typeMap:{},_bSubClass:false,_callStack:[],$static:function(_3){if(_3==null||_3==_2){return _3;}
if(_3.vj$!="undefined"&&typeof _3.vj$.$s!="undefined"){return _3.vj$.$s;}else{return _3.constructor;}},$ns:function(_4){var $s=vjo.$static(_4);return($s&&$s.vj$)?$s.vj$.b:_4;},isArray:function(_6){if(!_6){return false;}
return(_6.constructor==Array||(typeof _6=="object"&&_6.join&&_6.splice));},getType:function(_7){if(!_7){return;}
var _8=vjo._typeMap[_7],idx=_7.lastIndexOf("."),cn=(idx>-1)?_7.substring(idx+1):_7;if(_8){return _8.pkg[cn];}
return;},mixin:function(_9,_a){var _b=this.getType(_9);if(_b){if(_b.vj$&&_b.vj$._type==="mtype"){if(_b._props){throw"cannot mixin static props to an instance";}
this.extend(_a,_b._protos);}}},proxy:function(fn,_d){return function(){return fn.apply(_d,arguments);};},make:function(_e,_f){var a=arguments,len=a.length,tp=(typeof _f=="function"&&_f.vj$)?_f:this.getType(_f),_vjo={};if(len<2||!tp){throw"context and valid type are required";}
_forEach(_e.vj$,function(val,key){if(typeof val=="function"&&val.vj$&&val.vj$._type){this[key]=val;}},_vjo);var _13=Array.prototype.slice.call(a,2,len);return{_protos:null,protos:function(obj){this._protos=obj;return this;},endType:function(){var t=vjo.ctype(),clztype=tp.vj$._type,rv;if(clztype==="itype"){t.satisfies(tp);}else{if(clztype==="ctype"||clztype==="atype"){var _16=TypeMeta.get(tp.vj$._meta._metaId);if(_16&&!_16.completed){_16.complete();}
t.inherits(vjo.getType(tp.vj$._class));}else{throw"incompatible anonomyous type";}}
t.protos(this._protos);t.complete();t=t.endType();t.__donotconstruct=true;rv=new t;rv.vj$=t.vj$;vjo.extend(rv.vj$,_vjo);(function(){if(this.base){this.base.apply(this,arguments);}}).apply(rv,_13);if(rv.base&&rv._getBase){var fn=function(){};fn.prototype=rv._getBase();rv.base=new fn;rv.base._parent=rv;}
rv.vj$.parent=(tp._outer)?tp._outer:_e;rv.vj$.outer=tp._outer;if(this._protos&&this._protos.constructs){this._protos.constructs.apply(rv);}
return rv;}};},needs:function(clz,_19){if(!clz){return;}
if(!this._bScope){this._bScope={};}
var _1a=_createPkg(clz),cls=_1a.className,tp=_1a.pkg[cls];if(!tp){if(this.loader!=null){var _1b=this._bScope;this._bScope=null;this.loader.load(clz);this._bScope=_1b;tp=_1a.pkg[cls];}}
if(tp){if(typeof _19=="string"&&_19!=""){cls=_19;}
this._bScope[cls]=tp;}},forEach:_forEach,extend:function(){var a=arguments,len=a.length,target,source;if(len==1){target=this;source=a[0];}else{target=a[0];source=a[1];}
for(var _1d in source){var _1e=source[_1d];if(_1e!==undefined){target[_1d]=_1e;}}
if(source.toString!=Object.prototype.toString){target.toString=source.toString;}},needsLib:function(){},createArray:function(val,_20){var arr=[];if(arguments.length>1){for(var ii=0;ii<_20;ii++){if(arguments.length>2){var tmp=[val];for(var k=2;k<arguments.length;k++){tmp[tmp.length]=arguments[k];}
arr[ii]=vjo.createArray.apply(this,tmp);}else{arr[ii]=val;}}}
return arr;}};vjo.Object=function(){this.vj$={_class:"vjo.Object",_type:"ctype",Object:vjo.Object,_meta:{}};};vjo.Object.prototype={_hashCode:-1,constructs:function(){},getClass:_getClazz,hashCode:function(){if(this._hashCode==-1){this._hashCode=++vjo.Object._hashCounter;}
return this._hashCode;},equals:function(o){return(this===o);},toString:function(){return this.getClass().getName()+"@"+this.hashCode().toString(16);}};vjo.extend(vjo.Object,{vj$:{_class:"vjo.Object",_type:"ctype",Object:vjo.Object,_meta:{}},_hashCounter:0,instanceOf:function(o){return(o instanceof this);}});vjo.Class=function(clz,typ){this._name=clz,this._type=typ||"ctype",this.vj$={_class:"vjo.Class",_type:"ctype",Class:vjo.Class,_meta:{}};};vjo.Class.prototype=new vjo.Object();vjo.extend(vjo.Class.prototype,{getName:function(){var n=this._name;if(n){return n;}
return null;},getSimpleName:function(){var n=this.getName();var idx=n.lastIndexOf(".");if(idx!=-1){n=n.substring(idx+1);if(n){idx=n.lastIndexOf("$");if(idx>0){n=n.substring(idx+1);}}}
return n;},getPackageName:function(){var n=this.getName();if(n!=null){var i=n.lastIndexOf(".");if(i>=0){return n.substring(0,i);}}
return"";},isInterface:function(){return(this._type==="itype");},isInstance:function(obj){var tp=vjo.getType(this.getName());if(tp){return tp.instanceOf(obj);}
return false;},toString:function(){return(this.isInterface()?"interface ":"class ")+this.getName();},getClass:_getClazz});vjo.extend(vjo.Class,{forName:function(clz){var err="Type not found for '"+clz+"'";try{var o=eval(clz);if(o&&o.clazz){return o.clazz;}}
catch(e){}
throw err;},instanceOf:function(o){return(o instanceof vjo.Class);},clazz:new vjo.Class("vjo.Class","ctype")});vjo.Object.clazz=new vjo.Class("vjo.Object","ctype");var _34={map:{},inits:{},needs:{},stack:[],inners:{},loaded:{},addType:function(clz){if(!this.map[clz]){this.map[clz]=[];}},addDep:function(clz,dep){if(!clz){return;}
this.addType(clz);var aD=this.map[clz];aD[aD.length]=dep;if(vjo.loader){var stk=this.stack;if(stk.length==0){stk.push(clz);}
if(stk[stk.length-1]===clz){stk.push(dep);}}},addInner:function(clz,fn){if(!this.inners[clz]){this.inners[clz]=[];}
var ins=this.inners[clz];ins.push(fn);},execInners:function(clz){var ins=this.inners[clz];if(ins){var len=ins.length;for(var i=0;i<len;i++){var _41=ins[i];if(_41){_41.complete();}}}
this.inners[clz]=null;},removeDep:function(clz,dep){if(!vjo.loader){return;}
var stk=this.stack;if(stk[stk.length-1]===dep){stk.pop();}},addNeeds:function(clz){var n=this.needs[clz];if(n){while(n.length>0){n.pop()();}
this.needs[clz]=null;}},deferNeed:function(clz,fn){var n=this.needs;if(!n[clz]){n[clz]=[];}
n[clz].push(fn);},register:function(clz,fn){this.inits[clz]=fn;return true;},load:function(clz){if(!clz||this.loaded[clz]){return false;}
var _4d=this.map[clz],ins=this.inits;if(vjo.loader){var stk=this.stack,len=stk.length;if(len>0&&stk[len-1]===clz){stk.pop();}}
if(_4d&&!this.hasCirDep(clz)){var stk=[];this._pushDep(clz,stk,{});while(stk.length>0){var dep=stk.pop(),fn=ins[dep];this.addNeeds(dep);if(fn){fn.complete();}
ins[dep]=null;this.execInners(dep);this.loaded[dep]=true;}}
if((!_4d||_4d.length==0)){this.addNeeds(clz);if(ins[clz]){ins[clz].complete();ins[clz]=null;}
this.execInners(clz);this.loaded[clz]=true;}
return true;},_pushDep:function(clz,_51,_52){var arr=this.map[clz];_51.push(clz);_52[clz]=true;if(!arr||arr.length===0){return;}
var len=arr.length,i=0;for(;i<len;i++){var key=arr[i];if(!_52[key]){this._pushDep(key,_51,_52);}}
return;},_hasCirDep:function(clz,_57,_58){var arr=this.map[_57];if(!arr||arr.length===0){return false;}
var len=arr.length,i=0;for(;i<len;i++){var key=arr[i];if(vjo.loader){if(key===clz){return true;}}else{if(!vjo.getType(key)){return true;}}
if(!_58[key]){_58[key]=true;if(this._hasCirDep(clz,key,_58)){return true;}}}
return false;},hasCirDep:function(clz){var aD=this.map[clz],bInline=(!vjo.loader);if(aD){var len=aD.length;if(bInline){return this._hasCirDep(clz,clz,{});}else{var stk=this.stack,len2=stk.length;for(var i=0;i<len;i++){var dep=aD[i];for(var j=0;j<len2;j++){if(stk[j]===dep){return true;}}}
if(len2>0){return this._hasCirDep(clz,clz,{});}}}
return false;}};var _63={needs:function(clz,_65){if(!clz||this.vj$._meta._isInner){return this;}
var _66=[],useAlias=false;if(typeof clz=="string"){_66=[clz];useAlias=(_65)?true:false;}else{if(vjo.isArray(clz)){_66=clz;}else{return this;}}
_forEach(_66,function(val,key,obj){var cl=val,pObj=vjo._typeMap[val],idx=cl.lastIndexOf("."),cn=(idx>-1)?cl.substring(idx+1):cl,tp=(pObj)?pObj.pkg[cn]:null;_34.addDep(this.vj$._class,cl);if(!tp&&vjo.loader){var _6b=vjo._bScope;vjo._bScope=null;vjo.loader.load(cl);vjo._bScope=_6b;pObj=vjo._typeMap[clz];tp=(pObj)?pObj.pkg[cn]:null;}
_34.removeDep(this.vj$._class,cl);if(_65!==""){if(tp&&!(tp instanceof _MType)){var nm=(useAlias)?_65:cn,err=false;if(this.vj$[nm]&&this.vj$[nm]!==tp){if(_1){err=true;}else{throw"Name collision with type '"+nm+"' in need list.";}}
if(!err){this.vj$[nm]=tp;}}else{_34.deferNeed(this.vj$._class,(function(_6d,_6e,ctx){return function(){var tp=vjo.getType(_6e);if(tp&&tp.vj$&&!(tp instanceof _MType)){if(ctx.vj$[_6d]&&ctx.vj$[_6d]!==tp){throw"Name collision with "+nm+"in need list.";}
ctx.vj$[_6d]=vjo.getType(_6e);}};})((useAlias)?_65:cn,cl,this));}}},this);return this;},props:function(obj,_72){_forEach(obj,function(val,key,obj){if(_isValidProp(key)){var o=this[key]=val;if(_addInner(this,o,"s_inners",key)){if(this.vj$[key]){throw"'"+key+"' in type '"+this.vj$._class+"' conflicts with needed type name";}
var _v={};vjo.extend(_v,this.vj$);delete _v._meta;vjo.extend(_v,o.vj$);_v[key]=o;o.vj$=_v;if(!this.vj$._meta._isInner){_34.addInner(this.vj$._class,TypeMeta.get(o.vj$._meta._metaId));var rt=this.vj$._class;_updateInners(rt,rt+"."+key,o,true);}}else{if(typeof o=="function"&&!o._name&&!o.vj$){o._name=key;}}}},this);return this;},protos:function(obj,_7a){if(!obj){return;}
_forEach(obj,function(val,key,obj){if(key!="base"&&key!="b"){var _7e=this.prototype[key],isType=_isVjoType(val);if(_7e&&typeof _7e=="function"&&key.indexOf("constructs")!=0&&(key.indexOf("_ovld")===-1||key.indexOf("_ovld")!=(key.length-5))&&!_isVjoType(_7e)&&typeof val=="function"&&!isType){this.prototype[key]=(function(_7f,_80){return function(){var _81=this.base;this.base=(_80.prototype._getBase)?_80.prototype._getBase():null;if(this.base){this.base._parent=this;}
var t={},error=false,rv,out=this.vj$.outer;t.vj$=this.vj$;if(!out){_fixScope(_80,this);}
try{rv=_7f.apply(this,arguments);}
catch(e){error=e;}
if(!out){_fixScope(t,this);}
this.base=_81;if(error){throw error;}
return rv;};})(val,this);}else{if(isType&&!this.vj$._meta._isInner){_34.addInner(this.vj$._class,TypeMeta.get(val.vj$._meta._metaId));var rt=this.vj$._class;_updateInners(rt,rt+"."+key,val,false);}
this.prototype[key]=val;}
if(!_addInner(this,val,"_inners",key)){if(typeof val=="function"&&!val._name&&!isType){val._name=key;}}else{if(this.vj$[key]){throw"'"+key+"' in type '"+this.vj$._class+"' conflicts with needed type name";}}}
if(obj.toString!=Object.prototype.toString){this.prototype.toString=obj.toString;}},this);return this;},instanceOf:function(obj){return(obj instanceof this);},endType:function(){_updateInnerEtypes(this.vj$);_34.load(this.vj$._class);if(vjo.validateType){vjo.validateType(this);}
return this;}};var _85={satisfies:function(_86){var _87=[];if(vjo.isArray(_86)){_87=_86;}else{_87=[_86];}
_forEach(_87,function(val,key,obj){var cl=val,_86;if(_isVjoType(cl)){_86=cl;var clz=_86.vj$._class||"",idx=clz.lastIndexOf("."),cn=(idx>-1)?clz.substring(idx+1):clz;if(cn){this.vj$[cn]=_86;}}
var _8d=(_86)?_86:vjo.getType(cl);if(_8d){for(var i in _8d){var val=_8d[i];if(_isValidProp(i)&&!this[i]){this[i]=val;}}}},this);return this;},inherits:function(_8f,isB){if(!isB&&!_isValidInh(_8f)){throw"Cannot inherit from '"+_8f+"'";}
var _91=("vjo.Object"===_8f);if(!isB&&_91){return this;}
var _92;if(_isVjoType(_8f)){_92=_8f;var clz=_92.vj$._class||"",idx=clz.lastIndexOf("."),cn=(idx>-1)?clz.substring(idx+1):clz;if(cn){this.vj$[cn]=_92;}}else{if(_91){_92=vjo.Object;}else{_92=(this.vj$[_8f])?this.vj$[_8f]:vjo.getType(_8f);}}
if(_92){var _94=TypeMeta.get(_92.vj$._meta._metaId);if(_94&&!_94.completed){_94.complete();}
_createInherits(this,_92);}
return this;}};vjo.extend({sysout:{print:function(){if(typeof console!="undefined"){console.info.apply(this,arguments);}},println:function(){if(typeof console!="undefined"){console.info.apply(this,arguments);}},printStackTrace:function(){}},syserr:{print:function(){if(typeof console!="undefined"){console.warn.apply(this,arguments);}},println:function(){if(typeof console!="undefined"){console.warn.apply(this,arguments);}},printStackTrace:function(){}},jsunit:{assertEquals:function(){},assertTrue:function(){},assertFalse:function(){}}});function TypeMeta(_95,_96,cfg){this._needs=[];this._props=null;this._protos=null;this._satisfies=[];this._mixins=[];this._inherits=[];this._inits=null;this._completed=false;this._isInner=(_95)?false:true;this._name=_95;this._kind=_96;this.init(cfg);this.setup();}
TypeMeta.prototype={init:function(cfg){this._cfg={satisfiesFn:_85.satisfies,inheritsFn:_85.inherits,protosFn:_63.protos,postDefFn:function(){},typeDef:null,baseType:"vjo.Object"};if(cfg){vjo.extend(this._cfg,cfg);}},setup:function(){var t=this._type=this._cfg.typeDef||_createType(this._name,"itype"===this._kind);t.vj$._type=this._kind||"ctype";var id=this._name;if(this._isInner){id=TypeMeta.id();t.vj$._meta._isInner=true;}else{var _9b=_createPkg(id);if(!_9b.pkg[_9b.className]){_9b.pkg[_9b.className]=this._type;}
t.vj$[_9b.className]=t;_34.register(id,this);}
this._isDup=(!this._isInner&&TypeMeta.get(id)!=null);if(!this._isDup){TypeMeta.put(this,id);}
if(this._kind!="itype"){this._cfg.inheritsFn.call(this._type,this._cfg.baseType,true);}
t._inherits=null;_createClazz(t);t.vj$._meta._metaId=id;},needs:function(_9c,_9d){_63.needs.apply(this._type,arguments);return this;},singleton:function(){return this;},makeFinal:function(){return this;},satisfies:function(_9e){var _9f=[];if(vjo.isArray(_9e)){_9f=_9e;}else{_9f=[_9e];}
_forEach(_9f,function(val,key,obj){_63.needs.call(this._type,val);this._satisfies.push(val);},this);return this;},props:function(_a3){if(this._props){throw"multiple props blocks are not allowed";}
this._props=_a3;_63.props.apply(this._type,arguments);return this;},protos:function(_a4){if(this._protos){throw"multiple protos blocks are not allowed";}
this._protos=_a4;return this;},inherits:function(_a5){var _a6=[];if(vjo.isArray(_a5)){_a6=_a5;}else{_a6=[_a5];}
_forEach(_a6,function(val,key,obj){this.needs(val);this._inherits.push(val);},this);return this;},mixin:function(_aa){var _ab=[];if(vjo.isArray(_aa)){_ab=_aa;}else{_ab=[_aa];}
_forEach(_ab,function(val,key,obj){this.needs(val);this._mixins.push(val);},this);return this;},inits:function(fn){this._inits=fn;return this;},validateAndMerge:function(_b0,_b1){var p=(_b1)?_b0[_b1]:_b0;return function(val,key,obj){if(p[key]){throw"collision when mixing in '"+key+"' into "+this._name;}else{p[key]=val;}};},complete:function(){if(this._completed){return this;}
this._completed=true;_updateInnerEtypes(this._type.vj$);var _b6=this._mixins;var p={_props:{}};for(var i=0;i<_b6.length;i++){var m=vjo.getType(_b6[i]);if(!m||!m.vj$||m.vj$._type!="mtype"){throw _b6[i]+"is not a valid mtype.";}
var exp=m._expects;sats=m._satisfiers;if(!this._protos){this._protos={};}
if(!this._props){this._props={};}
_copyNS(m.vj$,this._type.vj$);var nm=m.vj$._class,idx=nm.lastIndexOf(".");var clz=(idx!=-1)?nm.substring(idx+1):nm;if(!this._type.vj$[clz]){this._type.vj$[clz]=this._type;}else{throw clz+" is already defined in the current namespace";}
_forEach(m._protos,this.validateAndMerge(this,"_protos"),this);_forEach(m._props,this.validateAndMerge(p,"_props"),this);for(var j=0;j<sats.length;j++){this._satisfies.push(sats[i]);}
if(exp){this._satisfies.push(exp);}}
_63.props.call(this._type,p._props);if(this._satisfies.length>0){this._cfg.satisfiesFn.call(this._type,this._satisfies);}
var _be=this._inherits.length;if(_be>0){if(this._kind!="itype"&&_be>1){throw"type can only inherit from one type";}
for(var i=0;i<_be;i++){this._cfg.inheritsFn.call(this._type,this._inherits[i]);}}
if(this._protos){this._cfg.protosFn.call(this._type,this._protos);}
this._cfg.postDefFn.call(this);if(this._inits&&!this._isDup){this._inits.call(this._type);}
return this;},canComplete:function(){var b=(this._inits==null);for(var i=0;i<this._inherits.length;i++){var s=this._inherits[i];if(!_isVjoType(s)){b=false;break;}}
for(var i=0;b&&i<this._satisfies.length;i++){var s=this._satisfies[i];if(!_isVjoType(s)){b=false;break;}}
for(var i=0;b&&i<this._mixins.length;i++){var m=this._mixins[i];if(!_isVjoType(m)){b=false;break;}}
return b;},endType:function(){if(!this._isInner){_34.load(this._name);if(vjo.validateType){vjo.validateType(this._type);}}else{if(this.canComplete()){this.complete();}}
return this._type;}};vjo.extend(TypeMeta,{_count:0,_pre:"tmp",_reg:{},id:function(){return this._pre+this._count++;},put:function(_c3,id){var nm=(id)?id:this._pre+this._count++;this._reg[nm]=_c3;},get:function(id){return this._reg[id];}});vjo.ctype=function(clz){var t=new TypeMeta(clz);return t;};vjo.type=function(clz){var t=new TypeMeta(clz);t.inits=function(fn){if(fn&&!this._isDup){fn.call(this._type);}
return this;};t.props=function(_cc){_63.props.apply(this._type,arguments);return this;};t.protos=function(_cd){_63.protos.apply(this._type,arguments);return this;};t.inherits=function(clz){_85.inherits.apply(this._type,arguments);return this;};t.satisfies=function(clz){_85.satisfies.apply(this._type,arguments);return this;};return t;};vjo.itype=function(clz){var t=new TypeMeta(clz,"itype",{inheritsFn:function(_d2){var _d3=(this.vj$[_d2])?this.vj$[_d2]:(this.vj$.b&&this.vj$.b[_d2])?this.vj$.b[_d2]:vjo.getType(_d2);if(_d3){_createInherits(this,_d3);}
return this;}});t._type.instanceOf=function(obj){var rv=true,proto=this.prototype;for(var _d6 in proto){if(_isValidInst(_d6)&&typeof obj[_d6]=="undefined"){rv=false;break;}}
return rv;};return t;};vjo.atype=vjo.ctype;function _addMixinMethods(to,_d8,ns){if(!_d8||typeof _d8!="object"){return;}
var b=true;for(var i in _d8){b=false;if(!reservedMProp[i]){to[i]=_d8[i];}}
return b;}
function _MType(clz){var t=this;t.vj$={_type:"mtype",_class:clz,_meta:{}};t._props=null;t._protos={};t._expects="";t._satisfiers=[];t.needs=function(){return _63.needs.apply(this,arguments);};t.props=function(_de){var p;if(!this._props){p={};}else{p=this._props;}
if(!_addMixinMethods(p,_de,this.vj$)){if(!this._props){this._props=p;}}
return this;};t.protos=function(_e0){_addMixinMethods(this._protos,_e0,this.vj$);return this;};t.expects=function(clz){this._expects=vjo.getType(clz);return this;};t.satisfies=function(clz){this._satisfiers[this._satisfiers.length]=clz;return this;};t.endType=function(){if(this.vj$._class){_34.load(this.vj$._class);}
return this;};return t;}
vjo.mtype=function(clz){var _e4=new _MType(clz);if(!clz||_isInnerClass(clz)){_e4.vj$._meta._isInner=true;}
if(!clz){return _e4;}
var _e5=_createPkg(clz);return(_e5.pkg[_e5.className])?_e4:(_e5.pkg[_e5.className]=_e4);};vjo.etype=function(clz){_createEnum();var _e7=function(_e8){this.vj$=_e7.vj$;if(_e8!=false){if(!this.constructs){throw"'"+this.vj$._class+"' cannot be instantiated";}
var rv=this.constructs.apply(this,_e8);}};_e7.vj$={_class:clz,_type:"etype",_meta:{}};_e7.instanceOf=function(o){return(o instanceof this);};var t=new TypeMeta(clz,"etype",{typeDef:_e7,baseType:"vjo.Enum",postDefFn:function(){if(this._eVals){this._type.values.call(this._type,this._eVals);}}});vjo.extend(t,{inherits:function(){throw"Invalid type definition. etype cannot be inheritted from another type";},values:function(_ec){this._eVals=_ec;return this;}});var en=t._type;en._enums=[];en.from=vjo.Enum.from;en.prototype.toString=vjo.Enum.prototype.name;en.values=function(_ee){if(arguments.length==0){if(typeof this._enums.slice!="undefined"){return this._enums.slice();}else{var a=[];for(var i=0;i<this._enums.length;i++){if(typeof this._enums[i]!="undefined"){a[i]=this._enums[i];}}
return a;}}else{var ord=0;if(typeof _ee=="string"&&_ee.length>0){while(_ee.indexOf(" ")>-1){_ee=_ee.replace(" ","");}
if(_ee.indexOf(",")>0){var a=_ee.split(","),t;if(a[0]&&a[0].indexOf(":")>0){throw"Invalid labels for etype values: "+a[0];}
for(var i=0,l=a.length;i<l;i++){var eV=a[i];if(i==0&&t&&t.length>0){eV=t[0];}else{if(a[i].indexOf(":")>-1){eV=a[i].split(":")[0];}}
this._enums[this._enums.length]=new en(false);this._enums[this._enums.length-1]._name=eV;}}else{this._enums[0]=new en(false);this._enums[0]._name=_ee;}}else{for(var itm in _ee){this._enums[this._enums.length]=new en(_ee[itm]);this._enums[this._enums.length-1]._name=itm;}}
for(var i=0,l=this._enums.length;i<l;i++){if(typeof this._enums[i]!="undefined"){var nm=this._enums[i]._name;if(this[nm]){throw"Invalid prop member. Cannot use etype value as prop member.";}
this[nm]=this._enums[i];this[nm]._ord=ord++;}}}
en.prototype.constructs=null;return this;};return t;};vjo.otype=function(clz){return{defs:function(_f6){return this;},endType:function(){}};};var _f7={},reservedProto={},reservedMProp={},reservedClz={},reservedInh={};_forEach("props protos inherits prototype inits satisfies b mixin _inherits _satisfiers singleton instanceOf vj$".split(" "),function(val,key,obj){this[val]=true;},_f7);_forEach("constructs getClass _getBase base vj$".split(" "),function(val,key,obj){this[val]=true;},reservedProto);_forEach("props protos _props _protos vj$ _expects expects _satisfiers satisfies endType".split(" "),function(val,key,obj){this[val]=true;},reservedMProp);_forEach("vjo.Class vjo.Object".split(" "),function(val,key,obj){this[val]=true;},reservedClz);function _createType(clz,isI){isI=isI||false;if(!_isValidClz(clz)){throw"Invalid type name '"+clz+"'";}
var base=function(isB){this.constructor=base;var _108=TypeMeta.get(base.vj$._meta._metaId);if(_108&&!_108._completed){_108.complete();_forEach(base.prototype,function(val,key,_10b){this[key]=val;},this);}
this.vj$=base.vj$;var t=this.vj$._type;if(!isB&&(t=="itype"||t=="atype"||t=="mtype")){throw t+" "+this.vj$._class+" cannot be instantiated";}
_processInners(this,base.vj$,base.clazz);if(!base.__donotconstruct){if(!this.constructs){this.constructs=function(){};}
var fn,c=this.constructs,dconstruct=false;if(this.base&&this._getBase){fn=function(){};fn.prototype=this._getBase();dconstruct=fn.prototype._constructs||false;}
if(dconstruct&&this.base){var cstr=c.toString();if(cstr.indexOf("this.base(")===-1&&cstr.indexOf("this.constructs")===-1){this.base();}}
var rv=c.apply(this,arguments);if(fn){this.base=new fn;this.base._parent=this;}
if(rv){return rv;}}
return null;};base._name="base";base.vj$={_class:clz,_meta:{}};base.instanceOf=function(o){return(o instanceof this);};return base;}
function _createClazz(typ){var nm=typ.vj$._class;if(typ.clazz&&typ.clazz._name){nm=typ.clazz._name;}
typ.clazz=new vjo.Class(nm,typ.vj$._type);if(typ.prototype){typ.prototype.getClass=_getClazz;}else{typ.getClass=_getClazz;}}
function _createInherits(_113,type){var _115=type.prototype;type.__donotconstruct=true;var cls=new type(true);delete type.__donotconstruct;cls.constructs=null;cls.constructor=_113;var _117={};cls.base=function(){var _118=this.base,_115=type.prototype,gb=_115._getBase,c=_115.constructs;if(_115.base){this.base=_115.base;}
var cstr=(c)?c.toString():"",b=(cstr.indexOf("this.base(")===-1&&cstr.indexOf("this.constructs")===-1);if(gb&&gb()._constructs&&b){this.base();}
if(_115.constructs){var isIn=this.vj$._meta._isInner,error=false;if(!isIn){_fixScope(type,this);}
try{_115.constructs.apply(this,arguments);}
catch(e){error=e;}
if(!isIn){_fixScope(_113,this);}
if(error){throw error;}}
this.base=_118;};cls._getBase=function(){return _117;};function createBaseRef(type,func,der){return function(){var scp=(this._parent)?this._parent:this,rv,error=false,cbase=scp.base;scp.base=(type.prototype._getBase)?type.prototype._getBase():null;if(scp.base){scp.base._parent=scp;}
_fixScope(type,scp);try{rv=func.apply(scp,arguments);}
catch(e){error=e;}
_fixScope(der,scp);scp.base=cbase;if(error){throw error;}
return rv;};}
for(var i in _115){var pt=_115[i];if(i==="toString"){continue;}
if(i==="constructs"&&typeof pt==="function"){if(type!==vjo.Object&&pt.length===0){_117._constructs=true;}}else{if(_115==vjo.Object.prototype||_isValidInst(i)){if(typeof pt==="function"&&!(pt instanceof RegExp)&&!pt.vj$){_117[i]=createBaseRef(type,pt,_113);var _121=pt.toString();if(_121.indexOf("this.base."+i+"(")!=-1){cls[i]=(function(fn){return function(){return fn.apply(this,arguments);};})(_117[i]);}else{if(!pt.vj$){cls[i]=(function(type,func,der){return function(){var _126=false,rv;_fixScope(type,this);try{rv=func.apply(this,arguments);}
catch(e){_126=e;}
_fixScope(der,this);if(_126){throw _126;}
return rv;};})(type,pt,_113);}else{cls[i]=pt;}}}else{if(pt&&pt.vj$&&pt.vj$._type&&pt.vj$._meta._isInner){if(!_113.vj$._meta._inners){_113.vj$._meta._inners={};}
_113.vj$._meta._inners[i]=pt;}
cls[i]=pt;}}}}
if(_117.toString!=vjo.Object.prototype.toString){_117.toString=createBaseRef(type,_115.toString,_113);}
_113.prototype=cls;_63.protos.call(_113,{getClass:cls.getClass});}
function _isVjoType(clz,_128){if(clz&&clz.vj$&&clz.vj$._type){if(typeof clz==="function"){return true;}else{if(clz.vj$._type==="mtype"&&!_128){return true;}}}
return false;}
function _createPkg(clz,_12a){if(!clz){return null;}
if(vjo._typeMap[clz]){return vjo._typeMap[clz];}
var _12b=clz.split("."),len=_12b.length;if(_12a){vjo._typeMap[clz]={pkg:{className:_12b[len-1]}};}else{var pkg=vjo.global;for(var i=0;i<len-1&&pkg&&_12b[i];i++){pkg=(pkg[_12b[i]])?pkg[_12b[i]]:pkg[_12b[i]]={};}
vjo._typeMap[clz]={pkg:pkg,className:(len>0)?_12b[len-1]:""};}
return vjo._typeMap[clz];}
function _createEnum(){if(typeof vjo.Enum=="function"){return;}
var nm="vjo.Enum";var _12f=vjo.ctype(nm).props({from:function(){if(!arguments[0]){throw"Invalid argument value: "+arguments[0];}
var s=arguments[0];if(arguments.length==2){s=arguments[1];if(s){var clz=arguments[0];try{var n=clz.getName();while(n.indexOf("$")>0){n=n.replace("$",".");}
var o=eval(n);if(o[s]){return o[s];}}
catch(a){}}
throw"No enum const "+arguments[0].getName()+"."+s;}else{if(this[s]){return this[s];}}
throw"No enum const "+s;}}).protos({_name:null,_ord:-1,constructs:function(){throw"cannot instantiate 'vjo.Enum'";},name:function(){return this._name;},ordinal:function(){return this._ord;},compareTo:function(o){if(o==null){throw"compare to Etype value cannot be null";}
return(this.ordinal()-o.ordinal());},equals:function(o){return(this===o);}}).endType();reservedClz[nm]=true;reservedInh[nm]=true;}
function _getClazz(){var n=clz=this.vj$._class;var idx=n.lastIndexOf(".");if(idx!=-1){clz=n.substring(idx+1);}
if(this.vj$[clz]){return this.vj$[clz].clazz;}
return null;}
function _updateInners(_138,_139,_13a,_13b){if(_13a&&_13a.vj$){_13a.vj$._class=_139;var idx=_139.lastIndexOf("."),snm=_139.substring(idx+1);_13a.vj$[snm]=_13a;if(_13a.clazz&&_138){if(_139.indexOf(_138)==0){var tmp=_139.replace(_138,"");while(tmp.indexOf(".")>-1){tmp=tmp.replace(".","$");}
_13a.clazz._name=_138+tmp;}else{_13a.clazz._name=_138+"$"+snm;}}
_createPkg(_139,true).pkg[snm]=_13a;var ins=(_13b)?_13a.vj$._meta.s_inners:_13a.vj$._meta._inners;if(ins){_forEach(ins,function(val,key){_forEach(_13a.vj$,function(val,key){if(!this[key]&&val&&val.vj$){this[key]=val;}},val.vj$);var m=TypeMeta.get(val.vj$._meta._metaId);if(val.vj$&&m){_34.addInner(_138,m);}
_updateInners(_138,_139+"."+key,val,_13b);});}}}
function _updateInnerEtypes(_144){if(!_144._class){return;}
var _145=_144._meta;if(_145.s_inners){for(var k in _145.s_inners){if(_145.s_inners[k].vj$._type=="etype"){for(var i=0;i<_145.s_inners[k]._enums.length;i++){_145.s_inners[k]._enums[i].vj$=_145.s_inners[k].vj$;_updateInnerEtypes(_145.s_inners[k]._enums[i].vj$);}}
_updateInnerEtypes(_145.s_inners[k].vj$);}}
if(_145._inners){for(var k in _145._inners){if(_145._inners[k].vj$._type=="etype"){if(!_145._inners[k].vj$._class){_145._inners[k].clazz._name=_145._inners[k].vj$._class=_145._class+"."+k;}
for(var i=0;i<_145._inners[k]._enums.length;i++){_145._inners[k]._enums[i].vj$=_145._inners[k].vj$;_updateInnerEtypes(_145._inners[k]._enums[i].vj$);}}
_updateInnerEtypes(_145._inners[k].vj$);}}}
function _addInner(clz,_149,_14a,key){if(!clz||!_149||!key){return false;}
if(_149.vj$&&_149.vj$._type&&_149.vj$._meta._isInner){if(!_149.vj$._class&&clz.vj$._class){var cn=_149.vj$._class=clz.vj$._class+"."+key;if(_149.clazz){_149.clazz._name=cn;}
_createPkg(cn,true).pkg[key]=_149;}
if(_14a){if(!clz.vj$._meta[_14a]){clz.vj$._meta[_14a]={};}
clz.vj$._meta[_14a][key]=_149;}
return true;}
return false;}
function _hasCollisionWithMixin(type,name,_14f){var mxns=type.vj$._meta.mixins;if(!mxns||mxns.length==0){return false;}
for(var i=0;i<mxns.length;i++){var mxn=mxns[i];if(_14f){if(mxn._props&&mxn._props[name]){return true;}}else{if(mxn._protos[name]){return true;}}}
return false;}
function _isValidInst(pVal){return!(reservedProto[pVal]===true);}
function _isValidClz(pVal){return!(reservedClz[pVal]===true);}
function _isValidInh(pVal){return!(reservedInh[pVal]===true);}
function _forEach(_156,_157,_158){if(!_156){return;}
var name,i=0,len=_156.length;if(!vjo.isArray(_156)){for(name in _156){if(_157.call(_158,_156[name],name,_156)===false){break;}}}else{for(;i<len;i++){if(_157.call(_158,_156[i],i,_156)===false){break;}}}
return _156;}
function _processInners(_15a,_15b,_15c){var _15d=(_15b)?_15b._meta._inners:null;if(!_15a||!_15b||!_15d||_15d.length==0){return;}
for(var k in _15d){_15a[k]=(function(t,type,nm){var fn=function(){var cn=t.vj$._class+"."+nm;var m=TypeMeta.get(type.vj$._meta._metaId);if(m){m.complete();}
type.__donotconstruct=true;var tp=new type();delete type.__donotconstruct;var _v={};vjo.extend(_v,_15b);delete _v._meta;vjo.extend(_v,type.vj$);_v[nm]=type;tp.vj$=_v;tp.vj$.outer=t;_processInners(tp,_v,type.clazz);type.vj$._class=tp.vj$._class=cn;var s=type.prototype;if(type.clazz&&!type.clazz._name){type.clazz._name=tp.vj$._class;}else{_createClazz(type);}
if(s.constructs){s.constructs.apply(tp,arguments);}
return tp;};fn._outer=t;return fn;})(_15a,_15d[k],k);_15a[k].vj$=_15d[k].vj$;}}
function _copyNS(from,to){_forEach(from,function(val,key){if(_isVjoType(val,true)){if(this[key]&&this[key]!==val){throw key+" is already defined in the current namespace";}
this[key]=val;}},to);}
function _isInnerClass(clz){if(!clz){return true;}else{if(clz.indexOf(".")==-1){return false;}}
var tp=clz;while((i=tp.lastIndexOf("."))>0){tp=tp.substring(0,i);if(vjo._typeMap[tp]){return true;}}
return false;}
function _isValidProp(pVal){return!(_f7[pVal]===true);}
function _fixScope(from,to){to.vj$=from.vj$;}})();

vjo.ctype("vjo.dsf.error.Error").protos({constructs:function(_1,_2,_3){this.message=_1;this.url=_2;this.lineNumber=_3;this.userAgent=navigator.userAgent;}}).endType();

vjo.ctype("vjo.dsf.error.ErrorHandlerManager").needs("vjo.dsf.error.Error").props({dsfErrors:[],handlers:[],add:function(_1,_2,_3){this.dsfErrors[this.dsfErrors.length]=new vjo.dsf.error.Error(_1,_2,_3);return true;},convertErrorTo:function(_4){var _5=_4?"<br>":"\n";var _6=_4?"<html><title>":"";var _7=_4?"</title><body>":_5;var _8=_4?"</body></html>":"";var _9=_4?"<h2>":"";var _a=_4?"</h2>"+_5:_5;var _b=_4?"<b>":"";var _c=_4?"</b>":"";var _d=_6+"V4 JS Errors"+_7;_d+=_9+"Total number of errors = "+this.dsfErrors.length+_a;for(var i=0;i<this.dsfErrors.length;i++){_d+=_b+"Message: "+_c+this.dsfErrors[i].message;_d+=_5;_d+=_b+"URL :"+_c+this.dsfErrors[i].url;_d+=_5;_d+=_b+"Line Number :"+_c+this.dsfErrors[i].lineNumber;_d+=_5;_d+=_b+"User Agent :"+_c+this.dsfErrors[i].userAgent;_d+=_5;_d+=_5;}
_d+=_8;return _d;},toHTML:function(){return this.convertErrorTo(true);},toText:function(){return this.convertErrorTo(false);},show:function(){var _f=window.open("","V4JSErrors","height=500,width=500,menubar=no,status=no,scrollbars=yes");if(window.focus){_f.focus();}
_f.document.write(this.toHTML());_f.document.close();},register:function(_10){if(!_10){return this;}
if(typeof _10!="object"){return this;}
this.handlers[this.handlers.length]=_10;return _10;},clear:function(){this.handlers.length=0;},process:function(_11,_12,_13){for(var i=0;i<this.handlers.length;i++){this.handlers[i].handle(_11,_12,_13);}},enableOnError:function(){window.onerror=function(_15,url,_17){vjo.dsf.error.ErrorHandlerManager.process(_15,url,_17);return true;};}}).endType();

vjo.ctype("vjo.dsf.error.DefaultErrorHandler").needs("vjo.dsf.error.ErrorHandlerManager").protos({constructs:function(){},handle:function(_1,_2,_3){vjo.dsf.error.ErrorHandlerManager.add(_1,_2,_3);}}).endType();

vjo.ctype("vjo.dsf.Event").protos({constructs:function(_1,_2,_3){this.src=_1;this.eventType=_2;this.nativeEvent=_3;}}).endType();

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
var b=this.isBound(id,_19),rv=this.register(id,_19,_1a,_1b);if(!b){var b=this.bind(id,_19);if(b==null){var len=this.unboundElems.length;this.unboundElems[len]=id;}}
return rv;},addEventListener:function(_1e,_1f,_20,_21,_22){var scp=_21||vjo.global;if(typeof _1e=="string"){_1e=this.vj$.Element.get(_1e);}
if(!_1e){return false;}
var _24=function(_25){var ev=_25||window.event;var rv=_20.call(scp,ev,_1f);if(rv===false){vjo.dsf.EventDispatcher.stopEvent(ev);}
if(typeof rv!="undefined"){return rv;}};if(window.addEventListener){_1e.addEventListener(_1f,_24,_22||false);this.registerNative(_1e,_1f,_24);return _24;}else{if(window.attachEvent){_1e.attachEvent("on"+_1f,_24);this.registerNative(_1e,_1f,_24);return _24;}}
_1e["on"+_1f]=_24;return false;},bind:function(id,_29){var _2a=this.vj$.Element.get(id);if(id=="body"||_2a==document.body){_2a=document.body;if(_29=="load"||_29=="unload"){var rv=this.addEventListener(window,_29,function(_2c){var oED=vjo.dsf.EventDispatcher;if(typeof oED.fCustomLoad[_29]=="function"){oED.fCustomLoad[_29]();}
oED.run(document.body,_2c||window.event,_29);oED.unregister("body",_29);oED.fCustomLoad={};});if(rv===false){if(_2a.vjLoadSet){return this;}else{_2a.vjLoadSet=true;var _2e=window["on"+_29]||"";if(_2e){this.fCustomLoad[_29]=_2e;}}}
return this;}}
if(_2a){this.addEventListener(_2a,_29,this.notifier,_2a);return this;}
return null;},notifier:function(_2f,_30){return vjo.dsf.EventDispatcher.run(this,_2f||window.event,_30);},reBind:function(){var eH=this.eventHandlers,uE=this.unboundElems,len=uE.length,tmp=[];for(var i=0;i<len;i++){var id=uE[i],hdls=eH[id];if(hdls){for(var _34 in hdls){if(!this.hasBinding(id,_34)){var rv=this.bind(id,_34);if(rv==null){tmp[tmp.length]=id;}}}}}
this.unboundElems=tmp;},isBound:function(id,_37){var _38=this.eventHandlers[id];return(_38&&_38[_37]&&_38[_37].length>0);},hasBinding:function(id,_3a){var nEH=this.nativeEventHandlers;if(nEH[id]&&nEH[id][_3a]){var aH=nEH[id][_3a],len=aH.length,rv=false;for(var i=0;i<len;i++){var str=aH[i].toString();if(str&&str.indexOf("vjo.dsf.EventDispatcher")!=-1){return true;}}}
return false;},removeEventListener:function(_3f,_40,_41){if(!_3f||!_40){return;}else{if(typeof _3f=="string"){_3f=this.vj$.Element.get(_3f);}}
if(window.addEventListener&&_41){_3f.removeEventListener(_40,_41,false);}else{if(window.attachEvent&&_41){_3f.detachEvent("on"+_40,_41);}else{_3f["on"+_40]=null;}}},detachNativeHandlers:function(_42,_43){var id=(_42==window)?"body":_42.id;var _45=this.nativeEventHandlers[id];if(_45&&_45[_43]){var h=_45[_43],len=h.length;for(var i=0;i<len;i++){this.removeEventListener(_42,_43,_45[_43][i]);}
_45[_43]=[];}},detachHandler:function(id,_49,_4a){var _4b=this.eventHandlers[id];if(!_4b||!_4b[_49]){return;}
var h=[],len=_4b[_49].length;for(var i=0;i<len;i++){if(_4a!=_4b[_49][i]){h[h.length]=_4b[_49][i];}}
this.eventHandlers[id][_49]=h;},detachHandlers:function(id,_4f){this.unregister(id,_4f);var _50=this.vj$.Element.get(id);if(id=="body"){_50=window;}
if(_50){this.detachNativeHandlers(_50,_4f);}},stopEvent:function(evt){this.stopPropagation(evt);this.preventDefault(evt);},stopPropagation:function(evt){if(evt.stopPropagation){evt.stopPropagation();}else{evt.cancelBubble=true;}},preventDefault:function(evt){if(evt.preventDefault){evt.preventDefault();}else{evt.returnValue=false;}},target:function(_54){return this.resolveTextNode((_54.target)?_54.target:_54.srcElement);},currentTarget:function(_55){return this.resolveTextNode((_55.currentTarget)?_55.currentTarget:_55.srcElement);},relatedTarget:function(_56){if(_56.relatedTarget){return this.resolveTextNode(_56.relatedTarget);}else{if((_56.type=="mouseover")&&_56.fromElement){return this.resolveTextNode(_56.fromElement);}else{if((_56.type=="mouseout")&&_56.toElement){return this.resolveTextNode(_56.toElement);}else{return null;}}}},resolveTextNode:function(_57){return(_57&&(_57.nodeType==3))?_57.parentNode:_57;},cleanUp:function(){var _58=this.nativeEventHandlers;for(var id in _58){for(var ev in _58[id]){if(ev!="unload"){this.detachHandlers(id,ev,true);}}}},getId:function(src,id){var _5d=id;if(_5d===null||!_5d){_5d=src.id;}
return _5d;},getBodyId:function(src,id){var _60=this.getId(src,id);if(!_60||src==document.body){_60="body";}
return _60;},unload:function(src,_62){return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"unload",_62));},change:function(src,_64){return this.process(this.getId(src),new vjo.dsf.Event(src,"change",_64));},submit:function(src,_66){return this.process(this.getId(src),new vjo.dsf.Event(src,"submit",_66));},reset:function(src,_68){return this.process(this.getId(src),new vjo.dsf.Event(src,"reset",_68));},select:function(src,_6a){return this.process(this.getId(src),new vjo.dsf.Event(src,"select",_6a));},blur:function(src,_6c){return this.process(this.getId(src),new vjo.dsf.Event(src,"blur",_6c));},focus:function(src,_6e){return this.process(this.getId(src),new vjo.dsf.Event(src,"focus",_6e));},keydown:function(src,_70){return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"keydown",_70));},keypress:function(src,_72){return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"keypress",_72));},keyup:function(src,_74){return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"keyup",_74));},click:function(src,_76){return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"click",_76));},dblclick:function(src,_78){return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"dblclick",_78));},mousedown:function(src,_7a){return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"mousedown",_7a));},mousemove:function(src,_7c){return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"mousemove",_7c));},mouseout:function(src,_7e){return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"mouseout",_7e));},mouseover:function(src,_80){return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"mouseover",_80));},mouseup:function(src,_82){return this.process(this.getBodyId(src),new vjo.dsf.Event(src,"mouseup",_82));},load:function(src,_84){return this.run(src,_84,"load");},run:function(src,_86,_87){var id=this.getBodyId(src);var _89=new vjo.dsf.Event(src,_87,_86);var rv=this.process(id,_89);if(_87==="load"&&id==="body"){this.unregister("body","load");}
return rv;}}).inits(function(){vjo.dsf.EventDispatcher.addEventListener(window,"load",function(){vjo.dsf.EventDispatcher.addEventListener(window,"unload",function(){vjo.dsf.EventDispatcher.cleanUp();});});}).endType();

if(!this.JSON){JSON=function(){function f(n){return n<10?'0'+n:n;}
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

vjo.ctype("vjo.dsf.XDomainRequest").needs("vjo.dsf.EventDispatcher").protos({constructs:function(){this.callbacks=[];this.sCallbackName="callback";this.sPreId="xdr_";this.sPreExtId=this.sPreId+"ext_";this.iCount=0;vjo.dsf.EventDispatcher.addEventListener(window,"load",this.onLoad,this);},onLoad:function(){this.bodyLoaded=true;},getReqDiv:function(){return document.getElementsByTagName(this.bodyLoaded?"body":"head")[0];},send:function(_1){if(!document.createElement||!_1){return;}
var _2="",eid="",rdm="_vrdm="+(new Date()).getTime();if(typeof _1=="string"){_2=_1;eid=this.sPreExtId+this.iCount++;}else{if(_1.objType=="dsf_Message"&&_1.svcConfig){var cb=this.createCallback(_1);eid=this.sPreId+this.callbacks[this.callbacks.length-1];_2=_1.svcConfig.url+"&callback="+cb;}}
var _4=null,doc;doc=document;var _5=this.createElement("script");_5.id=eid;_5.type="text/javascript";var _6=true;if(arguments.length>1){_6=arguments[1];}
if(_6){_2=_2+((_2.indexOf("?")==-1)?"?":"&")+rdm;}
_5.src=_2;this.getReqDiv().appendChild(_5);return eid;},createCallback:function(_7){var _8=this.callbacks.length,name=this.sCallbackName+_8,eid=this.sPreId+name;this.callbacks[_8]=name;this[name]=function(_9){var _a;try{_a=_9;}
catch(e){_a=new vjo.dsf.ServiceResponse();var _b=new vjo.dsf.Error();_b.id="SYS.JSON_PARSE_ERROR";_b.message="SYS.JSON_PARSE_ERROR";_a.errors=[_b];}
this.loaded(eid);_7.response=_a;vjo.dsf.ServiceEngine.handleResponse(_7);};var rv="vjo.dsf.XDomainRequest."+name;return rv;},loaded:function(_d){var e=document.getElementById(_d);if(e!=null){e.parentNode.removeChild(e);}},createElement:function(_f){return(typeof(createElementV4)!="undefined")?createElementV4(_f):document.createElement(_f);}}).inits(function(){vjo.dsf.XDomainRequest=new vjo.dsf.XDomainRequest();}).endType();

vjo.ctype("vjo.dsf.RemoteReqtHdl").needs("vjo.dsf.XDomainRequest").protos({constructs:function(){this.reqTimers={};this.timerCount=0;this.processed={};},handleRequest:function(_1){_1.trace=_1.trace+"-->RemoteHdl_"+_1.svcId;if(_1.svcConfig){this.invoke(_1);}},invoke:function(_2){var _3=vjo.dsf.Service,xmlHttpReq=_3.getXmlHttpReq(),requestUrl=this.appendPort80ForSafari(document.location.href,_2.svcConfig.url),cfg=_2.svcConfig;_2.status=-1;if(cfg.respMarshalling=="JSCALLBACK"){vjo.dsf.XDomainRequest.send(_2);return;}
try{var _4=(cfg.async===false)?false:true;xmlHttpReq.open(cfg.method,requestUrl,_4);}
catch(e){var _5=new vjo.dsf.ServiceResponse();var _6=new vjo.dsf.Error();_6.id="SYS.DARWIN_SERVICE_PROTOCOL_ERROR";_6.message="SYS.PROTOCOL_ERROR: Cannot open URL '"+requestUrl+"'";_5.errors=[_6];_2.response=_5;vjo.dsf.ServiceEngine.handleResponse(_2);return;}
var _7=this.timerCount++;this.setupReadyState(xmlHttpReq,_2,_7);if(cfg.method=="POST"){xmlHttpReq.setRequestHeader("Content-Type","application/x-www-form-urlencoded");xmlHttpReq.setRequestHeader("Content-Length",_2.rawRequest.length);xmlHttpReq.send(_2.rawRequest);}else{xmlHttpReq.send(null);}
if(cfg.timeout){var _8=this;this.reqTimers[_7]=window.setTimeout(function(){_8.timeout(xmlHttpReq,_2,_7);},cfg.timeout);}},setupReadyState:function(_9,_a,_b){var _c=this;_9.onreadystatechange=function(){if(_9.readyState!=4){return;}
if(_c.processed[_b]){return;}
_c.processed[_b]=true;var _d=_c.reqTimers[_b];if(_d){window.clearTimeout(_d);delete _c.reqTimers[_b];}
vjo.dsf.Service.callback(_9,_a);};},timeout:function(_e,_f,idx){if(this.processed[idx]){return;}
this.processed[idx]=true;delete _e.onreadystatechange;_e.abort();delete this.reqTimers[idx];var _11=new vjo.dsf.ServiceResponse();var _12=new vjo.dsf.Error();_12.id="SYS.DARWIN_SERVICE_PROTOCOL_ERROR";_12.message="SYS.PROTOCOL_ERROR: Service timed out.";_11.errors=[_12];_f.response=_11;vjo.dsf.ServiceEngine.handleResponse(_f);},appendPort80ForSafari:function(_13,_14){try{if(navigator.userAgent.toLowerCase().indexOf("safari")>=0){var _15="(([^:]*)://([^:/?]*)(:([0-9]+))?)?([^?#]*)([?]([^#]*))?(#(.*))?",ajaxUrl=_14,safariIssuePort="80";var _16=_13.match(_15);if(_16&&_16.length>=5&&_16[5]&&_16[5]==safariIssuePort){var _17=ajaxUrl.match(_15);if(_17&&_17[2]&&_17[2].length>0){if(_17[5]&&_17[5].length>0){}else{var _18="";if(_17[2]){_18+=_17[2]+"://";}
if(_17[3]){_18+=_17[3]+":"+safariIssuePort;}
if(_17[6]){_18+=_17[6];}
if(_17[8]){_18+="?"+_17[8];}
if(_17[10]){_18+="#"+_17[10];}
_14=_18;}}}}}
catch(e){}
return _14;}}).endType();

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

vjo.ctype("vjo.dsf.Message").protos({constructs:function(_1){this.objType="dsf_Message";this.svcId=_1;this.request;this.response;this.rawRequest="";this.clientContext={};this.trspType="InProc";this.status;this.svcConfig;this.returnData=true;this.trace="";this.v="0";}}).endType();

vjo.ctype("vjo.util.Map").protos({constructs:function(_1){this.javaClass=_1||"java.util.HashMap";this.map={};},get:function(_2){return this.map[_2];},put:function(_3,_4){return this.map[_3]=_4;},remove:function(_5){var _6=this.map[_5];delete this.map[_5];return _6;},size:function(){var _7=0;for(var _8 in this.map){_7++;}
return _7;}}).endType();

vjo.ctype("vjo.util.List").protos({constructs:function(_1){this.javaClass=_1||"java.util.ArrayList";this.list=[];},get:function(_2){if(this.size()>_2){return this.list[_2];}
return null;},add:function(_3){return this.list[this.size()]=_3;},remove:function(_4){var _5=this.size(),tmp=this.list,nlist=[],rv=false;for(var i=0;i<_5;i++){if(!rv&&tmp[i]===_4){rv=true;}else{nlist[nlist.length]=tmp[i];}}
this.list=nlist;return rv;},size:function(){return this.list.length;}}).endType();

vjo.needs("vjo.dsf.Json");vjo.ctype("vjo.dsf.ServiceEngine").needs(["vjo.dsf.RemoteReqtHdl","vjo.dsf.InProcReqtHdl","vjo.dsf.Service","vjo.dsf.ServiceResponse","vjo.dsf.SvcConfig","vjo.dsf.Message","vjo.util.Map","vjo.util.List"]).singleton().protos({STATUS:{ABORT:-1,JUMP:1},constructs:function(){this.svcReqtHdls={};this.svcRespHdls={};this.glbReqtHdls=[];this.glbRespHdls=[];this.trspReqtHdls={};this.trspRespHdls={};this.svcHdls={};this.inProcHdl=new vjo.dsf.InProcReqtHdl();this.remoteHdl=new vjo.dsf.RemoteReqtHdl();this.registerTrspReqtHdl("Remote",this.remoteTrspHdl);this.registerTrspRespHdl("Remote",this.remoteRespTrspHdl);},handleRequest:function(_1){var _2;if(_1.status!=this.STATUS.ABORT&&typeof _2=="undefined"){_2=this.processServiceRequestHandlers(_1);}
if(_1.status!=this.STATUS.ABORT&&typeof _2=="undefined"){_2=this.processGlobalRequestHandlers(_1);}
if(_1.status!=this.STATUS.ABORT&&typeof _2=="undefined"){_2=this.processTransportHandlers(_1);}
if(_1.status!=this.STATUS.ABORT&&(_1.trspType!="Remote"||typeof _2!="undefined")){this.handleResponse(_1,_2);}
return _1.returnData;},processServiceRequestHandlers:function(_3){var _4;var _5=this.svcReqtHdls[_3.svcId];if(_5){try{var _6=_5.length;for(var i=0;i<_6;i++){_3.trace=_3.trace+"-->svcReqtHdl_"+i;_5[i].handleRequest(_3);if(_3.status==this.STATUS.JUMP){_4="SVC";this.genResponseError(_3,"SYS.SVC_REQUEST_ERROR","SYS.SVC_REQUEST_ERROR");break;}}}
catch(e){_4="SVC";this.genResponseError(_3,"SYS.SVC_REQUEST_ERROR","SYS.SVC_REQUEST_ERROR");}}
return _4;},processGlobalRequestHandlers:function(_8){var _9;if(_8.status!=this.STATUS.JUMP){try{var _a=this.glbReqtHdls.length;for(var i=0;i<_a;i++){_8.trace=_8.trace+"-->glbReqtHdl_"+i;this.glbReqtHdls[i].handleRequest(_8);if(_8.status==this.STATUS.JUMP){_9="GLB";this.genResponseError(_8,"SYS.GLOBAL_REQUEST_ERROR","SYS.GLOBAL_REQUEST_ERROR");break;}}}
catch(e){_9="GLB";this.genResponseError(_8,"SYS.GLOBAL_REQUEST_ERROR","SYS.GLOBAL_REQUEST_ERROR");}}
return _9;},processTransportHandlers:function(_c){var _d;if(_c.status!=this.STATUS.JUMP&&_c.trspType){var _e=this.trspReqtHdls[_c.trspType];if(_e){try{var _f=_e.length;for(var i=0;i<_f;i++){_c.trace=_c.trace+"-->trspReqtHdl_"+i;_e[i].handleRequest(_c);if(_c.status==this.STATUS.JUMP){this.genResponseError(_c,"SYS.TRANS_REQUEST_ERROR","SYS.TRANS_REQUEST_ERROR");break;}}}
catch(e){this.genResponseError(_c,"SYS.TRANS_REQUEST_ERROR","SYS.TRANS_REQUEST_ERROR");}}
if(_c.status!=this.STATUS.JUMP&&_c.status!=this.STATUS.ABORT){if(_c.trspType==="Remote"){this.remoteHdl.handleRequest(_c);}else{this.inProcHdl.handleRequest(_c);}}}
return _d;},handleResponse:function(_11,_12){if(_11.trspType&&typeof _12=="undefined"){this.processTransResponseHandlers(_11);}
if(_12!="SVC"){this.processGlobalResponseHandlers(_11);}
this.processServiceResponseHandlers(_11);},processTransResponseHandlers:function(msg){var _14=this.trspRespHdls[msg.trspType];try{if(_14){for(var i=_14.length-1;i>=0;i--){msg.trace=msg.trace+"-->trspRespHdl_"+i;_14[i].handleResponse(msg);}}}
catch(e){this.genResponseError(msg,"SYS.TRANS_RESPONSE_ERROR","SYS.TRANS_RESPONSE_ERROR");}},processGlobalResponseHandlers:function(msg){try{for(var i=this.glbRespHdls.length-1;i>=0;i--){msg.trace=msg.trace+"-->glbRespHdl_"+i;this.glbRespHdls[i].handleResponse(msg);}}
catch(e){this.genResponseError(msg,"SYS.GLOB_RESPONSE_ERROR","SYS.GLOB_RESPONSE_ERROR");}},processServiceResponseHandlers:function(msg){var _19;if(msg.clientContext){_19=msg.clientContext.svcApplier;}
try{if(_19){if(typeof _19.onResponse=="function"){_19.onResponse(msg);}else{if(typeof _19=="function"){_19(msg);}}}}
catch(e){this.genResponseError(msg,"SYS.SVC_RESPONSE_ERROR","SYS.SVC_RESPONSE_ERROR");}
var _1a=this.svcRespHdls[msg.svcId];if(_1a){try{for(var i=_1a.length-1;i>=0;i--){msg.trace=msg.trace+"-->svcRespHdl_"+i;_1a[i].handleResponse(msg);}}
catch(e){this.genResponseError(msg,"SYS.SVC_RESPONSE_ERROR","SYS.SVC_RESPONSE_ERROR");}}},createHandler:function(_1c,_1d){if(typeof _1c[_1d]!="function"){if(typeof _1c=="function"){var _1e=_1c,obj={},self=this;obj[_1d]=function(){return _1e.apply(self,arguments);};_1c=obj;}}
return _1c;},registerSvcHdl:function(_1f,_20){if(!_1f||!_20){return;}
_20=this.createHandler(_20,"invoke");this.inProcHdl.registerSvcHdl(_1f,_20);},getSvcHdl:function(_21){return this.inProcHdl.getSvcHdl(_21);},registerSvcReqtHdl:function(_22,_23){if(!_22||!_23){return;}
if(typeof this.svcReqtHdls[_22]=="undefined"){this.svcReqtHdls[_22]=[];}
var _24=this.svcReqtHdls[_22];_24[_24.length]=this.createHandler(_23,"handleRequest");},registerSvcRespHdl:function(_25,_26){if(!_25||!_26){return;}
if(typeof this.svcRespHdls[_25]=="undefined"){this.svcRespHdls[_25]=[];}
var _27=this.svcRespHdls[_25];_27[_27.length]=this.createHandler(_26,"handleResponse");},registerGlbReqtHdl:function(_28){if(!_28){return;}
this.glbReqtHdls[this.glbReqtHdls.length]=this.createHandler(_28,"handleRequest");},registerGlbRespHdl:function(_29){if(!_29){return;}
this.glbRespHdls[this.glbRespHdls.length]=this.createHandler(_29,"handleResponse");},registerTrspReqtHdl:function(_2a,_2b){if(!_2a||!_2b){return;}
if(typeof this.trspReqtHdls[_2a]=="undefined"){this.trspReqtHdls[_2a]=[];}
var _2c=this.trspReqtHdls[_2a];_2c[_2c.length]=this.createHandler(_2b,"handleRequest");},registerTrspRespHdl:function(_2d,_2e){if(!_2d||!_2e){return;}
if(typeof this.trspRespHdls[_2d]=="undefined"){this.trspRespHdls[_2d]=[];}
var _2f=this.trspRespHdls[_2d];_2f[_2f.length]=this.createHandler(_2e,"handleResponse");},remoteTrspHdl:function(_30){var cfg=_30.svcConfig;if(!cfg||cfg.objType!="dsf_SvcConfig"){return;}else{if(cfg.respMarshalling=="JSCALLBACK"){if(typeof vjo.dsf.assembly!="undefined"&&typeof vjo.dsf.assembly.VjClientAssembler!="undefined"&&!vjo.dsf.assembly.VjClientAssembler.bBodyLoaded){vjo.dsf.assembly.VjClientAssembler.load(_30);_30.status=-1;return;}}}
if(_30.request&&_30.request.javaClass){delete _30.request.b;}
var svc=vjo.dsf.Service,requestParams=svc.generateReqParams(_30),requestUrl=cfg.url;if(_30.svcConfig.method=="GET"){requestUrl=requestUrl+"?"+requestParams;}else{_30.rawRequest=requestParams;}
_30.svcConfig.url=requestUrl;},remoteRespTrspHdl:function(_33){var _34=_33.response;if(_34!=null&&_34.data!=null){this.processData(_34.data);}},processData:function(_35){this.processObj(_35);for(var _36 in _35){var o=_35[_36];if(o!=null&&typeof o=="object"){this.processData(o);}}},processObj:function(obj){var _39=obj.javaClass;if(_39&&_39.length>0){if(/java.util.([^\s])*List/.test(_39)){this.addMethods(obj,vjo.util.List.prototype);}else{if(/java.util.([^\s])*Map/.test(_39)){this.addMethods(obj,vjo.util.Map.prototype);}}}},addMethods:function(obj,_3b){for(var key in _3b){obj[key]=_3b[key];}},genResponseError:function(msg,_3e,_3f){if(typeof msg.response=="undefined"){var _40=new vjo.dsf.ServiceResponse();msg.response=_40;}
var _41=new vjo.dsf.Error(_3e,_3f);msg.response.errors[msg.response.errors.length]=_41;},register:function(_42,_43,_44){var _s=vjo.dsf.ServiceEngine;switch(_42){case 0:_s.registerSvcHdl(_43,_44);break;case 1:_s.registerSvcReqtHdl(_43,_44);break;case 2:_s.registerGlbReqtHdl(_43,_44);break;case 3:_s.registerTrspReqtHdl(_43,_44);break;case 4:_s.registerSvcRespHdl(_43,_44);break;case 5:_s.registerGlbRespHdl(_43,_44);break;case 6:_s.registerTrspRespHdl(_43,_44);break;}}}).inits(function(){vjo.dsf.ServiceEngine=new vjo.dsf.ServiceEngine();}).endType();

vjo.ctype("vjo.dsf.assembly.VjClientAssemblerRequest").protos({constructs:function(_1,_2,_3,_4,_5){this.sUrl=_1;this.fCallback=_2;this.oScope=_3||window;this.sCallbackParam=_4;var b=_5;if(typeof(b)=="undefined"){b=true;}
this.bSendResponseOnLoad=b;}}).endType();

vjo.ctype("vjo.dsf.assembly.VjClientAssembler").needs(["vjo.dsf.EventDispatcher","vjo.dsf.ServiceEngine","vjo.dsf.assembly.VjClientAssemblerRequest"]).singleton().protos({constructs:function(){this.aCallbacks=[];this.aResponses={};this.aModels={};this.sPreCallback="_callback";this.bBodyLoaded=false;this.bLock=false;this.loaded={};},load:function(_1){var m=_1,url=m.sUrl||"";var _3=this.generateCallback(m);var cb="vjo.dsf.assembly.VjClientAssembler."+_3;if(m&&m.objType=="dsf_Message"){url=m.svcConfig.url+"?";url+=vjo.dsf.Service.generateReqParams(m)+"&callback="+cb;}else{if(m.sCallbackParam){url=url+"&"+m.sCallbackParam+"="+cb;}}
return vjo.dsf.XDomainRequest.send(url);},generateCallback:function(_5){var m=_5;var _7=this.aCallbacks.length;var _8=this.sPreCallback+_7;this.aCallbacks[_7]=_8;this.aModels[_8]=m;this[_8]=function(){if(this.loaded[_8]===true){return;}
this.loaded[_8]=true;if(this.bBodyLoaded||!m.bSendResponseOnLoad){if(m.objType=="dsf_Message"){m.response=this.getResponse(arguments[0]);vjo.dsf.ServiceEngine.handleResponse(m);}else{m.fCallback.apply(m.oScope,arguments);}}else{if(m.objType=="dsf_Message"){this.aResponses[_8]=this.getResponse(arguments[0]);m.response=this.aResponses[_8];}else{this.aResponses[_8]=arguments;}
if(this.bBodyLoaded){this.assemble();}}};return _8;},assemble:function(){this.bBodyLoaded=true;if(this.bLock){setTimeout("vjo.dsf.assembly.VjClientAssembler.assemble()",1000);return;}
this.bLock=true;try{for(var _9 in this.aResponses){this.loaded[_9]=true;var m=this.aModels[_9];if(this.aResponses[_9]!=null){if(m.objType=="dsf_Message"){vjo.dsf.ServiceEngine.handleResponse(m);}else{m.fCallback.apply(m.oScope,this.aResponses[_9]);}}
this.aResponses[_9]=null;}}
finally{this.bLock=false;}},getResponse:function(_b){var _c;try{_c=_b;}
catch(e){_c=new vjo.dsf.ServiceResponse();var _d=new vjo.dsf.Error();_d.id="SYS.JSON_PARSE_ERROR";_d.message="SYS.JSON_PARSE_ERROR";_c.errors=[_d];}
return _c;}}).inits(function(){vjo.dsf.assembly.VjClientAssembler=new vjo.dsf.assembly.VjClientAssembler();vjo.dsf.EventDispatcher.addEventListener(window,"load",function(){vjo.dsf.assembly.VjClientAssembler.assemble();});}).endType();

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

vjo.ctype("vjo.darwin.core.sitespeed.SiteSpeed").needs(["vjo.dsf.EventDispatcher","vjo.dsf.cookie.VjCookieJar"]).props({gauge:function(){this.gaugeInternal("");},gaugeUnload:function(){this.gaugeInternal("&ex2=1");},gaugeInternal:function(_1){var _2=vjo.dsf.cookie.VjCookieJar,sbf=_2.readCookie("ebay","sbf");_2.writeCookielet("ebay","sbf",_2.setBitFlag(sbf,20,1));if(typeof(oGaugeInfo)!="undefined"&&oGaugeInfo.bFlag!=1){if(oGaugeInfo.sent===true){return;}
oGaugeInfo.sent=true;var _3=new Image(1,1),delta=(new Date()).getTime()-oGaugeInfo.iST;if(_3){_3.src=oGaugeInfo.sUrl.replace(/&amp;/g,"&")+delta+_1;}}}}).inits(function(){var _4=vjo.dsf.cookie.VjCookieJar;sbf=_4.readCookie("ebay","sbf");b=(sbf)?_4.getBitFlag(sbf,20):0;if(typeof(oGaugeInfo)!="undefined"){oGaugeInfo.bFlag=b;oGaugeInfo.sent=false;}
var _5=vjo.darwin.core.sitespeed.SiteSpeed;vjo.dsf.EventDispatcher.addEventListener(window,"beforeunload",_5.gaugeUnload,_5);}).endType();

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

vjo.ctype("vjo.dsf.document.Form").props({get:function(_1){var f=document.forms[_1];return f?f:null;},getElem:function(_3,_4){var f=this.get(_3),e;if(f){e=f.elements[_4];}
return e?e:null;},submit:function(_6){var f=this.get(_6);if(f){f.submit();}},setAction:function(_8,_9){this.setAttr(_8,"action",_9);},getAction:function(_a){return this.getAttr(_a,"action");},setTarget:function(_b,_c){this.setAttr(_b,"target",_c);},getTarget:function(_d){return this.getAttr(_d,"target");},setAttr:function(_e,_f,_10){var f=this.get(_e);if(f){eval("f."+_f.toLowerCase()+"=psAttrValue;");}},getAttr:function(_12,_13){var f=this.get(_12),v=null;if(f){v=eval("f."+_13.toLowerCase());}
return v;}}).endType();

vjo.ctype("vjo.dsf.window.utils.VjWindow").props({open:function(_1,_2,_3,_4,_5,_6,_7){if(_5){var _8=(window.screen.width-_6)/2;var _9=(window.screen.height-_7)/2;_3+=",left="+_8+",top="+_9;}
return window.open(_1,_2,_3,_4);},location:function(_a){document.location.href=_a;},alert:function(_b){window.alert(_b);},confirm:function(_c){return window.confirm(_c);}}).endType();

vjo.ctype("vjo.darwin.component.common.actbutton.ActionButtonImgPreloader").needs("vjo.dsf.utils.Object").needs("vjo.dsf.client.Browser").needs("vjo.dsf.EventDispatcher").singleton().protos({constructs:function(){this.aLoaded=[];this.bPageLoaded=false;var _1=vjo.dsf.utils.Object.hitch(this,"preloadAll");vjo.dsf.EventDispatcher.add("body","load",_1);this.fixIE6ImgCaching();},preload:function(_2){if(!_2||!_2.length){return;}
var i=0;for(;i<_2.length;i++){if(!this.aLoaded[_2[i]]){if(!this.bPageLoaded){this.aLoaded[_2[i]]=true;}else{this.checkAndPreload(_2[i]);this.aLoaded[_2[i]]=true;}}}},preloadAll:function(){this.bPageLoaded=true;for(var _4 in this.aLoaded){this.checkAndPreload(_4);}},checkAndPreload:function(_5){if(typeof(_5)=="string"&&_5.match(/http.+/)){var _6=new Image();_6.src=_5;}},fixIE6ImgCaching:function(){var br=vjo.dsf.client.Browser;if(br.bIE&&br.iVer<7){var _8=function(){try{document.execCommand("BackgroundImageCache",false,true);}
catch(err){}};vjo.dsf.EventDispatcher.add("body","load",_8);}}}).inits(function(){vjo.darwin.component.common.actbutton.ActionButtonImgPreloader=new vjo.darwin.component.common.actbutton.ActionButtonImgPreloader();}).endType();

vjo.ctype("vjo.darwin.component.common.actbutton.ActionButton").needs("vjo.dsf.utils.Object").needs("vjo.dsf.utils.Handlers").needs("vjo.dsf.document.Element").needs("vjo.dsf.document.Form").needs("vjo.darwin.component.common.actbutton.ActionButtonImgPreloader").protos({constructs:function(_1,_2,_3,_4,_5,_6,_7,_8,_9,_a){var t=this;t.E=vjo.dsf.document.Element;t.H=vjo.dsf.utils.Handlers;t.sCompId=_1+"Anc";t.sAncId=_1+"Lnk";t.sSubId=_1+"Sub";t.sNoScriptId=_1+"NoScript";t.sFormName=_2;t.bAnonymousForm=_3;t.sUrl=_4;t.iState=_5;t.bShowClicked=_6;t.aStateClzes=_7;t.bForceSubmit=_9;t.fMouseMoveEvtHandler="undefined";if(t.bShowClicked){vjo.darwin.component.common.actbutton.ActionButtonImgPreloader.preload(_8);}
t.initMe(_a);},initMe:function(_c){var t=this,ancBtn=t.E.get(t.sCompId),noScript=t.E.get(t.sNoScriptId);if(noScript&&noScript.style){noScript.style.display="none";}
if(ancBtn){t.switchClz();}
if(_c){var _e=t.E.get(t.sAncId);if(t.bShowClicked){t.H.attachEvt(_e,"mousedown",t.onClick,t);t.H.attachEvt(_e,"mouseup",t.clicked,t);t.H.attachEvt(_e,"keypress",t.onEnter,t);}else{t.H.attachEvt(_e,"click",t.clicked,t);}}},enable:function(_f){this.iState=0;this.switchClz();return false;},monitorMouseOut:function(_10){var t=this;if(2!=t.iState){var _12={x:t.eventLeft(_10),y:t.eventTop(_10)};var _13=t.getAnchorPosition(t.sAncId);var _14={x:_13.x+t.E.get(t.sAncId).offsetWidth,y:_13.y+t.E.get(t.sAncId).offsetHeight};if(_12.x<_13.x||_12.x>_14.x||_12.y<_13.y||_12.y>_14.y){t.iState=0;t.switchClz();t.H.detachEvt(document.body,"mousemove",t.fMouseMoveEvtHandler);}
return false;}},isLeftClick:function(_15){if(window.event){if(window.event.button>1){return false;}}else{if(_15.nativeEvent){if(_15.nativeEvent.button>=1){return false;}}}
return true;},getAnchorPosition:function(_16){var _17=false;var _18={};var x=0,y=0;var _1a=false,use_css=false,use_layers=false;if(document.getElementById!=null){_1a=true;}else{if(document.all){use_css=true;}else{if(document.layers){use_layers=true;}}}
if(_1a&&document.all){x=this.getPageOffsetLeft(document.all[_16]);y=this.getPageOffsetTop(document.all[_16]);}else{if(_1a){var o=document.getElementById(_16);x=this.getPageOffsetLeft(o);y=this.getPageOffsetTop(o);}else{if(use_css){x=this.getPageOffsetLeft(document.all[_16]);y=this.getPageOffsetTop(document.all[_16]);}else{if(use_layers){var _1c=0;for(var i=0;i<document.anchors.length;i++){if(document.anchors[i].name==_16){_1c=1;break;}}
if(_1c===0){_18.x=0;_18.y=0;return _18;}
x=document.anchors[i].x;y=document.anchors[i].y;}else{_18.x=0;_18.y=0;return _18;}}}}
_18.x=x;_18.y=y;return _18;},getPageOffsetLeft:function(el){var ol=el.offsetLeft;while((el=el.offsetParent)!==null){ol+=el.offsetLeft;}
return ol;},getPageOffsetTop:function(el){var ot=el.offsetTop;while((el=el.offsetParent)!==null){ot+=el.offsetTop;}
return ot;},eventTop:function(_22){if(_22.pageY!=null){return _22.pageY;}else{if(document.documentElement&&document.documentElement.scrollTop){return _22.clientY+Math.max(document.documentElement.scrollTop,document.body.scrollTop);}else{return _22.clientY+document.body.scrollTop;}}},eventLeft:function(_23){if(_23.pageX!=null){return _23.pageX;}else{if(document.documentElement&&document.documentElement.scrollLeft){return _23.clientX+Math.max(document.documentElement.scrollLeft,document.body.scrollLeft);}else{return _23.clientX+document.body.scrollLeft;}}},onEnter:function(_24){if(2!=this.iState){if(_24.nativeEvent){if(_24.nativeEvent.keyCode==13){this.onClick(_24);return this.clicked(_24);}}}},onClick:function(_25){var t=this;if(!t.isLeftClick(_25)){return false;}
if(2!=t.iState){t.iState=1;if(t.bShowClicked){t.switchClz();t.fMouseMoveEvtHandler=t.H.attachEvt(document.body,"mousemove",t.monitorMouseOut,t);}}
return false;},clicked:function(_27){var t=this;if(!t.isLeftClick(_27)){return false;}
if(2!=t.iState&&t.bForceSubmit){t.iState=0;t.switchClz();if(t.bShowClicked){t.H.detachEvt(document.body,"mousemove",t.fMouseMoveEvtHandler);}
if(t.bAnonymousForm){var _29=t.E.get(t.sSubId);if(_29){_29.click();}}else{if(t.sFormName!="undefined"){var _2a=t.E.get(t.sFormName);if(typeof(window.external)!="undefined"&&typeof(window.external)!="unknown"&&window.external!=null&&typeof(window.external.AutoCompleteSaveForm)!="undefined"&&typeof(window.external.AutoCompleteSaveForm)!="unknown"){window.external.AutoCompleteSaveForm(_2a);}
vjo.dsf.document.Form.submit(t.sFormName);}else{if(t.sUrl){window.location=t.sUrl;}}}}
return false;},disable:function(_2b){if(2!=this.iState){this.iState=2;this.switchClz();}
return false;},switchClz:function(){var dom=vjo.dsf.document.Element;var _2d=dom.get(this.sCompId);if(_2d){_2d.className=this.aStateClzes[this.iState];this.displayCursor(dom.get(this.sAncId),"btnPointerCur");if(this.iState==0){_2d.style.opacity="";_2d.style.filter="";_2d.style.zoom="1";if(this.sFormName){}else{var anc=dom.get(this.sAncId);anc.href=this.sUrl;}}else{if(this.iState==2){_2d.style.opacity=".5";_2d.style.filter="alpha(opacity=50)";_2d.style.zoom="1";this.displayCursor(dom.get(this.sAncId),"btnDefCur");var btn=vjo.dsf.Element.get(this.sAncId);if(btn.tagName=="BUTTON"){btn.style.cursor="default";}
if(this.sFormName){}else{var anc=dom.get(this.sAncId);anc.href="#";}}}}},displayCursor:function(anc,_31){if(anc!==null){anc.className=_31;}}}).endType();

vjo.ctype("vjo.darwin.core.memberbadge.MemberBadge").needs(["vjo.dsf.document.Element","vjo.dsf.window.utils.VjWindow"]).props({openLink:function(_1){if(_1){var _2=_1;var _3=window.screen.width;vjo.dsf.window.utils.VjWindow.open(_1,"","width=440,height=500,resizable=yes,top=0,left="+(_3-450)+",location=no,menubar=no,scrollbars=no,status=no",false,false);return false;}}}).endType();

vjo.ctype("vjo.dsf.utils.Timer").needs("vjo.dsf.utils.Object").protos({timer:null,isRunning:false,interval:null,onTick:function(){},onStart:null,onStop:null,constructs:function(_1){this.interval=_1;},setInterval:function(ms){var t=this;if(t.isRunning){window.clearInterval(timer);}
t.interval=ms;if(t.isRunning){t.setInt();}},start:function(){var t=this;if(typeof t.onStart=="function"){t.onStart();}
t.isRunning=true;t.setInt();},stop:function(){var t=this;if(typeof t.onStop=="function"){t.onStop();}
t.isRunning=false;window.clearInterval(t.timer);},setInt:function(){var t=this;t.timer=window.setInterval(vjo.dsf.utils.Object.hitch(t,"onTick"),t.interval);}}).endType();

vjo.ctype("vjo.dsf.document.Shim").needs("vjo.dsf.client.Browser").props({add:function(_1,_2,_3){var f,p="px",w,h,s;if(this.check()){w=_1.offsetWidth;h=_1.offsetHeight;w+=_2?_2:0;h+=_3?_3:0;f=document.createElement("IFRAME");s=f.style;s.width=w+p;s.height=h+p;s.filter="chroma(color='white')";f.frameBorder=0;s.position="absolute";s.left="0"+p;s.top="0"+p;s.zIndex="-1";s.filter="Alpha(Opacity=\"0\")";if(document.location.protocol=="https:"){f.src="https://securepics.ebaystatic.com/aw/pics/s.gif";}
_1.appendChild(f);return f;}},remove:function(_5,_6){if(this.check()){if(_6&&_6.parentNode){_6.parentNode.removeChild(_6);}}},check:function(){var B=vjo.dsf.client.Browser;return(B.bIE||B.bFirefox);}}).endType();

vjo.ctype("vjo.dsf.document.Positioning").props({getScrollLeftTop:function(){var d=document,rv=[0,0],db=d.body,de=d.documentElement;if(db){rv[0]+=db.scrollLeft;rv[1]+=db.scrollTop;}
if(de){rv[0]+=de.scrollLeft;rv[1]+=de.scrollTop;}
return rv;},getOffsetLeft:function(_2){var e=_2,l=0;while(e){l+=e.offsetLeft;e=e.offsetParent;}
return l;},getOffsetTop:function(_4){var e=_4,t=0;while(e){t+=e.offsetTop;e=e.offsetParent;}
return t;},getClientWidth:function(){var s=self,d=document,de=d.documentElement,w;if(s.innerWidth){w=s.innerWidth;}else{if(de&&de.clientWidth){w=de.clientWidth;}else{w=d.body.clientWidth;}}
return w;},getClientHeight:function(){var s=self,d=document,de=d.documentElement,h;if(s.innerHeight){h=s.innerHeight;}else{if(de&&de.clientHeight){h=de.clientHeight;}else{h=d.body.clientHeight;}}
return h;},getEventLeftTop:function(_8){var u="undefined",evt=window.event||_8,xOff=(typeof(screenLeft)!=u)?screenLeft:screenX,yOff=(typeof(screenTop)!=u)?screenTop:(screenY+(outerHeight-innerHeight)-25);return[evt.screenX-xOff,evt.screenY-yOff];}}).endType();

vjo.ctype("vjo.darwin.app.common.mask.MaskHandlers").props({openService:function(_1,_2){var _3=vjo.Registry.get(_1);if(_3){_3.show(_2.clientContext);}},closeService:function(_4,_5){var _6=vjo.Registry.get(_4);if(_6){_6.hide();}}}).endType();

vjo.ctype("vjo.darwin.app.common.mask.Mask").needs("vjo.dsf.window.utils.VjWindowUtils").needs("vjo.dsf.EventDispatcher").needs("vjo.dsf.utils.Timer").needs("vjo.darwin.app.common.mask.MaskHandlers").needs("vjo.dsf.client.Browser").protos({constructs:function(_1,_2,_3,_4){var t=this;t.eElem=document.getElementById(_1);t.iOpacity=(_2)?parseFloat(_2):0;t.showing=false;t.zIndex=100;t.sIsActivated=false;t.oTimer=false;t.ED=vjo.dsf.EventDispatcher;t.pngalpha=false;t.pngnorm=false;t.strExt=".gif";t.B=vjo.dsf.client.Browser;t.png=_3;t.gif=_4;if(t.B.bIE&&t.B.bWin&&t.B.iVer>5&&t.B.iVer<=6){t.pngalpha=true;t.strExt=".png";}else{if((t.B.bFirefox)||(t.B.bSafari)||(t.B.bNav)||(t.B.bIE&&t.B.bMac)||(t.B.bOpera&&t.B.bWin&&t.B.iVer>=6)||(t.B.bOpera&&t.B.iVer>=6)||(t.B.bOpera&&t.B.bMac&&t.B.iVer>=5)||(t.B.bWebTV)||(t.B.bIE&&t.B.bWin&&t.B.iVer>6)){t.pngnorm=true;t.strExt=".png";}}},show:function(_6){var t=this;if(t.showing){return;}else{t.showing=true;}
t.formName=(_6)?_6.formName:"";t.zIndex=(_6&&_6.overlayPanelIndex)?_6.overlayPanelIndex:t.zIndex;t.render();t.disableSelects();t.startResizeListener();},render:function(){var t=this;t.VjWindowUtils=vjo.dsf.window.utils.VjWindowUtils;var _9=t.VjWindowUtils.scrollWidth();var _a=t.VjWindowUtils.scrollHeight();t.setStyle("display","block");t.setStyle("width",t.VjWindowUtils.toPixels(_9));t.setStyle("height",t.VjWindowUtils.toPixels(_a));t.setStyle("zIndex",t.zIndex);if(t.pngalpha){t.setStyle("background","transparent");t.setStyle("filter","progid:DXImageTransform.Microsoft.AlphaImageLoader(src="+t.png+", sizingMethod='scale')");try{t.eElem.filters["DXImageTransform.Microsoft.AlphaImageLoader"].enabled=true;}
catch(e){t.setDefaultBg();t.setStyle("filter","");}}else{if(t.pngnorm){t.setStyle("background","transparent url("+t.png+") repeat left top");}else{t.setDefaultBg();}}},setDefaultBg:function(){this.setStyle("background","transparent url("+this.gif+") repeat left top");},hide:function(){var t=this;if(!t.showing){return;}else{t.showing=false;}
t.setStyle("display","none");t.setStyle("width",t.VjWindowUtils.toPixels(0));t.setStyle("height",t.VjWindowUtils.toPixels(0));t.enableSelects();t.stopResizeListener();},disableSelects:function(){this.disabledSelects=new Array();var _c=document.body.getElementsByTagName("select");for(var _d=0;(_d<_c.length);_d++){var _e=_c[_d];if(_e.disabled){continue;}
var _f=(_e.form)?_e.form.name:null;if(_f==this.formName){continue;}
_e.disabled=true;this.disabledSelects.push(_e);}},enableSelects:function(){var _10=this.disabledSelects;for(var idx=0;(idx<_10.length);idx++){_10[idx].disabled=false;}},setStyle:function(_12,_13){var e=this.eElem;if(!e||!e.style){return;}
e.style[_12]=_13;},requireResize:function(){this.sIsActivated=true;},startResizeListener:function(){var t=this;t.ED.addEventListener(window,"resize",t.requireResize,t);if(!t.oTimer){t.oTimer=new vjo.dsf.utils.Timer();t.oTimer.interval=500;t.oTimer.onTick=function(){if(t.sIsActivated){t.render();t.sIsActivated=false;}};}
t.oTimer.start();},stopResizeListener:function(){var t=this;t.ED.removeEventListener(window,"resize",t.requireResize,t);if(t.oTimer){t.oTimer.stop();}}}).endType();

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

vjo.ctype("vjo.dsf.client.ActiveX").needs("vjo.dsf.client.Browser").props({init:function(){var oC=vjo.dsf.client.Browser;if(oC.bIE){var d=document;dw=function(s){d.writeln(s);};dw("<scr"+"ipt language=\"vbscript\" type=\"text/vbscript\">");dw("\tFunction vbCheckActiveXControl (pActXName)");dw("\t\taX = false");dw("\t\ton error resume next");dw("\t\taX = IsObject(CreateObject(pActXName))");dw("\t\tvbCheckActiveXControl = aX");dw("End Function");dw("</scr"+"ipt>");}},isLibLoaded:function(_4){var oC=vjo.dsf.client.Browser;return oC.bActiveXSupported&&vbCheckActiveXControl(_4);}}).inits(function(){vjo.dsf.client.ActiveX.init();}).endType();

vjo.ctype("vjo.darwin.core.rtm.RTMHelper").props({setRtm:function(_1){this.oRtm=_1;},render:function(_2){var t=this;if(t.oRtm){t.oRtm.renderPromos(_2);}}}).endType();

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
return _5;}});

vjo.ctype("vjo.dsf.flash.Version").needs(["vjo.dsf.client.Browser","vjo.dsf.client.ActiveX","vjo.dsf.utils.Bit","vjo.dsf.cookie.VjCookieJar"]).props({versions:[10,9],get:function(){var t=this,B=t.vj$.Browser,v=0,vs=t.versions,i,A=t.vj$.ActiveX,cv;cv=t.rw(false);if(cv){return(cv==1)?0:cv;}
if(B.bIE&&B.bWin&&!B.bOpera){for(i=0;i<vs.length;i++){if(A.isLibLoaded("ShockwaveFlash.ShockwaveFlash."+vs[i])){v=vs[i];break;}}}else{var n=navigator,pd,id,swf="Shockwave Flash";if(n.plugins[swf]){pd=n.plugins[swf].description;id=pd.indexOf("Flash")+5;v=parseInt(pd.substr(id,pd.length));}
if(B.bWebTV){v=3;}}
t.rw(true,v);return v;},rw:function(_3,_4){var t=this,n=t.vj$,C=n.VjCookieJar,B=n.Bit;cl=C.readCookie("ebay","sbf");if(!_3){return B.getMulti(cl,40,5);}else{if(_3){_4=(_4==0)?1:_4;C.writeCookielet("ebay","sbf",B.setMulti(cl,40,5,_4));}}}}).endType();

vjo.ctype("vjo.darwin.core.rtm.RTM").needs(["vjo.dsf.typeextensions.string.Comparison","vjo.dsf.cookie.VjCookieJar","vjo.dsf.client.Browser","vjo.dsf.client.ActiveX","vjo.dsf.EventDispatcher","vjo.dsf.flash.Version","vjo.darwin.core.rtm.RTMInit"]).protos({constructs:function(_1){this.oJSBean=_1||{};this.oJSBean.quickPids=this.oJSBean.quickPids||[];this.aContent=[];this.iContentLen=0;this.bTimedOut=false;this.oTimeoutId=null;this.iTIMEOUT=3000;this.oClient=vjo.dsf.client.Browser;this.bResponseReturned=false;this.iOrd=(new Date()).getTime();this.aGlobalNavPlacements=null;this.promos=new Object();this.globalAdOverlay=null;vjo.Registry.put("GlobalRtmInstance",this);this.response=null;this.renderedStatus={};this.bBodyLoaded=false;this.bOptimize=true;if(this.bOptimize){vjo.dsf.EventDispatcher.add("body","load",this.render,this);}},getFlashVersion:function(){return this.vj$.Version.get();},getEncodingType:function(){var _2=null;if(typeof(_GlobalNavHeaderUtf8Encoding)!="undefined"){_2=_GlobalNavHeaderUtf8Encoding?"UTF-8":"cp1252";}
return _2;},setInlineContent:function(_3){this.bResponseReturned=true;this.visiblePromos=new Array();if(this.bTimedOut){return;}
if(this.oTimeoutId){window.clearTimeout(this.oTimeoutId);}
this.hidePromos(this.promos);this.aContent=_3;this.iContentLen=this.aContent.length;var _4=false;this.bPromoScript=false;this.bPromoStyle=false;for(var i=0;i<this.iContentLen;i++){with(this){var _6=aContent[i];if(_6.JSMetaData){this.processJS(_6.JSMetaData);}else{if(_6.CSSMetaData){this.processCSS(_6.CSSMetaData);}else{this.processPromo(_6);}}}}
var _7=new vjo.dsf.Message("RTM_COMPLETE");_7.status=1;_7.vjRTMObject=this;vjo.dsf.ServiceEngine.handleRequest(_7);if(_4){var _7=new vjo.dsf.Message("PROMO_COMPLETE");_7.vjPromoClientObject=this;vjo.dsf.ServiceEngine.handleRequest(_7);}
return;},processJS:function(_8){if(_8.JSURLs){var _9=_8.JSURLs;for(var i=0;i<_9.length;i++){var _b=document.createElement("script");_b.type="text/javascript";_b.src=_9[i];document.getElementsByTagName("head")[0].appendChild(_b);}}
if(_8.EventHandlers){var _c=_8.EventHandlers;for(var i=0;i<_c.length;i++){var _b=document.createElement("script");_b.text=_c[i];document.getElementsByTagName("head")[0].appendChild(_b);}}},processCSS:function(_d){if(_d.CSSURLs){var _e=_d.CSSURLs;for(var i=0;i<_e.length;i++){var _10=document.createElement("link");_10.rel="stylesheet";_10.type="text/css";_10.href=_e[i];document.getElementsByTagName("head")[0].appendChild(_10);}}},processPromoResponse:function(_11){var map=_11.content.data.map;var _13=0;var id="-1";var _15=false;for(var _16 in map){var _17=_16.match(/ME([0-9]+)(.*)/);if(_17){_13++;id=_11.id;}}
var _18=map.VisualPreview;if(_18){this.loadVisualPreview(map.VisualPreview);}
var _19=map.Style;if(_19&&!this.bPromoStyle){this.loadStyle(_19);this.bPromoStyle=true;}
for(var _16 in map){var _17=_16.match(/ME([0-9]+)(.*)/);if(_17){promoLoaded=this.loadPromo(_11.id,_17[2],map[_16],_13>1);if(promoLoaded){_15=true;}}}
var _1a=map.Script;if(_1a&&!this.bPromoScript){this.bPromoScript=this.loadScript(_1a);}
if(_15){var _1b=this.getIndex(id);var _1c=this.getUIElement(this.oJSBean.htmlIds[_1b]);if(_1c&&!_1c.length){_1c.style.display="block";}}else{this.processNoneAd(_11);}
return;},loadVisualPreview:function(_1d){var _1e=this.getUIElement("VisualPreviewContent");if(_1e==null){_1e=document.createElement("div");_1e.name="VisualPreviewContent";document.body.appendChild(_1e);}
_1e.innerHTML=_1d;},loadStyle:function(_1f){var _20=document.body.appendChild(document.createElement("style"));_20.setAttribute("type","text/css");if(_20.styleSheet){_20.styleSheet.cssText=_1f;}else{_20.appendChild(document.createTextNode(_1f));}},loadPromo:function(id,_22,_23,_24){var _25=this.getIndex(id);var _26;if(_24){_26=this.oJSBean.merchPrefix+id+_22;}else{_26=this.oJSBean.htmlIds[_25];}
var _27=this.getUIElement(_26);if(this.getStatus(_26)){return true;}
if(_27&&!_27.length){this.addStatus(_26,true);var _28=document.createElement("div");_28.innerHTML=_23;_27.appendChild(_28);return true;}else{this.addStatus(_26,false);}
return false;},hidePromos:function(_29){for(var _2a in _29){var _2b=document.getElementById(_2a);if(_2b!=null){_2b.style.display="none";}}},loadScript:function(_2c){with(window){try{eval(_2c);return true;}
catch(except){}}
return false;},processPopUnderAd:function(pAd){var _2e="height="+pAd.height;_2e+=",width="+pAd.width;_2e+=",menubars=no,scrollbars=no'";var id="p_u_"+pAd.id;var _30=window.open("",id,_2e);if(_30){_30.blur();_30.document.open();_30.document.write(pAd.content);_30.document.close();}
return;},processDoubleClickAd:function(pAd){if(!pAd){return;}
var _32=this.oJSBean;var _33=this.getIndex(pAd.id);var id=_32.htmlIds[_33];var _35=this.getUIElement(id);var url=_32.dblclkUrls[_33];if(this.getStatus(id)){return;}
if(_35&&!_35.length){this.addStatus(id,true);if(pAd.content!=""){url+=pAd.content+";";}
url+="ord="+this.iOrd;_35.innerHTML=this.createIframe(id,url,_32.heights[_33],_32.widths[_33]);_35.style.display="block";}else{this.addStatus(id,false);}
return;},processNoneAd:function(pAd){var _38=this.oJSBean;var _39=this.getIndex(pAd.id);var id=_38.htmlIds[_39];var _3b=this.getUIElement(id);var _3c=_38.defaultUrls[_39];if(this.getStatus(id)){return;}
if(_3b&&!_3b.length){this.addStatus(id,true);if(!_3c||_3c=="collapse"||_3c==""){_3b.style.height="0px";_3b.style.height="0px";_3b.style.display="none";}else{_3b.innerHTML=this.createIframe(id,_3c,_38.heights[_39],_38.widths[_39]);}}else{this.addStatus(id,false);}
return;},processHTMLAd:function(pAd){if(pAd.height=="-1"||pAd.height=="9999"){pAd.height="auto";}
if(pAd.width=="-1"||pAd.width=="9999"){pAd.width="auto";}
var _3e=this.getIndex(pAd.id);var _3f=this.getUIElement(this.oJSBean.htmlIds[_3e]);if(this.getStatus(this.oJSBean.htmlIds[_3e])){return;}
if(_3f&&!_3f.length){this.addStatus(this.oJSBean.htmlIds[_3e],true);var _40=_3f.style,h,w;_40.height=h=(pAd.height.has("auto"))?pAd.height:pAd.height+"px";_40.width=w=(pAd.width.has("auto"))?pAd.width:pAd.width+"px";if(h!="auto"&&w!="auto"){_40.overflow="hidden";}
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
var _68=_62.url;var _69=_68.indexOf("&");var _6a=Math.random();_68=_68.substring(0,_69);_68=_68+"&p="+_62.pids.join(":")+"&ite=2"+"&to="+this.iTIMEOUT;_68=_68.replace("RtmCmd","RtmIt");var _6b=document.createElement("script");_6b.type="text/javascript";if(_6a<0.05){_6b.src=_68;document.getElementsByTagName("head")[0].appendChild(_6b);}
window.clearTimeout(this.oTimeoutId);var _6c=new vjo.dsf.Message("RTM_COMPLETE");_6c.status=0;_6c.vjRTMObject=this;vjo.dsf.ServiceEngine.handleRequest(_6c);return;},getBrowserWidth:function(){var b=this.oClient;var _6e=document.body.clientWidth;if(!b.bIE){_6e=window.innerWidth;}
return _6e;},popUp:function(_6f,_70,_71,_72,_73,_74,_75,_76,_77,_78,_79,_7a){var sP="";sP+=(_71!=null)?",width="+_71:"";sP+=(_72!=null)?",height="+_72:"";sP+=(_76!=null)?",screenX="+_76+",left="+_76:"";sP+=(_77!=null)?",screenY="+_77+",top="+_77:"";sP+=",toolbar="+((_74)?"1":"0");sP+=",location="+((_78)?"1":"0");sP+=",status="+((_73)?"1":"0");sP+=",scrollbars="+((_75)?"1":"0");sP+=",resizable="+((_79)?"1":"0");sP+=",menubar="+((_7a)?"1":"0");if(sP.length>0){sP=sP.substring(1);}
window.open(_70,_6f,sP);return false;},getSegment:function(_7c){var oCJ=vjo.dsf.cookie.VjCookieJar,e=oCJ.readCookie("etfc"),r=oCJ.readCookie("reg"),s=oCJ.readCookie("ebay","sin"),c,n;if(e=="0"){n="3";c="E";}else{if(e=="1"){n="4";c="C";}else{if(e=="2"){n="5";c="D";}else{if((e==""&&(r!=""&&r!=";"))||s=="in"||e=="5"){n="2";c="B";}else{n="1";c="A";}}}}
return _7c?c:n;},getCIDCookie:function(){var oCJ=vjo.dsf.cookie.VjCookieJar;var cid=oCJ.readCookie("npii","cguid");if(cid!="undefined"&&cid!=""){return cid;}},openReportAd:function(_80,_81,_82,_83,_84,_85,_86,_87,_88,_89){if(this.globalAdOverlay==null){this.globalAdOverlay=new vjo.darwin.core.rtm.ReportAd(_82,_83,_84,_85,_88,_89,_81);}
if(this.globalAdOverlay!=null){this.globalAdOverlay.init(_86,_87);this.globalAdOverlay.open(_80);}},closeReportAd:function(){if(this.globalAdOverlay!=null){if(this.globalAdOverlay){this.globalAdOverlay.close();}}},submitReportAd:function(){if(this.globalAdOverlay!=null){if(this.globalAdOverlay){this.globalAdOverlay.submitReport();}}},yahooAdBckGrnd:function(_8a){var _8b=_8a;var oFr=document.getElementById(_8b);oFr.style.visibility="visible";oFr.parentNode.style.backgroundColor="transparent";},addStatus:function(_8d,_8e){this.renderedStatus[_8d]=_8e;},getStatus:function(_8f){return this.renderedStatus[_8f];},storeResponse:function(_90){var t=this,c=_90,qp;t.response=c;t.aContent=c;t.iContentLen=c.length;if(t.bBodyLoaded){t.setInlineContent(c);}else{qp=t.oJSBean.quickPids;if(qp&&qp.length>0){t.renderPromos(qp);}}
if(_90!=null){for(var i=0;i<_90.length;i++){if(_90[i]!=null&&_90[i].content!=null&&_90[i].content.indexOf("openReportAd")>0){this.addReportAdScript();break;}}}
if(_90!=null&&_90.indexOf("openReportAd")>0){this.addReportAdScript();}},addReportAdScript:function(){try{if(this.oJSBean.reportAdJsUrl){var _93=document.createElement("script");_93.type="text/javascript";_93.src=this.oJSBean.reportAdJsUrl;document.getElementsByTagName("head")[0].appendChild(_93);}}
catch(er){}},render:function(){var t=this,r=t.response;if(r){t.setInlineContent(r);}
t.bBodyLoaded=true;},processPromo:function(_95){with(this){if(_95.type=="doubleclick"){processDoubleClickAd(_95);}else{if(_95.type=="html"){processHTMLAd(_95);}else{if(_95.type=="popUnder"){processPopUnderAd(_95);}else{if(_95.type=="promo"){bHasPromo=true;processPromoResponse(_95);}else{if(_95.type=="htmlform"){processHTMLFormAd(_95);}else{processNoneAd(_95);}}}}}}},renderPromos:function(_96){var r=this.response,i,j;if(r){for(j=0;j<r.length;j++){for(i=0;i<_96.length;i++){if(r[j].id==_96[i]+""){this.processPromo(r[j]);}}}}}}).props({siteCatalyst:function(_98,_99,PID,_9b){var esc=vjo.darwin.tracking.sitecatalyst;if(esc){var _9d="Search",loc="North";if(PID==188||PID==570){_9d="Browse";}
if(PID==569||PID==570){loc="Sky";}
var s=esc.EbaySiteCatalyst.s;s.linkTrackVars="prop27";s.linkTrackEvents="None";s.prop27=_99+":"+_9d+":"+PID+":"+_9b;if(_98){s.linkTrackVars="prop28";s.prop28=s.prop27;s.prop27=null;}
s.tl(true,"o","RTM "+loc+" Ad");}}}).endType();

vjo.ctype("vjo.darwin.core.moduletab.ModuleTab").protos({constructs:function(_1){var m=_1;var t=this;t.sHtmlName=m.htmlName;t.sactiveRgt=m.activeRgt;t.sactiveLft=m.activeLft;t.shighlightLft=m.highlightLft;t.shighlightRgt=m.highlightRgt;t.sinactiveLft=m.inactiveLft;t.sinactiveRgt=m.inactiveRgt;t.shoverLft=m.hoverLft;t.shoverRgt=m.hoverRgt;t.stabcontentOuterOff=m.tabcontentOuterOff;t.stabcontentOuterOn=m.tabcontentOuterOn;t.sContentElementId=null;t.sActiveBkgColor=m.activeBkgColor;t.aTabs=[];t.iCurrTab=m.activeTabId;}}).endType();

vjo.ctype("vjo.darwin.core.moduletab.TabProperties").needs("vjo.Registry").protos({_elm:null,_rg:null,constructs:function(_1){_elm=vjo.dsf.Element;_rg=vjo.Registry;m=_1;var t=this;t.hN=m.htmlName;t.iId=m.id;t.sUrl=m.url;t.bIsActive=m.active;t.bIsHighlighted=m.highlighted;t.bIsAjaxEnabled=m.ajaxEnabled;var _n=_rg.get(t.hN);if(_n!==null){var _4=_n.aTabs;_4[_4.length]=t;}
t.bindEventsInJS();t.attachEventListner(t.hN);},bindEventsInJS:function(){var t=this;vjEd=vjo.dsf.EventDispatcher,_tProp=vjo.darwin.core.moduletab.TabProperties;var _6=t.hN+"_tab_rgt_"+t.iId,tbL=t.hN+"_tab_lft_"+t.iId,tb=t.hN+"_"+t.iId;if(t.sUrl===null){t.sUrl="";}
var _f=function(_8){return _tProp.switchToTab(t.hN,t.iId,_8);};vjEd.add(tbL,"click",_f);vjEd.add(_6,"click",_f);vjEd.add(_6,"mouseover",function(_9){return _tProp.hoverChange(t.hN,t.iId,true);});vjEd.add(_6,"mouseout",function(_a){return _tProp.hoverChange(t.hN,t.iId,false);});},attachEventListner:function(_b){var t=vjo.darwin.core.moduletab.TabProperties,fn=function(){t.init(_b);};vjo.dsf.EventDispatcher.addEventListener(window,"load",fn);}}).props({tLft:"_tab_lft_",tRgt:"_tab_rgt_",tCnt:"_content_",tCurrId:"CurrId",switchToTab:function(_d,_e,_f){var t=this;var _11=_rg.get(_d),pArrTab=_11.aTabs,currId=_elm.get(_d+t.tCurrId).value;var _12;var to;for(i=0;i<pArrTab.length;i++){if(currId==pArrTab[i].iId){_12=i;}
if(_e==pArrTab[i].iId){to=i;}}
var tp=vjo.darwin.core.moduletab.TabProperties;var _15=_d+t.tRgt+pArrTab[_12].iId,fTbL=_d+t.tLft+pArrTab[_12].iId;var _16=_d+t.tRgt+pArrTab[to].iId,tTbL=_d+t.tLft+pArrTab[to].iId;if(!pArrTab){return;}
if(pArrTab[to].bIsActive===true||pArrTab[to].iId!=_e){return;}
var rt=_elm.get(_16),url=pArrTab[to].sUrl;if(_f&&(!pArrTab[to].bIsAjaxEnabled)&&(url&&url.length>1&&url.toLowerCase().indexOf("javascript")<0)){var src=_f.nativeEvent.target?_f.nativeEvent.target:_f.nativeEvent.srcElement;if(src.tagName.toLowerCase()==="a"){src.blur();return;}
document.location.href=pArrTab[to].sUrl;return;}
_elm.get(_d+t.tCurrId).value=_e;tp.setTabs(_e,pArrTab[_12],_d,_11);tp.setTabs(_e,pArrTab[to],_d,_11);},setClz:function(_19,_1a){var el;el=_elm.get(_19);if(el){el.className=_1a;}},setBkgColor:function(_1c,_1d){var el;el=_elm.get(_1c);if(el){el.style.backgroundColor=_1d;}},init:function(_1f){var t=this;var _21=_rg.get(_1f),pArrTab=_21.aTabs,tp=vjo.darwin.core.moduletab.TabProperties;var _22=_elm.get(_1f+tp.tCurrId).value;var to;for(i=0;i<pArrTab.length;i++){if(_22==pArrTab[i].iId){to=i;break;}}
if(pArrTab){if(pArrTab[to].bIsActive===true){return;}
for(var i=0;i<pArrTab.length;i++){tp.setTabs(_22,pArrTab[i],_1f,_21);}}},setTabs:function(_25,_26,_27,_28){var t=vjo.darwin.core.moduletab.TabProperties;var _2a=_27+t.tRgt+_26.iId,fTbL=_27+t.tLft+_26.iId;if(_25==_26.iId){_26.bIsActive=true;if(_28.sActiveBkgColor!==null){t.setBkgColor(fTbL,_28.sActiveBkgColor);t.setBkgColor(_2a,_28.sActiveBkgColor);}
t.setClz(fTbL,_28.sactiveLft);t.setClz(_2a,_28.sactiveRgt);t.setClz(_27+t.tCnt+_26.iId,_28.stabcontentOuterOn);}else{_26.bIsActive=false;t.setBkgColor(fTbL,"");t.setBkgColor(_2a,"");if(_26.bIsHighlighted===true){t.setClz(fTbL,_28.shighlightLft);t.setClz(_2a,_28.shighlightRgt);}else{t.setClz(fTbL,_28.sinactiveLft);t.setClz(_2a,_28.sinactiveRgt);}
t.setClz(_27+t.tCnt+_26.iId,_28.stabcontentOuterOff);}},hoverChange:function(_2b,_2c,_2d){var _2e=_rg.get(_2b),tp=vjo.darwin.core.moduletab.TabProperties,pArrTab=_2e.aTabs;for(var i=0;i<pArrTab.length;i++){if(pArrTab[i].iId==_2c){to=i;break;}}
var _30=_2b+tp.tRgt+pArrTab[to].iId,tTbL=_2b+tp.tLft+pArrTab[to].iId;if(pArrTab){if(pArrTab[to].bIsActive===true){return;}
if(pArrTab[to].bIsHighlighted===true){if(_2d===true){tp.setClz(tTbL,_2e.shighlightLft+" "+_2e.shoverLft);tp.setClz(_30,_2e.shighlightRgt+" "+_2e.shoverRgt);}else{tp.setClz(tTbL,_2e.shighlightLft);tp.setClz(_30,_2e.shighlightRgt);}}else{if(_2d===true){tp.setClz(tTbL,_2e.sinactiveLft+" "+_2e.shoverLft);tp.setClz(_30,_2e.sinactiveRgt+" "+_2e.shoverRgt);}else{tp.setClz(tTbL,_2e.sinactiveLft);tp.setClz(_30,_2e.sinactiveRgt);}}}}}).endType();

vjo.ctype("vjo.darwin.core.overlaypanel.VjOverlayPanelMessage").needs("vjo.dsf.Message").inherits("vjo.dsf.Message").protos({constructs:function(_1,_2){var t=this;t.base.call(t,_1);t.request=t.response={};t.sAnchorId=t.sBubbleId="";t.bCheckState=false;},setBubbleId:function(_4){this.sBubbleId=_4;},getBubbleId:function(){return this.sBubbleId;},setAnchorId:function(_5){this.sAnchorId=_5;},getAnchorId:function(){return this.sAnchorId;},getCheckState:function(){return this.bCheckState;}}).endType();

vjo.ctype("vjo.darwin.core.overlaypanelgroup.VjOverlayPanelGroup").needs("vjo.dsf.Message").needs("vjo.darwin.core.overlaypanel.VjOverlayPanelMessage").protos({constructs:function(_1,_2){this.aPostOpenedServiceIds=_1;this.aCloseServiceIds=_2;},handleOverlayPanelOpened:function(_3){var _4=_3.svcId,i;if(_4){for(i=0;i<this.aPostOpenedServiceIds.length;i++){if(_4!=this.aPostOpenedServiceIds[i]){var _5=new vjo.darwin.core.overlaypanel.VjOverlayPanelMessage(this.aCloseServiceIds[i]);vjo.dsf.ServiceEngine.handleRequest(_5);}}}
return false;},handleCloseAll:function(_6){for(i=0;i<this.aCloseServiceIds.length;i++){var _7=new vjo.darwin.core.overlaypanel.VjOverlayPanelMessage(this.aCloseServiceIds[i]);vjo.dsf.ServiceEngine.handleRequest(_7);}
return false;}}).endType();

vjo.ctype("vjo.darwin.core.overlaypanel.VjOverlayPanelController").needs("vjo.dsf.utils.Timer").needs("vjo.dsf.utils.Handlers").protos({H:vjo.dsf.utils.Handlers,constructs:function(){var t=this,H=t.H;t.aOlps=[];t.sIsActivated=false;H.attachEvt(window,"resize",t.initResize,t);var _2=function(){t.initCloseOnMouseOut();};window.setTimeout(_2,500);t.openAtZIndex=5000;},registerOverlayPanel:function(_3){var t=this;t.aOlps[t.aOlps.length]=_3;},initCloseOnMouseOut:function(){var t=this;t.oCloseTimer=new vjo.dsf.utils.Timer(100);var _6=t.oCloseTimer;_6.onTick=function(){var i=0,arr=t.aOlps,len=arr.length;while(i<len){var _8=arr[i];if(_8.bCloseOnMouseOut){_8.tryCloseOnMouseOut();}
i++;}};_6.start();},zIndex:function(){var o=this.openAtZIndex;o+=1;return o;},requireResize:function(){this.sIsActivated=true;},forceResize:function(_a){var t=this,i=0;for(;i<t.aOlps.length;i++){var tO=t.aOlps[i];if(_a&&tO.sOverlayDivId==_a){tO.onResize();break;}else{tO.onResize();}}},initResize:function(){var t=this,H=t.H;H.detachEvt(window,"resize",t.initResize);H.attachEvt(window,"resize",t.requireResize,t);t.requireResize();t.oResizeTimer=new vjo.dsf.utils.Timer();var _e=t.oResizeTimer;_e.interval=500;_e.onTick=function(){var bA=t.sIsActivated;if(bA){for(var i=0;i<t.aOlps.length;i++){var tO=t.aOlps[i];if(tO.bAdjustSize&&tO.bPanelOpen){tO.onResize();}}
bA=false;}};_e.start();},resizeOlps:function(){var i=0,t=this;for(;i<t.aOlps.length;i++){var tO=t.aOlps[i];if(tO.bPanelOpen){tO.onResize();}}}}).inits(function(){vjo.darwin.core.overlaypanel.VjOverlayPanelController=new vjo.darwin.core.overlaypanel.VjOverlayPanelController();}).endType();

vjo.ctype("vjo.darwin.core.overlaypanel.VjOverlayPanelOpenSvcHandler").needs("vjo.dsf.Message").protos({constructs:function(_1){this.sOverlayPanelJsCompId=_1;},invoke:function(_2){var _3=vjo.Registry.get(this.sOverlayPanelJsCompId),mO=_2.bOver,LS=_2.leftOffset,TS=_2.topOffset,mL=_2.left,mT=_2.top,pAC=_3.aAnchorCoor;_3.fSetNotchLocation=_2.fSetNotchLocation;_3.oSetNotchLocationOverrider=_2.oSetNotchLocationOverrider;var _4=true;if(typeof(mO)!="undefined"){_4=mO;}
_3.bCloseOnMouseOut=_4;_3.setAnchorName(_2.sAnchorId);_3.iTmpLftOfset=LS?LS:0;_3.iTmpTopOfset=TS?TS:0;pAC=(mL&&mT)?{x:mL,y:mT}:false;if(_2.bResetPosition){_3.bPanelOpen=false;}
_3.onOpenOverlayPanel(pAC||false);_2.returnData=false;return _2;}}).endType();

vjo.ctype("vjo.darwin.core.overlaypanel.VjOverlayPanelCloseSvcHandler").needs("vjo.dsf.Message").protos({constructs:function(_1){this.sOverlayPanelJsCompId=_1;},invoke:function(_2){var _3=vjo.Registry.get(this.sOverlayPanelJsCompId);_3.onClosePanel(_2.getCheckState?_2.getCheckState():false);return _2;}}).endType();

vjo.ctype("vjo.darwin.core.overlaypanel.VjOverlayPanel").needs("vjo.dsf.client.Browser").needs("vjo.dsf.document.Element").needs("vjo.dsf.document.Positioning").needs("vjo.dsf.document.Shim").needs("vjo.dsf.utils.Handlers").needs("vjo.dsf.Message").needs("vjo.dsf.utils.Object").needs("vjo.dsf.utils.Timer").needs("vjo.dsf.window.utils.VjWindowUtils").needs("vjo.darwin.core.overlaypanel.VjOverlayPanelController").needs("vjo.darwin.core.overlaypanel.VjOverlayPanelOpenSvcHandler").needs("vjo.darwin.core.overlaypanel.VjOverlayPanelCloseSvcHandler").inherits("vjo.dsf.utils.Timer").protos({H:vjo.dsf.utils.Handlers,B:vjo.dsf.client.Browser,PC:vjo.darwin.core.overlaypanel.VjOverlayPanelController,E:vjo.dsf.document.Element,S:vjo.dsf.document.Shim,O:vjo.dsf.utils.Object,M:vjo.dsf.Message,P:vjo.dsf.document.Positioning,W:vjo.dsf.window.utils.VjWindowUtils,constructs:function(_1){var t=this,E=t.E;t.sOverlayDivId=_1.overlayCompId;var _3=t.sOverlayDivId;t.sOlpId=_1.cmpWrapperId;t.oOlp=E.get(_3);t.oCntnt=E.get(_3+"olpcontent");t.oShdw=E.get(_3+"olpshadow");t.oArr=E.get(_3+"olparrow");t.pVt=_1.pointerVertical;t.sArrowVTL=_1.arrowVTL;t.sArrowVTR=_1.arrowVTR;t.sArrowVBL=_1.arrowVBL;t.sArrowVBR=_1.arrowVBR;t.sArrowTL=_1.arrowTL;t.sArrowTR=_1.arrowTR;t.sArrowBR=_1.arrowBR;t.sArrowBL=_1.arrowBL;t.iLastLeft=t.iLastTop=t.mouseState=-1;t.initDx=t.initDy=t.top=t.left=t.iTmpLftOfset=t.iTmpTopOfset=0;t.bShownInCenter=_1.isShownInCenter;t.iCnstLftOfset=_1.leftOffset;t.iCnstTopOfset=_1.topOffset;t.aAnchorCoor=false;t.sHAlgn=_1.horizontalAlign||false;t.sVAlgn=_1.verticalAlign||false;t.iShdwOfset=_1.shadowOffset;t.iOpDlay=_1.openDelay;t.iClsDlay=_1.closeDelay;t.bCloseOnMouseOut=_1.isCloseOnMouseOut;t.bHasMask=_1.hasMask;t.sOpMaskServId=_1.openMaskServiceName;t.sClsMaskServId=_1.closeMaskServiceName;t.sPostOpServId=_1.postOpenedServiceName;t.sPostClsServId=_1.postClosedServiceName;t.bAdjustSize=_1.adjustSize;t.fSetNotchLocation=t.oSetNotchLocationOverrider=t.closeByButton=t.bPanelOpen=t.bCheckState=false;t.sOpenServiceName="OPEN_OVERLAY_PANEL"+_3;t.sCloseServiceName="CLOSE_OVERLAY_PANEL"+_3;t.bSelfRegistry=_1.selfRegistry;t.bPromote2Body=_1.promote2Body;t.bPromoted=false;t.oWHeader=E.get(_3+"olpctnhd");t.oCntn=E.get(_3+"_cnt_n");t.oCntntBox=E.get(_3+"olpctnbox");t.oCloseBtn=E.get(_3+"closebtn");t.oCloseLink=E.get(_3+"closelink");t.sJsId=_1.cmpJsId;t.bHeaderDraggable=_1.headerDraggable;t.bDraggable=_1.draggable;t.bHasCloseBtn=_1.hasCloseBtn;t.bHasCloseLink=_1.hasCloseText;t.icmpWidth=_1.cmpWidth;t.sCmpOpacity=_1.cmpOpacity;t.PC.registerOverlayPanel(t);var _4=E.get("IFrameOff");t.IfrOFF=false;if(_4){t.IfrOFF=_4.value;}
if(t.bSelfRegistry){t.init();}
if(t.bPromote2Body){var _5=function(){t.promote2body(true);};t.H.attachEvt(window,"load",_5,window);}
t.strExt=".gif";t.pngs=_1.pngImgs;t.gifs=_1.gifImgs;if(t.B.bIE&&t.B.bWin&&t.B.iVer>5&&t.B.iVer<=6){t.pngalpha=true;t.strExt=".png";}else{if((t.B.bFirefox)||(t.B.bSafari)||(t.B.bNav)||(t.B.bIE&&t.B.bMac)||(t.B.bOpera&&t.B.bWin&&t.B.iVer>=6)||(t.B.bOpera&&t.B.iVer>=6)||(t.B.bOpera&&t.B.bMac&&t.B.iVer>=5)||(t.B.bWebTV)||(t.B.bIE&&t.B.bWin&&t.B.iVer>6)){t.pngnorm=true;t.strExt=".png";if(!t.B.bSafari&&t.oIFrame){t.oIFrame.style.visibility="hidden";}}}},init:function(){var t=this,H=t.H,open=new vjo.darwin.core.overlaypanel.VjOverlayPanelOpenSvcHandler(t.sJsId),close=new vjo.darwin.core.overlaypanel.VjOverlayPanelCloseSvcHandler(t.sJsId);H.attachSvcReqt(t.sOpenServiceName,open.invoke,open);H.attachSvcReqt(t.sCloseServiceName,close.invoke,close);if((t.bDraggable&&!t.bHeaderDraggable)&&t.oWHeader){H.attachEvt(t.oWHeader,"mousedown",t.startDrag,t);}
if(t.oOlp&&t.oOlp.firstChild){var sO=t.oOlp.firstChild;sO.setAttribute("style",t.sCmpOpacity);}
if(t.bDraggable&&t.oWHeader&&t.oWHeader.firstChild){var ss=t.oWHeader.firstChild;t.setStyle(ss,"cursor","move");}
if(t.bHeaderDraggable&&t.oCntn){H.attachEvt(t.oCntn,"mousedown",t.startDrag,t);t.setStyle(t.oCntn,"cursor","move");}
if(t.bCloseOnMouseOut&&t.oCntntBox){H.attachEvt(t.oCntntBox,"mouseover",t.panelMouseOver,t);H.attachEvt(t.oCntntBox,"mouseout",t.panelMouseOut,t);}
if(t.bHasCloseBtn&&t.oCloseBtn){H.attachEvt(t.oCloseBtn,"click",t.onClosePanel,t);}
if(t.bHasCloseLink&&t.oCloseLink){H.attachEvt(t.oCloseLink,"click",t.onClosePanel,t);H.attachEvt(t.oCloseLink,"mouseover",t.closeLinkMouseOver,t);H.attachEvt(t.oCloseLink,"mouseout",t.closeLinkMouseOut,t);}
if(t.icmpWidth!==0&&t.oOlp){t.setStyle(t.oOlp,"width",(t.icmpWidth+"PX"));}},setStyle:function(_9,_a,_b){_9.style[_a]=_b;},closeLinkMouseOut:function(){this.oCloseLink.style.textDecoration="none";},closeLinkMouseOver:function(){this.oCloseLink.style.textDecoration="underline";},promote2body:function(_c){var t=this;if(_c){t.E.promoteToBody(t.sOlpId);t.bPromoted=true;}},onResize:function(){var t=this;if(t.bPanelOpen){if(!t.bDragged){t.bPanelOpen=false;}
t.openOverlayPanel();}},onOpenOverlayPanel:function(_f){var t=this;if(t.bPromote2Body&&!t.bPromoted){t.promote2body(true);}
t.aAnchorCoor=_f?_f:false;t.openTimer=(t.iOpDlay<=0)?t.openOverlayPanel():window.setTimeout(t.O.hitch(t,"openOverlayPanel"),t.iOpDlay);},openOverlayPanel:function(){var t=this,openedInfo=false;if(t.openTimer){window.clearTimeout(t.openTimer);t.openTimer=null;}
if(!t.bPanelOpen){t.bPanelOpen=true;openedInfo=t.render();if(t.bHasMask){t.openMask();}
var _12=new t.M(t.sPostOpServId);if(openedInfo){_12.clientContext={x:openedInfo[0],y:openedInfo[1],w:openedInfo[2],h:openedInfo[3]};t.H.handle(_12);}}},openMask:function(){var t=this,maskMsg=new t.M(t.sOpMaskServId),cnt=t.oCntnt;try{var _14=cnt?cnt.getElementsByTagName("form")[0]:{name:""};maskMsg.clientContext={formName:_14.name||_14.id};}
catch(e){}
t.H.handle(maskMsg);},align:function(_15,_16,pad,_18){return _15+((_16=="right"||_16=="bottom")?pad:0)+((_16=="middle")?(pad/2):0)+_18;},getLeftOffset:function(){var t=this;return t.iCnstLftOfset+t.iTmpLftOfset;},getTopOffset:function(){var t=this;return t.iCnstTopOfset+t.iTmpTopOfset;},render:function(){var t=this,E=t.E,B=t.B,P=t.P,W=t.W,anc=E.get(t.sAnchorName),coordinates={x:0,y:0},openedInfo=false,AC=t.aAnchorCoor,LS=t.getLeftOffset(),TS=t.getTopOffset(),SS=t.iShdwOfset;if(!t.bShownInCenter){if(AC){coordinates.x=AC.x+LS;coordinates.y=AC.y+TS;}else{coordinates.x=(B.bIE&&B.fVer<7)?P.getOffsetLeft(anc)+1:P.getOffsetLeft(anc);coordinates.y=P.getOffsetTop(anc);coordinates.x=t.align(coordinates.x,t.sHAlgn,anc?anc.offsetWidth:0,LS);coordinates.y=t.align(coordinates.y,t.sVAlgn,anc?anc.offsetHeight:0,TS);}}
var olp=t.oOlp,olpS=olp.style;if(olpS){olpS.visibility="visible";olpS.zIndex=t.PC.zIndex();if(olpS.zIndex===0){t.promote2body();}}
openedInfo=t.fSetNotchLocation?t.fSetNotchLocation(t.oSetNotchLocationOverrider,t,olp,coordinates):t.setNotchLocation(olp,coordinates);t.oShdw.style.height=W.toPixels(t.oCntnt.offsetHeight+2);if(t.oIFrame){var s=t.oIFrame.style;s.width=W.toPixels(olp.offsetWidth+SS);s.height=W.toPixels(olp.offsetHeight+SS+2);}else{if(!t.IfrOFF){t.oIFrame=t.S.add(olp,SS,SS+2);}}
t.changeImage();return openedInfo;},changeImage:function(){var t=this;var e=vjo.dsf.Element;var _20=e.get(t.sOlpId);var _21=e.getElementsByTagClass(_20,"div","basOlp-scn");var _22=e.getElementsByTagClass(_20,"div","basOlp-s");var _23=null;for(var idx=0;idx<_22.length;idx++){if(_22[idx].className==="basOlp-s"){_23=_22[idx];break;}}
var _25=e.getElementByTagClass(_20,"div","basOlp-n");var _26=e.getElementByTagClass(_20,"div","basOlp-ctent");if(t.pngalpha){for(var cdx=0;cdx<_21.length;cdx++){t.applyStyle(_21[cdx],t.gifs[1]);_21[cdx].style.filter="";}
var _28="progid:DXImageTransform.Microsoft.AlphaImageLoader(src=";_23.style.filter=_28+t.pngs[0]+", sizingMethod='scale')";_25.style.filter=_28+t.pngs[0]+", sizingMethod='scale')";_26.style.filter=_28+t.pngs[0]+", sizingMethod='scale')";_23.style.background="transparent";_25.style.background="transparent";_26.style.background="transparent";try{var _29="DXImageTransform.Microsoft.AlphaImageLoader";_23.filters[_29].enabled=true;_25.filters[_29].enabled=true;_26.filters[_29].enabled=true;}
catch(e){for(var cdx=0;cdx<_21.length;cdx++){t.applyStyle(_21[cdx],t.gifs[1]);_21[cdx].style.filter="";}
t.applyStyle(_23,t.gifs[0]);t.applyStyle(_25,t.gifs[0]);t.applyStyle(_26,t.gifs[0]);_23.style.filter="";_25.style.filter="";_26.style.filter="";}}else{if(t.pngnorm){t.applyStyle(_23,t.pngs[0]);t.applyStyle(_25,t.pngs[0]);t.applyStyle(_26,t.pngs[0]);for(var cdx=0;cdx<_21.length;cdx++){t.applyStyle(_21[cdx],t.pngs[1]);}}else{t.applyStyle(_23,t.gifs[0]);t.applyStyle(_25,t.gifs[0]);t.applyStyle(_26,t.gifs[0]);for(var cdx=0;cdx<_21.length;cdx++){t.applyStyle(_21[cdx],t.gifs[1]);}}}},applyStyle:function(_2a,bkg){_2a.style.backgroundImage="url("+bkg+")";},setAnchorName:function(_2c){this.sAnchorName=_2c;},setBubbleDivId:function(_2d){this.sOverlayDivId=_2d;},onClosePanel:function(_2e){var t=this,CD=t.iClsDlay;t.bCheckState=_2e;t.closeTimer=(CD<=0)?t.closePanel():window.setTimeout(t.O.hitch(t,"closePanel"),CD);},closePanel:function(){var t=this,cT=t.closeTimer;if(cT){window.clearTimeout(cT);cT=null;}
if(t.bCheckState&&t.mouseState==1){t.bCheckState=false;return;}
t.hidePanel();if(t.mouseState==1){t.closeByButton=true;}
t.bDragged=false;},setNotchLocation:function(olp,_32){var t=this,W=t.W,olpS=olp.style,iX=_32.x,iY=_32.y,iW=olp.offsetWidth,iH=olp.offsetHeight,iScreenW=W.getBrowserWindowWidth(),iScreenH=W.getBrowserWindowHeight(),aScollLoc=W.getScrollXY(),iScrollX=aScollLoc[0],iScrollY=aScollLoc[1],arr=t.oArr;if(!t.bShownInCenter){if(t.pVt){var _34=iX-iScrollX,yLoc=iY-iScrollY,finalX=0,finalY=0,arrH=21,sAuto="auto",iAdj=5,bTop=true,bLeft=false,B=t.B,iGuard=1;if(!B.bIE){iAdj+=16;}
if((yLoc-iH-arrH)<0){finalY=iY+arrH+iGuard;bTop=false;}else{finalY=iY-iH-arrH-iGuard;}
if(iX-iW/2<0){finalX=0;}else{if(iX+iW/2+iAdj>iScreenW){finalX=iScreenW-iW-iAdj;bLeft=true;}else{finalX=iX-(iW/2);}}
if(arr){arr.style.left=W.toPixels(iX-finalX-(bLeft?arr.offsetWidth+iGuard:-iGuard));}
if(bTop){t.setArrDivStyle(arr,bLeft?t.sArrowVBL:t.sArrowVBR,sAuto,W.toPixels(-arrH));}else{t.setArrDivStyle(arr,bLeft?t.sArrowVTL:t.sArrowVTR,W.toPixels(-arrH),sAuto);}}else{var _34=iX-iScrollX,yLoc=iY-iScrollY,arrH=arr?21:0,arrW=arr?21:0,arrS=arr?14:0,finalX=0,finalY=0,finalW=iW+arrW,finalH=iH,sAuto="auto",iAdj=0,bLeft=false;if((_34+iW+arrW)<iScreenW||iW>iScreenW||(t.sHAlgn&&t.sHAlgn!="middle")){finalX=iX+arrW;bLeft=true;}else{if((_34-iW-arrW)<0){finalX=(iScreenW/2-iW/2)+iScrollX+t.getLeftOffset();}else{finalX=iX-iW-arrW;}}
if((yLoc+iH)<iScreenH){finalY=iY-arrS;t.setArrDivStyle(arr,bLeft?t.sArrowTL:t.sArrowTR,W.toPixels(arrS),sAuto);}else{if((yLoc+arrS)>iH){finalY=iY-iH+arrS;finalY-=t.getTopOffset()*2;t.setArrDivStyle(arr,bLeft?t.sArrowBL:t.sArrowBR,sAuto,W.toPixels(arrS));}else{if(yLoc<(iScreenH/2)&&(iH<iScreenH)){iAdj=(iY+iH)-(iScrollY+iScreenH);finalY=iY-iAdj-arrS;t.setArrDivStyle(arr,bLeft?t.sArrowTL:t.sArrowTR,W.toPixels(arrS+iAdj),sAuto);}else{iAdj=iScrollY-(iY-iH+arrS);finalY=iScrollY;t.setArrDivStyle(arr,bLeft?t.sArrowBL:t.sArrowBR,W.toPixels(yLoc-arrS),sAuto);}}}}}else{finalX=(iScreenW/2-iW/2)+iScrollX+t.getLeftOffset();finalY=iScreenH/2-iH/2;finalY=finalY>0?finalY:0;finalY=finalY+iScrollY+t.getTopOffset();}
olpS.left=W.toPixels(finalX);olpS.top=W.toPixels(finalY);return[finalX,finalY,finalW,finalH];},setArrDivStyle:function(_35,_36,_37,_38){if(_35){_35.className=_36;_35.style.top=_37;_35.style.bottom=_38;}},startDrag:function(_39){var t=this,P=t.P,W=t.W,olp=t.oOlp,SR=t.bSelfRegistry,H=t.H,db=document.body;if(!t.bMonitorDrag){t.bMonitorDrag=true;t.bDragged=true;var _3b=_39.nativeEvent;if(SR){_3b=_39.nativeEvent||_39;}
var _3c=W.eventTop(_3b),mouseX=W.eventLeft(_3b);t.initDx=P.getOffsetLeft(olp)-mouseX;t.initDy=P.getOffsetTop(olp)-_3c;t.top=t.iLastTop=_3c;t.left=t.iLastLeft=mouseX;t.fOnMouseMoveHdl=H.attachEvt(db,"mousemove",t.onMouseMove,t);t.fOnMouseUpHdl=H.attachEvt(db,"mouseup",t.onMouseUp,t);if(olp){olp.onselectstart=olp.onmousedown=function(){return false;};}
t.interval=50;t.onTick=function(){t.monitorMouseDrag();};t.start();}
if(SR){var _3d=_3b.target||_3b.srcElement;H.attachEvt(_3d,"mouseup",t.stopDrag,t);}},onMouseMove:function(_3e){var t=this,W=t.W;if(t.bMonitorDrag){var _40=window.event?window.event:_3e;var _41=W.eventTop(_40),leftValue=W.eventLeft(_40);if(_41>=0&&leftValue>=0){t.top=_41;t.left=leftValue;}}},onMouseUp:function(_42){var t=this,H=t.H,db=document.body;if(t.bMonitorDrag){H.detachEvt(db,"mousemove",t.fOnMouseMoveHdl);H.detachEvt(db,"mouseup",t.fOnMouseUpHdl);t.stopDrag();}},monitorMouseDrag:function(){if(this.bMonitorDrag){var t=this,P=t.P,W=t.W;if((t.left!=t.iLastLeft)||(t.top!=t.iLastTop)){var dx=t.left-t.iLastLeft,dy=t.top-t.iLastTop;t.iLastLeft=t.left;t.iLastTop=t.top;var olp=t.oOlp,olpS=olp.style,x=P.getOffsetLeft(olp),y=P.getOffsetTop(olp);olpS.left=W.toPixels(x+dx);olpS.top=W.toPixels(y+dy);}}},stopDrag:function(){var t=this;t.bMonitorDrag=false;var olp=t.oOlp;if(olp){olp.onselectstart=olp.onmousedown=null;}
t.stop();},panelMouseOver:function(){this.mouseState=1;},panelMouseOut:function(){this.mouseState=0;},tryCloseOnMouseOut:function(){var t=this;if(t.mouseState===0){t.mouseState=-1;if(!t.closeByButton){t.onClosePanel();}else{t.closeByButton=false;t.mouseState=-1;}}},hidePanel:function(){var t=this,H=t.H,M=t.M,olp=t.oOlp,olpS=olp.style;olpS.visibility="hidden";olpS.left="-1600px";t.bPanelOpen=false;t.iTmpLftOfset=t.iTmpTopOfset=0;if(t.bHasMask){H.handle(new M(t.sClsMaskServId));}
if(t.sPostClsServId){H.handle(new M(t.sPostClsServId));}
if(t.oIFrame){if(!t.fCleanUp){t.fCleanUp=function(){t.S.remove(t.oOlp,t.oIFrame);t.oIFrame=null;};H.attachEvt(window,"unload",t.fCleanUp,t);}}},setContentTitle:function(_4b){var t=this,tcnt=t.oTitlCntnr;if(!tcnt){tcnt=t.E.get(t.sOverlayDivId+"olpcontenttitle");}
if(tcnt){tcnt.innerHTML=_4b;}}}).endType();

vjo.ctype("vjo.darwin.core.overlaypanel.VjOverlayPanelEvtHandlers").needs("vjo.dsf.Message").needs("vjo.darwin.core.overlaypanel.VjOverlayPanelMessage").needs("vjo.dsf.window.utils.VjWindowUtils").props({M:vjo.darwin.core.overlaypanel.VjOverlayPanelMessage,handleOpenEvent:function(_1,_2){var _3=new this.M(_1);_3.setAnchorId(_2);return _3;},handleOpenEventXY:function(_4,x,y){var _7=new this.M(_4);_7.left=x;_7.top=y;_7.resetPosition=false;return _7;},handleOpenEventByLoc:function(_8,_9){var e=_9.nativeEvent,W=vjo.dsf.window.utils.VjWindowUtils,x=W.eventLeft(e),y=W.eventTop(e);return this.handleOpenEventXY(_8,x,y);},handleCloseEvent:function(_b){var _c=new this.M(_b);return _c;},handleCloseEventWithCheck:function(_d,_e){var _f=new this.M(_d);_f.bCheckState=_e;return _f;}}).endType();

vjo.ctype("vjo.darwin.core.roundedcornerpanel3.VjRoundedCornerPanel3SvcHandler").needs("vjo.dsf.Message").needs("vjo.dsf.document.Element").protos({constructs:function(_1){this.sRoundedCornerPanel3JsCompId=_1;},invoke:function(_2){var _e=vjo.dsf.document.Element;if(_2.changeType==="HEADER_CHANGE"){var _4=_e.get(_2.sElemId+"_t");var _5=_e.get(_2.sElemId+"_h");_5.className=_2.cssClz;if(_4){_4.className=_2.cssClz;}
this.setBorderToAnchor(_2.sElemId,_2.cssClz);}else{if(_2.changeType==="FOOTER_CHANGE"){var _6=_e.get(_2.sElemId+"_f");var _7=_e.get(_2.sElemId+"_b");var _8;_6.className=_2.cssClz;if(_7){_8=_7.className;if(_8.indexOf("c-sgf")>-1){_7.className=_2.cssClz;}else{_7.className=_8+" "+_2.cssClz;}}}else{if(_2.changeType==="SHOW_HIDE_CONTENT"){var _9=_e.get(_2.sElemId+"_c");var _a=_e.get(_2.sElemId);var _b=_a.className;if(_9.style.display=="none"){_9.style.display="";if(_b.indexOf("c-sgfx")>-1){_b=_b.replace("c-sgfx","");}}else{_9.style.display="none";_b+=" c-sgfx";}
_a.className=_b;}}}},setBorderToAnchor:function(_c,_d){var _e=vjo.dsf.document.Element;var _f=_e.get(_c+"_c-gr");var _10=_e.get(_c+"_c-bl");var _11=_e.get(_c+"_c-rd");var _12=_e.get(_c+"_c-yl");var _13=_e.get(_c+"_c-gy");if(_f!==null){if(_d==="c-gr"){_f.className="r3-gr-bdr";}else{_f.className="";}}
if(_10!==null){if(_d==="c-bl"){_10.className="r3-bl-bdr";}else{_10.className="";}}
if(_11!==null){if(_d==="c-rd"){_11.className="r3-rd-bdr";}else{_11.className="";}}
if(_12!==null){if(_d==="c-yl"){_12.className="r3-yl-bdr";}else{_12.className="";}}
if(_13!==null){if(_d==="c-gy"){_13.className="r3-gy-bdr";}else{_13.className="";}}}}).endType();

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

vjo.ctype("vjo.darwin.core.pageleveltab.TabProperties").needs("vjo.Registry").protos({_elm:null,_rg:null,mName:null,constructs:function(_1){_elm=vjo.dsf.Element;_rg=vjo.Registry;m=_1;var t=this;t.hN=m.htmlName;mName=m.htmlName;t.iId=m.id;t.sUrl=m.url;t.bIsActive=m.active;t.bIsHighlighted=m.highlighted;t.bIsAjaxEnabled=m.ajaxEnabled;var _n=_rg.get(t.hN);if(_n!==null){var _4=_n.aTabs;_4[_4.length]=t;}
t.bindEventsInJS(_n.hon);t.attachEventListner(t.hN);},bindEventsInJS:function(_5){var t=this,vjEd=vjo.dsf.EventDispatcher,_tProp=vjo.darwin.core.pageleveltab.TabProperties;var _7=t.hN+"_tab_rgt_"+t.iId,tbL=t.hN+"_tab_lft_"+t.iId,tb=t.hN+"_"+t.iId;if(t.sUrl===null){t.sUrl="";}
var _f=function(_9){return _tProp.switchToTab(t.hN,t.iId,_9);};vjEd.add(tbL,"click",_f);vjEd.add(_7,"click",_f);if(_5){vjEd.add(_7,"mouseover",function(_a){return _tProp.hoverChange(t.hN,t.iId,true);});vjEd.add(_7,"mouseout",function(_b){return _tProp.hoverChange(t.hN,t.iId,false);});}},attachEventListner:function(_c){var t=vjo.darwin.core.pageleveltab.TabProperties,fn=function(){t.init(_c);};vjo.dsf.EventDispatcher.addEventListener(window,"load",fn);}}).props({tLft:"_tab_lft_",tRgt:"_tab_rgt_",tCnt:"_content_",tCurrId:"CurrId",switchToTab:function(_e,_f,_10){var t=this;var _12=_rg.get(_e),pArrTab=_12.aTabs,currId=_elm.get(_e+t.tCurrId).value;var _13;var to;for(i=0;i<pArrTab.length;i++){if(currId==pArrTab[i].iId){_13=i;}
if(_f==pArrTab[i].iId){to=i;}}
var tp=vjo.darwin.core.pageleveltab.TabProperties;var _16=_e+t.tRgt+pArrTab[_13].iId,fTbL=_e+t.tLft+pArrTab[_13].iId;var _17=_e+t.tRgt+pArrTab[to].iId,tTbL=_e+t.tLft+pArrTab[to].iId;if(pArrTab){if(pArrTab[to].bIsActive===true){return;}
if(pArrTab[to].iId==_f){var rt=_elm.get(_17),url=pArrTab[to].sUrl;if(_10&&(!pArrTab[to].bIsAjaxEnabled)&&(url&&url.length>1&&url.toLowerCase().indexOf("javascript")<0)){var src=_10.nativeEvent.target?_10.nativeEvent.target:_10.nativeEvent.srcElement;if(src.tagName.toLowerCase()==="a"){src.blur();return;}
document.location.href=pArrTab[to].sUrl;return;}
_elm.get(_e+t.tCurrId).value=_f;tp.setTabs(_f,pArrTab[_13],_e,_12);tp.setTabs(_f,pArrTab[to],_e,_12);}}},setClz:function(_1a,_1b){var el;el=_elm.get(_1a);if(el){el.className=_1b;}},setBkgColor:function(_1d,_1e){var el;el=_elm.get(_1d);if(el){el.style.backgroundColor=_1e;}},init:function(_20){var t=this,to;var _22=_rg.get(_20),pArrTab=_22.aTabs,tp=vjo.darwin.core.pageleveltab.TabProperties;var _23=_elm.get(_20+tp.tCurrId).value;for(i=0;i<pArrTab.length;i++){if(_23==pArrTab[i].iId){to=i;break;}}
if(pArrTab){if(pArrTab[to].bIsActive===true){return;}
for(var i=0;i<pArrTab.length;i++){tp.setTabs(_23,pArrTab[i],_20,_22);}}},setTabs:function(_25,_26,_27,_28){var t=vjo.darwin.core.pageleveltab.TabProperties;var _2a=_27+t.tRgt+_26.iId,fTbL=_27+t.tLft+_26.iId;if(_25==_26.iId){_26.bIsActive=true;if(_28.sActiveBkgColor!==null){t.setBkgColor(fTbL,_28.sActiveBkgColor);t.setBkgColor(_2a,_28.sActiveBkgColor);}
t.setClz(fTbL,_28.sactiveLft);t.setClz(_2a,_28.sactiveRgt);t.setClz(_27+t.tCnt+_26.iId,_28.stabcontentOuterOn);}else{_26.bIsActive=false;t.setBkgColor(fTbL,"");t.setBkgColor(_2a,"");if(_26.bIsHighlighted===true){t.setClz(fTbL,_28.shighlightLft);t.setClz(_2a,_28.shighlightRgt);}else{t.setClz(fTbL,_28.sinactiveLft);t.setClz(_2a,_28.sinactiveRgt);}
t.setClz(_27+t.tCnt+_26.iId,_28.stabcontentOuterOff);}},hoverChange:function(_2b,_2c,_2d){var _2e=_rg.get(_2b),tp=vjo.darwin.core.pageleveltab.TabProperties,pArrTab=_2e.aTabs;for(var i=0;i<pArrTab.length;i++){if(pArrTab[i].iId==_2c){to=i;break;}}
var _30=_2b+tp.tRgt+pArrTab[to].iId,tTbL=_2b+tp.tLft+pArrTab[to].iId;if(pArrTab){if(pArrTab[to].bIsActive===true){return;}
if(pArrTab[to].bIsHighlighted===true){if(_2d===true){tp.setClz(tTbL,_2e.shighlightLft+" "+_2e.shoverLft);tp.setClz(_30,_2e.shighlightRgt+" "+_2e.shoverRgt);}else{tp.setClz(tTbL,_2e.shighlightLft);tp.setClz(_30,_2e.shighlightRgt);}}else{if(_2d===true){tp.setClz(tTbL,_2e.sinactiveLft+" "+_2e.shoverLft);tp.setClz(_30,_2e.sinactiveRgt+" "+_2e.shoverRgt);}else{tp.setClz(tTbL,_2e.sinactiveLft);tp.setClz(_30,_2e.sinactiveRgt);}}}}}).endType();

vjo.ctype("vjo.darwin.core.pageleveltab.PageLevelTab").needs("vjo.Registry").protos({constructs:function(_1){var t=_1,o=this;o.sHtmlName=t.htmlName;o.sactiveRgt=t.activeRgt;o.sactiveLft=t.activeLft;o.shighlightLft=t.highlightLft;o.shighlightRgt=t.highlightRgt;o.sinactiveLft=t.inactiveLft;o.sinactiveRgt=t.inactiveRgt;o.shoverLft=t.hoverLft;o.shoverRgt=t.hoverRgt;o.stabcontentOuterOff=t.tabcontentOuterOff;o.stabcontentOuterOn=t.tabcontentOuterOn;o.sContentElementId=null;o.sActiveBkgColor=t.activeBkgColor;o.aTabs=[];o.iCurrTab=t.activeTabId;o.hon=t.hoverOn;}});

vjo.needs("vjo.dsf.typeextensions.string.Comparison");vjo.ctype("vjo.dsf.typeextensions.string.Trim").endType();String.prototype.trim=function(){var s=this;while(s.substring(0,1).isAny(" ","\n","\r")){s=s.substring(1,s.length);}
while(s.substring(s.length-1,s.length).isAny(" ","\n","\r")){s=s.substring(0,s.length-1);}
return s;};

vjo.ctype("vjo.darwin.core.ebayheader.searchbox.SearchBox").needs("vjo.dsf.client.Browser").needs("vjo.dsf.typeextensions.string.Trim").props({Focus:function(_1){var _2=vjo.dsf.document.Element.get(_1);if(typeof(_2)!="undefined"&&_2){var B=vjo.dsf.client.Browser.bIE;if(B.bIE&&B.iVer==6){setTimeout(function(){_2.focus();},0);}else{_2.focus();}}},IeOptionDisabler:function(_4){if(vjo.dsf.client.Browser.bIE){var sl=vjo.dsf.document.Element.get(_4);if(sl){var _6;sl.onchange=function(){_6=this.selectedIndex=(this.options[this.selectedIndex].disabled)?_6:this.selectedIndex;};sl.onfocus=function(){_6=this.selectedIndex;};this.greydisabledoption(sl);}}},greydisabledoption:function(e){var i,op;for(i=0;i<e.options.length;i++){op=e.options[i];if(op.disabled){op.style.color="graytext";}}},AppendKW:function(p,e){var x="undefined";if(typeof(_GlobalNavHeaderAppendKW)!=x&&_GlobalNavHeaderAppendKW){var b=vjo.dsf.document.Element.get(e);if(typeof(b)!=x&&b){if(p.nativeEvent===null||p.nativeEvent===undefined){return;}
var t=p.nativeEvent.srcElement||p.nativeEvent.target;if(t!==null){var s=b.value;var s1=encodeURIComponent(s);var ot=t.getAttribute("act");if(typeof(ot)!=x&&ot&&s&&s.trim()){if(ot.substr(ot.length-1)!="/"){t.action=ot+"/"+s1;}else{t.action=ot+s1;}}}
var n=vjo.dsf.document.Element.get("_naf");if(typeof(n)!=x&&n){n.disabled=false;}}}}}).endType();

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

vjo.ctype("vjo.darwin.tracking.sojourner.SojData").singleton().protos({constructs:function(){this.sojD;}}).inits(function(){vjo.darwin.tracking.sojourner.SojData=new vjo.darwin.tracking.sojourner.SojData();}).endType();

vjo.ctype("vjo.darwin.tracking.sojourner.CalData").singleton().protos({constructs:function(){this.cal={};},setData:function(_1,_2){if(!_1||!_2){return;}
this.cal[_1]=_2;},getData:function(_3){if(_3){return this.cal[_3];}}}).inits(function(){vjo.darwin.tracking.sojourner.CalData=new vjo.darwin.tracking.sojourner.CalData();}).endType();

vjo.ctype("vjo.darwin.tracking.sojourner.TrackingRespHdl").needs(["vjo.darwin.tracking.sojourner.SojData","vjo.darwin.tracking.sojourner.CalData"]).props({handleResponse:function(_1){if(_1.errors&&_1.errors.length>0){vjo.darwin.tracking.sojourner.SojData.sojD="";}
if(_1.response&&_1.response.dataMap&&_1.response.dataMap.SOJDATA){vjo.darwin.tracking.sojourner.SojData.sojD=_1.response.dataMap.SOJDATA;}
if(_1.response&&_1.response.dataMap&&_1.response.dataMap.TDATA){vjo.darwin.tracking.sojourner.CalData.setData(_1.svcId,_1.response.dataMap.TDATA);}}}).endType();

vjo.ctype("vjo.darwin.core.ebayheader.autocomplete.AutoComplete").needs(["vjo.dsf.Message","vjo.dsf.Element","vjo.dsf.EventDispatcher","vjo.dsf.ServiceEngine","vjo.dsf.window.utils.VjWindowUtils"]).protos({constructs:function(_1,_2){var t=this,E=vjo.dsf.Element,ED=vjo.dsf.EventDispatcher;t.sFormId=_2;t.sAcDivId=_1+"_acdiv";t.acdivWidth=0;t.sShowImg=false;t.input=E.get(_1);t.input.setAttribute("AUTOCOMPLETE","OFF");t.acdiv=E.get(t.sAcDivId);var _4=document.forms;ED.add(_1,"keyup",function(e){var _6=e.nativeEvent.keyCode;var _7=new vjo.dsf.Message("SVC_GH_OUT");_7.clientContext={type:"kw_keyup",src:e.src,value:t.input.value,keyCode:_6};return _7;});ED.add(_1,"keydown",function(e){var _9=e.nativeEvent.keyCode;var _a=new vjo.dsf.Message("SVC_GH_OUT");_a.clientContext={type:"kw_keydown",src:e.src,value:t.input.value,keyCode:_9};return _a;});ED.add(_1,"mouseover",function(e){var _c=e.nativeEvent.keyCode;var _d=new vjo.dsf.Message("SVC_GH_OUT");_d.clientContext={type:"kw_mouseover",src:e.src};return _d;});ED.add(_1,"blur",function(e){var _f=e.nativeEvent.keyCode;var _10=new vjo.dsf.Message("SVC_GH_OUT");_10.clientContext={type:"kw_blur",src:e.src};return _10;});ED.add(this.sAcDivId,"click",function(e){var _12=new vjo.dsf.Message("SVC_GH_OUT");_12.clientContext={type:"show_click"};return _12;});vjo.dsf.ServiceEngine.registerSvcHdl("SVC_GH_IN",function(_13){var _14=_13.clientContext.type;if(_14=="kw_updvalue"){t.input.value=_13.clientContext.value;}else{if(_14=="kw_autocomplete"){if(_13.clientContext.bOn){t.input.blur();}
t.input.setAttribute("AUTOCOMPLETE",_13.clientContext.bOn?"ON":"OFF");if(_13.clientContext.bOn){t.input.blur();t.input.focus();}}else{if(_14=="search_updtrk"){for(i=0;i<_4.length;i++){if(_4[i].name==_2){for(var j=0;j<_4[i].length;j++){if(_4[i].elements[j].name=="_trksid"){var _16=_4[i].elements[j].value;var _17="";var _18=["p","m","l"];for(var _19 in _18){var _1a=new RegExp(_18[_19]+"[0-9]+(?=.|$)");var _1b=_16.match(_1a);var _1c=_13.clientContext.lnkStr.match(_1a);var _1d=_1c?_1c[0]:(_1b?_1b[0]:null);var _1e=_17.length>0?".":"";if(_1d){_17+=_1e+_1d;}}
_4[i].elements[j].value=_17;return;}}}}}else{if(_14=="search_submit"){for(i=0;i<_4.length;i++){if(_4[i].name==_2){_4[i].submit();return;}}}else{if(_14=="sug_icon_show"){if(_13.clientContext.bShow){t.showImage();}else{t.hideImage();}}else{if(_14=="kw_focus"){t.input.focus();t.input.value=t.input.value+"";}}}}}}});},showImage:function(){var t=this;if(t.sShowImg){return;}
if(t.acdivWidth===0){t.acdiv.style.display="inline";t.acdivWidth=t.acdiv.offsetWidth;}
var _20=t.acdivWidth+3;if("BackCompat"==document.compatMode){_20=_20-5;}
t.input.style.width=(t.input.clientWidth-_20)+"px";t.input.style.borderRightWidth="0px";t.acdiv.style.display="inline";t.sShowImg=true;},hideImage:function(){var t=this;if(!t.sShowImg){return;}
var _22=t.acdiv.offsetWidth-5;if("BackCompat"==document.compatMode){_22=_22+6;}
t.input.style.width=(t.input.clientWidth+_22)+"px";t.input.style.borderRightWidth="1px";t.acdiv.style.display="none";t.sShowImg=false;}}).props({init:function(_23,_24){new vjo.darwin.core.ebayheader.autocomplete.AutoComplete(_23,_24);}}).endType();

vjo.type("vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteEncoder").protos({aCharList:[[new RegExp("[%]","g"),"_"],[new RegExp("[.]","g"),"_2e"],[new RegExp("[+]","g"),"_2b"],[new RegExp("[']","g"),"_27"]],pseudoDiv:null,constructs:function(){this.pseudoDiv=document.createElement("div");},encode:function(_1){var _2=encodeURIComponent(_1),t=this;for(var j=0;j<t.aCharList.length;j++){var _4=t.aCharList[j];_2=_2.replace(_4[0],_4[1]);}
return _2;},decodeCookie:function(_5){var _6=_5||"";_6=_6.replace(new RegExp("[+]","g")," ");_6=decodeURIComponent(_6);return _6;},encodeHTML:function(_7){var e=this.pseudoDiv;if(typeof(e.textContent)!="undefined"){e.textContent=_7;}else{e.innerText=_7;}
return e.innerHTML;}});

vjo.needs("vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteEncoder");vjo.type("vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteCache").protos({oCache:{},oIndex:{},oReference:{},oLeaf:{},oEncoder:null,UN:"undefined",TFU:"function",TNU:"number",constructs:function(){this.oEncoder=new vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteEncoder();},add:function(_1){try{var t=this,kw=_1[0],kwList=_1[1],laList=_1[2],cacheItem=t.oCache[kw];if(_1.length>=3){t.addItem(kw,kw,"k",kwList);}else{try{if(typeof(kwList[0])=="string"){laList=null;t.addItem(kw,kw,"k",kwList);}else{if(typeof(kwList[0])=="object"&&(kwList[0]instanceof Array)){laList=_1[1];}}}
catch(err){laList=null;}}
if(laList!==null){for(var i=0;i<laList.length;i++){var _4=laList[i],lookAheadKw=kw+_4[0],lookAheadType=_4[1],lookAheadList=_4[2];if(typeof(lookAheadList)==t.UN||lookAheadList===null){lookAheadType="fd";lookAheadList=_4[0];}
t.addItem(lookAheadKw,kw,lookAheadType,lookAheadList);}}else{t.addItem(kw,kw,"null");}}
catch(e){}},addItem:function(_5,_6,_7,_8){var t=this;_5=_5.toLowerCase();_6=_6.toLowerCase();if(_7=="k"){t.oCache[_5]=t.buildItem("k",(typeof(_8)==t.TNU)?(""+_8):_8,_6);return t.oCache[_5];}else{if(_7=="f"){t.oIndex[_5]=t.buildItem("f",(typeof(_8)==t.TNU)?(""+_8):_8,_6);return t.oIndex[_5];}else{if(_7=="fd"){t.oReference[_5]=t.buildItem("fd",_8,_6);return t.oReference[_5];}else{if(_7=="null"){t.oLeaf[_5]=t.buildItem("null",null,_6);return t.oLeaf[_5];}}}}},buildItem:function(_a,_b,_c){var _d={"type":_a,"shortPrefix":_c};if(_b!==null){_d.keyword=_b;}
return _d;},get:function(_e,_f){_e=_e.toLowerCase();_f=_f.toLowerCase();var t=this,cacheItem=t.oCache[_e],indexItem=t.oIndex[_e],referenceItem=t.oReference[_e];if(typeof(cacheItem)!=t.UN&&typeof(cacheItem)!=t.TFU){return cacheItem;}
if(typeof(indexItem)!=t.UN&&typeof(indexItem)!=t.TFU){return indexItem;}
if(typeof(referenceItem)!=t.UN&&typeof(referenceItem)!=t.TFU){return referenceItem;}
if(_f){var _11=t.oLeaf[_f];if(typeof(_11)!=t.UN&&typeof(_11)!=t.TFU){return _11;}
var _12=[];for(var _13 in t.oIndex){var _14=t.oIndex[_13];if(_14.shortPrefix==_f){_12.push(_13);}}
_12.sort();if(_12.length===0){return null;}
var _15=_12.length-1;for(i=0;i<_12.length;i++){if(_e<_12[i]){_15=i-1;break;}}
if(_15<0){_15=0;}
return t.oIndex[_12[_15]];}
return null;}});

vjo.needs("vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteEncoder");vjo.type("vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteConfig").protos({oModel:{"rootDir":"autofill","listSize":7,"delayTime":100,"dirDepth":3,"noSugShowTime":1500,"svcIn":"SVC_GH_IN","svcOut":"SVC_GH_OUT"},defaultAlgo:"1",algoMap:{"1":"f","2":"a"},widthDef:{"unit":9,"min":{"px":233,"char":26},"max":{"px":400,"char":40}},oEncoder:null,constructs:function(_1){this.oEncoder=new vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteEncoder();this.loadConfig(_1);},loadConfig:function(_2){var t=this,tM=t.oModel;if(_2.version===null||_2.algorithm===null){tM.version=_2.algoVerMap[t.defaultAlgo];tM.algorithm=t.algoMap[t.defaultAlgo];}else{tM.version=_2.version;tM.algorithm=t.algoMap[_2.algorithm];}
tM.trkSuggest=_2.trkSuggest;tM.trkRS=_2.trkRS;tM.trkInput=_2.trkInput;tM.trkShow=_2.trkShow;tM.trkHide=_2.trkHide;tM.containerId=_2.containerId;tM.idList=_2.idList;tM.sugDivId=_2.sugDivId;tM.noSugDivId=_2.noSugDivId;tM.hideLnk=_2.hideLnk;tM.baseURL=_2.baseURL;tM.siteId=_2.siteId;tM.lastSearch=t.oEncoder.decodeCookie(_2.lastSearch);},updateWidthDef:function(_4){var t=this,inputWidth=parseInt(_4.offsetWidth,10);t.widthDef.max={"px":inputWidth,"char":Math.floor(inputWidth/t.widthDef.unit)-1};},getHideLnk:function(){return this.oModel.hideLnk;},getWidthUnit:function(){return this.widthDef.unit;},getWidthMin:function(){return this.widthDef.min;},getWidthMax:function(){return this.widthDef.max;},getRootDir:function(){return this.oModel.rootDir;},getListSize:function(){return this.oModel.listSize;},getDelayTime:function(){return this.oModel.delayTime;},getDirDepth:function(){return this.oModel.dirDepth;},getAlgorithm:function(){return this.oModel.algorithm;},getVersion:function(){return this.oModel.version;},getTrkSuggest:function(){return this.oModel.trkSuggest;},getTrkRS:function(){return this.oModel.trkRS;},getTrkInput:function(){return this.oModel.trkInput;},getTrkShow:function(){return this.oModel.trkShow;},getTrkHide:function(){return this.oModel.trkHide;},getContainerId:function(){return this.oModel.containerId;},getIdList:function(){return this.oModel.idList;},getKeyUpSvc:function(){return this.oModel.keyUpSvc;},getKeyDownSvc:function(){return this.oModel.keyDownSvc;},getInputOverSvc:function(){return this.oModel.inputOverSvc;},getUpdInputSvc:function(){return this.oModel.updInputSvc;},getFillContentSvc:function(){return this.oModel.fillContentSvc;},getInSvc:function(){return this.oModel.svcIn;},getOutSvc:function(){return this.oModel.svcOut;},getBaseURL:function(){return this.oModel.baseURL;},getSiteId:function(){return this.oModel.siteId;},getSugDivId:function(){return this.oModel.sugDivId;},getNoSugDivId:function(){return this.oModel.noSugDivId;},getNoSugShowTime:function(){return this.oModel.noSugShowTime;},getLastSearch:function(){return this.oModel.lastSearch;}});

vjo.needs("vjo.dsf.XDomainRequest");vjo.needs("vjo.dsf.utils.Timer");vjo.needs("vjo.dsf.document.Element");vjo.needs("vjo.dsf.ServiceEngine");vjo.needs("vjo.dsf.Message");vjo.needs("vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteCache");vjo.needs("vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteEncoder");vjo.type("vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteRequest").singleton().protos({aReqList:[],sRespSvc:"autofill_response",inProcess:false,bInit:false,vX:vjo.dsf.XDomainRequest,uN:"undefined",constructs:function(){},init:function(_1){var t=this;if(t.bInit){return;}
t.oConfig={baseURL:_1.baseURL,dirDepth:_1.dirDepth,rootDir:_1.rootDir,algorithm:_1.algorithm,version:_1.version,siteId:_1.siteId};t.oCache=new vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteCache();t.oEncoder=new vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteEncoder();t.oProcessTimer=new vjo.dsf.utils.Timer();t.oProcessTimer.setInterval(20);t.oProcessTimer.onTick=function(){if(t.inProcess){return;}
t.inProcess=true;try{t.processQue();t.cleanQue();}
catch(e){}
t.inProcess=false;};t.oProcessTimer.start();t.bInit=true;},processQue:function(){var t=this;if(t.aReqList.length===0){return;}
var _4=t.aReqList[0];if(_4&&(_4.state=="wait")){t.aReqList[0].state="process";var _5=t.vX.bUseIframe;try{t.vX.bUseIframe=false;t.aReqList[0].scriptId=t.vX.send(_4.url);}
catch(e){}
t.vX.bUseIframe=_5;}},cleanQue:function(){var t=this;var _7=[],i;for(i=0;i<t.aReqList.length;i++){var _8=t.aReqList[i];if(_8.state=="wait"){_7.push(_8);}else{if(_8.state=="process"){var _9=new Date(),time=_9.getTime()-_8.timestamp;if(time>=1000){t.removeScriptTag(_8.scriptId);t.sendRespService(true,_8.requester,_8.prefix,null,null,_8.shortPrefix);}else{_7.push(_8);}}}}
t.aReqList=_7;},removeScriptTag:function(_a){try{this.vX.getReqDiv().removeChild(vjo.dsf.document.Element.get(_a));}
catch(e){}},getRespSvc:function(){return this.sRespSvc;},getShortPrefix:function(_b){return _b.substr(0,this.oConfig.dirDepth+1);},buildPath:function(_c,_d){var t=this,pd=_d?10000:t.oConfig.dirDepth;var _f=_c.length>pd?_c.substr(0,pd):_c.substr(0,_c.length-1);var _10=_c.length>pd?_c.substr(pd,1):_c.substr(_c.length-1,1);var _11="";var _12=_f.toLowerCase();var _13=_10.toLowerCase();for(var i=0;i<_12.length;i++){_11+=t.oEncoder.encode(_12.charAt(i))+"/";}
return[_11,t.oEncoder.encode(_13),_f+_10];},buildURL:function(){var t=this,tO=t.oConfig,url=tO.baseURL;if(typeof(tO.version)==t.uN||tO.version===null){return null;}
if(url.lastIndexOf("/")<url.length-1){url+="/";}
url+=tO.rootDir+"/";url+=tO.algorithm+"/";url+=tO.siteId+"/";url+=tO.version+"/";return url;},addRequest:function(_16,_17,_18,_19){var t=this;var url=t.buildURL();var _1c=t.buildPath(_17,(_18=="fd"));if(url===null){return;}
if(url.lastIndexOf("/")<url.length-1){url+="/";}
url+=_1c[0]+_1c[1];if(_18=="f"&&typeof(_19)!=t.uN&&_19.length>0){url+=_19;}
url+=".js";var _1d=new Date();t.aReqList.push({"requester":_16,"prefix":_17,"shortPrefix":t.getShortPrefix(_17),"url":url,"state":"wait","timestamp":_1d.getTime(),"type":_18});},send:function(_1e,_1f){var t=this;if(!t.bInit){return;}
var _21=t.getShortPrefix(_1f);var _22=t.oCache.get(_1f,_21);if(_22===null){t.addRequest(_1e,_1f,"k");}else{if(_22.type=="f"||_22.type=="fd"){t.addRequest(_1e,_1f,_22.type,_22.keyword);}else{if(_22.type=="null"){t.sendRespService(true,_1e,_1f,[],[],_21);}else{t.sendRespService(false,_1e,_1f,_22.keyword,[],_21);}}}},sendRespService:function(_23,_24,_25,_26,_27,_28){var m=new vjo.dsf.Message(this.sRespSvc);if(_23){m.clientContext={"timeout":true,"prefix":_25,"shortPrefix":_28,"requestId":_24};}else{m.clientContext={"timeout":false,"prefix":_25,"shortPrefix":_28,"requestId":_24,"kwList":_26,"laList":_27};}
vjo.dsf.ServiceEngine.handleRequest(m);},handleResponse:function(_2a){var t=this,kw=_2a[0],i;if(!t.bInit){return;}
t.oCache.add(_2a);var _2c=[];for(i=0;i<t.aReqList.length;i++){var _2d=t.aReqList[i];if(_2d.state!="process"){continue;}
if(_2d.prefix.toLowerCase()==_2d.shortPrefix.toLowerCase()){if(_2d.prefix.toLowerCase()==kw.toLowerCase()){_2d.state="done";t.removeScriptTag(_2d.scriptId);t.sendRespService(false,_2d.requester,kw,_2a[1],_2a[2],_2d.shortPrefix);}}else{if(_2d.shortPrefix.toLowerCase()==kw.toLowerCase()){var _2e=t.oCache.get(_2d.prefix,_2d.shortPrefix);_2d.state="done";t.removeScriptTag(_2d.scriptId);if(_2d.type=="f"&&_2e.type!="k"){t.sendRespService(true,_2d.requester,_2d.prefix,null,null,_2d.shortPrefix);}else{_2c.push(_2d);}}}}
for(i=0;i<_2c.length;i++){var _2f=_2c[i];t.send(_2f.requester,_2f.prefix);}}}).inits(function(){vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteRequest=new vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteRequest();});

vjo.needs("vjo.dsf.document.Element");vjo.needs("vjo.dsf.utils.Object");vjo.needs("vjo.dsf.document.Shim");vjo.needs("vjo.dsf.EventDispatcher");vjo.needs("vjo.dsf.Message");vjo.needs("vjo.dsf.ServiceEngine");vjo.needs("vjo.dsf.cookie.VjCookieJar");vjo.needs("vjo.dsf.window.utils.VjWindowUtils");vjo.needs("vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteRequest");vjo.needs("vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteEncoder");vjo.type("vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteLayer").props({_do:function(_1){vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteRequest.handleResponse(_1);}}).protos({sCurKw:"",sLastKw:"",iCurSel:-1,aCurKwList:[],bInSugDiv:false,oRequest:vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteRequest,bLastQueryEmpty:false,oEncoder:new vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteEncoder(),VE:vjo.dsf.document.Element,VED:vjo.dsf.EventDispatcher,VO:vjo.dsf.utils.Object,VS:vjo.dsf.ServiceEngine,constructs:function(_2,_3){var t=this;var _5=t.VED,vO=t.VO,vS=t.VS;t.sRequestId=_3;t.oConfig=_2;vS.registerSvcHdl(t.oRequest.getRespSvc(),vO.hitch(t,"handleResp"));_5.add(_2.getContainerId(),"mouseover",function(){t.bInSugDiv=true;});_5.add(_2.getContainerId(),"mouseout",function(){t.bInSugDiv=false;});_5.addEventListener(window,"resize",t.onWindowResize,t);try{t.VE.promoteToBody(_2.getContainerId());}
catch(e){}
_5.add(_2.getHideLnk(),"click",function(_6){t.setHideSuggestion(true);t.selectSug(null);t.showSugDiv(false);t.setAutoComplete(true);t.showIcon(true);var _7=function(){t.createTrackingImg(t.oConfig.getTrkHide());};window.setTimeout(_7,500);return false;});vS.registerSvcHdl(_2.getOutSvc(),function(_8){var _9=_8.clientContext;t.setInput(_9.src);switch(_9.type){case"kw_keyup":t.kw_keyup(_9);break;case"kw_blur":t.kw_blur(_9);break;case"kw_keydown":t.kw_keydown(_9);break;case"kw_mouseover":t.kw_mouseover(_9);break;case"show_click":t.show_click(_9);break;}});t.updTrk(_2.getTrkInput());t.setAutoComplete(t.isHideSuggestion());},handleResp:function(_a){var t=this;var _c=t.oConfig;var _d=_a.clientContext,i;var _e=_d.kwList||[];var _f=_d.laList||[];if(t.sRequestId!=_d.requestId||_d.prefix.toLowerCase()!=t.getInputValue().toLowerCase()){return;}
var _10=t.getRecentSearch();if(_10&&_10.length>0){var _11=_10.toLowerCase();var _12=_11.indexOf(t.sCurKw.toLowerCase());var _13=false;var _14=_11;var _15=0;while(_12>=0){if(t.isWordStart(_11,_12+_15)){_13=true;break;}
_15=_12+1;_14=_14.substr(_12+1);_12=_14.indexOf(t.sCurKw.toLowerCase());}
if(_13){var _16=[];_16.push(_10);for(i=0;i<_e.length;i++){if(_e[i].toLowerCase()!=_11){_16.push(_e[i]);}}
if(_16.length>_c.getListSize()){_16.pop();}
_e=_16;}}
if(_d.timeout&&(_d.prefix==t.sCurKw)&&_e.length<1){if(!t.bLastQueryEmpty){t.showNoSugMessage(true);t.showSugDiv(true,_c.getWidthMin().px);}else{t.showSugDiv(false);}
t.bLastQueryEmpty=true;return;}
t.bLastQueryEmpty=false;t.showNoSugMessage(false);t.iCurSel=-1;t.aCurKwList=[];var _17=_c.getIdList();var _18=_17.length;if(_18>_c.getListSize()){_18=_c.getListSize();}
if(t.oInput){_c.updateWidthDef(t.oInput);}
var _19=t.getMaxKwLength(_e,_18)*_c.getWidthUnit();if(_19<_c.getWidthMin().px){_19=_c.getWidthMin().px;}else{if(_19>_c.getWidthMax().px){_19=_c.getWidthMax().px;}}
for(i=0;i<_18;i++){var en=_17[i],e=t.VE.get(en);if(e===null){continue;}
e.className="unsel";if(i<_e.length){e.innerHTML=t.genKwHTML(_e[i],t.sCurKw,_c.getWidthMax()["char"]);t.VE.toggleHideShow(en,true);var trk=(_e[i]==_10)?_c.getTrkRS():_c.getTrkSuggest();t.aCurKwList[i]={"divId":en,"sugKw":_e[i],"trk":trk};}else{t.VE.toggleHideShow(en,false);}}
t.showSugDiv(true,_19);},getInputValue:function(){var t=this;return t.oInput?t.oInput.value:"";},isHideSuggestion:function(){var vC=vjo.dsf.cookie.VjCookieJar,pbf=vC.readCookie("dp1","pbf"),bit=vC.getBitFlag(pbf,29);return bit==1;},setHideSuggestion:function(_1e){var vC=vjo.dsf.cookie.VjCookieJar,pbf=vC.readCookie("dp1","pbf");vC.writeCookielet("dp1","pbf",vC.setBitFlag(pbf,29,_1e?1:0));},getRecentSearch:function(){var lss=this.oConfig.getLastSearch();if(lss!==null&&lss.length>0){lss=lss.substring(lss.indexOf(".")+1);return lss;}
return"";},getMaxKwLength:function(_21,_22){if(!_21){return 0;}
var max=0;var _24=_21.length;if(_24>_22){_24=_22;}
for(var i=0;i<_24;i++){var len=_21[i].length;if(len>max){max=len;}}
return max;},isWordStart:function(pKw,_28){if(_28<=0||_28>pKw.length-1){return true;}
var _29=new RegExp("[\\s \\.,]");return pKw.substr(_28-1,1).search(_29)>=0;},genKwHTML:function(pKw,_2b,_2c){var _2d=_2b.length,out=pKw,t=this;var _2e=t.oEncoder,part;var _2f=-1;for(var i=0;i<pKw.length;i++){part=pKw.substr(i,_2d);if(part.length!=_2d){break;}
if(part.toLowerCase()==_2b.toLowerCase()&&t.isWordStart(pKw,i)){_2f=i;break;}}
var _31=pKw.substring(0,_2f);var _32=pKw.substr(_2f+_2d);var _33;if(_2f>=0){if(pKw.length>_2c){if(_31.length>_2c){out=_2e.encodeHTML(out.substr(0,_2c));out+="...";}else{if((_31.length+part.length)>_2c){_33=_2c-_31.length;part="<span class='hl'>"+_2e.encodeHTML(part.substr(0,_33))+"...</span>";out=_2e.encodeHTML(_31)+part;}else{_33=_2c-_31.length-part.length;out=_2e.encodeHTML(_31)+"<span class='hl'>"+_2e.encodeHTML(part)+"</span>"+_2e.encodeHTML(_32.substr(0,_33))+"...";}}}else{out=_2e.encodeHTML(_31)+"<span class='hl'>"+_2e.encodeHTML(part)+"</span>"+_2e.encodeHTML(_32);}}else{if(pKw.length>_2c){out=_2e.encodeHTML(out.substr(0,_2c));out+="...";}}
return out;},startKeyTimer:function(_34){var t=this;t.stopKeyTimer();var _36=function(){var _37=t.getInputValue();if(_34!=_37||_37.length<1){return;}
t.oRequest.send(t.sRequestId,_34);};t.oKeyTimer=window.setTimeout(_36,t.oConfig.getDelayTime());},stopKeyTimer:function(){var t=this;if(t.oKeyTimer){window.clearTimeout(t.oKeyTimer);t.oKeyTimer=null;}},getKwSelect:function(){var t=this;if(t.iCurSel<0){return t.sCurKw;}
return t.aCurKwList[t.iCurSel].sugKw;},selectSug:function(_3a,_3b,_3c){var t=this,kw=t.sCurKw;if(t.iCurSel>=0){t.unselectSug(t.aCurKwList[t.iCurSel].divId);}
if(_3a!==null){var e=t.VE.get(_3a);for(var i=0;i<t.aCurKwList.length;i++){var _40=t.aCurKwList[i];if(_40.divId==_3a){t.iCurSel=i;kw=_40.sugKw;break;}}
if(e){e.className="sel";}}else{t.iCurSel=-1;}
if(!_3c){t.updateInput(kw);}
if(!_3b){t.focusInput();}},unselectSug:function(_41){var t=this,e=t.VE.get(_41);if(e){e.className="unsel";}},createTrackingImg:function(_43){var _44=new Date(),r=_44.getTime();var _45=_43;if(_45.indexOf("?")>0){_45+="&"+r;}else{_45+="?"+r;}
var img=new Image();document.getElementsByTagName("body")[0].appendChild(img);img.setAttribute("src",_45);img.setAttribute("width","1");img.setAttribute("height","1");img.setAttribute("border","0");},sendInSvc:function(_47){var m=new vjo.dsf.Message(this.oConfig.getInSvc());m.clientContext=_47;this.VS.handleRequest(m);},isCtrlKey:function(_49){var _4a=[38,39,40,27];for(var i=0;i<_4a.length;i++){if(_4a[i]==_49){return true;}}
return false;},isIgnorKey:function(_4c){var _4d=[16,17,18];for(var i=0;i<_4d.length;i++){if(_4c==_4d[i]){return true;}}
return false;},isSugShown:function(){var t=this;var _50=t.VE.get(t.oConfig.getContainerId());disp=_50.currentStyle?_50.currentStyle.display:window.getComputedStyle(_50,null).getPropertyValue("display");return(disp!="none");},isNoSugMsgShow:function(){var t=this;var _52=t.VE.get(t.oConfig.getNoSugDivId());disp=_52.currentStyle?_52.currentStyle.display:window.getComputedStyle(_52,null).getPropertyValue("display");return(disp!="none");},showNoSugMessage:function(_53){var t=this;t.VE.toggleHideShow(t.oConfig.getSugDivId(),!_53);t.VE.toggleHideShow(t.oConfig.getNoSugDivId(),_53);if(_53){var _55=function(){if(t.isNoSugMsgShow()){t.showSugDiv(false);}};window.setTimeout(_55,t.oConfig.getNoSugShowTime());}},showSugDiv:function(_56,_57){var t=this;if(t.isHideSuggestion()&&_56){t.showIcon(true);return;}
var vS=vjo.dsf.document.Shim;var _5a=t.oConfig.getContainerId();var _5b=t.VE.get(_5a);if(_56){t.posLayer(_57);t.VE.toggleHideShow(_5a,true);if(t.oIframeShim){vS.remove(_5b,t.oIframeShim);}
t.oIframeShim=vS.add(_5b);var _5c=function(){t.regOverEvent(true);};window.setTimeout(_5c,100);}else{t.VE.toggleHideShow(_5a,false);if(t.oIframeShim){vS.remove(_5b,t.oIframeShim);t.oIframeShim=null;}
t.regOverEvent(false);}},regOverEvent:function(_5d){var t=this;var _5f=t.oConfig.getIdList(),i,id;for(i=0;i<_5f.length;i++){id=_5f[i];t.VED.unregister(id,"mouseover");t.VED.unregister(id,"click");if(_5d){t.VED.add(id,"mouseover",function(_60){t.bInSugDiv=true;t.selectSug(_60.src.id,false,true);});t.VED.add(id,"click",function(_61){t.selectSug(_61.src.id);t.submitForm();t.showSugDiv(false);});}}},posLayer:function(_62){var t=this;var _64=t.getAbsPos(t.oInput);if(_64===null){return false;}
var e=t.VE.get(t.oConfig.getContainerId());if(e===null){return false;}
e.style.left=_64.left+"px";e.style.top=_64.top+_64.height+"px";if((typeof(_62)!="undefined")&&(_62!==null)){e.style.width=parseInt(_62,10)+"px";}
return true;},getAbsPos:function(_66){var vW=vjo.dsf.window.utils.VjWindowUtils,t=this;if(_66===null){return null;}
return{"left":parseInt(vW.offsetLeft(_66),10),"top":parseInt(vW.offsetTop(_66),10),"height":parseInt(_66.offsetHeight,10),"width":parseInt(_66.offsetWidth,10)};},updateInput:function(pKw){var t=this;if(t.isSugShown()){t.sLastKw=pKw;t.sendInSvc({"type":"kw_updvalue","value":pKw});}},updTrk:function(_6a){this.sendInSvc({"type":"search_updtrk","lnkStr":_6a});},submitForm:function(){var t=this;if(t.iCurSel>=0){var trk=t.aCurKwList[t.iCurSel].trk;t.updTrk(trk);}
this.sendInSvc({"type":"search_submit"});},showIcon:function(_6d){this.sendInSvc({"type":"sug_icon_show","bShow":_6d});},focusInput:function(){this.sendInSvc({"type":"kw_focus"});},setAutoComplete:function(bOn){this.sendInSvc({"type":"kw_autocomplete","bOn":bOn});},setInput:function(_6f){if(!_6f){return;}
var t=this;if(!t.oInput){t.oInput=_6f;}
t.oConfig.updateWidthDef(_6f);},show_click:function(_71){var t=this;t.showIcon(false);t.setHideSuggestion(false);t.bLastQueryEmpty=false;t.oRequest.send(t.sRequestId,t.getInputValue());var _73=function(){t.createTrackingImg(t.oConfig.getTrkShow());};window.setTimeout(_73,500);t.setAutoComplete(false);},kw_blur:function(_74){var t=this;if(t.isSugShown()&&!t.isNoSugMsgShow()){if(!t.bInSugDiv){t.selectSug(null,true);t.showSugDiv(false);}else{t.focusInput();}}},kw_mouseover:function(_76){var t=this;if(t.isSugShown()&&!t.isNoSugMsgShow()){t.selectSug(null);}},kw_keydown:function(_78){var t=this,kc=_78.keyCode;if(kc==13&&t.isSugShown()&&!t.isNoSugMsgShow()){if(t.iCurSel>=0){t.updTrk(t.aCurKwList[t.iCurSel].trk);}
t.showSugDiv(false);}},kw_keyup:function(_7a){var t=this,kc=_7a.keyCode,cv=_7a.value;if(t.isIgnorKey(kc)){return;}
if(t.isSugShown()&&!t.isNoSugMsgShow()){if(t.isCtrlKey(kc)){var e,nextId,divId;switch(kc){case 38:nextId=t.iCurSel-1;if(nextId<-1){nextId=t.aCurKwList.length-1;}
divId=nextId>=0?t.aCurKwList[nextId].divId:null;t.selectSug(divId);t.iCurSel=nextId;break;case 40:nextId=t.iCurSel+1;if(nextId>=t.aCurKwList.length){nextId=-1;}
divId=nextId>=0?t.aCurKwList[nextId].divId:null;t.selectSug(divId);break;case 39:if(cv.length!==0){t.sCurKw=cv;t.startKeyTimer(cv);t.sLastKw=cv;}
break;case 27:t.selectSug(null);t.showSugDiv(false);break;}}else{if(cv.length!==0){if(t.sLastKw!=cv){t.sCurKw=cv;t.startKeyTimer(cv);t.sLastKw=cv;}}else{t.showSugDiv(false);t.showIcon(false);t.sCurKw=cv;t.sLastKw=cv;t.bLastQueryEmpty=false;}}}else{if(cv.length!==0){if(t.sLastKw!=cv){t.sCurKw=cv;t.startKeyTimer(cv);t.sLastKw=cv;}}else{t.sCurKw=cv;if(t.isHideSuggestion()){t.showIcon(false);}
t.sLastKw=cv;t.bLastQueryEmpty=false;}}},onWindowResize:function(_7d){if(this.isSugShown()){this.showSugDiv(true);}}});

vjo.needs("vjo.dsf.ServiceEngine");vjo.needs("vjo.dsf.EventDispatcher");vjo.needs("vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteConfig");vjo.needs("vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteLayer");vjo.needs("vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteRequest");vjo.type("vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteLayerInit").protos({constructs:function(_1){this.oModel=_1;var t=this;var _3=function(){var _4=new vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteConfig(t.oModel);var _5={baseURL:_4.getBaseURL(),dirDepth:_4.getDirDepth(),rootDir:_4.getRootDir(),algorithm:_4.getAlgorithm(),version:_4.getVersion(),siteId:_4.getSiteId()};vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteRequest.init(_5);new vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteLayer(_4,t.oModel.requestId);};vjo.dsf.EventDispatcher.addEventListener(window,"load",_3,window);vjo.type("vjo.darwin.domain.finding.autofill.AutoFill").props({_do:function(_6){vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteLayer._do(_6);}});}}).props({init:function(_7){new vjo.darwin.core.ebayheader.autocomplete.layer.AutoCompleteLayerInit(_7);}});

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

vjo.ctype("vjo.darwin.core.ebayheader.rover.FooterRover").needs("vjo.dsf.cookie.VjCookieJar").props({roverService:function(_1){this.command=_1;if(!_1){return;}
if(!this.isCookieValid()){return;}
vjo.dsf.EventDispatcher.addEventListener(window,"load",this.sendRequest,this);},sendRequest:function(){var _2=new vjo.dsf.assembly.VjClientAssemblerRequest(this.command,this.handleResponse,this,"cb",false);vjo.dsf.assembly.VjClientAssembler.load(_2);},isCookieValid:function(){var _3=vjo.dsf.cookie.VjCookieJar.readCookie("dp1","idm");if(!_3){return true;}else{return false;}},handleResponse:function(_4){if(_4&&_4.length>1){var _5=_4.length-1;for(i=0;i<_5;i++){this.createImage(_4[i]);}
this.setCookieExpiration(_4[_5]);}},createImage:function(_6){if(_6&&_6.length>1){var _7=document.createElement("IMG");_7.width="1";_7.height="1";_7.src=_6;document.body.appendChild(_7);}},setCookieExpiration:function(_8){if(typeof _8=="number"&&_8>0){vjo.dsf.cookie.VjCookieJar.writeCookielet("dp1","idm","1",_8/86400,"");}}}).endType();

// en_GB/e615i/SYS14_vjo_e615i8637324_1_en_GB
// b=8637324