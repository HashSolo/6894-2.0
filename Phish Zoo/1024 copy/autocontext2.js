if(typeof Begun!=="object"){
var Begun={};
}
if(typeof Begun.Error!=="object"){
Begun.Error={};
}
if(typeof Begun.loaderCallbacks==="undefined"){
Begun.loaderCallbacks=[];
}

Begun.loaderCallbacks.push(begun_load_autocontext);
function begun_load_autocontext(){

Begun.Scripts.importAllScripts({"acp/begun_utils.38449.js":true});

if(typeof Begun.Error.send==="undefined"){
Begun.Error.send=function(errorMessage,errorUrl,errorLine,loggerAddress){
if(typeof Begun.Error.sent[errorMessage]!=="undefined"){
return;
}
var defaultErrorLogger="http://autocontext.begun.ru/log_errors?";
var address=loggerAddress||window.begun_error_url||defaultErrorLogger;
var padId=window.begun_auto_pad;
var img=(new Image()).src=address+"e_msg="+encodeURIComponent(errorMessage)+"&e_url="+
encodeURI(errorUrl)+"&e_line="+errorLine+
"&pad_id="+padId+"&location="+encodeURI(document.location);
Begun.Error.sent[errorMessage]=true;
};
Begun.Error.sent={};
}

(function(){
if(!Begun.Autocontext){
var errorHandler=window.onerror;
window.onerror=function regErrors(msg,url,line){
if(errorHandler&&errorHandler instanceof Function){
errorHandler();
}
if(typeof url==="string"&&url.search(/autocontext/i)!==-1||typeof msg==="string"&&msg.search(/Begun/i)!==-1){
Begun.Error.send(msg,url,line);
}
};
}
})();

Begun.DOM_TIMEOUT=1;
Begun.REVISION='$LastChangedRevision: 39132 $';

Begun.Scripts.Callbacks['ac']=function(fileName){
if(!Begun.Autocontext&&Begun.Scripts.isLastRequired(fileName)){
Begun.Autocontext=new function(){
var _this=this;
this.dom_change=false;
this.multiple_feed=true;
this.scrollers=[];
this.options={
max_blocks_count:10,
max_scrollers:1,
max_scroll_banners:10,
fake_block_offset:200,
fake_block_high_limit:251
};
this.unhandledDebugs=[];

this.Thumbs=(function(){
var types={
'default':{
'width':42,
'height':42
},
'classic':{
'width':56,
'height':42
}
}
return{
getType:function(pad_id){
var sections=['autocontext','hypercontext'];
var banner=null;
for(var i=0;i<sections.length;i++){
if(_this.getBanner(sections[i],0,pad_id)){
banner=_this.getBanner(sections[i],0,pad_id);
break;
}
}
return _this.getThumbSrc(banner,false).indexOf('classic')!=-1?'classic':'default';
},
getDimentions:function(type){
return types[type];
}
}
})();

var Module=(function(){
var ext={
"auto_ppcall":"38520",
"auto_scroll":"39066",
"auto_rich":"38632",
"auto_accordion":"30573",
"auto_top":"34107",
"auto_catalog":"30833",
"auto_hyper":"36003",
"auto_photo":"36952",
"toolbar":"38449",
"catalog_tree":"1"
};
var loaded=[];
return{
updateAnticash:function(link){
var reModule=new RegExp("^"+_this.Strings.urls.base_scripts_url+"(\\w+)\.js$");
var parsed=reModule.exec(link);
if((parsed!==null)&&(parsed.length===2)&&
typeof ext[parsed[1]]!=="undefined"){
return ext[parsed[1]];
}else{
return false;
}
},
isLoaded:function(link){
return Begun.Utils.in_array(loaded,link);
},
load:function(link){
if(!this.isLoaded(link)){
var revNumber=this.updateAnticash(link);
if(revNumber){
Begun.Utils.includeScript(link.replace(/(.+)\/([^\/]+)\.js$/,"$1/acp/$2."+revNumber+".js",'write'));
}else{
Begun.Utils.includeScript(link,'write');
}
loaded.push(link);
}
},
getNames:function(what){
switch(what){
case"loaded":
return loaded.toString();
case"all":
default:
var allModules='';
var comma='';
for(var aModule in ext){
if(ext.hasOwnProperty(aModule)){
allModules+=comma+aModule;
comma=',';
}
}
return allModules;
}
},
initInBlock:function(block,pad){
var initExtraModule=function(objName,func,object){
if(!window.Begun[objName]){
window.setTimeout(function(){
initExtraModule(objName,func,object);
},Begun.DOM_TIMEOUT);
}else{
func(object);
}
};
if(_this.bannersContainViewType('rich',pad.pad_id)){
var richBlocksDiv=_this.Blocks.getDomObj(block.id);
if(richBlocksDiv){
Begun.Utils.addClassName(richBlocksDiv,'begun_auto_rich');
}
initExtraModule('richBlocks',_this.initAutoRichBlock,block);
}else if(Number(block.options.use_scroll)){
initExtraModule('Scroller',_this.initScrollBlock,block);
}else if(Number(block.options.use_accordion)){
initExtraModule('Accordion',_this.initAccordionBlock,block);
}else if(_this.Blocks.checkType(block,'top')){
initExtraModule('autoTop',_this.initAutoTopBlock,block);
}
}
};
})();

this.tplLoaded=function(tpl){
this.tplLoaded.notFinished[tpl]=false;
if(this.fillBlocks.delayedCall&&!ExtBlockTypes.isLoading()){
this.fillBlocks();
this.fillBlocks.delayedCall=false;
}
};
this.tplLoaded.notFinished={};

var ExtBlockTypes=(function(){
var ext={
"begun_tpl_block_120x600":"38796",
"begun_tpl_block_160x600":"38796",
"begun_tpl_block_200x300":"38796",
"begun_tpl_block_240x400":"38796",
"begun_tpl_block_468x60":"38796",
"begun_tpl_block_728x90":"38796",
"begun_tpl_block_flat":"38579",
"begun_tpl_block_horizontal":"37731",
"begun_tpl_block_rich":"38796",
"begun_tpl_block_square":"36621",
"begun_tpl_block_top":"36547",
"begun_tpl_block_vertical":"36547"
}

return{
isLoading:function(){
var isAny=false;
for(var status in _this.tplLoaded.notFinished){
if(_this.tplLoaded.notFinished.hasOwnProperty(status)){
if(_this.tplLoaded.notFinished[status]){
isAny=true;
break;
}
}
}
return isAny;
},
load:function(tplFileName){
if(typeof ext[tplFileName]==="undefined"){
return;
}
if(typeof _this.tplLoaded.notFinished[tplFileName]==="undefined"){
_this.tplLoaded.notFinished[tplFileName]=true;
Begun.Utils.includeScript(_this.Strings.urls.base_scripts_url+"acp/"+
tplFileName+"."+ext[tplFileName]+".js","write");
}
},
loadAll:function(){
for(var tplFileName in ext){
if(ext.hasOwnProperty(tplFileName)){
this.load(tplFileName);
}
}
}
};
})();

this.getModules=Module.getNames;

var UA_DESKTOP=0;
var UA_CLASSIC_MOBILE=1;
var UA_RICH_MOBILE=2;
var BLOCK_ID_TOP_MOBILE=2;
var FAKE_BLOCKS=[BLOCK_ID_TOP_MOBILE];
this.getBlockIdTopMobile=function(){
return BLOCK_ID_TOP_MOBILE;
};


this.Strings={
urls:{
begun:'http://www.begun.ru/',
autocontext:'http://autocontext.begun.ru/',
base_scripts_url:'http://autocontext.begun.ru/',
daemon:'http://autocontext.begun.ru/context.jsp?',
video:'http://video.begun.ru/vcp.swf',
thumbs:'http://thumbs.begun.ru/',
blank:'http://autocontext.begun.ru/img/blank.gif',
block_counter:'http://autocontext.begun.ru/blockcounter?pad_id={{pad_id}}&block={{block_id}}',
log_banners_counter:'http://autocontext.begun.ru/blockcounter?data={{data}}&log_visibility=1',
ppcalls_counter:'http://autocontext.begun.ru/ppcallcounter?data={{ppcall_data}}&ppcall_form={{ppcall_form}}&log_ppcall_visibility=1',
hyper_shadow_1:'http://autocontext.begun.ru/img/hyper-shadow-1.png',
hyper_shadow_2:'http://autocontext.begun.ru/img/hyper-shadow-2.png',
alter_geo_logger:'http://autocontext.begun.ru/altergeo?token={{token}}'
},
stubs:{
place_here:'&#1044;&#1072;&#1090;&#1100; &#1086;&#1073;&#1098;&#1103;&#1074;&#1083;&#1077;&#1085;&#1080;&#1077;',
all_banners:'&#1042;&#1089;&#1077; &#1086;&#1073;&#1098;&#1103;&#1074;&#1083;&#1077;&#1085;&#1080;&#1103;',
become_partner:'&#1057;&#1090;&#1072;&#1090;&#1100; &#1087;&#1072;&#1088;&#1090;&#1085;&#1077;&#1088;&#1086;&#1084;'
},
contacts:{
card:'&#1050;&#1086;&#1085;&#1090;&#1072;&#1082;&#1090;&#1099;',
ppcall:'&#1047;&#1074;&#1086;&#1085;&#1080;&#1090;&#1100;'
},
css:{
prefix:'begun',
block_prefix:'begun_block_',
scroll_table_prefix:'begun_adv_table_',
scroll_div_prefix:'begun_adv_common_',
catalog_search_wrapper:'begun_catalog_search_span',
catalog_results_wrapper:'begun_catalog_results_span',
catalog_cloud_wrapper:'begun_catalog_cloud_span',
thumb:'begun_adv_thumb',
thumb_default:'begun_adv_thumb_default',
thumb_classic:'begun_adv_thumb_classic',
favicon:'begun_adv_fav',
scroll:'begun_scroll',
alco_prefix:'begun_alco_',
logo_color:'#622678',
thumb_def_color:'#118F00',
thumb_def_color_hover:'#FF0000',
fix_layout:'begun_fix_layout'
},
js:{
banner_onclick:'Begun.Autocontext.clickBanner(event, this)',
ppcall_show:'Begun.Ppcall.showEnterForm'
}
};

var isBFSApplicable=function(){
return(typeof window.begun_multiple_feed!=="undefined"||_this.multiple_feed)&&!window.begun_block_ids;
};

var addFakeBlocks=function(){
var feed=_this.getFeed();
if(feed.blocks){
var _block=null;
var i=0;
while(_block=feed.blocks[i]){
if((_block)&&_block.id==BLOCK_ID_TOP_MOBILE){
_this.Blocks.add(_block);
}
i++;
}
}
}

var LoadingStrategy=function(){};
LoadingStrategy.prototype={
loadBlock:function(block_id){},
parseFeed:function(){}
};


var DS=function(){};
DS.prototype=new LoadingStrategy();
DS.prototype.loadBlock=function(block_id){
this.block_id=block_id;
if(!_this.initFeedLoad()){
_this.loadFeedDone();
}
arguments.callee=LoadingStrategy.loadBlock;
};
DS.prototype.parseFeed=function(){
_this.loadExtraResources();
var feed=_this.getFeed();
if(feed&&feed.blocks&&this.block_id){
addFakeBlocks();
var block=_this.Blocks.getBlockById(this.block_id,feed.blocks);
if(block){
_this.Blocks.push(block);
}
}
};


var FBS=function(){};
FBS.prototype=new LoadingStrategy();
FBS.prototype.parseFeed=function(){
var extendUndefinedFields=function(destination,source){
for(var property in source){
if(source.hasOwnProperty(property)){
if(typeof destination[property]==="undefined"){
destination[property]=source[property];
}else if(typeof destination[property]==="object"&&typeof source[property]==="object"){
destination[property]=extendUndefinedFields(destination[property],source[property]);
}
}
}
return destination;
};
_this.loadExtraResources();
var feed=_this.getFeed();
if(feed&&feed.blocks&&this.block_id){
addFakeBlocks();
var block=_this.Blocks.getBlockById(this.block_id,feed.blocks);
if(block){
extendUndefinedFields(block.options,window.begun_extra_block.options);
window.begun_extra_block.id=window.begun_block_id;
_this.Blocks.push(window.begun_extra_block);
}
}
};
FBS.prototype.loadBlock=function(block_id){
var feed=_this.getFeed();
if(feed&&feed.blocks){
_this.resetBannerIndex();
var sBanners=_this.getShownBanners();
if(typeof sBanners==="object"){
sBanners=sBanners.toString();
}else{
sBanners="";
}
_this.feedLoad({"banner_filter":sBanners});
}
this.block_id=block_id;
};


var BFS=function(){};
BFS.prototype=new LoadingStrategy();
BFS.prototype.loadBlock=function(block_id){
this.block_id=block_id;
var feed=_this.getFeed();
if(feed&&feed.blocks){
_this.resetBannerIndex();
var sBanners=_this.getShownBanners();
if(typeof sBanners==="object"){
sBanners=sBanners.toString();
}else{
sBanners="";
}
_this.feedLoad({"banner_filter":sBanners});
}
};
BFS.prototype.parseFeed=function(){
(new DS).parseFeed.apply(this);
};

this.getLoadingStrategy=function(){
if(window.begun_extra_block){
if(!arguments.callee.fbs){
arguments.callee.fbs=new FBS();
}
return arguments.callee.fbs;
}else if(isBFSApplicable()){
if(!arguments.callee.bfs){
arguments.callee.bfs=new BFS();
}
return arguments.callee.bfs;
}else{
if(!arguments.callee.ds){
arguments.callee.ds=new DS();
}
return arguments.callee.ds;
}
};
this.setOptions=function(options){
Begun.extend(_this.options,options||{});
};
this.requestParams={
"pad_id":'',
"block_id":'',
"n":'',
"lmt":Date.parse(document.lastModified)/1000,
"sense_mode":'custom',
"ut_screen_width":screen.width||0,
"ut_screen_height":screen.height||0,
"json":1,
"jscall":'loadFeedDone',
"condition_id":window.begun_condition_id||'',
"frm_level":'',
"frm_top":'',
"force_js_link":'',
"hookData":'',
"misc_id":window.begun_misc_id||window.misc_id,
"overridden":'',
"version":'',
"banner_filter":'',
"stopwords":window.stopwords||'',
"begun_self_keywords":window.begun_self_keywords||'',
"ref":document.referrer,
"real_refer":document.location
};
this.responseParams={};
this.prepareRequestParams=function(newValues){
var comma="";
_this.requestParams.pad_id=window.begun_auto_pad;
if(self.parent){
try{
var extra_data=self.parent.document.getElementById('bottomBannerDataId')||null;
if(extra_data&&extra_data.innerHTML){
_this.requestParams.hookData=extra_data.innerHTML;
}
}catch(e){}
}
var version=Begun.REVISION.replace(/\D/gi,'');
_this.requestParams.version=version;
if(typeof(window.begun_js_force_load)!='undefined'&&window.begun_js_force_load){
_this.requestParams.force_js_link=Module.getNames('all');
ExtBlockTypes.loadAll();
var moduleNames=_this.requestParams.force_js_link.split(',');
var baseUrl=_this.Strings.urls.base_scripts_url;
for(var j=0;j<moduleNames.length;j++){
if(moduleNames[j]!=="toolbar"){
Module.load(baseUrl+moduleNames[j]+".js");
}
}
}
var frame_level=(function(){
var level=0;
var _parent=self;
while(_parent!==top&&level<999){
_parent=_parent.parent;
level++;
}
return level;
})();
if(frame_level){
_this.requestParams.frm_level=frame_level;
try{
_this.requestParams.frm_top=top.location.href;
}catch(exc){
_this.requestParams.frm_top='top not accessible';
}
}

if(typeof Begun.Autocontext.isNotFirstRequest==="undefined"){
Begun.Autocontext.isNotFirstRequest=true;
comma=",";
_this.requestParams.block_id=BLOCK_ID_TOP_MOBILE;
window.altergeo_token_callback=function(token){
Begun.Utils.includeCounter(_this.Strings.urls.alter_geo_logger,{"token":token});
};
}

if(window.begun_block_ids){
Begun.Utils.includeCounter("http://autocontext.begun.ru/blockcustom?pad_id={{pad_id}}&log_block_ids=1",{"pad_id":(window.begun_auto_pad||'')});
_this.requestParams.block_id+=comma+window.begun_block_ids.replace(/\s/g,"");
}else{
if(window.begun_block_id&&isBFSApplicable()){
_this.requestParams.block_id+=comma+window.begun_block_id;
}
}

if(window.begun_request_params&&window.begun_request_params.constructor===Object){
window.begun_request_params.overridden=1;
Begun.extend(_this.requestParams,window.begun_request_params);
}
if(newValues){
Begun.extend(_this.requestParams,newValues);
}
var thePad=_this.getPad();
if(typeof thePad.rq==="undefined"){
thePad.rq=0;
}else{
thePad.rq++;
}
if(typeof Begun.pageId==="undefined"){
Begun.pageId=Math.floor(10000000000000*Math.random()+(new Date()).valueOf());
}
_this.requestParams.rq=(thePad.rq).toString();
_this.requestParams.rq_sess=Begun.pageId;
};
this.isEventTrackingOn=function(){
return _this.responseParams["track_events"];
};
this.init=function(){
_this.Customization.init();
_this.Pads.init();
_this.initCurrentBlock();
if(typeof arguments.callee.run==="undefined"){
arguments.callee.run=true;
}
};
this.initToolbar=function(debug){
if(Begun.Toolbar){
var toolbar=Begun.Toolbar.init(debug);
}
};
this.initHypercontextBlock=function(block,pad_id){
if(!Begun.Hypercontext||!block){
return;
}
this.hyperBlock=new Begun.Hypercontext(block,pad_id);
};
this.initPhotocontextBlock=function(block,pad_id){
if(!Begun.Photocontext||!block){
return;
}
this.photoBlock=new Begun.Photocontext(block,pad_id);
};
this.initScrollBlock=function(block){
if(Begun.Scroller){
var setHeight=function(block,scroll_div,scroll_table){
var trs=scroll_table.getElementsByTagName('tr');
var height;
var i;
var banners_count=Number(block.options.banners_count);
if(banners_count===1){
height=trs[0].offsetHeight;
for(i=1;i<trs.length;i++){
if(trs[i].offsetHeight>height){
height=trs[i].offsetHeight;
}
}
}else{
height=0;
for(i=0;i<banners_count;i++){
if(trs[i]){
var h=trs[i].offsetHeight;
height+=h;
}
}
}
scroll_div.style.height=height+'px';
scroll_div.style.overflow='hidden';
};
var init=function(block,scroll_div,scroll_table,is_horizontal){
var banners_count=Number(block.options.banners_count);
var banners_count_coef=Number(block.options.banners_count_coef)||1;
(function(){
if(!scroll_div.offsetHeight){
window.setTimeout(arguments.callee,Begun.DOM_TIMEOUT);
}
var scroller=(new Begun.Scroller(
scroll_table,
{
height:scroll_div.offsetHeight,
banners_count:banners_count,
banners_count_coef:banners_count_coef,
is_horizontal:is_horizontal,
scroll_timeout:(block&&block.options&&block.options.json&&block.options.json.scroll_timeout)||null,
fade_enabled:!Begun.Browser.IE
}
));

_this.scrollers.push(scroller);
scroller.start();
})();
};
var scroll_div=Begun.$(_this.Strings.css.scroll_div_prefix+block.id);
var scroll_table=Begun.$(_this.Strings.css.scroll_table_prefix+block.id);
var is_horizontal;
if(!block.scrolling&&Number(block.options.use_scroll)&&scroll_div&&scroll_table){
if(_this.Blocks.checkType(block,'horizontal')||_this.Blocks.checkType(block,'728x90')||_this.Blocks.checkType(block,'468x60')){
is_horizontal=true;
}else{
is_horizontal=false;
}
if(_this.Blocks.checkType(block,'vertical')||_this.Blocks.checkType(block,'flat')){
(function(block,scroll_div,scroll_table,is_horizontal){
if(scroll_table.offsetHeight){
setHeight(block,scroll_div,scroll_table);
scroll_div.style.width=scroll_div.offsetWidth+'px';
init(block,scroll_div,scroll_table,is_horizontal);
}else{
var func=arguments.callee;
window.setTimeout(function(){
func(block,scroll_div,scroll_table,is_horizontal);
},Begun.DOM_TIMEOUT);
}
})(block,scroll_div,scroll_table,is_horizontal);
}else if(_this.Blocks.checkType(block,'horizontal')){
(function(block,scroll_div,scroll_table,is_horizontal){
if(scroll_div.offsetHeight){
var tds=scroll_table.getElementsByTagName('td');
var displayProperty=Begun.Browser.IE&&Begun.Browser.version()<8?'':'table-cell';
var getScrollDivHeight=function(firstHalf){
for(var i=0;i<tds.length;i++){
if(i<Math.round(tds.length/2)){
tds[i].style.display=firstHalf?displayProperty:'none';
}else{
tds[i].style.display=firstHalf?'none':displayProperty;
}
tds[i].style.width=Math.floor(100*2/tds.length)+'%';
}
return scroll_div.offsetHeight;
}
scroll_div.style.height=Math.max(getScrollDivHeight(true),getScrollDivHeight(false))+'px';
init(block,scroll_div,scroll_table,is_horizontal);
}else{
var func=arguments.callee;
window.setTimeout(function(){
func(block,scroll_div,scroll_table,is_horizontal);
},Begun.DOM_TIMEOUT);
}
})(block,scroll_div,scroll_table,is_horizontal);
}else{
init(block,scroll_div,scroll_table,is_horizontal);
}
block.scrolling=true;
}
}
};
this.initAccordionBlock=function(block){
if(!Begun.Accordion){
return false;
}
var accordion_div=_this.Blocks.getDomObj(block.id);
if(!block.is_accordion_processing&&Number(block.options.use_accordion)&&accordion_div){
var accordion=(new Begun.Accordion(accordion_div));
accordion.init();
block.is_accordion_processing=true;
}
};
this.initAutoTopBlock=function(block){
if(!Begun.autoTop){
return false;
}
var auto_top_div=_this.Blocks.getDomObj(block.id);
if(!block.is_auto_top_processing&&_this.Blocks.checkType(block,'top')&&auto_top_div){
var divs=auto_top_div.getElementsByTagName('div');
var auto_top_div_inner=null;
for(var i=0,l=divs.length;i<l;i++){
if(Begun.Utils.hasClassName(divs[i],'begun_collapsable')){
auto_top_div_inner=divs[i];
}
}
var body=document.getElementsByTagName('body')&&document.getElementsByTagName('body')[0];
if(Begun.Browser&&Begun.Browser.IE&&(document.documentElement||body)&&auto_top_div_inner){
auto_top_div_inner.style.width=(document.documentElement.clientWidth||body.clientWidth)+'px';
window.onresize=function(){
auto_top_div_inner.style.width=(document.documentElement.clientWidth||body.clientWidth)+'px';
};
if(Begun.Browser.version()<=6){
auto_top_div.style.display='none';
}
}
var auto_top=(new Begun.autoTop(auto_top_div));
auto_top.init();
block.is_auto_top_processing=true;
}
};
this.getRichSizes=function(img,max){
var is_img_properties_equal=img.width==img.height;
var res={};
if(!is_img_properties_equal){
var max_property=img.width>img.height?'width':'height';
var min_property=max_property=='width'?'height':'width';
var ratio=img[min_property]/img[max_property];
res[max_property]=max;
img.style[max_property]=max+'px';
var max_min_property=Math.round(max*ratio);
res[min_property]=max_min_property;
img.style[min_property]=max_min_property+'px';
}else{
res.width=max;
res.height=max;
img.style.width=max+'px';
img.style.height=max+'px';
}
return res;
};
this.callRich=function(options,rich_blocks_div,block){
var rich_blocks=(new Begun.richBlocks(rich_blocks_div,options));
rich_blocks.init();
block.is_rich_blocks_processing=true;
};
this.initAutoRichBlock=function(block){
if(!Begun.richBlocks){
return false;
}
var rich_blocks_div=_this.Blocks.getDomObj(block.id);
if(!block.is_rich_blocks_processing&&rich_blocks_div){
var options={};
options.is_block_240x400=(_this.Blocks.checkType(block,'rich')||_this.Blocks.checkType(block,'240x400'));
var min=70;
var max=200;
if(typeof _this._big_rich_sizes==="undefined"){
_this._big_rich_sizes={};
}
var small_images=[];
var i;
var l;
var cells=Begun.Utils.getElementsByClassName(rich_blocks_div,'td','begun_adv_rich');
var ln=0;
for(i=0,l=cells.length;i<l;i++){
small_images[i]=Begun.Utils.getElementsByClassName(cells[i],'img','begun_adv_picture')&&Begun.Utils.getElementsByClassName(cells[i],'img','begun_adv_picture')[0];
if(small_images[i]){
ln+=2;
}
}
block.ln=ln;
for(i=0,l=small_images.length;i<l;i++){
if(small_images[i]){
var setSizes=function(num,image,max,key,obj,block){
var sizes=_this.getRichSizes.call(obj,image,max);
obj._big_rich_sizes['img_width_'+key+'_'+num]=sizes.width;
obj._big_rich_sizes['img_height_'+key+'_'+num]=sizes.height;
if(--block.ln==0){
options.num_pics=l;
options.sizes=obj._big_rich_sizes;
obj.callRich(options,rich_blocks_div,block);
image.onload=null;
}
}
var detectImgDimensions=function(img,i,value,key,obj,block){
if(img.complete){
setSizes(i,img,value,key,obj,block);
}else{
window.setTimeout((function(img,i,value,key,obj,block){
return function(){
detectImgDimensions(img,i,value,key,obj,block);
};
})(img,i,value,key,obj,block),Begun.DOM_TIMEOUT);
}
};
detectImgDimensions(small_images[i],i,min,'min',_this,block);
var big=new Image();
big.src=small_images[i].getAttribute('_big_photo_src');
detectImgDimensions(big,i,max,'max',_this,block);
}
}
}
};
this.initAutoCatalogBlock=function(block){
if(!Begun.Catalog){
return false;
}
if(!block.is_catalog_processing){
var feed=this.getFeed();
var catalog=(new Begun.Catalog(block,feed));
catalog.init();
block.is_catalog_processing=true;
}
};
this.resetMaxScrollers=function(){
_this.maxScrollers=_this.options.max_scrollers;
};
this.loadExtraResources=function(){
var feed=_this.getFeed();
if(!feed){
return;
}
var links=feed.links;
if(links){
var i=0;
var link=null;
while(link=links[i]){
switch(link.type){
case'js':
Module.load(link.url);
break;
case'css':
Begun.Utils.includeCSSFile(link.url);
break;
case'frame':
var vars={url:link.url};
document.write((new Begun.Template(_this.Tpls.getHTML('link_iframe'))).evaluate(vars));
break;
case'swf':
var isFlashInstalled=Begun.Utils.checkFlash();
if(isFlashInstalled){
var swf_url=link.url;
Begun.Utils.addEvent(window,'load',function(){
Begun.Utils.includeSWF(swf_url);
});
}
break;
case'img':
default:
(new Image()).src=link.url;
}
i++;
}
}
var isPpcall;
var isScroll;
var isRich;
var isTop;
var isHyper;
var isCatalog;
var isAccordion;
var isPhoto;
var isToolbar;
var blocks=feed.blocks;
for(var k=0;k<blocks.length;k++){
if(_this.Blocks.checkType(blocks[k],'top')){
isTop=true;
}
if(_this.Blocks.checkType(blocks[k],'hyper')){
isHyper=true;
}
if(_this.Blocks.checkType(blocks[k],'photo')){
isPhoto=true;
}
if(blocks[k].options){
if(Begun.Utils.inList(blocks[k].options.block_options,'JSCatalog')){
isCatalog=true;
}
if(blocks[k].options.use_scroll){
isScroll=true;
}
if(blocks[k].options.use_accordion){
isAccordion=true;
}
}
}
var feedBanners=feed.banners;
for(var bannersGroup in feedBanners){
if(feedBanners.hasOwnProperty(bannersGroup)&&feedBanners[bannersGroup].length){
for(var j=0;j<feedBanners[bannersGroup].length;j++){
if(feedBanners[bannersGroup][j].ppcall){
isPpcall=true;
}
if(feedBanners[bannersGroup][j].view_type&&feedBanners[bannersGroup][j].view_type.toLowerCase().indexOf("rich")!==-1){
isRich=true;
}
}
}
}
if(feed.debug&&feed.debug.request){
isToolbar=true;
}
var loadModule=function(flag,jsFile){
var baseUrl=_this.Strings.urls.base_scripts_url;
if(flag){
Module.load(baseUrl+jsFile);
}
};
loadModule(isPpcall,"auto_ppcall.js");
loadModule(isScroll,"auto_scroll.js");
loadModule(isRich,"auto_rich.js");
loadModule(isTop,"auto_top.js");
loadModule(isHyper,"auto_hyper.js");
loadModule(isCatalog,"catalog_tree.js");
loadModule(isCatalog,"auto_catalog.js");
loadModule(isAccordion,"auto_accordion.js");
loadModule(isPhoto,"auto_photo.js");
loadModule(isToolbar,"toolbar.js");
};
this.draw=function(){
if(!arguments.callee.run){
_this.Blocks.init();
}
arguments.callee.run=true;
_this.loadToolbar();
if(ExtBlockTypes.isLoading()){
_this.fillBlocks.delayedCall=true;
}else{
_this.fillBlocks();
}
};
this.useBlockIdDistr=function(){
return!!(_this.getBanner('autocontext',0)&&_this.getBanner('autocontext',0)["block_id"]);
};
this.isFakeBlockId=function(block_id){
return block_id<this.options.fake_block_high_limit&&block_id>this.options.fake_block_offset&&this.lastBlockId;
};
this.initCurrentBlock=function(){
var fakeBlockId;
if(typeof window.begun_auto_pad!=="undefined"&&window.begun_auto_pad>0&&
typeof window.begun_block_id!=="undefined"&&window.begun_block_id>0){
if(window.begun_extra_block&&typeof begunAutoRun!=="function"){
var total_banners=window.begun_total_banners||_this.getActualBlockBannersCount();
fakeBlockId=this.options.fake_block_offset+parseInt(total_banners);
this.lastBlockId=window.begun_block_id;
}
if(!window.begun_extra_block||!_this.isOldBlock()){
_this.printBlockPlace(window.begun_block_id);
}
_this.getLoadingStrategy().loadBlock(window.begun_block_id);
if(fakeBlockId){
window.begun_block_id=fakeBlockId;
}
_this.initFeedLoad();
}else if((_this.init.run)||(typeof window.begun_total_banners==="undefined"
&&typeof window.begun_block_ids==="undefined")){
Begun.Error.send("begun_block_id is missing",document.location,-1);
}
};
this.getActualBlockBannersCount=function(block){
if(typeof block==="undefined"){
if(typeof window.begun_extra_block!=="undefined"){
block=window.begun_extra_block;
}else{
return 0;
}
}
var coef=Math.ceil(Number(block.options.banners_count_coef))||1;
return Number(block.options.banners_count)*coef;
};
this.initFeedLoad=function(){
if(_this.isFeedStarted()){
return false;
}
if(isBFSApplicable()||window.begun_extra_block||!_this.getFeed()){
_this.setFeedStarted();
this.feedLoad();
}
return false;
};
this.feedLoad=function(paramsUpdate){
_this.prepareRequestParams(paramsUpdate);
Begun.Utils.includeScript(
(_this.Strings.urls.daemon+Begun.Utils.toQuery(_this.requestParams)).substring(0,1524).replace(/%[0-9a-fA-F]?$/,''),
'write',
undefined,
undefined,
'begunAds'
);
_this.requestParams.block_id="";
_this.requestParams.begun_self_keywords="";
return true;
};
this.getGraphHTML=function(graph_banner,callback,width,height,block_id){
width=width||240;
height=height||400;
var inlineStyles='';
var type='img';
if(("swf"==graph_banner.mime)||("application/x-shockwave-flash"==graph_banner.mime)){
type='swf';
}else if(("js"==graph_banner.mime)||("application/x-javascript"==graph_banner.mime)){
type='js';
Begun.Utils.includeScript(graph_banner.source,'append',callback||null);
}else if(!Begun.Browser.Gecko){
inlineStyles='width:'+width+'px;height:'+height+'px;';
}
var vars={'url':graph_banner.url,'source':graph_banner.source,'width':width,'height':height,'close_button':_this.getCloseButton(block_id),'styles':inlineStyles};
return(new Begun.Template(_this.Tpls.getHTML('search_banner_'+type))).evaluate(vars);
};
this.initFilledBannersData=function(block){
if(block&&!block.filled_banners_data){
block.filled_banners_data={
text:0,
graph:0,
code:0
};
}
};
this.insertNonTextBlock=function(block){
if(_this.Blocks.checkType(block,'hyper')){
return;
}
this.initFilledBannersData(block);
arguments.callee.blocksHandled=arguments.callee.blocksHandled||[];
if(Begun.Utils.in_array(arguments.callee.blocksHandled,block)){
return;
}else{
arguments.callee.blocksHandled.push(block);
}

var feed=_this.getFeed();
if(feed&&!feed.code_patched){
if(feed.code&&feed.banners&&!feed.banners.code){
feed.banners.code=feed.code;
}
feed.code_patched=true;
}
var block_id=block.id;
var codes=this.getBannersByBlockId(block_id,'code');
var graphs=this.getBannersByBlockId(block_id,'graph');
if(codes){
for(var i=0,l=codes.length;i<l;i++){
if(codes[i].js&&codes[i].js!=''){
Begun.Utils.evalScript(codes[i].js);
block.filled_banners_data.code++;
block.nonTextBannersInserted=true;
}
}
}
var type=block&&block.options&&block.options.dimensions&&block.options.dimensions.type;
if(graphs){
for(var i=0,l=graphs.length;i<l;i++){
if(graphs[i].loaded){
continue;
}
if(block.options.view_type&&block.options.view_type.indexOf('Graph')!=-1){
if(!arguments.callee.top_mobile_inserted&&block_id==BLOCK_ID_TOP_MOBILE&&Begun.Browser.getUaType()!==UA_DESKTOP){
_this.prepareTopMobileBlock(block.id);
arguments.callee.top_mobile_inserted=true;
}
var obj=_this.getGraphDimensions(block.options.view_type);
var html=_this.getGraphHTML(graphs[i],function(){
if(typeof begunJsBannerString==="string"){
var elem=_this.Blocks.getDomObj(block.id);
elem.innerHTML=begunJsBannerString;
}
},obj.width,obj.height,block_id);
block.filled_banners_data.graph++;
block.nonTextBannersInserted=true;
graphs[i].loaded=true;
var block_place=_this.Blocks.getDomObj(block_id);
if(html&&block_place){
block_place.innerHTML=html;
return;
}
}
}
}
if(!arguments.callee.top_mobile_inserted&&block_id==BLOCK_ID_TOP_MOBILE&&Begun.Browser.getUaType()!==UA_DESKTOP&&this.getBannersByBlockId(BLOCK_ID_TOP_MOBILE,'autocontext').length){
_this.prepareTopMobileBlock(block_id);
arguments.callee.top_mobile_inserted=true;
}

};
this.getGraphDimensions=function(src){
var dimensions=src.match(/(\d*)x(\d*)$/)||[0,0];
return{
width:dimensions[1],
height:dimensions[2]
}
};
this.getCloseButton=function(block_id){
return(block_id&&block_id==BLOCK_ID_TOP_MOBILE&&Begun.Browser.getUaType()==UA_RICH_MOBILE&&!Begun.Browser.Android)?
(new Begun.Template(_this.Tpls.getHTML('close_button'))).evaluate({
'block_id':block_id
}):'';
};
this.prepareTopMobileBlock=function(block_id){
try{
var vars={id:_this.Strings.css.block_prefix+block_id};
var block_wrapper=top.document.getElementById('begun_top_mobile_block_wrapper');
if(!block_wrapper){
var bo=top.document.getElementsByTagName('BODY');
var block_wrapper=top.document.createElement('div');
block_wrapper.id='begun_top_mobile_block_wrapper';
if(Begun.Browser.getUaType()==UA_RICH_MOBILE&&!Begun.Browser.Android){
block_wrapper.className='begun_top_mobile_bottom';
Begun.Utils.addEvent(window,'scroll',function(){
Begun.$('begun_top_mobile_block_wrapper').style.top=window.innerHeight+window.pageYOffset-Begun.$('begun_top_mobile_block_wrapper').offsetHeight+'px';
});
Begun.Utils.addEvent(window,'load',function(){
Begun.$('begun_top_mobile_block_wrapper').style.top=window.innerHeight+window.pageYOffset-Begun.$('begun_top_mobile_block_wrapper').offsetHeight+'px';
});
}
bo[0].insertBefore(block_wrapper,bo[0].firstChild);
}
block_wrapper.innerHTML=(new Begun.Template(_this.Tpls.getHTML('blck_place'))).evaluate(vars);
}catch(e){}
};
this.isOldBlock=function(){
var isPadNew=function(params){
if(!params||!window.begun_auto_pad){
return false;
}
return Begun.Utils.in_array(params.split(','),window.begun_auto_pad);
};
if(typeof _this.responseParams['old_blocks']!=="undefined"&&Number(_this.responseParams['old_blocks'])!=0&&typeof begunAutoRun=='function'){
var feed=_this.getFeed();
if(feed&&feed.cookies&&feed.cookies.js_force_new_pads&&isPadNew(feed.cookies.js_force_new_pads)){
return false;
}
return true;
}
return false;
};
this.renderOldBlock=function(){
if(_this.isOldBlock()){
begunAutoRun();
return true;
}
return false;
};
this.loadFeedDone=function(){
var i;
this.Callbacks.dispatch('feed','load',this,[window.begunAds]);
var extendVisualOptions=function(newVisualOptions){
Begun.extend(this.options.visual,newVisualOptions);
};
var setBlockBannerComponents=function(componentsParams){
for(var param in componentsParams){
if(componentsParams.hasOwnProperty(param)){
if(typeof this.options.visual[param]!=="object"){
this.options.visual[param]={};
}
if(!componentsParams[param]){
this.options.visual[param]["display"]="none";
}else{
this.options.visual[param]["display"]="";
}
}
}
};
var setThumbParams=function(visualParams){
var mixinThumb={};
for(var param in visualParams){
if(visualParams.hasOwnProperty(param)){
if(typeof mixinThumb.thumbStyles==="undefined"){
mixinThumb.thumbStyles={};
}
mixinThumb.thumbStyles[param]=visualParams[param];
}
}
Begun.extend(this.options.visual,mixinThumb);
};
var isExtraBlock=false;
for(i=0;window.begunAds&&window.begunAds.blocks&&i<window.begunAds.blocks.length;i++){
window.begunAds.blocks[i].setVisualOptions=extendVisualOptions;
if(typeof window.begunAds.blocks[i].options.json=="object"&&typeof window.begunAds.blocks[i].options.json.banner_components=="object"){
setBlockBannerComponents.call(window.begunAds.blocks[i],window.begunAds.blocks[i].options.json.banner_components);
}
window.begunAds.blocks[i].setBannerComponents=setBlockBannerComponents;
window.begunAds.blocks[i].setThumbOptions=setThumbParams;
_this.initFilledBannersData(window.begunAds.blocks[i]);

if(window.begunAds.blocks[i]&&window.begunAds.blocks[i].options&&window.begunAds.blocks[i].options.dimensions&&window.begunAds.blocks[i].options.dimensions.type&&window.begunAds.blocks[i].options.dimensions.type.toLowerCase){
ExtBlockTypes.load("begun_tpl_block_"+window.begunAds.blocks[i].options.dimensions.type);
}

if(this.isFakeBlockId(window.begunAds.blocks[i].id)&&window.begun_block_id==window.begunAds.blocks[i].id){
window.begunAds.blocks[i].id=this.lastBlockId;
isExtraBlock=true;
}
}
if(isExtraBlock){
for(var bannersType in window.begunAds.banners){
for(i=0;i<window.begunAds.banners[bannersType].length;i++){
if(window.begunAds.banners[bannersType][i].block_id==window.begun_block_id){
window.begunAds.banners[bannersType][i].block_id=this.lastBlockId;
}
}
}
window.begun_block_id=this.lastBlockId;
}
if(window.begun_extra_block&&window.begun_extra_block.options&&window.begun_extra_block.options.dimensions&&window.begun_extra_block.options.dimensions.type&&window.begun_extra_block.options.dimensions.type.toLowerCase){
ExtBlockTypes.load("begun_tpl_block_"+window.begun_extra_block.options.dimensions.type);
}
if(typeof Begun.Browser!='undefined'){
Begun.Browser.getUaType=function(){
return window.begunAds&&window.begunAds.params&&window.begunAds.params.is_mobile||0;
};
}
_this.getPad().feed=window.begunAds;
Begun.extend(_this.responseParams,_this.getFeed()&&_this.getFeed().params||{});
if(!_this.renderOldBlock()){
_this.getLoadingStrategy().parseFeed();
for(i=0;i<FAKE_BLOCKS.length;i++){
var block=_this.Blocks.getBlockById(FAKE_BLOCKS[i]);
if(block){
if(FAKE_BLOCKS[i]==BLOCK_ID_TOP_MOBILE){
for(var j=0;j<_this.getBlocks().length;j++){
if(!Begun.Utils.in_array(FAKE_BLOCKS,_this.getBlocks()[j].id)){
block.options.visual=_this.getBlocks()[j].options.visual;
break;
}
}
}
_this.insertNonTextBlock(block);
}
}
_this.draw();
}
};
this.printBlockPlace=function(block_id){
var vars={id:_this.Strings.css.block_prefix+block_id};
if(document.body){
document.write((new Begun.Template(_this.Tpls.getHTML('blck_place'))).evaluate(vars));
}else{
document.write("<body>"+(new Begun.Template(_this.Tpls.getHTML('blck_place'))).evaluate(vars)+"</body>");
}
};
this.printDefaultStyle=function(){
Begun.Utils.includeStyle(_this.Tpls.getCSS('default'),'write');
};
var getBGColor=function(block){
var bgcolor=Begun.Utils.getStyle(block,'background-color');
while(bgcolor=='transparent'){
if(block.nodeName=='BODY'){
var body_bg=Begun.Utils.getStyle(block,'background-color');
if(body_bg=='transparent'){
bgcolor='#FFFFFF';
}else{
bgcolor=Begun.Utils.getStyle(block,'background-color');
}
break;
}
block=block.parentNode;
bgcolor=Begun.Utils.getStyle(block,'background-color');
}
return bgcolor;
};
this.getLogoColor=function(styles,block_id){
var default_logo_color=_this.Strings.css.logo_color;
var channels;
var r;
var g;
var b;
var ok;
if(styles.block){
var is_logo_transparent=false;
if((styles.block.backgroundColor&&styles.block.backgroundColor.toLowerCase()=='transparent')||!styles.block.backgroundColor){
var block=_this.Blocks.getDomObj(block_id);
var toNumbers=function(str){
var ret=[];
str.replace(/(..)/g,function(str){
ret.push(parseInt(str,16));
});
return ret;
};
var areColorsTooClose=function(c1,c2,delta){
for(var i=0;i<arguments.length;i++){
if(0==arguments[i].indexOf('#')){
arguments[i]=toNumbers(arguments[i].slice(1));
}else{
return false;
}
}
delta=delta||100;
return(Math.sqrt((c1[0]-c2[0])*(c1[0]-c2[0])+(c1[1]-c2[1])*(c1[1]-c2[1])+(c1[2]-c2[2])*(c1[2]-c2[2]))<delta);
};
var convertColor=function(color_string){
if(color_string.charAt(0)=='#'){
color_string=color_string.substr(1,6);
}
color_string=color_string.replace(/ /g,'');
color_string=color_string.toLowerCase();
var simple_colors={aliceblue:'f0f8ff',antiquewhite:'faebd7',aqua:'00ffff',aquamarine:'7fffd4',azure:'f0ffff',beige:'f5f5dc',bisque:'ffe4c4',black:'000000',blanchedalmond:'ffebcd',blue:'0000ff',blueviolet:'8a2be2',brown:'a52a2a',burlywood:'deb887',cadetblue:'5f9ea0',chartreuse:'7fff00',chocolate:'d2691e',coral:'ff7f50',cornflowerblue:'6495ed',cornsilk:'fff8dc',crimson:'dc143c',cyan:'00ffff',darkblue:'00008b',darkcyan:'008b8b',darkgoldenrod:'b8860b',darkgray:'a9a9a9',darkgreen:'006400',darkkhaki:'bdb76b',darkmagenta:'8b008b',darkolivegreen:'556b2f',darkorange:'ff8c00',darkorchid:'9932cc',darkred:'8b0000',darksalmon:'e9967a',darkseagreen:'8fbc8f',darkslateblue:'483d8b',darkslategray:'2f4f4f',darkturquoise:'00ced1',darkviolet:'9400d3',deeppink:'ff1493',deepskyblue:'00bfff',dimgray:'696969',dodgerblue:'1e90ff',feldspar:'d19275',firebrick:'b22222',floralwhite:'fffaf0',forestgreen:'228b22',fuchsia:'ff00ff',gainsboro:'dcdcdc',ghostwhite:'f8f8ff',gold:'ffd700',goldenrod:'daa520',gray:'808080',green:'008000',greenyellow:'adff2f',honeydew:'f0fff0',hotpink:'ff69b4',indianred:'cd5c5c',indigo:'4b0082',ivory:'fffff0',khaki:'f0e68c',lavender:'e6e6fa',lavenderblush:'fff0f5',lawngreen:'7cfc00',lemonchiffon:'fffacd',lightblue:'add8e6',lightcoral:'f08080',lightcyan:'e0ffff',lightgoldenrodyellow:'fafad2',lightgrey:'d3d3d3',lightgreen:'90ee90',lightpink:'ffb6c1',lightsalmon:'ffa07a',lightseagreen:'20b2aa',lightskyblue:'87cefa',lightslateblue:'8470ff',lightslategray:'778899',lightsteelblue:'b0c4de',lightyellow:'ffffe0',lime:'00ff00',limegreen:'32cd32',linen:'faf0e6',magenta:'ff00ff',maroon:'800000',mediumaquamarine:'66cdaa',mediumblue:'0000cd',mediumorchid:'ba55d3',mediumpurple:'9370d8',mediumseagreen:'3cb371',mediumslateblue:'7b68ee',mediumspringgreen:'00fa9a',mediumturquoise:'48d1cc',mediumvioletred:'c71585',midnightblue:'191970',mintcream:'f5fffa',mistyrose:'ffe4e1',moccasin:'ffe4b5',navajowhite:'ffdead',navy:'000080',oldlace:'fdf5e6',olive:'808000',olivedrab:'6b8e23',orange:'ffa500',orangered:'ff4500',orchid:'da70d6',palegoldenrod:'eee8aa',palegreen:'98fb98',paleturquoise:'afeeee',palevioletred:'d87093',papayawhip:'ffefd5',peachpuff:'ffdab9',peru:'cd853f',pink:'ffc0cb',plum:'dda0dd',powderblue:'b0e0e6',purple:'800080',red:'ff0000',rosybrown:'bc8f8f',royalblue:'4169e1',saddlebrown:'8b4513',salmon:'fa8072',sandybrown:'f4a460',seagreen:'2e8b57',seashell:'fff5ee',sienna:'a0522d',silver:'c0c0c0',skyblue:'87ceeb',slateblue:'6a5acd',slategray:'708090',snow:'fffafa',springgreen:'00ff7f',steelblue:'4682b4',tan:'d2b48c',teal:'008080',thistle:'d8bfd8',tomato:'ff6347',turquoise:'40e0d0',violet:'ee82ee',violetred:'d02090',wheat:'f5deb3',white:'ffffff',whitesmoke:'f5f5f5',yellow:'ffff00',yellowgreen:'9acd32'};
for(var key in simple_colors){
if(color_string==key){
color_string=simple_colors[key];
}
}
var color_defs=[
{
re:/^rgb\((\d{1,3}),\s*(\d{1,3}),\s*(\d{1,3})\)$/,
example:['rgb(123, 234, 45)','rgb(255,234,245)'],
process:function(bits){
return[
parseInt(bits[1]),
parseInt(bits[2]),
parseInt(bits[3])
];
}
},
{
re:/^(\w{2})(\w{2})(\w{2})$/,
example:['#00ff00','336699'],
process:function(bits){
return[
parseInt(bits[1],16),
parseInt(bits[2],16),
parseInt(bits[3],16)
];
}
},
{
re:/^(\w{1})(\w{1})(\w{1})$/,
example:['#fb0','f0f'],
process:function(bits){
return[
parseInt(bits[1]+bits[1],16),
parseInt(bits[2]+bits[2],16),
parseInt(bits[3]+bits[3],16)
];
}
}
];
for(var i=0;i<color_defs.length;i++){
var re=color_defs[i].re;
var processor=color_defs[i].process;
var bits=re.exec(color_string);
if(bits){
channels=processor(bits);
r=channels[0];
g=channels[1];
b=channels[2];
ok=true;
}
}
r=(r<0||isNaN(r))?0:((r>255)?255:r);
g=(g<0||isNaN(g))?0:((g>255)?255:g);
b=(b<0||isNaN(b))?0:((b>255)?255:b);

var r=r.toString(16);
var g=g.toString(16);
var b=b.toString(16);
if(r.length==1){
r='0'+r;
}
if(g.length==1){
g='0'+g;
}
if(b.length==1){
b='0'+b;
}
return'#'+r+g+b;
};

var getRealBG=function(bgcolor){
var temp_stub=document.createElement('div');
temp_stub.style.height='0';
temp_stub.style.overflow='hidden';
temp_stub.style.backgroundColor=bgcolor;
document.body.appendChild(temp_stub);
var real_color=Begun.Utils.getStyle(temp_stub,'background-color');
temp_stub.parentNode.removeChild(temp_stub);
return real_color;
};
var bgcolor=getBGColor(block);
var temp_logo_color=getRealBG(styles.block.borderColor);
bgcolor=getRealBG(bgcolor);

bgcolor=convertColor(bgcolor);
temp_logo_color=convertColor(temp_logo_color);

if(bgcolor==temp_logo_color||areColorsTooClose(bgcolor,temp_logo_color)){
is_logo_transparent=true;
}
}
var is_default_color=((!styles.block.borderColor||styles.block.borderColor.toLowerCase()=='transparent')||(typeof styles.block.backgroundColor!=="undefined"&&typeof styles.block.borderColor!=="undefined"&&styles.block.backgroundColor.toLowerCase()==styles.block.borderColor.toLowerCase()));
return(is_default_color||is_logo_transparent)?default_logo_color:styles.block.borderColor;
}else{
return default_logo_color;
}
};
var prepareColorStyles=function(styles){
var checkColorDef=function(obj,prop){
if(obj!==null&&typeof obj!=="undefined"){
if(obj[prop]===""){
obj[prop]="transparent";
}
}
}

var checkBgColor=function(obj){
checkColorDef(obj,"backgroundColor");
}

var checkBorderColor=function(obj){
checkColorDef(obj,"borderColor");
}

var mkTransparentBordersForIE=function(obj){
if(obj!==null&&typeof obj!=="undefined"){
obj.filter="font-family:inherit;";
if(Begun.Browser.IE&&Begun.Browser.less(7)&&obj.borderColor&&(obj.borderColor.toLowerCase()=='transparent'||obj.borderColor=='')){
obj.borderColor="white";
obj.filter="filter:chroma(color=white);";
obj.transparentBorders='transparentBorders';
}
}
}

checkBgColor(styles.block_hover);
checkBorderColor(styles.block_hover);

checkBgColor(styles.block);
checkBorderColor(styles.block);

mkTransparentBordersForIE(styles.block);
mkTransparentBordersForIE(styles.block_hover);
};
this.printBlockStyle=function(block_id,styles,pad){
styles=styles||{};
var vars={};
var block=_this.Blocks.getBlockById(block_id,false,pad.pad_id);
vars.block_id=block_id||-1;
vars.phone_margin_top=1;
vars.phone_margin_top=styles.domain&&styles.domain.fontSize?styles.domain.fontSize-9:1;
vars.block_logo_color=this.getLogoColor(styles,block_id);
prepareColorStyles(styles);
for(var key in styles){
if(styles[key]&&styles.hasOwnProperty&&styles.hasOwnProperty(key)){
for(var key2 in styles[key]){
if(styles[key][key2]&&styles[key].hasOwnProperty&&styles[key].hasOwnProperty(key2)){
vars[key+':'+key2]=typeof styles[key][key2]=='number'?styles[key][key2]+'px':styles[key][key2];
}
}
}
}
var css_text=(new Begun.Template(_this.Tpls.getCSS('block'))).evaluate(vars);
css_text+=(new Begun.Template(_this.Tpls.getCSS('block_'+block.options.dimensions.type.toLowerCase()))).evaluate(vars);
var css_text_for_ie=(new Begun.Template(_this.Tpls.getCSS('forOperaIE'))).evaluate(vars);
if(Begun.Browser.IE||Begun.Browser.Opera){
css_text+=css_text_for_ie;
}
if(block_id==BLOCK_ID_TOP_MOBILE){
Begun.Utils.includeStyle(css_text,'append','begun-block-css-'+block_id,top);
}else{
Begun.Utils.includeStyle(css_text,'append','begun-block-css-'+block_id);
}
};
this.isFeedStarted=function(){
return!!_this.getPad().feed_started;
};
this.setFeedStarted=function(){
_this.getPad().feed_started=true;
};
this.getBannerIndex=function(pad_id,section,banner_id){
if(!banner_id){
return _this.getPad(pad_id).banner_index;
}else{
section=section||'autocontext';
var banner_index=0;
while(banner=_this.getBanner(section,banner_index,pad_id)){
if(banner.banner_id==banner_id){
return banner_index;
}
banner_index++;
}
}
};
this.setBannerIndex=function(index,pad_id){
_this.getPad(pad_id).banner_index=index;
};
this.incBannerIndex=function(pad_id){
_this.setBannerIndex(_this.getBannerIndex(pad_id)+1,pad_id);
};
this.resetBannerIndex=function(pad_id){
_this.setBannerIndex(0,pad_id);
};
this.registerShownBanner=function(shownBanner){
var bannerId=shownBanner&&shownBanner.banner_id;
if(!bannerId){
return;
}
if(!_this.banners){
_this.banners=[bannerId];
}else{
_this.banners.push(bannerId);
}
};
this.getShownBanners=function(){
return _this.banners;
};
this.getPad=function(pad_id){
return _this.Pads.getPad(pad_id||window.begun_auto_pad);
};
this.getFeed=function(pad_id){
return _this.getPad(pad_id).feed;
};
this.getBlock=function(index,pad){
if(typeof pad==="undefined"){
pad=_this.getPad();
}
var padBlocks=pad.blocks;
if(padBlocks.length>index){
return padBlocks[index];
}else{
return null;
}
};
this.getBlocks=function(pad_id){
var blocks=[];
if(pad_id){
blocks=_this.getPad(pad_id).blocks;
}else{
var pads=_this.Pads.getPads();
for(var i=0,l=pads.length;i<l;i++){
for(var j=0,n=pads[i].blocks.length;j<n;j++){
blocks.push(pads[i].blocks[j]);
}
}
}
return blocks;
};
this.getBanner=function(type,index,pad_id){
if(!_this.getFeed(pad_id)||typeof _this.getFeed(pad_id).banners==="undefined"||
typeof _this.getFeed(pad_id).banners[type]==="undefined"||
typeof _this.getFeed(pad_id).banners[type][index]!=="object"){
return null;
}
var banner=_this.getFeed(pad_id).banners[type][index];
banner.setImages=function(newImages){
if(typeof this.images==="undefined"){
this.images={};
}
Begun.extend(this.images,newImages);
};
banner.getThematics=function(){
var allThematics=this.thematics.split(',');
if(allThematics.length>0){
return allThematics;
}else{
return null;
}
};
return banner;
};
this.getBanners=function(pad_id){
return _this.getFeed(pad_id).banners;
};
this.getBannersByBlockId=function(block_id,type){
var i=0;
var obj=[];
var banner=null;
while(banner=_this.getBanner(type,i)){
if(banner.block_id==block_id){
obj[obj.length]=banner;
}
i++;
}
return obj;
};
this.getStub=function(type,pad_id){
return _this.getFeed(pad_id).stubs[type]||null;
};
this.getRichPictureSrc=function(banner){
var banner_id=banner.banner_id+'';
if(_this.Strings.urls.rich_picture_big&&_this.Strings.urls.rich_picture_small&&banner_id){
var small=(new Begun.Template(_this.Strings.urls.rich_picture_small)).evaluate({banner_id:banner_id});
var big=(new Begun.Template(_this.Strings.urls.rich_picture_big)).evaluate({banner_id:banner_id});
return{
small:small,
big:big
};
}
var src=_this.responseParams['thumbs_src']?'http://'+_this.responseParams['thumbs_src']+'/':_this.Strings.urls.thumbs;
var src_s;
var src_b;
if(banner_id&&banner_id.length>3){
src+='rich/';
src+=banner_id.charAt(banner_id.length-2);
src+='/'+banner_id.charAt(banner_id.length-1);
src+='/'+banner_id;
src_s=src+'s';
src_b=src+'b';
}else{
src_s=_this.Strings.urls.blank;
src_b=src_s;
}
if(banner.images&&banner.images.richpreview){
src_s=banner.images.richpreview;
}
if(banner.images&&banner.images.rich){
src_b=banner.images.rich;
}
return{
small:src_s,
big:src_b
};
};
this.getThumbSrc=function(banner,fake){
var src=_this.responseParams['thumbs_src']?'http://'+_this.responseParams['thumbs_src']+'/':_this.Strings.urls.thumbs;
var banner_id=banner.banner_id+'';
if(banner_id&&banner_id.length>3){
var bannerThematics=banner.getThematics();
var thematic=bannerThematics?(bannerThematics[0]+''):'1';
src+=banner_id.charAt(banner_id.length-2);
src+='/'+banner_id.charAt(banner_id.length-1);
src+='/'+banner_id+'.jpg';
src+='?t='+thematic+'&r='+banner_id.charAt(banner_id.length-3);
}else{
src=src+'empty.jpg';
}
if(banner.images&&banner.images.thematic){
src=banner.images.thematic;
}
if(Begun.Browser.IE&&Begun.Browser.version()<=6&&fake){
src=_this.Strings.urls.blank;
}
return src;
};
this.getFaviconSrc=function(banner){
var src=_this.responseParams['thumbs_src']?'http://'+_this.responseParams['thumbs_src']+'/':_this.Strings.urls.thumbs;
var banner_id=banner.banner_id+'';
if(banner_id&&banner_id.length>3){
src+='favicon/';
src+=banner_id.charAt(banner_id.length-2);
src+='/'+banner_id.charAt(banner_id.length-1);
src+='/'+banner_id+'.jpg';
}else{
src=_this.Strings.urls.blank;
}
if(banner.images&&banner.images.favicon){
src=banner.images.favicon;
}
return src;
};
this.getBannerContacts=function(banner,block,fullDomain,pad_id,section,banner_id){
var result=this.getBannerCardPPcallData(banner,block,pad_id,section,banner_id);
var banner_contacts_names=result.is_url_exist?['domain','geo']:['geo'];
return result.banner_contacts.concat(this.getBannerDomainGeoHTML(banner,block,banner_contacts_names,fullDomain));
};
this.getBannerCardPPcallData=function(banner,block,pad_id,section,banner_id){
var banner_contacts=[];
var is_url_exist=true;
var cards_mode=banner['cards_mode'];
var is_ppcall=banner['ppcall'];
var vars={};
function _card(use_phone){
vars.card_text=_this.Strings.contacts.card;
vars.url=_this.addMisc2URL(block.options.misc_id,banner.card);
vars.phone=use_phone?(new Begun.Template(_this.Tpls.getHTML('bnnr_phone'))).evaluate(vars):'';
vars.no_phone_class=use_phone?'':'begun_adv_phone_no_icon';
banner_contacts.push((new Begun.Template(_this.Tpls.getHTML('bnnr_card'))).evaluate(vars));
}
function _ppcall(use_phone){
var phone_msk=_this.getTextPhone(banner.descr);
var form_msk=_this.Strings.urls.base_scripts_url+'ppcall_form_msk.html?phone='+encodeURIComponent(phone_msk);
if(_this.isPpcallExperimentActivated()&&phone_msk){
banner.ppcall_form_type='text_phone';
vars.ppcall_text=phone_msk+'<span style="display:none !important;">&minus;</span>';
vars.link=form_msk;
}else{
banner.ppcall_form_type='default';
vars.ppcall_text=_this.Strings.contacts.ppcall;
vars.link=banner.ppcall_form?banner.ppcall_form:'';
}
vars.banner_index=_this.getBannerIndex(pad_id,section,banner_id);
vars.pad_id=window.begun_auto_pad||'';
vars.phone=use_phone?(new Begun.Template(_this.Tpls.getHTML('bnnr_phone'))).evaluate(vars):'';
vars.ua_type=Begun.Browser.getUaType();
banner_contacts.push((new Begun.Template(_this.Tpls.getHTML('bnnr_ppcall'))).evaluate(vars));
}
if(cards_mode=='Card'&&is_ppcall==false){
_card(true);
is_url_exist=false;
}else if(cards_mode=='Card'&&is_ppcall==true){
_ppcall(true);
_card(false);
is_url_exist=false;
}else if(cards_mode=='Url'&&is_ppcall==false){

}else if(cards_mode=='Url'&&is_ppcall==true){
_ppcall(true);
}else if(cards_mode=='Card, Url'&&is_ppcall==false){
_card(true);
}else if(cards_mode=='Card, Url'&&is_ppcall==true){
_ppcall(true);
_card(false);
}
return{
banner_contacts:banner_contacts,
is_url_exist:is_url_exist
};
};
this.getBannerDomainGeoHTML=function(banner,block,banner_contacts_names,fullDomain){
var banner_contacts=[];
var i=0;
var banner_contacts_name=null;
var vars={};
while(banner_contacts_name=banner_contacts_names[i]){
vars[banner_contacts_name]=banner[banner_contacts_name];
vars.status=banner.status;
vars.url=_this.addMisc2URL(block.options.misc_id,banner.url);
vars.fullDomain=fullDomain;
if(vars[banner_contacts_name]){
banner_contacts.push((new Begun.Template(_this.Tpls.getHTML('bnnr_'+banner_contacts_name))).evaluate(vars));
}
i++;
}
return banner_contacts;
};
this.addMisc2URL=function(misc_id,url){
return(misc_id>0?url+'&misc2='+(Number(misc_id)<<8):url);
};
this.clickBanner=function(click_event,orig_elem){
click_event=click_event||window.event;
if(click_event.done){
return;
}
var curr_elem=click_event.target||click_event.srcElement;
var isInsideTag=function(child_elem,parent_tag){
var child_elem_parent=child_elem;
do{
if(child_elem_parent.tagName&&child_elem_parent.tagName.toUpperCase()==parent_tag.toUpperCase()){
return true;
}
}while(child_elem_parent=child_elem_parent.parentNode);
return false;
};
if(curr_elem.tagName.toUpperCase()=='A'||isInsideTag(curr_elem,'A')){
click_event.done=true;
_this.Callbacks.dispatch('banner','click',curr_elem);
if(this.isEventTrackingOn()){
_this.clickHandler(orig_elem).apply(_this);
}
}else if(orig_elem.getAttribute('_url')){
var anyLink=curr_elem.getElementsByTagName("a")[0];
if(anyLink&&typeof anyLink.click!=="undefined"){
if(typeof click_event.preventDefault!=="undefined"){
click_event.preventDefault();
}else{
click_event.returnValue=false;
}
if(typeof click_event.stopPropagation!=="undefined"){
click_event.stopPropagation();
}else{
click_event.cancelBubble=true;
}
anyLink.click();
}else{
_this.Callbacks.dispatch('banner','click',curr_elem);
if(this.isEventTrackingOn()){
_this.clickHandler(orig_elem).apply(_this);
}
window.open(orig_elem.getAttribute('_url'));
}
}
};
this.getBannerHTML=function(banner,block,block_banner_count){
var BANNER_SHORT_PART_LENGTH=13;
function prepareBannerMode(banner){
banner=banner||{};
var possible_cards_modes=['Card, Url','Card','Url'];
if((!banner['cards_mode'])||!Begun.Utils.in_array(possible_cards_modes,banner['cards_mode'])){
banner['cards_mode']='Card, Url';
}
if(!banner['url']&&!banner['card']){
return{};
}
if(!banner['url']&&banner['card']){
banner['cards_mode']='Card';
}
if(banner['url']&&!banner['card']){
banner['cards_mode']='Url';
}
if(banner['cards_mode']=='Card'){
banner['url']=banner['card'];
}
return banner;
}

banner=prepareBannerMode(banner);
if(banner){
if(banner.domain){
banner.domain=banner.domain.replace(/&shy;/g,'');
banner.fullDomain=banner.domain;
if(banner.domain.match(/&#x426;&#x435;&#x43d;&#x430;: /)){
banner.status=banner.domain;
}else{
banner.status='http://'+banner.domain+'/';
if(banner.domain.length>2*BANNER_SHORT_PART_LENGTH+3){
banner.domain=banner.domain.substring(0,BANNER_SHORT_PART_LENGTH)
+'&hellip;'+banner.domain.slice(-BANNER_SHORT_PART_LENGTH);
}
}
}else{
banner.fullDomain=banner.domain=banner.status='';
}
banner.domain=banner.domain.replace(/\./g,'.&shy;');
var banner_contacts=_this.getBannerContacts(banner,block,banner.fullDomain);
var vars={};
Begun.extend(vars,banner);
if(block.options.visual.title&&block.options.visual.title.display&&block.options.visual.title.display=="none"){
vars.styleTitle=" style=\"display: none\"";
}
if(block.options.visual.text&&block.options.visual.text.display&&block.options.visual.text.display=="none"){
vars.styleText=" style=\"display: none\"";
}
if(block.options.visual.contact&&block.options.visual.contact.display&&block.options.visual.contact.display=="none"){
vars.styleContact=" style=\"display: none\"";
}
vars.contact=banner_contacts.join(_this.Tpls.getHTML('bnnr_glue'));
vars.url=_this.addMisc2URL(block.options.misc_id,banner.url);
vars.onclick=_this.Strings.js.banner_onclick;
vars.block_id=block.id;
vars.banner_id=banner.banner_id;
vars.id=block_banner_count||0;
vars.descr=vars.descr.replace(/(\,|\.|\?|\!|\:)(\S\D)/g,'$1 $2');
vars.banner_width=Math.round(100/Number(_this.getActualBlockBannersCount(block)))+'%';
if(_this.Blocks.checkType(block,'square')&&block.options.json&&block.options.json.col){
vars.banner_width=Math.round(100/Number(block.options.json.col))+'%';
}
vars.bnnr_alco=_this.checkBannerViewType(banner,'alco')?(new Begun.Template(_this.Tpls.getHTML('bnnr_alco_attn'))).evaluate({}):'';
var is_use_rich='';

vars.css_favicon=Number(block.options.show_favicons)?_this.Strings.css.favicon:'';
vars.favicon=Number(block.options.show_favicons)?'style="zoom:1;background-image:url('+_this.getFaviconSrc(banner)+') !important;background-repeat:no-repeat !important;"':'';
vars.thumb='';
vars.picture='';
var getThumbAdditionalStyles=function(){
var DEFAULT_STYLE="";
if(!block.options.visual.thumbStyles){
return DEFAULT_STYLE;
}
var styleString=" ";
for(visualParam in block.options.visual.thumbStyles){
if(block.options.visual.thumbStyles.hasOwnProperty(visualParam)){
styleString+=visualParam+":"+block.options.visual.thumbStyles[visualParam]+" !important;";
}
}
if(styleString===""){
styleString=DEFAULT_STYLE;
}
return styleString;
};
if(!vars.favicon){
if(_this.checkBannerViewType(banner,'rich')){
var pictures=_this.getRichPictureSrc(banner);
vars.picture=(new Begun.Template(_this.Tpls.getHTML('bnnr_picture'))).evaluate({src:pictures.small,big_photo_src:pictures.big,url:banner.url});
var is_use_rich='_rich';
}else{
vars.thumb=Number(block.options.show_thumbnails)?(new Begun.Template(_this.Tpls.getHTML('bnnr_thumb'))).evaluate({
url:banner.url,
src:_this.getThumbSrc(banner,true),
bgcolor:_this.Thumbs.getType()=='classic'?((typeof block.options.visual.thumb!='undefined')?block.options.visual.thumb.backgroundColor:_this.Strings.css.thumb_def_color):'transparent',
width:_this.Thumbs.getDimentions(_this.Thumbs.getType()).width,
height:_this.Thumbs.getDimentions(_this.Thumbs.getType()).height,
pngfix:(Begun.Browser.IE&&Begun.Browser.version()<=6)?'style="filter:progid:DXImageTransform.Microsoft.AlphaImageLoader(src=\''+_this.getThumbSrc(banner,false)+'\', sizingMethod=\'image\');"':'',
additionalStyles:getThumbAdditionalStyles(),
mouse_events:_this.Thumbs.getType()=='classic'?'onmouseover="this.style.background = \''+((typeof block.options.visual.thumb_hover!='undefined')?block.options.visual.thumb_hover.backgroundColor:_this.Strings.css.thumb_def_color_hover)+'\'" onmouseout="this.style.background = \''+((typeof block.options.visual.thumb!='undefined')?block.options.visual.thumb.backgroundColor:_this.Strings.css.thumb_def_color)+'\'"':''
}):'';
vars.picture=vars.thumb;
}
}
var is_use_accordion=Number(block.options.use_accordion)?'_use_accordion':'';
return(new Begun.Template(_this.Tpls.getHTML('banner_'+block.options.dimensions.type.toLowerCase()+is_use_rich+is_use_accordion))).evaluate(vars);
}else{
return'';
}
};
this.checkBannerViewType=function(banner,viewtype){
return Begun.Utils.inList(banner.view_type,viewtype);
};
this.addBannerViewType=function(banner,viewtype){
if(!this.checkBannerViewType(banner,viewtype)){
banner.view_type+=(banner.view_type?',':'')+viewtype;
}
};
this.removeBannerViewType=function(banner,viewtype){
if(this.checkBannerViewType(banner,viewtype)){
var reg=new RegExp('(,?\\s?|^)'+viewtype+'(,?\\s?|$)');
banner.view_type=banner.view_type.replace(reg,',');
}
};
this.bannersContainViewType=function(view_type,pad_id,section){
var section=section||'autocontext';
var hasViewType=false;
for(var i=0;i<_this.getBanners(pad_id)[section].length;i++){
if(_this.checkBannerViewType(_this.getBanners(pad_id)[section][i],view_type)){
hasViewType=true;
break;
}
}
return hasViewType;
};
this.getTableWithAds=function(blockId){
var getSingleTable=function(id){
var element=_this.Blocks.getDomObj(id);
if(!element){
return undefined;
}
var tables=element.getElementsByTagName("table");
for(var i=0;i<tables.length;i++){
if(tables[i].className&&tables[i].className.indexOf("begun_adv_table")>-1){
return tables[i];
}
}
return undefined;
};
switch(typeof blockId){
case"number":
case"string":
return getSingleTable(blockId);
default:
var blocks=_this.getBlocks();
var res=[];
for(var i=0;i<blocks.length;i++){
var tbl=getSingleTable(blocks[i].id);
if(tbl){
res.push(tbl);
}
}
return(res.length>0?res:undefined);
}
};
this.updateUrlParamInTd=function(td,param,value){
var updateParamInLink=function(link,param,value){
var hrefText=link.getAttribute("href");
var linkContent=link.firstChild.nodeValue;
if(hrefText.indexOf('http://')==-1&&hrefText.indexOf('https://')==-1){
return false;
}
var newHref;
if(hrefText.indexOf("?")===-1){
hrefText=hrefText+"?addingParams";
}
if(hrefText.indexOf("&"+param+"=")===-1){
newHref=hrefText+"&"+param+"="+value;
link.setAttribute("href",newHref);
link.firstChild.nodeValue=linkContent;
td.setAttribute("_url",newHref);
}else{
var firstPosition=hrefText.indexOf("&"+param+"=")+param.length+1;
var lastPosition=hrefText.indexOf("&",firstPosition+1);
if(lastPosition===-1){
newHref=hrefText.substring(0,firstPosition+1)+value;
link.setAttribute("href",newHref);
link.firstChild.nodeValue=linkContent;
td.setAttribute("_url",newHref);
}else{
newHref=hrefText.substring(0,firstPosition+1)+value+hrefText.slice(lastPosition);
link.setAttribute("href",newHref);
link.firstChild.nodeValue=linkContent;
td.setAttribute("_url",newHref);
}
}
};
var linksInTd=td.getElementsByTagName("a");
for(var i=0,len=linksInTd.length;i<len;i++){
updateParamInLink(linksInTd[i],param,value);
}
};
this.getBlockHTML=function(banners_html_arr,block,pad){
if(!banners_html_arr){
return'';
}
var banners_html=banners_html_arr.join('');
var stub_display=Number(_this.responseParams['stub'])?'':'none';
if(block.options&&(typeof block.options.json!='undefined')&&(typeof block.options.json.stub!='undefined')){
stub_display=Number(block.options.json.stub)?'':'none';
}
var logo_display='';
if(block.options&&(typeof block.options.json!='undefined')&&(typeof block.options.json.logo!='undefined')){
logo_display=(Number(block.options.json.logo))?'':'none';
}
var extended_block_class='';
if(stub_display=='none'&&logo_display=='none'){
extended_block_class='begun_extended_block';
}else if(stub_display=='none'&&logo_display==''){
extended_block_class='begun_extended_block_with_logo';
}
var blockBannersContainViewType=function(view_type,pad_id,section){
var section=section||'autocontext';
var hasViewType=false;
for(var i=0;i<_this.getBannersByBlockId(block.id,section).length;i++){
if(_this.checkBannerViewType(_this.getBannersByBlockId(block.id,section)[i],view_type)){
hasViewType=true;
break;
}
}
return hasViewType;
}
var rich_nopics_class='';
if((_this.Blocks.checkType(block,'rich')||_this.Blocks.checkType(block,'240x400')&&_this.Blocks.checkViewType(block,'rich'))&&!blockBannersContainViewType('rich',pad.pad_id)){
rich_nopics_class=' begun_rich_nopics';
}

var vars={};
var block_hover_html='';
var block_opts=block.options.visual||{};
if(block_opts.block&&block_opts.block_hover&&block_opts.block_hover.backgroundColor&&block_opts.block_hover.borderColor){
vars.bgcolor_over=block_opts.block_hover.backgroundColor;
vars.brdcolor_over=block_opts.block_hover.borderColor;
vars.bgcolor_out=block_opts.block.backgroundColor||'transparent';
vars.brdcolor_out=block_opts.block.borderColor||'transparent';
vars.block_id=block.id;
block_hover_html=(new Begun.Template(_this.Tpls.getHTML('blck_hover'))).evaluate(vars);
}
var pad_id=pad.pad_id;
vars={
block_id:block.id,
block_hover:block_hover_html,
banners:banners_html,
banners_count:banners_html_arr.length,

scroll_div_id:_this.Strings.css.scroll_div_prefix+block.id,
scroll_table_id:_this.Strings.css.scroll_table_prefix+block.id,
block_width:Number(block.options.dimensions.width)?Number(block.options.dimensions.width)+'px':'',
block_scroll_class:Number(block.options.use_scroll)?_this.Strings.css.scroll:'',
begun_url:_this.Strings.urls.begun,
become_partner_display:stub_display,
become_partner_url:_this.getStub('become_partner',pad_id),
become_partner_text:_this.Strings.stubs.become_partner,
place_here_display:stub_display,
place_here_url:_this.getStub('place_here',pad_id),
place_here_text:_this.Strings.stubs.place_here,
all_banners_display:stub_display,
all_banners_url:_this.getStub('all_banners',pad_id),
all_banners_text:_this.Strings.stubs.all_banners,
css_thumbnails:(Number(block.options.show_thumbnails)&&!Number(block.options.show_favicons)?_this.Strings.css.thumb+' '+(_this.Thumbs.getType()=='classic'?_this.Strings.css.thumb_classic:_this.Strings.css.thumb_default):'')+rich_nopics_class,
logo_display:logo_display,
close_button:_this.getCloseButton(block.id),
extended_block_class:extended_block_class,
block_alco:block.is_alco?(new Begun.Template(_this.Tpls.getHTML('block_alco'))).evaluate({}):'',
begun_alco_id:block.is_alco?_this.Strings.css.alco_prefix+block.id:'',
transparent_borders_class:(block_opts&&((block_opts.block&&block_opts.block.transparentBorders)||(block_opts.block_hover&&block_opts.block_hover.transparentBorders)))?'transparentBorders':'',
fix_layout:(Begun.Browser.IE&&Begun.Browser.version()<8&&document.compatMode&&document.compatMode=="CSS1Compat"&&_this.Blocks.isBlockFixed(block))?_this.Strings.css.fix_layout:''
};
var is_use_accordion=Number(block.options.use_accordion)?'_use_accordion':'';
return(new Begun.Template(_this.Tpls.getHTML('block_'+block.options.dimensions.type.toLowerCase()+is_use_accordion))).evaluate(vars);
};
this.clickHandler=function(targetTd){
return function(){
var nowTime=(new Date).valueOf();
this.updateUrlParamInTd(targetTd,"click_time",nowTime);
this.updateUrlParamInTd(targetTd,"frame_level",_this.requestParams.frm_level);
};
};
this.printBlock=function(banners_html,block,pad){
if(_this.isOldBlock()){
return;
}
if(banners_html.length){
var regEvents=function(){
if(!_this.isEventTrackingOn()){
return undefined;
}
var mouseOverHandler=function(targetTd){
return function(e){
if(!e){
var e=window.event;
}
var relTarget=e.relatedTarget||e.fromElement;
if(relTarget===targetTd){
return;
}
var tdElements=targetTd.getElementsByTagName("*");
for(var i=0;i<tdElements.length;i++){
if(tdElements[i]===relTarget){
return;
}
}
if(!arguments.callee.count){
arguments.callee.count=1;
}
var nowTime=(new Date).valueOf();
_this.updateUrlParamInTd(targetTd,"mouseover_time",nowTime);
_this.updateUrlParamInTd(targetTd,"mouseover_count",arguments.callee.count++);
};
};
var mouseDownHandler=function(targetTd){
return function(){
var nowTime=(new Date).valueOf();
_this.updateUrlParamInTd(targetTd,"mousedown_time",nowTime);
};
};
var tds=_this.getTableWithAds(block.id).getElementsByTagName("td");
var showTime=(new Date).valueOf();
for(var i=0;i<tds.length;i++){
_this.updateUrlParamInTd(tds[i],"show_time",showTime);
Begun.Utils.addEvent(tds[i],"mouseover",mouseOverHandler(tds[i]));
Begun.Utils.addEvent(tds[i],"mousedown",mouseDownHandler(tds[i]));
}
};
var elem=_this.Blocks.getDomObj(block.id);

if(!elem){
return false;
}
this.setExtraBlockResponseParams(block);
_this.dom_change=true;
var html=_this.getBlockHTML(banners_html,block,pad);
var show=showDefault=function(elem,html){
elem.innerHTML=html;
_this.dom_change=false;
regEvents();
};


if(Begun.Browser.IE){
show=function(elem,html){
var n=elem.cloneNode(true);
n.innerHTML=html;
elem.parentNode.insertBefore(n,elem);
elem.parentNode.removeChild(elem);
_this.dom_change=false;
regEvents();
};
var appendTableCell=function(tr,elem){
if(tr.offsetHeight){
var td=document.createElement('td');
tr.appendChild(td);
td.innerHTML=elem.outerHTML;
show(td.firstChild,html);
elem.parentNode.removeChild(elem);
}else{
var func=arguments.callee;
window.setTimeout(function(){
func(tr,elem);
},Begun.DOM_TIMEOUT);
}
};
var parent=null;
if((parent=elem.parentNode)&&(parent.tagName)&&(Begun.Utils.in_array(['ol','ul','li'],parent.tagName.toLowerCase()))){
window.setTimeout(function(){
var parent2=parent.parentNode;
parent2.insertBefore(elem,parent);
showDefault(elem,html);
},Begun.DOM_TIMEOUT);
}else if((parent)&&(parent=elem.parentNode.parentNode)&&(parent.tagName)){
try{
show(elem,html);
}catch(e){
switch(parent.tagName.toLowerCase()){
case'table':
var tr=document.createElement('tr');
window.setTimeout(function(){
parent.lastChild.appendChild(tr);
appendTableCell(tr,elem);
},Begun.DOM_TIMEOUT);
break;
case'tr':
window.setTimeout(function(){
appendTableCell(parent,elem);
},Begun.DOM_TIMEOUT);
break;
case'thead':
case'tbody':
case'tfoot':
var tr=document.createElement('tr');
window.setTimeout(function(){
parent.appendChild(tr);
appendTableCell(tr,elem);
},Begun.DOM_TIMEOUT);
break;
default:
_this.dom_change=false;
}
}
}else{
try{
show(elem,html);
}catch(e){
_this.dom_change=false;
}
}
}else{
show(elem,html);
}
return true;
}else{
return false;
}
};
this.hideBlock=function(block_id){
var elem=_this.Blocks.getDomObj(block_id);
if(elem){
elem.innerHTML='';
}
};
this.dispatchBlockDrawCallback=function(block){
if(block&&!block.drawCallbackDispatched){
_this.Callbacks.dispatch('block','draw',_this,[block]);
block.drawCallbackDispatched=true;
}
};
this.fillBlocks=function(){
var pad=_this.getPad();
var isValidSquareBlock=function(block){
return(_this.Blocks.checkType(block,'square')&&block.options.json&&block.options.json.row&&block.options.json.col);
};
var block=null;
var block_index=0;
var out_of_banners=false;
if(typeof arguments.callee.blocksHandled==="undefined"){
arguments.callee.blocksHandled=[];
}
while((block=_this.getBlock(block_index,pad))&&(!out_of_banners)){
if(!_this.Blocks.getDomObj(block.id)){
block_index++;
continue;
}
if(!Begun.Utils.in_array(arguments.callee.blocksHandled,block)){
_this.Callbacks.dispatch('block','predraw',_this,[block]);
if(block.options&&block.options.visual){
_this.printBlockStyle(block.id,block.options.visual,pad);
}
arguments.callee.blocksHandled.push(block);
}
if(block.loaded||_this.Blocks.isDeleted(block)){
block_index++;
continue;
}
if(block.nonTextBannersInserted){
_this.dispatchBlockDrawCallback(block);
block_index++;
continue;
}
if(Begun.Utils.inList((block.options&&block.options.block_options),'JSCatalog')){
var initAutoCatalog=function(block){
if(!Begun.Catalog||!Begun.$(_this.Strings.css.catalog_search_wrapper)||!Begun.$(_this.Strings.css.catalog_results_wrapper)||!Begun.$(_this.Strings.css.catalog_cloud_wrapper)||_this.dom_change){
window.setTimeout(function(){
initAutoCatalog(block);
},Begun.DOM_TIMEOUT);
}else{
_this.initAutoCatalogBlock(block);
}
};
initAutoCatalog(block);
block.loaded=true;
block_index++;
continue;
}else if(_this.Blocks.checkType(block,'hyper')){
var pad_id=window.begun_auto_pad;
var initHypercontext=function(block,pad_id){
if(!Begun.Hypercontext||_this.dom_change){
window.setTimeout(function(){
initHypercontext(block,pad_id);
},Begun.DOM_TIMEOUT);
}else{
_this.initHypercontextBlock(block,pad_id);
}
};
initHypercontext(block,pad_id);
block.loaded=true;
block_index++;
continue;
}else if(_this.Blocks.checkType(block,'photo')){
var pad_id=window.begun_auto_pad;
var initPhotocontext=function(block,pad_id){
if(!Begun.Photocontext||_this.dom_change){
window.setTimeout(function(){
initPhotocontext(block,pad_id);
},Begun.DOM_TIMEOUT);
}else{
_this.initPhotocontextBlock(block,pad_id);
}
};
initPhotocontext(block,pad_id);
block.loaded=true;
block_index++;
continue;
}
var banners_html=[];
var block_banner_count=0;
var banner=null;
this.setExtraBlockResponseParams(block);
if(Number(block.options.use_scroll)&&(Number(block.options.use_accordion)||_this.Blocks.checkType(block,'top')||_this.Blocks.checkType(block,'rich'))){
block.options.use_scroll=0;
}
var banners_count;
var banner_html;
if(_this.useBlockIdDistr()){
var i=0;
banners_count=_this.getActualBlockBannersCount(block);
while(banner=_this.getBanner('autocontext',i,pad.pad_id)){
_this.Callbacks.dispatch('banner','predraw',_this,[banner]);
if(banner.block_id&&(this.lastBlockId&&this.lastBlockId==block.id||banner.block_id==block.id)&&!banner.disabled){
banner_html='';
if(isValidSquareBlock(block)){
if(block_banner_count%Number(block.options.json.col)==0){
banner_html+='<tr>';
}
banner_html+=_this.getBannerHTML(banner,block,(block_banner_count+1));
if((block_banner_count+1)%Number(block.options.json.col)==0){
banner_html+='</tr>';
}
}else{
banner_html=_this.getBannerHTML(banner,block,(block_banner_count+1));
}
if(banner_html){
banners_html.push(banner_html);
block.filled_banners_data.text++;
_this.Callbacks.dispatch('banner','draw',_this,[banner]);
_this.registerShownBanner(banner);
}
if(_this.checkBannerViewType(banner,'alco')){
block.is_alco=true;
}
block_banner_count++;
}
i++;
}
}else{
banners_count=_this.getActualBlockBannersCount(block);
while(block_banner_count<banners_count){
banner=_this.getBanner('autocontext',_this.getBannerIndex(pad.pad_id),pad.pad_id)||null;
if(banner){
_this.Callbacks.dispatch('banner','predraw',_this,[banner]);
if(_this.checkBannerViewType(banner,'alco')){
block.is_alco=true;
}
banner_html='';
if(isValidSquareBlock(block)){
if(block_banner_count%Number(block.options.json.col)==0){
banner_html+='<tr>';
}
banner_html+=_this.getBannerHTML(banner,block,(block_banner_count+1));
if((block_banner_count+1)%Number(block.options.json.col)==0){
banner_html+='</tr>';
}
}else{
banner_html=_this.getBannerHTML(banner,block,(block_banner_count+1));
}
if(banner_html){
banners_html.push(banner_html);
block.filled_banners_data.text++;
_this.Callbacks.dispatch('banner','draw',_this,[banner]);
_this.registerShownBanner(banner);
}
}else{
out_of_banners=true;
break;
}
block_banner_count++;
_this.incBannerIndex(pad.pad_id);
}
}
if(isValidSquareBlock(block)&&block_banner_count<banners_count&&block_banner_count!=0){
for(block_banner_count;block_banner_count<banners_count;block_banner_count++){
var banner_html='';
if(block_banner_count%Number(block.options.json.col)==0){
banner_html+='<tr>';
}
banner_html+='<td>&nbsp;</td>';
if((block_banner_count+1)%Number(block.options.json.col)==0){
banner_html+='</tr>';
}
banners_html.push(banner_html);
}
}
if(_this.printBlock(banners_html,block,pad)){
block.loaded=true;
}
_this.dispatchBlockDrawCallback(block);
Module.initInBlock(block,pad);
block_index++;
}
};
this.fillBlocks.delayedCall=false;
this.begunToolbarLoaded=function(){
if(!Begun.Toolbar||!Begun.Toolbar.init){
return;
}
while(this.unhandledDebugs.length>0){
Begun.Toolbar.init(this.unhandledDebugs.pop());
}
};
this.loadToolbar=function(){
var feed=this.getFeed();
if(feed&&feed.debug){
var debugCopy={};
for(var debugEntity in window.begunAds.debug){
if(feed.debug.hasOwnProperty(debugEntity)){
debugCopy[debugEntity]=feed.debug[debugEntity];
}
}
this.unhandledDebugs.push(debugCopy);
delete feed.debug;
this.begunToolbarLoaded();
}
};
this.nullGlobalBlockParams=function(){
window.begun_block_id=null;
window.begun_extra_block=null;
};
this.setExtraBlockResponseParams=function(block){
block.options.use_scroll=typeof block.options.use_scroll!='number'?Number(_this.responseParams['autoscroll']):block.options.use_scroll;
block.options.show_thumbnails=typeof block.options.show_thumbnails!='number'||isNaN(block.options.show_thumbnails)?Number(_this.responseParams['thumbs']):block.options.show_thumbnails;
};
this.isPpcallExperimentActivated=function(){
var PPCALL_EXPERIMENT_ACTIVE=true;
return PPCALL_EXPERIMENT_ACTIVE&&_this.getFeed().params.user_region_id==12;
};
this.getTextPhone=function(text){
var phone='';
var re=/.*(\(495\)\s?\d{3}\-\d{2}\-\d{2}).*/;
if(text&&re.test(text)){
phone=text.replace(re,'$1');
}
return phone;
};
};

(function(){
var ac=Begun.Autocontext;

ac.Monitor=new function(){
var _this=this;
this.init=function(){
Begun.Utils.addEvent(window,'load',function(){
_this.prepare();
});
Begun.Utils.addEvent(window,'unload',function(){
_this.send(_this.data||'none');
});
Begun.Utils.addEvent(window,'scroll',function(){
_this.count();
});
};
this.prepare=function(){
var pads=ac.Pads.getPads();
if(pads.length===0){
Begun.Error.send("begun_auto_pad is missing",document.location,-1);
return;
}
for(var n=0,ln=pads.length;n<ln;n++){
for(var i=0,length=pads[n].blocks.length;i<length;i++){
var dom_obj=ac.Blocks.getDomObj(pads[n].blocks[i].id);
if(ac.Blocks.isDeleted(pads[n].blocks[i])||!dom_obj){
continue;
}
pads[n].blocks[i].hidden=false;
pads[n].blocks[i].dom_obj=dom_obj;
var banners_id=[];
var tds=dom_obj.getElementsByTagName('td');
for(var k=0,l=tds.length;k<l;k++){
if(tds[k].getAttribute('_banner_id')&&tds[k].getAttribute('_banner_id')!=''&&typeof(tds[k].getAttribute('_banner_id'))!=undefined){
banners_id[banners_id.length]=tds[k].getAttribute('_banner_id');
}
}
pads[n].blocks[i].banners_id=banners_id.join(',');
}
}
this.count();
};
this.count=function(){
var data=[];
var visibleBannersData=[];
var pads=ac.Pads.getPads();
for(var n=0,ln=pads.length;n<ln;n++){
var hiddenBannersObj=[];
var visibleBannersObj=[];
for(var i=0,l=pads[n].blocks.length;i<l;i++){
var viewportheight=Begun.Utils.countWindowSize().height;
var scrolledOfY=Begun.Utils.getScrollXY().y;
var dom_obj=pads[n].blocks[i].dom_obj;
if(dom_obj){
if(!pads[n].blocks[i].alreadySeen){
var graphBanner=(pads[n].blocks[i].options.view_type&&pads[n].blocks[i].options.view_type.indexOf('Graph')!=-1);
if(!pads[n].blocks[i].banners_id&&!graphBanner){
pads[n].blocks[i].alreadySeen=true;
continue;
}
var blockVisible=Begun.Utils.findPos(dom_obj)&&Begun.Utils.findPos(dom_obj).top<viewportheight+scrolledOfY;
if(blockVisible&&!graphBanner){
pads[n].blocks[i].hidden=false;
visibleBannersObj[visibleBannersObj.length]={
id:pads[n].blocks[i].id,
banners_id:pads[n].blocks[i].banners_id
}
var blockAlreadySeen=ac.Blocks.getBlockById(pads[n].blocks[i].id,hiddenBannersObj);
delete blockAlreadySeen;
pads[n].blocks[i].alreadySeen=true;
}else if(blockVisible&&graphBanner){
var graphBanners=ac.getBannersByBlockId(pads[n].blocks[i].id,'graph');
if(graphBanners.length&&graphBanners[0].show_url){
this.send('',graphBanners[0].show_url);
}
pads[n].blocks[i].alreadySeen=true;
}else if(!blockVisible&&!graphBanner){
pads[n].blocks[i].hidden=true;
hiddenBannersObj[hiddenBannersObj.length]={
id:pads[n].blocks[i].id,
banners_id:pads[n].blocks[i].banners_id
};
}else{}
}
}
}
if(hiddenBannersObj.length){
data[data.length]={
pad_id:pads[n].pad_id,
hidden:hiddenBannersObj
};
}else{
data=[];
}
if(visibleBannersObj.length){
this.send(Begun.Utils.toJSON({
pad_id:pads[n].pad_id,
visible:visibleBannersObj
}));
}
}
if(data.length){
this.data=data.length?Begun.Utils.toJSON(data):'none';
}
};
this.getCounterUrl=function(url){
return url||ac.Strings.urls.log_banners_counter;
};
this.send=function(data,url){
Begun.Utils.includeCounter(this.getCounterUrl(url),{
data:data
});
};
};

ac.Pads=new function(){
var pads=[];
this.init=function(){
if(typeof window.begun_auto_pad!=="undefined"&&!this.getPad()){
this.push(window.begun_auto_pad);
}
};
this.push=function(pad_id){
pads[pads.length]={
pad_id:pad_id,
feed:null,
blocks:[],
banner_index:0,
feed_started:false
};
};
this.getPad=function(pad_id){
pad_id=pad_id||window.begun_auto_pad;
for(var i=0,l=pads.length;i<l;i++){
if(pads[i].pad_id==pad_id){
return pads[i];
}
}
return null;
};
this.getPads=function(){
return pads;
};
};

ac.Blocks=new function(){
this.init=function(){
ac.resetBannerIndex();
ac.resetMaxScrollers();
};
this.add=function(elem,pad_id){
var blocks=ac.getPad(pad_id).blocks;
blocks[blocks.length]=elem;
};
this.push=function(elem,pad_id){
if(typeof window.begun_auto_pad!=="undefined"&&elem.id){
this.loadBlockCounter(window.begun_auto_pad,elem.id);
}
var blocks=ac.getPad(pad_id).blocks;
if(window.begun_extra_block){
blocks[0]=elem;
}else{
blocks[blocks.length]=elem;
}
if(!ac.isFeedStarted()){
ac.initFeedLoad();
}else if(!!ac.getFeed()){
ac.insertNonTextBlock(elem);
ac.draw();
}
ac.nullGlobalBlockParams();
};
this.del=function(block_id,pad_id){
var block=null;
var i=0;
var blocks=ac.getPad(pad_id).blocks;
while(block=blocks[i]){
if(block.id==block_id){
blocks[i].id=-1;
blocks[i].options.banners_count=0;
break;
}
i++;
}
};
this.deleteAll=function(pad_id){
var blocks=ac.getPad(pad_id).blocks;
while(blocks.pop()){}
};
this.isDeleted=function(block){
block.id==-1&&block.options.banners_count==0;
};
this.pushAll=function(blocks,pad_id){
this.deleteAll(pad_id);
this.init();

var block=null;
var i=0;
while(block=blocks[i]){
this.push(block);
i++;
}
};
this.loadBlockCounter=function(pad_id,block_id){
if(this.length>0){
Begun.Utils.includeCounter(ac.Strings.urls.block_counter,{'pad_id':pad_id,'block_id':block_id});
}
};
this.getBlockById=function(block_id,blocks,pad_id){
var block=null;
var i=0;
blocks=blocks||ac.getPad(pad_id).blocks;
while(block=blocks[i]){
if(block.id==block_id){
return block;
}
i++;
}
return null;
};
this.getDomObj=function(block_id){
var resultBlock;
if(ac.getBlockIdTopMobile()==block_id){
try{
resultBlock=top.document&&top.document.getElementById(ac.Strings.css.block_prefix+block_id);
}catch(e){}
}
return resultBlock||Begun.$(ac.Strings.css.block_prefix+block_id)||null;
};
this.checkType=function(block,type){
return(this.getBlockType(block)==type);
};
this.checkViewType=function(block,viewtype){
return Begun.Utils.inList((block.options&&block.options.view_type),viewtype);
};
this.getBlockType=function(block){
return block&&block.options&&block.options.dimensions&&block.options.dimensions.type&&block.options.dimensions.type.toLowerCase()||"";
};
this.isBlockFixed=function(block){
return(/(\d+)x(\d+)/.test(this.getBlockType(block))||this.checkType(block,'rich'));
};
};

ac.Callbacks=new function(){
var _callbacks={};
var _extend=function(destination,source){
for(var property in source){
if(typeof source[property]=='object'){
var new_obj={};
for(var property2 in source[property]){
if(typeof source[property][property2]=='function'){
if((typeof destination[property]!=="undefined")&&(typeof destination[property][property2]=='function')){
new_obj[property2]=function(old_func,new_func,property2){
return function(args){
old_func.apply(property2=='click'?this:ac,[args]);
new_func.apply(property2=='click'?this:ac,[args]);
};
}(destination[property][property2],source[property][property2],property2);
}else{
new_obj[property2]=function(func,property2){
return function(args){
func.apply(property2=='click'?this:ac,[args]);
};
}(source[property][property2],property2);
}
}
}
destination[property]=new_obj;
}
}
return destination;
};
this.register=function(callbacks){
_extend(_callbacks,callbacks);
};
this.dispatch=function(obj,method,context_obj,args){
if(_callbacks[obj]&&typeof _callbacks[obj][method]=='function'){
args=args||[];
_callbacks[obj][method].apply(context_obj||this,args);
}else{
return null;
}
};
this.getCallbacks=function(){
return _callbacks;
};
};

ac.Tpls=new function(){
var css={};
css['default']='\
#begun-default-css {display:none !important;}\
';
css['block']='.begun_adv * {clear:none !important;color:#000 !important;float:none !important;margin:0 !important;padding:0 !important;letter-spacing:normal !important;word-spacing:normal !important;z-index:auto !important;font-size:12px !important;font:normal normal 12px Arial,sans-serif !important;text-transform:none !important;list-style:none !important;position:static !important;text-indent:0 !important;visibility:visible !important;}.begun_adv .begun_adv_common tr,.begun_adv .begun_adv_common td,.begun_adv .begun_adv_common a,.begun_adv .begun_adv_common b,.begun_adv .begun_adv_common div,.begun_adv .begun_adv_common span,.begun_adv .begun_adv_sys *,.begun_adv .begun_adv_all *{background:none !important;border:none !important;}#begun_block_{{block_id}} a {display:inline !important;vertical-align:baseline !important;text-decoration:underline !important;}#begun_block_{{block_id}} {height:auto !important;}#begun_block_{{block_id}} .begun_adv {font:12px/18px Arial,sans-serif !important;color:#000 !important;text-align:left !important;-moz-box-sizing:border-box !important;-webkit-box-sizing:border-box !important;box-sizing:border-box !important;}#begun_block_{{block_id}} .begun_adv b {font-weight:bold !important;display:inline !important;}#begun_block_{{block_id}} .begun_adv td {font-size:11px !important;}#begun_block_{{block_id}} .begun_adv,#begun_block_{{block_id}} .begun_adv table,#begun_block_{{block_id}} .begun_adv td,#begun_block_{{block_id}} .begun_adv div {padding:0 !important;text-align:left !important;}#begun_block_{{block_id}} .begun_adv table {border:none !important;border-collapse:collapse !important;}#begun_block_{{block_id}} .begun_adv td {vertical-align:middle !important;}#begun_block_{{block_id}} .begun_adv_sys {width:100% !important;}#begun_block_{{block_id}} .begun_adv_sys_sign_up {vertical-align:middle !important;}#begun_block_{{block_id}} .begun_adv_cell,#begun_block_{{block_id}} .begun_adv_all {text-align:left !important;}#begun_block_{{block_id}} .begun_adv_bullit {color:#aaa !important;}#begun_block_{{block_id}} .begun_adv_title,#begun_block_{{block_id}} .begun_adv_text {white-space:normal !important;display:block !important;}#begun_block_{{block_id}} .begun_adv_title,#begun_block_{{block_id}} .begun_adv_title * {font-weight:bold !important;}#begun_block_{{block_id}} .begun_adv_sys_logo div {vertical-align:middle !important;}#begun_block_{{block_id}} .begun_adv_sys_logo a:link,#begun_block_{{block_id}} .begun_adv_sys_logo a:visited,#begun_block_{{block_id}} .begun_adv_sys_logo a:hover,#begun_block_{{block_id}} .begun_adv_sys_logo a:active {color:{{block_logo_color}} !important;text-decoration:none !important;font-weight:bold !important;font-style:italic !important;}#begun_block_{{block_id}} .begun_adv_sys_logo a {margin-top:-1px !important;}#begun_block_{{block_id}} .begun_adv_sys_sign_up div {text-align:right !important;}#begun_block_{{block_id}} .begun_adv_ext .begun_adv_sys_logo,#begun_block_{{block_id}} .begun_adv_ext .begun_adv_sys_logo * {font-size:13px !important;line-height:17px !important;}#begun_block_{{block_id}} .begun_adv_ver .begun_adv_cell .begun_adv_cell,#begun_block_{{block_id}} .begun_adv_ver .begun_adv_all {padding:5px 2px 4px 5px !important;}#begun_block_{{block_id}} .begun_adv_ext .begun_adv_sys_logo div {width:3.8em !important;height:2.7ex !important;left:-4px !important;position:relative !important;top:-2px !important;}#begun_block_{{block_id}} .begun_adv_ext .begun_adv_text {padding:2px 0 4px 0 !important;}#begun_block_{{block_id}} .begun_adv_ext .begun_adv_contact span {padding-right:0.2em !important;}#begun_block_{{block_id}} .begun_adv_ext.begun_adv_ver .begun_adv_sys_sign_up {padding-right:5px !important;}#begun_block_{{block_id}} .begun_adv_fix .begun_adv_cell {padding:0 5px 0 9px !important;}#begun_block_{{block_id}} .begun_adv_fix .begun_adv_cell,#begun_block_{{block_id}} .begun_adv_fix .begun_adv_cell * {font-size:11px !important;line-height:11px !important;}#begun_block_{{block_id}} .begun_adv_fix .begun_adv_title {font-size:12px !important;line-height:13px !important;margin-bottom:2px !important;}#begun_block_{{block_id}} .begun_adv_fix .begun_adv_title * {font-size:12px !important;line-height:13px !important;}#begun_block_{{block_id}} .begun_adv_fix .begun_adv_text,#begun_block_{{block_id}} .begun_adv_fix .begun_adv_text * {font-size:11px !important;line-height:12px !important;}#begun_block_{{block_id}} .begun_adv_fix .begun_adv_sys_logo,#begun_block_{{block_id}} .begun_adv_fix .begun_adv_sys_logo * {font-size:13px !important;line-height:17px !important;}#begun_block_{{block_id}} .begun_adv_fix .begun_adv_all,#begun_block_{{block_id}} .begun_adv_fix .begun_adv_all *,#begun_block_{{block_id}} .begun_adv_fix .begun_adv_sys_sign_up,#begun_block_{{block_id}} .begun_adv_fix .begun_adv_sys_sign_up * {font:9px/11px Tahoma,Arial,sans-serif !important;}#begun_block_{{block_id}} .begun_adv_fix .begun_adv_sys_logo {position:relative !important;}#begun_block_{{block_id}} .begun_adv_fix .begun_adv_sys_logo div {height:17px !important;float:left !important;}#begun_block_{{block_id}} .begun_adv_fix .begun_adv_common {overflow:hidden !important;}#begun_block_{{block_id}}.begun_auto_rich .begun_adv_fix .begun_adv_common {overflow:visible !important;}#begun_block_{{block_id}} .begun_adv_fix .begun_adv_text {padding:2px 0 !important;}#begun_block_{{block_id}} .begun_adv_fix .begun_adv_contact span {padding-right:2px !important;}#begun_block_{{block_id}} .begun_adv_fix_ver .begun_adv_sys_logo,#begun_block_{{block_id}} .begun_adv_ext .begun_adv_sys_logo {padding-left:9px !important;}#begun_block_{{block_id}} .begun_adv_fix_ver .begun_adv_sys_logo div {width:51px !important;}#begun_block_{{block_id}} .begun_adv_fix_ver .begun_adv_sys_sign_up div {width:93% !important;}#begun_block_{{block_id}} .begun_adv_fix_ver .begun_adv_all {height:18px !important;padding:2px 0 0 9px !important;}#begun_block_{{block_id}} .begun_adv_fix_ver .begun_adv_block {margin:5px 0 !important;}#begun_block_{{block_id}} .begun_adv_fix_hor .begun_adv_common {margin-top:7px !important;text-align:left !important;}#begun_block_{{block_id}} .begun_adv_fix_hor .begun_adv_block {margin:0 !important;}#begun_block_{{block_id}} .begun_adv_fix_hor .begun_adv_sys_logo {width:53px !important;float:left !important;padding:0 !important;}#begun_block_{{block_id}} .begun_adv_fix_hor .begun_adv_sys_logo div {width:53px !important;}#begun_block_{{block_id}} .begun_adv_fix_hor .begun_adv_sys_sign_up {width:53px !important;float:left !important;clear:left !important;}#begun_block_{{block_id}} .begun_adv_fix_hor .begun_adv_sys_sign_up div {padding-left:4px !important;text-align:left !important;}#begun_block_{{block_id}} .begun_adv_fix_hor .begun_adv_table {margin-left:60px !important;display:block !important;}#begun_block_{{block_id}} .begun_adv_ext .begun_adv_cell .begun_adv_cell,#begun_block_{{block_id}} .begun_adv_ext .begun_adv_all {padding:5px 2px 4px 5px !important;}#begun_block_{{block_id}} .begun_adv .begun_adv_fav {padding-left:22px !important;background-position:left 1px !important;background-repeat:no-repeat !important;}#begun_block_{{block_id}} .begun_adv .banners_count_1 .begun_adv_fav {padding-left:0 !important;background-position:-1000px -1000px !important;}#begun_block_{{block_id}} .begun_adv_fav .begun_adv_title a {background-position:-1000px -1000px !important;}#begun_block_{{block_id}} .begun_adv_text,#begun_block_{{block_id}} .begun_adv_text * {color:#000 !important;}#begun_block_{{block_id}} .begun_adv_block {border:none !important;cursor:pointer !important;cursor:hand !important;}#begun_block_{{block_id}} .begun_scroll {overflow:hidden !important;}#begun_block_{{block_id}} .begun_scroll {position:relative !important;}#begun_block_{{block_id}} .begun_adv .begun_adv_cell .begun_adv_phone * {font-size:1px !important;}#begun_block_{{block_id}} .begun_adv_phone {width:12px !important;margin:1px 3px 0 0 !important;position:absolute !important;top:0 !important;left:0 !important;}#begun_block_{{block_id}} .begun_adv_phone_wrapper {padding-left:15px !important;white-space:nowrap !important;position:relative !important;display:inline-block !important;_display:inline !important;zoom:1 !important;}#begun_block_{{block_id}} .begun_adv_phone_wrapper.begun_adv_phone_no_icon {padding-left:0 !important;}#begun_block_{{block_id}} div.begun_adv_contact > .begun_adv_phone {margin:0 5px 0 0 !important;}#begun_block_{{block_id}} .begun_adv_phone b {border:#118f00 solid 0 !important;height:1px !important;font-size:1px !important;line-height:1px !important;display:block !important;overflow:hidden !important;}#begun_block_{{block_id}} .begun_adv_phone .p0,#begun_block_{{block_id}} .begun_adv_phone .p1,#begun_block_{{block_id}} .begun_adv_phone .p3,#begun_block_{{block_id}} .begun_adv_phone .p5,#begun_block_{{block_id}} .begun_adv_phone .p8 {background-color:#118f00 !important;}#begun_block_{{block_id}} .begun_adv_phone .p1,#begun_block_{{block_id}} .begun_adv_phone .p7,#begun_block_{{block_id}} .begun_adv_phone .p8 {margin:0 1px !important;}#begun_block_{{block_id}} .begun_adv_phone .p2,#begun_block_{{block_id}} .begun_adv_phone .p7 {border-width:0 4px !important;}#begun_block_{{block_id}} .begun_adv_phone .p3,#begun_block_{{block_id}} .begun_adv_phone .p6 {margin:0 2px !important;}#begun_block_{{block_id}} .begun_adv_phone .p0 {margin:0 3px !important;}#begun_block_{{block_id}} .begun_adv_phone .p4 {border-width:0 3px !important;}#begun_block_{{block_id}} .begun_adv_phone .p5 {margin:0 4px !important;}#begun_block_{{block_id}} .begun_adv_phone .p6 {border-width:0 2px !important;}#begun_block_{{block_id}} .begun_adv_phone .p8 {height:2px !important;}#begun_block_{{block_id}} .begun_adv_phone b {border-color:{{domain:color}} !important;}#begun_block_{{block_id}} .begun_adv_phone .p0,#begun_block_{{block_id}} .begun_adv_phone .p1,#begun_block_{{block_id}} .begun_adv_phone .p3,#begun_block_{{block_id}} .begun_adv_phone .p5,#begun_block_{{block_id}} .begun_adv_phone .p8 {background-color:{{domain:color}} !important;}#begun_block_{{block_id}} .begun_adv_phone {font-size:11px !important;line-height:11px !important;margin-top:{{phone_margin_top}} px !important;}#begun_block_{{block_id}} .begun_adv_title a,#begun_block_{{block_id}} .begun_adv_title a * {color:{{title:color}} !important;}#begun_block_{{block_id}} .begun_adv .begun_adv_title a:hover,#begun_block_{{block_id}} .begun_adv .begun_adv_title a:hover * {color:#f00 !important;color:{{title_hover:color}} !important;}#begun_block_{{block_id}} .begun_adv_title,#begun_block_{{block_id}} .begun_adv_title * {font-size:{{title:fontSize}} !important;}#begun_block_{{block_id}} .begun_adv_all,#begun_block_{{block_id}} .begun_adv_all * {color:{{domain:color}} !important;font-size:{{domain:fontSize}} !important;}#begun_block_{{block_id}} .begun_adv_text,#begun_block_{{block_id}} .begun_adv_text * {color:{{text:color}} !important;font-size:{{text:fontSize}} !important;text-decoration:none !important;}#begun_block_{{block_id}} .begun_adv_contact,#begun_block_{{block_id}} .begun_adv_contact a,#begun_block_{{block_id}} .begun_adv_contact span {color:{{domain:color}} !important;font-size:{{domain:fontSize}} !important;}#begun_block_{{block_id}} .begun_adv_sys_sign_up,#begun_block_{{block_id}} .begun_adv_sys_sign_up * {color:{{domain:color}} !important;font-size:{{domain:fontSize}} !important;}#begun_block_{{block_id}} .begun_adv_contact a {color:{{domain:color}} !important;text-decoration:none !important;}#begun_block_{{block_id}} .begun_adv .begun_adv_thumb .begun_thumb {float:left !important;display:block !important;_display:inline !important;z-index:1 !important;overflow:hidden !important;zoom:1 !important;margin:6px auto 5px 7px !important;}#begun_block_{{block_id}} .begun_adv .begun_adv_thumb .begun_thumb img {z-index:20 !important;}#begun_block_{{block_id}} .begun_adv .begun_adv_rich .begun_adv_image {float:left !important;margin-right:10px !important;top:8px !important;width:70px !important;height:70px !important;position:relative !important;}#begun_block_{{block_id}} .begun_adv_ver .begun_adv_rich .begun_adv_image {top:3px !important;}#begun_block_{{block_id}} .begun_adv .begun_adv_thumb .begun_adv_block {margin-left:60px !important;_zoom:1 !important;}#begun_block_{{block_id}} .begun_adv .begun_adv_thumb_default .begun_adv_block {margin-left:60px !important;}#begun_block_{{block_id}} .begun_adv .begun_adv_thumb_classic .begun_adv_block {margin-left:74px !important;}#begun_block_{{block_id}} .begun_adv .begun_adv_rich .begun_adv_block {margin-left:80px !important;}#begun_block_{{block_id}} .begun_adv_accordion .accordion_section .begun_adv_cell .begun_adv_block .begun_adv_title,#begun_block_{{block_id}} .begun_adv_accordion .accordion_section .begun_adv_cell .begun_adv_block .section {border:1px solid transparent !important;_border:none !important;}#begun_block_{{block_id}} .begun_adv_accordion .accordion_section .begun_adv_cell {vertical-align:top !important;}#begun_block_{{block_id}} .begun_adv_accordion .begun_adv_block {margin:5px 0 0 !important;}#begun_block_{{block_id}}.begun_auto_rich .begun_adv .banners_count_1 .begun_adv_cell,#begun_block_{{block_id}}.begun_auto_rich .begun_adv .banners_count_1 .begun_adv_cell *,#begun_block_{{block_id}}.begun_auto_rich .begun_adv .banners_count_1 .begun_adv_text,#begun_block_{{block_id}}.begun_auto_rich .begun_adv .banners_count_1 .begun_adv_text *,#begun_block_{{block_id}}.begun_auto_rich .begun_adv .banners_count_1 .begun_adv_title,#begun_block_{{block_id}}.begun_auto_rich .begun_adv .banners_count_1 .begun_adv_title * {text-align:left !important;}#begun_block_{{block_id}} .begun_adv .begun_adv_cell .begun_adv_phone_wrapper .begun_adv_phone *,#begun_block_{{block_id}}.begun_auto_rich .begun_adv .begun_adv_cell .begun_adv_phone_wrapper .begun_adv_phone * {font-size:1px !important;}#begun_block_{{block_id}} .begun_adv_ver .begun_adv_phone {margin-top:3px !important;}#begun_block_{{block_id}} .begun_adv .begun_alco_message {padding:12px 10px 15px 20px !important;position:relative !important;top:0px !important;font-size:9px !important;line-height:1.2em !important;color:#333333 !important;text-transform:uppercase !important;background-color:#F0F0F0 !important;}#begun_block_{{block_id}} .begun_adv .begun_alco_message span.begun_alco_attention,#begun_block_{{block_id}} .begun_adv .begun_adv_title span.begun_alco_attention {color:#FF0000 !important;}#begun_block_{{block_id}} .begun_adv .begun_alco_message span.begun_alco_attention {left:10px !important;position:absolute !important;top:10px !important;}#begun_block_{{block_id}} .begun_adv .begun_adv_title span.begun_alco_attention {margin-left:5px !important;font-weight:bold !important;}#begun_block_{{block_id}} .begun_adv_ver .begun_alco_message {padding:10px 7px 10px 7px !important;}#begun_block_{{block_id}} .begun_adv_ver .begun_alco_message span.begun_alco_attention {position:static !important;top:0 !important;left:0 !important;margin-right:5px !important;}#begun_block_{{block_id}} #begun_alco_{{block_id}}.begun_adv_ver {border-collapse:collapse !important;}\
#begun_block_{{block_id}} .begun_adv {\
background-color: {{block:backgroundColor}}; /* no !important for hover */\
border: 1px solid {{block:borderColor}}; /* no !important for hover */\
{{block:filter}} /* no !important for hover */\
}\
#begun_block_{{block_id}} .begun_adv *, .begun_adv *:hover {\
width: auto; /* no !important for rich-images */\
height: auto; /* no !important for rich-images */\
background: none; /* no !important for hover */\
border: none; /* no !important for hover */\
}\
#begun_block_{{block_id}} .begun_adv.begun_hover {\
background-color: {{block_hover:backgroundColor}}; /* no !important for hover */\
border: 1px solid {{block_hover:borderColor}}; /* no !important for hover */\
{{block_hover:filter}} /* no !important for hover */\
}\
#begun_block_{{block_id}} .begun_adv .begun_adv_rich .begun_active_image {\
z-index:1000;\
}\
#begun_block_{{block_id}} .begun_adv .begun_adv_rich .begun_active_image img {\
z-index:1000 !important;\
}\
#begun_block_{{block_id}} .begun_adv .begun_adv_rich .begun_adv_image img {\
border:1px solid {{block:borderColor}};\
position:absolute !important;\
top:0;\
left:0;\
z-index:20;\
cursor:pointer;\
}\
#begun_block_{{block_id}} .begun_adv .begun_adv_rich .begun_adv_picture {\
/*width:70px;\
height:70px;*/\
position:absolute !important;\
z-index:20;\
}\
#begun_block_{{block_id}} .begun_adv_accordion .begun_adv_table tr.accordion_section .section {\
padding-top:1px;\
height:1px;\
overflow:hidden;\
position:relative !important;\
}\
#begun_block_{{block_id}} .begun_adv.begun_adv_accordion .begun_adv_common .begun_adv_table tr.accordion_section.expanded td {\
background-color:{{accordion:backgroundColor}} !important;\
}\
#begun_block_{{block_id}} .begun_collapsed {\
height:45px !important;\
overflow:hidden !important;\
}\
#begun_block_{{block_id}} .begun_collapsed .begun_adv_title {\
margin-bottom:30px !important;\
}\
';
css['forOperaIE']='\
#begun_block_{{block_id}} .begun_adv_contact span.begun_adv_phone {\
float:none !important;\
position:static !important;\
vertical-align: top;\
display:inline-block !important;\
}\
#begun_block_{{block_id}} .begun_adv_phone_wrapper {\
padding-left:0 !important;\
position:static !important;\
display:inline !important;\
}\
';
var html={};
html['blck_place']='<div id="{{id}}"></div>';
html['link_iframe']='<iframe src="{{url}}" style="height:0;width:0;border:0"></iframe>';
html['bnnr_glue']=' <span class="begun_adv_bullit"> &#149; </span> ';
html['bnnr_phone']='\
<span class="begun_adv_phone"><b class="p0"></b><b class="p1"></b><b class="p2"></b><b class="p4"><b class="p3"></b></b><b class="p5"></b><b class="p6"><b class="p1"></b></b><b class="p7"></b><b class="p8"></b></span>\
';
html['bnnr_card']='\
<span class="begun_adv_phone_wrapper {{no_phone_class}}">{{phone}}<span class="begun_adv_card"><a target="_blank" href="{{url}}" class="snap_noshots">{{card_text}}</a></span></span>\
';
html['bnnr_ppcall']='\
<span class="begun_adv_phone_wrapper"><a href="javascript:void(0)" onclick="'+ac.Strings.js.ppcall_show+'({{banner_index}}, this, event, {{pad_id}}, \'{{link}}\', {{ua_type}})">{{phone}}<span class="begun_adv_card" title="&#1047;&#1074;&#1086;&#1085;&#1086;&#1082;&#32;&#1073;&#1077;&#1089;&#1087;&#1083;&#1072;&#1090;&#1085;&#1099;&#1081;">{{ppcall_text}}</span></a></span>\
';
html['bnnr_domain']='\
<span class="begun_adv_contact"><a class="snap_noshots" target="_blank" href="{{url}}" onmouseover="status=\'{{status}}\';return true" onmouseout="status=\'\';return true" title="{{fullDomain}}">{{domain}}</a></span> \
';
html['bnnr_geo']='\
<span class="begun_adv_city"><a class="snap_noshots" target="_blank" href="{{url}}" onmouseover="status=\'{{status}}\';return true" onmouseout="status=\'\';return true" title="{{fullDomain}}">{{geo}}</a></span>\
';
html['bnnr_thumb']='\
<a href="{{url}}" class="begun_thumb snap_noshots" style="width:{{width}}px !important;height:{{height}}px !important;{{additionalStyles}}" target="_blank"><img src="{{src}}" {{pngfix}} {{mouse_events}} width="{{width}}" height="{{height}}" alt="" style="width:{{width}}px !important;background-color:{{bgcolor}};" /></a>\
';
html['bnnr_picture']='\
<div class="begun_adv_image"><a href="{{url}}" class="snap_noshots" target="_blank"><img src="{{src}}" _big_photo_src="{{big_photo_src}}" _small_photo_src="{{src}}" class="begun_adv_picture" alt="" /></a></div>\
';
html['block_alco']='\
<div class="begun_alco_message"><span class="begun_alco_attention">*</span>\
&#1063;&#1088;&#1077;&#1079;&#1084;&#1077;&#1088;&#1085;&#1086;&#1077; &#1091;&#1087;&#1086;&#1090;&#1088;&#1077;&#1073;&#1083;&#1077;&#1085;&#1080;&#1077; &#1072;&#1083;&#1082;&#1086;&#1075;&#1086;&#1083;&#1103; &#1074;&#1088;&#1077;&#1076;&#1080;&#1090; &#1042;&#1072;&#1096;&#1077;&#1084;&#1091; &#1079;&#1076;&#1086;&#1088;&#1086;&#1074;&#1100;&#1102;\
</div>\
';
html['bnnr_alco_attn']='\
<span class="begun_alco_attention">*</span>\
';
html['blck_hover']=' onmouseover="Begun.Utils.addClassName(this, \'begun_hover\');" onmouseout="Begun.Utils.removeClassName(this, \'begun_hover\');"';
html['search_banner_swf']='<object classid="clsid:D27CDB6E-AE6D-11cf-96B8-444553540000" width="{{width}}" height="{{height}}"><param name="movie" VALUE="{{source}}&link1={{url}}"><param name="wmode" value="opaque"><param name="allowScriptAccess" value="always"><param name="quality" VALUE="high"><embed src="{{source}}&link1={{url}}" quality="high" width="{{width}}" height="{{height}}" type="application/x-shockwave-flash" wmode="opaque"></embed></object>';
html['search_banner_js']='';
html['search_banner_img']='<a href="{{url}}" target="_blank"><img src="{{source}}&redir=1" border="0" width="{{width}}" height="{{height}}" style="{{styles}}" /></a>{{close_button}}';
html['close_button']='<a href="javascript:void(0);" class="begun_close_button" onclick="document.getElementById(\'begun_block_{{block_id}}\').parentNode.removeChild(document.getElementById(\'begun_block_{{block_id}}\'));" title="&#1047;&#1072;&#1082;&#1088;&#1099;&#1090;&#1100;">&#215;</a>';

this.getCSS=function(type){
return css[type];
};
this.getHTML=function(type){
return html[type];
};
this.addTpls=function(newTpls){
var types=['html','css'];
var i=0;
var type=null;
var is_default_css_override=false;
if(css['default']&&window['begun_css_tpls']&&window['begun_css_tpls']['default']&&css['default']!=window['begun_css_tpls']['default']){
is_default_css_override=true;
}
var tplContainer=typeof newTpls==="undefined"?window:newTpls;
while(type=types[i]){
if(typeof tplContainer['begun_'+type+'_tpls']!=="undefined"){
var j=0;
var tpl=null;
while(tpl=tplContainer['begun_'+type+'_tpls'][j]){
Begun.extend(eval(type),tpl);
j++;
}
}
i++;
}
return is_default_css_override;
};
};

ac.Customization=new function(){
var _this=this;
var callbacksLogged=false;
this.init=function(){
if(typeof window.begun_urls!=="undefined"){
_this.setURLs(window.begun_urls);
window.begun_urls=null;
}
if(typeof window.begun_callbacks!=="undefined"){
if(!callbacksLogged){
Begun.Utils.includeCounter("http://autocontext.begun.ru/blockcustom?pad_id={{pad_id}}&log_callbacks=1",{"pad_id":(window.begun_auto_pad||'')});
callbacksLogged=true;
}
_this.setCallbacks(window.begun_callbacks);
window.begun_callbacks=null;
}
if(_this.setTpls()||!arguments.callee.run){
ac.printDefaultStyle();
arguments.callee.run=true;

}
};
this.setURLs=function(urls){
Begun.extend(ac.Strings.urls,urls||{});
};
this.setCallbacks=function(callbacks){
ac.Callbacks.register(callbacks||{});
};
this.setTpls=function(newTpls){
ac.Tpls.addTpls(newTpls);
};
};
})();

(function(){
var ac=Begun.Autocontext;

function onContent(f){
var a,d=document,w=window,c="__onContent__",e="addEventListener",o="opera",r="readyState",
s="<scr".concat("ipt defer src='//:' on",r,"change='if (this.",r,"==\"complete\") {this.parentNode.removeChild(this);",c,"()}'></scr","ipt>");
w[c]=(function(o){
return function(){
w[c]=function(){};
for(a=arguments.callee;!a.done;a.done=1){
f(o?o():o);
}
};
})(w[c]);
if(d[e]){
d[e]("DOMContentLoaded",w[c],false);
}
var bb=Begun.Browser;
if((bb.WebKit)||(bb.Opera&&bb.less(9))){
(function(){
/loaded|complete/.test(d[r])?w[c]():setTimeout(arguments.callee,1);
})();
}else if(bb.IE){
d.write(s);
}
}

function logPpcalls(){
var ppcalls=(function(){
var res="[";
var separatePads="";
var separateBlocks;
var pagePads=ac.Pads.getPads();
var experimentData=function(pad_id){
return"\"region_id\":"+(ac.getFeed(pad_id)&&ac.getFeed(pad_id).params&&ac.getFeed(pad_id).params.user_region_id?ac.getFeed(pad_id).params.user_region_id:"\"none\"")+",";
};
var isAnyPpcallAd=false;
for(var i=0;i<pagePads.length;i++){
res+=separatePads+"{\"pad_id\":"+pagePads[i].pad_id+","+experimentData(pagePads[i].pad_id)+"\"blocks\":[";
separateBlocks="";
for(var j=0;j<pagePads[i].blocks.length;j++){
var blockElement=ac.Blocks.getDomObj(pagePads[i].blocks[j].id);
if(!blockElement){
continue;
}
var links=blockElement.getElementsByTagName("a");
var quantity=0;
for(var k=0;k<links.length;k++){
var onClickAttr=links[k].getAttribute("onclick");
if(onClickAttr&&onClickAttr.toString().indexOf(ac.Strings.js.ppcall_show)!==-1){
quantity++;
isAnyPpcallAd=true;
}
}
res+=separateBlocks+"{\"id\":"+pagePads[i].blocks[j].id+",\"ppcall_count\":"+quantity+"}";
separateBlocks=",";
}
res+="]}";
separatePads=",";
}
if(isAnyPpcallAd){
return res+"]";
}else{
return"";
}
})();
var ppcall_form=(function(){
var types={
"default":0,
"direct_phone":0,
"text_phone":0
};
var pads=ac.Pads.getPads();
for(var n=0,ln=pads.length;n<ln;n++){
var banners=ac.getBanners(pads[n].pad_id)['autocontext'];
var banners_length=banners.length;
for(var i=0;i<banners_length;i++){
if(banners[i].ppcall&&banners[i].ppcall_form_type){
types[banners[i].ppcall_form_type]+=1;
}
}
}
return types;
})();
if(ppcalls.length>2){
Begun.Utils.includeCounter(ac.Strings.urls.ppcalls_counter,{'ppcall_data':encodeURIComponent(ppcalls),'ppcall_form':encodeURIComponent(Begun.Utils.toJSON(ppcall_form))});
}
}

Begun.Utils.addEvent(window,'load',function(){
logPpcalls();
});

onContent(function(){
ac.Callbacks.dispatch('blocks','draw',ac);
if(!Begun.Browser.IE){
var styleElements=document.getElementsByTagName("style");
for(var i=0;i<styleElements.length;i++){
var styleId=styleElements[i].getAttribute("id");
if((!styleId||!/begun-block-css-/.test(styleId))&&/#begun_block_/i.test(styleElements[i].innerHTML)){
Begun.Utils.includeCounter("http://autocontext.begun.ru/blockcustom?pad_id={{pad_id}}&log_css_override=1",{"pad_id":(window.begun_auto_pad||'')});
break;
}
}
}
ac.domContentLoaded=true;
});

ac.Monitor.init();
})();
Begun.Autocontext.init();
}
}

if(typeof Begun.Autocontext==="object"){
Begun.Autocontext.init();
}

Begun.Scripts.addStrictFunction(Begun.Scripts.Callbacks['ac']);
}

if(typeof Begun.Scripts!=="object"){
(function(){
var scripts={
"begun_scripts":"29561"
};
var baseUrl=window.begun_urls&&window.begun_urls.base_scripts_url?window.begun_urls.base_scripts_url:"http://autocontext.begun.ru/";
for(var scriptName in scripts){
if(scripts.hasOwnProperty(scriptName)){
document.write("<scr"+"ipt type=\"text/javascript\" src=\""+baseUrl+"acp/"+scriptName+"."+scripts[scriptName]+".js"+"\"></scr"+"ipt>");
}
}
})();
}else{
if(typeof Begun.Scripts.addStrictFunction!=="undefined"){
begun_load_autocontext();
}
}
