/* Touch Clarity. http://www.touchclarity.com
 * Copyright (c) Touch Clarity Ltd 2001-2006. All rights reserved. Patent Pending.
 */

function tc_configured() {
  tc_tag_version="3.5";
  tc_dtimeout=5000;
  tc_d_loc=window.location;
  tc_sent=0;
  if (typeof tc_server_url==tc_ud||typeof tc_site_id==tc_ud) return false;
  if (typeof tc_log_page==tc_ud||tc_log_page=="") tc_log_page="logging.html";
  tc_layer=(document.layers?1:0);
  tc_frame=(document.getElementById||document.all?1:0);
  tc_timeout=(typeof tc_timeout==tc_ud?tc_dtimeout:tc_timeout*1000);
  tc_encfn=(typeof encodeURIComponent!=tc_ud?encodeURIComponent:escape);
  tc_http="http"+(tc_d_loc.href.substring(0,6)=="https:"?"s":"")+"://";
  tc_server_url=tc_http+tc_server_url;
  tc_url=((typeof tc_page_alias!=tc_ud)?tc_page_alias:tc_d_loc.href);
  if (typeof tc_extra_info!=tc_ud) tc_url+=(tc_url.indexOf("?")>0?"&":"?")+tc_extra_info;
  tc_extra_info="";
  tc_products=(typeof tc_products==tc_ud?"":tc_products);
  tc_ccs = new Array();
  tc_referrer=(typeof tc_referrer!=tc_ud&&tc_referrer!=""&&tc_referrer!=null?tc_referrer:(typeof document.referrer==tc_ud?tc_ud:(document.referrer==null?"null":(document.referrer==""?"empty":document.referrer))));
  tc_time = new Date().getTime();
  return true;
}

function tc_log(alias, products, displayed) {
  if (!tc_logging_active) return;
  alias=tc_fixURL(alias);
  tc_image=new Image();
  tc_image.src=tc_get_log_URL("i",alias,tc_products,new Date().getTime(), displayed);
}

function tc_redirect(target,url,alias,winproperties,products,script) {
  if (typeof url==tc_ud||url=="") return;
  if (typeof script==tc_ud||script=="") script="tc_d_loc.href='"+url+"'";
  url=tc_fixURL(url);
  if (typeof alias==tc_ud) alias=url;
  alias=tc_fixURL(alias);
  if (typeof target==tc_ud||target==""||target=="_self") {
    if (tc_logging_active) {
      tc_timer=new Image();
      tc_timer.onload=function() { eval(script); clearTimeout(tc_timeout_id); }
      tc_timer.onerror=function() { eval(script); clearTimeout(tc_timeout_id); }
      tc_timer.src=tc_get_log_URL("i",alias,products,new Date().getTime());
      tc_timeout_id=setTimeout(script,tc_timeout);
    } else { eval(script); }
  } else if (typeof target=="object"&&target.document) {
    if (tc_logging_active) { tc_timer=new Image();tc_timer.src=tc_get_log_URL("i",alias,products); }
    target.location.href=url;
  } else { tc_open_window(target,url,alias,winproperties,products); }
}

function tc_open_window(name,url,alias,winproperties,products) {
  if (typeof url==tc_ud||url=="") return false;
  if (tc_logging_active) { tc_timer=new Image();tc_timer.src=tc_get_log_URL("i",alias,products,new Date().getTime()); }
  if (typeof winproperties==tc_ud) return window.open(url,name);
  else { return window.open(url,name,winproperties) }
}

function tc_dltime() {
  if (!(document.getElementById||document.all)) return false;
  if (tc_logging_active&&(typeof tc_done!="undefined")&&tc_done&&!tc_sent) { setTimeout("tc_dltime()",1000);return false; }
  var sent=tc_sent;
  tc_image=new Image();
  tc_image.src=tc_get_log_URL("d");
  return sent;
}

function tc_get_log_URL(type,locn,products,time,displayed) {
  if (typeof type==tc_ud) type='i';
  var url=tc_server_url+"/"+type+"?siteID="+tc_site_id;
  if (type!="d") {
    url+="&ts="+(typeof time!=tc_ud?time:tc_time);
    var al = tc_isAlias(locn);
    if (typeof tc_containers!=tc_ud) {for(var cc=0; cc < tc_containers.length; cc++) {url+="&ccID="+tc_containers[cc];}}
    if (type=="c") url+="&log=no";
    if (al) url+="&alias=true";
    if ((typeof products!=tc_ud)&&products.length) url+="&prod="+tc_encfn(products);
    if (typeof displayed!=tc_ud) url+=displayed;
    if (locn==tc_ud) locn=tc_d_loc;
    locn=tc_encfn(locn);
    while (locn.length>1999-url.length) locn=locn.substring(0,locn.lastIndexOf(tc_encfn("&")));
    url+="&location="+locn;
    var dg=new Object();
    dg.tagv=tc_tag_version;
    dg.tz=0-(new Date().getTimezoneOffset());
    dg.r=tc_encfn(tc_referrer);
    if (al) dg.aliased=tc_encfn(tc_d_loc.href);
    dg.title=tc_encfn(document.title);
    if (screen) {dg.cd=screen.colorDepth;dg.ah=screen.availHeight;dg.aw=screen.availWidth;dg.sh=screen.height;dg.sw=screen.width;dg.pd=screen.pixelDepth;}
    for (var key in dg){
        if((typeof(dg[key])!=('function'))&&(typeof(dg[key])!=('array'))&&(typeof(dg[key])!=('object'))){
            var param="&"+key+"="+dg[key];
            if (url.length+param.length<2000){url+=param;}
            else {break;}
        }
    }
  } else {
    url+="&dlts="+tc_time+"&dl="+(new Date().getTime()-tc_time);
  }
  return url;
}

function tc_fixURL(url) {
  if (url=="") { return tc_d_loc.href }
  if ((url.substring(0,4)!='http')&&(url.substring(0,1)!="/")) { url=tc_d_loc.pathname.substring(0,tc_d_loc.pathname.lastIndexOf('/')+1)+url }
  if (url.substring(0,1)=="/") { url=tc_http+tc_d_loc.host+url }
  return url;
}

function tc_isAlias(alias) {
  alias=(typeof alias!=tc_ud?alias:(typeof tc_page_alias==tc_ud?"":tc_page_alias));
  alias=tc_fixURL(alias);
  if (alias.indexOf("?")>0) alias=alias.substring(0,alias.indexOf("?"));
  return (alias != tc_http+tc_d_loc.host+tc_d_loc.pathname);
}

function tc_loader() {
  tc_ud="undefined";
  if (typeof tc_log_path==tc_ud||tc_log_path=="") tc_log_path="/touchclarity";
  if ((typeof tc_containers!="undefined")&&(tc_containers.length>0)) { document.write("<scr"+"ipt language='JavaScript' type='text/javascript' src='" + tc_log_path + "/optimise.js'></scr"+"ipt>\n"); }
  else {
    if (tc_logging_active&&tc_configured()&&(typeof tc_done==tc_ud||tc_done==false)) {
      url=tc_fixURL(tc_url);
      tc_image=new Image();
      tc_image.onload=function(){tc_sent=true;}
      tc_image.src=tc_get_log_URL("i",url,tc_products,tc_time);
    }
    tc_done = true;
  }
}

tc_loader();
