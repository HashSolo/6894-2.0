/* Touch Clarity. http://www.touchclarity.com
 * Copyright (c) Touch Clarity Ltd 2001-2006. All rights reserved. Patent Pending.
 */

function tc_optimise(id,default_content) {
  if (tc_ccs[id]==null) { tc_ccs[id]=new tc_cc(default_content);
  } else { tc_ccs[id].default_content=default_content; }
  if (tc_ccs[id].content!="") {
    if (tc_ccs[id].content!="__default__") {
      document.write(tc_ccs[id].content);
      tc_reporter+=tc_ccs[id].reporter;
      tc_ccs[id].displayed=true;
    } else { document.write(tc_ccs[id].default_content); }
  } else if (tc_frame) {
    document.write("<div id='tc_content_"+id+"'></div>");
    tc_ccs[id].rendered=true;
    if (typeof default_content!='undefined') {
      var mt=(document.all?"document.all":"document.getElementById");
      tc_ccs[id].timeout_id=setTimeout("if(!tc_ccs['"+id+"'].displayed){tc_ccs['"+id+"'].defaulted=true;"+mt+"('tc_content_"+id+"').innerHTML=tc_ccs['"+id+"'].default_content;}tc_report();",tc_timeout);
    }
  } else { document.write(default_content); }
  tc_report();
}

function tc_set_content(cc_id,co_id,str) {
  if (tc_ccs[cc_id]==null) tc_ccs[cc_id]=new tc_cc('');
  if (tc_ccs[cc_id].defaulted) return;
  if (tc_ccs[cc_id].rendered) {
    if (str=='__default__') str=tc_ccs[cc_id].default_content;
    if (document.getElementById) {document.getElementById("tc_content_"+cc_id).innerHTML+=str}
    else if (document.all) {document.all("tc_content_"+cc_id).innerHTML+=str}
    tc_reporter+="&displayed="+co_id;
    tc_ccs[cc_id].displayed=true;
  } else { tc_ccs[cc_id].content += str; tc_ccs[cc_id].reporter+="&displayed="+co_id;}
}

function tc_cc(default_content) {
  this.rendered=false;
  this.displayed=false;
  this.defaulted=false;
  this.content="";
  this.default_content=default_content;
  this.timeout_id=null;
  this.reporter="";
}

function tc_report() {
  if (tc_reported) return;
  var def="";
  for (var cc=0; cc < tc_containers.length; cc++) {
    var id=tc_containers[cc];
    if ((tc_ccs[id]==null) || (!tc_ccs[id].displayed && !tc_ccs[id].defaulted)) return;
    if (tc_ccs[id].defaulted) def+="&default="+id;
  }
  tc_log(tc_url,tc_products,tc_reporter+def);
  tc_reported=true;
}

function tc_opt_loader() {
  if (tc_logging_active&&tc_configured()&&(typeof tc_done==tc_ud||tc_done==false)) {
    tc_tag_version="3.5";
    tc_log_call="<scr"+"ipt src='"+(typeof tc_test!=tc_ud?tc_test:tc_get_log_URL("c",tc_url,tc_products,tc_time))+"'></scr"+"ipt>";
    if (tc_frame) document.writeln("<iframe name='tc_iframe' src='"+tc_log_path+'/'+tc_log_page+"' width=0 height=0 style='position:absolute;left:-100px'></iframe>");
    tc_done = true;
  }
}

var tc_reporter="";
var tc_reported=false;
tc_opt_loader();