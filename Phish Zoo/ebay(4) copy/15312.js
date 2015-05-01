var rdm_id          = 15312;
var rdm_site_id       = 5161;
var rdm_show_desc     = 0;
//var rdm_show_desc   = 0;
var rdm_show_photo    = 1;
//var rdm_show_photo  = 1;
var rdm_show_align    = 'center';
var title_position    = 1;
var const_title_before= 0;
var const_title_after = 1;

var rdm_class_title   = '';
var rdm_class_desc    = '';
var rdm_class_photo   = '';
var rdm_style_photo   = '';

var rdm_header        = '<table border="0" width="100%"><tr>';
var rdm_footer        = '</tr></table>';
var rdm_delimeter     = '';

var rdm_line_before   = '<td>';
var rdm_line_after    = '</td>';

var rdm_title_before  = '';
var rdm_title_after   = '<br>';

var rdm_desc_before   = '';
var rdm_desc_after    = '';

var rdm_data          = '';
var rdm_img_size      = ' width="90" height="90" ';
var rdm_img_path      = 'http://img.readme.ru/news/90/';
var rdm_css_styles    = 'div#rdminfrm_15312 div {padding: 2px} div#rdminfrm_15312 ul {list-style-type: none} div#rdminfrm_15312 table {border: none; background-color: #CCCCCC} div#rdminfrm_15312 table td {width: 50%; text-align: left; vertical-align: top; ; border: none} div#rdminfrm_15312 div.rdm_title_15312 a, div#rdminfrm_15312 div.rdm_title_15312 a:hover {font-family: Tahoma; font-weight: normal; font-size: 12px; color: #333333; text-decoration: underline} div#rdminfrm_15312 div.rdm_img_15312 {float: left} div#rdminfrm_15312 div.rdm_img_15312 img {border: none} div#rdminfrm_15312 div.rdm_text_15312 a, div#rdminfrm_15312 div.rdm_text_15312 a:hover {font-family: Tahoma; font-weight: normal; font-size: 12px; color: black; text-decoration: none}';

var rdm_initid        = 'rdminfrm_15312';
var rdm_initid_tags   = 'rdminfrm_15312_tags';

var rdm_host          = 'http://click.readme.ru/';

var rdm_num           = 2;

var rdm_domain        = 'http://click.readme.ru/';

var rdm_GET_Keys;
var rdm_GET_Values;
var rdm_GET_Count = 0;
var rdm_GET_Default = '';
var rdm_context = '';

function getCookie(name){var dc=document.cookie;var prefix=name+"=";var begin=dc.indexOf("; "+prefix);if(begin==-1){begin=dc.indexOf(prefix);if(begin!=0){return null;}}else{begin += 2;}var end=dc.indexOf(";", begin);if (end==-1){end=dc.length;}return unescape(dc.substring(begin+prefix.length,end));}
function setCookie(name,value,expires,path,domain,secure){document.cookie=name+"="+escape(value)+((expires)?"; expires="+expires.toGMTString():"")+((path)?"; path="+path:"")+((domain)?";domain="+domain:"")+((secure)?"; secure":"");}

var rdm_init          = document.getElementById(rdm_initid);

function RDMI(){if(rdm_init){rdm_init.innerHTML = rdm_data;}}
function RDMH(){rdm_data+=rdm_header;}
function RDMF(){rdm_data+=rdm_footer;RDMI();}
function RDMAdd(title,news_url,url,desc,photo,special){

    tit = '<div class="rdm_title_'+rdm_id+'"><a target="_blank" href="'+news_url+'/in.php?id='+url+'">'+title+'<'+'/a></div>';
    str=rdm_line_before;
    if(title_position == const_title_before) {str+= tit;}
    if(rdm_show_photo&&photo!='') {str+='<div class="rdm_img_'+rdm_id+'"><a target="_blank" href="'+news_url+'/in.php?id='+url+'"><img src="'+rdm_img_path+photo+'" '+rdm_img_size+' border=0 /><'+'/a></div>';}
    if(title_position == const_title_after) {str+= tit;}
    if(rdm_show_desc&&desc!='') {str+='<div class="rdm_text_'+rdm_id+'"><a target="_blank" href="'+news_url+'/in.php?id='+url+'" >'+desc+'<'+'/a></div>';}
    rdm_data+=str+rdm_line_after;
}

rdm_context_link = rdm_context != '' ? 'context='+encodeURIComponent(rdm_context) : '';

if (rdm_init) {
    document.write('<style type="text/css">'+rdm_css_styles+'<'+'/style>');
    document.write('<scr'+'ipt charset="utf-8" language="javascript" type="text/javascript" src="'+rdm_host+'js/h.js"><'+'/scr'+'ipt>');
    document.write('<scr'+'ipt charset="utf-8" language="javascript" type="text/javascript" src="'+rdm_host+'js/id/15312.php?'+rdm_context_link+'"><'+'/scr'+'ipt>');
    document.write('<scr'+'ipt charset="utf-8" language="javascript" type="text/javascript" src="'+rdm_host+'js/f.js"><'+'/scr'+'ipt>');
}