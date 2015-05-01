hcSetImageGo("need_help","need_help_on.gif",false);
hcSetImageGo("need_close","close_on.gif",false);
var needhelpwidth=document.all.need_help.width
var needhelpheight=document.all.need_help.height
var needcloseheight=document.all.need_close.height
document.all.need_help.width= "1";
document.all.need_help.height= "1";
document.all.need_close.width= "1";
document.all.need_close.height= "1";
document.all.mylayer.style.visibility = "visible";


function special508() {
document.all.mylayer.style.visibility = "hidden";
document.all.need_help.width= needhelpwidth;
document.all.need_help.height=needhelpheight;
document.all.need_close.height=needcloseheight;
document.all.need_close.width=needhelpwidth;
}

setTimeout("special508()",2000);
