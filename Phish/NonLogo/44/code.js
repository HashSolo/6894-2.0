<!--
//function disableselect(e){
//return false
//}

function reEnable(){
//return true
}

//if IE4+
//document.onselectstart=new Function ("return false")

//if NS6
//if (window.sidebar){
//document.onmousedown=disableselect
//document.onclick=reEnable
//}


//CLICKFORCE
/* Configuration */
var num = 1;
var pissoff = 'Die Registrierung ist noch nicht vervollständigt.\n\nBitte klicken Sie zuerst auf einen der Banner!';

allow = Array();
allow[num] = 2;
function gotoit(link,target){
 if (link != num){
   allow[link] = 2;
 } else {
   for (i=1;i<=num;i++){
      if (allow[i] != 2){
         i = num + 1; lemmeIn = 0;
      } else {
         lemmeIn = 1;
      }
   }
   if (lemmeIn == 1){
      if (target != '+'){window.location = target;}
   } else {
      alert(pissoff);
   }
 }
}

var snum = 1;
var adele = 'Bitte besuchen Sie vor der Anmeldung einen der Sponsoren!';
sallow = Array();
sallow[snum] = 2;
function go2it(val){
 if (val != snum){ sallow[val] = 2;
 } else {
 for (i=1;i<=snum;i++){ if (sallow[i] != 2){ i = snum + 1; slemmeIn = 0;} else { slemmeIn = 1;}}
 if (slemmeIn == 1){ check_boxes();} else { alert(adele);}
 }
}
//--->
