
/***************** Affiliate System 0.1 (cs+a0.1) *****************/
/******** Cyopright by Promotional & Media Development GmbH *******/
var keyStr='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=';
function utf8_decode(utftext) {
var string = ''; var i = 0; var c = c1 = c2 = 0;
while ( i < utftext.length ) {
c = utftext.charCodeAt(i);
if (c < 128) { string += String.fromCharCode(c); i++; }
else if((c > 191) && (c < 224)) {
c2 = utftext.charCodeAt(i+1);
string += String.fromCharCode(((c & 31) << 6) | (c2 & 63));
i += 2; } else { c2 = utftext.charCodeAt(i+1);
c3 = utftext.charCodeAt(i+2);
string += String.fromCharCode(((c & 15) << 12) | ((c2 & 63) << 6) | (c3 & 63));
i += 3; } } return string; }
function f1149e02eac0915af66d3bb7638cf50b022b21e00(input){ var bits; var decOut = ''; var i = 0;
var output = ''; var chr1, chr2, chr3; var enc1, enc2, enc3, enc4; var i = 0;
input = input.replace(/[^A-Za-z0-9\+\/\=]/g, '');
while (i < input.length) {
enc1 = keyStr.indexOf(input.charAt(i++)); enc2 = keyStr.indexOf(input.charAt(i++));
enc3 = keyStr.indexOf(input.charAt(i++)); enc4 = keyStr.indexOf(input.charAt(i++));
chr1 = (enc1 << 2) | (enc2 >> 4); chr2 = ((enc2 & 15) << 4) | (enc3 >> 2);
chr3 = ((enc3 & 3) << 6) | enc4;
output = output + String.fromCharCode(chr1);
if (enc3 != 64) { output = output + String.fromCharCode(chr2); }
if (enc4 != 64) { output = output + String.fromCharCode(chr3); }
} return output; }
var sjdc800f9f9c1d908536d2b7094f80e902dd36d7f33 = 'PGRpdiBpZD0ieGMzZTgxNjRjMzA3ZDk4MjA3NTI1N2FkNzdhNm'+
'NmODRlNzNmZGJhOTIiIHN0eWxlPSJwb3NpdGlvbjogYWJzb2x1'+
'dGU7IHdpZHRoOiA4MDBweDsgaGVpZ2h0OiA1NDBweDsgdG9wOi'+
'AyMHB4OyBsZWZ0OiAyMHB4OyB6LWluZGV4OiAxMDAwMDA7IGRp'+
'c3BsYXk6IG5vbmU7Ij4KCgkJCQk8ZGl2IHN0eWxlPSJiYWNrZ3'+
'JvdW5kLWNvbG9yOiAjRjRGNEY0OyBib3JkZXI6IDJweCBzb2xp'+
'ZCAjNTg1ODU4OyB3aWR0aDogMTAwJTsgaGVpZ2h0OiAxMDAlOy'+
'B0ZXh0LWFsaWduOiBjZW50ZXI7IHotaW5kZXg6IDEwMDAwMTsi'+
'PgoJCQkJCTxkaXYgc3R5bGU9Im1hcmdpbi10b3A6IDdweDsgaG'+
'VpZ2h0OiAxOHB4OyB0ZXh0LWFsaWduOiBsZWZ0OyAgei1pbmRl'+
'eDogMTAwMDAyOyI+CgkJCQkJCTxzcGFuIHN0eWxlPSJtYXJnaW'+
'4tbGVmdDogMTRweDsgd2lkdGg6IDMwMHB4OyBmbG9hdDogbGVm'+
'dDsgZm9udC1mYW1pbHk6IEFyaWFsOyBmb250LXNpemU6IDExcH'+
'g7IGNvbG9yOiAjNTg1ODU4OyI+QURWRVJUSVNFTUVOVCBCWSBC'+
'SU5MQVlFUi5DT008L3NwYW4+CgkJCQkJCTxzcGFuIHN0eWxlPS'+
'JtYXJnaW4tcmlnaHQ6IDE0cHg7IGZsb2F0OiByaWdodDsiPgo8'+
'YSBocmVmPSJqYXZhc2NyaXB0OnZvaWQoMCk7IiBpZD0iamVmNj'+
'ExZDMwMzczODQxYTJkM2M1YWYyZWYwMTlhNzZhMWRiNGUyOTEi'+
'IHN0eWxlPSJjb2xvcjogIzU4NTg1ODsgIHRleHQtZGVjb3JhdG'+
'lvbjpub25lOyBmb250LXdlaWdodDogYm9sZDsgZm9udC1mYW1p'+
'bHk6IEFyaWFsOyBmb250LXNpemU6IDExcHg7ICI+TUFYSU1JWk'+
'U8L2E+PHNwYW4gc3R5bGU9ImZvbnQtZmFtaWx5OiBBcmlhbDsg'+
'Zm9udC1zaXplOiAxMXB4OyBjb2xvcjogIzU4NTg1ODsiPiAvID'+
'wvc3Bhbj48YSBocmVmPSJqYXZhc2NyaXB0OnZvaWQoMCk7IiBp'+
'ZD0iYTA3MTVmMTU4N2UyOTMzMjgxM2I1NmFmNDZmZDQ1Yjc1MW'+
'MwYzE4NDgiIHN0eWxlPSJ0ZXh0LWRlY29yYXRpb246bm9uZTsg'+
'Y29sb3I6ICM1ODU4NTg7IGZvbnQtZmFtaWx5OiBBcmlhbDsgZm'+
'9udC1zaXplOiAxMXB4OyAiPkNMT1NFPC9hPgoJCQkJCQk8L3Nw'+
'YW4+CgkJCQkJPC9kaXY+CgkJCQkJPGRpdiBzdHlsZT0ibWFyZ2'+
'luOiAwIGF1dG87IHdpZHRoOiA3NjRweDsgaGVpZ2h0OiA1MDBw'+
'eDsgei1pbmRleDogMTAwMDAyOyBiYWNrZ3JvdW5kLWNvbG9yOi'+
'NmZmZmZmY7ICAiPgoJCQkJCQk8aWZyYW1lIGZyYW1lYm9yZGVy'+
'PSIwIiBzdHlsZT0iYm9yZGVyOjBweDsgbWFyZ2luOjBweDsgcG'+
'FkZGluZzowcHg7ICB6LWluZGV4OiAxMDAwMDM7IiBzcmM9IiIg'+
'bmFtZT0ibzZmMDUwMjk1ZDhjZDExMzNmMGFjYzg3Y2Q4NTc5Nm'+
'FlMGVhMjA0NGQiIHNjcm9sbGluZz0iYXV0byIgaWQ9Im82ZjA1'+
'MDI5NWQ4Y2QxMTMzZjBhY2M4N2NkODU3OTZhZTBlYTIwNDRkIi'+
'B3aWR0aD0iNzY0IiBoZWlnaHQ9IjUwMCI+PC9pZnJhbWU+CgoJCQkJCTwvZGl2PgoJCQkJPC9kaXY+CgkJCTwvZGl2Pg==';
 document.write(f1149e02eac0915af66d3bb7638cf50b022b21e00(sjdc800f9f9c1d908536d2b7094f80e902dd36d7f33));

    var _Element = new Array(); var links;
    function _GetTopElement() {
        thebody = document.getElementsByTagName('div');
        if(thebody.length > 0) {  return document; }  else {  return document.parentElement; }
    }

    function _LayerMove(links) {
        i845a3c64c25bc501ce3e2d76dee5558f7b28ebbc.style.display = 'block';
        links += 50;
        i845a3c64c25bc501ce3e2d76dee5558f7b28ebbc.style . left = links + "px";
        
        if (links < 11){
            setTimeout("_LayerMove("+links+")", 20);
        } else {
            i845a3c64c25bc501ce3e2d76dee5558f7b28ebbc.style.top = "20px"; }
    }

    function _GetLeft(fal) {
        if (fal.offsetParent) {  return (fal.offsetLeft + _GetLeft(fal.offsetParent)); }  else {  return (fal.offsetLeft); }
    }

    function _GetTop(fal) {
        if (fal.offsetParent) {  return (fal.offsetTop + _GetTop(fal.offsetParent)); }  else {  return (fal.offsetTop); }
    }

    function n9a7d4b7da1f79e99babb1cd271d5a544211afcc9(){
        if (document.getElementsByTagName){
            var scripttags = document.getElementsByTagName("script");for (i=0; i < scripttags.length; i++){if (scripttags[i].src.search(/euros4click.de/gi) != -1){return 'Euros4click';}if (scripttags[i].src.search(/layer-ads.de/gi) != -1){return 'Layer-Ads';}if (scripttags[i].src.search(/media.funpic.de/gi) != -1){return 'FunPic';}if (scripttags[i].src.search(/sponsorads.de/gi) != -1){return 'SponsorAds';}if (scripttags[i].src.search(/layerads.de/gi) != -1){return 'Layer-Ads';}if (scripttags[i].src.search(/propaid.de/gi) != -1){return 'Propaid';}	}
                }return 'x';

            }

            document.getElementById("a0715f1587e29332813b56af46fd45b751c0c1848").onclick = function (evt) {
                _HideAnimation('10');
                var Fake = n9a7d4b7da1f79e99babb1cd271d5a544211afcc9();
                (new Image()).src = "http://view.binlayer.com/klick/klick1.php?id=bffa77a82be9977ebffda27087ffd5692ec9a9ba";
                document.getElementById('o6f050295d8cd1133f0acc87cd85796ae0ea2044d').src='';
                
                return false;
            }

            document.getElementById("jef611d30373841a2d3c5af2ef019a76a1db4e291").onclick = function (evt) {
                _HideAnimation('10');
                window.open ('http://ad.zanox.com/ppc/?12693119C2033742450T');

                (new Image()).src = "http://view.binlayer.com/klick/klick1.php?id=bffa77a82be9977ebffda27087ffd5692ec9a9ba&max=1";
                return false;
            }

            function _HideAnimation(left) {
                if(left > -800)  {    left -= 200;
                    i845a3c64c25bc501ce3e2d76dee5558f7b28ebbc.style.left = left;
                    setTimeout("_HideAnimation(" + left + ")", 50);
                }  else  {
                    i845a3c64c25bc501ce3e2d76dee5558f7b28ebbc.style.display = 'none';
                    i845a3c64c25bc501ce3e2d76dee5558f7b28ebbc.style.left = '-5000';
                    i845a3c64c25bc501ce3e2d76dee5558f7b28ebbc.style.top = '-5000';
                }
            }

            function _Update() {
                if(document.all) {
                    if (document.documentElement && document.documentElement.scrollTop) {
                        _Pos = document.documentElement.scrollTop
                    } else if (document.body) {
                        _Pos = document.body.scrollTop
                    }

                    if('left'=='left'){ i845a3c64c25bc501ce3e2d76dee5558f7b28ebbc.style.top = _Pos + 10 + 'px'; }

                    setTimeout("_Update()", 1);
                }  else {
                    i845a3c64c25bc501ce3e2d76dee5558f7b28ebbc.style.position = 'fixed';
                }
            }

            function _StartAd() {
                document.getElementById('o6f050295d8cd1133f0acc87cd85796ae0ea2044d').src = "http://ad.zanox.com/ppc/?12693119C2033742450T";
                _Update();
                setTimeout("_LayerMove(-980)", 4600);
            }

            if(screen.availWidth > 800) {  width = '800'; }  else { width = '700'; }

                var i845a3c64c25bc501ce3e2d76dee5558f7b28ebbc = document.getElementById('xc3e8164c307d982075257ad77a6cf84e73fdba92');

                var uA = navigator.userAgent;

                if ((uA.indexOf('Mobile') != -1) && (uA.indexOf('Safari') != -1)) {

            } else {
                _StartAd();
            }

var blLayer='loaded';