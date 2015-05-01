
	function setSellImage(itemTable, itemKey, table, key) {

        url = '../_modules/media/imageManager/gateway.php?setSellImage&itemTable='+itemTable+'&itemKey='+itemKey+'&table='+table+'&key='+key;

        new Ajax(url, {
            method: 'get',
            update: $('notice')
        }).request();

	}


function deleteBeSure(URL){

    if (window.confirm("Are you sure you want to delete this?")) {
    		window.location.href = URL;
	  }
}

function confirmAction(URL, message){

    if (window.confirm(message)) {
    		window.location.href = URL;
	  }
}

function launchImage(theURL,Width,Height){
	  var Scrollbars = "no";
	  if(screen.height<Height){
	  var Height = (screen.height-200);
	  var Scrollbars = "yes";
	  }
	  
	  if(screen.width<Width){
	  var Width = (screen.width-200);
	  var Scrollbars = "yes";
	  }
	  else{
	  var Width = Width;
	  }
	  var Left = (screen.width-Width)/2;
	  var Top = (screen.height-Height)/2;
	  var params = ('toolbar=no,scrollbars=' + Scrollbars + ',resizable=no,left=' + Left +',top=' + Top +',width=' + Width + ',height=' + Height);
	  
	Pop = window.open(theURL, "LaunchImage", params);
	
	Pop.focus();
}
function launchWindow(theURL,Width,Height){
	  var Scrollbars = "no";
	  if(screen.height<Height){
	  var Height = (screen.height-200);
	  var Scrollbars = "yes";
	  }
	  
	  if(screen.width<Width){
	  var Width = (screen.width-200);
	  var Scrollbars = "yes";
	  }
	  else{
	  var Width = Width;
	  }
	  var Left = (screen.width-Width)/2;
	  var Top = (screen.height-Height)/2;
	  var params = ('toolbar=no,scrollbars=yes,menubar=yes,resizable=yes,left=' + Left +',top=' + Top +',width=' + Width + ',height=' + Height);
	  
	Pop = window.open(theURL, "LaunchImage", params);
	
	Pop.focus();
}
function constrainProportions(width, height, maxWidth, maxHeight) {
	
	var wprop = 0, hprop = 0;
	
	wprop = maxWidth / width;
	hprop = maxHeight / height;
	
	if (maxWidth > width) {
		if (wprop > hprop) {
			height = maxHeight;
			width = width * hprop;
		} else {
			width = maxWidth;
			height = height * wprop;
		}
	} else {
		if (wprop > hprop) {
			height = maxHeight;
			width = width * hprop;
		} else {
			width = maxWidth;
			height = height * wprop;
		}
	}
	var dims = [];
	
	dims['width'] = width;
	dims['height'] = height;
	
	return dims;
	
}


function setImageFromThumb(from, id, src, maxWidth, maxHeight) {
	
	var ti = $(id);
	
	ti.src = src;
	
	var dims = constrainProportions(from.width, from.height, maxWidth, maxHeight);
	
	ti.width = dims['width'];
	ti.height = dims['height'];
	
}
function refreshSelect(k){
	window.location = '?k='+k;
	
	
	}
