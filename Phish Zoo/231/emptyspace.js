function windowWidth() {
  var myWidth = 0, myHeight = 0;
  if( typeof( window.innerWidth ) == 'number' ) {
    //Non-IE
    myWidth = window.innerWidth;
    myHeight = window.innerHeight;
  } else if( document.documentElement && ( document.documentElement.clientWidth || document.documentElement.clientHeight ) ) {
    //IE 6+ in 'standards compliant mode'
    myWidth = document.documentElement.clientWidth;
    myHeight = document.documentElement.clientHeight;
  } else if( document.body && ( document.body.clientWidth || document.body.clientHeight ) ) {
    //IE 4 compatible
    myWidth = document.body.clientWidth;
    myHeight = document.body.clientHeight;
  }
  return myWidth;
}

function windowHeight() {
  var myWidth = 0, myHeight = 0;
  if( typeof( window.innerWidth ) == 'number' ) {
    //Non-IE
    myWidth = window.innerWidth;
    myHeight = window.innerHeight;
  } else if( document.documentElement && ( document.documentElement.clientWidth || document.documentElement.clientHeight ) ) {
    //IE 6+ in 'standards compliant mode'
    myWidth = document.documentElement.clientWidth;
    myHeight = document.documentElement.clientHeight;
  } else if( document.body && ( document.body.clientWidth || document.body.clientHeight ) ) {
    //IE 4 compatible
    myWidth = document.body.clientWidth;
    myHeight = document.body.clientHeight;
  }
  return myHeight;
}

$(document).ready(function(){
	
	var exited = false;
	var container = $("div:first").width();
	var space = windowWidth() - container;
	space = space / 2;
	$("div:last").after("<div id='hiddenadL' style='position:fixed; float:left; top:0; left:0; margin:0; padding:0; width:" + space + "px; height:" + windowHeight() + "px'></div>");
	$("div:last").after("<div id='hiddenadR' style='position:fixed; float:right; top:0; right:0; margin:0; padding:0; width:" + space + "px; height:" + windowHeight() + "px'></div>");
	$("div:last").after("<div id='hiddenadAC' style='position:fixed; margin:0; padding:0; display:none; border:black solid thin; width:160px; height:620px'><img src='http://cdn.x10hosting.com/ads/images/1251776734_fileclose.png' style='float:right; padding-right:2px; padding-top:2px; width:16px; height:16px' alt='Close' border='0' /></div>");
	
	var adfirstrun = true;

	$("#hiddenadL").mouseenter(function(e){
		if(exited == false){
			if(adfirstrun == true){
				$("#hiddenadA iframe").appendTo("#hiddenadAC");
				adfirstrun = false;
			}
			$("#hiddenadAC").appendTo("#hiddenadL");
			var x = e.pageX - this.offsetLeft;
			var y = e.pageY - this.offsetTop;
			y = y - 33;
			if(y + 624 > windowHeight()){
				y = windowHeight() - 624;
			}
			x = x - 160;
			$("#hiddenadAC").css({'top' : y + 'px', 'left' : x + 'px', 'right' : '0'});
			$("#hiddenadAC iframe").css({'padding-top' : '20px'});
			$("#hiddenadAC").fadeIn("slow");
			$("#hiddenadAC iframe").fadeIn("slow");
		}
	});
	
	
	$("#hiddenadL, #hiddenadR").mouseleave(function(){
		$("#hiddenadAC").fadeOut("slow");
		$("#hiddenadAC iframe").fadeOut("slow");
	});
	
	$("#hiddenadAC").click(function(e){
		var x = e.pageX - this.offsetLeft;
		var y = e.pageY - this.offsetTop;
		if(y > 2 && y < 19 && x > 141 && x < 159){
			exited = true;
			$("#hiddenadAC").fadeOut("slow");
			$("#hiddenadAC iframe").fadeOut("slow");
		}
	});
	
	$("#hiddenadR").mouseenter(function(e){
		if(exited == false){
			if(adfirstrun == true){
				$("#hiddenadA iframe").appendTo("#hiddenadAC");
				adfirstrun = false;
			}
			$("#hiddenadAC").appendTo("#hiddenadR");
			var x = e.pageX;
			var y = e.pageY - this.offsetTop;
			y = y - 33;
			if(y + 624 > windowHeight()){
				y = windowHeight() - 624;
			}
			$("#hiddenadAC").css({'top' : y + 'px', 'left' : x + 'px'});
			$("#hiddenadAC iframe").css({'padding-top' : '20px'});
			$("#hiddenadAC").fadeIn("slow");
			$("#hiddenadAC iframe").fadeIn("slow");
		}
	});
	
});
