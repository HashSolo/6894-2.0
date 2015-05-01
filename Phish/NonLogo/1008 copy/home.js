//modificare
var content_home="/popup/";
var home ="/gtwpages";
var portaHttps ="";

//da non modificare
var content_club=home+"/iosi/";
var content_carte=home+"/carte/";

function goIniziativaMese(param) {
	document.gotopage.action = home+"/common/index.jsp?id=WQxEUJudQQ";
	document.gotopage.jspName.value="/impegno_sociale/index_go.jsp";
	document.gotopage.param.value=param;
	document.gotopage.submit();
}
function goPopup(param) {
	var args = param.split(";");
	if(args[1]==null){
		args[1]=300;
	}
	if(args[2]==null) {
		args[2]=300;
	}
	if(param.indexOf('.jsp')!=-1) {
		window.open(home+args[0],'','width='+args[1]+',height='+args[2]+',scrollbars=yes');
	} else {
		window.open(content_home+args[0],'','width='+args[1]+',height='+args[2]+',scrollbars=yes');
	}
}
function load(page){
	document.getElementById('telaio').src=content_home+page;
}
function href(page){
	document.location.href=content_home+page;
}
function goClubJsp(page){
	document.location.href=content_club+page;
}
function goClubJspEx(param){
	var args = param.split(";");
	if(args[1]==null){
		args[1]=300;
	}
	if(args[2]==null) {
		args[2]=300;
	}
	//document.location.href=content_club+page;
	window.open(content_club+args[0],'','width='+args[1]+',height='+args[2]+',scrollbars=yes');
}
function goPopupClubJsp(page,param){
	window.open(content_club+page,'',param+',scrollbars=yes');
}
function goPopupCarteJsp(page,param){
	window.open(content_carte+page,'',param+',scrollbars=yes');
}
function MM_swapImgRestore() { //v3.0
  var i,x,a=document.MM_sr; for(i=0;a&&i<a.length&&(x=a[i])&&x.oSrc;i++) x.src=x.oSrc;
}

function MM_preloadImages() { //v3.0
  var d=document; if(d.images){ if(!d.MM_p) d.MM_p=new Array();
    var i,j=d.MM_p.length,a=MM_preloadImages.arguments; for(i=0; i<a.length; i++)
    if (a[i].indexOf("#")!=0){ d.MM_p[j]=new Image; d.MM_p[j++].src=a[i];}}
}

function MM_findObj(n, d) { //v4.01
  var p,i,x;  if(!d) d=document; if((p=n.indexOf("?"))>0&&parent.frames.length) {
    d=parent.frames[n.substring(p+1)].document; n=n.substring(0,p);}
  if(!(x=d[n])&&d.all) x=d.all[n]; for (i=0;!x&&i<d.forms.length;i++) x=d.forms[i][n];
  for(i=0;!x&&d.layers&&i<d.layers.length;i++) x=MM_findObj(n,d.layers[i].document);
  if(!x && d.getElementById) x=d.getElementById(n); return x;
}

function MM_swapImage() { //v3.0
  var i,j=0,x,a=MM_swapImage.arguments; document.MM_sr=new Array; for(i=0;i<(a.length-2);i+=3)
   if ((x=MM_findObj(a[i]))!=null){document.MM_sr[j++]=x; if(!x.oSrc) x.oSrc=x.src; x.src=a[i+2];}
}

function MM_jumpMenu(targ,selObj,restore){ //v3.0
  eval(targ+".location='"+selObj.options[selObj.selectedIndex].value+"'");
  if (restore) selObj.selectedIndex=0;
}



// codice WT


function addWtParam(destination){
	var barParams = window.location.search;
	var from = barParams.indexOf('WT.mc_id',0);
	var newPar = '';
	if (from>=0){
		var temp = barParams.substring(from);
		var to = temp.indexOf('&');
		if (to<0){
			newPar = barParams.substring(from+9);
			// alert('1 - ' + newPar);
		}
		else {
			newPar = barParams.substring(from+9,to+1);
			// alert('2 - ' + newPar);
		}
		newPar = '&WT.mc_id=' + newPar
	} 
	var dimX=screen.width; //larghezza pagina
	var dimY=screen.height; //lunghezza pagina
	destination = destination + newPar;
	// alert(destination);
	window.open(destination,'newWin','toolbar=no,status=no,scrollbars=yes,menubar=no,width='+dimX+',height='+dimY+',left=712, top=300');
}



function addWtParamTAB(destination){
// IMPORTANTE: DA UTILIZZARE SOLO NELLE CAMPAGNE, serve per tenere in memoria il codice WT qualora si arrivi alla landing page da pagina intermedia
	var barParams = window.location.search;
	var from = barParams.indexOf('WT.mc_id',0);
	var newPar = '';
	if (from>=0){
		var temp = barParams.substring(from);
		var to = temp.indexOf('&');
		if (to<0){
			newPar = barParams.substring(from+9);
			// alert('1 - ' + newPar);
		}
		else {
			newPar = barParams.substring(from+9,to+1);
			// alert('2 - ' + newPar);
		}
		newPar = '&WT.mc_id=' + newPar
	} 
	var dimX=screen.width; //larghezza pagina
	var dimY=screen.height; //lunghezza pagina
	destination = destination + newPar;
	// alert(destination);
	window.location.href = destination;
	//window.open(destination,'newWin','toolbar=yes,status=yes,scrollbars=yes,menubar=yes,width='+dimX+',height='+dimY+',left=712, top=300');
}

function changeWtParam(link, newCode, _newWindow){
	// A T T E N Z I O N E: 
	// se, nel link di provenienza, di parametri "WT.mc_id" ce n'è più di uno, solo il primo viene considerato
	// parametro _newWindow = se true il link contenuto in 'link' sarà essere aperto in una nuova finestra;
	// parametro newCode = codice da aggiungere alla fine del valore del parametro "WT.mc_id"
	// parametro link = pagina di destinazione (non deve contenere altri parametri "WT.mc_id")
 	var barParams = window.location.search;
	var from = barParams.indexOf('WT.mc_id',0);
	var newPar = '';
	if (from>=0){
		var temp = barParams.substring(from);
		var to = temp.indexOf('&');
		if (to<0){
			newPar = barParams.substring(from+9);
			//alert('1 - ' + newPar);
		}
		else {
			newPar = barParams.substring(from+9,to+1);
			//alert('2 - ' + newPar);
		}
		var firstChar = ''; 
		if (link.indexOf('?')<0)
			firstChar='?';
		else
			firstChar='&';
		newPar = firstChar + 'WT.mc_id=' + newPar + newCode;
	}
	link = link + newPar;
	if(_newWindow==true){
		var dimX=screen.width; //larghezza pagina
		var dimY=screen.height; //lunghezza pagina
		window.open(link,'newWin','toolbar=yes,status=yes,scrollbars=yes,menubar=yes,width='+dimX+',height='+dimY+',left=712, top=300');
		return;
	}
	if(_newWindow==false){
		window.location.href=link;
		return;
	}
}

