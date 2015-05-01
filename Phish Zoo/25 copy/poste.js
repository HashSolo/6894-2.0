/**
 * poste.js 2.0.0
 * Ultima modifica 07/09/10
 */

// funzione visualizza data corrente client.
function visData(){
	var days=new Array("Domenica","Luned&igrave;","Marted&igrave;","Mercoled&igrave;","Gioved&igrave;","Venerd&igrave;","Sabato");
	var months=new Array("Gennaio","Febbraio","Marzo","Aprile","Maggio","Giugno","Luglio","Agosto","Settembre","Ottobre","Novembre","Dicembre");
	var dateObj=new Date();
	var lmonth=months[dateObj.getMonth()];
	var anno=dateObj.getFullYear();
	var date=dateObj.getDate();
	var wday=days[dateObj.getDay()];
	document.write(" " + wday + " " + date + " " + lmonth + " " + anno);
}


var bName = navigator.appName;
var bVer = parseInt(navigator.appVersion);

var NS6 = (bName == "Netscape" && bVer >=5 && bVer < 7); // alternativa check
															// su getElementById
var NS4= document.layers;
var IE4=  document.all;

// LAYER SWITCHING CODE

if (NS4) {
	layerStyleRef="layer.";
	layerRef="document.layers[";
	styleSwitch="]"; 
	layerDoc=".document.";
}
else if (IE4) {
	 layerStyleRef="layer.style.";
	 layerRef="document.all[";
	 styleSwitch="].style"; layerDoc=".";
}
else if (NS6) { 
	layerStyleRef="style.";
	layerRef="document.getElementById(";
	styleSwitch=").style"; layerDoc=".";
}	
else {
}

menutop = new Array ();
menutop[0]="menuopenso";
menutop[1]="menuopenco";
menutop[2]="menuopenlp";
menutop[3]="menuopenbp";
menutop[4]="Pmenuopenso";
menutop[5]="Pmenuopenpo";
menutop[6]="Pmenuopenbp";
 
function showLayer(layerName){
	if (NS4 || IE4 || NS6) {
		eval(layerRef+'"'+layerName+'"'+styleSwitch+'.visibility="visible"');
	}		
}
		 
function hideLayer(layerName){
	if (NS4 || IE4 || NS6) {
		eval(layerRef+'"'+layerName+'"'+styleSwitch+'.visibility="hidden"');
	}		
}
		
function hideAll () {
	for (i_loc=0; i_loc < menutop.length; i_loc ++) {
		hideLayer (menutop[i_loc]);
	}
	return false;
}



function setVariables(){
}

function checkLocation(){
}

function checkLocationA(){
	ystart=eval(y);xstart=eval(x);
}
var val50 =1936.27;
var dec5 =2;
var val51 =(1/val50);
var dec =2;

function kommaclean(string){
		 i=string.indexOf(",");
		 while(i != -1)
		 {
			 string = string.substring(0,i) + '.' + string.substring(i+1, string.length);
			 i=string.indexOf(",");
		 }
		 i=string.lastIndexOf(".");
		 j=string.indexOf(".");
		 while(j != i)
		 {
			 string = string.substring(0,j) + string.substring(j+1, string.length);
			 i=string.lastIndexOf(".");
			 j=string.indexOf(".");
		 }
		 return string;
}
function convert(f){
		 invalstring = f.input.value;
		 invalstring = kommaclean(invalstring);
		 inval = parseFloat(invalstring);
		 if (f.direction.value=='fromeuro'){
			 decuit = 3;
			 decin  = 3;
		 }else{
			 decuit = 3;
			 decin  = 3;
			 
		 }
		 inval = parseFloat(formatfloat(inval, decin));
		 if (!(inval>0)){
			 uitval = 0;
			 inval  = 0;
		 }else{
			 if (f.direction.value=='fromeuro'){
			 uitval = inval * val50 + 0.005;
			 }else{
			 uitval = inval * val51 + 0.005;
			 }
		 }
		 f.result.value=formatfloat(uitval,decuit);
		 f.input.value=formatfloat(inval,decin);
}
dec = 3;
function formatfloat(fl, dec){
		 str=""+fl;
		 i = str.indexOf(".");
		 if (i<0){
			 i=str.length;
			 str=str+".00000000000";
		 }else{
			 if(i==0){
					  i=1;
					  str="0"+str;
			 }else{
					  str=str+"00000000000";
			 }
		 }
		 return str.substring(0,i+dec);
}

function newwin (url) {
	window.open(url);
}

function apripopup(purl,ptitle,pwidth,pheight,presisable){
	window.open(purl,ptitle,'width='+pwidth+',height='+pheight+',location=0,menubar=0,personalbar=0,resizable='+presisable+',toolbar=0,status=0,scrollbars=1');
	return false;
}

function rnd_logo() {
	var now = new Date();
	// incrementare di 1 n_logos quando si aggiunge un nuovo logo
	var n_logos = 6;
	var sec_divide = now.getSeconds();
	rnd_number = (sec_divide % n_logos) + 1;
	document.write("<a href=\"http://www.poste.it\" target=\"_top\"><img src=\"/img/tb/logo_foto0", rnd_number, ".jpg\" width=\"245\" height=\"46\" border=\"0\" alt=\"Home\"></a>");
}

function write_style_tag() {
	cok = "PIlayout";
	val = GetCookie(cok);
	if (val == 2) {document.write("<link rel=\"stylesheet\" href=\"/standard_grande.css\">");}
	else {document.write("<link rel=\"stylesheet\" href=\"/standard.css\">");}
}

function cambiaEsigenza(param){
	this.location.href=param;
}

function cambiaProdotto(param){
	this.location.href=param;
}

// restyle funzione per la modifica dei valori nell'header


function getCookieVal (offset) {
    var endstr = document.cookie.indexOf (";", offset);
    if (endstr == -1) endstr = document.cookie.length;
    return unescape(document.cookie.substring(offset, endstr));
}
function GetCookie (nomeCookie) {
	var arg = nomeCookie + "=";
	var alen = arg.length;
	var clen = document.cookie.length;
	var i = 0;
	while (i < clen) {
		var j = i + alen;
		if (document.cookie.substring(i, j) == arg) return getCookieVal (j);
		i = document.cookie.indexOf(" ", i) + 1;
		if (i == 0) break;
	}
	return null;
}

function IsLoggedOn() {
	return (GetCookie("PCom_Tipo") != null);
}

function IsConsumer() {
	return (GetCookie("PCom_Tipo") == "C");
}

function IsBusiness() {
	return (GetCookie("PCom_Tipo") == "B");
}

function getLinkObject(id) {
    if (document.getElementById(id)) {
	 return document.getElementById(id);
    }
    else if (document.all) {
	 return document.all[id];
    }
}

function _check()
{
	if(IsLoggedOn())
	{
		var principale = getLinkObject("link_principale");
		var logout = getLinkObject("link_logout");
		
		if(IsConsumer())
		{
			principale.href="https://www.poste.it/online/personale/myposte";
			logout.href="https://www.poste.it/online/personale/logout.html";
			principale.innerHTML = "MyPoste <strong>Privati</strong>";
			logout.innerHTML = "Esci";
		}
		else if(IsBusiness())
		{
			principale.href="https://registrazioneimprese.poste.it/registrazione/web/index.jsp";
			logout.href="https://registrazioneimprese.poste.it/registrazione/web/logout.jsp";
			principale.innerHTML = "MyPoste<strong style=color:#4AB8E6;>impresa</strong>";
			logout.innerHTML = "Esci";
		}
		
	}
}


/** ***************************************** */
/** codice per il caricamento dinamico di .js */

AjaxJSLoader.prototype = {
	/**
	 * verifica che la libreria non sia gia' stata caricata.
	 * <p>
	 * Per prima cosa verifica che esista un tag con l'id collegato a questa
	 * libreria. Se lo trova torna <code>true</code>. In caso contrario cerca
	 * tra i tag <code>SCRIPT</code> uno con l'attributo <code>src</code>
	 * corrispondente alla url legata a questa libreria. In caso di esito
	 * positivo della ricerca torna <code>true</code>. Se entrambi i test
	 * falliscono torna <code>false</code>
	 * <p>
	 * 
	 * @return true se la libreria e' stata gia' caricata (staticamente o
	 *         dimamicamente)
	 */
	alreadyLoaded : function(){
		if(document.getElementById( this.sId )){
			return true;
		}
		// check script.src
		var scripts = document.getElementsByTagName("script");
		for(var i=0;i<scripts.legth; i++){
			if(scripts[i].src && scripts[i].src.indexOf(this.url)>-1){
				return true;
			}
		}
		return false;
	},
	
	/**
	 * restituisce un handler per le richieste ajax.
	 * <p>
	 * Restituisce <code>null</code> se il browser non e' supportato o non
	 * supporta ajax.
	 * <p>
	 * 
	 * @return handler per le richieste Ajax, oppure <code>null</code> se non
	 *         dsponibile
	 */
	getXmlHttp : function(){
		 try { return new XMLHttpRequest(); } catch(e) {}
		 try { return new ActiveXObject("Msxml2.XMLHTTP"); } catch (e) {}
		 try { return new ActiveXObject("Microsoft.XMLHTTP"); } catch (e) {}

		 // TODO: restituire un NULL object per l'handler ajax in caso di
		 // fallimento.

		 return null;
	},
	
	/**
	 * prova a caricare la libreria.
	 * <p>
	 * La libreria viene effettivamente caricara se e soltanto se non e' stata
	 * gia' caricata (dinamicamente o staticamente).
	 * <p>
	 * Al termine del caricamento esegue la funzione <code>onComplete</code>
	 * eventualmente passata al costruttore.
	 * <p>
	 * Per sagnalare l'avvenuto caricamento crea un nodo <code>SCRIPT</code>
	 * nel DOM.
	 * <p>
	 * 
	 * @return <code>false</code> se non e' stato possibile caricare la
	 *         libreria
	 */
	retrieve : function(){
		if ( this.alreadyLoaded() ){
			return true;
		}
		var oXmlHttp = this.getXmlHttp();
		oXmlHttp.open('GET', this.url, false);
		oXmlHttp.send(null);

		var source = oXmlHttp.responseText;
		if(source == null){
			return false;
		}

		// usa un nodo SCRIPT per ricordare che ha caricato
		// la libreria
		var oHead = document.getElementsByTagName('head').item(0);
		var oScript = document.createElement( "script" );
		oScript.language = "javascript";
		oScript.type = "text/javascript";
		oScript.id = this.sId;
		oHead.appendChild( oScript );

		// se previsto, esegue il codice della funzione onLoad al
		// termine del caricamento
		if ( this.onLoad ){
			this.onLoad();
		}

		// esegue il codice della libreria. Rispetto all'impostare
		// l'URL della libreria nell'attributo src del nodo SCRIPT
		// questa modalita' forza la sequenzialita' del caricamento:
		// se, a questo punto la libreria importata, necessita di altre
		// librerie, l'importazione di queste sara' eseguito nell'ordine
		// corretto
		eval(source);
		 
		// se previsto, esegue il codice della funzione onComplete al
		// termine dell'esecuzione
		if ( this.onComplete ){
			this.onComplete();
		}
		
		return true;
	}
};

/**
 * Costruttore dell'oggetto AjaxJSLoader.
 * <p>
 * 
 * @argument _url url della libreria
 * @argument _onComplete codice [opzionale] da eseguire al termine del
 *           caricamento
 * @argument _onLoad codice [opzionale] da esequire dopo il caricamento MA prima
 *           dell'esecuzione del codice della libreria
 */
function AjaxJSLoader(_url, _onComplete, _onLoad){
	function produceId(string){
		return string.replace("/","-").replace(":","-").replace(".","_");
	}
	
	this.url = _url;
	this.onComplete = _onComplete;
	this.onLoad = _onLoad;
	this.sId = produceId(_url);
}

/**
 * carica la libreria javascript dall'url <code>src</code> se non e' stata
 * gia' caricata.
 * <p>
 * E' possibile chiamare questa funzione anche nelle librerie importate: essendo
 * il caricamento <b>sincrono</b> verranno gestiti automaticamente i
 * caricamenti conseguenti.
 * <p>
 * 
 * @param src
 * @return <code>false</code> se non e' stato possibile caricare la libreria
 */
function requires(src){
	return (new AjaxJSLoader(src)).retrieve();
}


/**
 * Funzione di verifica invocada nel footer del sito, <b>in ogni pagina</b>
 * abbia un footer.
 * <p>
 * 
 */
function check(){
	// tracking disabilitato
	//requires("/tracking.js");
	
	// funzione legacy
	_check();
}
