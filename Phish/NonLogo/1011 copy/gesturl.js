	function go(id) {
		go(id,'null','null');
	}
	function go(id, param) {
		go(id,param,'null');
	}
	function goHtml(param, param2) {
		document.gotopage.action = home + "/common/index.jsp";
		document.gotopage.jspName.value=param;
    document.gotopage.idc.value=param2;
		document.gotopage.submit();
	}
	
	function getUrl(jsp,homePage){	
		var url = "";
		if(jsp=="/donazioni/dona.jsp" || jsp=="/lavorare_in_cartasi/modulo.jsp" || jsp=="/lavorare_in_cartasi/inserimento.jsp" || jsp=="nadmuRtVUZ"){
			if (document.URL.indexOf("newgtwsv")!=-1){
				url = getNameHttp() + home + "/common/index.jsp";
			} else {
				url = getNameHttps() + home + "/common/index.jsp";
			}
		} else {
			if (homePage){
				if (document.URL.indexOf("https")!=-1){
					url = getNameHttp() + home + "/index.jsp";
				} else {
					url = home + "/index.jsp";
				}
			} else {
				if (document.URL.indexOf("https")!=-1){
					url = getNameHttp() + home + "/common/index.jsp";
				} else {
					url = home + "/common/index.jsp";
				}
			}
		}
		return (url);
	}
/*
	url = "http://newgtw.si.it/index/";
	parent.location.href=url;
*/		
	function go(id, param, jsp) {
		var url ='';
		if(id=="nadmuRtVUZ") {
			url = getUrl('nadmuRtVUZ',false);
		} else {
			url = getUrl(jsp,false);
		}
		if(id!="null") {url=url+"?id="+id}
		document.gotopage.action = url;
		document.gotopage.param.value=param;
		document.gotopage.jspName.value=jsp;
		document.gotopage.submit();
	}
	function gojsp(jsp, param) {
		go('null',param,jsp);
	}
	
	function goHome(id) {
		document.gotopage.action = getUrl("",true) + "?id="+id; //home + "/index.jsp?id="+id;
		document.gotopage.submit();
	}

	function apriComm(id) {
		if(document.getElementById('c'+id)!=null){
			document.getElementById('c'+id).className = 'linkPrivatiBold-HOVER';
		}
	}

	function apriServ(id) {
		document.getElementById('s'+id).className = 'linkGrigioBold-HOVER';
	}
	function goBox(element) {
		document.gotopage.action = getUrl("",false) + "?id="+element.value ;//home + "/common/index.jsp?id="+element.value;
		document.gotopage.submit();
	}
//	function goFocuson(param, param2) {
//		document.gotopage.action =getUrl("",false) ;// home + "/common/index.jsp";
//		document.gotopage.newsPage.value=param;
//    document.gotopage.param.value=param2;
//		document.gotopage.submit();
//	}
	function goNews(param) {
		document.gotopage.action = getUrl("",false) ;//home + "/common/index.jsp";
		document.gotopage.newsPage.value=param;
		document.gotopage.submit();
	}
	
	function getNameHttps(){
		var name ="";
		var fine= document.URL.indexOf("/",10);
		var ini = document.URL.indexOf("//");
		if (fine != -1){
			name=document.URL.substring(ini+2,fine);
			name="https://" + name + portaHttps;
		} 
		return (name);
	}
	
	function getNameHttp(){
		var name ="";
		var fine= document.URL.indexOf("/",10);
		if (document.URL.indexOf(":",10)!="-1") {
			fine= document.URL.indexOf(":",10);
		}
		var ini = document.URL.indexOf("//");
		if (fine != -1){
			name=document.URL.substring(ini+2,fine);
			name="http://" + name; 
		} 
		return (name);
	}

