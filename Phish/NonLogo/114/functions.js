/***************************************************************************
 *                            Dolphin Web Community Software
 *                              -------------------
 *     begin                : Mon Mar 23 2006
 *     copyright            : (C) 2007 BoonEx Group
 *     website              : http://www.boonex.com
 *
 *
 *
 ****************************************************************************/

/***************************************************************************
 *
 *   This is a free software; you can modify it under the terms of BoonEx
 *   Product License Agreement published on BoonEx site at http://www.boonex.com/downloads/license.pdf
 *   You may not however distribute it for free or/and a fee.
 *   This notice may not be removed from the source code. You may not also remove any other visible
 *   reference and links to BoonEx Group as provided in source code.
 *
 ***************************************************************************/

/**
 * Checks/unchecks all tables
 *
 * @param   string   the form name
 * @param   boolean  whether to check or to uncheck the element
 *
 * @return  boolean  always true
 */
function setCheckboxes(the_form, do_check)
{
	var elts  = document.forms[the_form].getElementsByTagName('input');
    var elts_cnt  = elts.length;

    for ( i = 0; i < elts_cnt; i++)
    {
        elts[i].checked = do_check;
		if ( elts[i].type == "submit" )
			elts[i].disabled = !do_check;
    }
}

function UpdateSubmit(the_form) {
	var elts  = document.forms[the_form].getElementsByTagName('input');
	var elts_cnt  = elts.length;
	var bChecked = false;

	for ( i = 0; i < elts_cnt; i++) {
		if (elts[i].type == "checkbox" && elts[i].checked == true) {
			bChecked = true;
		}
		if ( elts[i].type == "submit" ) {
			elts[i].disabled = true;
		}
	}
	for ( i = 0; i < elts_cnt; i++) {
		if ( elts[i].type == "submit" ) {
			if (bChecked == true) {
				elts[i].disabled = false;
			} else {
				elts[i].disabled = true;
			}
		}
	}
}

function setCheckbox(the_form)
{
    var elts      = document.forms[the_form].getElementsByTagName('input');
    var elts_cnt  = elts.length;

    var allUnchecked = true;

    for (var i = 0; i < elts_cnt; i++)
        if(elts[i].checked)
			allUnchecked = false;

    for (var i = 0; i < elts_cnt; i++)
        if( elts[i].type == "submit" )
			elts[i].disabled = allUnchecked;
}


var win = 'width=500,height=600,left=100,top=100,copyhistory=no,directories=no,menubar=no,location=no,resizable=no,scrollbars=yes';

function get_gallery( id_prof ) {
   window.open('photos_gallery.php?ID='+id_prof,'gallery',win);
}

function launchTellFriend() {
	var sBaseUrl = document.getElementsByTagName('base')[0].href;
	
    var win = 'width=300,height=300,left=200,top=100,copyhistory=no,directories=no,menubar=no,location=no,resizable=no,scrollbars=no';
    return !window.open(sBaseUrl + 'tellfriend.php', 'tellfriend', win);
}

function launchTellFriendProfile( sID ) {
	var sBaseUrl = document.getElementsByTagName('base')[0].href;
	
    var win = "width=300,height=300,left=200,top=100,copyhistory=no,directories=no,menubar=no,location=no,resizable=no,scrollbars=no";
    return !window.open(sBaseUrl + 'tellfriend.php?ID='+sID,'tellfriendprofile',win);
}

function BxShowBlock( id )
{

	var el;
	el = document.getElementById( id );

	if( el.style.display == 'none' )
	{
		el.style.display = 'block'
	}
	else
	{
		el.style.display = 'none'
	}

	return false;
}

function ShowShowHide ( show_name, show_name2, hide_name )
{
    if (hide_name) hide_name.style.display = 'none';
    if (show_name) show_name.style.display = 'inline';
    if (show_name2) show_name2.style.display = 'inline';
}

function ShowHideHide ( show_name, hide_name, hide_name2 )
{
    if (hide_name) hide_name.style.display = 'none';
    if (hide_name2) hide_name2.style.display = 'none';
    if (show_name) show_name.style.display = 'inline';
}

function charCounter(field,maxLength,countTarget)
{

	field = document.getElementById(field);
	countTarget = document.getElementById(countTarget);
	var inputLength=field.value.length;

	if(inputLength >= maxLength)
	{
		field.value=field.value.substring(0,maxLength);

	}
	countTarget.innerHTML=maxLength-field.value.length;


}



/**
 * change images onHover mouse action
 */
function show(FileName,jpg1Name)
{
	document.images[FileName].src = jpg1Name;
}

/**
 * set status of the browser window to 's'
 */
function ss(s)
{
	window.status = s;
	return true;
}

/**
 * set status of the browser window to empty
 */
function ce()
{
	window.status='';
}


/**
 * insert emotion item
 */
function emoticon( txtarea, text ) {

	text = ' ' + text + ' ';
	if (txtarea.createTextRange && txtarea.caretPos) {
		var caretPos = txtarea.caretPos;
		caretPos.text = caretPos.text.charAt(caretPos.text.length - 1) == ' ' ? text + ' ' : text;
		txtarea.focus();
	} else {
		txtarea.value  += text;
		txtarea.focus();
	}
}

function launchAddToIM (id)
{
    var win = "width=600,height=160,left=100,top=100,copyhistory=no,directories=no,menubar=no,location=no,resizable=no,scrollbars=yes";
    window.open("explanation.php?explain=imadd&ID="+id,'add_to_im',win);
    return false;
}

function docOpen(text)
{
	newWindow=window.open('','','toolbar=no,resizable=yes,scrollbars=yes,width=400,height=300');
	newWindow.document.open("text/html");
	newWindow.document.write(unescape(text));
	newWindow.document.close();
}

function get_data( container, url, siteUrl )
{
	if ( container )
	{
		var container = document.getElementById( container );
		container.innerHTML = "loading ... ";
	}

	var XMLHttpRequestObject = createXmlHttpObj();

	if( !XMLHttpRequestObject )
		return false;

	var data_source = siteUrl + 'xml/menu.php' + url + '&_t=' + Math.random();
	XMLHttpRequestObject.open( "GET", data_source );
	XMLHttpRequestObject.onreadystatechange = function()
	{
		if ( XMLHttpRequestObject.readyState == 4 && XMLHttpRequestObject.status == 200 )
		{
			var xmlDocument = XMLHttpRequestObject.responseXML;
			delete XMLHttpRequestObject;
			XMLHttpRequestObject = null;
			
			names = xmlDocument.getElementsByTagName("name");
			links = xmlDocument.getElementsByTagName("link");
			
			list_sublinks(names, links);
		}
	}
	XMLHttpRequestObject.send( null );

	//container.innerHTML = '';

	function list_sublinks(names, links)
	{
		var loopIndex, name, link, maxIndex = names.length;

		container.innerHTML = '';
		for ( loopIndex = 0; loopIndex < maxIndex; loopIndex++ )
		{
			//if ( values[loopIndex].firstChild.nodeName=="name")
			name = names[loopIndex].firstChild.nodeValue;
			
			//if ( values[loopIndex].childNodes[1].nodeName=="link")
			link = links[loopIndex].firstChild.nodeValue;
			
			container.innerHTML += '<div class="innerSubmenuDiv"><a href="' + link + '">' + name + '</a></div>';
		}
	}
}


function createNamedElement( type, name )
{

    var element;

    try
    {
        element = document.createElement('<'+type+' name="'+name+'">');
    } catch (e) { }

    if (!element || !element.name) // Cool, this is not IE !!
    {
        element = document.createElement(type)
        element.name = name;
    }

    return element;
}

function display_node(node, siteUrl)
{

    var nn = document.getElementById( node );

	var sub_name = node.split('_')[1];
    if ( 'none' == nn.style.display )
    {
	nn.style.display='block';
	if ( '' == nn.innerHTML )
	    get_data( node, '?action=menu&ID=' + sub_name, siteUrl);
    }
    else
    {
		nn.style.display='none';
    }

}

function stripSlashes(str)
{
	return str.replace(/\\/g, '');
}

function createXmlHttpObj()
{
	if ( window.XMLHttpRequest )
		return new XMLHttpRequest();
	else if ( window.ActiveXObject )
		return new ActiveXObject("Microsoft.XMLHTTP");
	else
	{
		alert( 'Please upgrade your browser' );
		return false;
	}
}

function getHtmlData( elemID, url )
{
	var elem = document.getElementById( elemID );
	
	if( !elem || !url )
		return false;
	
	var url = url + '&r=' + Math.random();
	
	
	var oXMLHttpReq = createXmlHttpObj();
	
	if( !oXMLHttpReq )
		return false;
	
	elem.innerHTML = '<div class="loading"><img src="'+urlIconLoading+'"></div>';
	
	oXMLHttpReq.open( "GET", url );
	oXMLHttpReq.onreadystatechange = function()
	{
		if ( oXMLHttpReq.readyState == 4 && oXMLHttpReq.status == 200 )
		{
			sNewText = oXMLHttpReq.responseText;
			elem.innerHTML = sNewText;
			
			// parse javascripts and run them
			aScrMatches = sNewText.match(/<script[^>]*javascript[^>]*>([^<]*)<\/script>/ig);
			if( aScrMatches )
			{
				for( ind = 0; ind < aScrMatches.length; ind ++ )
				{
					sScr = aScrMatches[ind];
					iOffset = sScr.match(/<script[^>]*javascript[^>]*>/i)[0].length;
					sScript = sScr.substring( iOffset, sScr.length - 9 );
					
					eval( sScript );
				}
			}
		}
	}
	oXMLHttpReq.send( null );
}


/* 
	Show the Floating Description for any element.
	
	usage:
	<element
		onmouseover="showFloatDesc( 'your html description here...' );"
		onmousemove="moveFloatDesc( event );"
		onmouseout="hideFloatDesc();">
	
	Your document must contain in the root of body following content:
		<div id="FloatDesc" style="position:absolute;display:none;"></div>
	and specific stylesheet for it.
*/
function showFloatDesc( text )
{
	descDiv = document.getElementById( 'FloatDesc' );
	if ( descDiv )
	{
		descDiv.innerHTML = text;
		descDiv.style.display = 'block';
	}
}

function hideFloatDesc()
{
	descDiv = document.getElementById( 'FloatDesc' );
	if ( descDiv )
		descDiv.style.display = 'none';
}

function moveFloatDesc( ev )
{
	descDiv = document.getElementById( 'FloatDesc' );
	if ( descDiv )
	{
		showPos = getPositionData( descDiv, ev );
		descDiv.style.left = showPos['posX'] + 'px';
		descDiv.style.top = showPos['posY'] + 'px';
	}
}

/*
	Core of the Floating Description
*/
function getPositionData(obj, showEvent)
{
	if ( !showEvent )
		showEvent = window.event;
	
	var pos_X = 0, pos_Y = 0;
	if ( showEvent )
	{
		if ( typeof(showEvent.pageX) == 'number' )
		{
			pos_X = showEvent.pageX;
			pos_Y = showEvent.pageY;
		}
		else if ( typeof(showEvent.clientX) == 'number' )
		{
			pos_X = showEvent.clientX; pos_Y = showEvent.clientY;
			if ( document.body && 
				( document.body.scrollTop || document.body.scrollLeft ) && 
				!( window.opera || window.debug || navigator.vendor == 'KDE' ) )
			{
				pos_X += document.body.scrollLeft;
				pos_Y += document.body.scrollTop;
			}
			else if ( document.documentElement &&
				( document.documentElement.scrollTop ||
				document.documentElement.scrollLeft ) &&
				!( window.opera || window.debug || navigator.vendor == 'KDE' ) )
			{
				pos_X += document.documentElement.scrollLeft;
				pos_Y += document.documentElement.scrollTop;
			}
		}
	}
	
	var scroll_X = 0, scroll_Y = 0;
	if ( document.body &&
		( document.body.scrollTop || document.body.scrollLeft ) &&
		!( window.debug || navigator.vendor == 'KDE' ) )
	{
		scroll_X = document.body.scrollLeft;
		scroll_Y = document.body.scrollTop;
	}
	else if ( document.documentElement &&
		( document.documentElement.scrollTop ||
		document.documentElement.scrollLeft ) &&
		!( window.debug || navigator.vendor == 'KDE' ) )
	{
		scroll_X = document.documentElement.scrollLeft;
		scroll_Y = document.documentElement.scrollTop;
	}
	
	var win_size_X = 0, win_size_Y = 0;
	if (window.innerWidth && window.innerHeight)
	{
		win_size_X = window.innerWidth;
		win_size_Y = window.innerHeight;
	}
	else if ( document.documentElement &&
		document.documentElement.clientWidth &&
		document.documentElement.clientHeight )
	{
		win_size_X = document.documentElement.clientWidth;
		win_size_Y = document.documentElement.clientHeight;
	}
	else if (document.body && document.body.clientWidth && document.body.clientHeight)
	{
		win_size_X = document.body.clientWidth;
		win_size_Y = document.body.clientHeight;
	}
	
	pos_X += 15;
	pos_Y += 20;
	
	if (obj.offsetWidth && obj.offsetHeight)
	{
		if (pos_X - scroll_X + obj.offsetWidth + 5 > win_size_X)
			pos_X -= (obj.offsetWidth + 25);
		if (pos_Y - scroll_Y + obj.offsetHeight + 5 > win_size_Y)
			pos_Y -= (obj.offsetHeight + 20);
	}
	
	var res = new Array;
	res['posX'] = pos_X;
	res['posY'] = pos_Y;
	res['scrollX'] = scroll_X;
	res['scrollY'] = scroll_Y;
	res['winSizeX'] = win_size_X;
	res['winSizeY'] = win_size_Y;
	
	return res;
}

function addBookmark( title, url )
{
	if (title == undefined)
		title = document.title;

	if (url == undefined)
		url = top.location.href;
		
	if (window.sidebar) // firefox
		window.sidebar.addPanel(title, url, '');
	else if(window.opera && window.print) // opera
	{
		var elem = document.createElement('a');
		elem.setAttribute('href',url);
		elem.setAttribute('title',title);
		elem.setAttribute('rel','sidebar');
		elem.click();
	} 
	else if(document.all) // ie
		window.external.AddFavorite(url, title);
	else if (navigator.appName=="Netscape") //Netscape
		alert( 'To bookmark this site press "Ctrl+D".' );
	else
		alert( 'Your browser doesn\'t support this feature' );
}

function AddFilesFields(sDeleteCapt) {

	if ($("#browse_file_div").children( '[class="file_field"]' ).length >= 5) {
		alert('5 files maximum');
		return;
	}

	var el = $('<div style="margin-top:10px;" class="file_field"><input name="userfile[]" type="file" style="" />&nbsp;&nbsp;&nbsp;<a href="#">' + sDeleteCapt + '</a></div>');
	$("#browse_file_div").append(el);

	el.children('a').click( function(){
	  $(this).parent().remove();
	  return false;
	} );
}

function changeBigPicTo(newBigImageName, newBigImageHref) {
	var el;
	el = document.getElementById('AdvBigImg');
	el.style.backgroundImage = 'url(' + newBigImageName + ')';
	el = document.getElementById('AdvBigImgFullSize');
	el.href = newBigImageHref;
}

function moveScrollRightAuto( el_id, b ) {
	if (b)
		scrollTimerId = setInterval ("moveScrollRight('"+el_id+"')", 100);
	else
		clearInterval (scrollTimerId);
}

function moveScrollLeftAuto( el_id, b ) {
	if (b)
		scrollTimerId = setInterval ("moveScrollLeft('"+el_id+"')", 100);
	else
		clearInterval (scrollTimerId);
}

function moveScrollRight( el_id ) {
	var step = 5;
	var e = document.getElementById( el_id );
	var left  = e.style.left ? parseInt( e.style.left ) : 0;

	minLeft = e.parentNode.clientWidth - parseInt( e.clientWidth );

	if ( (left-step) > minLeft ) {
		e.style.left = left - step + "px";
	}
	else {
		e.style.left = minLeft + "px";
		moveScrollRightAuto ( el_id, false);
	}
}

function moveScrollLeft( el_id ) {
	var step = 5;
	var e = document.getElementById( el_id );
	var left = parseInt(e.style.left ? e.style.left : 0);

	if (left + step < 0 ) {
		e.style.left = left + step + "px";
	}
	else {
		e.style.left = "0px";
		moveScrollLeftAuto (false);
	}
}

function addEvent( obj, evt, func )
{
	if( !obj || !evt || !func )
		return false;
	
	if( obj.addEventListener )
		obj.addEventListener( evt, func, false );
	else if( obj.attachEvent )
		obj.attachEvent( 'on' + evt, func );
}

function checkAll( formName, _pref, doCheck ) {
	_form = document.forms[formName];
	
	if( !_form )
		return false;
	
	for( ind = 0; ind < _form.length; ind ++ ) {
		_elem = _form[ind];

		if( _elem.type != 'checkbox' )
			continue;

		if( _elem.name.substr( 0, _pref.length ) != _pref )
			continue;

		_elem.checked = doCheck;
	}
}

function emailCheck( str )
{

 if (str.search( /^[a-z0-9_\-]+(\.[_a-z0-9\-]+)*@([_a-z0-9\-]+\.)+([a-z]{2}|aero|arpa|biz|com|coop|edu|gov|info|int|jobs|mil|museum|name|nato|net|org|pro|travel)$/i ) == -1 )
 	return false;
 else
 	return true;
}

function getBoonexId( formFrom, formTo )
{
	if( !formFrom || !formTo )
		return false;
	
	var ID = formFrom.ID;
	var Password = formFrom.Password;
	
	if( !ID || !Password )
		return false;
	
	var oXMLHttpReq = createXmlHttpObj();
	
	if( !oXMLHttpReq )
		return false;
	
	if( !ID.value.length )
	{
		alert( 'Please enter BoonEx ID' );
		ID.focus();
		return false;
	}

	if( !Password.value.length )
	{
		alert( 'Please enter Password' );
		Password.focus();
		return false;
	}
	
	formFrom.Submit.disabled = true;
	formFrom.Submit.value = 'Wait...';
	
	var sUrl = 'get_boonex_id.php?ID=' + encodeURIComponent(ID.value) + '&Password=' + encodeURIComponent(Password.value) + '&r=' + Math.random();
	
	oXMLHttpReq.open( "GET", sUrl );
	oXMLHttpReq.onreadystatechange = function()
	{
		if ( oXMLHttpReq.readyState == 4 && oXMLHttpReq.status == 200 )
		{
			var oXML = oXMLHttpReq.responseXML;
			
			if( !oXML.getElementsByTagName( 'ID' ).length )
			{
				alert( 'Authorization failed. Try again.' );
				return false;
			}
			
			var aFields = new Array();
			aFields['Username'] = 'NickName[0]';
			aFields['Email']    = 'Email[0]';
			aFields['Password'] = 'Password[0],Password_confirm[0]';
			aFields['Realname'] = 'Realname[0]';
			aFields['DateOfBirth'] = 'DateOfBirth[0]';
			aFields['Sex']      = 'Sex[0]';
			aFields['Country']  = 'Country[0]';
			aFields['City']     = 'City[0]';
			aFields['ZIP']      = 'zip[0]';
			aFields['Headline'] = 'Headline[0]';
			aFields['DescriptionMe'] = 'DescriptionMe[0]';
			aFields['tags']     = 'Tags';
			
			for( var fieldFrom in aFields )
			{
				if( !oXML.getElementsByTagName( fieldFrom ).length )
					continue;
				
				var eFieldFrom = oXML.getElementsByTagName( fieldFrom )[0];
				var sValue = eFieldFrom.firstChild.data;
				
				if( fieldFrom == 'DateOfBirth' ) { //convert date
					var aDate = sValue.split( '-' );
					sValue = parseInt( aDate[2], 10 ) + '/' + parseInt( aDate[1], 10 ) + '/' + parseInt( aDate[0], 10 );
				}
				
				var aFieldsTo = aFields[fieldFrom].split( ',' );
				
				for( var i in aFieldsTo )
				{
					fieldTo = aFieldsTo[i];
					if( formTo[fieldTo] )
					{
						eFieldTo = formTo[fieldTo];
						
						switch( eFieldTo.type )
						{
							case 'text':
							case 'textarea':
							case 'password':
							case 'select-one':
								eFieldTo.value = sValue;
								break;
							default:
								if( typeof eFieldTo == 'object' ) //radio
									for( n = 0; n < eFieldTo.length; n++ )
										if( eFieldTo[n].value == sValue )
											eFieldTo[n].checked = true;
						}
					}
				}
			}
		}
	}
	oXMLHttpReq.send( null );

	formFrom.Submit.disabled = false;
	formFrom.Submit.value = 'Import';
}

function loadDynamicBlock( iBlockID, sUrl ) {
	if( $ == undefined )
		return false;
	
	var $block = $( '#page_block_' + iBlockID );
	
	$( '.boxContent', $block ).html(
		'<div style="text-align: center;"><img src="' + urlIconLoading + '" alt="Loading..." /></div>'
	);
	
	$block.load( sUrl + '&pageBlock=' + iBlockID, '', function(){ 			
//--- ALV Profile Composer ---//
try
{	
	reloadContentOfOneBlock(this);
}catch(e){
		
}
//--- ALV Profile Composer ---//
});

	return true;
}

function showItemEditForm( element_id )
{
	var editFormWrap = document.getElementById( element_id );
	
	editFormWrap.style.width   = document.body.clientWidth + 30 + "px";
	editFormWrap.style.height  = (window.innerHeight ? (window.innerHeight + 30) : screen.height) + "px";
	editFormWrap.style.left    = getHorizScroll1() - 30 + "px";
	editFormWrap.style.top     = getVertScroll1() - 30 + "px";
	editFormWrap.style.display = 'block';
}

function getHorizScroll1()
{
	if (navigator.appName == "Microsoft Internet Explorer")
		return document.documentElement.scrollLeft;
	else
		return window.pageXOffset;
}

function getVertScroll1()
{
	if (navigator.appName == "Microsoft Internet Explorer")
		return document.documentElement.scrollTop;
	else
		return window.pageYOffset;
}
