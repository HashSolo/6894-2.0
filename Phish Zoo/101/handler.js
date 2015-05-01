var TA = new Array();
var nr=0;
var handles = new Array();
var columns = new Array();
var counters = new Array();
var haspackages = false;
handlesvisible=0;











/**
Written By John Bakker

Feel free to change,fry and/or eat .. just be sure to drop me a line :)

Usage:

  _tT.define('template1','Hello {name}');
  alert(_tT.parse('template1',{name:'john'}));



**/

function wordWrap(string, m, b, c){
    var i, j, s, r = string.split("\n");
    if(m > 0) for(i in r){
        for(s = r[i], r[i] = ""; s.length > m;
            j = c ? m : (j = s.substr(0, m).match(/\S*$/)).input.length - j[0].length
            || m,
            r[i] += s.substr(0, j) + ((s = s.substr(j)).length ? b : "")
        );
        r[i] += s;
    }
    return r.join("\n");
}
/**
Creates the container.. you can have multiple templates easily and can reference them by name.
**/
function tTemplatesContainer()
{
	this.templates = new Array();
}
/**
	you define a template here
**/
tTemplatesContainer.prototype.define = function(name,template)
{
	newtemplate = this.templates.length;
	this.templates[newtemplate]= new tTemplate(name,template);
}
tTemplatesContainer.prototype.parse = function(name,vals)
{
	empty ='';
	if(typeof vals!='object')
	{
		returndata ='';
	}
	else
	{
		lu = this.lookup(name);
		if(lu>-1)
		{
			
			returndata = this.templates[lu].parse(vals);
	
		}
		else
		{
			returndata = '';
		}
	}
	
	return returndata;
	
	
}
tTemplatesContainer.prototype.lookup = function(name)
{
	for(tT=0;tT<this.templates.length;tT++)
	{
		//now it reads throught the templates 
		try
		{
			//just in case
			if(this.templates[tT].name==name)
			{
				return tT;
			}
		}
		catch(e)
		{
			//whoops :)
		}
	}
	
	return -1;
}

function tTemplate(name,content)
{
	this.name    = name;
	this.content = content;
}

tTemplate.prototype.parse = function(vals)
{
	output = this.content;
	for (var i in vals)
	{
		cl = "/{"+i+"}/g";
		eval("output = output.replace("+cl+",vals[i])");
		
		
	}
	return output;
}
 
  _tT = new tTemplatesContainer('_tT');
  









function registercolumn(column)
{
	newpos = columns.length;
	columns[newpos]= new Array();
	columns[newpos].columnname= column;

}
function registerhandle(handle)
{
	newpos = handles.length;
	handles[newpos]= new Array();
	handles[newpos].handlename= handle;
	document.getElementById(handle).style.display='none';
}
function togglehandles()
{
	if(handlesvisible==0)
	{
		for(uu=0;uu<columns.length;uu++)
		{
			document.getElementById(columns[uu].columnname).style.border='1px dashed gray';
		}
		for(uu=0;uu<handles.length;uu++)
		{
			document.getElementById(handles[uu].handlename).style.display='';
		}
		handlesvisible=1;
	}
	else
	{
		for(uu=0;uu<columns.length;uu++)
		{
			document.getElementById(columns[uu].columnname).style.border='';
		}
		for(uu=0;uu<handles.length;uu++)
		{
			document.getElementById(handles[uu].handlename).style.display='none';
		}
		handlesvisible=0;
	}

}

function menumouseover(item,color)
{
	document.getElementById('menuitem'+item+'Container').className='menuboxover';
	document.getElementById('menuitem'+item+'tl').src='http://layoutcdn4.mijndomein.nl/Layout/corners/images/corner_3_'+color+'_tl.png';
	document.getElementById('menuitem'+item+'tr').src='http://layoutcdn4.mijndomein.nl/Layout/corners/images/corner_3_'+color+'_tr.png';
	document.getElementById('menuitem'+item+'bl').src='http://layoutcdn4.mijndomein.nl/Layout/corners/images/corner_3_'+color+'_bl.png';
	document.getElementById('menuitem'+item+'br').src='http://layoutcdn4.mijndomein.nl/Layout/corners/images/corner_3_'+color+'_br.png';

}

function menumouseout(item,color)
{
	color='ffffff';
	document.getElementById('menuitem'+item+'Container').className='menuboxout';
	document.getElementById('menuitem'+item+'tl').src='http://layoutcdn4.mijndomein.nl/Layout/corners/images/corner_3_'+color+'_tl.png';
	document.getElementById('menuitem'+item+'tr').src='http://layoutcdn4.mijndomein.nl/Layout/corners/images/corner_3_'+color+'_tr.png';
	document.getElementById('menuitem'+item+'bl').src='http://layoutcdn4.mijndomein.nl/Layout/corners/images/corner_3_'+color+'_bl.png';
	document.getElementById('menuitem'+item+'br').src='http://layoutcdn4.mijndomein.nl/Layout/corners/images/corner_3_'+color+'_br.png';
}

function defaultval(val,type,item)
{

	
	if(type=='click')
	{
		if(item.value==val)
		{
			item.value='';
		}
		else
		{

		}
	}
	else if(type=='clickpwd')
	{
		if(item.value==val)
		{
			item.value='';
			//item.type='password';
		}
		else
		{

		}
	}
	else if(type=='changepwd')
	{
		if(item.value=='')
		{

			item.value=val;
			//item.type='text'
		}
		else
		{

		}
	}
	else
	{

		if(item.value=='')
		{

			item.value=val;
		}
		else
		{

		}
	}

}
function registercounterupdate(itemid,type)
{
	newpos = counters.length;

	counters[newpos]=new Array();
	counters[newpos].itemid=itemid;
	counters[newpos].itemtype=type;
}
function updatecounters()
{
	if(counters.length>0)
	{
		nr++;
		TA[nr] = new TAjax();
		TA[nr].cn = 'TA['+nr+']';

		TA[nr].Sourcefile= "/teller";

		TA[nr].onReadyresponsecommand = 'updatecountersfinish(getresponse('+nr+'))';
		TA[nr].doRequest();

	}
}
function putcomma(numbert)
{
	if(numbert.length>3)
	{
		newstr = '';
		commas = Math.floor(numbert.length/3);
		start = numbert.length-(commas*3);
		for(oo=0;oo<commas;oo++)
		{
			if(oo>0)
			{
				newstr +='.';
			}
			newstr += numbert.substr(start+(oo*3),3)
		}
		if(start>0)
		{
			newstr = numbert.substr(0,start)+'.'+newstr;


		}
		return newstr;
	}
	return numbert;



}
function updatecountersfinish(msg)
{

	msglist = msg.split(',');
	totaldomains = putcomma(msglist[0]);
	todaydomains = putcomma(msglist[1]);
	totalusers = putcomma(msglist[2]);
	todayusers = putcomma(msglist[3]);
	//todaytransfers = putcomma(msglist[4]);

	for(hh=0;hh<counters.length;hh++)
	{
		if(counters[hh].itemtype=='totaldomains')
		{
			document.getElementById(counters[hh].itemid).innerHTML=totaldomains;
		}
		else if(counters[hh].itemtype=='todaydomains')
		{
			document.getElementById(counters[hh].itemid).innerHTML=todaydomains;
		}
		else if(counters[hh].itemtype=='totalusers')
		{
			document.getElementById(counters[hh].itemid).innerHTML=totalusers;
		}
		else if(counters[hh].itemtype=='todayusers')
		{
			document.getElementById(counters[hh].itemid).innerHTML=todayusers;
		}
		/**
		else if(counters[hh].itemtype=='todaytransfers')
		{
			document.getElementById(counters[hh].itemid).innerHTML=todaytransfers;
		}
		*/
	}

	setTimeout('updatecounters();',30000);
}

function startonload()
{
	nr=0;
	setTimeout('updatecounters();',30000);
	
}


function formvalchange(frm,nm,item,prep)
{
	//alert(prep+'->'+item.value);
	document.getElementById(frm+nm+'value').innerHTML=prep+' '+item.value;
}

function calculatetotal(prep,extensionlist)
{
	var exts = extensionlist.split(',');
	var total = 0;
	for (var j=0; j<exts.length; j++)
	{
		try
		{
			price = document.getElementById(exts[j]+'value').innerHTML.substring(2)*1;
			total += price;

		}
		catch(err)
		{

		}
	}
	if (total > 0) {
		haspackages = true;
	}
	else {
		haspackages = false;
	}
	document.getElementById('totalprice').innerHTML=prep+' '+total.toFixed(2);
}
function hasPackage(text) {
	if (haspackages) {
		return true;
	}
	else {
		alert(text);
		return false;
	}
}


function feedbacksubmit()
{
	document.getElementById('feedback_title1').style.display="none";
	document.getElementById('feedback_title2').style.display="block";
	document.getElementById('feedback_title3').style.display="none";
	document.getElementById('feedbackform').style.display='none';
	//now send it !





	var ftitle =document.getElementById('feedback_title').value;
	var fdescr =document.getElementById('feedback_descr').value;
	var ftopic = document.getElementById('feedback_topic').value;
	if((fdescr==''))
	{
		document.getElementById('feedback_title1').style.display="none";
		document.getElementById('feedback_title2').style.display="none";
		document.getElementById('feedback_title3').style.display="block";
		document.getElementById('feedbackform').style.display='block';
		return false;
	}



	doctosend = 'title='+encodeURIComponent(ftitle);
	doctosend +='&fdescr='+encodeURIComponent(fdescr);
	doctosend +='&ftopic='+encodeURIComponent(ftopic);
	doctosend +='&screeninfo='+encodeURIComponent(screen.width+'x'+screen.height+'x'+screen.colorDepth);

	doctosend += '&URL='+window.location;









	nr++;
	TA[nr] = new TAjax();
	TA[nr].cn = 'TA['+nr+']';

	TA[nr].Sourcefile= "/sendfeedback";
	TA[nr].doctosend= doctosend;

	TA[nr].onReadyresponsecommand = 'feedbackrecieved(getresponse('+nr+'))';
	TA[nr].doPost();



}
function feedbackrecieved()
{
	setTimeout('feedbackhidethanks()',4000);
}
function feedbackhidethanks()
{
	document.getElementById('feedback_title1').style.display="block";
	document.getElementById('feedback_title2').style.display="none";
	document.getElementById('feedback_title3').style.display="none";


}

function voiceyouropinion()
{
	if(document.getElementById('feedbackform').style.display!='block')
	{
		document.getElementById('feedbackform').style.display='block';
	}
	else
	{
		document.getElementById('feedbackform').style.display='none';
	}

}
function extraextensions() {
	document.getElementById('extraextensions').className = 'displaytable';
	document.getElementById('extralink').className = 'displaynone';
}
function domainlistSubmit(interfaced, text) {
	form = document.getElementById('domainlistform');
	if (interfaced) {
		doc = new Array();
		var i = 0;
		for(var id in form.elements)	{
			val = form.elements[id].value;
			if (val) {
				doc[i] = form.elements[id].name+'='+val;
				i++;
			}
		}
		if (hasPackage(text)) {
			puntapi.DoCommand2(basepath+'/frontpage/step2', doc.join('&'));
			//return false;
		}
		else {
			return false;
		}
	}
	else {
		if (hasPackage(text)) {
			form.submit();
		}
		else {
			return false;
		}

	}
}


function texts()
{
	this.texts=new Object;	
}
texts.prototype.define = function(tag,value)
{
	var newtag = tag.replace(/_/,'');
	this.texts[newtag]=value;
	
}
texts.prototype.get = function(tag)
{
	var newtag = tag.replace(/_/,'');
	
	if(typeof this.texts[newtag]!='undefined')
	{
		return this.texts[newtag];
	}
	else
	{
		return tag;
	}
	
}
_texts = new texts();




			function md3fpadddiscount()
			{
				var code = document.getElementById('discount').value;
				if(code)
				{
					tmpdoc = 'discountcode='+code;
					_tajax.makeCall('/updatecart',{method:'post',onFinish:function(resp){md3fpbuildcartdata(resp);},doc:tmpdoc,weight:1});
				}				
			}
			function md3fpremovediscount(code)
			{
				if(code)
				{
					tmpdoc = 'removediscount='+code;
					_tajax.makeCall('/updatecart',{method:'post',onFinish:function(resp){md3fpbuildcartdata(resp);},doc:tmpdoc,weight:1});
				}				
			}			
			function md3fpremovecartitem(domain)
			{
				var tmpdoc='removedomain='+encodeURIComponent(domain)
				_tajax.makeCall('/updatecart',{method:'post',onFinish:function(resp){md3fpbuildcartdata(resp);},doc:tmpdoc,weight:1});
				
			}
			function md3fpbuildcartdata(response)
			{
				var tmplocation = window.location+'';
				if(tmplocation.indexOf('/basket')>-1)
				{
					location.reload('/basket');
					return false;
				}
								
				try
				{
					//fp_inittemplates();
					var step='/step1';

					var xml = response.xml;
	
					var result = xml.getElementsByTagName('cartupdate').item(0);
					var items = result.getElementsByTagName('cartitem');
					var topcarttext = '';
					var itemcount = 0;
					var labelpackages = new Array();
					labelpackages[1] = _texts.get('frontpage_totalpackage');
					labelpackages[2] = _texts.get('frontpage_parkingpackage');
					labelpackages[3] = _texts.get('frontpage_emailpackage'); 	
					var totalpricetext = _texts.get('frontpage_totalprice');
					var topcarttext1 = _texts.get('frontpage_items_in_shoppingcart');
					var topcarttext2 = _texts.get('frontpage_item_in_shoppingcart');
					var incvattext = _texts.get('frontpage_inc_vat');
					var totaltext1 = _texts.get('frontpage_total');
					
					if(items.length)
					{
						var total = 0;
						var rowclr = 'a';
						
						var contentblock = _tT.parse('cartcontenttop',{});
						
						var productOverview = _tT.parse('productoverviewtop',{});
						
						
								

						
						for (var i=0 ; i < items.length ; i++)
						{
							if(rowclr == 'a')
							{
								rowclr = 'b';
							}
							else
							{
								rowclr = 'a';
							}

							var item = items.item(i);
							
							if (item.getAttribute('type') == 'discount')
							{
								if(rowclr == 'a')
								{
									rowclr = 'b';
								}
								else
								{
									rowclr = 'a';
								}
								
								var disctext = item.getElementsByTagName('discounttext')[0].firstChild.nodeValue;
								var disc = item.getElementsByTagName('discount')[0].firstChild.nodeValue;
								var disccode = item.getElementsByTagName('discountcode')[0].firstChild.nodeValue;
								
								total -= parseFloat(disc);
								
								contentblock += _tT.parse('cartcontentdiscount',{	row:rowclr
																				,	discounttext:disctext
																				,	discount:disc
																				,	discountcode:disccode
																				});
								
								if (step == 'step4')
								{
									productOverview += _tT.parse('productoverviewitemdiscount',{discounttext:disctext,discount:disc});
								}
							}
							else
							{
								var domainname = item.getElementsByTagName('domain')[0].firstChild.nodeValue;
								var fulldomainname = item.getElementsByTagName('fulldomain')[0].firstChild.nodeValue;

								var price = item.getElementsByTagName('price')[0].firstChild.nodeValue;
								total += parseFloat(price);
								itemcount += 1;
							
								var packagename = item.getAttribute('name');
								var packagetype = item.getAttribute('type');
								//md3fprecievepackageupdate(domainname,packagetype);
								if(item.getAttribute('transfer') == 'yes')
								{
									contentblock += _tT.parse('cartcontenttransfer',{	row:rowclr
																					,	domain:domainname
																					,	fulldomain:fulldomainname
																					,	packagename:packagename
																					,	price:'&euro;'+price
																					});
								}
								else
								{
									contentblock += _tT.parse('cartcontent',{	row:rowclr
																			,	domain:domainname
																			,	fulldomain:fulldomainname
																			,	packagename:packagename
																			,	price:'&euro;'+price
																			});
								}
								
								if (step == 'step4')
								{
									packagename = labelpackages[item.getAttribute('type')];
									productOverview += _tT.parse('productoverviewitem',{domain:domainname,packagename:packagename,price:price});
								}
							}														
						}
						
						if(i > 1)
						{
							topcarttext = topcarttext1;
						}
						else
						{
							topcarttext = topcarttext2;
						}
						if (document.getElementById('shoppingcartitems'))
						{
							document.getElementById('shoppingcartitems').innerHTML = _tT.parse('topcartitems'
																							  ,{message:topcarttext
																							  ,	itemcount:itemcount
																							  });
						}
						
						
						var totalcheck = new RegExp('[\.]');
						if (totalcheck.test(total))
						{
							total = total + '0';
						}
						else
						{
							total = total + '.00';
						}
							
						contentblock += _tT.parse('cartcontentbottom',{totaltext:totalpricetext+'('+incvattext+')'
																	  ,	totalprice:'&euro;'+total
																	  });
						
						if (step == 'step4')
						{
							productOverview += _tT.parse('productoverviewbottom',{	totaltext:totaltext1
																				 ,	incvat:incvattext
																				 ,	total:total
																				 });
							
							document.getElementById('products_overview').innerHTML = productOverview;
						}

						document.getElementById('cartcontent').innerHTML = contentblock;

					}
					else
					{
						
						document.getElementById('shoppingcartitems').innerHTML = '&#160;';
						document.getElementById('cartcontent').innerHTML = _tT.parse('cartcontentempty',{message:_texts.get('frontpage_noshoppingcartitems')});
					}		
								
				}
				catch(e)
				{
					/*
					alert(response.text);
					alert(e);
					*/
					
				}
				
				
				if(tmplocation.indexOf('/step1')>-1)
				{
					md3fpdomainlistRecievefullUpdate(response);
				}
				if(tmplocation.indexOf('/step4')>-1)
				{
					location.reload('/step4');
				}
				

							
			}




/** 
Templates

**/

_tT.define('pricecolorblock','<table cellspacing="0" cellpadding="0" width="50px"><tr><td style="padding:0px;width:5px;"><div class="PBC_{packagename}_left">&#160;</div></td><td style="height:12px;font-size:12px;padding:0px;line-height:12px;font-weight:bold;text-align:right;width:40px;" class="md{packagename}_bg md{fontcolor}">{price}</td><td style="padding:0px;width:5px;"><div class="PBC_{packagename}_right">&#160;</div></td></tr></table>');
_tT.define('cartcontentempty','{message}');
_tT.define('cartcontenttop','<form method="post" action="/basket" onsubmit="return md3domainchecksend(\'express\');" name="md3basketform" id="md3basketform"><input name="do" value="updateproducts" type="hidden"><div class="list"><table width="100%" cellpadding="0" cellspacing="0"><tbody><tr><td><table width="100%" cellpadding="0" cellspacing="0"><tbody><tr><td width="5"><img src="/Layout/corners/images/corner_5_aeaeae_tl.png" width="5" height="5"></td><td class="firstline" width="100%"><img src="/Layout/Mijndomein/blank.gif" width="100%" height="5"></td><td width="5"><img src="/Layout/corners/images/corner_5_aeaeae_tr.png" width="5" height="5"></td></tr></tbody></table></td></tr><tr><td><table width="100%" cellpadding="0" cellspacing="0"><tbody>');
_tT.define('cartcontentbottom','<tr><td colspan="4" style="background-color: rgb(174, 174, 174);"><img src="/Layout/Mijndomein/blank.gif" width="100%" height="1"></td></tr><tr class="row_a"><td id="_total" class="basket_total_small" align="right" colspan="2">{totaltext}: {totalprice}</td><td></td></tr></tbody></table></td></tr><tr><td class="lastline"><table width="100%" cellpadding="0" cellspacing="0"><tbody><tr><td align="right"><table style="padding-right: 8px;" cellpadding="0" cellspacing="0"><tbody><tr></tr></tbody></table></td></tr></tbody></table></td></tr><tr><td><table width="100%" cellpadding="0" cellspacing="0"><tbody><tr><td width="5"><img src="/Layout/corners/images/corner_5_eeeeee_bl.png" width="5" height="5"></td><td class="lastline"><img src="/Layout/Mijndomein/blank.gif" width="100%" height="5"></td><td width="5"><img src="/Layout/corners/images/corner_5_eeeeee_br.png" width="5" height="5"></td></tr></tbody></table></td></tr></tbody></table></div></form>');

_tT.define('cartcontent','<tr class="row_{row}"><td width="200">{domain}</td><td align="right" width="40"><table width="50" cellpadding="0" cellspacing="0"><tbody><tr><td style="padding: 0px; width: 5px;"><div class="PBC_{packagename}_left">&nbsp;</div></td><td style="padding: 0px; height: 12px; font-size: 12px; line-height: 12px; font-weight: bold; color: white; text-align: right; width: 40px;" class="md{packagename}_bg">{price}</td><td style="padding: 0px; width: 5px;"><div class="PBC_{packagename}_right">&nbsp;</div></td></tr></tbody></table></td><td></td></tr>');
_tT.define('cartcontenttransfer','<tr class="row_{row}" width="100"><td>{domain}</td><td align="right" width="40"><table width="50" cellpadding="0" cellspacing="0"><tbody><tr><td style="padding: 0px; width: 5px;"><div class="PBC_{packagename}_left">&nbsp;</div></td><td style="padding: 0px; height: 12px; font-size: 12px; line-height: 12px; font-weight: bold; color: white; text-align: right; width: 40px;" class="md{packagename}_bg">{price}</td><td style="padding: 0px; width: 5px;"><div class="PBC_{packagename}_right">&nbsp;</div></td></tr></tbody></table></td><td></td></tr><tr class="row_{row}"><td>Verhuizingskosten</td><td align="right" width="40"><table width="50" cellpadding="0" cellspacing="0"><tbody><tr><td style="padding: 0px; width: 5px;"><div class="PBC_yellow_left">&nbsp;</div></td><td style="padding: 0px; height: 12px; font-size: 12px; line-height: 12px; font-weight: bold;" class="mdlightyellow_bg mddarkgray" align="right">&#8364;0,00</td><td style="padding: 0px; width: 5px;"><div class="PBC_yellow_right">&nbsp;</div></td></tr></tbody></table></td><td align="right" width="15"></td></tr>');
_tT.define('cartcontentdiscount','<tr class="row_{row}"  width="100"><td>{discounttext}</td><td align="right" width="40"><table width="50" cellpadding="0" cellspacing="0"><tbody><tr><td style="padding: 0px; width: 5px;"><div class="PBC_discount_left">&nbsp;</div></td><td style="padding: 0px; height: 12px; font-size: 12px; line-height: 12px; font-weight: bold;" class="mddiscount_bg mddarkgray" align="right">-&#8364;{discount}</td><td style="padding: 0px; width: 5px;"><div class="PBC_discount_right">&nbsp;</div></td></tr></tbody></table></td><td></td>');

_tT.define('topcartitems','<table class="clickable" onclick="location.href=\'/basket\';" cellpadding="0" cellspacing="0"><tbody><tr><td><div class="ICON4_basket" style="display: inline-block; height: 16px;">&nbsp;</div></td><td>{itemcount}&nbsp;{message}</td></tr></tbody></table>');
_tT.define('productoverviewtop','<table width="90%" cellpadding="0" cellspacing="0"><tbody>');
_tT.define('productoverviewitem','<tr><td width="50%">{domain}</td><td align="right" width="15%" nowrap="nowrap">&#8364;{price}</td><td width="5%">&nbsp;</td><td width="30%" nowrap="nowrap">{packagename}</td></tr>');
_tT.define('productoverviewitemdiscount','<tr><td width="50%">{discounttext}</td><td align="right" width="15%" nowrap="nowrap">-&#8364;{discount}</td><td width="5%">&nbsp;</td><td width="30%" nowrap="nowrap"></td></tr>');
_tT.define('productoverviewbottom','<tr><td class="overview_total" align="right">{totaltext}:&nbsp;</td><td class="overview_total" align="right" nowrap="nowrap">&#8364;{total}<br><div style="font-size: 10px;">{incvat}</div></td><td colspan="2">&nbsp;</td></tr><tr><td colspan="4">&nbsp;</td></tr></tbody></table>');
