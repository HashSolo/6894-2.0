//$version: \include\js\lib\features\ads\ads.js@@\main\14 $
//<!--
// This is part of the ebay JS toolbox library
// All rights reserved.
//
// File:	 Ads.js
// Desc:	 Ads implementation, depends on toolbox
//
// Change Log:
// iiimmddyyn  nnnnn  Description
// ----------  -----  --------------------------------------------------------
// sp0310041          Initial creation
// sp0322041          Add vars for default values
// sp0409041          Add cats to search ad
// sp0618041   199568 JS errors on NS4.x
// jcc0618041         Add setCustomAdParam()
// jcc0701041  207377 Malformed page (downloading as .zip)
// sp1008041   240455 "dcopt=ist" added to ad tags
// jce093004		  Fixing writeAds to pick up categories on page if defined
//

//Default values

var defDARTSite = "general";
var defAdZone = "overview";
var defMaxCatLevels = 6;

var defMainAdWidth = 468;
var defMainAdHeight = 60;
var defSideAdWidth = 120;
var defSideAdHeight = 60;

//var doubleClickHost = "http://ebay.doubleclick.net/";
var ebayDoc;
var adload;

var ebAdRand = (new Date()).getTime();

//Generic Ad
function writeAd(pDARTSite, pZone, pTile, pW, pH, pIFW, pIFH, pCustomParams)
{
	if (pZone.length == 0)
		pZone = "home";
		
	var cfg1 = getStandardAdConfig("generic", pDARTSite, [pZone], pTile, pW, pH, pIFW, pIFH);
	setCustomAdParam(cfg1, pCustomParams);

	var h = getStandardAdHTML(cfg1);
	document.write(h);
//	alert("Generic Ad: \n\n" + h);
}

//Home page Ad
//Example: writeHomepageAd("signout", "default", 1, 275, 300)
function writeHomepageAd(pDARTSite, pZone, pTile, pW, pH, pIFW, pIFH)
{
	var cfg1 = getStandardAdConfig("homePage", pDARTSite, [pZone], pTile, pW, pH, pIFW, pIFH);

	var h = getStandardAdHTML(cfg1);
	document.write(h);
	//alert("Homepage Ad: \n\n" + h);
}

function getSearchAdConfig(pType,pTile,pW,pH,pDW,pDH)
{
	var cfg = getStandardAdConfig(pType, "search", ["keywords"], pTile, pW, pH, pDW, pDH);
	if (!setCatNumbers(cfg))
		cfg.addParam("cat", "0");
		
	return cfg;
}

//Search results page Ad
function writeSearchAd(pKeyword, pHideLinkAds, pCustomParams, pMainW, pMainH, pSide1W, pSide1H, pSide2W, pSide2H)
{//debugger;

	pMainW = pMainW || defMainAdWidth;
	pMainH = pMainH || defMainAdHeight;
	pSide1W = pSide1W || defSideAdWidth;
	pSide1H = pSide1H || defSideAdHeight;
	pSide2W = pSide2W || defSideAdWidth;
	pSide2H = pSide2H || defSideAdHeight;
		
	var cfg1 = getSearchAdConfig("searchLeft", 1, pMainW, pMainH);
	setCustomAdParam(cfg1, pCustomParams);
	var cfg2 = null, cfg3 = null;
	if (pHideLinkAds != true)
	{
		cfg2 = getSearchAdConfig("searchTop", 2, pSide1W, pSide1H, defSideAdWidth, defSideAdHeight);
		setCustomAdParam(cfg2, pCustomParams);
		cfg3 = getSearchAdConfig("searchBottom", 3, pSide2W, pSide2H, defSideAdWidth, defSideAdHeight);
		setCustomAdParam(cfg3, pCustomParams);
	}
	
	pKeyword = pKeyword || keyword;
		
	if (pKeyword && pKeyword.length > 0)
	{
		cfg1.addParam("kw", pKeyword);
		if (pHideLinkAds != true)
		{
			cfg2.addParam("kw", pKeyword);
			cfg3.addParam("kw", pKeyword);
		}
	}
	
	var h = getSearchListingAdHTML(cfg1, cfg2, cfg3);
	document.write(h);
	//alert("Search Ad: \n\n" + h);
}


function setCustomAdParam(pCfg, pParams)
{
	if (typeof(pParams) == "string")
		pParams = [pParams];
	if (pParams && pParams.length > 0)
	{
		for (var i = 0; i < pParams.length; i++)
		{
			if (typeof (pParams[i]) != 'undefined')
			{
   				if (pParams[i].indexOf("=") != -1)
   				{
   					var tmp = pParams[i].split("=");
   					pCfg.addParam(tmp[0], tmp[1]);
   				}
			}
		}
	}
}


function getListingAdConfig(pType,pDart,pAdZones,pTile,pWidth,pHeight)
{
	var cfg = getStandardAdConfig(pType, pDart, pAdZones, pTile, pWidth||defMainAdWidth, pHeight||defMainAdHeight);
	if (!setCatNumbers(cfg))
		cfg.addParam("cat", "0");
		
	return cfg;
}

//Listings page ad
function writeListingAd(pHideLinkAds, pCustomParams)
{//debugger;
	var DARTSite = defDARTSite;
	if (typeof(category_level0) != "undefined" && category_level0.length > 0 && category_level0 != "0")
		DARTSite = category_level0;
	else if (typeof(cat0_name) != "undefined" && cat0_name.length > 0)
		DARTSite = cat0_name;
	
	//Ebay motors
	if (DARTSite == "ebay_motors" || DARTSite.toLowerCase() == "ebay motors")
		DARTSite = "motors";

	
	DARTSite = DARTSite.substring(0, 23);

	var adZones = [defAdZone];
	//Limit only one Zone, pass the rest in params
	if (typeof(cat1_name) != "undefined" && cat1_name.length > 0)
		adZones = [cat1_name];
	
	var cfg1 = null, cfg2 = null, cfg3 = null;
	var cfg1 = getListingAdConfig("cfg1", DARTSite, adZones, null);
	setCustomAdParam(cfg1, pCustomParams);
		
	if (DARTSite != "6000" && DARTSite != "motors" && pHideLinkAds != true)
	{
		cfg2 = getListingAdConfig("cfg2", DARTSite, adZones, 2, defSideAdWidth, defSideAdHeight);			
		setCustomAdParam(cfg2, pCustomParams);
		cfg3 = getListingAdConfig("cfg3", DARTSite, adZones, 3, defSideAdWidth, defSideAdHeight);
		setCustomAdParam(cfg3, pCustomParams);
	}
	var h = getSearchListingAdHTML(cfg1, cfg2, cfg3);
	document.write(h);
	//alert("Listings Ad: \n\n" + h);
}

function getStandardAdConfig(pName, pDARTSite, pZones, pTile, pW, pH, pIW, pIH)
{
	var cfg = new EBayAdConfig(pName);
	cfg.tile = pTile;
	cfg.width = pW;
	cfg.height = pH;
	cfg.ifWidth = pIW || pW;
	cfg.ifHeight = pIH || pH;
	cfg.DARTSite = new String(pDARTSite);
	cfg.zones = pZones;
	setCatNumbers(cfg);
	if (typeof(ebSelCBObj) != "undefined" && ebSelCBObj != null)
	{
		var cb = ebSelCBObj.name.toLowerCase();
		if (cb != "ebay")
			cfg.addParam("!category", cb);
	}
	//BUGDB00240455 - Missing Key-Value from Doubleclick Ad Calls
	cfg.addParam("dcopt", "ist");
	return cfg;
}

function getStandardAdHTML(pCfg)
{//debugger;
	if (typeof(ebayDoc) == "undefined" || !ebayDoc)
		ebayDoc = new EbayDocument(window, "Ads");
	var ebayAd1 = new EbayAd(ebayDoc, ebayDoc, pCfg.name, pCfg, ebAdRand);
	ebayAd1.setCountryGlobals = setAdCountryGlobals;
	ebayAd1.setCountryLocals = setAdCountryLocals;
	
	return ebayAd1.toHTML();
}

function getSearchListingAd(pDoc, pType, pCfg, pRand)
{
	var ebayAd = new EbayAd(pDoc, pDoc, pType, pCfg, pRand);
	ebayAd.setCountryGlobals = setAdCountryGlobals;
	ebayAd.setCountryLocals = setAdCountryLocals;
	
	return ebayAd;
}

function getSearchListingAdHTML(pCfg1, pCfg2, pCfg3)
{
	if (typeof(ebayDoc) == "undefined" || !ebayDoc)
		ebayDoc = new EbayDocument(window, "Ads");

	var ebayAd1 = getSearchListingAd(ebayDoc, "main", pCfg1, ebAdRand);
	var ebayAd2, ebayAd3;
	if (pCfg2)
		ebayAd2 = getSearchListingAd(ebayDoc, "top", pCfg2, ebAdRand);
	if (pCfg3)
		ebayAd3 = getSearchListingAd(ebayDoc, "bottom", pCfg3, ebAdRand);

	var sIm = 'src="http://pics.ebaystatic.com/aw/pics/x.gif"';
	var h = '<table width="100%" border="0" cellspacing="0" cellpadding="5">';
	h += '<tr><td colspan="3"><img ' + sIm;
	h += ' width="100%" height="1" border="0"></td></tr>';

	h += '<tr><td align="left">' + ebayAd1.toHTML() + '</td>';
	if (pCfg2 && pCfg3)
	{
		h += '<td align="left"><table border="0" align="left" valign="center" cellspacing="5" cellpadding="0">';
		h += '<tr><td align="left">' + ebayAd2.toHTML() + '</td>';
		h += '<td align="left">' + ebayAd3.toHTML() + '</td></tr>';
		h += '</table></td>';
	}
	h += '<td align="left" width="100%"><img ' + sIm;
	h += ' width="1" height="1" border="0"></td></tr>';
	h += '</table>';

	return h;
}

function setCatNumbers(pCfg)
{
	var cs = true, isAdded = false;
	var aCats = new Array();
	if(typeof(defMaxCatLevels)!="undefined")
	{
	for (var i=0; i<defMaxCatLevels && (cs != "0"); i++)
	{
		if(eval("typeof(category_level"+i+")!=\"undefined\""))
		{	
			cs = eval("category_level"+i);
			aCats[i]=cs;
			isAdded = true;
		}
		else
		{
			aCats[i]="";
		}
	}
	pCfg.categories = aCats;
	}
			
	return isAdded;
}


//========================================================================================================
//SP - Backward compatibility=============Avoid using these functions=====================================
//========================================================================================================
function createAdStr(pW, pH, pTile, pIFWidth, pIFHeight)
{
	//Homepage goes here
	if (slOut.length > 0)
	{
		var tmp = slOut.split("/");
		if (tmp[0] == "motors" || tmp[0] == "ebaymotors")
			tmp[0] = "6000";
		writeAd(tmp[0]?tmp[0]:"", tmp[1]?tmp[1]:"", pTile, pW, pH, pIFWidth, pIFHeight, []);
	}
	return "";
}


function catAdTop()
{
	var h = '<tr><td colspan=2 align=center bgcolor=FFFFFF>'+createAdStr(120,60,1)+'<BR><BR></td></tr>';
	document.write(h);
	//alert(adStr);
}

	
function catAdBottom()
{
	var h = '<tr><td align=left bgcolor=FFFFFF>'+createAdStr(234,60,1)+'</td>';
	document.write(h);
}


function catAdLayer()
{
	//Default search/listings ads for NS 4.x
	if (typeof(dcAdLayerSrc) != "undefined")
	{
		var ls = '<LAYER SRC="' + dcAdLayerSrc[0] + '" width=120 height=60 visibility="hidden" ';
		ls += 'onLoad="moveToAbsolute(US_ad0_layer.pageX,US_ad0_layer.pageY);clip.height=60;clip.width=120;';
		ls += ' visibility=\'show\';"></LAYER>';
		
		if (dcAdLayerSrc[1])
		{
			ls += '<LAYER SRC="' + dcAdLayerSrc[1] + '" width=234 height=60 visibility="hidden" ';
			ls += 'onLoad="moveToAbsolute(US_ad1_layer.pageX,US_ad1_layer.pageY);clip.height=60;clip.width=234;';
			ls += ' visibility=\'show\';"></LAYER>';
		}
		document.write(ls);
	}
}


function catAdMotors()
{
	var adStr = '<table border="0" width="120" align="center" cellpadding="0" cellspacing="0">';
	adStr += '<tr><td nowrap valign="top" align="center" width="120">'+createAdStr(120,60,1);
	adStr += '</td></tr></table>';
	document.write(adStr);
	//alert(adStr);
}

if (typeof(catIndexCatName) != "undefined")
{
	slStr = "catindex";
	if (catIndexCatName == "")
		catIndexCatName = "general";

	//Motors has subcat in the page		
	slOut = catIndexCatName;
	if (catIndexCatName.indexOf("/") == -1)			//jad061703
		slOut += "/home";
		
	window.ad_top = catAdTop;
	window.ad_bottom = catAdBottom;
	window.write_adlayer = catAdLayer;
	window.write_ad = catAdMotors;
	adload = true;
}
else 
{
	// This was used by Search & Listings pages, 
	// Since it's going to be converted to use new Ads code, this is section is not required
	/*
	if(typeof(isHomepage) == "undefined" || !isHomepage)
	{
		//If it's not the homepage, For search and listings, write ads now. 
		if (stringContains(thisPageURL,"listings."))
			writeListingAd();
		else
			writeSearchAd(keyword);
	}
	*/
}

//----------For Reference, don't delete this section--------------
/*
 Antiques (#20081)  
 Art (#550)  
 Books (#267)  
 Business & Industrial (#12576)  
 Cameras & Photo (#625)  
 Clothing, Shoes & Accessories (#11450)  
 Coins (#11116)  
 Collectibles (#1)  
 Computers & Networking (#58058)  
 Consumer Electronics (#293)       
 Crafts (#14339)  
 Dolls & Bears (#237)  
 DVDs & Movies (#11232)  
 Entertainment Memorabilia (#45100)  
 Health & Beauty (#26395)  
 Home & Garden (#11700)  
 Jewelry & Watches (#281)  
 Music (#11233)  
 Musical Instruments (#619)  
 Pottery & Glass (#870)         
 Real Estate (#10542)  
 Specialty Services (#316)  
 Sporting Goods and Fan Shop (#382)  
 Sports Cards and Memorabilia (#64482)  
 Stamps (#260)  
 Tickets (#1305)  
 Toys & Hobbies (#220)  
 Travel (#3252)  
 Video Games (#1249)  
 Everything Else (#99) 
 Automobiles (#6000)
*/
//-----------------------------------------------------------
//-->
