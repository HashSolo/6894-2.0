var temp = { v2:false, v3:false, v4:false, v5:false, v6:false, v7:false, v8:false, v9:false  };
function FlashDetectorFactory()
{
	var versions = { v2:false, v3:false, v4:false, v5:false, v6:false, v7:false, v8:false, v9:false  };
	var n = navigator;
	var closure =
	{
		installed : function()
		{
			var ok = false;
			for (v in versions) { ok = ok || versions[v]; }
			return(ok);
		},
		atLeast : function(want)
		{
			var ok = false;
			for (v in versions) { ok = ok || ((((v.search(/(\d+(\.\d+)?)/) != -1) ? RegExp.$1 : 0) >= want) && versions[v]); }
			return(ok);
		},
		renderMovie : function(args)
		{
			String.prototype.interpolate=function(){var s=this;while(s.search(/\$\{([^\}]+)\}/)!=-1){s=s.replace(/\$\{([^\}]+)\}/, eval(RegExp.$1));}return(s);} 
			document.writeln(("<object classid='CLSID:D27CDB6E-AE6D-11CF-96B8-444553540000'"+
			" codebase='http://download.macromedia.com/pub/shockwave/cabs/flash/swflash.cab#version=7,0,19,0'"+
			" id='${args.name}' width='${args.width||100}' height='${args.height||100}' align='middle' alt='${args.alt||''}'>"+
			"  <param name='allowScriptAccess' value='sameDomain'/>"+
			"  <param name='movie' value='${args.path}'/>"+
			"  <param name='play' value='${args.play||'true'}'/>"+
			"  <param name='quality' value='high'/>"+
			"  <param name='loop' value='${args.loop||'false'}'/>"+
			"  <param name='longdesc' value='${args.longdesc||''}'/>"+
			"<embed src='${args.path}' quality='high'"+
			" swLiveConnect='false' scale='default' width='${args.width||100}' height='${args.height||100}'"+
			" loop='${args.loop||'false'}' play='${args.play||'true'}' title='${args.alt||''}' name='${args.name}' longdec='${args.longdesc||''}' align='middle'"+
			" allowScriptAccess='sameDomain' type='application/x-shockwave-flash'"+
			" pluginspage='http://www.macromedia.com/go/getflashplayer' />"+
			"</object>").interpolate());
		},

		renderAlternative : function(args)
		{
			String.prototype.interpolate=function(){var s=this;while(s.search(/\$\{([^\}]+)\}/)!=-1){s=s.replace(/\$\{([^\}]+)\}/, eval(RegExp.$1));}return(s);} 
							
			document.writeln("<p class='${args.cssclass}'>To view this security demonstration you will need Flash Player version 7 or a newer release. The latest version of Flash can be downloaded from the <a href='http://www.macromedia.com/go/getflashplayer' target='_blank' title='Download the latest verison of Flash'>Adobe Web site</a>.</p>".interpolate());													
			document.writeln("<p class='${args.cssclass}'>If you would prefer not to download the latest version now then you can view our <a href='${args.alternative}?brand=${args.brand}&type=${args.type}' target='${args.target}' title='${args.title}'>alternative version</a> of the demonstration.</p>".interpolate());													
		}
	};
	if ((n.appVersion.indexOf("MSIE") != -1) && (n.appVersion.indexOf("Windows") != -1))
	{
		var w = document.writeln;
		w('<SCR' + 'IPT LANGUAGE=VBScript\>');
		w('on error resume next');
		for (i=2;i<=8;i++) w('temp.v'+i+' = IsObject(CreateObject("ShockwaveFlash.ShockwaveFlash.'+i+'"))');
		w('on error goto 0');
		w('</SCR' + 'IPT\>');
		versions = temp;
	}
	else if ((n.plugins) && (n.plugins["Shockwave Flash 2.0"] || n.plugins["Shockwave Flash"]))
	{
		var version = n.plugins["Shockwave Flash" + (n.plugins["Shockwave Flash 2.0"] ? " 2.0" : "")].description.search(/(\d+\.\d+)/) != -1 ? RegExp.$1 : 0;
		versions["v"+version] = true;
	}
	return(closure);
}		
var FlashDetector = FlashDetectorFactory();
