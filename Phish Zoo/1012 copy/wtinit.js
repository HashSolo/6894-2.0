// WebTrends SmartSource Data Collector Tag
// Version: 1.1.1
// Created: 2/12/2008 5:47:24 AM
function DcsInit(){
	var that=this;
	this.dcsid="dcsg9f09p10000g4ee5r58dz8_5b8f";
	this.domain="statse.webtrendslive.com";
	this.enabled=true;
	this.exre=(function(){return(window.RegExp?new RegExp("dcs(uri)|(ref)|(aut)|(met)|(sta)|(sip)|(pro)|(byt)|(dat)|(p3p)|(cfg)|(redirect)|(cip)","i"):"");})();
	this.fpc="WT_FPC";
	this.fpcdom=".cartasi.it";
	this.i18n=false;
	this.images=[];
	this.index=0;
	this.qp=[];
	this.re=(function(){return(window.RegExp?(that.i18n?{"%25":/\%/g}:{"%09":/\t/g,"%20":/ /g,"%23":/\#/g,"%26":/\&/g,"%2B":/\+/g,"%3F":/\?/g,"%5C":/\\/g,"%22":/\"/g,"%7F":/\x7F/g,"%A0":/\xA0/g}):"");})();
	this.onsitedoms="";
	this.downloadtypes="xls,doc,pdf,txt,csv,zip";
	this.rightclicktypes="xls,doc,pdf,txt,csv,zip";
	this.timezone=1;
	this.trackevents=true;
	(function(){if(that.enabled&&(document.cookie.indexOf(that.fpc+"=")==-1)&&(document.cookie.indexOf("WTLOPTOUT=")==-1)){document.write("<scr"+"ipt type='text/javascript' src='"+"http"+(window.location.protocol.indexOf('https:')==0?'s':'')+"://"+that.domain+"/"+that.dcsid+"/wtid.js"+"'><\/scr"+"ipt>");}})();
}
var DCS={};
var WT={};
var DCSext={};
var dcsInit=new DcsInit();