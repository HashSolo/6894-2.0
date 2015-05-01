function demoPopup(url,ie_argu,ff_argu)
{
var browserName=navigator.appName;
if (browserName=="Microsoft Internet Explorer")
{
 openWindowATC(url,'','','',ie_argu,windowNamer(url));
}
else
{
  openWindowATC(url,'','','',ff_argu,windowNamer(url));
}
}



