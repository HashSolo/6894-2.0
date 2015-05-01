	 function ImageOff(imgname, imgpath)
							 {
								//alert(imgpath);
								var path = imgpath.toString();
								var index = path.lastIndexOf('/');
								//alert(index);
								var newPath = path.substring(0,index);
								//alert(newPath);
								document[imgname].src = newPath+"/"+imgname+"_off.gif";
							 }
							function ImageOn(imgname, imgpath)
							 {
								//alert(imgpath);
								var path = imgpath.toString();
								var index = path.lastIndexOf('/');
								var newPath = path.substring(0,index);
								document[imgname].src = newPath+"/"+imgname+"_on.gif";
							 }
	 
	 // browser detect params

							var isIE = (navigator.appVersion.indexOf("MSIE") != -1) ? true : false;

							var isWin = (navigator.appVersion.toLowerCase().indexOf("win") != -1) ? true : false;

							var isMac = (navigator.appVersion.toLowerCase().indexOf("mac") != -1) ? true : false;

							var isOpera = (navigator.userAgent.indexOf("Opera") != -1) ? true : false;

							// swf params

							var jsVersion = 1.0;

							var requiredMajorVersion = 7;

							var requiredMinorVersion = 0;

							var requiredRevision = 0;

							var requiredVersion = 7; // correct version of flash player required

							var maxVersion = 9;

							var actualVersion = 0;

							var flash2Installed = false;

							var flash3Installed = false;

							var flash4Installed = false;

							var flash5Installed = false;

							var flash6Installed = false;

							var flash7Installed = false;

							var flash8Installed = false;

							var flash9Installed = false;

							// write vb swf detection

							document.write('<SCR' + 'IPT LANGUAGE=VBScript> \n');

							document.write('on error resume next \n');

							for (var i = 2; i <= maxVersion; i++) {

								document.write('flash' + i + 'Installed = (IsObject(CreateObject("ShockwaveFlash.ShockwaveFlash." + CStr(' + i + ')))) \n');

							}

							document.write('<\/SCR' + 'IPT> \n');

							function JSGetSwfVerIE() {

								if (navigator.plugins != null && navigator.plugins.length > 0) {

									if (navigator.plugins["Shockwave Flash 2.0"] || navigator.plugins["Shockwave Flash"]) {

										var isVersion2 = navigator.plugins["Shockwave Flash 2.0"] ? " 2.0" : "";

										var flashDescription = navigator.plugins["Shockwave Flash" + isVersion2].description;

										var flashVersion = parseInt(flashDescription.substring(16));

										flash2Installed = flashVersion == 2;

										flash3Installed = flashVersion == 3;

										flash4Installed = flashVersion == 4;

										flash5Installed = flashVersion == 5;

										flash6Installed = flashVersion == 6;

										flash7Installed = flashVersion == 7;

										flash8Installed = flashVersion == 8;

										flash9Installed = flashVersion >= 9;

									}

								}

								for (var i = 2; i <= maxVersion; i++) {

									if (eval("flash" + i + "Installed") == true) actualVersion = i;

								}

								if(navigator.userAgent.indexOf("WebTV") != -1) actualVersion = 4;

								if (actualVersion >= requiredVersion) {

									// flash player is present and version is OK

									return true;

								}else{

									return false;

								}

							}

							function JSGetSwfVer(i){

								if (navigator.plugins != null && navigator.plugins.length > 0) {

									if (navigator.plugins["Shockwave Flash 2.0"] || navigator.plugins["Shockwave Flash"]) {

										var swVer2 = navigator.plugins["Shockwave Flash 2.0"] ? " 2.0" : "";

										var flashDescription = navigator.plugins["Shockwave Flash" + swVer2].description;

										descArray = flashDescription.split(" ");

										tempArrayMajor = descArray[2].split(".");

										versionMajor = tempArrayMajor[0];

										versionMinor = tempArrayMajor[1];

										if ( descArray[3] != "" ) {

											tempArrayMinor = descArray[3].split("r");

										} else {

											tempArrayMinor = descArray[4].split("r");

										}

										versionRevision = tempArrayMinor[1] > 0 ? tempArrayMinor[1] : 0;

										flashVer = versionMajor + "." + versionMinor + "." + versionRevision;

									} else {

										flashVer = -1;

									}

								}

								else if (navigator.userAgent.toLowerCase().indexOf("webtv/2.6") != -1) flashVer = 4;

								else if (navigator.userAgent.toLowerCase().indexOf("webtv/2.5") != -1) flashVer = 3;

								else if (navigator.userAgent.toLowerCase().indexOf("webtv") != -1) flashVer = 2;

								else {

									flashVer = -1;

								}

								return flashVer;

							}

							function DetectFlashVer(reqMajorVer, reqMinorVer, reqRevision) {

								reqVer = parseFloat(reqMajorVer + "." + reqRevision);

								if(isIE && isWin && !isOpera){

									return JSGetSwfVerIE();

								}else{

									for (i=25;i>0;i--) {

										versionStr = JSGetSwfVer(i);

										if (versionStr == -1) {

											return false;

										} else if (versionStr != 0) {

											if(isIE && isWin && !isOpera) {

												tempArray = versionStr.split(" ");

												tempString = tempArray[1];

												versionArray = tempString .split(",");

											} else {

												versionArray = versionStr.split(".");

											}

											versionMajor = versionArray[0];

											versionMinor = versionArray[1];

											versionRevision = versionArray[2];

											versionString = versionMajor + "." + versionRevision;

											versionNum = parseFloat(versionString);

											if ( (versionMajor > reqMajorVer) && (versionNum >= reqVer) ) {

												return true;

											} else {

												return ((versionNum >= reqVer && versionMinor >= reqMinorVer) ? true : false );

											}

										}

									}

								}

							}