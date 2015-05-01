
(function( $ )
{

      if ( typeof (Array.prototype.indexOf) == 'undefined' ) {
            Array.prototype.indexOf = function(item) {
                  for(var i = 0; i < this.length; i++ ) {if ( this[i] == item )return i;}
                  return -1;
            }
      }
      String.prototype.splitUrls = function() {
          var urls = this.split(',http');
            for(var i = 0; i < urls.length; i++ ) {if ( i > 0 ) urls[i] = 'http' + urls[i];}
            return urls;
      }
      var debug = false;
      jQuery.fn.log = function (msg) {
            if ( debug && console )console.log("%s: %o", msg, this);
            return this;
    };
 
      jQuery.fn.extend({
            executeOnce: function(label,delay, fn, option) {return this.each(function() {jQuery.timer.delayExecute(this, delay, label, fn, option);});},
            stopExecution: function(label) {return this.each(function() {jQuery.timer.remove(this, label);});}
      });
      
      jQuery.extend({
            timer: {
                  delayExecute: function(element, delay, label, fn, option) {
                        var timer = jQuery.data(element, label);
                        if ( timer ) {
                              if ( timer.timerId ) {
                                    if ( option == 'block' ) return;
                                    window.clearTimeout(timer.timerId);
                                    timer.timerId = null;
                              }
                        } else {
                              timer = {};
                              jQuery.data(element, label, timer);
                        }
                        var timerId;
                        var handler = function() {
                              if ( timer.timerId == timerId ) {
                                    timer.timerId = null;
                                    fn.call(element);
                              } else {
                              }
                        };
                        timerId = timer.timerId = window.setTimeout(handler,delay);
                  
                  },
                  remove: function(element, label) {
                        var timer = jQuery.data(element, label);
                        if ( timer ) {
                              if ( timer.timerId ) {
                                    window.clearTimeout(timer.timerId);
                                    timer.timerId = null;
                              }
                        }
                  }
            }
      });


      $.fn.document = function() {
            var element = this[0];
            if ( element.nodeName.toLowerCase() == 'iframe' )
                  return element.contentWindow.document;
            else
                  return $(this);
      };
      
      var getTransparent = function(el){
            for(var n=0;n<el.parents().length;n++){
                  var parent = el.parents().get(n);
                  var pBg = $.browser.safari ? $(parent).css("background") : $(parent).css("background-color");
                  if(pBg!='' && pBg!='transparent'){
                        return pBg;
                  }
            }
            return '#FFF';
      }

      $.fn.hoverClass = function(clsName) {
            this.bind('mouseover',function(e){$(this).addClass(clsName);}).bind('mouseout',function(e){$(this).removeClass(clsName);});
        return this;
      };
      
      $.fn.loadContent = function(url,data,callback,errorCallback ) {
            var parent = this;
            $.ajax({
                  type: "GET",
                  url: url,
                  //data: $("#flip").serialize(),
                  beforeSend: function(){
                  },
                  success: function(html){
                        $(html).appendTo(parent).find('input:text').inputFields();
                        callback(html);
                  },
                error: errorCallback
            });
      };
      
      var fadeFx = function( opts) {
            opts.showAnim        = { opacity: 1 };
            opts.hideAnim   = { opacity: 0 };
            opts.cssBeforeShow = { opacity: 0,display:''};
            opts.cssAfterShow = { opacity:''};

        if ( opts.hideSpeed == 0 ) opts.hideSpeed = 50;
          if ( opts.showSpeed == 0 ) opts.showSpeed = 50;
            opts.sync=false;
            opts.beforeHide.push(function(toHide){
                var offset = toHide.position();
                  this.cssBeforeHide = {top:offset.top + 'px',left:offset.left + 'px',opacity:1,zIndex:100000,position:'absolute'}; 
                  this.cssAfterHide = {position:'',top:'',left:'',opacity:'',zIndex : '',display:'none'};
            })
            opts.beforeShow.push(function(toShow){
                var h = toShow.outerHeight();
                var w = toShow.outerWidth();
                var ph = toShow.parent().outerHeight();
                var pw = toShow.parent().outerWidth();
                var y = (ph-h)/2;
                  var x = (pw-w)/2;
                  var offset = this.jqContentContainer.position();
            toShow.css({position:'absolute',top: (offset.top + y) + 'px',left: (offset.left + x) + 'px'});
            });
            opts.cssAfterShow = {zIndex:'',position : '',left:'',top:'',textAlign:'center',verticalAlign:'middle'};
            opts.cssAfterShow = {zIndex:''};

      };

      var vScrollFx = function( opts) {
            opts.hideAnim   = { height: 'hide' };
            opts.showAnim = { height: 'show' };
            opts.cssAfterShow = { display: '',top:'0px'}
            opts.hideSpeed = 1300;
          opts.showSpeed = 1300;
      };

      var tossFx = function(opts) {
            if (opts.xOffset == 0 )
                  opts.xOffset = 100;
            if ( opts.yOffset == 0 )
                  opts.yOffset = 50;
            opts.beforeHide.push(function(toHide){
                var offset = toHide.position();
                  var color = getTransparent(toHide);
                  this.cssBeforeHide = {top:offset.top + 'px',left:offset.left + 'px',opacity:1,zIndex:100000,backgroundColor:color,position:'absolute'};
                  
                  this.cssAfterHide = {position:'',top:0,left:0,opacity:'',zIndex : '',display:'none'};
                  this.cssBeforeShow = {opacity:0,zInex:990000,display:''};//,verticalAlign: 'middle'};//top:'',left:'',position:'static'};
                  this.cssAfterShow = {opacity:'',zIndex:''};
                  this.hideAnim = {left: (offset.left + this.xOffset) + 'px', top: (offset.top - this.yOffset) + 'px',opacity:0};

            });
            if ( this.removeAfterHide ) {
                  opts.afterHide.push(function(toHide){toHide.remove();});
            }
            if ($.browser.msie) { 
                  opts.beforeShow.push(function(toShow){
                  if ( this.dynamicLoad && toShow[0].tagName == 'IMG' && this.center ) {
                        var h = toShow.outerHeight();
                        var ph = toShow.parent().outerHeight();
                              var m = (ph-h)/2;
                              if ( m > 0 ) toShow.css('marginTop', m + 'px');
                  }
                  });
        }
            opts.showAnim = { opacity:1 };
            opts.sync = true;
            opts.hideSpeed = 1200;
            opts.showSpeed = 500;
      };

      var zoomFx = function(opts) {
            opts.beforeHide.push(function(toHide){
                  var pw = toHide.parent().outerWidth();
                  var ph = toHide.parent().outerHeight();
                  var w = toHide.outerWidth();
                  var h = toHide.outerHeight();
                  var x = (pw-w)/2;
                  var y = (ph-h)/2;
                var offset = toHide.position();
                  var color = getTransparent(toHide);
                  this.cssBeforeHide = {top:offset.top + 'px',left:offset.left + 'px',width:w + 'px',height: h + 'px',opacity:1,zIndex:100000,backgroundColor:color,position:'absolute'};
                  this.jqContentContainer.css('display','block');
                  offset = this.jqHideTarget.position();

                  this.hideAnim={width:this.jqHideTarget.outerWidth() + 'px',height:this.jqHideTarget.outerHeight() + 'px',top:offset.top + 'px',left: offset.left + 'px'};

            });
            if ( this.removeAfterHide ) {
                  opts.afterHide.push(function(toHide){toHide.remove();});
            }
            opts.beforeShow.push(function(toShow){
                var offset = this.jqShowSrc.position();
                  var ws = this.jqShowSrc.outerWidth();
                  var hs = this.jqShowSrc.outerHeight();
                  var pw = this.jqContentContainer.outerWidth();
                  var ph = this.jqContentContainer.outerHeight();
                  var w = toShow.outerWidth();
                  var h = toShow.outerHeight();
                  var x = (pw-w)/2;
                  var y = (ph-h)/2;
                  var color = getTransparent(toShow);
                  this.cssBeforeShow = {top:offset.top + 'px',left:offset.left + 'px',width:ws + 'px',height: hs + 'px',opacity:1,zIndex:100000,backgroundColor:color,position:'absolute'};
                  var parent = this.jqContentContainer;
                  offset = this.jqContentContainer.position();
                  this.showAnim = { width: w + 'px',height: h + 'px',left: (offset.left + x)+ 'px',top: (offset.top + y)+ 'px'};

            });

            opts.cssAfterShow = {zIndex:'',position : '',left:'',top:'',textAlign:'center'};
            opts.cssAfterShow = {zIndex:''};
            opts.sync = true;
            if ( opts.hideSpeed == 0 ) opts.hideSpeed = 200;
            if ( opts.showSpeed == 0 ) opts.showSpeed = 200;
      };

      var zoomxFx = function(opts) {
            opts.beforeHide.push(function(toHide){
                  var pw = toHide.parent().outerWidth();
                  var ph = toHide.parent().outerHeight();
                  var w = toHide.outerWidth();
                  var h = toHide.outerHeight();
                  var x = (pw-w)/2;
                  var y = (ph-h)/2;
                  this.cssBeforeHide = {};
                  this.hideAnim={width:'hide'};
                  //if ($.browser.msie) { 
                  //if ( this.dynamicLoad && toHide[0].tagName == 'IMG' && this.center ) {
                  if ( this.center ) {
                        var h = toHide.outerHeight();
                        var ph = toHide.parent().outerHeight();
                              var m = (ph-h)/2;
                              if ( m > 0 ) toHide.css('marginTop', m + 'px');
                  }
            //}

            });
            if ( this.removeAfterHide ) {
                  opts.afterHide.push(function(toHide){toHide.remove();});
            }
            opts.beforeShow.push(function(toShow){
                  var pw = toShow.parent().outerWidth();
                  var ph = toShow.parent().outerHeight();
                  var w = toShow.outerWidth();
                  var h = toShow.outerHeight();
                  var x = (pw-w)/2;
                  var y = (ph-h)/2;
                  this.cssBeforeShow = {};
                  this.showAnim = { width:'show'};
                  //if ($.browser.msie) { 
                  //if ( this.dynamicLoad && toShow[0].tagName == 'IMG' && this.center ) {
                  if ( this.center ) {
                        var h = toShow.outerHeight();
                        var ph = toShow.parent().outerHeight();
                              var m = (ph-h)/2;
                              if ( m > 0 )
                        toShow.css('marginTop', m + 'px');
                  }
            //}

            });

            //opts.sync = true;
            opts.hideSpeed = 800;
            opts.showSpeed = 1200;
      };
      
      var shuffleFx = function(opts) {
            opts.beforeHide.push(function(toHide){
                  var w = toHide.parent().outerWidth();
                  var offset = toHide.position();
                  var offset2 = toHide.parent().position();
                  var color = getTransparent(toHide);
                  //toHide.css('background-color', color);
                  this.cssBeforeHide = {top:offset.top + 'px',left:offset.left + 'px',zIndex:800000,position:'absolute',backgroundColor:color};
                  this.cssAfterHide = {top:'',left:''};//,zIndex :'',position:''};//,opacity:''};
                  this.hideAnim = {left: '-=' + (w+this.xOffset) + 'px',top:'-=' + this.yOffset + 'px'};//,opacity:.4};
                  this.hideAnim2 = {left: '+=' + (w+this.xOffset) + 'px',top:'+=' + this.yOffset + 'px'};//,opacity:0};

                  this.cssBeforeShow = {zIndex:990000,opacity:0,position:'absolute',backgroundColor:color};
                  this.cssAfterShow = {zIndex: ''};
                  this.showAnim = {opacity:1};

            });
            opts.doHideAnimate = function(jqToHide,jqToShow) {
                  var mgr = this;
                  jqToHide.animate(this.hideAnim, this.hideSpeed).animate(mgr.hideAnim2, mgr.hideSpeed, function() { //
                              jqToHide.hide();
                              mgr.doAfterHide(jqToHide);
            });
                  this.doShow(jqToShow);
            }

            opts.beforeShow.push(function(toShow){
                  
                  var pw = toShow.parent().outerWidth();
                  var ph = toShow.parent().outerHeight();
                  var w = toShow.outerWidth();
                  var h = toShow.outerHeight();
                  var x = (pw-w)/2;
                  var y = (ph-h)/2;
                  this.cssBeforeShow = {};
                  if ($.browser.msie) { 
                  if ( this.center ) {
                        var h = toShow.outerHeight();
                        var ph = toShow.parent().outerHeight();
                              var m = (ph-h)/2;
                              if ( m > 0 )
                        toShow.css('marginTop', m + 'px');
                  }
            }

            });

            opts.sync = true;
            opts.hideSpeed = 500;
            opts.showSpeed = 100;

      };
      var replaceImage = function (jqToHide,jqToShow) {
            this.doBeforeHide(jqToHide);
            jqToHide.attr('src',jqToShow.attr('src'));
            this.doAfterShow(jqToShow);
      }
      var replaceContent = function (jqToHide,jqToShow) {
            this.doBeforeHide(jqToHide);
            jqToHide.hide();
            this.doAfterHide(jqToHide);
            jqToShow.show();                
            this.doAfterShow(jqToShow);

      }
            
      var defaultDisplayMgr = {
          fx:'',
            hideAnim: null,
          hideSpeed: 0,
            cssBeforeHide : {},
            cssAfterHide : {},
            beforeHide : [],
            afterHide : [],
          showAnim : null,
          showSpeed:0,
            cssBeforeShow : {},
            cssAfterShow : {},
            beforeShow : [],
            afterShow : [],
            sync: false,
            removeAfterHide : false,
            xOffset :0,
            yOffset :0,
            doBeforeHide : function(jqToHide) {
                  var param = [jqToHide];
                  for(i=0; i < this.beforeHide.length; i++ )
                        this.beforeHide[i].apply(this,param);
                  jqToHide.css(this.cssBeforeHide);
            },
            doAfterHide : function(jqToHide) {
                  var param = [jqToHide];
                  for(i=0; i < this.afterHide.length; i++ )
                        this.afterHide[i].apply(this,param);
                  jqToHide.css(this.cssAfterHide);
                  jqToHide.hide();
            if (typeof this.onHide == 'function') {
                  this.onHide();
                  }
            },
            doBeforeShow : function(jqToShow) {
                  var param = [jqToShow];
                  for(i=0; i < this.beforeShow.length; i++ )
                        this.beforeShow[i].apply(this,param);
                  jqToShow.css(this.cssBeforeShow);
            },
            doAfterShow : function(jqToShow) {

                  var param = [jqToShow];
                  for(i=0; i < this.afterShow.length; i++ )
                        this.afterShow[i].apply(this,param);
            jqToShow.css(this.cssAfterShow);
                  jqToShow.css('opacity',1).show(); // opacity bit needed to handle reinit case
                  if ($.browser.msie) jqToShow[0].style.removeAttribute('filter');
            if (typeof this.onShow == 'function') {
                  this.onShow();
            }
            },
            doHideAnimate : function(jqToHide,jqToShow) {
                  var mgr = this;
                  jqToShow.hide();
                  jqToHide.animate(this.hideAnim, this.hideSpeed, function() { //
                        jqToHide.hide();
                        mgr.doAfterHide(jqToHide);
                        if( !mgr.sync ) mgr.doShow(jqToShow);
            });
                  if ( mgr.sync ) this.doShow(jqToShow);
            },
            doShowAnimate : function(jqToShow) {
                  var mgr = this;
            jqToShow.animate(this.showAnim, this.showSpeed, function() {
                        jqToShow.show();
                        mgr.doAfterShow(jqToShow);
             });
            },
            doShow : function (jqToShow) {
                  this.doBeforeShow(jqToShow);
                  if ( this.showAnim && this.showSpeed ) {
                        this.doShowAnimate(jqToShow);
                  } else {
                        jqToShow.show();
                        this.doAfterShow(jqToShow);
            }
            },
            doHide : function(jqToHide,jqToShow) {
                  this.doBeforeHide(jqToHide);
                  if ( this.hideAnim && this.hideSpeed ) {
                        this.doHideAnimate(jqToHide,jqToShow);
                  } else {
                        jqToHide.hide();
                        this.doAfterHide();
                      this.doShow(jqToShow);
              }
            },
            toggleDisplay : function(jqToHide,jqToShow) {
                  this.doHide(jqToHide,jqToShow);
            },
            init : function () {
                  switch( this.fx ) {
                        case 'toss':
                              tossFx(this);
                              break;
                        case 'zoom':
                              zoomFx(this);
                              break;
                        case 'vScroll':
                              vScrollFx(this);
                              break;
                        case 'fade':
                              fadeFx(this);
                              break;
                        case 'shuffle':
                              shuffleFx(this);
                              break;
                        case 'replaceContent':
                              this.toggleDisplay = replaceContent;
                              break;
                        case 'replaceImage':
                              this.toggleDisplay = replaceImage;
                              break;
                  }

            }
      };
      
      var defaultContentMgr = {
            id : "contentMgr",
            display:$.extend({},defaultDisplayMgr),
            alternateDisplay: null,
            contents:null,
            urlSelect:null,
            activeIndex: 0,
            maxIndex:0,
            wrapIndex : true,
            dynamicLoad : false,
            removeAfter : false,
            isImage : false,
            urlList : null,
            jqContainer: null,
            jqContents : null,
            jqCacheContents : null,
            cache : false,
            slideShowDelay : 500,
            slideShowCycle : 2000,
            slideShowEnabled : false,
            beforeDynamicLoad : function () {
            },
            afterDynamicLoad : function () {
            },
            //initList : [],
            onShow : function () {
//it is possible that hide have not finish
                  //this.resetActiveContent();
//do on certian condition only
                  this.jqContents = this.jqContainer.find(this.contents);
                  this.jqActiveContent = this.jqContents.filter(':visible');

                  if ( this.slideShowEnabled )
                        this.delayNext();
            },
            onHide : function () {
            },
            delayNext : function (delay) {
                  if ( this.slideShowEnabled ) {
                        var mgr = this;
                        var handler = function () {
                              mgr.next();
                        }
                        this.jqContainer.executeOnce(this.id+'sstimer',delay || this.slideShowCycle, handler, 'block');
                  }
            },
            startSlideShow : function() {
                  this.slideShowEnabled = true;
                  this.delayNext(this.slideShowDelay);
            },
            stopSlideShow : function () {
                  this.slideShowEnabled = false;
                  this.jqContainer.stopExecution(this.id+'sstimer');
            },
            resetActiveContent : function () {
                  var previous = this.jqActiveContent;
                  this.jqActiveContent = this.jqContents.filter(':visible');
                  if ( this.jqActiveContent.length > 1 ) {
                        var idx = this.jqActiveContent.index(previous[0]);
                        this.jqActiveContent = $(this.jqActiveContent[idx == 0 ? 1 :0 ]);
                  }
            },
            switchToIndex : function(idx) {
                  if ( idx != this.activeIndex ) {
                        if ( this.cache ) {
                              var url = this.urlList[idx];
                              if ( this.urlList[url] ) {
                                    this.switchToContent(this.urlList[url]); 
                                    this.activeIndex = idx;
                              } else {
                                    this.loadContent(url);
                              }
                        } else if ( this.dynamicLoad ) {
                              this.loadContent(this.urlList[idx]);
                        } else {
                              this.switchToContent($(this.jqContents[idx]));
                              this.activeIndex = idx;
                        }
                  }
            },
            switchToContent : function(jqTarget) {
                  if ( this.jqContents.length > 1 )
                        this.jqActiveContent = this.jqContents.filter(':visible');
                  else
                        this.jqActiveContent = this.jqContents;
                  this.display.jqShowSrc = this.jqShowSrc;
                  this.display.jqHideTarget = this.jqCurrent;
                  this.display.toggleDisplay(this.jqActiveContent,jqTarget);
            },
            previous : function() {
                  if ( this.activeIndex > 1 ) {
                        this.switchToIndex(this.activeIndex-1);
                  } else if ( this.wrapIndex ) {
                        this.switchToIndex(this.maxIndex);
                  }
            },
            next : function() {
                  if ( this.activeIndex < this.maxIndex ) {
                        this.switchToIndex(this.activeIndex + 1);
                  } else {
                        if (this.wrapIndex || this.slideShowEnabled ) {
                        this.switchToIndex(0);
                        }
                  }
            },
            loadContent : function (url,idx) {
                  this.beforeDynamicLoad();
                  if ( this.isImage ) {
                        this.dynamicLoadImage(url);
                  } else {
                        //call  before dynamic load
                        //opts.jqSelected.addClass('loading');
                        var mgr = this;
                        this.jqActiveContent.parent().loadContent(url,null,function(html) {mgr.afterLoad(url,$(html),false);},null);
                  }
            },
            dynamicLoadImage : function (url) {
                  var imgPreloader = new Image();
                  var mgr = this;
                  imgPreloader.onload=function(){
                        mgr.afterLoad(url,$(imgPreloader),mgr.display.fx != 'replaceImage');
                  }
                  imgPreloader.src = url;
            },
            afterLoad : function (url,elem,append) {
                  this.afterDynamicLoad();
                  var index = this.urlList.indexOf(url);
                  if ( append )
                        elem.hide().appendTo(this.jqActiveContent.parent());
                  //elem.find('input:text').inputFields();

                  if ( this.cache   ) {
                        this.urlList[url] = elem;
                        this.switchToIndex(index);
                  } else {
                        this.activeIndex = index;
                        this.switchToContent(elem); 
                  }
            },
            initContext : function () {
                  this.resetActiveContent();
                  if ( this.urlSelect || this.getUrls) {
                        this.dynamicLoad = this.display.dynamicLoad = true;
                        if ( this.alternateDisplay )
                            this.alternateDisplay.dynamicLoad = true;
                        var urls;
                        //for backward compatibility - should be remove
                        if ( this.getUrls ) {
                            urls = this.getUrls();
                              this.maxIndex = this.urlList.length -1;
                        } else {
                            if ( this.urlSelect.charAt(0) == '[' && this.urlSelect.charAt(this.urlSelect.length - 1) == ']' ) {
                            //if ( this.urlSelect == '[img]' ) {
                                  urls = this.jqContainer.attr(this.urlSelect.substring(1,this.urlSelect.length-1));
                            } else {
                                urls = this.jqContainer.find(this.urlSelect);
                                if ( urls.length > 0 ) {
                                      urls = urls.attr('urls');
                                }
                            }
                            if ( urls.length > 0 ) {
                                  this.urlList = urls.splitUrls();
                                  this.maxIndex = this.urlList.length -1;
                            }
                        }
                        
                        if ( this.isImage ) {
                              this.activeIndex = urls.indexOf(this.jqContents[0].src);
                        //else {
                              //this.activeIndex = this.jqContents.index(this.jqActiveContent[0]);
                        }
                  } else {
                        this.maxIndex = this.jqContents.length - 1;
                        this.activeIndex = this.jqContents.index(this.jqActiveContent[0]);
                  }
            },
            init : function () {
                  var mgr = this;
                  if ( this.jqContainer ) {
                  this.jqContents = this.jqContainer.find(this.contents);
                  }
                  if ( this.jqContents ) {
                        this.display.jqContentContainer = this.jqContents.parent();
                        if ( this.alternateDisplay ) this.alternateDisplay.jqContentContainer = this.jqContents.parent();
                        this.initContext();
                  }

                  //replcae image only if dynamic and isImage
                  if ( this.display.fx == '' ) {
                        if ( this.dynamicLoad && this.isImage )
                              this.display.fx = 'replaceImage';
                        else
                              this.display.fx = 'replaceContent';
                  }
                  if ( this.dynamicLoad && this.fx == 'toss' )
                        this.display.removeAfterHide = true;
                  this.display.init();
                  this.display.onShow = function () {
                        mgr.onShow();
                  }
                  this.display.onHide = function () {
                        mgr.onHide();
                  }
                  if ( this.alternateDisplay) {
                      if ( this.alternateDisplay.fx == '' ) {
                            if ( this.dynamicLoad && this.isImage )
                                  this.alternateDisplay.fx = 'replaceImage';
                            else
                                  this.alternateDisplay.fx = 'replaceContent';
                      }
                      this.alternateDisplay.init();
                      this.alternateDisplay.onShow = function () {
                            mgr.onShow();
                      }
                      this.alternateDisplay.onHide = function () {
                            mgr.onHide();
                      }
                  }
                  //(this.initList.length);
                  if ( this.initList )
                  for(i=0; i < this.initList.length; i++ )
                        this.initList[i].apply(this);

            }
      };
      
      function extendDisplay(displayOptions) {
          var display=$.extend({},defaultDisplayMgr,displayOptions);
            display.beforeHide = [].concat(defaultDisplayMgr.beforeHide);
            if ( displayOptions && displayOptions.beforeHide )
                  display.beforeHide.concat(displayOptions.beforeHide);
            display.afterHide = [].concat(defaultDisplayMgr.afterHide);
            if ( displayOptions && displayOptions.afterHide )
                  display.afterHide.concat(displayOptions.afterHide);
            display.beforeShow = [].concat(defaultDisplayMgr.beforeShow);
            if ( displayOptions && displayOptions.beforeShow )
                  display.beforeShow.concat(displayOptions.beforeShow);
            display.afterShow = [].concat(defaultDisplayMgr.afterShow);
            if ( displayOptions && displayOptions.afterShow )
                  display.afterShow.concat(displayOptions.afterShow);
            return display;
      }

      $.fn.configureContentManager = function(options) {
            if ( options.contents == undefined )
            options.contents = '>';

          var settings = $.extend({},defaultContentMgr,options);
          settings.display = extendDisplay(options.display);
          if ( options.alternateDisplay ) {
              settings.alternateDisplay = extendDisplay(options.alternateDisplay);
          }

            var doClone = this.length > 1;
            this.each(function() {
                  var config = doClone ? $.extend({},settings) : settings;
                  config.jqContainer = $(this);
                  if ( doClone ) {
                        config.display = $.extend({},settings.display);
                        if ( config.alternateDisplay ) {
                            config.alternateDisplay = $.extend({},settings.alternateDisplay);
                        }
                  }
                  config.init();
                jQuery.data(this, config.id, config);
            });
            return this;

      };
      
      jQuery.fn.extend({
            startSlideShow: function() {
                  this.each(function() {
                  var contentMgr = jQuery.data(this, defaultContentMgr.id);
                        if ( contentMgr )
                              contentMgr.startSlideShow();
                  });
                  return this;
            },
            stopSlideShow: function() {
                  return this.each(function() {
                  var contentMgr = jQuery.data(this, defaultContentMgr.id);
                        if ( contentMgr )
                              contentMgr.stopSlideShow();
                  });
            }
      });


      /*
       * $('conatiner').hoverSlideShow(options)
       */
      $.fn.hoverSlideShow = function(options) {
            this.configureContentManager(options);
            return this.each(function() {
                  $(this).bind('mouseleave',function(e) {
                        $(this).stopSlideShow();
                  }).bind('mouseenter',function(e) {
                        $(this).startSlideShow();
                  });
            });
      };

      var navMgr = {
          navId : null,
            locked:false,
            navs : 'li',
            selectedClass:'selected',
            disabledClass:'disabled',
            hoverClass: 'hover',
            nextButton: null,
            previousButton : null,
        previousButtonHoverClass : null,
        nextButtonHoverClass : null,
            activeClass : 'active',
            contentContainer:null,
            initList : [],
            activeNavIndex : 0,
            getActiveIndex : function () {
                return this.activeNavIndex;
            },
            beforeDynamicLoad : function () {
                  if ( this.onloadClass) {
                        this.jqSelected.addClass(this.onloadClass);
                  }
            },
            afterDynamicLoad : function () {
                  if ( this.onloadClass) {
                        this.jqSelected.removeClass(this.onloadClass);
                  }
            },
            navSelection : function () {
                  this.jqShowSrc = this.jqSelected;
                  this.activeNavIndex = this.jqNavs.index(this.jqSelected);
                  this.switchToIndex(this.activeNavIndex);
            },
            nextTab : function () {
                var mgr = this;
                  if ( !mgr.locked) {
                      mgr.locked = true;
                      mgr.jqCurrent = mgr.jqNavs.filter('.' + mgr.selectedClass);
                      mgr.activeNavIndex = mgr.jqNavs.index(mgr.jqCurrent);
                      mgr.nextTabIndex();
                      mgr.jqSelected = $(mgr.jqNavs[mgr.activeNavIndex]);
                      mgr.navSelection();
              }
            },
            nextTabIndex : function() {
                if (this.activeNavIndex == this.maxIndex )
                    this.activeNavIndex =0;
              else
                        this.activeNavIndex++;
        },
        previousTabIndex : function () {  
            if (this.activeNavIndex == 0 )
                        this.activeNavIndex =this.maxIndex;
                  else
                        this.activeNavIndex--;
          },
            addToInitList : function () {
                this.initList.push(function() {
                        var mgr = this;
                        if ( mgr.navId ){
                    jQuery.data(mgr.jqNavContainer[0], mgr.navId,mgr);
                }
                this.jqNavs = this.jqNavContainer.find(this.navs);
                        if ( this.jqNavs.length == 0)
                              $.log("can't find " + this.navs);
                    this.locked =false;
                    if ( this.nextButton ) {
                        this.jqNext = this.jqNavContainer.find(this.nextButton);
                              if ( this.activeClass && this.wrapIndex )
                                    this.jqNext.addClass(this.activeClass);
                              if ( this.nextButtonHoverClass )
                                  this.jqNext.hoverClass(this.nextButtonHoverClass);
                        this.jqNext.click(function() {
                                    if ( $(this).hasClass(mgr.activeClass) && !mgr.locked) {
                              if (mgr.alternateDisplay ) mgr.display = mgr.alternateDisplay;
                                        mgr.nextTab();
                                    }
                              });
                    }
                    if ( this.previousButton ) {
                        this.jqPrevious = this.jqNavContainer.find(this.previousButton);
                              if ( this.activeClass && this.wrapIndex )
                                    this.jqPrevious.addClass(this.activeClass);
                              if ( this.previousButtonHoverClass) 
                                  this.jqPrevious.hoverClass(this.previousButtonHoverClass);
                        this.jqPrevious.click(function() {
                                    if ( $(this).hasClass(mgr.activeClass) && !mgr.locked) {
                                          mgr.jqCurrent = mgr.jqNavs.filter('.' + mgr.selectedClass);
                                    mgr.activeNavIndex = mgr.jqNavs.index(mgr.jqCurrent);
                                    mgr.previousTabIndex();
                                          mgr.jqSelected = $(mgr.jqNavs[mgr.activeNavIndex]);
                                          mgr.locked = true;
                              if (mgr.alternateDisplay ) mgr.display = mgr.alternateDisplay;
                                          mgr.navSelection();
                                    }
                              });
                    }
                    if ( this.hoverClass ) {
                        this.jqNavs.hoverClass(this.hoverClass);
                        if ( this.jqNext )
                            this.jqNext.hoverClass(this.hoverClass);
                        if ( this.jqPrevious )
                            this.jqPrevious.hoverClass(this.hoverClass);
                    }
                    var mgr = this;
                    this.display.afterHide.push(function() {
                          mgr.jqCurrent.removeClass(mgr.selectedClass);
                    });
                    this.display.afterShow.push(function() {
                          if ( mgr.jqSelected ) {
                                mgr.jqSelected.addClass(mgr.selectedClass);
                                mgr.jqSelected = null;
                          }
                          //update the next/previous if exist
                          mgr.locked = false;
                    });
                    this.mainDisplay = this.display;
                    if ( this.alternateDisplay ) {
                        this.alternateDisplay.afterHide.push(function() {
                              mgr.jqCurrent.removeClass(mgr.selectedClass);
                        });
                        this.alternateDisplay.afterShow.push(function() {
                              if ( mgr.jqSelected ) {
                                    mgr.jqSelected.addClass(mgr.selectedClass);
                                    mgr.jqSelected = null;
                              }
                              //update the next/previous if exist
                              mgr.locked = false;
                        });
                    }
                      this.jqNavs.bind('click', function(e) {
                            e.preventDefault();
                    if (mgr.locked ) {
                                  this.blur();
                            return false;
                      }                 
                        //selected
                            mgr.jqCurrent = mgr.jqNavs.filter('.' + mgr.selectedClass);
                            mgr.activeNavIndex = mgr.jqNavs.index(mgr.jqCurrent);
                              mgr.jqSelected = $(this);

                        // if animation is still running, tab is selected or disabled or onClick callback returns false stop here
                  // check if onClick returns false last so that it is not executed for a disabled tab
                      if (mgr.locked || mgr.jqSelected.is('.' + mgr.selectedClass) || mgr.jqSelected.is('.' + mgr.disabledClass) ) {
                                  this.blur();
                            return false;
                      }
                      mgr.locked = true;
                      mgr.display = mgr.mainDisplay;
                            mgr.navSelection();
                      });
                        
            });
          }
      };
      
      /*
       * $('conatiner of tabs').navs(options)
       */
      $.fn.navs = function(options) {
          var settings = $.extend({},navMgr,options);
          settings.initList = [];
          settings.jqNavContainer = this;
            settings.addToInitList();
            var contents = $(settings.contentsContainer);
            if ( contents.length == 0 )
                  $.log('unable to find nav content' + settings.contentsContainer);
            else
                  contents.configureContentManager(settings);
            return this;
      };
      $.fn.applyToNavMgr = function(id,fn){
        var mgr = jQuery.data(this[0],id);
        if ( mgr != null )
                  fn.apply(this, [mgr]);
            return this;
      }

      $.fn.allowDrag = function(src,options){
            var dragMgr = null;
            var _drag = function(e) {
                  e.preventDefault();
                  if ( dragMgr )
                        dragMgr.drag(e);
            }
            var _dragStop = function(e) {
                  e.preventDefault();
                  if ( dragMgr )
                        dragMgr.dragStop(e);
            }
            var mleave = function(e) {
                  e.preventDefault();
                  if (! e.relatedTarget) {
                        if ( dragMgr )
                              dragMgr.dragStop(e);
                  }
            }

            var defaultDragMgr = {
                  sensitivity:0,
                  trackMouseMove : function(e) {
                  },
                  drag : function(e) {
                        //if ( e.which == 1 ) {
                        if ( this.dragMode == 'start' ) {
                              var xDelta = e.pageX - this.mX;
                              var yDelta = e.pageY - this.mY;
                              if ( ( Math.abs(xDelta) > dragMgr.sensitivity) || (Math.abs(yDelta)  > dragMgr.sensitivity) ) {
                                    this.dragMode = 'dragging';
                              }
                        }
                        if ( this.dragMode == 'dragging' ) {
                              this.trackMouseMove(e);
                        }
                        /*} else {
                              this.dragStop(e);
                        }*/
                  },
                  dragStop : function (e) {
                        if ( this.dragMode ) {
                              $(document).unbind('mousemove',_drag).unbind('mouseup',_dragStop).unbind('mouseout',mleave);
                              dragMgr = null;
                              this.dragMode = null;
                        }
                  }
            };
            var content = this;
            $(src,this).bind('mousedown',function (e) { 
                  if ( dragMgr == null || !dragMgr.dragMode ) {
                      e.preventDefault();
                        dragMgr =$.extend({},defaultDragMgr, options);
                        dragMgr.mX = dragMgr.oX = e.pageX;
                        dragMgr.mY = dragMgr.oY = e.pageY;
                        dragMgr.offset = content.position();

                        dragMgr.dragMode='start';
                        $(document).bind('mouseout',mleave).bind('mouseup',_dragStop).bind('mousemove',_drag);
                  }
            }).css({cursor:'move'});
            
            return this;
            
      };

      /*
       * Layer object for popup menu, tooltip, dialog (modal/none modal)
       * shimlayer is needed only for <= IE6
       * shimlayer sizes modal dialog: full window size; none-modal, tooltip, popup: size of container
       * zIndex 1000 *n reserve for shim layer
       * zIndex 1000 * n + 1 reserver for modal overlay
       * zIndex 1000 * n  + 2 reserve for each container 
       */   
      var _needShimLayer = ($.browser.msie && ($.browser.version < 7));
      $.layerMgr = {
                  _layers : [],
                  newLayer : function( content,options ) {
                        var defaultLayer = {
                              cloneData : false,
                              onOpen : null,
                              onClose : null,
                              onOpenEvents : [],
                              onCloseEvents : [],
                              onInit : function() {},
                              addOnOpenEvent : function (f) {
                                    this.onOpenEvents.push(f);
                              },
                              open: function (external) {
                                    if (!external && $.isFunction(this.onOpen)) {
                                          // execute the onOpen callback 
                                      this.onOpen.apply(this);
                                    }
                                    else {
                                          if ( this.cloneData ) {
                                          this.jqClonedContent = this.jqContent.clone().appendTo('body');
                                          }
                                          if ( this.jqShim ) {
                                                this.jqShim.show();
                                          }
                                          if ( this.jqOverlay ){
                                                this.jqOverlay.show();
                                                //this.jqOverlay.fadeIn('slow');
                                          }
                                          this.jqContent.show();
                                          //this.container.slideDown('slow');
                                    }
                                    // bind default events
                                    this.bindEvents();
                              },
                              close: function (external) {
                                    //call close on all layers with higher index than the current index
                                    $.layerMgr._layers.pop();
                                    if ($.isFunction(this.onClose) && !external) {
                                          // execute the onClose callback
                                          this.onClose.apply(this);
                                    }
                                    else {
                                        for(var i=0; i < this.onCloseEvents.length; i++ ){
                                              this.onCloseEvents[i].apply(this);
                                        }
                                          this.jqContent.hide();
                                          if ( this.cloneData ) {
                                              this.jqContent.remove();
                                                this.jqContent = this.jqClonedContent;
                                          }
                                          if ( this.jqOverlay )
                                                this.jqOverlay.hide();
                                                //this.jqOverlay.fadeOut('slow');
                                          if (this.jqShim) {
                                                //if ( this.jqShim == this._modalShimLayer )
                                                //    this.jqShim.hide();
                                                //else
                                                      this.jqShim.remove();
                                          }
                                        this.unbindEvents();
                                        if ( this.src ) {
                                              this.src._layer = null;
                                              this.src = null;
                                        }
                                    }
                              },
                              trackMouseMove :function( e ) {
                                    var xDelta = e.pageX - this.mX;
                                    var yDelta = e.pageY - this.mY;
                                    if ( ( Math.abs(xDelta) > this.sensitivity) || (Math.abs(yDelta)  > this.sensitivity) ) {
                                          var offset = this.jqContent.position();
                                          this._moveTo((offset.left + xDelta),(offset.top + yDelta));
                                          this.mX = e.pageX;
                                          this.mY = e.pageY;
                                    }

                              },
                              bindEvents: function () {
                                    for(var i=this.onOpenEvents.length-1; i >= 0; i-- )
                                          this.onOpenEvents[i].apply(this);
                              },
                              unbindEvents: function () {
                              },
                              createShimLayer : function() {
                                    if ( _needShimLayer )
                                          return $('<iframe src="javascript:false;" frameBorder=0 class="shim">')
                                                .css({position: 'absolute',height: '100%',width: '100%',top: '0px',left: '0px'})
                                                .appendTo('body');
                                    return null;
                              },
                              _moveTo : function ( left, top ) {
                                    var css = {position:'absolute', top:top+'px', left:left+'px'};
                                    if ( this.jqShim ) {
                                          this.jqShim.css(css);
                                    }
                                    this.jqContent.css(css);
                                    //alert(this.jqContent.html());
                              },
                              createOverlay : function () { return null; }

                        };
            
                        function setZindex(layer, index ) {
                              index = (index + 1) * 1000;
                              if ( layer.jqShim )
                                    layer.jqShim.css({zIndex : index});
                              ++index;
                              if ( layer.jqOverlay )
                                    //layer.jqOverlay.css({zIndex : index, opacity : .5});
                                    layer.jqOverlay.css({zIndex : index});
                              ++index;
                              layer.jqContent.css({zIndex : index});
                        }
                        function initLayer (layer,content) {
                              layer.layers = $.layerMgr._layers;
                              layer.layers.push(layer);
                              var index = layer.layers.length;
                              layer.index = index;
                              layer.jqShim = layer.createShimLayer();
                              layer.jqOverlay = layer.createOverlay();
                              layer.jqContent = $(content);
                              layer.onInit();
                              setZindex(layer,index);
                        };
                        var layer = $.extend({}, defaultLayer, options);
                        initLayer(layer,content);
                        return layer;
                  }
            };
            
      jQuery.extendLayer = function(defaultOpts,options) {
            var opts =$.extend({}, defaultOpts, options);
        opts.onOpenEvents = defaultOpts.onOpenEvents ? [].concat(defaultOpts.onOpenEvents): [];
            if ( options && options.onOpenEvents )
                  opts.onOpenEvents = opts.onOpenEvents.concat(options.onOpenEvents );
        opts.onCloseEvents = defaultOpts.onCloseEvents ? [].concat(defaultOpts.onCloseEvents): [];
            if ( options && options.onCloseEvents )
                  opts.onCloseEvents = opts.onCloseEvents.concat(options.onCloseEvents );
            return opts;
      };
      function fitHorizontal(x,w) {
          x -= $(document).scrollLeft(); 
          return x >= 0  && $(window).width()>= (x + w); 
    }
      function fitVertical(y,h) {
          y -= $(document).scrollTop(); 
          return y >= 0 && $(window).height()>= (y + h); 
    }

      /*
       * $(content).popupLayer(eventSrc,options)
       */
      $.fn.popupLayer = function(src,options) {
            var defaultPopuLayer = {
                  positionBy : 'source',
                  //source ref - left is same as bottomLeft
                  positionRef : 'bottomLeft',
                  targetRef : 'topLeft',
                  pointerLength : 0,
                  sourceMovable : false,
                  xOffset : 0,
                  yOffset : 0,
                  beforeOpenPopup : function () {},
                  targetLeftSrcLeftX : function(srcX,srcW,popupW){
                        this.srcHorPos = 'left';
                      var x = srcX + this.xOffset;
                      if ( fitHorizontal(x,popupW) ) return x;
                var x2 = srcX + srcW - popupW - this.xOffset;
                      if ( fitHorizontal(x2,popupW) ) {this.srcHorPos = 'right';return x2;}
                  return x; 
                  },                
                  targetLeftSrcRightX : function(srcX,srcW,popupW){
                        this.srcHorPos = 'left';
                      var x = srcX + srcW + this.xOffset;
                      if ( fitHorizontal(x,popupW) ) return x;
                      //try aligning right
                      var x2 = srcX + srcW - popupW - this.xOffset;
                      if ( fitHorizontal(x2,popupW) ) {this.srcHorPos = 'right';return x2;}
                  return x; 
                  },                
                  targetRightSrcLeftX : function(srcX,srcW,popupW){
                        this.srcHorPos = 'right';
                      var x = srcX - popupW + this.xOffset;
                      if ( fitHorizontal(x,popupW) ) return x;
                      //try aligning right
                      var x2 = srcX - this.xOffset;
                      if ( fitHorizontal(x2,popupW) ) {this.srcHorPos = 'left';return x2;}
                      //align to the edge of the window
                  return x; 
                  },                
                  targetRightSrcRightX : function(srcX,srcW,popupW){
                        this.srcHorPos = 'right';
                      var x = srcX + srcW - popupW + this.xOffset;
                      if ( fitHorizontal(x,popupW) ) return x;
                      //try aligning left
                      var x2 = srcX - this.xOffset;
                      if ( fitHorizontal(x2,popupW) ) {this.srcHorPos = 'left';return x2;}
                  return x; 
                  },                
                  targetTopSrcTopY : function(srcY,srcH,popupH){
                        this.srcVerPos = 'top';
                      var y = srcY + this.pointerLength + this.yOffset;
                      if ( fitVertical(y,popupH) ) return y;
                      //try aligning top
                      var y2 = srcY - popupH - this.pointerLength - this.yOffset;
                      if ( fitVertical(y2,popupH) ) {this.srcVerPos = 'bottom';return y2;};
                  return y; 
                  },                
                  targetTopSrcBottomY : function(srcY,srcH,popupH){
                        this.srcVerPos = 'top';
                      var y = srcY + srcH + this.pointerLength + this.yOffset;
                      if ( fitVertical(y,popupH) ) return y;
                      var y2 = srcY - popupH - this.pointerLength - this.yOffset;
                      if ( fitVertical(y2,popupH) ) {this.srcVerPos = 'bottom';return y2;}
                  return y; 
                  },                
                  targetBottomSrcTopY : function(srcY,srcH,popupH){
                        this.srcVerPos = 'bottom';
                      var y = srcY -popupH - this.pointerLength + this.yOffset;
                      if ( fitVertical(y,popupH) ) return y;
                      //try aligning bottom
                      var y2= srcY + srcH + this.pointerLength - this.yOffset;
                      if ( fitVertical(y2,popupH) ) {this.srcVerPos = 'top';return y2;};
                        return y;
                  },                
                  targetBottomSrcBottomY : function(srcY,srcH,popupH){
                        this.srcVerPos = 'bottom';
                      var y = srcY + srcH -popupH - this.pointerLength + this.yOffset;
                      if ( fitVertical(y,popupH) )return y;
                      var y2 = srcY + srcH + + this.pointerLength - this.yOffset;
                      if ( fitVertical(y2,popupH) ){this.srcVerPos = 'top';return y2;};
                  return y; 
                  },                
                  calculateX : function(srcX,srcW,popupW,srcAlign,targetAlign){
                        if ( targetAlign == 'left' ) {
                          if ( srcAlign == 'left' ) {
                              return this.targetLeftSrcLeftX(srcX,srcW,popupW);
                          } else {
                              return this.targetLeftSrcRightX(srcX,srcW,popupW);
                          }
                      } else {
                          if ( srcAlign == 'left' ) {
                              return this.targetRightSrcLeftX(srcX,srcW,popupW);
                          } else {
                              return this.targetRightSrcRightX(srcX,srcW,popupW);
                          }
                      }
                  },
                  calculateY : function(srcY,srcH,popupH,srcAlign,targetAlign){
                      if ( targetAlign == 'middle' || srcAlign == 'middle') {
                              this.srcVerPos = 'middle'
                        return y2 = srcY + srcH/2 - this.yOffset;
                      } if ( targetAlign == 'top' ) {
                          if ( srcAlign == 'top' ) {
                              return this.targetTopSrcTopY(srcY,srcH,popupH);
                          } else {
                              return this.targetTopSrcBottomY(srcY,srcH,popupH);
                          }
                      } else {
                          if ( srcAlign == 'top' ) {
                              return this.targetBottomSrcTopY(srcY,srcH,popupH);
                          } else {
                              return this.targetBottomSrcBottomY(srcY,srcH,popupH);
                          }
                      }
                  },
                  openPopup : function ( src ) {
                        this.src = src;
                        this.beforeOpenPopup();
                        if ( this.sourceMovable ) {
                            this.jqSrcContainer = $(this.src).parent();
                            this.jqSrcContainer.css({position:'relative'});
                            this.jqContent.remove().css({position:'absolute'});
                            this.jqSrcContainer.append(this.jqContent);
                            if ( this.jqShim ){
                                this.jqShim.remove().appendTo(this.jqSrcContainer);
                            }
                      } 
                      
                      //if ( this.jqContent.parent().length == 0 )
                      //    this.jqContent.appendTo('body');
                        src._layer = this;
                        //quickly show the container and its content so that browser can 
                        // calculate actual width and height
                        this.jqContent.show();
                        var w = this.jqContent.outerWidth();
                        var h = this.jqContent.outerHeight();
                        this.left=0;
                        this.top = 0;
                        this.offset(src,w,h);
                        this.jqContent.hide();
                        this._sizeTo(w,h);
                        this._moveTo(this.left,this.top);
                        this.open();
                  },
                  getVerPos : function (ref) {
                        if ( ref.indexOf('bottom') >= 0 ) return 'bottom';
                        if ( ref.indexOf('middle') >= 0 ) return 'middle';
                        return 'top';
                  },
                  getHorPos : function (ref) {
                        if ( ref.indexOf('Left') >= 0 ) return 'left';
                        return 'right';
                  },
                  offset : function(src,w,h) {
                              var jqSrc = $(src);
                              if ( this.positionBy == 'source' ) {
                                    if ( this.positionRef == 'left' ) this.positionRef = 'topLeft';
                                    this.verPos = this.getVerPos(this.positionRef);  
                                    this.horPos = this.getHorPos(this.positionRef);
                                    var offset = jqSrc.offset();
                                    this.left = this.calculateX(offset.left,jqSrc.outerWidth(),w,this.horPos,this.getHorPos(this.targetRef));
                                    this.top = this.calculateY(offset.top,jqSrc.outerHeight(),h,this.verPos,this.getVerPos(this.targetRef));
                                    if ( this.pointerLength > 0 ) {
                                          this.positionClass = '.' + this.srcVerPos + '-' + this.srcHorPos + '-arrow';;
                                          $(this.positionClass,this.jqContent).show();
                                    }
                                    if ( this.sourceMovable ) {
                                          offset = this.jqContent.parent().offset();
                                          this.left -= offset.left;
                                          this.top -= offset.top;
                                    }
                              } else {
                                    this.left = this.mX + this.xOffset;
                                    this.top = this.mY + this.yOffset;
                              }
                        },
                  _sizeTo : function ( w, h ) {
                        if ( this.jqShim ) {
                              var css = {width:w+'px', height:h+'px'};
                              this.jqShim.css(css);
                        }
                  }
            };
            var opts = $.extend({}, defaultPopuLayer, options);
            var popup = $.layerMgr.newLayer( this,opts );
            popup.openPopup(src);
            return this;
      };
      
      /*
       * $(content).popup(eventSrc,options)
       */
      var _singlePopupop = null;
      $.fn.popup = function (eventSrc,options) {
            var closePopup = function () {
                if ( _singlePopupop ) {
                    _singlePopupop.close();
                _singlePopupop = null;
                }
            }
            closePopup();
          var defaultPopup = {
                  hooverPopup : false,
                closeOnClick : true,
                  onOpenEvents : [
                        function () {
                        _singlePopupop = this;
                        if ( this.closeOnClick )
                            $(document).bind('click',closePopup);
                              if ( this.hooverPopup ) {
                                    //set delay on mouseleave
                                    this.jqContent.bind("mouseleave", closePopup);
                              }

                  }],
                  onCloseEvents : [
                        function () {
                    if ( this.closeOnClick )
                        $(document).unbind('click',closePopup);
                        if ( this.hooverPopup ) {
                            this.jqContent.unbind("mouseleave", closePopup);
                        }
                     _singlePopupop = null;
                  }]

            };
            var opts = $.extendLayer(defaultPopup,options);
            $(this).popupLayer(eventSrc,opts);
            return this;
      };

      /*
       * $(eventSrcs).attachHoverPopup(content,options)
       */
      $.fn.attachPopup = function(content,startevent,options) {
            // default configuration options
            var defaultPopup = {
                  delay: 150,
                  sensitivity: 5,
                  hooverPopup : true,
                  onCloseEvents : [
                        function () {
                        cleanPopup(this.src);
                  }],
                  onOpenEvents : [
                        function () {
                              var popup = this;
                              var src = this.src;
                              this.jqContent.bind("click", function(e) {
                                    popup.close();
                              })
                              .bind("mouseenter", function(e) {
                                    cleanPopup(src);
                              })
                              .bind("mouseleave", function(e) {
                                    //close popup but delay it just in case user move back to the src that trigger
                                    //of this popup
                                    hidePopup(src,'container mouseleave');
                              });
                              $('.closeBtn',popup.jqContent).bind('click',function (e) {
                              popup.close();
                          });

                        }]
            };
            var options = $.extendLayer(defaultPopup,options);
            
            var showPopup = function (content,src,opts) {
                  cleanPopup(src);
                  $(content).popup(src,opts);
            };
            var hidePopup = function(src,tag) {
                  if ( src._layer )
                        src._hoverPopup_t = setTimeout(function() { if( src._hoverPopup_t && src._layer)src._layer.close();cleanPopup(src);},options.delay);
            };
            var cleanPopup = function(src) {
                  if (src && src._hoverPopup_t) { clearTimeout(src._hoverPopup_t); src._hoverPopup_t=null;}
            };
            var jqContent = $(content);
            var events;
            if ( options.positionBy == 'mouse') {
                  events = ' mousemove mouseleave';
            } else {
                  events = ' mouseleave';
            }
            events = startevent + events;
            return this.each(function(){
                  var src = this;
                  $(src).bind(events,function(e) {
                        // next three lines copied from jQuery.hover, ignore children onMouseOver/onMouseOut
                if ( e.type != "click") {
                        var p = (e.type == "mouseenter" ? e.fromElement : e.toElement) || e.relatedTarget;
                        while ( p && p != this ) { try { p = p.parentNode; } catch(e) { p = this; } }
                        if ( p == this ) { return false; }
                        }
                        
                        // copy objects to be passed into t (required for event object to be passed in IE)
                        var ev = jQuery.extend({},e);

                        // else e.type == "onmouseover"
                        if (e.type == "mousemove") {
                              if ( src._layer ) {
                                    //move to new location
                                    var xDelta = e.pageX - src._layer.mX;
                                    var yDelta = e.pageY - src._layer.mY;
                                    if ( ( Math.abs(xDelta) > options.sensitivity) || (Math.abs(yDelta)  > options.sensitivity) ) {
                                          var offset = src._layer.jqContent.position();
                                          src._layer._moveTo((offset.left + xDelta),(offset.top + yDelta));
                                          src._layer.mX = e.pageX;
                                          src._layer.mY = e.pageY;
                                    }

                              }
                        } else if (e.type == "mouseenter" || e.type == "click") {
                              if ( src._layer ) {
                                    //popup already opend, clear the timer of the 
                                    //layer so that it does not close the layer
                                    cleanPopup(src);
                              } else {
                                    //delay opening of popup
                                    options.mX = e.pageX;
                                    options.mY = e.pageY;
                                    src._hoverPopup_t = setTimeout(function() { showPopup(content,src,options);},options.delay);
                              }

                        } else {
                              if ( src._layer ) {
                                    hidePopup(src,'src ml');
                              } else {
                                    cleanPopup(src);
                              }
                        }
                        return false;
                  });
            });
      };

      /*
       * $(eventSrcs).attachHoverPopup(content,options)
       */
      $.fn.attachHoverPopup = function(content,options) {
          this.attachPopup(content,"mouseenter",options);
          return this;
      };

      var defaultDialog = {
            center: true,
            draggable: true,
            sensitivity:1,
            onOpenEvents : [
                  function () {
                        var popup = this;
                        $('.close',popup.jqContent).click(function (e) {
                              e.preventDefault();
                              popup.close();
                        });
                        if ( popup.center ) {
                          var w = popup.jqContent.outerWidth();
                          var h = popup.jqContent.outerHeight();
                          var top = ($(window).height() -h )/2 + $(document).scrollTop();
                          top = Math.max(top,0);
                          var left = ($(window).width() -w )/2 + $(document).scrollLeft();
                          left = Math.max(left,0);
                          if ( this.jqShim)
                                this.jqShim.css({position:'absolute',top: top + 'px',left: left + 'px',width:w + 'px', height:h + 'px'});
                          this.jqContent.css({top:top + 'px', left:left + 'px'});
                    }

                        var dragOpts = {popup:this,
                              trackMouseMove :function( e ) {
                                  var pX = e.pageX;
                                  var pY = e.pageY;
                                    var xDelta = pX - this.mX;
                                    var yDelta = pY - this.mY;
                                    if ( ( Math.abs(xDelta) > this.sensitivity) || (Math.abs(yDelta)  > this.sensitivity) ) {
                                        xDelta = pX - this.oX;
                                        yDelta = pY - this.oY;
                                        var w = this.popup.jqContent.width();
                                        var h = this.popup.jqContent.height();
                                        var left = this.offset.left + xDelta;
                                        var dW = $(document).width() - 5; 
                                        if ( left + w >= dW )
                                            left = dW - w;
                                        if ( left < 0 )
                                            left = 0;
                                        var top = this.offset.top + yDelta;
                                        var dH = $(document).height() - 5;
                                        if ( top + h >= dH )
                                            top = dH - h;
                                        if ( top < 0 )
                                            top = 0;
                                          this.popup._moveTo(left,top);
                                          this.mX = e.pageX;
                                          this.mY = e.pageY;
                                    }

                              }
                        };
                        popup.jqContent.allowDrag('.dialogHeading',dragOpts);

                  }]
                  
      };
      $.fn.dialog = function (options) {
            var opts = $.extendLayer(defaultDialog,options);
            $(this).css({left:'-1000px'}).show();
            var h = $(this).outerHeight();
            $(this).css({height: h + 'px'}).hide();
            var dialog = $.layerMgr.newLayer( this,opts );
            dialog.open();
            return this;
      };
      
      
      /*
       * $(content).modal(options)
       */
      var _modalDialog = null;
      var defaultModal = {
            center: true,
            draggable: false,
            _modalOverlay : null,
            _modalShimLayer : null,
            createOverlay : function () { return this._modalOverlay; },
            createShimLayer : function() { 
                  if (_needShimLayer) {
                        this._modalShimLayer = $('<iframe src="javascript:false;" frameBorder=0 class="shim" style="display:none;top:expression(((document.documentElement.scrollTop || document.body.scrollTop) + Math.round(50 * (document.documentElement.offsetHeight || document.body.clientHeight) / 100)) + \'px\');">')
                                    .appendTo('body');
                }
                return this._modalShimLayer; 
            },
            onOpenEvents : [
                  function () {
                        var popup = this;
                        $('.close',popup.jqContent).click(function (e) {
                              e.preventDefault();
                              popup.close();
                        });
                    if ( this.center ) {
                          var w = this.jqContent.outerWidth();
                          var h = this.jqContent.outerHeight();
                            this.jqContent.css({marginLeft:'-' + (w/2) + 'px',marginTop: '-' + (h/2) + 'px'});
                            //alert(this.jqContent.parent().css("position"));
                            if ($.browser.msie ) {
                                if ( ($.browser.version < 7)) {
                                      var wHeight = $(document.body).height() + 'px';
                                      var wWidth = $(document.body).width() + 'px';
                                      if ( this._modalShimLayer )
                                    this._modalShimLayer.css({position:'absolute',height: h,width:w, left: '50%',marginLeft:'-' + (w/2) + 'px',marginTop: '-' + (h/2) + 'px'});
                            var element = this.jqContent[0];
                            if (element.style.setExpression) { // IE
                                element.style.setExpression('top',  
                                "((document.documentElement.scrollTop || document.body.scrollTop) + Math.round(50 * (document.documentElement.offsetHeight || document.body.clientHeight) / 100)) + \'px\'");
                                //element = this._modalShimLayer[0];
                                //element.style.setExpression('top',  
                                //"((document.documentElement.scrollTop || document.body.scrollTop) + Math.round(50 * (document.documentElement.offsetHeight || document.body.clientHeight) / 100)) + \'px\'");
                           }
                                  }
                                  if ( $.browser.version >= 7 && $.browser.version < 8 )
                                      this._modalOverlay.css({opacity:.5});
                        if ( ($.browser.version >= 8 )) {
                                      this.jqContent.css({top:'50%',left:'50%'});
                                  } else {
                                    this.jqContent.css({position: 'absolute'});
                                  }

                            }       
                      }

                  }],
            onCloseEvents : [
                  function () {
                     _modalDialog = null;
                  }]

                  
      };
      $.fn.modal = function (options) {
          if ( _modalDialog != null ) {
              _modalDialog.close();
              return this;
          } 
            if ( defaultModal._modalOverlay == null && defaultModal._modalShimLayer == null) {
                  if (_needShimLayer) {
                        /*defaultModal._modalShimLayer = $('<iframe src="javascript:false;" frameBorder=0 style="top:expression(((document.documentElement.scrollTop || document.body.scrollTop) + Math.round(50 * (document.documentElement.offsetHeight || document.body.clientHeight) / 100)) + \'px\');">')
                                    .hide()
                                    .appendTo('body');*/
                  } else {
                  defaultModal._modalOverlay = $('<div>').attr('id', 'modalOverlay').appendTo('body');
                  }

            }
            var opts = $.extendLayer(defaultModal,options);
            _modalDialog = $.layerMgr.newLayer( this,opts );
            _modalDialog.open();
            
            return this;
      };
    jQuery.closeModalDialog = function () {
      if ( _modalDialog  ) 
            _modalDialog.close();
    };

      $.fn.configureDynamicLoadModalDialog = function(options) {
            var opts = $.extendLayer({showModal : function() {$(this.content).modal()}},options);
            this.click( function(e) {
                  var content = $(opts.content);
                  if ( content.length == 0 ) {
                        $('body').loadContent(opts.url,null,function() {opts.onLoad();opts.showModal();},null);
                  } else {
                        opts.showModal();
                  }
                  e.preventDefault();
            });
      }
    var defaultBubbleHelp = {
              width:'320px',
                  delay: 350,
                  className : 'help-tip',
                  positionRef : 'topRight',
                  targetRef : 'bottomLeft',
                  xOffset : -20,
                  pointerLength : 14, 
                  onOpenEvents : [
                        function () {
                        var popup = this;
                        $('.closeBtn',popup.jqContent).click(function (e) {
                              //e.preventDefault();
                              popup.close();
                          });
                        }],
            onCloseEvents : [
                function () {
                    if ( this.positionClass )
                                $(this.positionClass).hide();
                    this.container.remove();
                    }],
                // overwriteing targetLeftSrcRightX of supperclass since right alligned help arrow needs extra 10 pixels shift to the left
                  targetLeftSrcRightX : function(srcX,srcW,popupW){
                        this.srcHorPos = 'left';
                      var x = srcX + srcW + this.xOffset;
                      if ( fitHorizontal(x,popupW) ) return x;
                      //try aligning right
                      var x2 = srcX + srcW - popupW - this.xOffset-10;
                      if ( fitHorizontal(x2,popupW) ) {this.srcHorPos = 'right';return x2;}
                  return x; 
                  },                
                xoffset : function(src,w,h) {
                        var jqSrc = $(src);
                        var sw = jqSrc.outerWidth();
                      var sh = jqSrc.outerHeight();

                  var offset = jqSrc.offset();
                  var horPos = 'right';
                  var verPos = 'bottom';
                        var left = offset.left + (sw/2) + this.xOffset;
                    if ($(window).width()-(left -16 - $(document).scrollLeft())> w) {
                        this.xOffset = -10;
                        horPos = 'left';
                        this.left = left - 16;
                    } else {
                        this.left = left-w+ (sw/2) + 9 + this.xOffset;
                        horPos = 'right';
                    }
                  var top = offset.top + this.yOffset;
                        if ( $(window).height()-(top + 32 -$(document).scrollTop()+sh)> h) {
                            this.top = top + 32;
                            verPos = 'top';
                        } else {
                            this.top = top - h - 9;
                            verPost = 'bottom';
                        }
                        

                        this.positionClass = '.' + verPos + '-' + horPos + '-arrow';;
                        $(this.positionClass,this.jqContent).show();
                        

                  },
            container : null
    };
      
      jQuery.extendBubbleHelp = function(options) {
            options = $.extendLayer(defaultBubbleHelp,options);
          options.container =
                  $('<div class="' + options.className + '"><div id="bubble"><table border="0" cellpadding="0" cellspacing="0" width="100%">' +
                        '<tr><td class="help-tip-rtl round-top-left">&nbsp;</td><td class="help-tip-rtm round-top-middle">&nbsp;</td><td class="help-tip-rtr round-top-right">&nbsp;</td></tr>' +
                        '<tr><td class="help-tip-rml round-middle-left">&nbsp;</td><td class="help-tip-rm round-middle">' +            
                        '<table style="float: left; display: inline;" border="0" cellpadding="0" cellspacing="0" width="100%">' +
                        '<tr><td align="left" valign="top" width="100%"><div class="popText"></div></td>' +
                        '<td style="padding-left: 3px;" valign="top" width="19px"><div class="closeBtn"></div></td></tr></table>' +
                        '</td><td class="help-tip-rmr round-middle-right">&nbsp;</td></tr>' +
                        '<tr><td class="help-tip-rbl round-bottom-left">&nbsp;</td><td class="help-tip-rbm round-bottom-middle" width="100%">&nbsp;</td><td class="help-tip-rbr round-bottom-right">&nbsp;</td></tr></table></div>' +
                        '<div class="top-left-arrow">&nbsp;</div><div class="top-right-arrow">&nbsp;</div><div class="bottom-left-arrow">&nbsp;</div><div class="bottom-right-arrow">&nbsp;</div><div class="middle-left-arrow">&nbsp;</div></div>').hide();
        return options;
    };
      
      $.fn.attachHoverBubbleHelp = function(content,options) {
            var opts = $.extendBubbleHelp(options);
            opts.beforeOpenPopup = function() {
                  $('.popText',opts.container).html(content);
                  opts.container.css({width:opts.width}).appendTo('body');
            }
            this.each(function() { $(this).attachHoverPopup(opts.container[0],opts)});
    };
      $.fn.attachStaticBubbleHelp = function(content,options) {
            var opts = $.extendBubbleHelp(options);
            opts.closeOnClick =false;
            opts.beforeOpenPopup = function() {
                  $('.popText',this.container).html(content);
                  this.container.css({width:opts.width}).appendTo('body');
            }
        opts.container.popup(this,opts);
            return this;
    };
    
    var dynamicBubbleHelp=null;
      $.fn.attachDynamicHoverBubbleHelp = function(content,options) {
            if (!dynamicBubbleHelp)
                  dynamicBubbleHelp = $.extendBubbleHelp(options);
            dynamicBubbleHelp.beforeOpenPopup = function() {
                  $('.popText',dynamicBubbleHelp.container).html($(this.src).data("bubbleHelp"));
                  dynamicBubbleHelp.container.css({width:dynamicBubbleHelp.width}).appendTo('body');
            }
          this.each(function() {
            $(this).data("bubbleHelp", content);
            $(this).attachHoverPopup(dynamicBubbleHelp.container[0],dynamicBubbleHelp)
            if (this._layer) // if layer visible, update its content
                        $('.popText', this._layer.jqContent).html(content);
          });
    };
   

      var fieldHelpPopup = null;    
      var inputFieldsOptions = {className :'tipField', charCountClass:'charCount'};
      $.fn.inputFields = function() {
            options = inputFieldsOptions;
            if( options.className ) {
                  $(window).unload(function() {$('.' + options.className).val('');});
                $('form').submit(function() {$(this).find('.' + options.className).val('');});
            }
            this.each(function() {
            var $this = $(this);
                  var fieldTip = $this.attr('title');
                  if ( fieldTip) {
                        if ($.browser.msie && $this.val() == fieldTip) {
                        $this.val('');
                        }
                        if ($this.val() == '') {
                        $this.addClass(options.className);
                              $this.val(fieldTip);
                        }
                  }

            var keycount = function(i) {
                        var max = $(this).attr('maxlength');
                        var val = $(this).attr('value');
                        var cur = 0;
                        if(val)
                        cur = val.length;
                        var left = max-cur;
                        var c = $(this).next("span").find(".counter");
                        c.text(left.toString());
                        if(left <= 10)
                        c.css("background","#F4F379");
                        else
                        c.css("background","none");
            };
                  $this.focus(function() {
                        var $$this = $(this);
                  if ($$this.is('.' + options.className)) {
                        $$this.val('').removeClass(options.className);
                  }
                        if ( fieldHelpPopup ) {
                              fieldHelpPopup.close();
                              fieldHelpPopup = null;
                        }
                        var fieldHelp = $$this.attr('fieldHelp');
                        if ( fieldHelp ) {
                              fieldHelpPopup = $.extendBubbleHelp({className : 'input-help-tip'});
                              fieldHelpPopup.positionRef = 'middleRight';
                              fieldHelpPopup.targetRef = 'topLeft';
                              fieldHelpPopup.xOffset = 20;
                              fieldHelpPopup.yOffset = 15;
                              fieldHelpPopup.closeOnClick =false;
                    fieldHelpPopup.sourceMovable = true;
                              fieldHelpPopup.beforeOpenPopup = function() {
                                    $('.popText',this.container).html(fieldHelp);
                                    this.container.css({width:'420px'});
                              };
                              fieldHelpPopup.container.popup(this,fieldHelpPopup);
                              fieldHelpPopup = this._layer;
                        }
                        if ($$this.is('.' + options.charCountClass)) {
                        var max = $$this.attr('maxlength');
                              if ( max ) {
                              var val = $$this.val();
                              var cur = 0;
                              if(val) // value="", or no value at all will cause an error
                                          cur = val.length;
                              var left = max-cur;
                              $$this.after("<span><span class='counter'>"+ left.toString()+"</span> characters remaining</span>");
                              // Style as desired
                              var c = $$this.next("span").find(".counter");
                              c.css("margin-left","10px");
                              c.css("padding", "0 3px 0 3px")
                              c.css("border", "1px solid #ccc")
                              if(left <= 10)
                                    c.css("background","#F4F379");
                              else
                                    c.css("background","none");
                                    $$this.keyup(keycount);

                              }

                        }
                  });
    
                  $this.blur(function() {
                        if ( fieldHelpPopup ) {
                              fieldHelpPopup.close();
                              fieldHelpPopup = null;
                        }
                        var $$this = $(this);
                  if ($$this.val() == '') {
                        $$this.addClass(options.className);
                        $$this.val(fieldTip);
                        }
                        if ($$this.is('.' + options.charCountClass)) {
                              $$this.next('span').remove();
                        }
                  });
                  //return this;
            });
            return this;

      };
      $.fn.resetInputFields = function() {
          this.find('.' + inputFieldsOptions.className).val('').removeClass(inputFieldsOptions.className);
   };
    var MonthDays = new Array(31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31);
    function GetDaysCountForMonth(SomeYear, SomeMonth) {
        return ((SomeMonth == 1) && ((SomeYear % 400 == 0) || ((SomeYear % 4 == 0) && (SomeYear % 100 != 0)))) ? 29 : MonthDays[SomeMonth];
    } 
      var defaultCalendar = {
              init: false,
                setCalendarContent:function(y,m) {
                    var calPopup = this;
                    var nextMonthName = MonthNames[(m < 11 ? m+1 :0)];
                    var monthName = MonthNames[m];
                    var prevMonthName = MonthNames[(m > 0 ? m -1 :11)];
                    var dt = new Date(y, m, 1);
                    this.firstDay = dt.getDay();
                    this.dayCount = GetDaysCountForMonth(y,m);
                    this.displayedYear = y;
                    this.displayedMonth = m;

                    var HTML = '<table class="calendar" cellspacing="0" cellpadding="1"><tr class="caltitle"><td>';
                    if ( y > this.minYear || m > 0 ) {
                          HTML += '<a id="calPrev" href="#"><img src="' + PrevURL + '" title="' + prevMonthName + '" border="0"></a>';
                    } else {
                          HTML += '&nbsp;';
                    }
                    HTML += '</td><td nowrap align="center" colspan="5">' + monthName + '&nbsp;' + y + '</td><td>';
                    if ( y < this.maxYear || m < 11 ) {
                    HTML += '<a id="calNext" href="#"><img src="' + NextURL + '" title="' + nextMonthName + '" border="0"></a>';
                } else {
                          HTML += '&nbsp;';
                }
                HTML += '</td></tr><tr>';
                for (var w=0;w<7;w++) 
                    HTML += '<td class="calhd">' + WeekDays[w] + '</td>';
                    HTML += '</tr>';
                    var sameMonthAndYear = this.displayedYear == this.valueYear && this.displayedMonth == this.valueMonth;
                    var todayMonthAndYear = this.displayedYear == this.todayYear && this.displayedMonth == this.todayMonth;
                    for (var j=0;j< 6;j++) {
                          HTML += '<tr>';
                          for (var i=1;i<=7;i++) {
                          var Day = (j * 7) + (i - this.firstDay);
                          if ((Day >= 1) && (Day <= this.dayCount)) {
                                      var cls = 'calcell';
                                if ( sameMonthAndYear && (Day == this.valueDay)) {
                                            cls += " calseldate";
                                }
                                if ( todayMonthAndYear && (Day == this.todayDay)){
                                              cls += " caltoday";
                                }   
                                HTML += '<td align="center" class="' + cls + '">' + Day + '</td>';
                          }
                          else HTML += '<td>&nbsp;</td>'; 
                          }
                          HTML += '</tr>';
                    }
                    HTML += '</table>';
                this.calendarPopup = $('#CalendarPopup');
                if ( this.calendarPopup.length == 0 ) {
                    $('<div id="CalendarPopup" class="popupWindow"></div>').appendTo('body');
                    this.calendarPopup = $('#CalendarPopup');
                }
                    this.calendarPopup.html(HTML);
                    this.calendarPopup.find('#calPrev').click(function(e){calPopup.previous();return false;});
                    this.calendarPopup.find('#calNext').click(function(e){calPopup.next();return false;});
                    this.calendarPopup.find('.calcell').hoverClass('calhover').click(function(e){calPopup.select(this);return false;});
              },
              next : function() {
                    var m = this.displayedMonth;
                    var y = this.displayedYear;
                    if ( m < 11 ) {
                        m++;
                    } else {
                          m = 0;
                          y++;
                    }
                    this.setCalendarContent(y,m);
              },
              previous : function() {
                    var m = this.displayedMonth;
                    var y = this.displayedYear;
                    if ( m > 0 ) {
                          m--;
                    } else {
                          m = 11;
                          y--;
                    }
                    this.setCalendarContent(y,m);
              },
              select : function(src) {
                  //set the year to be the same as the calendar year
                    for(var i = 0;i<this.yearElement.options.length;++i){
                          if(this.yearElement.options[i].value == this.displayedYear){
                                this.yearElement.selectedIndex = i;
                                break;
                          }
                    }
            
                    var m = "" + this.displayedMonth;
                    for(var i = 0;i<this.monthElement.options.length;++i){
                          if(this.monthElement.options[i].value == m){
                                this.monthElement.selectedIndex = i;
                                break;
                          }
                    }
                    this.monthChanged();
                    var day = $(src).html().trim();
                    for(var i = 0;i<this.dayElement.options.length;++i){
                          if(this.dayElement.options[i].value == day){
                                this.dayElement.selectedIndex = i;
                                break;
                          }   
                    }
                    $(this.src).trigger("change");
                    this.onChange();
                this.close();
            
              },  
              yearChanged : function (elm) {
                    this.monthChanged(elm);
              },
              monthChanged : function (elm) {
                    if ( this.monthElement.options[this.monthElement.selectedIndex].value != '' && 
                          this.yearElement.options[this.yearElement.selectedIndex].value != '' ) {
                          this.setCurrentValues();
                          var newDays = GetDaysCountForMonth(this.valueYear,this.valueMonth);
                          var dayList = this.dayElement;
                          var dayInc = 1;
                          if ( dayList.options[0].value == '') {
                              dayInc = 0;
                              newDays++;
                          }
                          if ( newDays != dayList.length ) {
                                var oldSize = dayList.length;
                                for (var k=Math.min(newDays,oldSize);k<Math.max(newDays,oldSize);k++) {
                                          (k >= newDays) ? dayList.options[newDays] = null : dayList.options[k] = new Option(dayInc + k, dayInc + k);
                                }
                                var dayPick = Math.min(this.valueDay, newDays);
                                if ( dayList.options[0].value != '')
                                      dayPick--;
                                if ( dayPick >= dayList.length)
                                    dayPick = dayList.length-1;
                                dayList.options[dayPick].selected = true;
                          }
                          if ( elm )
                              this.onChange();
                    }   
          },
              setCurrentValues : function(t) {
                    var Today = new Date();
                    var yr = this.yearElement.options[this.yearElement.selectedIndex].value;
                    if ( yr == '' )
                          yr = this.yearElement.options[1].value;
                    this.valueYear = parseInt(yr,10);
                    if ( this.monthElement.options.length > 12 ) {
                          this.valueMonth = this.monthElement.selectedIndex > 0 ? this.monthElement.selectedIndex - 1 : Today.getMonth();
                    } else {
                          this.valueMonth = this.monthElement.selectedIndex;
                    }
                    if ( this.dayElement.options[0].value == '') {
                          this.valueDay = this.dayElement.selectedIndex > 0 ? this.dayElement.selectedIndex : Today.getDate();
                    } else {
                          this.valueDay = this.dayElement.selectedIndex + 1;
                    }
              },
              onOpen : function () {
                    this.setCurrentValues();
                  this.setCalendarContent(this.valueYear,this.valueMonth);
                  this.open(true);
              },
              onChange : function () {
              }

          };      
      $.fn.calendar = function(options) {
        this.each(function() {
              var settings = $.extend({},defaultCalendar,options);
                settings.name = $(this).attr('id');  
              settings.dayElement = $('#' + settings.name +'Day')[0];
              settings.monthElement = $('#' + settings.name +'Month')[0];
              settings.yearElement =  $('#' + settings.name +'Year')[0];
              var yr = settings.yearElement.options[0].value;
              if ( yr == "" )
                  yr = settings.yearElement.options[1].value;
              settings.minYear = parseInt(yr,10);
                  //alert(settings.minYear);
              yr = settings.yearElement.options[settings.yearElement.length-1].value;
              settings.maxYear = parseInt(yr,10);
                settings.monthChanged(); //settings.monthElement);
                var Today = new Date();
                settings.todayYear = Today.getFullYear();
                settings.todayMonth = Today.getMonth();
                settings.todayDay = Today.getDate();
                settings.setCurrentValues();
                settings.setCalendarContent(settings.valueYear,settings.valueMonth);
                //**
                $(settings.yearElement).bind('change',function(e){settings.yearChanged(this);}); 
              $(settings.monthElement).bind('change',function(e){settings.monthChanged(this);}); 
              $(settings.dayElement).bind('change',function(e){settings.onChange();}); 
            settings.onChange();

                $(this).show().attachPopup(settings.calendarPopup,"click",settings);

          });
          return this;
    };
      
})(jQuery);

var favorites = function() {
            this.domain = 'kijiji.com'; // default value
            this.panelDisplayed = true;
            
            function setDomain(newDomain) {
                  this.domain = newDomain;
            }
            
            function setPanelDisplayed(newPanelDisplayed) {
                  this.panelDisplayed = newPanelDisplayed;
            }
            
            function getPanelDisplayed() {
                  return this.panelDisplayed;
            }
            
            function positionPanel(){
                  var panelHeight = $("#myFavorites-panel").outerHeight(true);;
                  var panelWidth =$("#myFavorites-panel").outerWidth(true);
                  var topPosition = document["documentElement"].clientHeight + document["documentElement"].scrollTop - panelHeight;
                  var leftPosition = document["documentElement"].clientWidth + document["documentElement"].scrollLeft - panelWidth;
                  $("#myFavorites-panel").css("top",topPosition+"px");
                  $("#myFavorites-panel").css("left",leftPosition+"px");
            }
            
            //dirty flag shows whether client-side storage is out of date
            function refreshPanel(dirtyFlag){
                  if(getFavoriteAdIds().length == 0){
                        $("#myFavorites-panel").fadeOut('fast');
                  }
                  else{
                  panelRetrievalAction(dirtyFlag, function(panelData){
                        showPanel(panelData);
                        
                        //fix some css-styles for Internet Explorer
                        var liWidth = $("#myFavorites-panel li").outerWidth(true);
                        var ulSize = $("#myFavorites-panel ul").children("li").size();
                        var paddingSum = (parseInt($("#myFavorites-panel ul").css("padding-right"))) + (parseInt($("#myFavorites-panel form").css("padding-left")));     
                        var panelWidth = paddingSum + liWidth*ulSize;
                        var panelElement = $("#myFavorites-panel-inner").not(".closed");

                        if($.browser.msie && parseInt(jQuery.browser.version)=="6.0"){
                              if(!panelElement.hasClass('closed'))
                                    panelElement = $("#myFavorites-panel");
                              
                              $("#myFavorites-panel-tab").width($("#myFavorites-panel-tab h2").width());
                        }
                        
                        var panelTabWidth = $("#myFavorites-panel-tab").outerWidth(true);
                        
                        // Fix for Firefox 2
                        if($.browser.mozilla  && jQuery.browser.version.substr(0,3)=="1.8")
                              panelTabWidth += 35;
                        
                        if(!$("#myFavorites-panel-inner").hasClass('closed'))
                              panelTabWidth += 11;
                        
                        if(panelWidth && (panelWidth > panelTabWidth))
                              panelElement.width(panelWidth);
                        else
                              panelElement.width(panelTabWidth);
                  
                        /* Append Shim Layer and fix position for IE6 */
                        if (jQuery.browser.msie && parseInt(jQuery.browser.version)<=6) {
                              var shimLayer = $('<iframe class="shim-layer">');
                              shimLayer.height(panelElement.height());
                               
                              if(! $('#myFavorites-panel-inner').hasClass('closed'))
                                    shimLayer.css({ width:  panelElement.width()+9, left: '-11px'});
                              
                               shimLayer.appendTo('#myFavorites-panel');
                               positionPanel();
                        }
                  });   
            }
            }
            
            function panelRetrievalAction(dirtyFlag, callbackForPanelData){
                  panelData = getFlashValue("panelData");
                  if(panelData == null || dirtyFlag.panelDataDirty){
                        // add timestamp as parameter in order to deal with IE ajax request caching problems!
                        $.get('/c-FavoritesPanel',{opened: getPanelPreferenceOpen(), nocache: new Date().getTime()} , function(data){
                              if (data.indexOf("myFavorites-panel-inner") != -1) {
                                    writeFlashValue("panelData",data);
                                    writeFlashValue("panelTimestamp",new Date().getTime());
                                    callbackForPanelData(data);
                              }
                        });
                  } else {
                        callbackForPanelData(panelData);
                  }
            }
            
            function panelDataExpired(){
                  var panelDataTimestamp = getFlashValue("panelTimestamp");                 
                  //panelData should always be refreshed after 1 day 
                  if( new Date().getTime() - panelDataTimestamp > 86400000)               
                        return true;
                  return false;             
          }
            
            function hidePanel(){
                  writeFlashValue("panelOpen" , "false");   
                  refreshPanel({panelDataDirty:true});
            }
            
            function openPanel(){
                  writeFlashValue("panelOpen" , "true");    
                  refreshPanel({panelDataDirty:true});
            }     
            
            function getPanelPreferenceOpen(){
                  if(getFlashValue("panelOpen") == "true")
                        return true;
                  //default preference is closed
                  if(getFlashValue("panelOpen")  == null)
                        return false;
                  if(getFlashValue("panelOpen") == "false")
                        return false;
            }
            
            function firstPanelVisit(){               
                  return getFlashValue("panelOpen") == null;
            }           
            
            function showPanel(data){
                  $('#myFavorites-panel').show().html(data);
            }
            
            function flashAvailable(){    
                  //feature is supported since flashplayer version 9
                  if(swfobject.getFlashPlayerVersion().major == (0 || 6 || 7 || 8) )
                        return false;
                  return true;
            }                   
         
         function getFlashValue(key){
                  var obj = swfobject.getObjectById("flashCookie");                                                        
                  if (obj && typeof obj.getFlashCookieValue != "undefined") {
                        if (key == "panelData" || key == "panelTimestamp") {
                              if (IsNC2_On) {
                                    var indx = this.domain.indexOf('.');
                                    key += "-" + (indx>=0 ? this.domain.substring(indx+1) : this.domain);
                              } else key += "-" + this.domain;
                        }
                        return obj.getFlashCookieValue(key);    
                  }
                  else throw "FlashCookie no Access";
          } 
          
          function writeFlashValue(key,value){                      
            var obj = swfobject.getObjectById("flashCookie");                                         
                  if (obj && typeof obj.setFlashCookieValue != "undefined") {
                        if (key == "panelData" || key == "panelTimestamp") {
                              if (IsNC2_On) {
                                    var indx = this.domain.indexOf('.');
                                    key += "-" + (indx>=0 ? this.domain.substring(indx+1) : this.domain);
                              } else key += "-" + this.domain;
                        }
                        obj.setFlashCookieValue(key,value);
                  }
                  else  throw "FlashCookie no Access";
          }   
          
      function panelActivated(){
            return getPanelDisplayed() && flashAvailable();
      } 
          
                  function getFavoriteAdIds(){
                        var adIds = new Array(), savedAds="";
                        if (IsNC2_On) {
                              // remove starting snippet which is the key (saved-ad-ids sit right to '.*?saved_s=')
                              savedAds=readCookie("ad_ids_s").replace(/.*?saved_s=/,"");
                        } else {
                              // remove starting snippet which is the key (saved-ad-ids sit right to '.*?saved=')
                              savedAds=readCookie("ad_ids").replace(/.*?saved=/,"");
                        }
                        // get rid of timestamps (pattern :3333333 or pattern #333333)
			if(savedAds.indexOf(":") >=0)
                              savedAds=savedAds.replace(/:\d+/g,"");
                        else if(savedAds.indexOf("#") >=0)                              
                               savedAds=savedAds.replace(/#\d+/g,"");
                        if(savedAds!="")
                              adIds =  savedAds.split("&");
                        return adIds;
                  }

                  
                  //TODO: check if there is not already a common lib-function for reading cookie values
                  function readCookie(n) {
                        var cookiecontent = new String();
                        if(document.cookie.length > 0) {
                              var cookiename = n+ '=';
                              var cookiebegin = document.cookie.indexOf(cookiename);
                              var cookieend = 0;
                        if(cookiebegin > -1) {
                              cookiebegin += cookiename.length;
                              cookieend = document.cookie.indexOf(";",cookiebegin);
                              if(cookieend < cookiebegin) { cookieend = document.cookie.length; }
                                    cookiecontent = document.cookie.substring(cookiebegin,cookieend);
                              }
                        }
                        return unescape(cookiecontent);
                  }
                

      //PUBLIC API, visible to clients
      return{

            init:function(domain, panelDisplayed){
                  setDomain(domain);
                  setPanelDisplayed(panelDisplayed);
                  
                  if(getFavoriteAdIds().length == 0)
                        panelDisplayed = false;
                  
                  if (panelDisplayed) {
                        //no panel for non flash, all further methods for panel retrieval aren't checking flash availability again, because 
                        //they cannot be reached (no callbacks active) and thus are protected by non-flash behaviour.       
                        if(flashAvailable()){
                        
                              if(panelDataExpired()){
                                    //sync-data (active ads could have gotten inactive by now)
                                    $.get('/c-SaveAd', {Action:'sync'},function(data){
                                          refreshPanel( {panelDataDirty:true} );
                                    });
                              }
                              else {
                                    refreshPanel( {panelDataDirty:false} );
                              }
                        }
                  }
                  if (this.onPostInit)    this.onPostInit();
            },

            addFavorite: function(adId, addSuccessCallback){      
                  if(panelActivated())
                        //panel not initialized yet, then user should be shown the panel 
                        if(firstPanelVisit())
                              writeFlashValue("panelOpen" , "true");    
                        
                  // use post to write data...
                  $.post('/c-SaveAd', {AdId:adId ,Action:'add',AdActivityType:'ViewAd'},function(data){
                        if (data == 'ok') {
                              addSuccessCallback(adId);
                              if (panelActivated()) {
                                    refreshPanel({panelDataDirty:true});
                              }
                        } else if (data.indexOf('pagestatusajax') != -1) {
                              // show error message - TODO: better way to see of an error message needs to be shown.
                              $('#pagestatus_new').css("display","");
                              $('#pagestatus_new').html(data);
                              // go to top of the page in case an error is displayed!
                              window.location.href = "#top";
                        }
            });
            },      
            
            removeFavorite: function(adId, removeSuccessCallback){
                  if(panelActivated())
                        //panel not initialized yet, then user should be shown the panel 
                        if(firstPanelVisit() && panelActivated())
                              writeFlashValue("panelOpen" , "true");
                              
                  // use post to remove ad
                  $.post('/c-SaveAd', {AdId:adId ,Action:'remove',AdActivityType:'ViewAd'},function(data){
                        if (data == 'ok') {
                              removeSuccessCallback(adId);
                              if (panelActivated()) {
                                    refreshPanel({panelDataDirty:true});
                              }
                        }
            });    
            },
    
          isFavorite: function(adId){
                        var isSaved=false;
                        $.each(getFavoriteAdIds(),function(i,savedAdId){
                              if(adId == savedAdId)
                                    isSaved=true;
                        });               
                        return isSaved;
          },      
            
            openPanel: function(){
                  openPanel();
            },
            
            flashAvailable: function(){
                  return flashAvailable();
            },
            
            closePanel: function(){
                  hidePanel();
            },
            
            positionPanel: function(){
                  positionPanel();
            },
            
            refreshPanelStorage: function(){
                  panelRetrievalAction({panelDataDirty:true}, function(panelData){});                       
            }
      };
            
}(); 

$(document).ready(function() {
      $('input:text,textarea').inputFields();
});

/******** Debugging functions *********/
function printProps(obj) {
      var res = "";
      for(var prop in obj) {
            if (typeof(obj[prop])!="function")
                  res += prop+"="+obj[prop]+"\t";
      }
      alert(res)
}
function printFuncs(obj) {
      var res = "";
      for(var prop in obj) {
            if (typeof(obj[prop])=="function")
                  res += prop+"\n";
      }
      alert(res)
}
/************ Drop-down multi-level menu **************
      zindex = Z-index of the menu
      shadow = 0-none, 1-quick, 2-nice shadow
      fadeIn = fade in of menu in msec
      fadeOut = fade out of menu in msec
*/
(function($){  
      $.fn.kjmenu_makeMenu = function(options) {
            var defaults = {
                  zindex:100,
                  shadow: 2,
                  fadeIn:0,
                  fadeOut:0,
                  indentPixels:15,
                  maxitems:22
            };  
            var options = $.extend(defaults, options);  

            return this.each(function() {
                  $(this).attr("kjmenu", this.id);
                  this.menu_root = {items:new Array(),id:this.id,opt:options,offset:0};
                  this.menu_root.root = this.menu_root;
                  if (options.data)
                        $(this.menu_root).kjmenu_fromLocString(options.data);                   

                  // building Ruler hidden container (to measure string width in pixels)
                  var rhtml="<span id='kjr_"+this.id+"' style='position:absolute;top:0px;left:0px;visibility:hidden;display:inline-block'><span id='kjr_cell_"+this.id+"' class='kjmenu_item kjmenu_over kjmenu_over_clickable' style='display:inline-block;white-space:nowrap'>test</span></span>";
                  $(document.body).append(rhtml);
                  this.menu_root.ruler = $('#kjr_'+this.id);
                  this.menu_root.ruler_cell = $('#kjr_cell_'+this.id);

                  $(document).bind("mouseover",this.menu_root,function(e) {
                        if (e.data.el) {
                              var mitem = e.data;
                              if (!mitem.opt.timer) {
                                    mitem.opt.timer = setTimeout(function() {
                                          mitem.opt.timer = null;
                                          $(mitem).kjmenu_remove(true);
                                    }, 100);
                              }
                        }
                  });
                  $(this).bind("mouseover",function(){
                        var root = this.menu_root;
                        if (root.opt.timer) {
                              clearTimeout(root.opt.timer);
                              root.opt.timer=null;
                        }
                        if (!root.el) {
                              $("[kjmenu]").each(function() {
                                    $(this.menu_root).kjmenu_remove(true);
                              });
                              $(this).kjmenu_create(root);
                        }
                        return false;
                  });
            });  
      }; 
      $.fn.extend({
            /** Parses location data string:
                  parent id,own id,name,value,clickable,indentation 
            */
            kjmenu_fromLocString : function (data) {
                  var ritem = this.get(0);
                  var map = new Array(), mitem;
                  var items = data.split("\004"),pitem;
                  for (var i=0; i<items.length; i++) {
                        if (items[i].length>0) {
                              var parts = items[i].split("\003");
                              mitem = {
                                    pid:parts[0], id:parts[1], name:parts[2], value:parts[3], 
                                    clickable:typeof(parts[4])!="undefined" ? parts[4]-0 : 1, 
                                    style:typeof(parts[5])!="undefined" && (parts[5]-0>0) ? "padding-left:"+((parts[5]-0)*ritem.opt.indentPixels)+"px" : null, 
                                    root:ritem,offset:0,parent:ritem
                              };
                              map[mitem.id] = mitem;
                              if (!mitem.pid)
                                    ritem.items[ritem.items.length]=mitem;
                        }
                  }
                  for(var key in map) {
                        mitem = map[key];
                        if (mitem.pid) {
                              pitem = map[mitem.pid];
                              if (pitem) {
                                    if (!pitem.items) pitem.items = new Array();
                                    pitem.items[pitem.items.length]=mitem;
                                    mitem.parent=pitem;     
                              }
                        }
                  }
            }
            ,kjmenu_create : function (mitem) {
                  if (mitem.el)     return;

                  // calculate how many items will fit vertically
                  var pdiv = this;
                  var ypos = !mitem.parent ? pdiv.offset().top+pdiv.height() : 0;
                  var itemheight = mitem.root.ruler.height();
                  mitem.maxitems = Math.min(mitem.root.opt.maxitems, Math.max(3, Math.floor(($(window).height()-ypos-60)/itemheight)));

                  $(document.body).append($().kjmenu_mainHTML(mitem));
                  mitem.el = $('#kjm_'+mitem.id);
                  mitem.el.css({zIndex:mitem.root.opt.zindex++});
                  mitem.el.kjmenu=mitem;
                  
                  // filling items
                  $(mitem).kjmenu_update(mitem);

                  // positioning menu
                  if (!mitem.parent) {
                        var xpos = pdiv.offset().left;
                        xpos = Math.min($(window).width()-mitem.el.width()-5, xpos);
                        mitem.el.css({top:(pdiv.offset().top+pdiv.height()), left:xpos});
                  } else {
                        var pardiv = mitem.parent.el;
                        var ypos = pdiv.offset().top-4;
                        ypos = (ypos+mitem.el.height()<$(window).height()) ? ypos : Math.max(0,ypos+pdiv.height()+4-mitem.el.height());
                        var xpos = pardiv.offset().left+pardiv.width()+1;
                        if($(window).width()-mitem.el.width()-5 < xpos)
                        	xpos = pardiv.offset().left-mitem.el.width()-1;
                        mitem.el.css({top:ypos, left:xpos});
                  }
                  // instant show or fadein
                  if (mitem.root.opt.fadeIn>0)
                        mitem.el.fadeIn(mitem.root.opt.fadeIn);
                  else
                        mitem.el.show();

                  //if IE6 we add iframe
                  if($.browser.msie ){
                        mitem.el.append("<iframe style='display:block;position:absolute;top:0;left:0;z-index:-1;filter:mask();width:"+(mitem.el.width())+";height:"+(mitem.el.height())+"'/>");
                  } else {
                        // add shadow (if needed)
                        if (mitem.root.opt.shadow)
                              mitem.el.kjmenu_dropShadow({type:mitem.root.opt.shadow});
                  }

                  // binding mouse events for menu body
                  mitem.el.bind("mousedown", function () {return false;});
                  mitem.el.bind("mouseover", mitem, function (e) {
                        var mitem = e.data;
                        if (mitem.root.opt.timer) {
                              clearTimeout(mitem.root.opt.timer);
                              mitem.root.opt.timer=null;
                        }
                        return false;
                  });

                  // binding events for top scroll button
                  $("#kjm_tblup_"+mitem.id).each(function(i, el){
                        $(this).bind("click dblclick", el, function () {
                              var offset=Math.max(0, mitem.offset-mitem.maxitems);
                              if (offset != mitem.offset) {
                                    mitem.offset = offset;
                                    $(mitem).kjmenu_update(mitem);
                              }
                        });
                        $(this).bind("mouseover", el, function () {
                              $("#kjt_"+mitem.id+" td.kjmenu_over").removeClass("kjmenu_over").removeClass("kjmenu_over_expand").removeClass("kjmenu_over_clickable");
                              if (mitem.expanded_child)
                                    $(mitem.expanded_child).kjmenu_remove(true);
                              mitem.expanded_child=null;
                        });
                  });
                  
                  // binding events for bottom scroll button
                  $("#kjm_tbldown_"+mitem.id).each(function(i, el){
                        $(this).bind("click dblclick", el, function () {
                              var offset=Math.min(mitem.offset+mitem.maxitems, mitem.items.length-mitem.maxitems);
                              if (offset != mitem.offset) {
                                    mitem.offset = offset;
                                    $(mitem).kjmenu_update(mitem);
                              }
                        });
                        $(this).bind("mouseover", el, function () {
                              $("#kjt_"+mitem.id+" td.kjmenu_over").removeClass("kjmenu_over").removeClass("kjmenu_over_expand").removeClass("kjmenu_over_clickable");
                              if (mitem.expanded_child)
                                    $(mitem.expanded_child).kjmenu_remove(true);
                              mitem.expanded_child=null;
                        });
                  });
            }
            ,kjmenu_mainHTML : function (mitem) {
                  // measure max width in pixels
                  var width = $().kjmenu_measure(mitem)+mitem.root.opt.indentPixels+6;
                                    
                  // building html for whole menu (without items inside yet)                                
                  var html = new Array(), item;
                  html[html.length]="<div id='kjm_"+mitem.id+"' class='kjmenu_div' style='display:none;top:0px;left:0px'>";
                  if (mitem.items.length>mitem.maxitems) {
                        // top scroll button
                        html[html.length]="<table id='kjm_tblup_"+mitem.id+"' height=26 width="+width+" cellpadding=0 cellspacing=0 border=0><tr>"
                        html[html.length]="<td class='kjmenu_fillbox kjmenu_scroll' width=6 style='background-position:0px 0px'></td>";
                        html[html.length]="<td class='kjmenu_fillbox kjmenu_scroll' align=center style='background-position:0px -26px;background-repeat:repeat-x'>";
                        html[html.length]="<div class='kjmenu_fillbox kjmenu_scroll' style='width:13px;height:26px;background-position:-32px 0px'></div></td>";
                        html[html.length]="<td class='kjmenu_fillbox kjmenu_scroll' width=6 style='background-position:-45px 0px'></td>";
                        html[html.length]="</tr></table>";
                  } else {
                        html[html.length]="<table height=4 width="+width+" cellpadding=0 cellspacing=0 border=0><tr>"
                        html[html.length]="<td class='kjmenu_fillbox kjmenu_frame' width=5 style='background-position:0px 0px'></td>";
                        html[html.length]="<td class='kjmenu_fillbox kjmenu_frame' style='background-position:0px -4px;background-repeat:repeat-x'><div class='kjmenu_fillbox' style='width:4px;height:4px'></div></td>";
                        html[html.length]="<td class='kjmenu_fillbox kjmenu_frame' width=5 style='background-position:-5px 0px'></td>";
                        html[html.length]="</tr></table>";
                  }

                  // placeholder for items
                  html[html.length]="<div class='kjmenu_body' style='width:"+(width-4)+"px'>";
                  html[html.length]="<table id='kjt_"+mitem.id+"' width=100% class='kjmenu' cellpadding=0 cellspacing=0 border=0>";
                  html[html.length]="</table></div>";

                  if (mitem.items.length>mitem.maxitems) {
                        // bottom scroll button
                        html[html.length]="<table id='kjm_tbldown_"+mitem.id+"'height=26 width="+width+" cellpadding=0 cellspacing=0 border=0><tr>"
                        html[html.length]="<td class='kjmenu_fillbox kjmenu_scroll' width=6 style='background-position:0px -78px'></td>";
                        html[html.length]="<td class='kjmenu_fillbox kjmenu_scroll' align=center style='background-position:0px -52px;background-repeat:repeat-x'>";
                        html[html.length]="<div class='kjmenu_fillbox kjmenu_scroll' style='width:13px;height:26px;background-position:-32px -78px'></div></td>";
                        html[html.length]="<td class='kjmenu_fillbox kjmenu_scroll' width=6 style='background-position:-45px -78px'></td>";
                        html[html.length]="</tr></table>";
                  } else {
                        html[html.length]="<table height=4 width="+width+" cellpadding=0 cellspacing=0 border=0><tr>"
                        html[html.length]="<td class='kjmenu_fillbox kjmenu_frame' width=5 style='background-position:0px -8px'></td>";
                        html[html.length]="<td class='kjmenu_fillbox kjmenu_frame' style='background-position:0px -12px;background-repeat:repeat-x'><div class='kjmenu_fillbox' style='width:4px;height:4px'></div></td>";
                        html[html.length]="<td class='kjmenu_fillbox kjmenu_frame' width=5 style='background-position:-5px -8px'></td>";
                        html[html.length]="</tr></table>";
                  }
                  html[html.length]="</div>";
                  return html.join('');
            }
            ,kjmenu_update : function (mitem) {
                  var maxindx = Math.min(mitem.offset+mitem.maxitems,mitem.items.length);
                  var html = new Array(), item;
                  for (var i=mitem.offset; i<maxindx; i++) {
                        item = mitem.items[i];
                        html[html.length]="<tr><td nowrap class='kjmenu_item ";
                        if (item.clickable)
                              html[html.length]=" kjmenu_clickable";
                        if (item.items && item.items.length>0)
                              html[html.length]=" kjmenu_expand";
                        html[html.length]="'";
                        if (item.style)   html[html.length]=" style='"+item.style+"'";
                        html[html.length]=">"+item.name+"</td></tr>";
                  }
                  $("#kjt_"+mitem.id).html(html.join(''));

                  $("#kjt_"+mitem.id+" td").each(function(i, el){
                        var cmenu = mitem.items[i+mitem.offset];
                        $(this).bind("mouseover", cmenu, function (e) {
                              var cmenu = e.data;
                              $("#kjt_"+mitem.id+" td.kjmenu_over").removeClass("kjmenu_over").removeClass("kjmenu_over_expand").removeClass("kjmenu_over_clickable");
                              $(this).addClass("kjmenu_over");
                              if(cmenu.clickable)
                                    $(this).addClass("kjmenu_over_clickable");
                              if (cmenu.items && cmenu.items.length>0)
                                    $(this).addClass("kjmenu_over_expand");
                              if (!cmenu.el) {
                                    if (mitem.expanded_child)
                                          $(mitem.expanded_child).kjmenu_remove(true);
                                    if (cmenu.items) {
                                          $(this).parent().kjmenu_create(cmenu);
                                          mitem.expanded_child = cmenu;
                                    }
                              }
                        }); 
                        $(this).bind("click", el, function () {
                              if (cmenu.clickable && typeof(mitem.root.opt.OnSelect)!="undefined") {
                                    mitem.root.opt.OnSelect (cmenu);
                                    $(document).trigger("mouseover");
                              }
                        });
                  });
                  
                  var enbl = (mitem.offset>0);
                  $("#kjm_tblup_"+mitem.id+" div").css("background-position",(enbl ? -6 : -32)+"px 0px");
                  $("#kjm_tblup_"+mitem.id+" .kjmenu_scroll").css("cursor", (enbl?"pointer":"default"));

                  enbl = (mitem.items && mitem.offset+mitem.maxitems<mitem.items.length);
                  $("#kjm_tbldown_"+mitem.id+" div").css("background-position",(enbl ? -6 : -32)+"px -78px");
                  $("#kjm_tbldown_"+mitem.id+" .kjmenu_scroll").css("cursor",(enbl?"pointer":"default"));
            }
            ,kjmenu_remove : function (drill) {
                  var mitem = this.get(0);
                  if (mitem.el) {
                        mitem.el.kjmenu_removeShadow();
                        if (mitem.root.opt.fadeOut>0) {
                              mitem.el.fadeOut(mitem.root.opt.fadeOut,function() {
                                    $(this).remove();
                                    mitem.el=null;
                              });
                        } else {
                              mitem.el.remove();
                              mitem.el=null;
                        }
                        if (drill && typeof(mitem.items)!="undefined") {
                              for(var i=0; i<mitem.items.length; i++) {
                                    if (mitem.items[i].el)
                                          $(mitem.items[i]).kjmenu_remove(mitem.items[i], true);
                              }
                        }
                  }
            }
            ,kjmenu_measure : function (mitem) {
                  if (mitem.items && mitem.items.length>0) {
                        if (!mitem.maxstring) {
                              var maxlen=0;
                              for (var i=0; i<mitem.items.length; i++) {
                                    if (mitem.items[i].name.length>maxlen) {
                                          maxlen = mitem.items[i].name.length;
                                          mitem.maxstring=mitem.items[i].name;
                                    }
                              }
                        }
                        mitem.root.ruler_cell.html(mitem.maxstring);
                        return mitem.root.ruler.width();
                  }
                  return 0;
            }
            ,kjmenu_dropShadow : function(options) {
                  var opt = $.extend({
                        inner_offx:2, offx:4,
                        inner_offy:2, offy:4,
                        opacity : 0.3,
                        bgcolor: "black",
                        type: 2
                        }, options);
                  var jdiv = this;
                  var divShadow = $("<div style='background-color:transparent'></div>")
                        .css({
                              left: jdiv.position().left+opt.offx-opt.inner_offx,
                              marginTop: jdiv.css("marginTop"),
                              marginRight: jdiv.css("marginRight"),
                              marginBottom: jdiv.css("marginBottom"),
                              marginLeft: jdiv.css("marginLeft"),
                              position: "absolute",
                              top: jdiv.position().top + opt.offy-opt.inner_offy,
                              width: jdiv.outerWidth(),
                              height: jdiv.outerHeight(),
                              zIndex: 1,
                              backgroundColor: opt.bgcolor,
                              opacity:opt.opacity
                        });
                  var jshad = $(divShadow);
                  if (opt.type==2) {
                        jshad.css({
                              backgroundColor: 'transparent',
                              opacity:1
                        });

                        var styleFn = function (opac) {
                              var opacIE = Math.floor(opac*100);
                              return "background-color:"+opt.bgcolor+";opacity:"+opac+";filter:\"alpha(opacity="+opacIE+"\");*filter:alpha(opacity="+opacIE+");filter:progid:DXImageTransform.Microsoft.Alpha(opacity="+opacIE+")";
                        }
                        var html = new Array();
                        html[html.length] = "<table width="+jdiv.outerWidth()+" height="+(jdiv.outerHeight())+" cellpadding=0 cellspacing=0 border=0 style='empty-cells:show'>";
                        var ostepy = opt.opacity/opt.offy, opstep,cop,width1=jdiv.outerWidth()-opt.offx,height1=jdiv.outerHeight()-opt.offy*2;
                        for (var y=0; y<opt.offy; y++) {
                              cop = ostepy*(y+1);
                              opstep = cop/opt.offx;
                              html[html.length]="<tr><td width="+width1+" height=1></td>";
                              for (var i=0; i<opt.offx; i++) {
                                    cop = opstep*(opt.offx-i);
                                    html[html.length]="<td width=1 height=1 style='"+styleFn(cop)+"'></td>";
                              }
                              html[html.length]="</tr>\n";
                        }
                        html[html.length]="<tr><td width="+width1+" height="+height1+"></td>";
                        opstep = opt.opacity/opt.offx;
                        for (var i=0; i<opt.offx; i++) {
                              cop = opt.opacity-opstep*i;
                              html[html.length]="<td width=1 style='"+styleFn(cop)+"'></td>";
                        }
                        html[html.length]="</tr>\n";
                        for (var y=0; y<opt.offy; y++) {
                              cop = ostepy*(opt.offy-y);
                              opstep = cop/opt.offx;
                              html[html.length]="<tr><td width="+width1+" height=1 style='"+styleFn(cop)+"'></td>";
                              for (var i=0; i<opt.offx; i++) {
                                    cop = opstep*(opt.offx-i);
                                    html[html.length]="<td width=1 height=1 style='"+styleFn(cop)+"'></td>";
                              }
                              html[html.length]="</tr>\n";
                        }
                        html[html.length]="</table>";
                        divShadow.html(html.join(''));
                  }
                  
                  jdiv.after(divShadow);
                  jdiv.data("kj_shadow",divShadow);
            }
            ,kjmenu_removeShadow : function() {
                  var jdiv = $(this);
                  var sdiv = jdiv.data("kj_shadow");
                  if (sdiv) 
                        $(sdiv).remove();             
                  jdiv.removeData("kj_shadow");
            }
      });
      
      //autocomplete
      var defaultAutoComplete = {
                enabled : true,
                autocompleteMode : false,
                active : null,
                typingTimeout : null,
                suppressKey : false,
                maxChars : 3,
                timeout: 1000,
                maxEntries : 7,
                list : null,
                listKeyVal : null,
                  hideLabel : 'Hide',
                  KEY : {    
                    ESC: 27,    
                      RETURN: 13,    
                      TAB: 9,  
                      BS: 8,   
                      DEL: 46,   
                      UP: 38,   
                      DOWN: 40,
                              SPACE: 32,
                              SLASH: 92
                },
                getList: function(input,url) {
                      $.getJSON(url, "val=" + input.val() + "&jsoncallback=?", function(json) {input.trigger("updateList", [json]);});
                  },
                  template: function(str) { return "<li>" + this.insertText(str) + "</li>"; },
                  insertText: function(str) {return str; },
                  match: function(typed,isCaseInsensitive) {
                      if(this.highlight){ 
                        return this;
                      }else{
                        return isCaseInsensitive?this.match(new RegExp(typed,"i")):this.match(new RegExp(typed)); 
                      }
                    },
                  wrapper: "<ul></ul>",
            updateDropDown : function(usrVal) {
               var len = usrVal.length;
                      var count = 0;
                      var self = this;
                      var list = $(this.list)
                              .filter(function() { return count < self.maxEntries && self.match.call(this, usrVal,self.isCaseInsensitive) && ++count; })
                                    .map(function() {
                                        var thisElem;
                                        if(this.highlight){ 
                                          thisElem = this.highlight;
                                        }else{
                                            thisElem = this;
                                            var idx;
                                              if(self.isCaseInsensitive){
                                                idx = thisElem.toLowerCase().indexOf(usrVal);
                                              }else{
                                                    idx = thisElem.indexOf(usrVal);
                                                }
                                                thisElem = thisElem.substring(0,idx) + "<b>" + thisElem.substring(idx,idx+len) + "</b>" + thisElem.substring(idx+len);
                                          }
                                          var node = $(self.template(thisElem))[0];
                                          if(this.highlight)
                                                $.data(node, "originalObject", this.highlight);
                                          else
                                                $.data(node, "originalObject", this);
                                          return node; 
                                    });
                        $("body").trigger("off.autocomplete");
                        if(!list.length) return false;
                      var container = list.wrapAll(self.wrapper).parents(":last").children();
                      // IE seems to wrap the wrapper in a random div wrapper so
                        // drill down to the node in opt.wrapper.
                      var wrapper_tagName = $(self.wrapper)[0].tagName;
                        for (;container[0].tagName !== wrapper_tagName; container = container.children(':first')) {}
                  container = container.wrap("<div class='jq-ui-autocomplete'></div>").parent();
                  container.append("<div class='autohide'><a href='#'>" + self.hideLabel + "</a></div>");
                    $('a',container).bind('click',function(){$("body").trigger("cancel.autocomplete");self.enabled=false;return false;});
                        var offset = self.jqInput.offset();
                      self.container = container
                              .css({top: offset.top + self.jqInput.outerHeight(), left: offset.left, width: (self.jqInput.outerWidth()-2) + 'px'})
                              .appendTo("."+self.containerStyleClass);
                      self.setAutocompleteMode(container, self.jqInput, list.length);
                  },
                  setAutocompleteMode : function(container, input, size) {
                      var original = input.val(); var selected = -1; var self = this;
                  self.autocompleteMode = true;
                        
                  $("body").one("cancel.autocomplete", function() {
                        input.trigger("cancel.autocomplete"); 
                        $("body").trigger("off.autocomplete"); 
                        input.val(original); 
                  });
                  $("body").bind("xactivate.autocomplete", function() {
                        // Try hitting return to activate autocomplete and then hitting it again on blank input      
                        // to close it.  w/o checking the active object first this input.trigger() will barf.      
                        self.active && input.trigger("xactivate.autocomplete", [$.data(self.active[0], "originalObject")]);      
                        $("body").trigger("off.autocomplete");
                  });
                        
                  $("body").one("off.autocomplete", function(e, reset) {
                        container.remove();
                        self.autocompleteMode = false;
                        input.unbind("keydown.autocomplete");
                        $("body").add(window).unbind("click.autocomplete").unbind("cancel.autocomplete").unbind("xactivate.autocomplete");
                  });
                        
                  // If a click bubbles all the way up to the window, close the autocomplete
                  $("body").bind("click.autocomplete", function() {$("body").trigger("cancel.autocomplete"); });
            
                  var select = function() {
                        self.active = $("li", container).removeClass("active").slice(selected, selected + 1).addClass("active");
                          input.trigger("itemSelected.autocomplete", [$.data(self.active[0], "originalObject")]);
                          input.val(self.insertText($.data(self.active[0], "originalObject")));
                    };
                        
                  container.find('ul').mouseover(function(e) {
                        // If you hover over the container, but not its children, return
                        if(e.target == container[0] || $(e.target).is('ul') ) return;
                        // Set the selected item to the item hovered over and make it active
                        selected = $("li", container).index($(e.target).is('li') ? $(e.target)[0] : $(e.target).parents('li')[0]); 
                        self.active = $("li", container).removeClass("active").slice(selected, selected + 1).addClass("active");
                        //select();
                  }).click(function(e){
                      if(e.target == container[0] || $(e.target).is('ul') ) return;
                        selected = $("li", container).index($(e.target).is('li') ? $(e.target)[0] : $(e.target).parents('li')[0]); 
                        self.active = $("li", container).removeClass("active").slice(selected, selected + 1).addClass("active");
                          input.val(self.insertText($.data(self.active[0], "originalObject")));
                          $("body").trigger("off.autocomplete");
                          if(self.submitOnlick)
                          	input[0].form.submit();
                    });
                        
                  input.bind("keydown.autocomplete", function(e) {
                      var KEY = self.KEY;
                        var k = e.which || e.keyCode; // in IE e.which is undefined
                        if(k == KEY.ESC) { $("body").trigger("cancel.autocomplete"); } 
                                                else if(k == KEY.BS && original.length == 1) { $("body").trigger("cancel.autocomplete"); }
                        else if(k == KEY.RETURN) { $("body").trigger("xactivate.autocomplete"); } 
                        else if(k == KEY.UP || k == KEY.TAB || k == KEY.DOWN) {          
                            switch(k) {           
                                case KEY.DOWN:          
                                case KEY.TAB:              
                                      selected = selected >= size - 1 ? 0 : selected + 1; break;
                                case KEY.UP:             
                                      selected = selected <= 0 ? size - 1 : selected - 1; break;            
                                      default: break;          
                              }          
                              select();        
                        } else { return true; }  
                        this.suppressKey =  true;
                    });
              },
              preventTabInAutocompleteMode :    function (e) {
                      var k = e.which || e.keycode;   
                      if ( this.autocompleteMode && k == this.KEY.TAB) { 
                            e.preventDefault();   
                      }   
                }
                              
              
          };
                  
      $.fn.autocomplete = function(opt) {
            opt = $.extend({},defaultAutoComplete,opt);

            return this.each(function() {             
                $(this)
                      .keydown(function(e) {                            
                              opt.preventTabInAutocompleteMode(e);   
                        })
                        .keydown(function(e){ //holy shit, back button is keydown in IE
                    if ( !opt.enabled ) return;
                              var k = e.keyCode || e.which;
                              var bs = false;
                      var KEY = opt.KEY;
                              if((k == KEY.DEL || k == KEY.BS) && $.browser.msie){ //don't execute on FF cus back button is keypress
                                    opt.typingTimeout = window.setTimeout(function() { $(e.target).trigger("autocomplete");}, opt.timeout);
                              }
                        })
                        .keypress(function(e) {
                    if ( !opt.enabled ) return;
                      var KEY = opt.KEY;
                              var typingTimeout = opt.typingTimeout;   
                              var k = e.keyCode || e.which; // keyCode == 0 in Gecko/FF on keypress   
                              if(typingTimeout) window.clearInterval(typingTimeout);        
                              if( opt.suppressKey ) {
                                    opt.suppressKey = false;
                                    //note that IE does not generate keypress for arrow/tab keys
                                    if(k == KEY.TAB || k == KEY.UP || k == KEY.DOWN) return false;
                              }
                              if(opt.autocompleteMode && (k == KEY.UP || k == KEY.DOWN)) return false;
                              if(opt.autocompleteMode && k < 32 && k != KEY.BS && k != KEY.DEL) return false;
                              else if (k == KEY.BS || k==KEY.DEL || k==KEY.SPACE || k > 32) { // more than ESC and RETURN and the like
                                    opt.typingTimeout = window.setTimeout(function() { $(e.target).trigger("autocomplete"); }, opt.timeout);
                            }
                        })
                        .bind("autocomplete", function() {
                            if ( !opt.enabled ) return false;
                              var self = $(this);
                              var eVal = self.val().toLowerCase();
                              var eLen = eVal.length;
                              //opt.listKeyVal is not defined at this time. Looks like we don't have to do this checking - need to confirm
                              if(eLen > opt.maxChars /*&& eVal.substring(0,opt.maxChars) == opt.listKeyVal.substring(0,opt.maxChars)*/){
                                  if ( opt.list ) {
                                    opt.updateDropDown(eVal);
                                          }
                                          return false;
                              }
                              var jUrl = opt.baseUrl;
                              if(eVal == "" || eVal == null){return false;}
                              var arrayVal = new Array();
                              if(eLen > opt.maxChars){
                                    eLen = opt.maxChars;
                              }
                              for(var i=0;i<=eLen-1; i++){
                                    arrayVal[i] = encodeURIComponent(eVal.charAt(i)).toLowerCase();
                              }
                                          
                              for(var k=0;k<=arrayVal.length-1;k++){
                                    var aVal = arrayVal[k];
                                    if(aVal.length > 1 && aVal.match("%") != null){
                                          aVal = aVal.replace(/%/g,"_");
                                          if(aVal.indexOf("_") == 0){
                                                arrayVal[k] = aVal.substr(1);
                                          }else{
                                                arrayVal[k] = aVal;
                                          }
                                    }
                              }
                              var len = arrayVal.length;
                              if(len == 1){
                                    jUrl += arrayVal[0] + ".js";
                              }else if(len >1){
                                    for(var i=0;i<=len-1;i++){
                                          if(i != len-1){
                                                jUrl += arrayVal[i] + "/";
                                          }
                                          if(i == len-1){
                                                jUrl += arrayVal[i] + ".js";
                                          }
                                    }
                              }else{
                                    return false;
                              }
                              self.one("updateList", function(e, list) {
                                  if(opt.fixedResults){
                                    opt.list = opt.fixedResults.concat(list);
                                  }else{
                                    opt.list = list;
                                  }
                                    
                                    opt.listKeyVal =self.val().toLowerCase();
                                    opt.jqInput = self;
                                    opt.updateDropDown(opt.listKeyVal);
                            });
                        opt.getList(self,jUrl);
                });
            
          });
    };      
})(jQuery);

(function($){  
      $.fn.jqslider_make = function(options) {
            var defaults = {
                  cname: "jqslider",
                  units:"km",
                  min_value: 1,
                  max_value:80,
                  maxmax_value:300, // max value for max_value
                  value:1
            };  
            var opts = $.extend(defaults, options);  
            return this.each(function() {
                  $this = $(this);
                  $this.data("opts",opts);
                  $this.addClass(opts.cname);

                  var range = opts.max_value-opts.min_value, grades=4;
                  for (var i=Math.min(Math.floor(range/2),10); i > 1; i--) {
                        if (range%i==0) {
                              grades=i;
                              break;
                        }
                  }
                  if (grades<3)     grades*=2;
                  
                  var tdw = Math.floor(100/grades);
                  var html="<div class='"+opts.cname+"_poi'><div class='wrap'><div class='brand_border poi_text_wrap'><div class='poi_text'></div></div></div></div>";
                  html+="<div class='ruler'><table width='100%' class='"+opts.cname+"_tbl' cellpadding=0 cellspacing=0 border=0><tr>";
                  for (var i=0; i<grades; i++)
                        html+="<td width='"+tdw+"%' class='brand_border"+(i==0?" td1":"")+"'>&nbsp;</td>";
                  html+="</tr></table>";
                  html+="<span class='"+opts.cname+"_text' style='float:left'>"+opts.min_value+opts.units+"</span><span id='range_max' class='"+opts.cname+"_text' style='float:right;text-align:right'>"+opts.max_value+opts.units+"</span></div>";
                  $this.html(html);
                  
                  var $poi = $("."+opts.cname+"_poi");
                  var $tbl = $("."+opts.cname+"_tbl");
                  var dragging = false, drag_off=0, old_val, old_max;
                  $("."+opts.cname+"_poi").bind("mousedown", function(evnt) {
                        old_val=opts.value;
                        old_max = opts.max_value;
                        dragging=true;
                        var pos = $poi.offset();
                        drag_off = evnt.pageX-(pos.left+$poi.width()/2);
                        return false;
                  });
                  $(document.body).bind("mouseup", function(evnt) {
                        var tmp = dragging;
                        dragging=false;
                        if (tmp && old_val!=opts.value) {
                              $this.triggerHandler("slider_change", {value:opts.value, old_value:old_val});
                        }
                  });
                  $(document.body).bind("mousemove", function(evnt) {
                        if (dragging) {
                              var xpos = evnt.pageX-drag_off;
                              var off=$tbl.offset();
                              var nval=Math.min(opts.maxmax_value, Math.max(opts.min_value, opts.min_value+Math.floor((xpos-off.left)*(old_max-opts.min_value)/$tbl.width()+0.5)));
                              if (nval != opts.value) {
                                    if (nval > old_max) {
                                           opts.max_value = nval;
                                           $("#range_max").html(nval+opts.units);
                                    } else if (nval <= old_max && opts.max_value > old_max) {
                                          opts.max_value = old_max;
                                           $("#range_max").html(opts.max_value+opts.units);
                                    }
                                    $this.jqslider_update(nval);
                              }
                              return false;
                        }
                  });
                  
                  $this.jqslider_update(opts.value, true);
                  
            });
      };
      $.fn.extend({
            jqslider_update : function (nval, init) {
                  $this = this;
                  var opts = $this.data("opts");
                  if ( opts == null ) return;
                  opts.value=nval;
                  $("."+opts.cname+"_poi .poi_text").html(init && opts.value_init ? opts.value_init : (opts.value+opts.units));
                  $this.find(".jqslider_text").css("color",init && opts.value_init ? "#aaa" : "");

                  var $poi = $("."+opts.cname+"_poi");
                  var $tbl = $("."+opts.cname+"_tbl");
                  $poi.css({
                        left:Math.min($tbl.width(), Math.floor((opts.value-opts.min_value)*$tbl.width()/(opts.max_value-opts.min_value)+0.5))-$poi.width()/2 
                  });
            }
      });
})(jQuery);

(function($) {
      $.fn.ellipsis = function(enableUpdating){
            var s = document.documentElement.style;
            if (!('textOverflow' in s || 'OTextOverflow' in s)) {
                  return this.each(function(){
                        var el = $(this);
                        if(el.css("overflow") == "hidden"){
                              var originalText = el.html();
                              var w = el.width();
                              
                              var t = $(this.cloneNode(true)).hide().css({
                        'position': 'absolute',
                        'width': 'auto',
                        'overflow': 'visible',
                        'max-width': 'inherit'
                    });
                              el.after(t);
                              
                              var text = originalText;
                              while(text.length > 0 && t.width() > el.width()){
                                    text = text.substr(0, text.length - 1);
                                    t.html(text + "...");
                              }
                              el.html(t.html());
                              
                              t.remove();
                              
                              if(enableUpdating == true){
                                    var oldW = el.width();
                                    setInterval(function(){
                                          if(el.width() != oldW){
                                                oldW = el.width();
                                                el.html(originalText);
                                                el.ellipsis();
                                          }
                                    }, 200);
                              }
                        }
                  });
            } else return this;
      };
})(jQuery);


function urlHelper (data) {
      this.fromString = function(inp) {
            this._map = {};
            var list = inp.split('&'),name,value,indx;
            for(var i=0; i<list.length; i++) {
                  name = list[i];         value=null;
                  indx = name.indexOf('=');
                  if (indx>0) {
                        value=name.substring(indx+1);
                        name=name.substring(0,indx);
                  }
                  this._map[name]=value;
            }
      }
      this.toString = function() {
            var res="";
            for(var key in this._map) {
                  if (res.length>0) res+="&";
                  res+=encodeURIComponent(key);
                  if (this._map[key])
                        res+="="+encodeURIComponent(this._map[key]);
            }
            return res;
      }
      this.addParam = function (name,value) {
            this._map[key]=value;
      }
      this.removeParam = function (name) {
            var arr = {};
            for(var key in this._map) {
                  if (key!=name)
                        arr[key]=this._map[key];
            }
            this._map=arr;
      }
      
      this._map = {};
      if (data)         this.fromString(data);
}

function setLastVisitedAreaInCookie(areaId)
{
   var today = new Date();
   var expiry = new Date(today.getTime() + 90 * 24 * 60 * 60 * 1000); // plus 90 days
   var urlDomain = document.domain;
   var indexOfFirstDot = urlDomain.indexOf('.');
   var cookieDomain = urlDomain.substring(indexOfFirstDot, urlDomain.length);
   var cookieName = "site_preference";
   var cookiePath = "/";
   var cookieDecoderChar = "V"; //void/no encoding
   var value = "default_area="+encodeURIComponent(areaId);//forming cookielets
   value = cookieDecoderChar + value;
   
   setCookie(cookieName, value, expiry, cookiePath, cookieDomain,null);
 }
 
function setCookie(name, value, expires, path, domain, secure)
{
    var temp = name + "=" + value + 
            ((expires == null) ? "" : ("; expires=" + expires.toGMTString())) +
            ((path == null) ? "" : ("; path=" + path)) +
            ((domain == null) ? "" : ("; domain=" + domain)) +
            ((secure == null) ? "" : ("; secure=" + secure)) ;
    document.cookie = temp;
}

function readCookie(name) {
      var toRet = null
      if(!name) return null;
      var nameEQ = name + "=";
      var ca = document.cookie.split(';');
      for(var i=0;i < ca.length;i++) {
            var c = ca[i];
            while (c.charAt(0)==' ') c = c.substring(1,c.length);
            if (c.indexOf(nameEQ) == 0){
                  toRet =  c.substring(nameEQ.length,c.length);
                  if(toRet.length >= 1) toRet =  toRet.substring(1,toRet.length); //remove version info
            }                 
      }
      return toRet;
}

function readCookielet(cookieName, cookieletName){
      var toRet = null;
      if(!(cookieName && cookieletName)) return;
      var cookieValue = readCookie(cookieName);
      if(cookieValue == null) return null;
      var cookieletNameEQ = cookieletName + "=";
      var ca = cookieValue.split('^');
      for(var i=0;i < ca.length;i++) {
            var c = ca[i];
            while (c.charAt(0)==' ') c = c.substring(1,c.length);
            if (c.indexOf(cookieletNameEQ) == 0){
                   toRet = c.substring(cookieletNameEQ.length,c.length);
                   break;
            }                       
      }
      return toRet;
}
            
            
function decodeBase64(rv){
      var len=rv.length,ret="",i=0;var chr1,chr2,chr3="";
      var enc1,enc2,enc3,enc4="";
      var aChar="ABCDEFGHIJKLMNOPQRSTUVWXYZ"+"abcdefghijklmnopqrstuvwxyz"+"0123456789+/=*";
      var test=new RegExp("[^A-Za-z0-9+/=*]");
      if(test.exec(rv)){return;}
      do{
            enc1=aChar.indexOf(rv.charAt(i++));
            enc2=aChar.indexOf(rv.charAt(i++));
            enc3=aChar.indexOf(rv.charAt(i++));
            enc4=aChar.indexOf(rv.charAt(i++));
            chr1=(enc1<<2)|(enc2>>4);
            chr2=((enc2&15)<<4)|(enc3>>2);chr3=((enc3&3)<<6)|enc4;ret+=String.fromCharCode(chr1);
            if(!(enc3>=64)) ret+=String.fromCharCode(chr2);
            if(!(enc4>=64)) ret+=String.fromCharCode(chr3);
            chr1=chr2=chr3=enc1=enc2=enc3=enc4="";}while(i<len);return ret;
      }


function decodeUTF8(s){
      var len=s.length;
      var rs="";
      var i=0;
      var c=c1=c2=0;
      while(i<len){
            c=s.charCodeAt(i);
            if(c<128){
                  rs+=String.fromCharCode(c);
                  i++;
            }
            else if((c>191)&&(c<224)){
                  c2=s.charCodeAt(i+1);
                  rs+=String.fromCharCode(((c&31)<<6)|(c2&63));i+=2;
            }
            else{
                  c2=s.charCodeAt(i+1);
                  c3=s.charCodeAt(i+2);
                  rs+=String.fromCharCode(((c&15)<<12)|((c2&63)<<6)|(c3&63));i+=3;}
            }
      return rs;
}


function showHeaderWithGreetings(){
      var withGreetings = document.getElementById('withGreetings');
      var withoutGreetings = document.getElementById('withoutGreetings');
      var encodedUserGreeting = readCookielet('na', 'ug');
      if(encodedUserGreeting !=null)
      {
            withGreetings.innerHTML = withGreetings.innerHTML.replace('replacewithusergreetings',decodeUTF8(decodeBase64(encodedUserGreeting)));
            withoutGreetings.innerHTML = withGreetings.innerHTML;
      }

}

function DoNav(xRowId){
      var row = document.getElementById(xRowId);
      row.className='highlight';
      row.onmouseout = new Function("this.className=''");
      var link = row.getElementsByTagName("a");
      if(link[0].onclick){
            row.onclick = eval(link[0].onclick);
      }
      else{
            row.onclick = new Function("document.location.href='" + link[0].href + "'");
      }
}

function SortResults(frm){
  var list = frm.sortBy;
  var sortOption = list.options[list.selectedIndex].value;
      document.location = sortOption;
}
//use by view ad
var sp = '[\\w/\\#~:.?+=%@!\\-;,]|(&(?!nbsp;))';
var sp2 = '\\w/\\#~:.?+=%@!\\-;,&';
var atext = '[a-zA-Z0-9]';
var letter = '[a-zA-Z]';
var letDig = '[a-zA-z0-9]';
var letDigHyp = '[a-zA-Z0-9-]';
var localPart = atext + '+([a-zA-Z0-9-_.]+)*';
var rfcLabel = letter + letDigHyp + '*' + letter;
var rfcLabel2 = letDig + letDigHyp + '*';
var domain = rfcLabel2 + '(\\.' + rfcLabel2 + ')*';
var emailUrl = localPart + '@' + domain;
var protocol = '((http|https):\/\/(www.)?|www.)';
var port = '(:\d+)?';
var wwwUrl = '(' + protocol + domain + port + '(' + sp + '*)*)(?=[,.:?\-]*(?:[^' + sp2 + ']|$))';
function AdLink(aLink){
      window.location = aLink;
}

function trackClick(trackerurl) {
      if (trackerurl!="undefined" && trackerurl) {
            (new Image).src='http://altfarm.mediaplex.com/ad/ck/'+trackerurl;
      }
      return true;
}

function trackROI(trackerurl) {
      if (trackerurl!="undefined" && trackerurl) {          
            (new Image).src='http://altfarm.mediaplex.com/ad/bk/'+trackerurl;
      }
      return true;
}

(function($) {
      $.fn.make_imggal = function(options){
            var defaults = {
                  cname : "imggal",
                  fullImgUrl : null,
                  index : 0,
                  images : [],
                  speed:400
            };  

            return this.each(function() {
                  var opts = $.extend(defaults, options);
                  var $this = $(this);
                  
                  if (opts.fullImgUrl) {
                        $("[imggal='viewimg'], [imggal='main']").click (function() {
                              document.location.assign(opts.fullImgUrl+(opts.index>0 ? "&ImageIndex="+opts.index : ""));
                        });  
                  }
                  $this.find("[imggal='prev']").click(function () {
                        if (opts.images.length>1) {
                              var indx = opts.index-1;
                              if (indx<0) indx = opts.images.length-1;
                              $this.selectImage(indx);
                        }
                  });
                  $this.find("[imggal='next']").click(function () {
                        if (opts.images.length>1) {
                              var indx = opts.index+1;
                              if (indx>=opts.images.length) indx = 0;
                              $this.selectImage(indx);
                        }
                  });
                  $this.find("[imggal='thumb']").click(function () {
                        var indx=$(this).attr("imgindx")-0;
                        if (opts.index!=indx)
                              $this.selectImage(indx, true);
                  });
                  $this.data("opts", opts);
            });  
      };

      $.fn.extend({
            selectImage : function (indx, animate) {
                  if ($("#imggal_anim1").size()>0)    return;           
            
                  var $this = $(this);
                  var opts = $this.data("opts");

                  var mainimg = $("[imggal='main'] img").get(0);
                  mainimg._animate_done = !animate;
                  mainimg._load_done=false;
                  mainimg.onload = function() {
                        mainimg._load_done = true;
                        mainimg.onupdate();
                  }
                  mainimg.onupdate = function() {
                        if (mainimg._load_done && mainimg._animate_done) {
                              mainimg._load_done = false;
                              $(mainimg).show();
                              opts.index=indx;
                              $this.trigger("select", opts);
                              $("#imggal_anim1").remove(); 
                        }
                  }
                  $(mainimg).hide();
                  mainimg.src=opts.images[indx];

                  var calcbounds = function(img, bounds) {
                        var scale = Math.min(bounds.width/img.width, bounds.height/img.height);
                        return {width: Math.floor(img.width*scale), height: Math.floor(img.height*scale)};
                  }
                  var getrect = function($obj) {
                        var pos = $obj.offset();
                        return {left:pos.left, top:pos.top, width:$obj.width(), height:$obj.height()};              
                  }
                  
                  if (animate) {                      
                        var $anim1_src = $this.find("[imggal='thumb'][imgindx='"+indx+"'] img").eq(0);
                        var $anim1_targ = $("[imggal='main']").eq(0);
                        var anim1_src_rect = getrect($anim1_src), anim1_targ_rect=getrect($anim1_targ);
                        var imgsize = calcbounds({width:anim1_src_rect.width, height:anim1_src_rect.height}, {width: anim1_targ_rect.width, height: anim1_targ_rect.height});
                        anim1_targ_rect = {
                              left: anim1_targ_rect.left+(anim1_targ_rect.width-imgsize.width)/2,
                              top: anim1_targ_rect.top+(anim1_targ_rect.height-imgsize.height)/2,
                              width: imgsize.width, height: imgsize.height
                        };
                        
                        var $anim2_src = $("[imggal='main']").eq(0);
                        var $anim2_targ = $this.find("[imggal='thumb'][imgindx='"+opts.index+"'] img").eq(0);
                        
                        var anim2_src_rect = getrect($anim2_src), anim2_targ_rect=getrect($anim2_targ);
                        imgsize = calcbounds({width: anim2_targ_rect.width, height: anim2_targ_rect.height}, {width: anim2_src_rect.width, height: anim2_src_rect.height});
                        anim2_src_rect = {
                              left: anim2_src_rect.left+(anim2_src_rect.width-imgsize.width)/2,
                              top: anim2_src_rect.top+(anim2_src_rect.height-imgsize.height)/2,
                              width: imgsize.width, height: imgsize.height
                        };

                        var html = "<img id='imggal_anim2' width="+anim2_src_rect.width+" height="+anim2_src_rect.height+" src='"+opts.images[opts.index]+"' style='position:absolute;left:"+anim2_src_rect.left+"px;top:"+anim2_src_rect.top+"px'>";
                        html += "<img id='imggal_anim1' width="+anim1_src_rect.width+" height="+anim1_src_rect.height+" src='"+$anim1_src.get(0).src+"' style='position:absolute;left:"+anim1_src_rect.left+"px;top:"+anim1_src_rect.top+"px'>";                    
                        $(document.body).append(html);

                        $("#imggal_anim2").load(function() {
                              $("#imggal_anim1").animate({
                                    left : anim1_targ_rect.left,
                                    top : anim1_targ_rect.top,
                                    width: anim1_targ_rect.width,
                                    height: anim1_targ_rect.height
                              }, opts.speed, function() {
                                    mainimg._animate_done = true;
                                    mainimg.onupdate();
                              });
                              $("#imggal_anim2").animate({
                                    left : anim2_targ_rect.left,
                                    top : anim2_targ_rect.top,
                                    width: anim2_targ_rect.width,
                                    height: anim2_targ_rect.height
                              }, opts.speed, function() {
                                    $("#imggal_anim2").remove();
                              });
                        });
                  }
            }
      });
})(jQuery);
(function($){ 
      $.fn.locMenu_makeMenu = function(options) {
      
            var defaults = {
                  zindex:100,
                  shadow: 2,
                  fadeIn:0,
                  fadeOut:0,
                  indentPixels:15,
                  maxitems:100
            };  
            var options = $.extend(defaults, options);  
            if(options.page != "SiteHome"){
            
                  return this.each(function() {
                        $(this).locMenu_start(options);
                  });
            }
            else{
                  $(this).locMenu_start(options);
            }
      }; 
      $.fn.extend({
            locMenu_start : function(options){
                  $(this).attr("locMenu", this.id);
                  this.menu_root = {items:new Array(),items2:new Array(),items3:new Array(),items4:new Array(),items2a:new Array(),items3a:new Array(),items4a:new Array(),id:this.id,opt:options};
                  this.menu_root.root = this.menu_root;
                  if (options.data)
                        $(this.menu_root).locMenu_fromLocString(options.data,options.loc);
                  if(options.page != "SiteHome"){
                        $(document).bind("click",this.menu_root,function(e) {
                              var contr = $("#ChgLocationPanel").get(0);
                              if(!$(this).containsElement(contr,e.target)){
	                              if (e.data.el) {
	                                    var mitem = e.data;
	                                    if (!mitem.opt.timer) {
	                                          mitem.opt.timer = setTimeout(function() {
	                                                mitem.opt.timer = null;
	                                                $(mitem).locMenu_remove(true);
	                                          }, 400);
	                                    }
	                              }
	                          }
                        });
                  }
                  var root = this.menu_root;
                  if (root.opt.timer) {
                        clearTimeout(root.opt.timer);
                        root.opt.timer=null;
                  }
                  if (!root.el) {
                        $("[locMenu]").each(function() {
                              $(this.menu_root).locMenu_remove(true);
                        });
                        $(this).locMenu_create(root);       
                  }
                  return false;
            }
            ,locMenu_fromLocString : function (data, loc) {
                  var ritem = this.get(0);
                  var mitem, mitem2, mitem3,mitem4;
                  var a = new Array(); var b = new Array(); var c = new Array();var d = new Array();
                  for(var i=0; i<data.level1.length; i++){
                        a[i] = data.level1[i].name+"^"+data.level1[i].id;
                        if(data.level1[i].id == loc[0] && data.level1[i].level2){
                              for(var j=0; j<data.level1[i].level2.length; j++){
                                    b[j] = data.level1[i].level2[j].name+"^"+data.level1[i].level2[j].id+"^"+data.level1[i].id+"^"+i;
                                    if(data.level1[i].level2[j].id == loc[1] && data.level1[i].level2[j].level3){
                                          for(var k=0; k<data.level1[i].level2[j].level3.length; k++){
                                                c[k] = data.level1[i].level2[j].level3[k].name+"^"+data.level1[i].level2[j].level3[k].id+"^"+data.level1[i].level2[j].id+"^"+(j+i);
                                                if(data.level1[i].level2[j].level3[k].id == loc[2] && data.level1[i].level2[j].level3[k].level4){
                                                      for(var l=0; l<data.level1[i].level2[j].level3[k].level4.length; l++){
                                                            d[l] = data.level1[i].level2[j].level3[k].level4[l].name+"^"+data.level1[i].level2[j].level3[k].level4[l].id+"^"+data.level1[i].level2[j].level3[k].id+"^"+(i+j+k);
                                                      }
                                                }
                                          }
                                    }
                              }
                        }
                  }
                  for (var i=0; i<a.length; i++) {
                        var parts = a[i].split("^");
                        mitem = {pid:0,id:parts[1],name:parts[0],value:ritem.opt.lUrl+"?id="+parts[1]+"&CatId="+ritem.opt.cId+"&ruq="+ritem.opt.ref,root:ritem,parent:ritem};
                        ritem.items[ritem.items.length]=mitem;
                  }
                  for (var i=0; i<b.length; i++) {
                        var parts = b[i].split("^");
                        mitem2 = {pid:parts[2],id:parts[1],name:parts[0],value:ritem.opt.lUrl+"?id="+parts[1]+"&CatId="+ritem.opt.cId+"&ruq="+ritem.opt.ref,pos:parts[3],root:ritem,parent:ritem};
                        ritem.items2[ritem.items2.length]=mitem2;
                  }
                  for (var i=0; i<c.length; i++) {
                        var parts = c[i].split("^");
                        mitem3 = {pid:parts[2],id:parts[1],name:parts[0],value:ritem.opt.lUrl+"?id="+parts[1]+"&CatId="+ritem.opt.cId+"&ruq="+ritem.opt.ref,pos:parts[3],root:ritem,parent:ritem};
                        ritem.items3[ritem.items3.length]=mitem3;
                  }
                  for (var i=0; i<d.length; i++) {
                        var parts = d[i].split("^");
                        mitem4 = {pid:parts[2],id:parts[1],name:parts[0],value:ritem.opt.lUrl+"?id="+parts[1]+"&CatId="+ritem.opt.cId+"&ruq="+ritem.opt.ref,pos:parts[3],root:ritem,parent:ritem};
                        ritem.items4[ritem.items4.length]=mitem4;
                  }
            }
            ,locMenu_create : function (mitem) {
                  if (mitem.el)     return;
                  $("#changeLocDiv").css("background-image","url(http://pic.classistatic.com/image/pics/classifieds/button_close.gif)");//ignore img name, arrow points up
                  // calculate how many items will fit vertically
                  var pdiv = this;
                  if(mitem.root.opt.page == "SiteHome"){pdiv = $("#topCCB");}
                  var ypos = !mitem.parent ? pdiv.offset().top+pdiv.height() : 0;
                  //var itemheight = mitem.root.ruler.height();
                  mitem.maxitems = mitem.root.opt.maxitems;//Math.min(mitem.root.opt.maxitems, Math.max(3, Math.floor(($(window).height()-ypos-60)/itemheight)));//22

                  $(document.body).append($().locMenu_mainHTML(mitem));
                  mitem.el = $('#kjm_'+mitem.id);
                  mitem.el.css({zIndex:mitem.root.opt.zindex++});
                  mitem.el.locMenu=mitem;
                  
                  // filling items
                  $(mitem).locMenu_update(mitem);

                  // positioning menu
                  if (!mitem.parent) {
                        var xpos = pdiv.offset().left;
                        xpos = Math.min($(window).width()-mitem.el.width()-5, xpos);
                        if($.browser.safari){xpos = xpos - 15;}
                        mitem.el.css({top:(pdiv.offset().top+pdiv.height()), left:xpos});
                  } else {
                        var pardiv = mitem.parent.el;
                        var ypos = pdiv.offset().top-4;
                        ypos = (ypos+mitem.el.height()<$(window).height()) ? ypos : Math.max(0,ypos+pdiv.height()+4-mitem.el.height());
                        mitem.el.css({top:ypos, left:(pardiv.offset().left+pardiv.width()+1)});
                  }
                        mitem.el.show();

                  //if IE6 we add iframe
                  if($.browser.msie ){
                        mitem.el.append("<iframe style='display:block;position:absolute;top:0;left:0;z-index:-1;filter:mask();width:"+(mitem.el.width())+";height:"+(mitem.el.height())+"'/>");
                  }

                  // binding mouse events for menu body
                  mitem.el.bind("mousedown", function () {return false;});
                  mitem.el.bind("mouseover", mitem, function (e) {
                        var mitem = e.data;
                        if (mitem.root.opt.timer) {
                              clearTimeout(mitem.root.opt.timer);
                              mitem.root.opt.timer=null;
                        }
                        return false;
                  });
                  
                  $("#updateBtn").click(function() {
		        	mitem.root.opt.OnUpdate();
				   });
				  $("#cancelBtn").click(function() {
    					if (!mitem.opt.timer) {
                              mitem.opt.timer = setTimeout(function() {
                                    mitem.opt.timer = null;
                                    $(mitem).locMenu_remove(true);
                              }, 400);
                        }
                        return false;
				   });
                  
            }
            ,locMenu_mainHTML : function (mitem) {
                   var colspan,btntxt;
                   if(mitem.root.opt.lv4Exists){
                   		colspan = 7;
                   }else{
                   		colspan = 6;
                   }
                  // measure max width in pixels
                  //var width = $().locMenu_measure(mitem)+mitem.root.opt.indentPixels+6;
                  var width = mitem.root.opt.width;
                  // building html for whole menu (without items inside yet)                                
                  var html = new Array();
                   html[html.length]="<div id='ChgLocationPanel'>"
                  if(mitem.root.opt.page == "SiteHome"){
                  		btntxt = mitem.root.opt.goTxt;
                        html[html.length]="<div id='kjm_"+mitem.id+"' class='locMenu_div' style='width:90%;display:none;top:0px;left:0px'>";
                        html[html.length]="<table height=4 width=100% cellpadding=0 cellspacing=0 border=0><tr>";
                  }
                  else {
                        btntxt = mitem.root.opt.updateTxt;
                        html[html.length]="<div id='kjm_"+mitem.id+"' class='kjmenu_div' style='display:none;top:0px;left:0px'>";
                        html[html.length]="<table height=4 width="+width+" cellpadding=0 cellspacing=0 border=0><tr>";
                  }
                  html[html.length]="<td class='kjmenu_fillbox locMenu_frameTLC' width=10 style=''></td>";
                  html[html.length]="<td class='kjmenu_fillbox locMenu_frameT' style=''><div class='kjmenu_fillbox' style='width:4px;height:10px'></div></td>";
                  html[html.length]="<td class='kjmenu_fillbox locMenu_frameTRC' width=10 style=''></td>";
                  html[html.length]="</tr></table>";
                  if(mitem.root.opt.page != "SiteHome"){html[html.length]="<div class='locMenu_body' style='width:"+(width-8)+"px'>";}
                  else{html[html.length]="<div class='locMenu_body2'>";}
                  html[html.length]="<table id='' class='kjmenu' cellpadding=0 cellspacing=0 border=0 width='100%'>";
                  html[html.length]="<tr><td nowrap='true' colspan='"+colspan+"'><div class='selectLabel'>"+mitem.root.opt.txt1+"<div></td></tr>";
                  if(mitem.root.opt.lv4Exists){
	                  html[html.length]="<tr><td nowrap='true' valign='top' width='23%'><div class='locLabel' id='locLabel1'>"+mitem.root.opt.level1Txt+"<div></td><td>&nbsp;</td><td nowrap='true' valign='top' width='23%'><div class='locLabel' id='locLabel2'>"+mitem.root.opt.level2Txt+"<div></td><td>&nbsp;</td><td nowrap='true' valign='top' width='23%'><div class='locLabel' id='locLabel3'>"+mitem.root.opt.level3Txt+"<div></td><td>&nbsp;</td><td nowrap='true' valign='top' width='23%'><div class='locLabel' id='locLabel4'>"+mitem.root.opt.level4Txt+"<div></td></tr>";
	                  html[html.length]="<tr><td nowrap='true' id='kjt_td' valign='top' width='23%'>&nbsp;</td><td>&nbsp;</td><td nowrap='true' id='kjt_td2' valign='top' width='23%'>&nbsp;</td><td>&nbsp;</td><td nowrap='true' id='kjt_td3' valign='top' width='23%'>&nbsp;</td><td>&nbsp;</td><td nowrap='true' id='kjt_td4' valign='top' width='23%'>&nbsp;</td></tr>";
	              }else{
                  	html[html.length]="<tr><td nowrap='true' valign='top' width='30%'><div class='locLabel' id='locLabel1'>"+mitem.root.opt.level1Txt+"<div></td><td>&nbsp;</td><td nowrap='true' valign='top' width='30%'><div class='locLabel' id='locLabel2'>"+mitem.root.opt.level2Txt+"<div></td><td>&nbsp;</td><td nowrap='true' valign='top' width='30%'><div class='locLabel' id='locLabel3'>"+mitem.root.opt.level3Txt+"<div></td></tr>";
                  	html[html.length]="<tr><td nowrap='true' id='kjt_td' valign='top' width='30%'>&nbsp;</td><td>&nbsp;</td><td nowrap='true' id='kjt_td2' valign='top' width='30%'>&nbsp;</td><td>&nbsp;</td><td nowrap='true' id='kjt_td3' valign='top' width='30%'>&nbsp;</td></tr>";
				  }
                  html[html.length]="<tr><td nowrap='true' colspan='"+colspan+"' style='text-align:center;'>";
                  html[html.length]="<div style='margin:15px 10px 10px 10px'>";
                  html[html.length]="<input type='button' id='updateBtn' class='newButton'  value='"+btntxt+"' >";
                  if(mitem.root.opt.page != "SiteHome"){
                  	html[html.length]="<a style='margin-left:10px' href='javascript:' id='cancelBtn'>"+mitem.root.opt.cancelTxt+"</a>";
                  }
                  html[html.length]="</div>";
                  html[html.length]="</td></tr>";
                  html[html.length]="</table>";
                  html[html.length]="</div>";
                  if(mitem.root.opt.page != "SiteHome"){
                        html[html.length]="<table height=4 width="+width+" cellpadding=0 cellspacing=0 border=0><tr>";
                  }
                  else{
                        html[html.length]="<table height=4 width=100% cellpadding=0 cellspacing=0 border=0><tr>";
                  }
                  html[html.length]="<td class='kjmenu_fillbox locMenu_frameBLC' width=10 style=''></td>";
                  html[html.length]="<td class='kjmenu_fillbox locMenu_frameB' style=''><div class='kjmenu_fillbox' style='width:4px;height:10px'></div></td>";
                  html[html.length]="<td class='kjmenu_fillbox locMenu_frameBRC' width=10 style=''></td>";
                  html[html.length]="</tr></table>";
                
                  html[html.length]="<div id='siteHomeFooter' style='text-align:center;'>&nbsp;</div>";
                  html[html.length]="</div>";
                  html[html.length]="</div>";
                  return html.join('');
            }
            ,locMenu_update : function (mitem) {
                  var maxindx = Math.min(mitem.maxitems,mitem.items.length);
                  var maxindx2 = Math.min(mitem.maxitems,mitem.items2.length);
                  var maxindx3 = Math.min(mitem.maxitems,mitem.items3.length);
                  var maxindx4 = Math.min(mitem.maxitems,mitem.items4.length);
                  var html = new Array();
                  var html2 = new Array();
                  var html3 = new Array();
                  var html4 = new Array();
                  var item, item2, item3,item4;
                  var pos3 = 0,posN2;
                  var sitehome = mitem.root.opt.page == "SiteHome";
                  if(maxindx3 != 0){$("#locLabel3").html(mitem.root.opt.level3Txt);}
                 
                  if(maxindx4 != 0){$("#locLabel4").html(mitem.root.opt.level4Txt);}
                 
                  for (var i=0; i<maxindx; i++) {
                        item = mitem.items[i];
                        html[html.length]="<div class='locMenu_item ";
                        if(!sitehome && maxindx2 == 0)html[html.length]="locMenu_over2 ";
                        //if(mitem.root.opt.clk == "true"){html[html.length]="kjmenu_clickable";}
                        html[html.length]="'>"+item.name+"</div>";
                  }
                  for (var i=0; i<maxindx2; i++) {
                        item2 = mitem.items2[i];
                        if(i==0){
                              if(maxindx2==2){var posF = 0;}
                              else{var posF = Math.floor(maxindx2/2);}
                              var posN = item2.pos-posF;
                              if(posN > 0){
                                    pos3 = item2.pos-posN;
                                    for(var j=0;j<posN;j++){
                                          html2[html2.length]="<div style='padding:2px 14px 5px 0px;'>&nbsp;</div>";
                                    }
                              }
                              else{pos3 = item2.pos}
                        }
                        html2[html2.length]="<div class='locMenu_item ";
                        if(!sitehome && maxindx3 == 0)html2[html2.length]="locMenu_over2 ";
                        html2[html2.length]="'>"+item2.name+"</div>";
                        posN2 = posN;
                  }
                  for (var i=0; i<maxindx3; i++) {
                        item3 = mitem.items3[i];
                        if(i==0){
                              if(maxindx3==2){var posF = 0;}
                              else{var posF = Math.floor(maxindx3/2);}
                              var posN = (item3.pos-pos3)-posF;
                              if(posN > 0){
                              		pos3 = parseInt(item3.pos) - parseInt(posN);
                                    for(var j=0;j<posN;j++){
                                          html3[html3.length]="<div style='padding:2px 14px 5px 0px;'>&nbsp;</div>";
                                    }
                              }else{pos3 = item3.pos}
                        }
                        html3[html3.length]="<div class='locMenu_item ";
                        if(!sitehome && maxindx4 == 0)html3[html3.length]="locMenu_over2 ";
                        html3[html3.length]="'>"+item3.name+"</div>";
                  }

                  for (var i=0; i<maxindx4; i++) {
                        item4 = mitem.items4[i];
                        if(i==0){
                              if(maxindx4==2){var posF = 0;}
                              else{var posF = Math.floor(maxindx4/2);}
                              var posN = (item4.pos-pos3)-posF;
                              if(posN > 0){
                                    for(var j=0;j<posN;j++){
                                          html4[html4.length]="<div style='padding:2px 14px 5px 0px;'>&nbsp;</div>";
                                    }
                              }
                        }
                        html4[html4.length]="<div class='locMenu_item locMenu_over2";
                        html4[html4.length]="'>"+item4.name+"</div>";
                  }
                  $("#kjt_td").html(html.join(''));
                  $("#kjt_td2").html(html2.join(''));
                  $("#kjt_td3").html(html3.join(''));
                  if(mitem.root.opt.lv4Exists){
                  	$("#kjt_td4").html(html4.join(''));
                  }
                  
                  $(this).initLoc(mitem,pos3,posN2);
                  
            }
            ,initLoc : function(mitem,pos3,posN2){
                  if(!posN2 || posN2 < 0){
                        posN2 = 0;
                   }
                  $("#kjt_td div.locMenu_item").each(function(i, el){
                        
                        var cmenu = mitem.items[i];
                        if(cmenu.id == mitem.root.opt.loc[0]){
                              $("#kjt_td div.locMenu_overA").removeClass("locMenu_overA");
                              $("#kjt_td div.locMenu_over2").removeClass("locMenu_over2");
                        	 if(mitem.items2.length != 0){$(this).addClass("locMenu_overA");}
                              else{$(this).addClass("locMenu_over2");}
                        }
                        if(mitem.root.opt.clk == "true"){$(this).css("text-decoration","underline");}
                        $(this).bind("click", cmenu, function (e) {
                              $("#locLabel3").html("");
                              $("#locLabel4").html("");
                              var cmenu = e.data;
                              var b = new Array();
                              var data = cmenu.root.opt.data;
                              for(var i=0; i<data.level1.length; i++){
                                    if(data.level1[i].id == cmenu.id && data.level1[i].level2){
                                          for(var j=0; j<data.level1[i].level2.length; j++){
                                         
                                                b[j] = data.level1[i].level2[j].name+"^"+data.level1[i].level2[j].id+"^"+data.level1[i].id+"^"+(i-pos3);
                                          }
                                          break;
                                    }
                              }
                              $("#kjt_td div.locMenu_overA").removeClass("locMenu_overA");
                              $("#kjt_td div.locMenu_over2").removeClass("locMenu_over2");
                              if(b.length == 0){$(this).addClass("locMenu_over2");}
                              else{$(this).addClass("locMenu_overA");}
                              if(mitem.root.opt.clk == "false"){$(this).css("cursor","default");}
                              $(this).updateL2(mitem.root.opt.data,cmenu);
                        });
                        $(this).bind("click", el, function (e) {
                              if(mitem.root.opt.clk == "false"){mitem.root.opt.clkableLink = false; return false;}
                              if (mitem.root.opt.selectedLoc) {
                                    mitem.root.opt.selectedLoc = cmenu.id;
                                    if(mitem.root.opt.page != "SiteHome"){$(document).trigger("mouseover");}
                              }
                        });
                  });
                  $("#kjt_td2 div.locMenu_item").each(function(i, el){
                        var cmenu2 = mitem.items2[i];
                        if(cmenu2.id == cmenu2.root.opt.loc[1]){
                        	  $("#kjt_td2 div.locMenu_overA").removeClass("locMenu_overA");
                              $("#kjt_td2 div.locMenu_over2").removeClass("locMenu_over2");
                              if(mitem.items3.length != 0){$(this).addClass("locMenu_overA");}
                              else{$(this).addClass("locMenu_over2");}
                        }
                        $(this).css("text-decoration","underline");
                        $(this).bind("click", cmenu2, function (e) {
                              $("#locLabel4").html("");
                              var cmenu2 = e.data;
                              var c = new Array();
                              var data = cmenu2.root.opt.data;
                              for(var i=0; i<data.level1.length; i++){
                                    if(data.level1[i].id == cmenu2.pid && data.level1[i].level2){
                                          for(var j=0; j<data.level1[i].level2.length; j++){
                                                if(data.level1[i].level2[j].id == cmenu2.id && data.level1[i].level2[j].level3){
                                                      for(var k=0; k<data.level1[i].level2[j].level3.length; k++){
                                                    
                                                            c[k] = data.level1[i].level2[j].level3[k].name+"^"+data.level1[i].level2[j].level3[k].id+"^"+data.level1[i].level2[j].id+"^"+(j+i-pos3);
                                                      }
                                                      break;
                                                }
                                          }
                                          break;
                                    }
                              }
                              $("#kjt_td2 div.locMenu_overA").removeClass("locMenu_overA");
                              $("#kjt_td2 div.locMenu_over2").removeClass("locMenu_over2");
                              if(c.length == 0){$(this).addClass("locMenu_over2");}
                              else{$(this).addClass("locMenu_overA");}
                              
                             $(this).updateL3(c,cmenu2,posN2);
                        });
                        $(this).bind("click", el, function () {
                        	  mitem.root.opt.clkableLink = true;
                              if (mitem.root.opt.selectedLoc) {
                                  	mitem.root.opt.selectedLoc = cmenu2.id;
                                    if(mitem.root.opt.page != "SiteHome"){$(document).trigger("mouseover");}
                              }
                        });
                  });
                  
                  $("#kjt_td3 div.locMenu_item").each(function(i, el){
                  
                        var cmenu3 = mitem.items3[i];
                        if(cmenu3.id == cmenu3.root.opt.loc[2]){
                             $("#kjt_td3 div.locMenu_overA").removeClass("locMenu_overA");
                             $("#kjt_td3 div.locMenu_over2").removeClass("locMenu_over2");
                        	 if(mitem.items4.length != 0){$(this).addClass("locMenu_overA");}
                              else{$(this).addClass("locMenu_over2");}
                        }
                        
                        $(this).css("text-decoration","underline");
                        $(this).bind("click", cmenu3, function (e) {
                              var cmenu3 = e.data;
                              if(mitem.root.opt.lv4Exists){//level4 exists
	                              var d = new Array();
	                              var data = cmenu3.root.opt.data;
	                              for(var i=0; i<data.level1.length; i++){
	                                    if(data.level1[i].id == mitem.items2[0].pid && data.level1[i].level2){//all elements in mitem.items2 has the selected province id. 
	                                          for(var j=0; j<data.level1[i].level2.length; j++){
	                                                if(data.level1[i].level2[j].id == cmenu3.pid && data.level1[i].level2[j].level3){
	                                                      for(var k=0; k<data.level1[i].level2[j].level3.length; k++){                                                       
	                                                            if(data.level1[i].level2[j].level3[k].id == cmenu3.id && data.level1[i].level2[j].level3[k].level4){
	                                                                  for(var l=0; l<data.level1[i].level2[j].level3[k].level4.length; l++){
	                                                                        d[l] = data.level1[i].level2[j].level3[k].level4[l].name+"^"+data.level1[i].level2[j].level3[k].level4[l].id+"^"+data.level1[i].level2[j].level3[k].id+"^"+(k+j+posN2-pos3);
	                                                                  }
	                                                                  break;
	                                                            }
	                                                      }
	                                                      break;
	                                                }
	                                          }
	                                          break;
	                                    }
	                              }
	                              $("#kjt_td3 div.locMenu_overA").removeClass("locMenu_overA");
	                              $("#kjt_td3 div.locMenu_over2").removeClass("locMenu_over2");
	                              
	                              if(d.length == 0){$(this).addClass("locMenu_over2");}
	                              else{$(this).addClass("locMenu_overA");}
	                              $(this).updateL4(d,cmenu3);
	                            //end if statement level4
	                            }else{ 
                                  	$("#kjt_td3 div.locMenu_over2").removeClass("locMenu_over2");
	                              	$(this).addClass("locMenu_over2");
	                            }
                        });
                        $(this).bind("click", el, function () {
                              mitem.root.opt.clkableLink = true;
                              if (mitem.root.opt.selectedLoc) {
                                    mitem.root.opt.selectedLoc = cmenu3.id;
                                    if(mitem.root.opt.page != "SiteHome"){$(document).trigger("mouseover");}
                              }
                        });
                  });
                  
                  if(mitem.root.opt.lv4Exists){//level4 exists
                  
	                  $("#kjt_td4 div.locMenu_item").each(function(i, el){
	                        var cmenu4 = mitem.items4[i];
	                        if(cmenu4.id == cmenu4.root.opt.loc[3]){
                                 $("#kjt_td4 div.locMenu_overA").removeClass("locMenu_overA");
                             	 $("#kjt_td4 div.locMenu_over2").removeClass("locMenu_over2");
	                              $(this).addClass("locMenu_over2");
	                        }
	                        $(this).css("text-decoration","underline");
	                        $(this).bind("click", cmenu4, function (e) {
	                              var cmenu4 = e.data;
	                              $("#kjt_td4 div.locMenu_over2").removeClass("locMenu_over2");
	                              $(this).addClass("locMenu_over2");
	                        });
	                        $(this).bind("click", el, function () {
	                              mitem.root.opt.clkableLink = true;
	                              if (mitem.root.opt.selectedLoc) {
	                                  
	                                    mitem.root.opt.selectedLoc = cmenu4.id;
	                                    if(mitem.root.opt.page != "SiteHome"){$(document).trigger("mouseover");}
	                              }
	                        });
	                  });
	             }
            }
            ,updateL2 : function(data,L1){
             
                  var ritem = L1.root;
                  var b = new Array();
                  var mitem2;
                  var pos3 = 0;
                  for(var i=0; i<data.level1.length; i++){
                        if(data.level1[i].id == L1.id && data.level1[i].level2){
                              for(var j=0; j<data.level1[i].level2.length; j++){
                                    b[j] = data.level1[i].level2[j].name+"^"+data.level1[i].level2[j].id+"^"+data.level1[i].id+"^"+i;
                              }
                              break;
                        }
                  }
                  ritem.items2a = [];
                  for (var i=0; i<b.length; i++) {
                        var parts = b[i].split("^");
                        mitem2 = {pid:parts[2],id:parts[1],name:parts[0],value:ritem.opt.lUrl+"?id="+parts[1]+"&CatId="+ritem.opt.cId+"&ruq="+ritem.opt.ref,pos:parts[3],root:ritem,parent:ritem};
                        ritem.items2a[ritem.items2a.length]=mitem2;
                  }
                  var maxindx2 = Math.min(ritem.maxitems,ritem.items2a.length);
                  
                  var html2 = new Array();
                  var item2,posN;              
                  for (var i=0; i<maxindx2; i++) {
                        item2 = ritem.items2a[i];
                        if(i==0){
                                    if(maxindx2==2){var posF = 0;}
                                    else{var posF = Math.floor(maxindx2/2);}
                                    posN = item2.pos-posF;
                                    if(posN > 0){
                                          pos3 = item2.pos-posN;
                                          for(var j=0;j<posN;j++){
                                                html2[html2.length]="<div style='padding:2px 14px 5px 0px;'>&nbsp;</div>";
                                          }
                                    }
                                    else{pos3 = item2.pos}
                        }
                        html2[html2.length]="<div class='locMenu_item";
                        html2[html2.length]="'>"+item2.name+"</div>";
                  }
                  $("#kjt_td2").html("");
                  $("#kjt_td3").html("");
                  $("#kjt_td4").html("");
                  $("#kjt_td2").html(html2.join(''));
                  $("#kjt_td2 div.locMenu_item").each(function(i, el){
                        var cmenu2 = ritem.items2a[i];
                        $(this).css("text-decoration","underline");
                        $(this).addClass("locMenu_over2");
                        $(this).bind("click", cmenu2, function (e) {
                         
                              $("#locLabel4").html("");
                              var cmenu2 = e.data;
                              var c = new Array();
                              var data = ritem.root.opt.data;
                              for(var i=0; i<data.level1.length; i++){
                              
                                    if(data.level1[i].id == cmenu2.pid && data.level1[i].level2){
                                          for(var j=0; j<data.level1[i].level2.length; j++){
                                                if(data.level1[i].level2[j].id == cmenu2.id && data.level1[i].level2[j].level3){
                                                      for(var k=0; k<data.level1[i].level2[j].level3.length; k++){
                                                            c[k] = data.level1[i].level2[j].level3[k].name+"^"+data.level1[i].level2[j].level3[k].id+"^"+data.level1[i].level2[j].id+"^"+(j+i-pos3);
                                                      }
                                                      break;
                                                }
                                          }
                                          break;
                                    }
                              }
                              $("#kjt_td2 div.locMenu_overA").removeClass("locMenu_overA");
                              $("#kjt_td2 div.locMenu_over2").removeClass("locMenu_over2");
                              if(c.length == 0){$(this).addClass("locMenu_over2");}
                              else{$(this).addClass("locMenu_overA");}
                            
                              $(this).updateL3(c,cmenu2,posN);
                        });
                        $(this).bind("click", el, function () {
                              ritem.root.opt.clkableLink = true;
                              if (ritem.root.opt.selectedLoc) {
                                    ritem.root.opt.selectedLoc = cmenu2.id;
                                    if(ritem.root.opt.page != "SiteHome"){$(document).trigger("mouseover");}
                              }
                        });
                  });
            }
      
            ,updateL3 : function(c,L2,posN2){
                  var ritem = L2.root;
                  var mitem3;
                  var pos3 = 0;
                  ritem.items3a = [];
                  for (var i=0; i<c.length; i++) {
                        var parts = c[i].split("^");
                        mitem3 = {pid:parts[2],id:parts[1],name:parts[0],value:ritem.opt.lUrl+"?id="+parts[1]+"&CatId="+ritem.opt.cId+"&ruq="+ritem.opt.ref,pos:parts[3],root:ritem,parent:ritem};
                        
                        ritem.items3a[ritem.items3a.length]=mitem3;
                  }
                  var maxindx3 = Math.min(ritem.maxitems,ritem.items3a.length);
                  var maxindx4 = Math.min(ritem.maxitems,ritem.items4a.length);
                  
                   if(!posN2 || posN2 < 0){
                        posN2 = 0;
                   }
                  
                  var html3 = new Array();
                  var item3;
                  if(maxindx3 == 0){$("#locLabel3").html("");}
                  else{$("#locLabel3").html(ritem.opt.level3Txt);}
                 
                  
                  for (var i=0; i<maxindx3; i++) {
                        item3 = ritem.items3a[i];
                        if(i==0){
                              if(maxindx3==2){var posF = 0;}
                              else{var posF = Math.floor(maxindx3/2);}
                              var posN = item3.pos-posF;
                              if(posN > 0){
                                  pos3 = parseInt(item3.pos) - parseInt(posN);
                                    for(var j=0;j<posN;j++){
                                          html3[html3.length]="<div style='padding:2px 14px 5px 0px;'>&nbsp;</div>";
                                    }
                              }else{pos3 = item3.pos}
                        }
                        html3[html3.length]="<div class='locMenu_item";
                        html3[html3.length]="'>"+item3.name+"</div>";
                  }
                  $("#kjt_td3").html("");
                  if(ritem.opt.lv4Exists){//if level4 
                  	$("#kjt_td4").html("");
                  }//end if
                  $("#kjt_td3").html(html3.join(''));
                  
                  $("#kjt_td3 div.locMenu_item").each(function(i, el){
                        var cmenu3 = ritem.items3a[i];
                              
                        $(this).css("text-decoration","underline");
                         $(this).addClass("locMenu_over2");
                         $(this).bind("click", cmenu3, function (e) {
                              var cmenu3 = e.data;
                              if(ritem.opt.lv4Exists){//if level4 
	                              var d = new Array();
	                              var data = ritem.root.opt.data;
	                              for(var i=0; i<data.level1.length; i++){
	                                    if(data.level1[i].id == L2.pid && data.level1[i].level2){
	                                          for(var j=0; j<data.level1[i].level2.length; j++){
	                                                if(data.level1[i].level2[j].id == cmenu3.pid && data.level1[i].level2[j].level3){
	                                                      for(var k=0; k<data.level1[i].level2[j].level3.length; k++){
	                                                            if(data.level1[i].level2[j].level3[k].id == cmenu3.id && data.level1[i].level2[j].level3[k].level4){
	                                                                  for(var l=0; l<data.level1[i].level2[j].level3[k].level4.length; l++){
	                                                                  		d[l] = data.level1[i].level2[j].level3[k].level4[l].name+"^"+data.level1[i].level2[j].level3[k].level4[l].id+"^"+data.level1[i].level2[j].level3[k].id+"^"+(k+j+posN2-pos3);
	                                                                  }
	                                                                  break;
	                                                            }                                         
	                                                      }
	                                                      break;
	                                                }
	                                          }
	                                          break;
	                                    }
	                              }
	                              $("#kjt_td3 div.locMenu_overA").removeClass("locMenu_overA");
	                              $("#kjt_td3 div.locMenu_over2").removeClass("locMenu_over2");
	                              if(d.length == 0){$(this).addClass("locMenu_over2");}
	                              else{$(this).addClass("locMenu_overA");}
	                              $(this).updateL4(d,cmenu3); 
	                              
	                               //end if  
	                          }else {                   
                               	$("#kjt_td3 div.locMenu_over2").removeClass("locMenu_over2");
                              	$(this).addClass("locMenu_over2");
                              }
                              
                        });
                        $(this).bind("click", el, function () {
                              ritem.root.opt.clkableLink = true;
                              if (ritem.root.opt.selectedLoc) {
                                    ritem.root.opt.selectedLoc = cmenu3.id;
                                    if(ritem.root.opt.page != "SiteHome"){$(document).trigger("mouseover");}
                              }
                        });
                  });
            },
           
            updateL4 : function(d,L3){
            var ritem = L3.root;
            var mitem4;
            ritem.items4a = [];
            for (var i=0; i<d.length; i++) {
                  var parts = d[i].split("^");
                  mitem4 = {pid:parts[2],id:parts[1],name:parts[0],value:ritem.opt.lUrl+"?id="+parts[1]+"&CatId="+ritem.opt.cId+"&ruq="+ritem.opt.ref,pos:parts[3],root:ritem,parent:ritem};
                  ritem.items4a[ritem.items4a.length]=mitem4;
            }
            var maxindx4 = Math.min(ritem.maxitems,ritem.items4a.length);
            var html4 = new Array();
            var item4;
            if(maxindx4 == 0){$("#locLabel4").html("");}
            else{$("#locLabel4").html(ritem.opt.level4Txt);}
            for (var i=0; i<maxindx4; i++) {
                  item4 = ritem.items4a[i];
                  if(i==0){
                        if(maxindx4==2){var posF = 0;}
                        else{var posF = Math.floor(maxindx4/2);}
                        var posN = item4.pos-posF;
                        if(posN > 0){
                              for(var j=0;j<posN;j++){
                                    html4[html4.length]="<div style='padding:2px 14px 5px 0px;'>&nbsp;</div>";
                              }
                        }
                  }
                  html4[html4.length]="<div class='locMenu_item";
                  html4[html4.length]="'>"+item4.name+"</div>";
            }
            $("#kjt_td4").html("");
            $("#kjt_td4").html(html4.join(''));
            $("#kjt_td4 div.locMenu_item").each(function(i, el){
                  var cmenu4 = ritem.items4a[i];
                  $(this).css("text-decoration","underline");
                   $(this).addClass("locMenu_over2");
                  $(this).bind("click", cmenu4, function (e) {
                        var cmenu4 = e.data;
                        $("#kjt_td4 div.locMenu_over2").removeClass("locMenu_over2");
                        $(this).addClass("locMenu_over2");
                  });
                  $(this).bind("click", el, function () {
                  		ritem.root.opt.clkableLink = true;
                        if (ritem.root.opt.selectedLoc) {
                              ritem.root.opt.selectedLoc = cmenu4.id;
                              if(ritem.root.opt.page != "SiteHome"){$(document).trigger("mouseover");}
                        }
                  });
            });
      },
        containsElement : function(container,element) {
            while ((element !== null) && (element.id !== container.id) && (element.parentNode !== null)) { element = element.parentNode; }
            return (element.id === container.id);
		},
        
            locMenu_remove : function (drill) {
                  var mitem = this.get(0);
                  if (mitem.el) {
                        if(mitem.el.parent().length != 0){$("#changeLocDiv").css("background-image","url(http://pic.classistatic.com/image/pics/classifieds/button_open.gif)");}//ignore img name, arrow points down
                        mitem.el.remove();
                        mitem.el=null;
                        if (drill && typeof(mitem.items)!="undefined") {
                              for(var i=0; i<mitem.items.length; i++) {
                                    if (mitem.items[i].el){
                                          $(mitem.items[i]).locMenu_remove(mitem.items[i], true);
                                    }
                              }
                        }
                  }
            }
      });               
})(jQuery);

