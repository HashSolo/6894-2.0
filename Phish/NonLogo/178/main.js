var isIE = false, is_ie6 = false, is_ie7 = false;

var ua = navigator.userAgent.toLowerCase();
isIE = (ua.indexOf("msie") != -1 && ua.indexOf("opera") == -1 && ua.indexOf("webtv") == -1);
if (isIE){
  var is_major = parseInt(navigator.appVersion);
  is_ie6 = (isIE && (is_major == 4) && (ua.indexOf("msie 6.")!=-1) );
  is_ie7 = (isIE && (is_major == 4) && (ua.indexOf("msie 7.")!=-1) );
}  

window.onload = function() {  
  resizeMainPart();  
  resizeBlockServices();
  resizeWindow();

}

window.onresize = function() { 
  resizeMainPart(); 
  resizeBlockServices(); 
  resizeWindow();
}

window.onscroll = function() { 
  drawHeaderTarif(); 
}

function resizeBlockServices(){
   			   
  var tarif_main = document.getElementById('tarif_main');
  var servises = document.getElementById('servises');
  if (tarif_main && servises){
    var x;	  	  
    if (self.innerWidth) x = self.innerWidth; // общий синтаксис	      	
    else if (document.documentElement && document.documentElement.clientWidth) x = document.documentElement.clientWidth;	// IE 6 Strict Mode	  
    else if (document.body) x = document.body.clientWidth; // Остальные версии IE	
    
    if (x < 865) { //900
      servises.style.margin = '0px';      
      tarif_main.style.width = '100%';
      
      var items = xGetElementsByClassName('tarif_link', document.getElementById('tarif_main'), 'div');
      for (var i = 0; i < items.length; ++i)      
        items[i].style.bottom = '30px';    
    }
    else{
      servises.style.margin = '0px 30px';      
      tarif_main.style.width = '';                                                            
      tarif_main.style.width = document.getElementById('block_tarif').clientWidth * 3 + 54 +'px';
  
      var items = xGetElementsByClassName('tarif_link', document.getElementById('tarif_main'), 'div');
      for (var i = 0; i < items.length; ++i)
        items[i].style.bottom = '';
    }           
  }    
}

function changeSubmenuWidth_ie(type){
    
  var menu;
  if (is_ie6){    
    menu = xGetElementsByTagName('table', document.getElementById('menu_'+type));
    if (menu) menu = menu[0];
   } 
  else{    
    menu = document.getElementById('menu_'+type);
  }
  
  var submenu = document.getElementById('submenu_'+type);
    
  if (menu){
     
    if (is_ie6) menu.style.visibility = 'visible';
    else {       
      submenu.style.display = 'block';
      submenu.style.width = '0';
    }   
    
    var items = xGetElementsByClassName('menu_item', menu, 'div');
    if (items.length > 0){        
       
      var item_width = items[0].clientWidth;
            
      for (var i = 1; i < items.length; ++i){
        if (items[i].clientWidth > item_width)                               
          item_width = items[i].clientWidth;
      }      
      submenu.style.width = item_width + 'px';
                                                       		
    } // items.length > 0
    
    if (is_ie6) menu.style.visibility = '';
    else submenu.style.display = '';
    
  } // menu.length > 0     
}


function changeSubmenuWidths(){
  if (isIE){
    changeSubmenuWidth_ie('about_company');
    changeSubmenuWidth_ie('services');
    changeSubmenuWidth_ie('spec_offer');        
    changeSubmenuWidth_ie('support');
  } 
}
 
function resizeMainPart() {

  var main_part = document.getElementById('main_part');
  if (main_part){  
    var windowHeight = getPageSize()['windowHeight'];
    var pageHeight   = getPageSize()['pageHeight'];  
  
    if (windowHeight < 550){
      main_part.style.minHeight = '57%';
      if (isIE){
        if (pageHeight > windowHeight) main_part.style.height = '';
        else main_part.style.height = '65%'; 
      }        
    }
    else if (windowHeight < 700){  
      main_part.style.minHeight = '69%';
      if (isIE){
        if (pageHeight > windowHeight) main_part.style.height = '';
        else main_part.style.height = '74%';       
      }          
    }
    else{
      var windowWidth = getPageSize()['windowWidth'];
  
      if (windowWidth > 1280) main_part.style.minHeight = '74%';
      else main_part.style.minHeight = '77%';
      
      if (isIE) {    
        if (pageHeight > windowHeight) main_part.style.height = '';
        else main_part.style.height = '80%';      
      }  
    }
  }
  //changeSubmenuWidths();  
}

/* -------------------------------------------------------------------------- */
/*                             Popup                                          */
/* -------------------------------------------------------------------------- */
function showPopup(name_popup){
 
  var popup = document.getElementById('popup');   
  if (popup){
    top_otstup = self.pageYOffset || (document.documentElement && document.documentElement.scrollTop) || (document.body && document.body.scrollTop);
    
    var y;	  	
    if (self.innerHeight) y = self.innerHeight; // общий синтаксис	      	  	
    else if (document.documentElement && document.documentElement.clientHeight) y = document.documentElement.clientHeight;	// IE 6 Strict Mode
   	else if (document.body) y = document.body.clientHeight;  // Остальные версии IE
   	  
    popup.className = 'show';
    
    var popup_content = document.getElementById(name_popup);
    popup_content.className = 'popup_content show';
        
    var height_popup = popup_content.offsetHeight;   
    popup.style.top = Math.round((y - height_popup)/2) - 10 + top_otstup + 'px';  
    
    resizeWindow();
      
    var obertka = document.getElementById('obertka');
      
    //obertka.style.background = '#000';   
    obertka.style.opacity      = '0.6'; //для DOM браузеров 
    obertka.style.MozOpacity   = '0.6'; //(для старых Mozilla 1.6 и ниже, Firefox 0.8)
    obertka.style.KhtmlOpacity = '0.6'; //(для Safari 1.1, Konqueror 3.1)
    obertka.style.filter ="progid:DXImageTransform.Microsoft.Alpha(opacity=60)"; // ИЕ      
    obertka.style.height = getPageSize()['pageHeight'] + 'px';
  	obertka.className = 'show';
  	
    return false;
  }
}

function resizeWindow(){  

  var obertka = document.getElementById('obertka');
  var popup = document.getElementById('popup');

  if (obertka && popup){
    var x;	  	  
    if (self.innerWidth) x = self.innerWidth; // общий синтаксис	      	
    else if (document.documentElement && document.documentElement.clientWidth) x = document.documentElement.clientWidth;	// IE 6 Strict Mode	  
    else if (document.body) x = document.body.clientWidth; // Остальные версии IE	
    
    if (x < 800)	{ // 1024
      if (popup) popup.style.left = '50px'; 
      obertka.style.width = '775px'; //1002
      obertka.style.height = document.body.scrollHeight;		    
    }
    else{
      if (popup) popup.style.left = '172px';    
    	obertka.style.width = document.body.offsetWidth;
      obertka.style.height = document.body.scrollHeight;    
      resizeHeaderTarif();
    }
    drawHeaderTarif();
  }
}


function resizeHeaderTarif(){

  var header_tarif = document.getElementById('header_tarif');
  if (header_tarif) {
    top_otstup = self.pageYOffset || (document.documentElement && document.documentElement.scrollTop) || (document.body && document.body.scrollTop);  
    if (top_otstup > 295){            
      header_tarif.style.position = 'static';       
      header_tarif.style.backgroundColor = '';    
      header_tarif.style.width = '';      
    }
  }        
}

function hidePopup(){
  var popup = document.getElementById('popup');
  if (popup.className == 'show'){
    popup.className = 'hidden';
    var popups = xGetElementsByClassName('popup_content', document.getElementById('popup_main_content'), 'div');
    for (var i = 0; i < popups.length; ++i) {
      popups[i].className = 'popup_content hidden';
    }		
  	
  	var obertka = document.getElementById('obertka');
  	//obertka.style.backgroundColor = '#000';
  	obertka.style.opacity      = '1'; //для DOM браузеров 
  	obertka.style.MozOpacity   = '1'; //(для старых Mozilla 1.6 и ниже, Firefox 0.8)
  	obertka.style.KhtmlOpacity = '1'; //(для Safari 1.1, Konqueror 3.1)
  	obertka.style.filter = "progid:DXImageTransform.Microsoft.Alpha(opacity=100)"; // ИЕ
  	obertka.className = 'hidden';
 	}
  return false;
}

function xGetElementsByClassName(clsName, parentEle, tagName) {
  var found = new Array();
  var re = new RegExp('\\b'+clsName+'\\b', 'i');
  var list = xGetElementsByTagName(tagName, parentEle);
  for (var i = 0; i < list.length; ++i) {
    if (list[i].className.search(re) != -1) {
      found[found.length] = list[i];
    }
  }
  return found;
}

function xGetElementsByTagName(tagName, parentEle){
  var list = null;
  tagName = tagName || '*';
  parentEle = parentEle || document;  

  if (isIE) {
    if (tagName == '*') list = parentEle.all;
    else list = parentEle.all.tags(tagName);
  }
  else if (parentEle.getElementsByTagName) list = parentEle.getElementsByTagName(tagName);
  return list || new Array();
}

function drawHeaderTarif(){

  var header_tarif = document.getElementById('header_tarif');
    
  if (header_tarif){
    var top_otstup = self.pageYOffset || (document.documentElement && document.documentElement.scrollTop) || (document.body && document.body.scrollTop);
    
    top_otstup = top_otstup; //25
    
   
    if (top_otstup > 360){ //270 284
  
      header_tarif.style.height = '65px';
      
      header_tarif.style.position = 'absolute';
      
      header_tarif.style.top = top_otstup + 'px';
      header_tarif.style.left = '30px';      
               
      var x;	  	
      if (self.innerWidth) x = self.innerWidth; // общий синтаксис	      	  	
      else if (document.documentElement && document.documentElement.clientWidth) x = document.documentElement.clientWidth;	// IE 6 Strict Mode
     	else if (document.body) x = document.body.clientWidth;  // Остальные версии IE
      
      if (x < 778){ //1002
      	if (isIE) header_tarif.style.width = '708px';        //755      
      	else header_tarif.style.width = '714px';           //762
      }  
      else{
        if (isIE) header_tarif.style.width = x-60+'px'; //x-237+'px';      
        else header_tarif.style.width = x-77+'px';  //x-255+'px'; 254
      }    
          
    }
    else{     
      header_tarif.style.height = '';
      header_tarif.style.position = 'static';
      header_tarif.style.backgroundColor = '';    
      header_tarif.style.width = '';          
    } 
  } 
}

function  getPageSize(){
	var xScroll, yScroll;

	if (window.innerHeight && window.scrollMaxY) {
		xScroll = document.body.scrollWidth;
		yScroll = window.innerHeight + window.scrollMaxY;
	} else if (document.body.scrollHeight > document.body.offsetHeight){ // all but Explorer Mac
		xScroll = document.body.scrollWidth;
		yScroll = document.body.scrollHeight;
	} else if (document.documentElement && document.documentElement.scrollHeight > document.documentElement.offsetHeight){ // Explorer 6 strict mode
		xScroll = document.documentElement.scrollWidth;
		yScroll = document.documentElement.scrollHeight;
	} else { // Explorer Mac...would also work in Mozilla and Safari
		xScroll = document.body.offsetWidth;
		yScroll = document.body.offsetHeight;
	}

	var windowWidth, windowHeight;
	if (self.innerHeight) { // all except Explorer
		windowWidth = self.innerWidth;
		windowHeight = self.innerHeight;
	} else if (document.documentElement && document.documentElement.clientHeight) { // Explorer 6 Strict Mode
		windowWidth = document.documentElement.clientWidth;
		windowHeight = document.documentElement.clientHeight;
	} else if (document.body) { // other Explorers
		windowWidth = document.body.clientWidth;
		windowHeight = document.body.clientHeight;
	}

	// for small pages with total height less then height of the viewport
	if(yScroll < windowHeight){
		pageHeight = windowHeight;
	} else {
		pageHeight = yScroll;
	}

	// for small pages with total width less then width of the viewport
	if(xScroll < windowWidth){
		pageWidth = windowWidth;
	} else {
		pageWidth = xScroll;
	}

	return {'pageWidth':pageWidth, 'pageHeight':pageHeight, 'windowWidth':windowWidth, 'windowHeight':windowHeight};
}

/* -------------------------------------------------------------------------- */
function moveRightMenu(item, index){
  if (is_ie6){
   var item = document.getElementById(item);
   item.style.zIndex = index;    
  }
}

/* -------------------------------------------------------------------------- */
/*                  CLASS "Chooser" with round button                         */
/* -------------------------------------------------------------------------- */
var is_selector_hover = false;
var is_selector_open = false;
var timer = null;
var obj = new Array();

function Chooser() {
  
  var obj_;
  var choose_name, choose_items, count_elems_in_column;  
  
  var is_exist_selector = false;
  var dd_height_before = 0, dd_height_after = 0, main_choose_height_before = 0, main_choose_height_after = 0;  
  var class_name_block, class_name_a;
  var width_a;
  var is_resize_selector = true;
  var is_resize_inner_div = false;    
  
  this.create_ = function(choose_name_, choose_items_, default_value_, count_elems_in_column_, width_main, width_a_, width_selector, shift_left) {
  
    this.defineParams(choose_name_, choose_items_, count_elems_in_column_);
        
    var default_value = default_value_ ? default_value_ : getFirstKeyHash(choose_items);
    width_a = width_a_;
        
    obj[choose_name] = this.obj_;
    
    this.defineSelectedClassName('block_selector', 'selected_item_choose');
    
    if (!width_main && !width_selector) width_selector = '216px';     
           
    if (document.getElementById('main_choose_'+ choose_name)){ 
      
      defineChooserSize();
      
      var style_main = ""; 
      if (width_main) style_main = style_main + "width:"+ width_main +";";
            
      var style_ = ""; 
      if (width_selector) style_ = style_ + "width:"+ width_selector +";";
      if (shift_left) style_ = style_ + "margin-left: -"+ shift_left +";";
      
      var hidden_name = document.getElementById('term_sale_' +choose_name) ? document.getElementById('term_sale_' +choose_name).name : "choose_"+ choose_name;
       
      document.getElementById('main_choose_' + choose_name).innerHTML =         
      "<input type='hidden' name='"+ hidden_name +"' id='choose_"+ choose_name +"' value='"+ default_value +"'>" +
      "<a name='a_"+ choose_name +"' class='hidden'>&nbsp;</a>" +
      "<div style='padding-right: 18px !important;'>" +      
      "<div id='block_choose_"+ choose_name + "' class='block_choose'"+ (style_main ? "style='"+ style_main +"'" : "") +" onMouseOut='if (is_selector_open) obj[\""+ choose_name_ +"\"].exitSelector();'>" +                      
      "  <a href='#a_"+ choose_name +"' id='a_choose_"+ choose_name + "'" +                          
      "     onClick='if (is_selector_open) obj[\""+ choose_name_ +"\"].closeSelector_AcceptValue();  else{ choose_name_=\""+ choose_name_ +"\"; obj[\""+ choose_name_ +"\"].showSelector();} return false;'>" +
      "     <span class='a_choose_selected' id='a_choose_selected_"+ choose_name+ "'><span>"+ choose_items_[default_value] + "</span></span><div></div></a>" +
      "  <div id='block_selector_"+ choose_name + "' class='hidden'"+ (style_ ? "style='"+ style_ +"'" : "") +" onMouseOver='setSelectorHover();' onMouseOut='unsetSelectorHover(); obj[\""+ choose_name +"\"].exitSelector();'>" +      
      "  </div>" +    
      "</div></div>";                  
    }
    if (is_resize_selector) createSelector();
  };
  
  this.showSelector = showSelector;
  this.exitSelector = exitSelector;
  this.closeSelector_AcceptValue = closeSelector_AcceptValue;
  this.createSelector = createSelector;
  //this.clearSelectorItems = clearSelectorItems;
  
  //function clearSelectorItems(){ is_exist_selector = false; };
  this.unsetResizeSelector = function(){
    is_resize_selector = false;
  }; 

          
  this.defineSelectedClassName = function(class_name_block_, class_name_a_){
    class_name_block = class_name_block_;
    class_name_a = class_name_a_;                
  }; 
    
  this.defineParams = function(choose_name_, choose_items_, count_elems_in_column_){    
    choose_name = choose_name_;
    choose_items = choose_items_;
    if (count_elems_in_column_) count_elems_in_column = count_elems_in_column_;
  }; 
    
  function showSelector(e){      
             
      is_selector_open = true;
      
      window.clearTimeout(timer);
         
      e = e || window.event;
  
      var block_selector = document.getElementById('block_selector_' + choose_name);
      block_selector.className = class_name_block;
      
      if ((class_name_block == 'block_selector_domain') && isIE)
        block_selector.style.visibility = 'visible';
        
      if (is_ie7) block_selector.style.visibility = 'visible'; //patch for IE7 for not recover
                
      if ( is_exist_selector == false )
        createSelector();                   
      
      block_selector.className = class_name_block;
      /*
      if (isIE) {
        block_selector.style.background = 'url(i/n.gif) repeat-x';
        block_selector.style.backgroundColor = '#F7F7ED';
      }        
      */
      if (e && typeof e.preventDefault != 'undefined') e.preventDefault();
      
      if (is_resize_selector) resizeChooserHeight(true);
      
      if ( ((choose_name == 'whois') || (choose_name == 'domain')) && !is_resize_inner_div ) 
        resizeInnerDiv();       
             
  }  
  
  function createSelector(){
      is_exist_selector = true;
    
      var block_selector = document.getElementById('block_selector_' + choose_name);      
            
      if ( (getLengthHash(choose_items) > count_elems_in_column) || (choose_name == 'whois') ){
          
           block_selector.style.top = '30px';
           
           var inner_div;           
           // ----- create inner divs -----              
           var count_columns = parseInt(getLengthHash(choose_items)/count_elems_in_column);
           if ((getLengthHash(choose_items) % count_elems_in_column) > 0) count_columns++;
                
           for (var i = 1; i <= count_columns; ++i){                               
             inner_div = document.createElement('div');        
             inner_div.className= 'left_';
             inner_div.style.marginRight = '10px'; 
             
             inner_div.id = 'inner_choose_'+ choose_name +'_column'+i;
            
             //addHandler(inner_div, 'click', clickOnDiv);
             addHandler(inner_div, 'mouseover', setSelectorHover);
             //addHandler(inner_div, 'mouseout', unsetSelectorHover);
                            
             block_selector.appendChild(inner_div);      
          }
                     
          // ----- create elements in inner divs -----        
          var num = 1;		
          var div_id;
          var i = 1;
          for (var k in choose_items){        	                     
            if (i > count_elems_in_column) {
              num = parseInt(i/count_elems_in_column);              
              if ((i % count_elems_in_column) > 0) { num++ };
            } 
            div_id = 'inner_choose_'+ choose_name +'_column'+num;

            inner_div = document.getElementById(div_id);

            if (inner_div)                                  
              inner_div.appendChild(createSelectedItem(k, choose_items[k]));
              //inner_div.appendChild(createSelectedItem(choose_items[i-1]));
              
            ++i;                          
          }  
                               
      } // end if (choose_items.length > count_elems_in_column)
      else{         
        for (var k in choose_items)
          block_selector.appendChild(createSelectedItem(k, choose_items[k]));
      }      	 
  }
  
  function createSelectedItem(key, value){
  
      var a = document.createElement('a');          
      a.style.color='#fff'; //ie 7&8 fix for link color      
      a.id = "a_selector_"+ choose_name +"_"+ key ;
      a.href = '#a_'+ choose_name;// +"_"+ key ;       
      if (width_a) a.style.width = width_a +'px';        
      
      if (key == document.getElementById('choose_'+ choose_name).value)                  
        a.className = class_name_a;      
      
      var div_ = document.createElement('div');                                     
      a.appendChild(div_);
      
      a.appendChild(document.createTextNode(value));
      
      addHandler(a, 'click', closeSelector);
      addHandler(a, 'mouseover', setSelectorHover);      
      //addHandler(a, 'mouseout', unsetSelectorHover);
      
      return a;
      
      function closeSelector(e){
      
        e = e || window.event;
        
        if (e.target) targ = e.target;                  // Mozila
      	else if (e.srcElement) targ = e.srcElement;     // IE      	 
      	if (targ.nodeType == 3) targ = targ.parentNode; // defeat Safari bug
      		       
        unsetSelectorHover(); // ???
        
        var a_items = xGetElementsByClassName(class_name_a, document.getElementById('block_selector_'+ choose_name), 'a');         
        if (a_items[0]) a_items[0].className = '';
        
        targ.className = class_name_a;
        
        closeSelector_AcceptValue(key, value);
        if (e && typeof e.preventDefault != 'undefined')
          e.preventDefault();
        
        return false;
            
      }  
  }

  function resizeInnerDiv(){
    
    var width = 0; 
    var width_inner_div;
    var i = 1;
    var min_width_inner_div = 60; 
    var width_inner_div; 
    
    while (inner_div = document.getElementById('inner_choose_'+ choose_name +'_column'+i)){
      width_inner_div = document.getElementById('inner_choose_'+ choose_name +'_column'+i).clientWidth;
      if (width_inner_div < min_width_inner_div) width_inner_div = min_width_inner_div;                            
      resizeWidthAinDiv(inner_div, width_inner_div);      
      width = width + width_inner_div;      
      ++i;        
    }     
    
    var block_selector = document.getElementById('block_selector_' + choose_name);
    if ((width > 206) || (i>3) ){
      //alert('i = ' + i);             
      block_selector.style.width = width + (i-1)*18 +(is_ie6 ? 5 : 0)+ 'px';     
      block_selector.style.right = -15 - (is_ie6 ? 6 : 0) - 3*(i-1) + 'px'; //-6      
    }   
    is_resize_inner_div = true;        
  }

  function resizeWidthAinDiv(inner_div, new_width){
    var list = xGetElementsByTagName('a', inner_div);
    for (var i = 0; i < list.length; ++i) {
      list[i].style.width = new_width + 'px';               
    }    
  }
  
  function closeSelector_AcceptValue(key, value) {      
       
      if (!is_selector_hover) {
        
        is_selector_open = false;
        
        if (value){ 
          document.getElementById('choose_' + choose_name).value = key;
          document.getElementById('a_choose_selected_' + choose_name).innerHTML = value;
        }
        
        var block_selector = document.getElementById('block_selector_' + choose_name);
        
        if ((class_name_block == 'block_selector_domain') && isIE)                   
          block_selector.style.visibility = 'hidden';             
        else  
          block_selector.className = 'hidden'; 
                 
        if (is_resize_selector) resizeChooserHeight(false);    
      }
      return false;  
  }


  function exitSelector() {             
      timer = setTimeout("obj[\""+ choose_name +"\"].closeSelector_AcceptValue()", 500);       
  }

  /* ---------------------------- Resize Сhooser ----------------------------------- */  
  
  function resizeChooserHeight(is_increase){    
    
    var dd_m = document.getElementById('dd_m_' + choose_name);
    
    if (getLengthHash(choose_items) > 2){
      var count_buts_in_column = (getLengthHash(choose_items) > count_elems_in_column) ? count_elems_in_column : getLengthHash(choose_items);   
    
      if (dd_m){      
        var main_choose = document.getElementById('main_choose_' + choose_name);        
         
        if (is_increase){
        
          if (dd_height_after > 0){
            dd_m.style.height = dd_height_after + 'px';
            main_choose.style.height = main_choose_height_after + 'px';
          }  
          else{        
            dd_height_before = dd_m.clientHeight;                    
            dd_height_after = dd_height_before + 21*(count_buts_in_column - 2);
                        
            if (is_ie7) dd_height_after = dd_height_after + 21;                        
            dd_m.style.height = dd_height_after + 'px';
          
            main_choose_height_before = main_choose.clientHeight;
            main_choose_height_after = main_choose_height_before + 21*count_buts_in_column;        
            main_choose.style.height = main_choose_height_after + 'px';            
          }
        } // increase
        else{                  
          main_choose.style.height = main_choose_height_before + 'px';

          if (isIE)          
            dd_m.style.height = (dd_height_before - 21) + 'px';     
          else
            dd_m.style.height = (dd_height_before - 21*2)  + 'px';            
        } // decrease
      } // dd_m exists
      
      else{
                            
        var td_name = choose_name.substr(0, choose_name.indexOf("_"));        
        var td = document.getElementById('td_' + td_name);
        if (td){
        
          if (is_increase){
          
            if (dd_height_after > 0){
              td.style.height = dd_height_after + 'px';              
            }  
            else{        
              dd_height_before = td.clientHeight;                    
              dd_height_after = dd_height_before + 21*(count_buts_in_column);
              
              if (getLengthHash(choose_items) > count_elems_in_column) dd_height_after = dd_height_after + 50;
                                                                
              td.style.height = dd_height_after + 'px';            
            }
          } // increase
          else{                                
            td.style.height = dd_height_before + 'px';            
          } // decrease
        
        } // td exists           
      
      } 
      
    } // count of items > 2
  }
  
  function defineChooserSize(){        
    var dd_m = document.getElementById('dd_m_' + choose_name);  
    if (dd_m){        
      if (isIE){
        if (is_ie6)      
          dd_m.style.height = '65px';        
        else            
          dd_m.style.minHeight = '34px';        
      }
      else     
        dd_m.style.minHeight = '34px';
    }    
  }
  
} // end of class 'Chooser'

function setSelectorHover(){    
  is_selector_hover = true;  
}
function unsetSelectorHover(){   
  is_selector_hover = false; 
}



/* -------------------------------------------------------------------------- */
/*              CLASS "DomainChooser" with round button                       */
/* -------------------------------------------------------------------------- */
/*
var domains = new Array( '.ru', '.su', '.com', '.net', '.org',
                         '.info', '.biz', '.tv', '.ws', '.bz', '.cc',
                         '.name', '.mobi', '.in', '.me', '.eu', '.asia',
                         '.co.uk', '.org.uk', '.me.uk', '.msk.ru', '.spb.ru', '.net.ru',
                         '.org.ru', '.pp.ru', '.com1.ru', '.kod812.ru', '.kod095.ru' );
*/
function DomainChooser() {

  this.createDomain = function(domains, default_value) {
    var choose_name_ = 'domain';
    this.defineParams(choose_name_, domains, 10);    
    
    obj[choose_name_] = this.obj_;
    this.defineSelectedClassName('block_selector_domain', 'selected_item_choose_domain');
    
    if (document.getElementById('main_choose_'+ choose_name_)){ 

      var hidden_name = document.getElementById('choose_zone') ? document.getElementById('choose_zone').name : "zone";
      default_value = default_value ? default_value : getFirstKeyHash(domains);
          
      document.getElementById('main_choose_' + choose_name_).innerHTML =         
      "<input type='hidden' name='"+ hidden_name +"' id='choose_"+ choose_name_ +"' value='"+ default_value +"'>" +            
      "<div id='block_choose_"+ choose_name_ + "' class='block_choose_domain' onMouseOut='if (is_selector_open) obj[\""+ choose_name_ +"\"].exitSelector();'>" +
      "  <a href='#a_"+ choose_name_ +"' id='a_choose_"+ choose_name_ + "'" +        
      "     onClick='if (is_selector_open) obj[\""+ choose_name_ +"\"].closeSelector_AcceptValue();  else{ choose_name_=\""+ choose_name_ +"\"; obj[\""+ choose_name_ +"\"].showSelector();}'>" +                                                 
      "     <span class='a_choose_selected' id='a_choose_selected_"+ choose_name_+ "'><span>"+ domains[default_value] + "</span></span><div></div></a>" +      
      "  <div id='block_selector_"+ choose_name_ + "' class='hidden' onMouseOver='setSelectorHover();' onMouseOut='unsetSelectorHover(); obj[\""+ choose_name_ +"\"].exitSelector();'>" +
      "  </div>" +    
      "</div>";
      
      this.createSelector();
    }
  };

} // end of class 'DomainChooser'
DomainChooser.prototype = new Chooser(); 

/* -------------------------------------------------------------------------- */
/*              CLASS "WhoisChooser" with round button                       */
/* -------------------------------------------------------------------------- */

function WhoisChooser() {

  this.createDomain = function(domains, default_value) {
    var choose_name_ = 'whois';
    this.defineParams(choose_name_, domains, 10);    
    
    obj[choose_name_] = this.obj_;
    this.defineSelectedClassName('block_selector_domain', 'selected_item_choose_domain');
           
    if (document.getElementById('main_choose_'+ choose_name_)){ 

      var hidden_name = document.getElementById('choose_whois_zone') ? document.getElementById('choose_whois_zone').name : "zone";
      default_value = default_value ? default_value : getFirstKeyHash(domains);
      
      document.getElementById('main_choose_' + choose_name_).innerHTML =               
      "<input type='hidden' name='"+ hidden_name +"' id='choose_"+ choose_name_ +"' value='"+ default_value +"'>" +            
      "<div id='block_choose_"+ choose_name_ + "' class='block_choose_domain' onMouseOut='if (is_selector_open) obj[\""+ choose_name_ +"\"].exitSelector();'>" +                
      "  <a href='#a_"+ choose_name_ +"' id='a_choose_"+ choose_name_ + "'" +
      "     onClick='if (is_selector_open) obj[\""+ choose_name_ +"\"].closeSelector_AcceptValue();  else{ choose_name_=\""+ choose_name_ +"\"; obj[\""+ choose_name_ +"\"].showSelector();}'>" +                           
      "     <span class='a_choose_selected' id='a_choose_selected_"+ choose_name_+ "'><span>"+ domains[default_value] + "</span></span><div></div></a>" +            
      "  <div id='block_selector_"+ choose_name_ + "' class='hidden' onMouseOver='setSelectorHover();' onMouseOut='unsetSelectorHover(); obj[\""+ choose_name_ +"\"].exitSelector();'>" +      
      "  </div>" +    
      "</div>";
      
      
      var block_selector = document.getElementById('block_selector_'+ choose_name_);

      this.createSelector();     
    }
  };
    
} // end of class 'WhoisChooser'
WhoisChooser.prototype = new Chooser(); 

/* -------------------------------------------------------------------------- */
function getFirstKeyHash(hash){
  for (var i in hash) return i;
  return null;
}

function getLengthHash(hash){
  var len = 0;
  for (var i in hash) ++len;
  return len;
}

function addHandler(obj, event, handler) {   
  if (typeof obj.addEventListener != 'undefined') obj.addEventListener(event, handler, true);
  if (typeof obj.attachEvent != 'undefined') obj.attachEvent('on'+event, handler);    
}

function trim(string){
  return string.replace(/(^\s+)|(\s+$)/g, "");
} 
/* -------------------------------------------------------------------------- */

function initializeDomain(name){
  var block = document.getElementById('noscript_block_'+name);
  if (block){
    block.className = "hidden";    
    
    var selector = xGetElementsByTagName('select', block);
    if (selector[0]){          
      var a_options = new Array();
      var option_selected = null;      
      for (var i = 0; i < selector[0].length; ++i){     
        a_options[selector[0].options[i].value] = selector[0].options[i].text;
        if (selector[0].options[i].selected) option_selected = selector[0].options[i].value;                            
      }                          

      var chooser = (name == 'whois') ? new WhoisChooser() : new DomainChooser();      
      chooser.obj_ = chooser;  
      chooser.createDomain(a_options, option_selected);
    }
    block.innerHTML = '&nbsp;';
  }      
}
/* -------------------------------------------------------------------------- */

function show_hideBlock(id){
  var block = document.getElementById(id);  
  if (block){ 
  	if (block.className == 'hidden')
      block.className = 'block';	  			
  	else	
  		block.className = 'hidden';
	}
	return false;
}

function hideBlock(id){
  var block = document.getElementById(id);  
  if (block)   	
  		block.className = 'hidden';	
	return false;
}

function clearDefaultValue(element_id, def_value){
  var element = document.getElementById(element_id);  
  if (element && element.value == def_value)
    element.value = '';
}

/* -------------------------------------------------------------------------- */
function check_email(e){
  ok = "1234567890qwertyuiop[]asdfghjklzxcvbnm.@-_QWERTYUIOPASDFGHJKLZXCVBNM";
  
  for(i=0; i < e.length ;i++)
    if(ok.indexOf(e.charAt(i))<0) 
      return (false);
    	    
  if (document.images) {
    re = /(@.*@)|(\.\.)|(^\.)|(^@)|(@$)|(\.$)|(@\.)/;
    re_two = /^.+\@(\[?)[a-zA-Z0-9\-\.]+\.([a-zA-Z]{2,4}|[0-9]{1,3})(\]?)$/;
    if (!e.match(re) && e.match(re_two)) 
      return (-1);		       
  }
}

