var Tracker = {
  category: false,
  enabled: false,
  userID: 'unknown',
  RTTrackerFile: 'out',

   // Random comment change to get this to Prod, because builds are annoying

  init: function(categoryID,userID) {
    if(categoryID!=null) {
      this.setCategory(categoryID);
    }
    if(userID!=null) {
      this.userID = userID;
    }

    if(typeof(cmSetProduction)=='function') {
      if(!this.enabled) {
        cmSetProduction();
        this.enabled = true;
      }
    } else {
      this.enabled = false;
    }
  },
  
  RTTrack: function (eventID){
  	var img = new Image();
  	img.src = 'http://38.103.1.95/tracker/tracker.php?action=track&out=' + this.RTTrackerFile + '&eventID=' + eventID + '&rand=' + Math.random();
  },

  track: function(eventID,cat) {
    if(!this.enabled) {
      return;
    }

    cmCreatePageviewTag(eventID, cat||this.category, null, null);
  },

  trackSearch: function(eventID,query,numResults) {
    if(!this.enabled) {
      return;
    }

    cmCreatePageviewTag(eventID, this.category, query, numResults);
  },

  trackError: function(eventID) {
    if(!this.enabled) {
      return;
    }

    cmCreateErrorTag(eventID, this.category);
  },

  startConversion: function(eventID, points) {
    if(!this.enabled) {
      return;
    }
    
    if(arguments.length < 2) {
      points = 0;
    }

    cmCreateConversionEventTag(eventID, "1", this.category, points);
  },

  completeConversion: function(eventID, points) {
    if(!this.enabled) {
      return;
    }

    if(arguments.length < 2) {
      points = 0;
    }

    cmCreateConversionEventTag(eventID, "2", this.category, points);
  },

  productView: function(productID, productName) {
    if(!this.enabled) {
      return;
    }

    if(productName==null) {
      productName = productID;
    }
    cmCreateProductviewTag(productID,productName,this.category);
  },

  purchaseClick: function(productID, productName, price) {
    //shop5

    if(!this.enabled) {
      return;
    }

    if(productName==null) {
      productName = productID;
    }
    cmCreateShopAction5Tag(productID,productName,1,price,this.category);
    cmDisplayShop5s();
  },

	cartPrice: 0,
  purchaseComplete: function(productID, productName, price, isCart) {
    //shop9
    var orderID;

    if(!this.enabled) {
      return;
    }

    if(productName==null) {
      productName = productID;
    }

    if(typeof(sessionToken)!='undefined' && sessionToken.length>15) {
      orderID = sessionToken.substring(sessionToken.length-10);
    } else {
      orderID = this.userID;
    }

    cmCreateShopAction9Tag(productID,productName,1,price,this.userID,orderID,price,this.category);
    
    this.cartPrice += price;
	if(!isCart) {
          cmDisplayShop9s();
		  cmCreateOrderTag(orderID,price,0,this.userID,null,null,null);
	  }
  },

	orderComplete: function() {
        if(typeof(sessionToken)!='undefined' && sessionToken.length>15) {
		  orderID = sessionToken.substring(sessionToken.length-10);
		} else {
		  orderID = this.userID;
		}         

 		cmDisplayShop9s();
 		cmCreateOrderTag(orderID,this.cartPrice,0,this.userID,null,null,null);
	},

  setCategory: function(newCategory) {
    this.category = newCategory;
  },
  setUserID: function(newuserid) {
    this.userID = newuserid;
  },

  loadCMLibraries: function() {
    var files = ["http://images.freewebs.com/JS/CM/eluminate.js", "http://images.freewebs.com/JS/CM/cmdatatagutils.js"];
    for(var i=0;i<files.length;i++) {
      loadScript(files[i]);
    }
  },

  loadScript: function(src, appendTo) {
    if(arguments.length > 1) {
      var script = document.createElement('script');
      script.src = this.url;
      script.type = "text/javascript";
      appendTo?appendTo.appendChild(script):document.body.appendChild(script);
    } else {
      document.write('<script type="text/javascript" src="' + src + '"></script>');
    }
  },
  
  makeCMSafe: function(str) {
    return str.replace(/\W/g,"").substring(0,15);
  }
};
