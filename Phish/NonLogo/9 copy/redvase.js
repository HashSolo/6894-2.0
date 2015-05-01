/*
RedVase, version 1.6
(c) Copyright 2007 Bravenet Media Network. All Rights Reserved.
*/

if (!Array.prototype.push) {
  // implements Array.push since this may not be available
  Array.prototype.push = function() {
    var al = arguments.length; var l = this.length;
    for ( var i = 0; i < al; ++i ) {
      this[l+i] = arguments[i];
    }
    return this.length;
  };
}

if ( typeof(RedVase) == 'undefined' ) {
  var RedVase = (function() {
    var _url_base = 'http://redvase.bravenet.com';
    var _page_ads = [];

    // converts ad hash into url
    function _to_url(ad) {
      var url = [ _url_base ];
      var query_string = [ ];

      // construct base url
      url.push((ad.content == 'html' || ad.content == 'iframe' ? 'creative' : ad.content)); delete ad.content;
      url.push(ad.publisher); delete ad.publisher;
      url.push(ad.kind); delete ad.kind;
      if (ad.alternate) { url.push(ad.alternate); delete ad.alternate; }

      // append the iframe formats if needed
      if (ad.format) { 
        var matches = ad.format.match(RedVase.FORMAT_REGEX)
        query_string.push('ifh=' + matches[1]);
        query_string.push('ifw=' + matches[2]);
        delete ad.format; 
      }

      // append randomizer
      query_string.push('r=' + (ad.random || new Date().getTime()))
      if (ad.random) { delete ad.random; }

      // unshift any other params so randomizer is last
      for (var key in ad) {
        if (typeof(ad[key]) == 'string') query_string.unshift([key] + '=' + ad[key]);
      }

      return url.join('/') + '?' + query_string.join('&');
    }

    // injects the ad onto the document
    function _show(ad) {
      document.writeln('<script src="' + _to_url(ad) + '" type="text/javascript" charset="utf-8"><\/script>');
    }

    // shortcut method
    function _record_and_show(ad) {
      _page_ads.push(ad);
      _show(ad);
    }

    // converts options hash to attribute string
    function _to_pop_options(hash) {
      if (typeof hash == 'string') return hash;
      var result = [];
      for (var key in hash) {
        result.push(key+'='+hash[key]);
      }
      return result.join(',');
    }

    return {
      placement: function(block) {
        var ad = {};
        block(ad);
        if ((new RedVase.Sanitizer(ad)).check()) {
          _record_and_show(ad);
        }
      },
      show_popunder: function(url, name, options) {
        if (!url) throw "required url missing";
        name = name || '_blank';
        options = _to_pop_options(options || {});
        return window.open(url, name, options);
      }
    };
  })();

  RedVase.FORMAT_REGEX = /(\d+)x(\d+)/;
  RedVase.Sanitizer = function(ad) {
    this.ad = ad;
    this.sane = true;
  };
  RedVase.Sanitizer.prototype  = {
    update: function(result) {
      if (this.sane && !result) {
        this.sane = false;
      }
      return result;
    },
    assert_exists: function(flag) {
      return this.update(flag);
    },
    assert_string: function(flag, allow_undefined) {
      if (typeof(allow_undefined) === 'undefined') allow_undefined = false;
      return (allow_undefined || this.update(this.assert_exists(flag)) && typeof(flag) == 'string');
    },
    assert_content: function(flag) {
      if (flag != 'html' && flag != 'pop' && flag != 'iframe') {
        flag = 'html';
      }
      return flag;
    },
    assert_format: function(flag, allow_undefined) {
      if (typeof(allow_undefined) === 'undefined') allow_undefined = false;
      if (typeof(flag) === 'undefined' && allow_undefined) return null;
      
      if ( this.update( this.assert_string(flag, true) && RedVase.FORMAT_REGEX.test(flag) ) ) {
        return flag.match(RedVase.FORMAT_REGEX)[0];
      }
      return null;
    },
    check: function() {
      var ad = this.ad;
      // make sure ad.content is set
      ad.content = this.assert_content(ad.content);
      this.assert_string(ad.publisher);
      this.assert_string(ad.kind);
      this.assert_string(ad.alternate, true);
      ad.format = this.assert_format(ad.format, true);
      if (ad.format === null) delete ad.format;
      return this.sane;
    }
  };
}

if (typeof(redvase_ad) != 'undefined') {
  (function(redvase_ad_var) {
      RedVase.placement(function(ad) {
      for (var k in redvase_ad_var) {
        ad[k] = redvase_ad_var[k];
      }
    });
  })(redvase_ad);
  redvase_ad = null;
}
