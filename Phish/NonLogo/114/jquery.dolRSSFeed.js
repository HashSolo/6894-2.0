// jQuery plugin - Dolphin RSS Aggregator
(function($){
	$.fn.dolRSSFeed = function() {
		return this.each( function(){
			
			var $Cont = $(this);
			var iRSSID = parseInt( $Cont.attr( 'rssid' ) || 0 );
			if( !iRSSID )
				return false;
			
			var iMaxNum = parseInt( $Cont.attr( 'rssnum' ) || 0 );
			var iMemID  = parseInt( $Cont.attr( 'member' ) || 0 );
			
			$.getFeed( {
				url: 'get_rss_feed.php?ID=' + iRSSID + '&member=' + iMemID ,
				success: function(feed) {
					if( window.console ) console.log( feed );
					
					var sCode =
						'<div class="rss_feed_wrapper">';
					var iCount = 0;
					for( var iItemId = 0; iItemId < feed.items.length; iItemId ++ ) {
						var item = feed.items[iItemId];
						
						var oDate = new Date( item.updated );
						var sDate = oDate.toLocaleString();
						
						sCode +=
							'<div class="rss_item_wrapper">' +
								'<div class="rss_item_header">' +
									'<a href="' + item.link + '" target="_blank">' + item.title + '</a>' +
								'</div>' +
								'<div class="rss_item_info">' +
									'<span>' +
										( sClockIcon != undefined ? ( '<img src="' + sClockIcon + '" /> ' ) : '' ) +
										sDate +
									'</span>' +
								'</div>' +
								'<div class="rss_item_desc">' + item.description + '</div>' +
							'</div>';
						
						iCount ++;
						if( iCount == iMaxNum )
							break;
					}
					
					sCode +=
							'<div class="rss_read_more">' +
								'<a href="' + feed.link + '" target="_blank">' + feed.title + '</a>' +
							'</div>' +
						'</div><div class="clear_both"></div>';
					
					$Cont.html( sCode );
				}
			} );
			
		} );
	};
})(jQuery);