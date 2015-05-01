// jQuery plugin - Dolphin Top Menu
(function($){
	$.fn.dolTopMenu = function() {
		return this.each( function(){
			
			var show = function() {
				var o = $(this).attr('showsub');
				$( o ? o : this ).attr( 'hover', 'true' ).show();
				
				//alert( $activeSub );
				if( typeof $activeSub != 'undefined' )
					$activeSub.hide();
			};
			
			var hide = function() {
				var o = $(this).attr('showsub');
				var $o = $( o ? o : this );
				
				$o.attr( 'hover', 'false' );
				
				setTimeout( function() {
					if( $o.attr( 'hover' ) != 'true' ) {
						$o.hide();
						
						if( typeof $activeSub != 'undefined' && $inactiveSubs.filter('[hover=true]').length == 0 )
							$activeSub.show();
					}
				}, 100 );
			};
			
			var $inactive  = $( 'a', this );
			var $active    = $( 'b:first', this );
			
			var $activeSub;
			var $inactiveSubs;
			
			if( $active.length && $active.attr('showsub') )
				$activeSub = $( $active.attr('showsub') );
			
			$inactive.hover( show, hide ).each( function() {
				if( !$inactiveSubs )
					$inactiveSubs = $( $(this).attr('showsub') );
				else
					$inactiveSubs = $inactiveSubs.add( $(this).attr('showsub') );
			} );
		
			$inactiveSubs.hover( show, hide );
		} );
	};
})(jQuery);