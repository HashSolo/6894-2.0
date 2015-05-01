HW.onload(function() {
	new HW.ClearDefault('inputSearch');
});
		
HW.ClearDefault = function(cls) {
	if(cls) {
		var inputs = $$(cls,document.body,'input');
		for(var i=0;i<inputs.length;i++) {
			var elm = this;
			(function(){
				var obj = inputs[i];
				inputs[i] = HW.extendObject(inputs[i],elm.Element);
				HW.attachEvent(inputs[i],'focus',function() {obj.focusHandler()});
				HW.attachEvent(inputs[i],'blur',function() {obj.blurHandler()});
			})()
		}
	}
}

HW.ClearDefault.prototype = {
	expClass:'clearField',
	Element:{
		focusHandler:function() {
			if (this.value == this.defaultValue) {this.value = '';}
			HW.removeClass(this,'clearField');
		},
		blurHandler:function() {
			if (this.value == "") {
				this.value= this.defaultValue;
				HW.addClass(this,'clearField');
			}
		}
	}
}
