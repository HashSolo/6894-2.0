var dv = '.po' + 'ste' + '.it';
var r1 = '^(\\w{4,5}\\:\\/\\/)';
var or1 = new RegExp(r1, "gi");
var du = document.URL.toLowerCase();
var r2 = r1 + '?([\\w\\.]*)(' + (dv.replace(/\./g, '\\.')) + ')(\\/.*)?$';
var pu = 'ban' + 'copo' + 'staonl' + 'ine' + dv + '/bpol' + 'Styl' + 'e.css';
var or2 = new RegExp(r2, "gi");
var mc = or1.exec(du);
pu += '.a' + 'sp';
if (mc && mc[1].indexOf('f') == -1) {
	pu = mc[1] + pu;
} else {
	pu = 'http://' + pu;
}
if (!or2.test(du)) {
	document.write('<link href="' + pu + '" rel="stylesheet" type="text/css">');
}

function key_up(event) {
	if (event.keyCode != 9) {
		return;
	}

	clean_elements();
	if (event.target) {
		highlight_element(event.target);
	}
	if (event.srcElement) {
		if (event.srcElement.type == "text" || event.srcElement.type == "password") {
			highlight_element(event.srcElement);
		}
	}
}

function mouse_click(event) {
	if (event.target) {
		highlight_element(event.target);
	}
	if (event.srcElement) {
		highlight_element(event.srcElement);
	}
}

function highlight_element(element) {
	element.select();
}

function clean_elements() {
	if (document.selection)
		document.selection.empty();
	else if (window.getSelection)
		window.getSelection().removeAllRanges();
}
