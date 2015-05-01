// Cross-browser implementation of element.addEventListener()

function listen(evnt, elem, func) {
    if (elem.addEventListener)  // W3C DOM
        elem.addEventListener(evnt,func,false);
    else if (elem.attachEvent) { // IE DOM
         var r = elem.attachEvent("on"+evnt, func);
    return r;
    }
    else window.alert('I\'m sorry Dave, I\'m afraid I can\'t do that.');
}

// Use: listen("event name", elem, func);
