var SortableTable = new Class ({

	// pass in the table element to get this started
	initialize: function (table, options) {

		this.options = Object.extend(
			{
				ascCl: 'asc',
				descCl: 'desc',
				primaryCl: 'primary',
				secondaryCl: 'secondary',
				altCl: 'alt',
				overCl: 'over',
				dateCl: 'date'
			} , options || {} );

		this.table = $(table);
		this.thead = this.table.getElement('thead');
		this.tbody = this.table.getElement('tbody');
		this.tfoot = this.table.getElement('tfoot');

		this.headers = this.thead.getElements('th');
		this.rows = this.tbody.getElements('tr');
		this.cells = this.tbody.getElements('td');

		this.row_count = this.rows.length;
		this.col_count = this.headers.length;
		this.cell_count = this.cells.length;

		this.sortasc = true;
		this.sortasc2 = true;
		this.sortby;
		this.sortby2;

		this.order = new Array();
		for (i=this.row_count-1;i>=0;i--) {
			this.order[i] = i;
		}
		
		for (i=(this.col_count-1);i>=0;i--) {
			this.headers[i].addEvent('click',function(i){this.sort(i)}.pass(i,this));
			this.headers[i].addEvent('mouseover',function(i){this.headers[i].addClass(this.options.overCl)}.pass(i,this));
			this.headers[i].addEvent('mouseout',function(i){this.headers[i].removeClass(this.options.overCl)}.pass(i,this));
			this.headers[i].setStyle('cursor','pointer');
		}

		this.month_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

		this.getdata();
		
		this.sort(0);
		

	},

	// put all the table data in arrays
	// special case for dates
	getdata: function () {

		this.data = new Array();

		for (i=(this.row_count-1);i>=0;i--) {
			this.data[i] = new Array();
			for (j=(this.col_count-1);j>=0;j--) {

				var cell = this.cells[((i*this.col_count)+j)];
				var datum = '';

				var classes = cell.getProperty('class').split(" ");
				for (k=classes.length-1;k>=0;k--) {
					if (classes[k].contains('sortby:')) {
						datum = classes[k].split(":")[1];
					}
				}

				if (this.headers[j].hasClass(this.options.dateCl) && !datum) {
					datum = new Date(cell.getText()).getTime();
				} else if (!datum) {
                    var datum = cell.getText().clean();
                    datum = datum.replace(/,/, '');
                    if (datum.match(/^\d+$/)) {
                        datum = datum-0; // force to integer type
                    } else {
                        datum = datum.replace(/^(\d\d)\/(\d\d)\/(\d\d\d\d)/, "$3$1$2");
                    }
				}

				this.data[i][j] = datum;
			}
		}


	},

	// called by clicking on a header
	sort: function (col) {

		this.headers.each(function(th){
			th.removeClass(this.options.primaryCl);
			th.removeClass(this.options.secondaryCl);
			th.removeClass(this.options.ascCl);
			th.removeClass(this.options.descCl);
		}.bind(this));

		if (this.sortby == col) {
			this.sortasc = this.sortasc ? false : true;
		} else {
			this.sortasc2 = this.sortasc;
			this.sortasc = true;
			this.sortby2 = this.sortby;
			this.sortby = col;
		}

		this.headers[this.sortby].addClass(this.options.primaryCl);
		if (this.sortasc) {
			this.headers[this.sortby].addClass(this.options.ascCl);
		} else {
			this.headers[this.sortby].addClass(this.options.descCl);
		}
		if (this.sortby2 || this.sortby2 === 0) {
			this.headers[this.sortby2].addClass(this.options.secondaryCl);
			if (this.sortasc2) {
				this.headers[this.sortby2].addClass(this.options.ascCl);
			} else {
				this.headers[this.sortby2].addClass(this.options.descCl);
			}
		}

		this.order.sort(this.compare.bind(this));

		this.build();

	},

	// custom sort function
	compare: function (a, b) {

		if (this.data[a][this.sortby] == this.data[b][this.sortby]) {
			if (this.data[a][this.sortby2] == this.data[b][this.sortby2]) return 0;
			order = (this.data[a][this.sortby2] < this.data[b][this.sortby2]) ? -1 : 1;
			return this.sortasc2 ? order : order*-1;
		} else {
			order = (this.data[a][this.sortby] < this.data[b][this.sortby]) ? -1 : 1;
			return this.sortasc ? order : order*-1;
		}

	},

	// reorganize the table rows based on the new order
	build: function () {

		//this.rows.each(function(tr){tr.removeClass(this.options.altCl)}.bind(this));

		for(i=0;i<this.row_count;i++) {
			this.tbody.adopt(this.rows[this.order[i]]);
			if (i % 2 == 0) this.rows[this.order[i]].addClass(this.options.altCl);
			else this.rows[this.order[i]].removeClass(this.options.altCl);
		}

	}

});