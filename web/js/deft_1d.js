// Keep global
var results = 0
var json_string = ''
var data = 0

// Load the visualization API and the corechart pacakge
google.load("visualization", "1", {packages:["corechart"]});

$("#google_chart").html("Loading Google Charts API...")

// Hommade function to check validity of JSON string. From StackOverflow
function IsJsonString(str) {
    try {
        JSON.parse(str);
    } catch (e) {
        return false;
    }
    return true;
}

// Homemade assertion function (off stackoverflow)
function assert(condition, message) {
    if (!condition) {
        message = message || "Assertion failed";
        if (typeof Error !== "undefined") {
            throw new Error(message);
        }
        throw message; // Fallback
    }
}


// Loads a file
function readSingleFile(e) {
    $("#input_source_user").prop('checked', true)

    if ($("#automatic_box").prop('checked')) {
        $("#box_min").val("auto")
        $("#box_max").val("auto")
    }

    // Get file
    var file = e.target.files[0];
    
    // Verify file is valid
    if (!file) {
        return;
    }

    // Create a new FileReader instance
    var reader = new FileReader();

    // Define a function to load contents
    reader.onload = function(e) {

        // Gets contents from file
        var contents = e.target.result;

        // Remove lines that contain hashes
        contents = contents.replace(/^\#.*$/mg, '');

        // Clean contents so just numbers and spaces
        contents = contents.replace(/[^0-9\-\+\.]/g,' ')

        // Replace contiguous blocks of whitespace with single '\n's
        contents = contents.replace(/\s+/g,'\n')

        // Remove whitespace at beginning of file
        contents = contents.replace(/^\s+|\s+$/g,'')

        // Split contents into a list of numbers
        data_strings = contents.split('\n')

        // Convert data to floats
        data = []
        for (i = 0; i < data_strings.length; i++) {
            if (isNaN(data_strings[i])) {
                throw new Error('ERROR: Non-numeric values in selected file.');
            } else {
                data[i] = parseFloat(data_strings[i])
            }
        }

        // Create object to write to form
        obj = {data:data, description:"User data.", 
                box_min:"auto", box_max:"auto", alpha:3}

        $("#data").val(obj.data.join(',\n'))

        // Set file name
        full_fake_path = $("#file_selector").val()
        obj.description = full_fake_path.replace(/^.*[\\\/]/,'')
        $("#file_name").html(obj.description)
        $("#file_name").attr('href','file://' + full_fake_path)
        set_form_values(obj)

        //run_deft()

        //draw_chart()
    };

    // I guess readAsText is method of FileReader. 
    reader.readAsText(file);
}

// Set event handler
document.getElementById('file_selector')
    .addEventListener('change', readSingleFile, false);

// Just run DEFT
run_deft = function() {
    // Data to send to server
    form_data = $("#form").serialize()

    // Tell user that we're waiting for a density estimate from the server
    $("#google_chart").html("<h4> Waiting for density estimte from server... </h4>")

    // Get data from server in JSON format
    json_string = $.ajax({
        type: "POST",
        data: form_data,
        url: "run_deft_1d.php",
        dataType:  "json",
        async: false
    }).responseText;

    // Display object
    $("#google_chart").html('<pre>' + json_string + '</pre>')

    // Validate JSON string
    assert(IsJsonString(json_string))

    // Convert results into JavaScript object (results_object is global)
    results = JSON.parse(json_string)

    // Run KDE if requested
    if ($("#show_kde").is(":checked")) {
        run_kde()
    }
    // Run MaxEnt if requested
    if ($("#show_maxent").is(":checked")) {
        run_maxent()
    }
}


// Have server perform maxent density estimation
run_maxent = function() {
    // Data to send to server
    form_data = $("#form").serialize()

    // Tell user that we're waiting for a density estimate from the server
    $("#google_chart").html("<h4> Waiting for KDE estimte from server... </h4>")

    // Get data from server in JSON format
    json_string = $.ajax({
        type: "POST",
        data: form_data,
        url: "run_maxent_1d.php",
        dataType:  "json",
        async: false
    }).responseText;

    // Display object
    $("#google_chart").html('<pre>' + json_string + '</pre>')

    // Convert results into JavaScript object (results_object is global)
    kde_results = JSON.parse(json_string)

    // Store maxent results, but that's it
    results.Q_maxent = kde_results.Q_maxent
}


// Has server perform kernel density estimation
run_kde = function() {
    // Data to send to server
    form_data = $("#form").serialize()

    // Tell user that we're waiting for a density estimate from the server
    $("#google_chart").html("<h4> Waiting for KDE estimte from server... </h4>")

    // Get data from server in JSON format
    json_string = $.ajax({
        type: "POST",
        data: form_data,
        url: "run_kde_1d.php",
        dataType:  "json",
        async: false
    }).responseText;

    // Display object
    $("#google_chart").html('<pre>' + json_string + '</pre>')

    // Convert results into JavaScript object (results_object is global)
    kde_results = JSON.parse(json_string)

    // Store kde results, but that's it
    results.Q_kde = kde_results.Q_kde
}

// Evaluates a function, defined by func_text, at each point in x_grid
eval_function = function(x_grid, func_text) {
    G = x_grid.length
    f = []
    for (i=0; i<G; i++) {
        x = x_grid[i]
        f_at_x = eval(func_text) // func_text must be a function of x
        f.push(f_at_x)
    }
    return f
}

// Just draw chart
draw_chart = function() {
    
    // Set plotting colors
    histogram_color = "#DDD" //"#66CCFF"
    deft_color = "#009933"
    true_color = "#000000"
    kde_color = "#CC66FF" //"#FF9900"
    maxent_color = "#FF9900"

    // Used to set ylims
    Q_max = Math.max.apply(null, results.Q_star)

    // If there is a true distribution, compute it
    if ($("#input_source_simulation").prop('checked') && $("#pdf").val()) {
        pdf_text = $("#pdf").val()
        Q_true = eval_function(results.x_grid, pdf_text)
        h = results.h
        G = results.G
        
        // Integrate Q_true
        Q_true_integral = 0
        for (i=0; i<G; i++) {
            Q_true_integral += Q_true[i]*h
        }

        // Normalize Q_true
        for (i=0; i<G; i++) {
            Q_true[i] /= Q_true_integral
        }
        
        // Compute differential entropy in bits
        e_true = 0
        for (i=0; i<Q_true.length; i++) {
            e_true += -h * Q_true[i] * Math.log2(Q_true[i])
        }

        // Store results
        results.Q_true = Q_true
        results.e_true = e_true
    }

    // Display entropy estimate
    e_mean = results.e_mean
    e_std = results.e_std

    // Determine number of digits to display
    for (digits=0; digits<7; digits++) {
        if (e_std >= 4*Math.pow(10,-(digits))) {
            break;
        }
    }

    disp_string = "<center>Estimated entropy:&nbsp;&nbsp;" + e_mean.toFixed(digits) 
        + ' &plusmn; ' + e_std.toFixed(digits) + ' bits<br>'
    if ($("#input_source_simulation").prop('checked') && $("#pdf").val()) {
        e_true = results.e_true
        
        z = (e_true - e_mean)/e_std
        disp_string += 'True entropy:&nbsp;&nbsp;' + e_true.toFixed(digits) 
            + ' bits&nbsp;&nbsp;(z-score: ' + z.toFixed(1) + ')' 
    }
    disp_string += '</center>'
    $("#entropy").html(disp_string)

    // Create Google Data Table object
    data_table = new google.visualization.DataTable();

    // Add x-values to data table
    cols = [results.x_grid]
    data_table.addColumn({id:'x', type:'number', role:'domain', label:'x'});

    // Initialize list of plots
    series_specs = {}
    num_plots = 0

    // If user wants to view histogram
    if ($("#show_histogram").is(':checked')) {
        cols.push(results.R)
        data_table.addColumn({id:'R', type:'number', role:'data', label:'Histogram'})
        series_specs[num_plots] = {type: "bars", color:histogram_color}
        num_plots += 1
    }

    // If user wants to view Q_star
    if ($("#show_Q_star").is(':checked')) {
        cols.push(results.Q_star)
        data_table.addColumn({id:'Q_star', type:'number', role:'data', label:'DEFT'})
        series_specs[num_plots] = {type: "line", color:deft_color, lineWidth:4}
        num_plots += 1
    }

    // If user whishes to show sampled densities
    if ($("#num_posterior_samples").val() > 0) {
        num_samples = parseInt($("#num_posterior_samples").val())
        assert(num_samples <= results.Q_samples.length)
        for (n=0; n<num_samples; n++) {
            Q = results.Q_samples[n]
            cols.push(Q)
            this_id = 'Q_posterior_'+toString(n)
            data_table.addColumn({id:this_id, type:'number', role:'data', label:'Posterior'})
            if (n==0) {
                series_specs[num_plots] = {type: "line", color:deft_color, lineWidth:0.35, visibleInLegend: false}
            }
            else {
                series_specs[num_plots] = {type: "line", color:deft_color, lineWidth:0.35, visibleInLegend: false}
            }
            num_plots += 1
        }
    }

    // Check whether user wants to show true distribution 
    show_true = $("#show_pdf").is(':checked') && $("#input_source_simulation").prop('checked') && $("#pdf").val()
    // If so
    if (show_true) {

        // Compute true distribution
        pdf_text = $("#pdf").val()

        // Add plotting info
        cols.push(results.Q_true)
        data_table.addColumn({id:'Q_true', type:'number', role:'data', label:'True'})
        series_specs[num_plots] = {type: "line", color:true_color, lineWidth:2}
        num_plots += 1
    }

    // If user wishes to see the kde estimate
    if ($("#show_kde").is(':checked')) {
         cols.push(results.Q_kde)
         data_table.addColumn({id:'Q_kde', type:'number', role:'data', label:'KDE'})
         series_specs[num_plots] = {type: "line", color:kde_color, lineWidth:4}
         num_plots += 1
    }

    // If user wishes to see the maxent estimate
    if ($("#show_maxent").is(':checked')) {
         cols.push(results.Q_maxent)
         data_table.addColumn({id:'Q_maxent', type:'number', role:'data', label:'MaxEnt'})
         series_specs[num_plots] = {type: "line", color:maxent_color, lineWidth:4}
         num_plots += 1
    }

    // If user wishes to show error bars 
    // WARNING: HAS TO GO LAST IN LIST OF PLOTS. 
    // OTHERWISE FORMATTING OF OTHER PLOTS GETS FUCKED UP. REASON UNKNOWN.
    /*
    if ($("#show_errorbars").is(':checked')) {
        // Append data columns
        cols.push(results.Q_lb)
        cols.push(results.Q_ub)

        // Add columns to data_table
        data_table.addColumn({id:'dQ', type:'number', role:'interval', label:'Q_star - dQ'});
        data_table.addColumn({id:'dQ', type:'number', role:'interval', label:'Q_star + dQ'});
        
        // Add specificaitons for data columns
        series_specs[num_plots] = {type: "interval", visibleInLegend: false, lineWidth:0},
        series_specs[num_plots+1] = {type: "interval", visibleInLegend: false, lineWidth:0}

        // Increment the number of things that are being plotted
        num_plots += 2
    }
    */

    // Transform cols into rows
    num_cols = cols.length
    rows = []
    for (i=0; i<results.G; i++) {
        rows.push([])
        for (j=0; j<num_cols; j++) {
            rows[i].push(cols[j][i])
        }
    }

    // Insert rows into data table
    data_table.addRows(rows)

    // Specify options, including the series_specs defined above
    options = {
        //title: $("#data_description").val(),
        interval: {'dQ': { 
            style:'area', 
            curveType:'function', 
            fillOpacity:0.3, 
            color:deft_color }},
        series: series_specs,
        bar: {groupWidth: "100%"},
        vAxis: {
            viewWindow: {min: 0, max: 1.5*Q_max}, 
            gridlines:{count:0},
        },
        hAxis: {
            viewWindow: {min: results.box_min, max: results.box_max}, 
            gridlines: {color: "none"},
            baselineColor:"none",
            title: results.N.toString() + " data points"
        },
        legend:{position:'top'},
        chartArea: {'width': '90%', 'height': '80%'},
    };

    // Get DIV element into which chart will be placed 
    chart_div = document.getElementById('google_chart');

    // Instantiate a chart to be displayed in the appropriate DIV
    chart = new google.visualization.ComboChart(chart_div);

    // Wait for the chart to finish drawing before calling the getImageURI() method.
    google.visualization.events.addListener(chart, 'ready', function () {
        // Display link to .png file
        $png_link_html = '<small> Results: <a href="' + chart.getImageURI() + '" target="_blank"> image (.png)  </a>  or ';
        $("#output_links").html(
            $png_link_html + 
            "<a id='csv_link'> table (.csv) </a> </small>"
        );
        $("#csv_link").click(function() {
            csv_out = dataTableToCSV(data_table);
            downloadCSV(csv_out);
        })
    });

    // Draw the chart
    chart.draw(data_table, options);
}


// Set a callback to run when the Goggle Visualization API is loaded
google.setOnLoadCallback(function() {
    get_simulated_data();
})


// Gets simulated data from the server
get_simulated_data = function(event) {
    $("#input_source_simulation").prop('checked', true)

    $("#data").html("Waiting for data from server...")

    // Serialize form data so can send it to server
    form_data = $("#form").serialize()

    // Send data to server and catch response
    jsonData = $.ajax({
        type: "POST",
        data: form_data,
        url: "simulate_data_1d.php",
        dataType:  "json",
        async: false
        }).responseText;

    // Convert JSON string to data
    obj = JSON.parse(jsonData)

    // Get function
    if (obj.hasOwnProperty('pdf_js')) {
        $("#pdf").val(obj.pdf_js)
    }

    set_form_values(obj)

    run_deft()

    draw_chart()
}


// Get example data from a file
get_example_data = function(event) {
    $("#input_source_example").prop('checked', true)

    $("#data").html("Waiting for data from server...")

    // Serialize form data so can send it to server
    form_data = $("#form").serialize()

    // Send data to server and catch response
    jsonData = $.ajax({
        type: "POST",
        data: form_data,
        url: "load_example_data_1d.php",
        dataType:  "json",
        async: false
        }).responseText;

    // Convert JSON string to data
    obj = JSON.parse(jsonData)

    set_form_values(obj)

    run_deft()

    draw_chart()

}

set_form_values = function(obj) {

    if (!obj.hasOwnProperty('data')) {
        throw new Error('ERROR: No data to process!')
    }

    // Keep only 5 decimal places3
    for (i = 0; i < obj.data.length; i++) {
        obj.data[i] = obj.data[i].toPrecision(5)
    }

    data_string = obj.data.join('\n')

    $("#data").val(data_string)

    if (obj.hasOwnProperty('description')) {
        //$("#data_description").val(obj.description)
        $("#data_title").html("<h4> " + obj.description.trim() + "</h4>")
    }

    use_presets = 
        ($("#input_source_example").prop('checked') && 
        $("#use_example_presets").is(':checked'))
        ||
        ($("#input_source_simulation").prop('checked') && 
        $("#use_simulation_presets").is(':checked'));


    if (use_presets) {

        if (obj.hasOwnProperty('box_min'))
            $("#box_min").val(obj.box_min)

        if (obj.hasOwnProperty('box_max'))
            $("#box_max").val(obj.box_max)

        if (obj.hasOwnProperty('alpha'))
            $("#alpha").val(obj.alpha)

        if (obj.hasOwnProperty('periodic')) {
            if (obj.periodic)
                $("#periodic").val('True')
        } else {
            if (obj.periodic)
                $("#periodic").val('False')
        }

    }

}



//From https://gist.github.com/pilate/1477368 
function dataTableToCSV(dataTable_arg) {
    var dt_cols = dataTable_arg.getNumberOfColumns();
    var dt_rows = dataTable_arg.getNumberOfRows();
    
    var csv_cols = [];
    var csv_out;
    
    // Iterate columns
    for (var i=0; i<dt_cols; i++) {
        // Replace any commas in column labels
        csv_cols.push(dataTable_arg.getColumnLabel(i).replace(/,/g,""));
    }
    
    // Create column row of CSV
    csv_out = csv_cols.join(",")+"\r\n";
    
    // Iterate rows
    for (i=0; i<dt_rows; i++) {
        var raw_col = [];
        for (var j=0; j<dt_cols; j++) {
            // Replace any commas in row values
            raw_col.push(dataTable_arg.getFormattedValue(i, j, 'label').replace(/,/g,""));
        }
        // Add row to CSV text
        csv_out += raw_col.join(",")+"\r\n";
    }

    return csv_out;
}

 //From https://gist.github.com/pilate/1477368 
function downloadCSV (csv_out) {
    var blob = new Blob([csv_out], {type: 'text/csv;charset=utf-8'});
    var url  = window.URL || window.webkitURL;
    var link = document.createElementNS("http://www.w3.org/1999/xhtml", "a");
    link.href = url.createObjectURL(blob);
    link.download = "deft_estimate.csv"; 

    var event = document.createEvent("MouseEvents");
    event.initEvent("click", true, false);
    link.dispatchEvent(event); 
}