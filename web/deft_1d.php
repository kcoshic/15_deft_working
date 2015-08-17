<?php echo file_get_contents("./top.html"); ?>

<!-- Marke selection in menubar -->
<script>
$("#menubar_1d").addClass("active");
</script>

<form id='form' 
      method='post' 
      enctype='multipart/form-data'>

    <!-- Display data that will be used by DEFT 1D-->
    <div class="row">            

        <div class="col-sm-12">
            <center>

            <!-- DIV title for chart title -->
            <div id="data_title"><h4>&nbsp;</h4></div> 

            <!-- DIV for Google Chart -->
            <div id="google_chart" style="width: 600px; height: 300px"></div>

            <!-- This is where the data is stored -->
            <input type="hidden" id="data" name="data">

            <!-- DIV for entropy estimate -->
            <br>
            <div id="entropy">&nbsp;<br>&nbsp;</div>

            <!-- DIV for displaying links to .png and .csv files -->
            <div id='output_links' class="row" >
            &nbsp;
            </div>

            </center>

        </div>

    </div>

    <hr>

    <div class="row">
        <div class="col-sm-12">
            <center>

            <input type="checkbox" 
                id="show_histogram"
                onclick="draw_chart();"
                checked> 
             Data &nbsp;&nbsp; 

            <input type="checkbox" 
                id="show_Q_star"
                onclick="draw_chart();"
                checked> 
             Estimate &nbsp;&nbsp; 

            <!--
            <input type="checkbox" 
                id="show_errorbars"
                onclick="draw_chart();"
                value='False'> 
             Errorbars &nbsp;&nbsp; 
            -->

            Posterior samples  
            <select id='num_posterior_samples' 
                name='num_posterior_samples'
                onchange="draw_chart();">
                <option value='0' selected='selected'>0</option>
                <option value='5'>5</option>
                <option value='20'>20</option>
                <option value='100'>100</option>
            </select>
            <input type="hidden" name="max_posterior_samples" value="100">
            &nbsp; &nbsp;

            <input type="checkbox" 
                id="show_maxent"
                onclick="
                    if ($('#show_maxent').is(':checked')) {
                        run_maxent()
                    } 
                    draw_chart()
                    "> 
            MaxEnt
            <small>
            <select id='maxent_order' 
                name='maxent_order'
                onchange="
                    $('#show_maxent').prop('checked',true); 
                    run_maxent(); 
                    draw_chart()">
                <option value='1' selected='selected'>1</option>
                <option value='2'>2</option>
                <option value='3' selected='selected'>3</option>
                <option value='4'>4</option>
                <option value='5'>5</option>
                <option value='6'>6</option>
                <option value='7'>7</option>
                <option value='8'>8</option>
                <option value='9'>9</option>
                <option value='10'>10</option>
                <option value='15'>15</option>
                <option value='20'>20</option>
            </select>
            </small>
            &nbsp; &nbsp;

            <input type="hidden" id="pdf">
            <input type="checkbox" 
                id="show_kde"
                onclick="
                    if ($('#show_kde').is(':checked')) {
                        run_kde()
                    } 
                    draw_chart()
                    "> 
            KDE &nbsp;&nbsp;

            <input type="checkbox" 
                id="show_pdf" 
                onclick="draw_chart();"> 
            True

            </center> 
        </div>

    </div>

    <hr>

    <div class="row">
        <!-- Specify data -->
        
        <div class="col-sm-2">
            <center>
            <a class="btn btn-md btn-success" role="button" 
               id="run_deft_button"
               onclick="
               if ($('#input_source_simulation').is(':checked')) {
                   get_simulated_data();
               }
               else if ($('#input_source_example').is(':checked')) {
                   get_example_data();
               }
               else if ($('#input_source_user').is(':checked')) {
                    run_deft()
                    draw_chart()
               }
               else {
                    $('#google_chart').html('<h4> ERROR! Cant identify input type </h4>');
               }
               ">
               Get data <br> and <br> run DEFT 
            </a>
            </center>
        </div>

        <div class="col-sm-10" id="simulate_data_div">
            
            <!-- Press button to simulate data -->
            <p>
            <label 
                onclick="
                    $('#input_source_simulation').prop('checked',true)
                    $('#show_pdf').prop('disabled', false)
                ">
            <input type="radio" 
                id="input_source_simulation" 
                name="input_source" value="simulation" >
            Simulated data: &nbsp; 

            <!-- Select probability distribution -->
            <small>
            <select name='distribution' 
                    width='30' >
                <option value="gaussian">
                    Gaussain 
                </option>
                <option value="narrow">
                    Gaussian mixture, narrow separaiton
                </option>
                <option value="wide" selected='selected'>
                    Gaussian mixture, wide separation
                </option>
                <option value="accordian">
                    Accordian
                </option>
                <option value="foothills">
                    Foothills
                </option>
                <option value="goalposts">
                    Goalposts
                </option>
                <option value="towers">
                    Towers
                </option>
                <option value="uniform">
                    Uniform 
                </option>
                <option value="beta_convex">
                    Convex beta 
                </option>
                <option value="beta_concave">
                    Concave beta 
                </option>
                <option value="exponential">
                    Exponential 
                </option>
                <option value="gamma">
                    Gamma 
                </option>
                <option value="triangular">
                    Triangular 
                </option>
                <option value="laplace">
                    Laplace 
                </option>
                <option value="vonmises">
                    von Mises 
                </option>
            </select>
            &nbsp; &nbsp;

            <!-- Select number of data points -->
            N:
            <select name='num_samples'>
                <option value='10'> 10 </option>
                <option value='30'> 30 </option>
                <option value='100' selected='selected'> 100 </option>
                <option value='300'> 300 </option>
                <option value='1000'> 1,000 </option>
                <option value='10000'> 10,000 </option>
                <option value='100000'> 100,000 </option>
            </select> &nbsp; &nbsp;
            <input type="checkbox" id="use_simulation_presets" checked> 
            Presets
            </small>

            </label>
            </p>
        
            <p>
            <label onclick="
                $('#input_source_example').prop('checked',true);
                $('#show_pdf').prop('checked', false)
                $('#show_pdf').prop('disabled', true)
                ">
            <input 
                type="radio" 
                id="input_source_example" 
                name="input_source" 
                value="example" > 
            Example data:
            <small>
            
            <!-- Grab data from file-->
          
            <select name='example_data_file' 
                    width='30' >
                <option value="old_faithful_eruption_times.dat">
                    Old Faithful eruption times
                </option>
                <option value="old_faithful_waiting_times.dat">
                    Old Faithful waiting times
                </option>
                <option value="buffalo_snowfall.dat">
                    Buffalo snowfall
                </option>
                <option value="treatment_length.dat">
                    treatment length
                </option>
            </select>
            &nbsp;
            <input type="checkbox" id="use_example_presets" checked> 
            Presets
            </small>
            </label>
            </p>

            <p>
            <label onclick="
                $('#input_source_user').prop('checked',true)
                $('#show_pdf').prop('checked', false)
                $('#show_pdf').prop('disabled', true)
                ">
            <input type="radio" id="input_source_user" name="input_source" value="user">  
            Your data: &nbsp;
            <small>

            <input type="checkbox" id="automatic_box" name="automatic_box"checked> 
            Set box automatically. &nbsp;

            <input type="file" id="file_selector" style="display: none;" />
            <input 
                type="button" 
                id="load_file" 
                value="Choose file:" 
                onclick="document.getElementById('file_selector').click()">
            <a id="file_name"></a>
            </small>
            </label>
            </p>
            
            
        </div>

    </div>

    <hr>

    <div class="row">

        <div class="col-sm-3">
            <center>
            <a class="btn btn-md btn-success" role="button"
               onclick="run_deft(); draw_chart()">
               Rerun DEFT
            </a>
            </center>
        </div>

        <div class="col-sm-9" >
            <center>

            <!-- User specifies number of grid points and bounding box --> 
            
            &alpha;:
            <select id='alpha' name='alpha' width='1' onchange="$('#maxent_order').val($('#alpha').val())">
                <option value='1'>1</option>
                <option value='2'>2</option>
                <option value='3' selected='selected'>3</option>
                <option value='4'>4</option>
                <option value='5'>5</option>
            </select>.
            &nbsp;&nbsp;

            G:
            <select name='num_gridpoints' width='1'>
                <option value='20'> 20 </option>
                <option value='50'> 50 </option>
                <option value='100' selected='selected'> 100 </option>
                <option value='200'> 200 </option>
                <option value='500'> 500 </option>
                <option value='1000'> 1000 </option>
                <option value='2000'> 2000 </option>
            </select>.
            &nbsp;&nbsp;

            Box: [ 
            <input id='box_min' 
                   type='text' 
                   name='box_min' 
                   value='-6' 
                   size='4'
                   style="font-size:small"> 
            ,
            <input id='box_max'
                   type='text' 
                   name='box_max' 
                   value='6' 
                   size='4'
                   style="font-size:small"> ].
            &nbsp;&nbsp;

            <!-- Choose whether to enforce periodic boundary conditions -->
            Periodic:
            <select id='periodic' name='periodic'>
                <option value='False' selected='selected'>no</option>
                <option value='True'>yes</option>
            </select>. 

            </center>
        </div>

    </div>

</form>

<!-- 
JavaScript code for this page.
This is what does all the heavy lifting 
-->
<script src="js/deft_1d.js"></script>



<?php echo file_get_contents("./bottom.html"); ?>
   

    

