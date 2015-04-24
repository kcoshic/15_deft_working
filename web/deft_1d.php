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

            <!-- DIV title for Google Chart -->
            <div id="data_title"><h4>&nbsp;</h4></div> 

            <!-- DIV for Google Chart -->
            <div id="google_chart" style="width: 500px; height: 250px"></div>

            <!-- This is where the data is stored -->
            <input type="hidden" id="data" name="data">

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

            <input type="checkbox" 
                id="show_errorbars"
                onclick="draw_chart();"
                checked> 
             Errorbars &nbsp;&nbsp; 

             Plausible  
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
                id="show_pdf" 
                onclick="draw_chart();"> 
             True &nbsp;&nbsp; 

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
                id="show_gmm"
                onclick="
                    if ($('#show_gmm').is(':checked')) {
                        run_gmm()
                    } 
                    draw_chart()
                    "> 
            GMM &nbsp;&nbsp;

            </center> 
        </div>

    </div>

    <hr>

    <div class="row">
        <!-- Specify data -->
        
        <div class="col-sm-4" id="simulate_data_div">
            
            <!-- Press button to simulate data -->
            <center>
            <input type="radio" id="input_source_simulation" name="input_source" value="simulation" style="display:none"> 
            <p>
            <a class="row btn btn-md btn-success" role="button"
                 onclick="get_simulated_data();">
                Simulate data
            </a>
            </p>
            <small>
            <!-- Select probability distribution -->
            <p>
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
            </p>
            

            <!-- Select number of data points -->
            <p>
            N:
            <select name='num_samples' 
                    width='5'>
                <option value='10'> 10 </option>
                <option value='30'> 30 </option>
                <option value='100' selected='selected'> 100 </option>
                <option value='300'> 300 </option>
                <option value='1000'> 1,000 </option>
                <option value='10000'> 10,000 </option>
                <option value='100000'> 100,000 </option>
            </select>. &nbsp;
            <input type="checkbox" id="use_simulation_presets" checked> 
            Use presets
            </p>
            </small>
            </center>

        </div>

        <div class="col-sm-4">

            <center>
            <input type="radio" id="input_source_example" name="input_source" value="example" style="display:none"> 

            <p>
            <!-- Press button to simulate data -->
            <a class="btn btn-md btn-success" role="button"
               onclick="get_example_data()">
               Load example data
            </a>
            </p>
            <small>
            <!-- Grab dta from file-->
            <p>
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
            </select>.
            </p>

            <p>
            <input type="checkbox" id="use_example_presets" checked> 
            Use presets
            </p>
            </small>
            </center>

        </div>
        <div class="col-sm-4"> 

            <center>
            <input type="radio" id="input_source_user" name="input_source" value="user" style="display:none">  

            <p>
            <input type="file" id="file_selector" style="display: none;" />
            <a class="btn btn-md btn-success" role="button" 
               onclick="document.getElementById('file_selector').click();">Upload your own data </a>
            </p>

            <small>
            <p>
            <input type="checkbox" id="automatic_box" name="automatic_box"checked> 
            Set box automatically
            </p>
            </small>
            
            </center>
        </div>
    </div>

    <hr>

    <div class="row">

        <div class="col-sm-9" >
            <center>

            <!-- User specifies number of grid points and bounding box --> 
            
            &alpha;:
            <select id='alpha' name='alpha' width='1'>
                <option value='1'>1</option>
                <option value='2'>2</option>
                <option value='3' selected='selected'>3</option>
            </select>.
            &nbsp;&nbsp;

            G:
            <select name='num_gridpoints' width='1'>
                <option value='30'> 30 </option>
                <option value='100' selected='selected'> 100 </option>
                <option value='300'> 300 </option>
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

        <div class="col-sm-3">
            <center>
            <a class="btn btn-md btn-success" role="button"
               onclick="run_deft(); draw_chart()">
               Rerun DEFT 1D
            </a>
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
   

    

