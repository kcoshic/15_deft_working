
<?php 

# simulate_data_1d.php
#
# Simulates data. Retrieves simulation specification as form data via POST. 
# Returns simulated data in JSON format. 
#

# Sanitizes a string. Use IMMEDIATELY upon processing information from $POST. 
# Use even if the user is not prompted to manually input that value, because
# hackers can still do this.
function sanitizeString($var)
{
    $var = stripslashes($var);
    $var = strip_tags($var);
    $var = htmlentities($var);
    return $var;
}

# Test for valid JSON string
function isJson($string) {
	json_decode($string);
	return (json_last_error() == JSON_ERROR_NONE);
}

if (!$_POST) {
	echo "ERROR: No data in $_POST.";
}

# Get name of distribution
$distribution = sanitizeString($_POST['distribution']);

# Get number of samples to generate
$num_samples = sanitizeString($_POST['num_samples']);

# Construct command to simulate data
$command = '../sim/simulate_data_1d.py --distribution=' 
    . $distribution 
    . ' --num_samples=' 
    . $num_samples
    . ' --json';

// Save last command in file. Useful for debugging purposes
$myfile = fopen("commandline/last_sim_command.sh","w");
fwrite($myfile, $command . "\n");
fclose($myfile);
        
# Execute shell code. Output should be in JSON format
$json_data = shell_exec($command);

# Check for validity of JSON string
if (!isJson($json_data)) {
	throw new Exception('ERROR: JSON string returned form simulate_data_1d.py is not valid. ');
}

# Send JSON formatted data back to client
echo $json_data;

?>
    