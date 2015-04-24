
<?php // getData.php
// Performs DEFT density estimation
// Sends density estimate back to client in JSON format

$data_directory = 'data/';

// Remove keys set to auto
foreach($_POST as $key => $value) {
    if ($value == 'auto') {
        unset($_POST[$key]);
        //echo '$_POST[' . $key . '] removed!<br>';
    }
}

$data_string = sanitizeString($_POST['data']);

// Write data to temporary file
$data_file_name = 'uploads/temp.dat';
$data_file = fopen($data_file_name, 'w') or 
                die("Unable to open file!");
fwrite($data_file, $data_string);
fclose($data_file);

// Build commandline command
$command = 'cat ' . $data_file_name . ' | ../code/kde_1d.py';

if (array_key_exists('num_gridpoints', $_POST))
    $command .= " --num_gridpoints=". sanitizeString($_POST['num_gridpoints']);

if (array_key_exists('box_min', $_POST))
    $command .= " --box_min=" . sanitizeString($_POST['box_min']);

if (array_key_exists('box_max', $_POST))
    $command .= " --box_max=" . sanitizeString($_POST['box_max']);

if (array_key_exists('periodic', $_POST)) {
    if (sanitizeString($_POST['periodic']) == 'True') {
        $command .= ' --periodic';
    }
}

$command .= ' --json';

// Save last command in file. Useful for debugging purposes
$myfile = fopen("commandline/last_kde_1d_command.sh","w");
fwrite($myfile, $command . "\n");
fclose($myfile);

// IMPORTANT DEBUGGING COMMENT
//echo "Command:<br><pre>" . $command . "</pre><br>";

// Execute shell code
$json_data = shell_exec($command);

// SHOULD SEND ESTIMATION PARAMETERS BACK AS WELL
// Send JSON formatted data back to client
echo $json_data;

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


?>
    