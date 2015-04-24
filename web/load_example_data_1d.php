
<?php 

# load_example_data_1d.php
#
# Loads a data file into memory. Prints it to user in JSON format
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

# Make sure there is data in $_POST
if (!$_POST) {
	throw new Exception('ERROR: $_POST has no content. ');
}

# Get name of example data file
$file_name = '../data/' . sanitizeString($_POST['example_data_file']);

# Open file
$handle = fopen($file_name, "r");

# Make sure file is open
if (!$handle)
    throw new Exception('ERROR: could not open file ' . $file_name);

// Read in file line by line
$json_string = '{';
$in_header = True;
while (!feof($handle)) {
    $line = fgets($handle);

    if ($line[0] == "#") {
        $json_string .= chop(substr($line, 1)) . ',';
    }
    elseif (strlen(chop($line)) > 0) {

        // If just exiting header, add "data"
        if ($in_header) {
            $in_header = False;
            $json_string .= '"data":[' . chop($line);
        }
        // Otherwise, just add entry to data array
        else {
            $json_string .= ', ' . chop($line);
        }
    }
}
$json_string .= ']}';

// Close file
fclose($handle);
   
# Check for validity of JSON string
#if (!isJson($json_string)) {
#	throw new Exception('ERROR: JSON string returned form load_example_data_1d.py is not valid. ');
#}

# Send JSON formatted data back to client
echo $json_string;

?>
    