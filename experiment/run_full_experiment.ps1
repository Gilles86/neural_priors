param(
    [Parameter(Mandatory=$true)]
    [string]$subject,

    [Parameter(Mandatory=$true)]
    [int]$session,

    [Parameter(Mandatory=$true)]
    [string]$initialWidth
)

# Define Python interpreter command
$pythonCommand = "python"

# Get the current working directory
$currentDirectory = Get-Location

# Define relative paths for script files
$script1 = Join-Path $currentDirectory "examples.py"
$script2 = Join-Path $currentDirectory "feedback.py"
$script3 = Join-Path $currentDirectory "task.py"
$script4 = Join-Path $currentDirectory "score.py"

# Define common arguments
$commonArgs = "--settings"
$commonArgsValue = "scanner"
$calibrateArg = "--calibrate_eyetracker"

# Run script 1 with --calibrate_eyetracker argument
Write-Host "Executing command: $pythonCommand $script1 $subject $session 1 $initialWidth $calibrateArg $commonArgs $commonArgsValue"
& $pythonCommand $script1 $subject $session 1 $initialWidth $calibrateArg $commonArgs $commonArgsValue

# Run script 2
Write-Host "Executing command: $pythonCommand $script2 $subject $session 1 $initialWidth $commonArgs $commonArgsValue"
& $pythonCommand $script2 $subject $session 1 $initialWidth $commonArgs $commonArgsValue

# Run script 3 for the first block
for ($i = 1; $i -le 4; $i++) {
    Write-Host "Executing command: $pythonCommand $script3 $subject $session $i $initialWidth $commonArgs $commonArgsValue"
    & $pythonCommand $script3 $subject $session $i $initialWidth $commonArgs $commonArgsValue
}

# Determine final width for the second block
if ($initialWidth -eq "narrow") {
    $finalWidth = "wide"
} else {
    $finalWidth = "narrow"
}

# Run script 1 for the second block
Write-Host "Executing command: $pythonCommand $script1 $subject $session 5 $finalWidth $commonArgs $commonArgsValue"
& $pythonCommand $script1 $subject $session 5 $finalWidth $commonArgs $commonArgsValue

# Run script 2 for the second block
Write-Host "Executing command: $pythonCommand $script2 $subject $session 5 $finalWidth $commonArgs $commonArgsValue"
& $pythonCommand $script2 $subject $session 5 $finalWidth $commonArgs $commonArgsValue

# Run script 3 for the second block
for ($i = 5; $i -le 8; $i++) {
    Write-Host "Executing command: $pythonCommand $script3 $subject $session $i $finalWidth $commonArgs $commonArgsValue"
    & $pythonCommand $script3 $subject $session $i $finalWidth $commonArgs $commonArgsValue
}

# Run script 4
Write-Host "Executing command: $pythonCommand $script4 $subject $session $commonArgs $commonArgsValue"
& $pythonCommand $script4 $subject $session $commonArgs $commonArgsValue
