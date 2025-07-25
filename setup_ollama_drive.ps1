# Create directory for Ollama models on D drive
$ollamaDir = "D:\Ollama\models"
New-Item -ItemType Directory -Force -Path $ollamaDir

# Set OLLAMA_MODELS environment variable permanently
[System.Environment]::SetEnvironmentVariable("OLLAMA_MODELS", $ollamaDir, [System.EnvironmentVariableTarget]::User)

# Add to PATH if not already present
$currentPath = [System.Environment]::GetEnvironmentVariable("Path", [System.EnvironmentVariableTarget]::User)
if ($currentPath -notlike "*$ollamaDir*") {
    [System.Environment]::SetEnvironmentVariable("Path", "$currentPath;$ollamaDir", [System.EnvironmentVariableTarget]::User)
}

Write-Host "Ollama model directory set to: $ollamaDir"
Write-Host "Please restart your terminal and Ollama service for changes to take effect."
