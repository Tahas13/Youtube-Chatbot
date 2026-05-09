Param(
    [string]$ApiBase = "http://localhost:8000"
)

Write-Host "Setting VITE_API_BASE to $ApiBase in .env.local"

$envPath = Join-Path $PSScriptRoot ".env.local"
if (-Not (Test-Path $envPath)) {
    Write-Host "Creating .env.local"
    "# Local dev override for the backend API base" | Out-File -FilePath $envPath -Encoding utf8
    "VITE_API_BASE=$ApiBase" | Out-File -FilePath $envPath -Append -Encoding utf8
    "VITE_CHAT_STREAMING=true" | Out-File -FilePath $envPath -Append -Encoding utf8
} else {
    (Get-Content $envPath) -replace 'VITE_API_BASE=.*', "VITE_API_BASE=$ApiBase" | Set-Content $envPath -Encoding utf8
}

Write-Host "Installing dependencies (npm install)"
npm install

Write-Host "Building extension (npm run build)"
npm run build

$dist = Join-Path $PSScriptRoot 'dist'
Write-Host "Build finished. Load unpacked extension from: $dist"
Write-Host "Open chrome://extensions/ → Developer mode → Load unpacked → select the folder above."