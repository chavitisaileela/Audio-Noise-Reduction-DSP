# Audio Noise Reduction - Automated GitHub Push Script
# User: chavitisaileela (Chaviti Sai Leela)
# Repo: Audio-Noise-Reduction-DSP

$GitHubUsername = "chavitisaileela"
$repoName = "Audio-Noise-Reduction-DSP"
$repoUrl = "https://github.com/$GitHubUsername/$repoName.git"

Write-Host "`n========================================================" -ForegroundColor Cyan
Write-Host "   Audio Noise Reduction: Direct GitHub Push Script     " -ForegroundColor Cyan
Write-Host "   Target: $repoUrl" -ForegroundColor Cyan
Write-Host "========================================================`n" -ForegroundColor Cyan

# 1. Check if git is available
$gitCmd = Get-Command git -ErrorAction SilentlyContinue

if (-not $gitCmd) {
    Write-Host "[1/4] Git is not installed on this system. Installing Git via winget..." -ForegroundColor Yellow
    winget install --id Git.Git -e --source winget --accept-source-agreements --accept-package-agreements
    
    # Refresh PATH in current PowerShell session
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    
    # Check default install path
    if (Test-Path "C:\Program Files\Git\cmd\git.exe") {
        $env:Path += ";C:\Program Files\Git\cmd"
    } elseif (Test-Path "$env:LOCALAPPDATA\Programs\Git\cmd\git.exe") {
        $env:Path += ";$env:LOCALAPPDATA\Programs\Git\cmd"
    }
}

$gitCmd = Get-Command git -ErrorAction SilentlyContinue
if (-not $gitCmd) {
    Write-Host "`n[!] Git was installed, but Windows needs a new terminal session to load the PATH." -ForegroundColor Yellow
    Write-Host "Please close this PowerShell window, open a NEW PowerShell window, and run:" -ForegroundColor Cyan
    Write-Host "cd C:\Users\venka\.gemini\antigravity\scratch\audio_noise_reduction"
    Write-Host ".\push_to_github.ps1"
    exit 1
}

Write-Host "[OK] Git is available: $(git --version)" -ForegroundColor Green

# 2. Configure Git user details
git config --global user.name "Chaviti Sai Leela"
git config --global user.email "chavitisaileela@users.noreply.github.com"

# 3. Initialize repository
Write-Host "`n[2/4] Initializing local Git repository..." -ForegroundColor Yellow
if (-not (Test-Path ".git")) {
    git init
}

# 4. Add all files
Write-Host "[3/4] Staging files and creating commit..." -ForegroundColor Yellow
git add .
git commit -m "Initial commit: Audio Noise Reduction DSP mini-project"

# 5. Set branch and remote
Write-Host "[4/4] Connecting and pushing to GitHub..." -ForegroundColor Yellow
git branch -M main

$existingRemote = git remote get-url origin 2>$null
if ($existingRemote) {
    git remote set-url origin $repoUrl
} else {
    git remote add origin $repoUrl
}

Write-Host "Pushing code to $repoUrl..." -ForegroundColor Cyan
Write-Host "(A browser window will appear asking you to sign in to GitHub)`n" -ForegroundColor DarkGray

git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n==================================================================" -ForegroundColor Green
    Write-Host " SUCCESS! Code pushed to GitHub successfully!" -ForegroundColor Green
    Write-Host " Repository URL: $repoUrl" -ForegroundColor Green
    Write-Host " Next Step: Deploy on Streamlit at https://share.streamlit.io" -ForegroundColor Cyan
    Write-Host "==================================================================" -ForegroundColor Green
} else {
    Write-Host "`n[!] Push failed. Please verify:" -ForegroundColor Yellow
    Write-Host "1. Is the repository created at: https://github.com/chavitisaileela/Audio-Noise-Reduction-DSP ?"
    Write-Host "2. Did you sign in to GitHub on the browser prompt?"
}
