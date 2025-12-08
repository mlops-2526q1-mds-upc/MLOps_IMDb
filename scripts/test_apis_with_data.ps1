# Test APIs with real data from processed datasets
# Usage: .\scripts\test_apis_with_data.ps1 [-prod]
#   -prod: Use production server (http://34.135.54.197) instead of localhost

param(
    [switch]$prod
)

$ErrorActionPreference = "Continue"

# Set base URL based on -prod parameter
if ($prod) {
    $baseUrl = "http://34.135.54.197"
    Write-Host "Using PRODUCTION server: $baseUrl" -ForegroundColor Magenta
} else {
    $baseUrl = "http://localhost"
    Write-Host "Using LOCAL server: $baseUrl" -ForegroundColor Cyan
}

# API endpoints - properly construct URLs using string concatenation
$sentimentApiUrl = $baseUrl + ":8001"
$spamApiUrl = $baseUrl + ":8000"
$sentimentInternalUrl = $baseUrl + ":9001"
$spamInternalUrl = $baseUrl + ":9000"

Write-Host "=== Testing Sentiment API with IMDb Test Data ===" -ForegroundColor Cyan

# Read samples from IMDb test data
$imdbData = Import-Csv -Path "data\processed\imdb_test_clean.csv" -Encoding UTF8
# Randomly select 10 samples
$imdbSamples = $imdbData | Get-Random -Count 10

Write-Host "`nTesting Sentiment API with 10 samples..." -ForegroundColor Yellow
$sentimentResults = @()

foreach ($sample in $imdbSamples) {
    $text = $sample.text
    $expectedLabel = [int]$sample.label
    
    try {
        $body = @{text = $text} | ConvertTo-Json -Compress -Depth 10
        $response = Invoke-RestMethod -Uri "$sentimentApiUrl/predict" -Method Post -ContentType "application/json; charset=utf-8" -Body $body
        
        $match = ($expectedLabel -eq $response.label)
        $sentimentResults += [PSCustomObject]@{
            ExpectedLabel = $expectedLabel
            PredictedLabel = $response.label
            PredictedSentiment = $response.sentiment
            Probability = $response.probability
            Match = $match
        }
        
        Write-Host "  Expected: $expectedLabel, Predicted: $($response.label) ($($response.sentiment)), Prob: $([math]::Round($response.probability, 3))" -ForegroundColor $(if ($match) { "Green" } else { "Red" })
    }
    catch {
        Write-Host "  Error: $_" -ForegroundColor Red
    }
    
    Start-Sleep -Milliseconds 100
}

$sentimentAccuracy = ($sentimentResults | Where-Object { $_.Match }).Count / $sentimentResults.Count
Write-Host "`nSentiment API Accuracy: $([math]::Round($sentimentAccuracy * 100, 2))% ($(($sentimentResults | Where-Object { $_.Match }).Count)/$($sentimentResults.Count))" -ForegroundColor $(if ($sentimentAccuracy -ge 0.8) { "Green" } else { "Yellow" })

Write-Host "`n=== Testing Spam API with Spam Test Data ===" -ForegroundColor Cyan

# Read samples from spam test data
$spamData = Import-Csv -Path "data\processed\spam_test_clean.csv" -Encoding UTF8
# Randomly select 10 samples
$spamSamples = $spamData | Get-Random -Count 10

Write-Host "`nTesting Spam API with 10 samples..." -ForegroundColor Yellow
$spamResults = @()

foreach ($sample in $spamSamples) {
    # Use clean_text if available, otherwise use text
    $text = if ($sample.clean_text) { $sample.clean_text } else { $sample.text }
    $expectedLabel = [int]$sample.label
    
    try {
        $body = @{text = $text} | ConvertTo-Json -Compress -Depth 10
        $response = Invoke-RestMethod -Uri "$spamApiUrl/predict" -Method Post -ContentType "application/json; charset=utf-8" -Body $body
        
        $match = ($expectedLabel -eq $response.label)
        $spamResults += [PSCustomObject]@{
            ExpectedLabel = $expectedLabel
            PredictedLabel = $response.label
            Probability = $response.probability
            Match = $match
        }
        
        $labelText = if ($response.label -eq 1) { "SPAM" } else { "NOT SPAM" }
        Write-Host "  Expected: $expectedLabel, Predicted: $($response.label) ($labelText), Prob: $([math]::Round($response.probability, 3))" -ForegroundColor $(if ($match) { "Green" } else { "Red" })
    }
    catch {
        Write-Host "  Error: $_" -ForegroundColor Red
    }
    
    Start-Sleep -Milliseconds 100
}

$spamAccuracy = ($spamResults | Where-Object { $_.Match }).Count / $spamResults.Count
Write-Host "`nSpam API Accuracy: $([math]::Round($spamAccuracy * 100, 2))% ($(($spamResults | Where-Object { $_.Match }).Count)/$($spamResults.Count))" -ForegroundColor $(if ($spamAccuracy -ge 0.8) { "Green" } else { "Yellow" })

Write-Host "`n=== Generating Monitoring Data ===" -ForegroundColor Cyan

# Generate 60 unique predictions for sentiment API
Write-Host "`nGenerating 60 unique predictions for Sentiment API..." -ForegroundColor Yellow

# Randomly select 60 samples from the full dataset
$imdbMoreSamples = $imdbData | Get-Random -Count 60
$sentimentCount = 0
$sentimentFailed = 0

foreach ($sample in $imdbMoreSamples) {
    try {
        $body = @{text = $sample.text} | ConvertTo-Json -Compress -Depth 10
        $null = Invoke-RestMethod -Uri "$sentimentApiUrl/predict" -Method Post -ContentType "application/json; charset=utf-8" -Body $body
        $sentimentCount++
        if ($sentimentCount % 10 -eq 0) {
            Write-Host "  Sentiment: $sentimentCount/60 predictions..." -ForegroundColor Gray
        }
    }
    catch {
        $sentimentFailed++
        Write-Host "  Error on sentiment sample ${sentimentCount}: $_" -ForegroundColor Red
    }
    
    Start-Sleep -Milliseconds 50
}

Write-Host "  Sentiment API: $sentimentCount successful, $sentimentFailed failed" -ForegroundColor $(if ($sentimentFailed -eq 0) { "Green" } else { "Yellow" })

# Generate 60 unique predictions for spam API
Write-Host "`nGenerating 60 unique predictions for Spam API..." -ForegroundColor Yellow

# Randomly select 60 samples from the full dataset
$spamMoreSamples = $spamData | Get-Random -Count 60
$spamCount = 0
$spamFailed = 0

foreach ($sample in $spamMoreSamples) {
    try {
        $text = if ($sample.clean_text) { $sample.clean_text } else { $sample.text }
        $body = @{text = $text} | ConvertTo-Json -Compress -Depth 10
        $null = Invoke-RestMethod -Uri "$spamApiUrl/predict" -Method Post -ContentType "application/json; charset=utf-8" -Body $body
        $spamCount++
        if ($spamCount % 10 -eq 0) {
            Write-Host "  Spam: $spamCount/60 predictions..." -ForegroundColor Gray
        }
    }
    catch {
        $spamFailed++
        Write-Host "  Error on spam sample ${spamCount}: $_" -ForegroundColor Red
    }
    
    Start-Sleep -Milliseconds 50
}

Write-Host "  Spam API: $spamCount successful, $spamFailed failed" -ForegroundColor $(if ($spamFailed -eq 0) { "Green" } else { "Yellow" })

# Check monitoring status
Write-Host "`n=== Checking Monitoring Status ===" -ForegroundColor Cyan

try {
    $driftStatus = Invoke-RestMethod -Uri "$sentimentInternalUrl/monitoring/drift"
    Write-Host "`nDrift Status:" -ForegroundColor Yellow
    Write-Host "  Status: $($driftStatus.status)" -ForegroundColor $(if ($driftStatus.drift_detected) { "Red" } else { "Green" })
    Write-Host "  Sample Size: $($driftStatus.sample_size)" -ForegroundColor Gray
    if ($driftStatus.p_value) {
        Write-Host "  P-Value: $($driftStatus.p_value)" -ForegroundColor Gray
    }
}
catch {
    Write-Host "  Could not check drift status: $_" -ForegroundColor Red
}

try {
    $stats = Invoke-RestMethod -Uri "$sentimentInternalUrl/monitoring/stats"
    Write-Host "`nSentiment Prediction Stats:" -ForegroundColor Yellow
    Write-Host "  Total Predictions: $($stats.total_predictions)" -ForegroundColor Gray
    Write-Host "  Positive: $($stats.positive_predictions), Negative: $($stats.negative_predictions)" -ForegroundColor Gray
    Write-Host "  Avg Confidence: $([math]::Round($stats.avg_confidence, 3))" -ForegroundColor Gray
}
catch {
    Write-Host "  Could not check sentiment stats: $_" -ForegroundColor Red
}

try {
    $spamStats = Invoke-RestMethod -Uri "$spamInternalUrl/monitoring/stats"
    Write-Host "`nSpam Prediction Stats:" -ForegroundColor Yellow
    Write-Host "  Total Predictions: $($spamStats.total_predictions)" -ForegroundColor Gray
    Write-Host "  Positive: $($spamStats.positive_predictions), Negative: $($spamStats.negative_predictions)" -ForegroundColor Gray
    Write-Host "  Avg Confidence: $([math]::Round($spamStats.avg_confidence, 3))" -ForegroundColor Gray
}
catch {
    Write-Host "  Could not check spam stats: $_" -ForegroundColor Red
}

Write-Host "`n=== Test Complete ===" -ForegroundColor Cyan

