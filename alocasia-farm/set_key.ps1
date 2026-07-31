# 로보플로우 API 키를 farm_env.bat 에 안전하게 써 넣는다.
#
# 사람이 직접 편집하다 나는 사고를 전부 막는 것이 목적이다:
#   · farm_env.example.bat(깃에 올라가는 견본)을 대신 고치는 실수
#   · 배치 파일에서 set X="키" 라고 써서 따옴표까지 값이 되는 문제
#   · 붙여넣을 때 딸려오는 줄 끝 공백
#   · 메모장이 UTF-8 BOM 을 붙여 첫 줄(@echo off)이 깨지는 문제
#
# set_key.bat 이 이 파일을 불러 준다 — 직접 실행할 일은 없다.

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$envFile = Join-Path $PSScriptRoot 'farm_env.bat'
$example = Join-Path $PSScriptRoot 'farm_env.example.bat'

Write-Host ''
Write-Host '==========================================================' -ForegroundColor Cyan
Write-Host '   로보플로우 API 키 넣기' -ForegroundColor Cyan
Write-Host '==========================================================' -ForegroundColor Cyan
Write-Host ''
Write-Host '  app.roboflow.com > 우측 상단 프로필 > Settings > API Keys 에서'
Write-Host '  Private API Key 를 복사해 오세요. (rf_ 로 시작하는 공개키는 안 됩니다)'
Write-Host ''
Write-Host '  붙여넣기: 창에서 마우스 오른쪽 클릭 또는 Ctrl+V'
Write-Host ''

$raw = Read-Host '  키'

# --- 다듬기 --------------------------------------------------------------
$key = $raw.Trim()
foreach ($q in @('"', "'")) {
    if ($key.Length -ge 2 -and $key.StartsWith($q) -and $key.EndsWith($q)) {
        $key = $key.Substring(1, $key.Length - 2).Trim()
        Write-Host "  · 감싼 $q 를 떼고 저장합니다." -ForegroundColor Yellow
    }
}

if ([string]::IsNullOrWhiteSpace($key)) {
    Write-Host ''
    Write-Host '  [중단] 키가 비어 있습니다. 아무것도 바꾸지 않았습니다.' -ForegroundColor Red
    exit 1
}
if ($key -match '\s') {
    Write-Host ''
    Write-Host '  [중단] 키 중간에 공백이 있습니다 — 붙여넣다 잘린 것 같습니다.' -ForegroundColor Red
    Write-Host '         다시 복사해서 실행해 주세요. 아무것도 바꾸지 않았습니다.' -ForegroundColor Red
    exit 1
}
if ($key.StartsWith('rf_')) {
    Write-Host ''
    Write-Host '  [중단] rf_ 로 시작하는 키는 공개(publishable) 키라 워크플로에서 막힙니다.' -ForegroundColor Red
    Write-Host '         Private API Key 를 넣어 주세요. 아무것도 바꾸지 않았습니다.' -ForegroundColor Red
    exit 1
}

# --- farm_env.bat 준비 ---------------------------------------------------
if (-not (Test-Path -LiteralPath $envFile)) {
    if (-not (Test-Path -LiteralPath $example)) {
        Write-Host ''
        Write-Host '  [중단] farm_env.example.bat 을 찾을 수 없습니다.' -ForegroundColor Red
        Write-Host '         alocasia-farm 폴더에서 실행하고 계신지 확인해 주세요.' -ForegroundColor Red
        exit 1
    }
    Copy-Item -LiteralPath $example -Destination $envFile
    Write-Host ''
    Write-Host '  · farm_env.bat 이 없어서 견본에서 새로 만들었습니다.' -ForegroundColor Yellow
}

# --- 키 줄 교체 (없으면 추가) --------------------------------------------
$lines   = [System.IO.File]::ReadAllLines($envFile)
$pattern = '^\s*set\s+ROBOFLOW_API_KEY\s*='
$newLine = "set ROBOFLOW_API_KEY=$key"

$found = $false
$out = foreach ($line in $lines) {
    if (-not $found -and $line -match $pattern) {
        $found = $true
        $newLine
    } else {
        $line
    }
}
if (-not $found) { $out = @($out) + @('', 'rem 로보플로우 Private API Key', $newLine) }

# BOM 없는 UTF-8 로 쓴다. 메모장이 붙이는 BOM 은 배치의 첫 줄을 깨뜨린다.
[System.IO.File]::WriteAllLines($envFile, [string[]]$out, (New-Object System.Text.UTF8Encoding $false))

$head = $key.Substring(0, [Math]::Min(4, $key.Length))
Write-Host ''
Write-Host '  ✔ farm_env.bat 에 저장했습니다.' -ForegroundColor Green
Write-Host "     앞자리 $head…  ($($key.Length)자)" -ForegroundColor Green
Write-Host '     로보플로우 대시보드에 보이는 키 앞자리와 같은지 확인해 보세요.'
Write-Host ''
Write-Host '  이제 서버를 껐다 켜면 됩니다 (start.bat).'
Write-Host '  서버 창에 [분석 엔진] workflow · ... 가 뜨면 성공입니다.'
Write-Host ''
