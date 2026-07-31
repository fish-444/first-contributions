# farm_env.bat 의 로보플로우 설정을 싹 비우고 새로 받아 쓴다.
#
# 사람이 직접 편집하다 나는 사고를 전부 막는 것이 목적이다:
#   · farm_env.example.bat(깃에 올라가는 견본)을 대신 고치는 실수
#   · 배치 파일에서 set X="키" 라고 써서 따옴표까지 값이 되는 문제
#   · 붙여넣을 때 딸려오는 줄 끝 공백
#   · 메모장이 UTF-8 BOM 을 붙여 첫 줄(@echo off)이 깨지는 문제
#   · 옛 값이 어딘가 남아 새 값과 섞이는 문제 → 관련 줄을 통째로 새로 쓴다
#
# 로보플로우와 무관한 설정(FARM_PORT, FARM_DB, CONFIDENCE 등)은 그대로 옮긴다.
# set_key.bat 이 이 파일을 불러 준다 — 직접 실행할 일은 없다.

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$envFile = Join-Path $PSScriptRoot 'farm_env.bat'
$backup  = Join-Path $PSScriptRoot 'farm_env.bak'

# 이 접두사로 시작하는 설정은 전부 지우고 새로 쓴다
$roboflowPattern = '^\s*set\s+ROBOFLOW_'

function Show-Line($text, $color) { Write-Host $text -ForegroundColor $color }

Write-Host ''
Show-Line '==========================================================' Cyan
Show-Line '   로보플로우 설정 새로 넣기' Cyan
Show-Line '==========================================================' Cyan
Write-Host ''
Write-Host '  기존 로보플로우 설정(키·워크스페이스·워크플로)을 전부 지우고'
Write-Host '  새로 받습니다. 나머지 설정과 farm.db(식물 정보)는 그대로 둡니다.'
Write-Host ''

# --- 기존 파일에서 로보플로우 밖 설정만 건져 둔다 ------------------------
$keep = @()
$hadFile = Test-Path -LiteralPath $envFile
if ($hadFile) {
    Copy-Item -LiteralPath $envFile -Destination $backup -Force
    foreach ($line in [System.IO.File]::ReadAllLines($envFile)) {
        if ($line -match '^\s*set\s+' -and $line -notmatch $roboflowPattern) { $keep += $line }
    }
    Show-Line "  · 기존 farm_env.bat 을 farm_env.bak 으로 백업했습니다." DarkGray
    if ($keep.Count) {
        Show-Line "  · 로보플로우와 무관한 설정 $($keep.Count)줄은 그대로 옮깁니다." DarkGray
    }
}
Write-Host ''

# --- 1) API 키 -----------------------------------------------------------
Write-Host '  [1/3] Private API Key'
Write-Host '        https://app.roboflow.com/s-workspace-br86f/settings/api'
Write-Host '        복사 아이콘을 쓰세요. 드래그로 고르면 잘리거나 공백이 딸려옵니다.'
Write-Host ''
$raw = Read-Host '        키'

$key = $raw.Trim()
foreach ($q in @('"', "'")) {
    if ($key.Length -ge 2 -and $key.StartsWith($q) -and $key.EndsWith($q)) {
        $key = $key.Substring(1, $key.Length - 2).Trim()
        Show-Line "        · 감싼 $q 를 떼고 저장합니다." Yellow
    }
}

if ([string]::IsNullOrWhiteSpace($key)) {
    Write-Host ''
    Show-Line '  [중단] 키가 비어 있습니다. 아무것도 바꾸지 않았습니다.' Red
    exit 1
}
if ($key -match '\s') {
    Write-Host ''
    Show-Line '  [중단] 키 중간에 공백이 있습니다 — 붙여넣다 잘린 것 같습니다.' Red
    Show-Line '         다시 복사해서 실행해 주세요. 아무것도 바꾸지 않았습니다.' Red
    exit 1
}
if ($key.StartsWith('rf_')) {
    Write-Host ''
    Show-Line '  [중단] rf_ 로 시작하는 키는 공개(publishable) 키라 막힙니다.' Red
    Show-Line '         Private API Key 를 넣어 주세요. 아무것도 바꾸지 않았습니다.' Red
    exit 1
}

# --- 2) 워크스페이스 -----------------------------------------------------
Write-Host ''
Write-Host '  [2/3] 워크스페이스 이름  (그냥 엔터 = s-workspace-br86f)'
$workspace = (Read-Host '        워크스페이스').Trim().Trim('"').Trim("'").Trim()
if ([string]::IsNullOrWhiteSpace($workspace)) { $workspace = 's-workspace-br86f' }

# --- 3) 워크플로 ID ------------------------------------------------------
Write-Host ''
Write-Host '  [3/3] 워크플로 ID  (그냥 엔터 = find-old-leaf-and-others)'
$workflow = (Read-Host '        워크플로 ID').Trim().Trim('"').Trim("'").Trim()
if ([string]::IsNullOrWhiteSpace($workflow)) { $workflow = 'find-old-leaf-and-others' }

# --- 파일 쓰기 -----------------------------------------------------------
$out = @(
    '@echo off',
    'rem ════════════════════════════════════════════════════════════════',
    'rem  이 파일은 set_key.bat 이 만들어 줍니다. 손으로 고치지 마세요.',
    'rem  키를 바꾸시려면 set_key.bat 을 다시 더블클릭하시면 됩니다.',
    'rem  이 파일은 깃에 올라가지 않습니다 (키가 새 나가지 않게).',
    'rem ════════════════════════════════════════════════════════════════',
    '',
    'rem 로보플로우 Private API Key',
    "set ROBOFLOW_API_KEY=$key",
    '',
    'rem 워크플로 방식',
    "set ROBOFLOW_WORKSPACE=$workspace",
    "set ROBOFLOW_WORKFLOW_ID=$workflow"
)
if ($keep.Count) {
    $out += @('', 'rem 그 밖의 설정 (이전 farm_env.bat 에서 그대로 옮김)')
    $out += $keep
}

# BOM 없는 UTF-8 로 쓴다. 메모장이 붙이는 BOM 은 배치의 첫 줄을 깨뜨린다.
[System.IO.File]::WriteAllLines($envFile, [string[]]$out, (New-Object System.Text.UTF8Encoding $false))

$head = $key.Substring(0, [Math]::Min(4, $key.Length))
Write-Host ''
Show-Line '  ✔ farm_env.bat 을 새로 썼습니다.' Green
Write-Host ''
Show-Line "      키          $head…  ($($key.Length)자)" Green
Show-Line "      워크스페이스  $workspace" Green
Show-Line "      워크플로     $workflow" Green
Write-Host ''
Write-Host '  키 앞자리가 대시보드에 보이는 것과 같은지 확인해 보세요.'
Write-Host '  다르면 붙여넣기가 잘못된 겁니다 — 다시 실행하시면 됩니다.'
Write-Host ''
Write-Host '  이제 서버를 껐다 켜세요 (start.bat).'
Write-Host '  서버 창에 이렇게 뜨면 성공입니다:'
Show-Line "      [키] $head… ($($key.Length)자)" DarkGray
Show-Line "      [분석 엔진] workflow · $workflow" DarkGray
Write-Host ''
