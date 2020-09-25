$list = Get-ChildItem -Path '.\img\new\' | 
        Where-Object { $_.PSIsContainer -eq $false -and $_.Extension -ne '.csv' }

ForEach($pic in $list) {
  $pic.Name | Out-File -Append .\img\new\images.csv
}