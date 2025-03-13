# 80 % : 50 MB
$chunkSize = 40 * 1024 * 1024
$inputfile = ".\data\neural_network\pneumonia_model.h5"
$outputPrefix = "model_part_"


$byte = [System.IO.File]::ReadAllBytes($inputfile)
$size = $byte.Length


$part_iter = 0
$remains = $size
for ($i = 0; $i -lt $size; $i += $chunkSize) {
    $chunk_size  = [Math]::Min($chunkSize, $size - $i)
    $chunk = New-Object byte[] $chunk_size 

    [System.Buffer]::BlockCopy($byte, $i, $chunk, 0, $chunk_size)

    $outputfile = "${outputPrefix}${part_iter}.pth"
    [System.IO.File]::WriteAllBytes($outputfile, $chunk)
    
    $remains -= $chunk_size 
    $part_iter++
    Write-Output "Iteration $part_iter : chunked $chunk_size => remains $remains bytes"
}