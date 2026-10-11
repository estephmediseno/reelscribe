# reelscribe

Descarga y transcribe videos de TikTok, Instagram Reels y YouTube Shorts en archivos individuales por cada video.

## Salida por cada video

Por cada URL procesada se generan tres archivos en `transcriptions/`:

- `NN_<plataforma>_<autor>_<id>.md`: Metadatos del video, transcripción completa en texto limpio y desglose por marcas de tiempo.
- `NN_<plataforma>_<autor>_<id>.srt`: Subtítulos estándar con tiempos de inicio y fin.
- `NN_<plataforma>_<autor>_<id>.json`: Datos estructurados con metadatos de `yt-dlp`, texto completo, segmentos y marcas de tiempo por palabra.
- `resumen.md`: Resumen global de todos los videos del lote.

## Uso en WSL

```bash
reelscribe <url1> <url2> ...
```

O pasando un archivo con una URL por línea:

```bash
reelscribe -f urls.txt -o transcriptions/lote_01
```
